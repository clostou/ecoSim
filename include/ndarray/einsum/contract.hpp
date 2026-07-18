/**
 * @file contract.hpp
 * @brief 固定秩二元 einsum 收缩主内核
 *
 * 核心接口：
 *   contract<LhsLabels, RhsLabels, OutLabels>(lhs, rhs)
 *     → ndarray<T, OutLabels::size>
 *
 * 内部流程：
 *   1. 编译期将标签分类为 B(batch) / M(lhs-free) / N(rhs-free) / K(contraction)
 *   2. 编译期生成 lhs→[B,M,K]、rhs→[B,K,N] 的置换
 *   3. 运行期折叠各组轴为单个 extent，得到 4D 逻辑视图
 *   4. 执行 batched GEMM-like 融合内循环
 *   5. 将 [B,M,N] 结果反置换到 OutLabels 轴序
 *
 * 同时提供一元 reduce_labels<InLabels, OutLabels>(x)。
 */

#pragma once

#include "labels.hpp"
#include "../core/axis.hpp"
#include "../core/shape.hpp"
#include "../core/stride.hpp"
#include "../core/layout.hpp"
#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"

#include <cassert>
#include <type_traits>

namespace nd {
namespace einsum {

// ================================================================
//  detail: 编译期 / 运行期辅助
// ================================================================

namespace detail {

/**
 * @brief 按置换数组重排视图的形状和步长元数据
 * @param perm 目标轴顺序在原始轴中的下标
 */
template <std::size_t N, std::size_t Rank>
void apply_perm(const std::array<std::size_t, N>& perm,
                const nd::shape<Rank>& src_sh,
                const nd::strides<Rank>& src_st,
                nd::shape<N>& dst_sh,
                nd::strides<N>& dst_st) {
    for (std::size_t i = 0; i < N; ++i) {
        dst_sh[i] = src_sh[perm[i]];
        dst_st[i] = src_st[perm[i]];
    }
}

// ---- 折叠连续组轴为单个 extent ----
// groups 数组：每组的轴数。
// 输入为 N 个轴的 shape/strides，输出为 G 个的 extent/folded_stride。
// folded_stride[g] = 最内层轴（该组最后一个轴）的 stride。
// 运行期检查组内轴连续性（若不连续，仍可工作，但内循环需用组内 stride）。
// 为简化，我们只折叠 extent，内循环使用原始 stride 遍历。

/// @brief 计算一组连续轴的总长度乘积
template <std::size_t Rank>
index_t group_extent(const nd::shape<Rank>& sh,
                     std::size_t start, std::size_t count) {
    index_t ext = 1;
    for (std::size_t i = start; i < start + count; ++i)
        ext *= sh[i];
    return ext;
}

/**
 * @brief 将局部分组内的平坦索引映射为原视图中的物理偏移
 *
 * @details
 * 该辅助函数使收缩内核可在非连续 stride 布局上工作，而无需显式物理转置输入。
 */
template <std::size_t Rank>
sindex_t flat_to_offset(index_t flat,
                        const nd::shape<Rank>& sh,
                        const nd::strides<Rank>& st,
                        std::size_t start, std::size_t count) {
    sindex_t off = 0;
    for (std::size_t i = start + count; i > start; --i) {
        std::size_t d = i - 1;
        index_t idx = flat % sh[d];
        flat /= sh[d];
        off += static_cast<sindex_t>(idx) * st[d];
    }
    return off;
}

/// @brief 生成从原标签序到目标标签序的编译期置换数组
template <typename OrigLabels, typename TargetLabels>
constexpr auto make_perm() {
    return perm_indices<OrigLabels, TargetLabels>();
}

} // namespace detail

// ================================================================
//  contract<LhsLabels, RhsLabels, OutLabels>(lhs, rhs)
// ================================================================

/**
 * @brief 执行固定秩二元 einsum 收缩
 * @tparam LhsLabels 左输入标签序列
 * @tparam RhsLabels 右输入标签序列
 * @tparam OutLabels 输出标签序列
 * @param lhs 左输入视图
 * @param rhs 右输入视图
 * @return 结果数组，秩为 `OutLabels::size`
 */
template <typename LhsLabels, typename RhsLabels, typename OutLabels,
          typename T, std::size_t LRank, std::size_t RRank>
auto contract(ndview<T, LRank> lhs, ndview<T, RRank> rhs) {
    using cl = classify<LhsLabels, RhsLabels, OutLabels>;
    using B = typename cl::B;
    using M = typename cl::M;
    using N = typename cl::N;
    using K = typename cl::K;

    static_assert(LRank == LhsLabels::size, "LhsLabels size must match lhs rank");
    static_assert(RRank == RhsLabels::size, "RhsLabels size must match rhs rank");

    // 目标标签：lhs → [B, M, K]，rhs → [B, K, N]，out → [B, M, N]
    using lhs_target = concat_t<concat_t<B, M>, K>;
    using rhs_target = concat_t<concat_t<B, K>, N>;
    using out_canonical = concat_t<concat_t<B, M>, N>;

    static_assert(lhs_target::size == LRank, "Lhs label classification error");
    static_assert(rhs_target::size == RRank, "Rhs label classification error");
    static_assert(out_canonical::size == OutLabels::size, "Out label count mismatch");

    constexpr std::size_t B_cnt = B::size;
    constexpr std::size_t M_cnt = M::size;
    constexpr std::size_t N_cnt = N::size;
    constexpr std::size_t K_cnt = K::size;
    constexpr std::size_t OutRank = OutLabels::size;

    // 编译期生成置换
    constexpr auto lhs_perm = perm_indices<LhsLabels, lhs_target>();
    constexpr auto rhs_perm = perm_indices<RhsLabels, rhs_target>();

    // 运行期：按置换提取 shape/stride
    nd::shape<LRank>   lhs_sh;  nd::strides<LRank>   lhs_st;
    nd::shape<RRank>   rhs_sh;  nd::strides<RRank>   rhs_st;
    detail::apply_perm(lhs_perm, lhs.shape(), lhs.strides(), lhs_sh, lhs_st);
    detail::apply_perm(rhs_perm, rhs.shape(), rhs.strides(), rhs_sh, rhs_st);

    // 验证 shape 一致性
    // B 轴：lhs[0..B_cnt) == rhs[0..B_cnt)
    for (std::size_t i = 0; i < B_cnt; ++i)
        assert(lhs_sh[i] == rhs_sh[i] && "Batch axis size mismatch");
    // K 轴：lhs[B_cnt+M_cnt .. B_cnt+M_cnt+K_cnt) == rhs[B_cnt .. B_cnt+K_cnt)
    for (std::size_t i = 0; i < K_cnt; ++i)
        assert(lhs_sh[B_cnt + M_cnt + i] == rhs_sh[B_cnt + i] &&
               "Contraction axis size mismatch");

    // 折叠 extents
    index_t B_size = detail::group_extent(lhs_sh, 0, B_cnt);
    index_t M_size = detail::group_extent(lhs_sh, B_cnt, M_cnt);
    index_t K_size = (K_cnt > 0) ? detail::group_extent(lhs_sh, B_cnt + M_cnt, K_cnt) : index_t(1);
    index_t N_size = detail::group_extent(rhs_sh, B_cnt + K_cnt, N_cnt);

    // 构造输出（canonical 序 [B, M, N]）
    nd::shape<OutRank> can_sh;
    for (std::size_t i = 0; i < B_cnt; ++i) can_sh[i] = lhs_sh[i];
    for (std::size_t i = 0; i < M_cnt; ++i) can_sh[B_cnt + i] = lhs_sh[B_cnt + i];
    for (std::size_t i = 0; i < N_cnt; ++i) can_sh[B_cnt + M_cnt + i] = rhs_sh[B_cnt + K_cnt + i];

    ndarray<std::remove_const_t<T>, OutRank> result(can_sh);
    result.zero();
    auto can_st = row_major::compute_strides(can_sh);

    auto* lhs_ptr = lhs.data();
    auto* rhs_ptr = rhs.data();
    auto* out_ptr = result.data();

    // ---- 融合 batched contraction 内核 ----
    // C[b,m,n] += A[b,m,k] * B[b,k,n]
    // 使用原始多维 stride 遍历（非平坦遍历最内层可能不连续时仍正确）

    for (index_t b = 0; b < B_size; ++b) {
        sindex_t b_lhs = detail::flat_to_offset(b, lhs_sh, lhs_st, 0, B_cnt);
        sindex_t b_rhs = detail::flat_to_offset(b, rhs_sh, rhs_st, 0, B_cnt);
        sindex_t b_out = detail::flat_to_offset(b, can_sh, can_st, 0, B_cnt);

        for (index_t m = 0; m < M_size; ++m) {
            sindex_t m_lhs = detail::flat_to_offset(m, lhs_sh, lhs_st, B_cnt, M_cnt);
            sindex_t m_out = detail::flat_to_offset(m, can_sh, can_st, B_cnt, M_cnt);

            for (index_t k = 0; k < K_size; ++k) {
                sindex_t k_lhs = detail::flat_to_offset(k, lhs_sh, lhs_st,
                                                        B_cnt + M_cnt, K_cnt);
                sindex_t k_rhs = detail::flat_to_offset(k, rhs_sh, rhs_st,
                                                        B_cnt, K_cnt);

                auto a_val = lhs_ptr[b_lhs + m_lhs + k_lhs];

                for (index_t n = 0; n < N_size; ++n) {
                    sindex_t n_rhs = detail::flat_to_offset(n, rhs_sh, rhs_st,
                                                            B_cnt + K_cnt, N_cnt);
                    sindex_t n_out = detail::flat_to_offset(n, can_sh, can_st,
                                                            B_cnt + M_cnt, N_cnt);

                    out_ptr[b_out + m_out + n_out] += a_val * rhs_ptr[b_rhs + k_rhs + n_rhs];
                }
            }
        }
    }

    // ---- 如果 OutLabels 序与 canonical 序相同，直接返回 ----
    // 否则需要物理转置结果
    constexpr bool same_order = std::is_same_v<out_canonical, OutLabels>;
    if constexpr (same_order) {
        return result;
    } else {
        // 需要从 canonical [B,M,N] 排列到 OutLabels 序
        constexpr auto out_perm = perm_indices<out_canonical, OutLabels>();

        nd::shape<OutRank> final_sh;
        for (std::size_t i = 0; i < OutRank; ++i)
            final_sh[i] = can_sh[out_perm[i]];

        ndarray<std::remove_const_t<T>, OutRank> final_result(final_sh);
        auto final_st = row_major::compute_strides(final_sh);

        // 逐元素拷贝（通过 canonical 视图的置换遍历）
        // result 是连续的 canonical 序
        index_t total = result.size();
        for (index_t flat = 0; flat < total; ++flat) {
            // 解码 flat → canonical 多维索引
            index_t rem = flat;
            sindex_t src_off = 0;
            sindex_t dst_off = 0;
            // 从最后一维开始解码
            std::array<index_t, OutRank> idx;
            for (std::size_t d = OutRank; d > 0; --d) {
                idx[d - 1] = rem % can_sh[d - 1];
                rem /= can_sh[d - 1];
            }
            // src offset in canonical (contiguous row-major)
            for (std::size_t d = 0; d < OutRank; ++d)
                src_off += static_cast<sindex_t>(idx[d]) * can_st[d];
            // dst offset: permute idx
            for (std::size_t d = 0; d < OutRank; ++d) {
                // final_result 轴 d 对应 canonical 轴 out_perm[d]
                dst_off += static_cast<sindex_t>(idx[out_perm[d]]) * final_st[d];
            }
            final_result.data()[dst_off] = result.data()[src_off];
        }
        return final_result;
    }
}

/// @brief 数组重载的二元 einsum 收缩接口
template <typename LhsLabels, typename RhsLabels, typename OutLabels,
          typename T, std::size_t LRank, typename LL, typename LA,
          std::size_t RRank, typename RL, typename RA>
auto contract(const ndarray<T, LRank, LL, LA>& lhs,
              const ndarray<T, RRank, RL, RA>& rhs) {
    return contract<LhsLabels, RhsLabels, OutLabels>(lhs.cview(), rhs.cview());
}

// ================================================================
//  reduce_labels<InLabels, OutLabels>(x) — 一元 reduce
// ================================================================

/**
 * @brief 按标签保留集合执行一元规约
 * @tparam InLabels 输入标签序列
 * @tparam OutLabels 规约后保留的标签序列
 * @param v 输入视图
 * @return 按 `OutLabels` 排列的新数组
 */
template <typename InLabels, typename OutLabels,
          typename T, std::size_t Rank>
auto reduce_labels(ndview<T, Rank> v) {
    static_assert(Rank == InLabels::size, "InLabels size must match rank");

    // 被缩减的轴 = InLabels \ OutLabels
    using ReduceAxes = diff_t<InLabels, OutLabels>;
    constexpr std::size_t OutRank = OutLabels::size;
    constexpr std::size_t RedCnt = ReduceAxes::size;
    static_assert(OutRank + RedCnt == Rank, "Label count mismatch");

    // 目标排列：[OutLabels..., ReduceAxes...]
    using target = concat_t<OutLabels, ReduceAxes>;
    constexpr auto perm = perm_indices<InLabels, target>();

    nd::shape<Rank>   p_sh;
    nd::strides<Rank> p_st;
    detail::apply_perm(perm, v.shape(), v.strides(), p_sh, p_st);

    // 输出 shape
    nd::shape<OutRank> out_sh;
    for (std::size_t i = 0; i < OutRank; ++i) out_sh[i] = p_sh[i];

    index_t out_total = out_sh.total_size();
    index_t red_size = detail::group_extent(p_sh, OutRank, RedCnt);

    using V = std::remove_const_t<T>;
    ndarray<V, OutRank> result(out_sh);
    result.zero();
    auto out_st = row_major::compute_strides(out_sh);

    auto* src = v.data();
    auto* dst = result.data();

    for (index_t o = 0; o < out_total; ++o) {
        sindex_t o_off_src = detail::flat_to_offset(o, p_sh, p_st, 0, OutRank);
        sindex_t o_off_dst = detail::flat_to_offset(o, out_sh, out_st, 0, OutRank);
        V acc = V{0};
        for (index_t r = 0; r < red_size; ++r) {
            sindex_t r_off = detail::flat_to_offset(r, p_sh, p_st, OutRank, RedCnt);
            acc += static_cast<V>(src[o_off_src + r_off]);
        }
        dst[o_off_dst] = acc;
    }

    return result;
}

/// @brief 数组重载的一元标签规约接口
template <typename InLabels, typename OutLabels,
          typename T, std::size_t Rank, typename L, typename A>
auto reduce_labels(const ndarray<T, Rank, L, A>& arr) {
    return reduce_labels<InLabels, OutLabels>(arr.cview());
}

} // namespace einsum
} // namespace nd
