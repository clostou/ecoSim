/**
 * @file einsum.hpp
 * @brief 静态收缩算子：dot, matmul, batched_matmul, outer, trace
 *
 * 这些函数是 contract<> 的具名包装，覆盖设计文档 §9.1 列出的
 * 首版固定秩 einsum 模式。
 */

#pragma once

#include "labels.hpp"
#include "contract.hpp"
#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"
#include "../reduce/reduce.hpp"

#include <cassert>
#include <type_traits>

namespace nd {

// ================================================================
//  dot(a, b) — 向量点积  i,i->   返回标量
// ================================================================

/**
 * @brief 向量点积
 * @param a 左向量视图
 * @param b 右向量视图
 * @return 标量点积结果
 */
template <typename T>
std::remove_const_t<T> dot(ndview<T, 1> a, ndview<T, 1> b) {
    assert(a.size() == b.size() && "dot: size mismatch");
    using V = std::remove_const_t<T>;
    V acc = V{0};
    for (index_t i = 0; i < a.size(); ++i)
        acc += static_cast<V>(a.data()[static_cast<sindex_t>(i) * a.stride(0)]) *
               static_cast<V>(b.data()[static_cast<sindex_t>(i) * b.stride(0)]);
    return acc;
}

/// @brief 数组重载的点积接口
template <typename T, typename LA, typename AA,
          typename LB, typename AB>
auto dot(const ndarray<T, 1, LA, AA>& a,
         const ndarray<T, 1, LB, AB>& b) {
    return dot(a.cview(), b.cview());
}

// ================================================================
//  matmul(a, b) — 矩阵乘  ij,jk->ik
// ================================================================

/**
 * @brief 矩阵乘法，等价于标签收缩 `ij,jk->ik`
 * @param a 左矩阵视图
 * @param b 右矩阵视图
 * @return 结果矩阵
 */
template <typename T>
auto matmul(ndview<T, 2> a, ndview<T, 2> b) {
    using L = einsum::label_seq<'i','j'>;
    using R = einsum::label_seq<'j','k'>;
    using O = einsum::label_seq<'i','k'>;
    return einsum::contract<L, R, O>(a, b);
}

/// @brief 数组重载的矩阵乘法接口
template <typename T, typename LA, typename AA,
          typename LB, typename AB>
auto matmul(const ndarray<T, 2, LA, AA>& a,
            const ndarray<T, 2, LB, AB>& b) {
    return matmul(a.cview(), b.cview());
}

// ================================================================
//  batched_matmul(a, b) — 批矩阵乘  bij,bjk->bik
// ================================================================

/**
 * @brief 批矩阵乘法，等价于标签收缩 `bij,bjk->bik`
 * @param a 左批矩阵视图
 * @param b 右批矩阵视图
 * @return 批矩阵乘法结果
 */
template <typename T>
auto batched_matmul(ndview<T, 3> a, ndview<T, 3> b) {
    using L = einsum::label_seq<'b','i','j'>;
    using R = einsum::label_seq<'b','j','k'>;
    using O = einsum::label_seq<'b','i','k'>;
    return einsum::contract<L, R, O>(a, b);
}

/// @brief 数组重载的批矩阵乘接口
template <typename T, typename LA, typename AA,
          typename LB, typename AB>
auto batched_matmul(const ndarray<T, 3, LA, AA>& a,
                    const ndarray<T, 3, LB, AB>& b) {
    return batched_matmul(a.cview(), b.cview());
}

// ================================================================
//  outer(a, b) — 外积  i,j->ij
// ================================================================

/**
 * @brief 向量外积，等价于标签构造 `i,j->ij`
 * @param a 左向量视图
 * @param b 右向量视图
 * @return 外积矩阵
 */
template <typename T>
auto outer(ndview<T, 1> a, ndview<T, 1> b) {
    using V = std::remove_const_t<T>;
    index_t m = a.size();
    index_t n = b.size();
    ndarray<V, 2> result(shape<2>(m, n));
    for (index_t i = 0; i < m; ++i) {
        V ai = static_cast<V>(a.data()[static_cast<sindex_t>(i) * a.stride(0)]);
        for (index_t j = 0; j < n; ++j) {
            result(i, j) = ai * static_cast<V>(
                b.data()[static_cast<sindex_t>(j) * b.stride(0)]);
        }
    }
    return result;
}

/// @brief 数组重载的外积接口
template <typename T, typename LA, typename AA,
          typename LB, typename AB>
auto outer(const ndarray<T, 1, LA, AA>& a,
           const ndarray<T, 1, LB, AB>& b) {
    return outer(a.cview(), b.cview());
}

// ================================================================
//  trace<Axis0, Axis1>(a) — 沿两个轴求迹并降秩
//  trace 要求 shape[Axis0] == shape[Axis1]
//  结果秩 = Rank - 2
// ================================================================

/**
 * @brief 沿两个轴求迹并移除这两个轴
 * @tparam Axis0 第一个对角轴
 * @tparam Axis1 第二个对角轴
 * @param v 输入视图
 * @return 若 `Rank > 2` 返回降秩数组，否则返回标量
 */
template <std::size_t Axis0, std::size_t Axis1,
          typename T, std::size_t Rank>
auto trace(ndview<T, Rank> v) {
    static_assert(Rank >= 2, "trace requires Rank >= 2");
    static_assert(Axis0 < Rank && Axis1 < Rank, "Axis out of range");
    static_assert(Axis0 != Axis1, "Axis0 and Axis1 must be different");

    constexpr std::size_t A0 = (Axis0 < Axis1) ? Axis0 : Axis1;
    constexpr std::size_t A1 = (Axis0 < Axis1) ? Axis1 : Axis0;

    assert(v.shape(A0) == v.shape(A1) && "trace: axes must have same length");

    using V = std::remove_const_t<T>;
    constexpr std::size_t OutRank = Rank - 2;

    // 构造输出 shape：去掉 A0 和 A1
    shape<OutRank> out_sh;
    std::size_t j = 0;
    for (std::size_t i = 0; i < Rank; ++i) {
        if (i != A0 && i != A1) out_sh[j++] = v.shape(i);
    }

    ndarray<V, OutRank> result(out_sh);
    result.zero();
    auto out_st = row_major::compute_strides(out_sh);

    index_t trace_len = v.shape(A0);
    index_t out_total = out_sh.total_size();

    auto* src = v.data();
    auto* dst = result.data();

    // 遍历输出元素
    for (index_t o = 0; o < out_total; ++o) {
        // 解码 o → 输出多维索引
        index_t rem = o;
        sindex_t dst_off = 0;
        std::array<index_t, OutRank> oidx;
        for (std::size_t d = OutRank; d > 0; --d) {
            oidx[d - 1] = rem % out_sh[d - 1];
            rem /= out_sh[d - 1];
        }
        for (std::size_t d = 0; d < OutRank; ++d)
            dst_off += static_cast<sindex_t>(oidx[d]) * out_st[d];

        // 映射到源多维索引（跳过 A0, A1）
        sindex_t base_off = 0;
        j = 0;
        for (std::size_t d = 0; d < Rank; ++d) {
            if (d == A0 || d == A1) continue;
            base_off += static_cast<sindex_t>(oidx[j++]) * v.stride(d);
        }

        // 沿对角线求和
        V acc = V{0};
        for (index_t t = 0; t < trace_len; ++t) {
            sindex_t t_off = static_cast<sindex_t>(t) * v.stride(A0) +
                             static_cast<sindex_t>(t) * v.stride(A1);
            acc += static_cast<V>(src[base_off + t_off]);
        }
        dst[dst_off] = acc;
    }

    return result;
}

/// @brief 二阶方阵求迹的标量特化
template <std::size_t Axis0, std::size_t Axis1, typename T>
auto trace(ndview<T, 2> v) {
    static_assert(Axis0 < 2 && Axis1 < 2 && Axis0 != Axis1,
                  "Invalid trace axes for rank-2");
    assert(v.shape(0) == v.shape(1) && "trace: axes must have same length");
    using V = std::remove_const_t<T>;
    V acc = V{0};
    for (index_t t = 0; t < v.shape(0); ++t)
        acc += static_cast<V>(v(t, t));
    return acc;
}

/// @brief 数组重载的迹计算接口
template <std::size_t Axis0, std::size_t Axis1,
          typename T, std::size_t Rank, typename L, typename A>
auto trace(const ndarray<T, Rank, L, A>& arr) {
    return trace<Axis0, Axis1>(arr.cview());
}

} // namespace nd
