/**
 * @file utility.hpp
 * @brief 偏移量计算、边界检查、连续性判定等工具函数
 */

#pragma once

#include "shape.hpp"
#include "stride.hpp"
#include "layout.hpp"

#include <cassert>

namespace nd {

// ---- 负下标规范化 ----

/**
 * @brief 规范化可能为负的下标
 * @param idx 原始下标
 * @param dim_size 当前轴长度
 * @return 规范化后的下标
 */
inline constexpr sindex_t normalize_index(sindex_t idx, index_t dim_size) {
    return idx < 0 ? idx + static_cast<sindex_t>(dim_size) : idx;
}

// ---- 线性偏移量计算 ----

/**
 * @brief 由 stride 与多维下标计算线性偏移
 * @tparam Rank 维度数
 * @param st 各轴 stride
 * @param indices 每个轴的下标
 * @return 元素粒度的线性偏移
 */
template <std::size_t Rank, typename... Indices>
constexpr sindex_t compute_offset(const strides<Rank>& st, Indices... indices) {
    static_assert(sizeof...(Indices) == Rank,
                  "Number of indices must match Rank");
    const sindex_t idx[] = {static_cast<sindex_t>(indices)...};
    sindex_t offset = 0;
    for (std::size_t i = 0; i < Rank; ++i)
        offset += idx[i] * st[i];
    return offset;
}

/**
 * @brief 由 shape 和 stride 计算线性偏移，并自动处理负下标
 * @tparam Rank 维度数
 * @param sh 各轴长度
 * @param st 各轴 stride
 * @param indices 每个轴的下标
 * @return 元素粒度的线性偏移
 */
template <std::size_t Rank, typename... Indices>
constexpr sindex_t compute_offset_norm(const shape<Rank>& sh,
                                       const strides<Rank>& st,
                                       Indices... indices) {
    static_assert(sizeof...(Indices) == Rank,
                  "Number of indices must match Rank");
    const sindex_t raw[] = {static_cast<sindex_t>(indices)...};
    sindex_t offset = 0;
    for (std::size_t i = 0; i < Rank; ++i) {
        sindex_t idx = normalize_index(raw[i], sh[i]);
        offset += idx * st[i];
    }
    return offset;
}

// ---- 连续性判定 ----

/**
 * @brief 判断给定 shape 与 stride 是否为 row-major 连续布局
 * @tparam Rank 维度数
 * @param sh 各轴长度
 * @param st 各轴 stride
 * @return 若与默认连续布局一致则返回 true
 */
template <std::size_t Rank>
constexpr bool is_contiguous(const shape<Rank>& sh, const strides<Rank>& st) {
    auto expected = row_major::compute_strides(sh);
    for (std::size_t i = 0; i < Rank; ++i) {
        if (st[i] != expected[i]) return false;
    }
    return true;
}

// ---- 边界检查（支持负下标）----

#ifdef NDARRAY_DEBUG
/**
 * @brief 调试模式下执行边界检查
 * @tparam Rank 维度数
 * @param sh 形状
 * @param indices 每个轴的下标，支持负下标
 */
template <std::size_t Rank, typename... Indices>
void check_bounds(const shape<Rank>& sh, Indices... indices) {
    static_assert(sizeof...(Indices) == Rank);
    const sindex_t raw[] = {static_cast<sindex_t>(indices)...};
    for (std::size_t i = 0; i < Rank; ++i) {
        sindex_t idx = normalize_index(raw[i], sh[i]);
        assert(idx >= 0 && static_cast<index_t>(idx) < sh[i]
               && "Index out of bounds");
    }
}
#else
/// @brief 非调试模式下的空边界检查占位
template <std::size_t Rank, typename... Indices>
constexpr void check_bounds(const shape<Rank>&, Indices...) {}
#endif

} // namespace nd
