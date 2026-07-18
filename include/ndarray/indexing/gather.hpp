/**
 * @file gather.hpp
 * @brief take_axis<Axis> — 按 gather 语义沿指定轴抽取元素
 */

#pragma once

#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"

#include <cassert>
#include <type_traits>

namespace nd {

namespace detail {

/**
 * @brief `take_axis` 的递归实现
 *
 * @details
 * 该算法逐维递归遍历源张量；在普通轴上做一一映射，在 gather 轴上根据 `ids`
 * 读取指定层并复制到目标数组，因此结果始终是连续新数组而不是仿射 view。
 */
template <std::size_t Axis, std::size_t Dim,
          typename T, typename U, std::size_t Rank, typename IndexArray>
void take_axis_impl(ndview<T, Rank> src, ndview<U, Rank> dst,
                    const IndexArray& ids,
                    sindex_t src_off, sindex_t dst_off) {
    if constexpr (Dim == Rank) {
        // 叶节点：复制单个元素
        dst.data()[dst_off] = src.data()[src_off];
    } else if constexpr (Dim == Axis) {
        // gather 轴：用 ids 驱动
        for (std::size_t i = 0; i < ids.size(); ++i) {
            sindex_t idx = normalize_index(static_cast<sindex_t>(ids[i]),
                                           src.shape(Axis));
            assert(idx >= 0 && static_cast<index_t>(idx) < src.shape(Axis));
            take_axis_impl<Axis, Dim + 1>(
                src, dst, ids,
                src_off + idx * src.stride(Dim),
                dst_off + static_cast<sindex_t>(i) * dst.stride(Dim));
        }
    } else {
        // 普通轴：逐步遍历
        for (index_t i = 0; i < src.shape(Dim); ++i) {
            take_axis_impl<Axis, Dim + 1>(
                src, dst, ids,
                src_off + static_cast<sindex_t>(i) * src.stride(Dim),
                dst_off + static_cast<sindex_t>(i) * dst.stride(Dim));
        }
    }
}

} // namespace detail

// ================================================================
//  take_axis<Axis>  —  沿指定轴 gather
// ================================================================

/**
 * @brief 从视图沿指定轴按 gather 语义抽取元素
 * @tparam Axis 待抽取轴
 * @param v 输入视图
 * @param ids 目标轴上的索引序列，可重复、可乱序
 * @return 新分配的连续数组
 */
template <std::size_t Axis, typename T, std::size_t Rank, typename IndexArray>
ndarray<std::remove_const_t<T>, Rank> take_axis(ndview<T, Rank> v,
                                                 const IndexArray& ids) {
    static_assert(Axis < Rank, "Axis out of range");

    shape<Rank> out_shape;
    for (std::size_t i = 0; i < Rank; ++i)
        out_shape[i] = (i == Axis) ? ids.size() : v.shape(i);

    ndarray<std::remove_const_t<T>, Rank> result(out_shape);
    detail::take_axis_impl<Axis, 0>(v, result.view(), ids,
                                    sindex_t(0), sindex_t(0));
    return result;
}

/// @brief 从可写数组沿指定轴执行 gather
template <std::size_t Axis, typename T, std::size_t Rank,
          typename L, typename A, typename IndexArray>
ndarray<T, Rank> take_axis(ndarray<T, Rank, L, A>& arr,
                            const IndexArray& ids) {
    return take_axis<Axis>(arr.view(), ids);
}

/// @brief 从只读数组沿指定轴执行 gather
template <std::size_t Axis, typename T, std::size_t Rank,
          typename L, typename A, typename IndexArray>
ndarray<T, Rank> take_axis(const ndarray<T, Rank, L, A>& arr,
                            const IndexArray& ids) {
    return take_axis<Axis>(arr.cview(), ids);
}

} // namespace nd
