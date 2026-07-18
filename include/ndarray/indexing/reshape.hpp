/**
 * @file reshape.hpp
 * @brief 形状变换：reshape / flatten / permute / transpose / squeeze_axis / unsqueeze
 */

#pragma once

#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"

#include <cassert>
#include <type_traits>

namespace nd {

// ================================================================
//  reshape
// ================================================================

/**
 * @brief 对视图执行零拷贝 reshape
 * @tparam NewRank 输出秩
 * @param v 输入视图
 * @param new_shape 目标形状
 * @return 重解释后的新视图
 */
template <std::size_t NewRank, typename T, std::size_t Rank>
ndview<T, NewRank> reshape(ndview<T, Rank> v, const shape<NewRank>& new_shape) {
    assert(v.is_contiguous() && "reshape requires contiguous view");
    assert(v.size() == new_shape.total_size() && "reshape: total size mismatch");
    return ndview<T, NewRank>(v.data(), new_shape);
}

/// @brief 对可写数组执行 reshape，并返回视图
template <std::size_t NewRank, typename T, std::size_t Rank, typename L, typename A>
ndview<T, NewRank> reshape(ndarray<T, Rank, L, A>& arr, const shape<NewRank>& new_shape) {
    return reshape(arr.view(), new_shape);
}

/// @brief 对只读数组执行 reshape，并返回只读视图
template <std::size_t NewRank, typename T, std::size_t Rank, typename L, typename A>
ndview<const T, NewRank> reshape(const ndarray<T, Rank, L, A>& arr,
                                  const shape<NewRank>& new_shape) {
    return reshape(arr.cview(), new_shape);
}

// ================================================================
//  flatten / ravel  →  rank-1 view
// ================================================================

template <typename T, std::size_t Rank>
ndview<T, 1> flatten(ndview<T, Rank> v) {
    assert(v.is_contiguous() && "flatten requires contiguous view");
    return ndview<T, 1>(v.data(), shape<1>(v.size()));
}

/// @brief 对可写数组执行 flatten，并返回 rank-1 视图
template <typename T, std::size_t Rank, typename L, typename A>
ndview<T, 1> flatten(ndarray<T, Rank, L, A>& arr) {
    return flatten(arr.view());
}

/// @brief 对只读数组执行 flatten，并返回 rank-1 只读视图
template <typename T, std::size_t Rank, typename L, typename A>
ndview<const T, 1> flatten(const ndarray<T, Rank, L, A>& arr) {
    return flatten(arr.cview());
}

// ================================================================
//  permute<Perm...>  —  静态轴置换
// ================================================================

namespace detail {

template <std::size_t N>
constexpr bool is_valid_perm(const std::size_t (&perm)[N]) {
    bool seen[N] = {};
    for (std::size_t i = 0; i < N; ++i) {
        if (perm[i] >= N) return false;
        if (seen[perm[i]]) return false;
        seen[perm[i]] = true;
    }
    return true;
}

} // namespace detail

template <std::size_t... Perm, typename T, std::size_t Rank>
ndview<T, Rank> permute(ndview<T, Rank> v) {
    static_assert(sizeof...(Perm) == Rank, "Permutation length must equal Rank");
    constexpr std::size_t perm[] = {Perm...};
    static_assert(detail::is_valid_perm(perm), "Invalid permutation");

    shape<Rank>   new_shape;
    strides<Rank> new_strides;
    for (std::size_t i = 0; i < Rank; ++i) {
        new_shape[i]   = v.shape(perm[i]);
        new_strides[i] = v.stride(perm[i]);
    }
    return ndview<T, Rank>(v.data(), new_shape, new_strides);
}

/// @brief 对可写数组执行静态轴置换
template <std::size_t... Perm, typename T, std::size_t Rank, typename L, typename A>
ndview<T, Rank> permute(ndarray<T, Rank, L, A>& arr) {
    return permute<Perm...>(arr.view());
}

/// @brief 对只读数组执行静态轴置换
template <std::size_t... Perm, typename T, std::size_t Rank, typename L, typename A>
ndview<const T, Rank> permute(const ndarray<T, Rank, L, A>& arr) {
    return permute<Perm...>(arr.cview());
}

// ================================================================
//  transpose  —  permute 的别名 / 2D 特化
// ================================================================

/// @brief 通用静态 transpose，语义等价于 `permute<Perm...>`
template <std::size_t... Perm, typename T, std::size_t Rank>
ndview<T, Rank> transpose(ndview<T, Rank> v) {
    return permute<Perm...>(v);
}

/// @brief 对可写数组执行静态 transpose
template <std::size_t... Perm, typename T, std::size_t Rank, typename L, typename A>
ndview<T, Rank> transpose(ndarray<T, Rank, L, A>& arr) {
    return permute<Perm...>(arr.view());
}

/// @brief 对只读数组执行静态 transpose
template <std::size_t... Perm, typename T, std::size_t Rank, typename L, typename A>
ndview<const T, Rank> transpose(const ndarray<T, Rank, L, A>& arr) {
    return permute<Perm...>(arr.cview());
}

/// @brief 2D 无参 transpose，直接交换行列轴
template <typename T>
ndview<T, 2> transpose(ndview<T, 2> v) {
    return ndview<T, 2>(v.data(),
                        shape<2>(v.shape(1), v.shape(0)),
                        strides<2>(v.stride(1), v.stride(0)));
}

/// @brief 对二维可写数组执行无参 transpose
template <typename T, typename L, typename A>
ndview<T, 2> transpose(ndarray<T, 2, L, A>& arr) {
    return transpose(arr.view());
}

/// @brief 对二维只读数组执行无参 transpose
template <typename T, typename L, typename A>
ndview<const T, 2> transpose(const ndarray<T, 2, L, A>& arr) {
    return transpose(arr.cview());
}

// ================================================================
//  squeeze_axis<Axis>  —  删除长度为 1 的指定轴
// ================================================================

/**
 * @brief 删除指定的单个长度为 1 的轴
 * @tparam Axis 待删除轴编号
 */
template <std::size_t Axis, typename T, std::size_t Rank>
ndview<T, Rank - 1> squeeze_axis(ndview<T, Rank> v) {
    static_assert(Rank > 1, "Cannot squeeze a rank-1 view");
    static_assert(Axis < Rank, "Axis out of range");
    assert(v.shape(Axis) == 1 && "squeeze_axis: axis length must be 1");

    shape<Rank - 1>   new_shape;
    strides<Rank - 1> new_strides;
    std::size_t j = 0;
    for (std::size_t i = 0; i < Rank; ++i) {
        if (i == Axis) continue;
        new_shape[j]   = v.shape(i);
        new_strides[j] = v.stride(i);
        ++j;
    }
    return ndview<T, Rank - 1>(v.data(), new_shape, new_strides);
}

/// @brief 对可写数组执行 `squeeze_axis`
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
ndview<T, Rank - 1> squeeze_axis(ndarray<T, Rank, L, A>& arr) {
    return squeeze_axis<Axis>(arr.view());
}

/// @brief 对只读数组执行 `squeeze_axis`
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
ndview<const T, Rank - 1> squeeze_axis(const ndarray<T, Rank, L, A>& arr) {
    return squeeze_axis<Axis>(arr.cview());
}

// ================================================================
//  unsqueeze<Axis>  —  在指定位置插入长度为 1 的新轴
// ================================================================

/**
 * @brief 在指定位置插入长度为 1 的新轴
 * @tparam Axis 插入位置，可取区间 [0, Rank]
 */
template <std::size_t Axis, typename T, std::size_t Rank>
ndview<T, Rank + 1> unsqueeze(ndview<T, Rank> v) {
    static_assert(Axis <= Rank, "Axis out of range");

    shape<Rank + 1>   new_shape;
    strides<Rank + 1> new_strides;
    std::size_t j = 0;
    for (std::size_t i = 0; i < Rank + 1; ++i) {
        if (i == Axis) {
            new_shape[i] = 1;
            // stride 对 size-1 轴不影响寻址；取相邻轴乘积以兼容 is_contiguous 判断
            if (Axis < Rank) {
                new_strides[i] = static_cast<sindex_t>(v.shape(Axis))
                                 * v.stride(Axis);
            } else {
                new_strides[i] = 1;
            }
        } else {
            new_shape[i]   = v.shape(j);
            new_strides[i] = v.stride(j);
            ++j;
        }
    }
    return ndview<T, Rank + 1>(v.data(), new_shape, new_strides);
}

/// @brief 对可写数组执行 `unsqueeze`
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
ndview<T, Rank + 1> unsqueeze(ndarray<T, Rank, L, A>& arr) {
    return unsqueeze<Axis>(arr.view());
}

/// @brief 对只读数组执行 `unsqueeze`
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
ndview<const T, Rank + 1> unsqueeze(const ndarray<T, Rank, L, A>& arr) {
    return unsqueeze<Axis>(arr.cview());
}

} // namespace nd
