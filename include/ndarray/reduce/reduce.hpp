/**
 * @file reduce.hpp
 * @brief 通用 reduce 内核 + 内建 reduce 函数（sum / prod / min / max / mean）
 *
 * 三个主接口：
 *   reduce_axis<Axis>(src, init, reducer)       → ndarray<T, Rank-1>
 *   reduce_axis_keepdims<Axis>(src, init, reducer) → ndarray<T, Rank>
 *   reduce_all(src, init, reducer)               → T
 *
 * 内建包装（每个提供 <Axis> / _keepdims<Axis> / _all 三种）：
 *   求和、求积、最小值、最大值、均值
 */

#pragma once

#include "../core/axis.hpp"
#include "../core/shape.hpp"
#include "../core/stride.hpp"
#include "../core/layout.hpp"
#include "../core/traits.hpp"
#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>

namespace nd {

// ================================================================
//  detail：递归遍历 + 归约内核
// ================================================================

namespace detail {

// ---- reduce_axis 递归内核 ----
// 递归遍历除 Axis 以外的所有维度。
// 到达"叶"后（所有非 Axis 维遍历完），对 Axis 维做一次归约。

/**
 * @brief 计算除归约轴外的下一个外层遍历维度
 * @tparam Axis 被归约的轴
 * @tparam From 当前搜索起点
 * @tparam InRank 输入秩
 */
template <std::size_t Axis, std::size_t From, std::size_t InRank>
struct next_outer_dim {
    static constexpr std::size_t value =
        (From >= InRank) ? InRank :
        (From == Axis)   ? next_outer_dim<Axis, From + 1, InRank>::value :
                           From;
};

/// @brief `next_outer_dim` 的递归终止情形
template <std::size_t Axis, std::size_t InRank>
struct next_outer_dim<Axis, InRank, InRank> {
    static constexpr std::size_t value = InRank;
};

/**
 * @brief 递归遍历所有非归约轴，并在叶节点对归约轴执行一次折叠
 *
 * @details
 * 该实现保持输入视图的 stride 语义，不要求输入连续；仅输出结果使用连续 row-major 存储。
 */
template <std::size_t Axis, std::size_t Dim, std::size_t OutIdx,
          std::size_t InRank, std::size_t OutRank,
          typename T, typename U, typename Reducer>
void reduce_axis_recurse(const T* src, const shape<InRank>& src_sh,
                         const strides<InRank>& src_st,
                         U* dst, const strides<OutRank>& dst_st,
                         Reducer& reducer, U init,
                         sindex_t src_off, sindex_t dst_off) {
    if constexpr (Dim >= InRank) {
        // 所有非 Axis 维遍历完：对 Axis 维归约
        U acc = init;
        for (index_t k = 0; k < src_sh[Axis]; ++k) {
            acc = reducer(acc, static_cast<U>(
                src[src_off + static_cast<sindex_t>(k) * src_st[Axis]]));
        }
        dst[dst_off] = acc;
    } else {
        // 遍历当前维度 Dim，递归到下一个非 Axis 维
        constexpr std::size_t Next =
            next_outer_dim<Axis, Dim + 1, InRank>::value;
        for (index_t i = 0; i < src_sh[Dim]; ++i) {
            reduce_axis_recurse<Axis, Next, OutIdx + 1>(
                src, src_sh, src_st, dst, dst_st, reducer, init,
                src_off + static_cast<sindex_t>(i) * src_st[Dim],
                dst_off + static_cast<sindex_t>(i) * dst_st[OutIdx]);
        }
    }
}

/// @brief 对全部元素执行递归规约
template <std::size_t Dim, std::size_t Rank, typename T, typename U, typename Reducer>
void reduce_all_impl(const T* data, const shape<Rank>& sh, const strides<Rank>& st,
                     U& acc, Reducer& reducer, sindex_t off) {
    if constexpr (Dim == Rank - 1) {
        for (index_t i = 0; i < sh[Dim]; ++i) {
            acc = reducer(acc, static_cast<U>(
                data[off + static_cast<sindex_t>(i) * st[Dim]]));
        }
    } else {
        for (index_t i = 0; i < sh[Dim]; ++i) {
            reduce_all_impl<Dim + 1>(data, sh, st, acc, reducer,
                off + static_cast<sindex_t>(i) * st[Dim]);
        }
    }
}

} // namespace detail

// ================================================================
//  reduce_axis<Axis>  →  ndarray<U, Rank-1>
// ================================================================

/**
 * @brief 沿指定轴规约并移除该轴
 * @tparam Axis 被规约的轴
 * @param v 输入视图
 * @param init 归约初值
 * @param reducer 二元归约器
 * @return 去掉 `Axis` 后的新数组
 */
template <std::size_t Axis, typename T, std::size_t Rank, typename U, typename Reducer>
ndarray<U, Rank - 1> reduce_axis(ndview<T, Rank> v, U init, Reducer reducer) {
    static_assert(Rank >= 2, "reduce_axis requires Rank >= 2; use reduce_all for Rank==1");
    static_assert(Axis < Rank, "Axis out of range");

    // 构造输出 shape：去掉 Axis 维
    shape<Rank - 1> out_sh;
    std::size_t j = 0;
    for (std::size_t i = 0; i < Rank; ++i) {
        if (i != Axis) out_sh[j++] = v.shape(i);
    }

    ndarray<U, Rank - 1> result(out_sh);
    auto dst_st = row_major::compute_strides(out_sh);

    constexpr std::size_t First =
        detail::next_outer_dim<Axis, 0, Rank>::value;

    detail::reduce_axis_recurse<Axis, First, 0>(
        v.data(), v.shape(), v.strides(),
        result.data(), dst_st,
        reducer, init,
        sindex_t(0), sindex_t(0));

    return result;
}

/// @brief 数组重载的按轴规约接口
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A,
          typename U, typename Reducer>
ndarray<U, Rank - 1> reduce_axis(const ndarray<T, Rank, L, A>& arr,
                                  U init, Reducer reducer) {
    return reduce_axis<Axis>(arr.cview(), init, std::move(reducer));
}

// ================================================================
//  reduce_axis_keepdims<Axis>  →  ndarray<U, Rank>
// ================================================================

/**
 * @brief 沿指定轴规约但保留长度为 1 的轴
 * @tparam Axis 被规约的轴
 * @param v 输入视图
 * @param init 归约初值
 * @param reducer 二元归约器
 * @return 与输入同秩、但 `Axis` 长度为 1 的结果数组
 */
template <std::size_t Axis, typename T, std::size_t Rank, typename U, typename Reducer>
ndarray<U, Rank> reduce_axis_keepdims(ndview<T, Rank> v, U init, Reducer reducer) {
    static_assert(Axis < Rank, "Axis out of range");

    if constexpr (Rank == 1) {
        // Rank==1：reduce 到标量，keepdims 包装为 shape(1)
        U acc = init;
        for (index_t i = 0; i < v.size(); ++i) {
            acc = reducer(acc, static_cast<U>(
                v.data()[static_cast<sindex_t>(i) * v.stride(0)]));
        }
        ndarray<U, 1> result(shape<1>(1));
        result(0) = acc;
        return result;
    } else {
        auto reduced = reduce_axis<Axis>(v, init, reducer);
        // 在 Axis 位置插入长度为 1 的轴
        shape<Rank> out_sh;
        std::size_t j = 0;
        for (std::size_t i = 0; i < Rank; ++i) {
            if (i == Axis) {
                out_sh[i] = 1;
            } else {
                out_sh[i] = reduced.shape(j++);
            }
        }
        ndarray<U, Rank> result(out_sh);
        std::copy_n(reduced.data(), reduced.size(), result.data());
        return result;
    }
}

/// @brief 数组重载的 keepdims 规约接口
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A,
          typename U, typename Reducer>
ndarray<U, Rank> reduce_axis_keepdims(const ndarray<T, Rank, L, A>& arr,
                                       U init, Reducer reducer) {
    return reduce_axis_keepdims<Axis>(arr.cview(), init, std::move(reducer));
}

// ================================================================
//  reduce_all  →  标量 U
// ================================================================

/**
 * @brief 将整个输入规约为单个标量
 * @param v 输入视图
 * @param init 归约初值
 * @param reducer 二元归约器
 * @return 规约结果标量
 */
template <typename T, std::size_t Rank, typename U, typename Reducer>
U reduce_all(ndview<T, Rank> v, U init, Reducer reducer) {
    U acc = init;
    detail::reduce_all_impl<0>(v.data(), v.shape(), v.strides(),
                               acc, reducer, sindex_t(0));
    return acc;
}

/// @brief 数组重载的全量规约接口
template <typename T, std::size_t Rank, typename L, typename A,
          typename U, typename Reducer>
U reduce_all(const ndarray<T, Rank, L, A>& arr, U init, Reducer reducer) {
    return reduce_all(arr.cview(), init, std::move(reducer));
}

// ================================================================
//  内建 reduce 算子
// ================================================================

namespace ops {

/// @brief 求和归约器
struct add_reduce {
    template <typename V> V operator()(V a, V b) const { return a + b; }
};

/// @brief 求积归约器
struct mul_reduce {
    template <typename V> V operator()(V a, V b) const { return a * b; }
};

/// @brief 最小值归约器
struct min_reduce {
    template <typename V> V operator()(V a, V b) const { return a < b ? a : b; }
};

/// @brief 最大值归约器
struct max_reduce {
    template <typename V> V operator()(V a, V b) const { return a > b ? a : b; }
};

} // namespace ops

// ================================================================
//  sum
// ================================================================

/// @brief 沿指定轴求和
template <std::size_t Axis, typename T, std::size_t Rank>
auto sum(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis<Axis>(v, V{0}, ops::add_reduce{});
}

/// @brief 数组重载的按轴求和
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto sum(const ndarray<T, Rank, L, A>& arr) {
    return sum<Axis>(arr.cview());
}

/// @brief 沿指定轴求和并保留长度为 1 的轴
template <std::size_t Axis, typename T, std::size_t Rank>
auto sum_keepdims(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis_keepdims<Axis>(v, V{0}, ops::add_reduce{});
}

/// @brief 数组重载的 keepdims 求和
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto sum_keepdims(const ndarray<T, Rank, L, A>& arr) {
    return sum_keepdims<Axis>(arr.cview());
}

/// @brief 对全部元素求和
template <typename T, std::size_t Rank>
auto sum_all(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_all(v, V{0}, ops::add_reduce{});
}

/// @brief 数组重载的全量求和
template <typename T, std::size_t Rank, typename L, typename A>
auto sum_all(const ndarray<T, Rank, L, A>& arr) {
    return sum_all(arr.cview());
}

// ================================================================
//  prod
// ================================================================

/// @brief 沿指定轴求积
template <std::size_t Axis, typename T, std::size_t Rank>
auto prod(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis<Axis>(v, V{1}, ops::mul_reduce{});
}

/// @brief 数组重载的按轴求积
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto prod(const ndarray<T, Rank, L, A>& arr) {
    return prod<Axis>(arr.cview());
}

/// @brief 沿指定轴求积并保留长度为 1 的轴
template <std::size_t Axis, typename T, std::size_t Rank>
auto prod_keepdims(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis_keepdims<Axis>(v, V{1}, ops::mul_reduce{});
}

/// @brief 数组重载的 keepdims 求积
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto prod_keepdims(const ndarray<T, Rank, L, A>& arr) {
    return prod_keepdims<Axis>(arr.cview());
}

/// @brief 对全部元素求积
template <typename T, std::size_t Rank>
auto prod_all(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_all(v, V{1}, ops::mul_reduce{});
}

/// @brief 数组重载的全量求积
template <typename T, std::size_t Rank, typename L, typename A>
auto prod_all(const ndarray<T, Rank, L, A>& arr) {
    return prod_all(arr.cview());
}

// ================================================================
//  min
// ================================================================

/// @brief 沿指定轴求最小值
template <std::size_t Axis, typename T, std::size_t Rank>
auto min(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis<Axis>(v, std::numeric_limits<V>::max(), ops::min_reduce{});
}

/// @brief 数组重载的按轴最小值
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto min(const ndarray<T, Rank, L, A>& arr) {
    return min<Axis>(arr.cview());
}

/// @brief 沿指定轴求最小值并保留长度为 1 的轴
template <std::size_t Axis, typename T, std::size_t Rank>
auto min_keepdims(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis_keepdims<Axis>(v, std::numeric_limits<V>::max(), ops::min_reduce{});
}

/// @brief 数组重载的 keepdims 最小值
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto min_keepdims(const ndarray<T, Rank, L, A>& arr) {
    return min_keepdims<Axis>(arr.cview());
}

/// @brief 对全部元素求最小值
template <typename T, std::size_t Rank>
auto min_all(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_all(v, std::numeric_limits<V>::max(), ops::min_reduce{});
}

/// @brief 数组重载的全量最小值
template <typename T, std::size_t Rank, typename L, typename A>
auto min_all(const ndarray<T, Rank, L, A>& arr) {
    return min_all(arr.cview());
}

// ================================================================
//  max
// ================================================================

/// @brief 沿指定轴求最大值
template <std::size_t Axis, typename T, std::size_t Rank>
auto max(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis<Axis>(v, std::numeric_limits<V>::lowest(), ops::max_reduce{});
}

/// @brief 数组重载的按轴最大值
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto max(const ndarray<T, Rank, L, A>& arr) {
    return max<Axis>(arr.cview());
}

/// @brief 沿指定轴求最大值并保留长度为 1 的轴
template <std::size_t Axis, typename T, std::size_t Rank>
auto max_keepdims(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_axis_keepdims<Axis>(v, std::numeric_limits<V>::lowest(), ops::max_reduce{});
}

/// @brief 数组重载的 keepdims 最大值
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto max_keepdims(const ndarray<T, Rank, L, A>& arr) {
    return max_keepdims<Axis>(arr.cview());
}

/// @brief 对全部元素求最大值
template <typename T, std::size_t Rank>
auto max_all(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    return reduce_all(v, std::numeric_limits<V>::lowest(), ops::max_reduce{});
}

/// @brief 数组重载的全量最大值
template <typename T, std::size_t Rank, typename L, typename A>
auto max_all(const ndarray<T, Rank, L, A>& arr) {
    return max_all(arr.cview());
}

// ================================================================
//  mean
// ================================================================

/// @brief 沿指定轴求平均值
template <std::size_t Axis, typename T, std::size_t Rank>
auto mean(ndview<T, Rank> v) {
    auto s = sum<Axis>(v);
    using V = typename decltype(s)::value_type;
    V divisor = static_cast<V>(v.shape(Axis));
    for (index_t i = 0; i < s.size(); ++i)
        s.data()[i] /= divisor;
    return s;
}

/// @brief 数组重载的按轴平均值
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto mean(const ndarray<T, Rank, L, A>& arr) {
    return mean<Axis>(arr.cview());
}

/// @brief 沿指定轴求平均值并保留长度为 1 的轴
template <std::size_t Axis, typename T, std::size_t Rank>
auto mean_keepdims(ndview<T, Rank> v) {
    auto s = sum_keepdims<Axis>(v);
    using V = typename decltype(s)::value_type;
    V divisor = static_cast<V>(v.shape(Axis));
    for (index_t i = 0; i < s.size(); ++i)
        s.data()[i] /= divisor;
    return s;
}

/// @brief 数组重载的 keepdims 平均值
template <std::size_t Axis, typename T, std::size_t Rank, typename L, typename A>
auto mean_keepdims(const ndarray<T, Rank, L, A>& arr) {
    return mean_keepdims<Axis>(arr.cview());
}

/// @brief 对全部元素求平均值
template <typename T, std::size_t Rank>
auto mean_all(ndview<T, Rank> v) {
    using V = std::remove_const_t<T>;
    V s = sum_all(v);
    return s / static_cast<V>(v.size());
}

/// @brief 数组重载的全量平均值
template <typename T, std::size_t Rank, typename L, typename A>
auto mean_all(const ndarray<T, Rank, L, A>& arr) {
    return mean_all(arr.cview());
}

} // namespace nd
