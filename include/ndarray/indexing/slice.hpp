/**
 * @file slice.hpp
 * @brief 切片描述符（view_all / range / new_axis）与混合索引 slice() 函数
 */

#pragma once

#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"

#include <cassert>
#include <type_traits>

namespace nd {

// ================================================================
//  切片描述符类型
// ================================================================

/// @brief 切片描述符：保留该轴所有元素
struct view_all_t {};
inline constexpr view_all_t view_all{};

/// @brief 切片描述符：在该位置插入长度为 1 的新轴（不消耗输入轴）
struct new_axis_t {};
inline constexpr new_axis_t new_axis{};

/**
 * @brief 区间切片描述符
 *
 * @details
 * 使用半开区间 `[start, stop)` 与步长 `step` 描述单轴切片，支持负下标。
 */
struct range_desc {
    sindex_t start;
    sindex_t stop;
    sindex_t step;
};

/// @brief 创建区间切片描述符
inline range_desc range(sindex_t start, sindex_t stop, sindex_t step = 1) {
    return {start, stop, step};
}

// ================================================================
//  描述符萃取
// ================================================================

namespace detail {

// ---- 是否为合法描述符 ----

template <typename T, typename = void>
struct is_valid_desc : std::false_type {};

template <typename T>
struct is_valid_desc<T, std::enable_if_t<std::is_integral_v<T>>>
    : std::true_type {};

template <>
struct is_valid_desc<view_all_t, void> : std::true_type {};

template <>
struct is_valid_desc<range_desc, void> : std::true_type {};

template <>
struct is_valid_desc<new_axis_t, void> : std::true_type {};

// ---- 各描述符消耗的输入轴数 ----

template <typename T, typename = void>
struct desc_consumes { static constexpr std::size_t value = 1; };

template <>
struct desc_consumes<new_axis_t, void> { static constexpr std::size_t value = 0; };

// ---- 各描述符产生的输出轴数 ----

template <typename T, typename = void>
struct desc_produces { static constexpr std::size_t value = 1; };

template <typename T>
struct desc_produces<T, std::enable_if_t<std::is_integral_v<T>>>
{ static constexpr std::size_t value = 0; };

// ---- 对参数包求和 ----

template <template <typename, typename> class Trait, typename... Ts>
struct sum_trait;

template <template <typename, typename> class Trait>
struct sum_trait<Trait> { static constexpr std::size_t value = 0; };

template <template <typename, typename> class Trait, typename T, typename... Rest>
struct sum_trait<Trait, T, Rest...> {
    static constexpr std::size_t value =
        Trait<T, void>::value + sum_trait<Trait, Rest...>::value;
};

template <typename... Ts>
inline constexpr std::size_t total_produces_v =
    sum_trait<desc_produces, Ts...>::value;

template <typename... Ts>
inline constexpr std::size_t total_consumes_v =
    sum_trait<desc_consumes, Ts...>::value;

// ================================================================
//  递归处理描述符
// ================================================================

/// @brief 递归终止：所有描述符都已处理完毕
template <std::size_t InAxis, std::size_t OutAxis,
          std::size_t InRank, std::size_t OutRank>
void apply_descs(const shape<InRank>&, const strides<InRank>&,
                 sindex_t&, shape<OutRank>&, strides<OutRank>&) {}

/// @brief 处理 `view_all` 描述符，保留当前输入轴
template <std::size_t InAxis, std::size_t OutAxis,
          std::size_t InRank, std::size_t OutRank, typename... Rest>
void apply_descs(const shape<InRank>& in_sh, const strides<InRank>& in_st,
                 sindex_t& offset, shape<OutRank>& out_sh, strides<OutRank>& out_st,
                 view_all_t, Rest... rest) {
    out_sh[OutAxis]  = in_sh[InAxis];
    out_st[OutAxis]  = in_st[InAxis];
    apply_descs<InAxis + 1, OutAxis + 1>(in_sh, in_st, offset, out_sh, out_st, rest...);
}

/**
 * @brief 处理 `range_desc` 描述符
 *
 * @details
 * 该分支负责完成负下标规范化、边界夹紧、切片长度计算，以及输出 stride 的缩放。
 */
template <std::size_t InAxis, std::size_t OutAxis,
          std::size_t InRank, std::size_t OutRank, typename... Rest>
void apply_descs(const shape<InRank>& in_sh, const strides<InRank>& in_st,
                 sindex_t& offset, shape<OutRank>& out_sh, strides<OutRank>& out_st,
                 range_desc r, Rest... rest) {
    sindex_t dim = static_cast<sindex_t>(in_sh[InAxis]);

    // 负下标规范化
    sindex_t start = r.start < 0 ? r.start + dim : r.start;
    sindex_t stop  = r.stop  < 0 ? r.stop  + dim : r.stop;
    sindex_t step  = r.step;
    assert(step != 0 && "range step must not be zero");

    // 夹紧
    if (start < 0) start = 0;
    if (start > dim) start = dim;
    if (stop < 0) stop = 0;
    if (stop > dim) stop = dim;

    // 计算切片长度
    index_t len = 0;
    if (step > 0 && stop > start) {
        len = static_cast<index_t>((stop - start + step - 1) / step);
    } else if (step < 0 && start > stop) {
        len = static_cast<index_t>((start - stop - step - 1) / (-step));
    }

    out_sh[OutAxis] = len;
    out_st[OutAxis] = in_st[InAxis] * step;
    offset += start * in_st[InAxis];

    apply_descs<InAxis + 1, OutAxis + 1>(in_sh, in_st, offset, out_sh, out_st, rest...);
}

/// @brief 处理标量索引描述符，消去对应输入轴
template <std::size_t InAxis, std::size_t OutAxis,
          std::size_t InRank, std::size_t OutRank,
          typename Idx, typename... Rest,
          std::enable_if_t<std::is_integral_v<std::decay_t<Idx>>, int> = 0>
void apply_descs(const shape<InRank>& in_sh, const strides<InRank>& in_st,
                 sindex_t& offset, shape<OutRank>& out_sh, strides<OutRank>& out_st,
                 Idx idx, Rest... rest) {
    sindex_t dim = static_cast<sindex_t>(in_sh[InAxis]);
    sindex_t normalized = normalize_index(static_cast<sindex_t>(idx), in_sh[InAxis]);
    assert(normalized >= 0 && normalized < dim && "Scalar index out of bounds");
    offset += normalized * in_st[InAxis];
    apply_descs<InAxis + 1, OutAxis>(in_sh, in_st, offset, out_sh, out_st, rest...);
}

/// @brief 处理 `new_axis` 描述符，在输出中插入长度为 1 的新轴
template <std::size_t InAxis, std::size_t OutAxis,
          std::size_t InRank, std::size_t OutRank, typename... Rest>
void apply_descs(const shape<InRank>& in_sh, const strides<InRank>& in_st,
                 sindex_t& offset, shape<OutRank>& out_sh, strides<OutRank>& out_st,
                 new_axis_t, Rest... rest) {
    out_sh[OutAxis] = 1;
    out_st[OutAxis] = 0;
    apply_descs<InAxis, OutAxis + 1>(in_sh, in_st, offset, out_sh, out_st, rest...);
}

} // namespace detail

// ================================================================
//  slice() 自由函数
// ================================================================

/**
 * @brief 对 `ndview` 执行混合索引切片
 * @param v 输入视图
 * @param descs 描述符包，可混用整数、`view_all`、`range`、`new_axis`
 * @return 结果视图，输出秩由描述符在编译期推导
 */
template <typename T, std::size_t Rank, typename... Descs>
auto slice(ndview<T, Rank> v, Descs... descs)
    -> ndview<T, detail::total_produces_v<std::decay_t<Descs>...>>
{
    constexpr std::size_t consumed = detail::total_consumes_v<std::decay_t<Descs>...>;
    static_assert(consumed == Rank,
                  "Number of input-consuming descriptors must equal Rank");
    static_assert(std::conjunction_v<detail::is_valid_desc<std::decay_t<Descs>>...>,
                  "All descriptors must be view_all / range / new_axis or integer");

    constexpr std::size_t OutRank = detail::total_produces_v<std::decay_t<Descs>...>;
    static_assert(OutRank > 0, "Slice result must have at least rank 1");

    shape<OutRank>   out_sh;
    strides<OutRank> out_st;
    sindex_t         offset = 0;

    detail::apply_descs<0, 0>(v.shape(), v.strides(), offset, out_sh, out_st, descs...);

    return ndview<T, OutRank>(v.data() + offset, out_sh, out_st);
}

/// @brief 对可写数组执行混合索引切片
template <typename T, std::size_t Rank, typename Layout, typename Alloc, typename... Descs>
auto slice(ndarray<T, Rank, Layout, Alloc>& arr, Descs... descs)
    -> ndview<T, detail::total_produces_v<std::decay_t<Descs>...>>
{
    return slice(arr.view(), descs...);
}

/// @brief 对只读数组执行混合索引切片
template <typename T, std::size_t Rank, typename Layout, typename Alloc, typename... Descs>
auto slice(const ndarray<T, Rank, Layout, Alloc>& arr, Descs... descs)
    -> ndview<const T, detail::total_produces_v<std::decay_t<Descs>...>>
{
    return slice(arr.cview(), descs...);
}

} // namespace nd
