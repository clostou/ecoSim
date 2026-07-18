/**
 * @file labels.hpp
 * @brief 编译期标签系统，用于 einsum 收缩的轴分类与置换推导
 *
 * 核心类型：
 *   label_seq<Chars...>  —  编译期字符序列，表示一组标签
 *
 * 编译期算法：
 *   contains<Seq, C>       —  判断字符 C 是否在序列中
 *   index_of<Seq, C>       —  字符 C 在序列中的索引
 *   intersect<A, B>        —  两个序列的交集（保持 A 中的顺序）
 *   diff<A, B>             —  A 中不在 B 中的字符（保持 A 中的顺序）
 *   concat<A, B>           —  拼接两个序列
 *   unique<Seq>            —  去重（保持首次出现的顺序）
 *   perm_indices<From, To> —  从 From 到 To 的置换索引数组
 */

#pragma once

#include <cstddef>
#include <array>
#include <type_traits>

namespace nd {
namespace einsum {

// ================================================================
//  label_seq — 编译期字符序列
// ================================================================

template <char... Cs>
struct label_seq {
    /// @brief 标签数量
    static constexpr std::size_t size = sizeof...(Cs);
    /// @brief 以空字符结尾的标签数组，便于调试或编译期访问
    static constexpr char chars[sizeof...(Cs) + 1] = {Cs..., '\0'};

    /// @brief 返回第 `i` 个标签字符
    static constexpr char at(std::size_t i) { return chars[i]; }
};

// 空序列特化
template <>
struct label_seq<> {
    static constexpr std::size_t size = 0;
    static constexpr char chars[1] = {'\0'};
    /// @brief 空标签序列总是返回空字符
    static constexpr char at(std::size_t) { return '\0'; }
};

// ================================================================
//  contains<Seq, C>
// ================================================================

namespace detail {

template <char C, char... Cs>
struct contains_impl : std::false_type {};

template <char C, char First, char... Rest>
struct contains_impl<C, First, Rest...>
    : std::conditional_t<C == First, std::true_type,
                         contains_impl<C, Rest...>> {};

} // namespace detail

/// @brief 判断标签 `C` 是否属于序列 `Seq`
template <typename Seq, char C>
struct contains;

template <char... Cs, char C>
struct contains<label_seq<Cs...>, C> : detail::contains_impl<C, Cs...> {};

template <char C>
struct contains<label_seq<>, C> : std::false_type {};

template <typename Seq, char C>
inline constexpr bool contains_v = contains<Seq, C>::value;

// ================================================================
//  index_of<Seq, C>
// ================================================================

namespace detail {

template <std::size_t I, char C, char... Cs>
struct index_of_impl;

template <std::size_t I, char C, char First, char... Rest>
struct index_of_impl<I, C, First, Rest...>
    : std::conditional_t<C == First,
                         std::integral_constant<std::size_t, I>,
                         index_of_impl<I + 1, C, Rest...>> {};

template <std::size_t I, char C>
struct index_of_impl<I, C> : std::integral_constant<std::size_t, I> {
    // sentinel: returns size if not found
};

} // namespace detail

/// @brief 返回标签 `C` 在序列 `Seq` 中的下标；若不存在则返回 `Seq::size`
template <typename Seq, char C>
struct index_of;

template <char... Cs, char C>
struct index_of<label_seq<Cs...>, C> : detail::index_of_impl<0, C, Cs...> {};

template <typename Seq, char C>
inline constexpr std::size_t index_of_v = index_of<Seq, C>::value;

// ================================================================
//  intersect<A, B> — A ∩ B（保持 A 中顺序）
// ================================================================

namespace detail {

template <typename B, typename Acc, char... Cs>
struct intersect_impl;

template <typename B, typename Acc>
struct intersect_impl<B, Acc> { using type = Acc; };

template <typename B, char... AccCs, char C, char... Rest>
struct intersect_impl<B, label_seq<AccCs...>, C, Rest...>
    : std::conditional_t<
          contains_v<B, C>,
          intersect_impl<B, label_seq<AccCs..., C>, Rest...>,
          intersect_impl<B, label_seq<AccCs...>, Rest...>> {};

} // namespace detail

/// @brief 求两个标签序列的交集，并保持左侧序列中的相对顺序
template <typename A, typename B>
struct intersect;

template <char... As, typename B>
struct intersect<label_seq<As...>, B> {
    using type = typename detail::intersect_impl<B, label_seq<>, As...>::type;
};

template <typename A, typename B>
using intersect_t = typename intersect<A, B>::type;

// ================================================================
//  diff<A, B> — A \ B（保持 A 中顺序）
// ================================================================

namespace detail {

template <typename B, typename Acc, char... Cs>
struct diff_impl;

template <typename B, typename Acc>
struct diff_impl<B, Acc> { using type = Acc; };

template <typename B, char... AccCs, char C, char... Rest>
struct diff_impl<B, label_seq<AccCs...>, C, Rest...>
    : std::conditional_t<
          contains_v<B, C>,
          diff_impl<B, label_seq<AccCs...>, Rest...>,
          diff_impl<B, label_seq<AccCs..., C>, Rest...>> {};

} // namespace detail

/// @brief 求标签差集 `A \ B`，并保持 `A` 中的相对顺序
template <typename A, typename B>
struct diff;

template <char... As, typename B>
struct diff<label_seq<As...>, B> {
    using type = typename detail::diff_impl<B, label_seq<>, As...>::type;
};

template <typename A, typename B>
using diff_t = typename diff<A, B>::type;

// ================================================================
//  concat<A, B>
// ================================================================

/// @brief 拼接两个标签序列
template <typename A, typename B>
struct concat;

template <char... As, char... Bs>
struct concat<label_seq<As...>, label_seq<Bs...>> {
    using type = label_seq<As..., Bs...>;
};

template <typename A, typename B>
using concat_t = typename concat<A, B>::type;

// ================================================================
//  unique<Seq> — 去重（保持首次出现顺序）
// ================================================================

namespace detail {

template <typename Seen, typename Acc, char... Cs>
struct unique_impl;

template <typename Seen, typename Acc>
struct unique_impl<Seen, Acc> { using type = Acc; };

template <typename Seen, char... AccCs, char C, char... Rest>
struct unique_impl<Seen, label_seq<AccCs...>, C, Rest...>
    : std::conditional_t<
          contains_v<Seen, C>,
          unique_impl<Seen, label_seq<AccCs...>, Rest...>,
          unique_impl<concat_t<Seen, label_seq<C>>,
                      label_seq<AccCs..., C>, Rest...>> {};

} // namespace detail

/// @brief 对标签序列去重，并保持首次出现顺序
template <typename Seq>
struct unique;

template <char... Cs>
struct unique<label_seq<Cs...>> {
    using type = typename detail::unique_impl<label_seq<>, label_seq<>, Cs...>::type;
};

template <typename Seq>
using unique_t = typename unique<Seq>::type;

// ================================================================
//  perm_indices<From, To> — 置换索引数组
//  perm[i] = index_of<From, To::at(i)>
// ================================================================

namespace detail {

template <typename From, typename To, std::size_t... Is>
constexpr std::array<std::size_t, sizeof...(Is)>
make_perm_array(std::index_sequence<Is...>) {
    return {{ index_of_v<From, To::at(Is)>... }};
}

} // namespace detail

/**
 * @brief 计算从标签序列 `From` 到 `To` 的置换索引数组
 * @return `perm[i] = index_of_v<From, To::at(i)>`
 */
template <typename From, typename To>
constexpr auto perm_indices() {
    static_assert(From::size == To::size,
                  "perm_indices: From and To must have the same size");
    return detail::make_perm_array<From, To>(
        std::make_index_sequence<To::size>{});
}

// ================================================================
//  all_labels<Lhs, Rhs> — 所有标签的并集（去重）
// ================================================================

/// @brief 两个标签序列的并集，并保持首次出现顺序
template <typename Lhs, typename Rhs>
using all_labels_t = unique_t<concat_t<Lhs, Rhs>>;

// ================================================================
//  classify — B/M/N/K 标签分类
//  B = 出现在 Lhs、Rhs、Out 中
//  M = 出现在 Lhs、Out 中，不在 Rhs 中
//  N = 出现在 Rhs、Out 中，不在 Lhs 中
//  K = 出现在 Lhs、Rhs 中，不在 Out 中
// ================================================================

/**
 * @brief 按 einsum 语义将标签分类为 `B/M/N/K`
 *
 * @details
 * `B` 为 batch 标签，`M` 为左操作数自由标签，`N` 为右操作数自由标签，
 * `K` 为只出现在输入而不出现在输出中的收缩标签。
 */
template <typename Lhs, typename Rhs, typename Out>
struct classify {
    using lhs_rhs = intersect_t<Lhs, Rhs>;        // Lhs ∩ Rhs
    using B = intersect_t<lhs_rhs, Out>;           // batch
    using K = diff_t<lhs_rhs, Out>;                // contraction
    using M = diff_t<diff_t<Lhs, Rhs>, K>;         // lhs free (already not in Rhs)
    using N = diff_t<diff_t<Rhs, Lhs>, K>;         // rhs free (already not in Lhs)
};

} // namespace einsum
} // namespace nd
