/**
 * @file traits.hpp
 * @brief 类型萃取与前向声明
 */

#pragma once

#include <cstddef>
#include <type_traits>

namespace nd {

// ---- 前向声明 ----

template <typename T, std::size_t Rank, typename Layout, typename Allocator>
class ndarray;

template <typename T, std::size_t Rank>
class ndview;

// ---- 标量类型检查 ----

/**
 * @brief 判断类型是否为库支持的标量类型
 * @tparam T 待检查类型
 *
 * @details
 * 当前实现只接受 C++ 基本算术类型，以避免对象生命周期、自定义复制语义和
 * 复杂算子重载对表达式模板与内存布局带来的额外约束。
 */
template <typename T>
inline constexpr bool is_scalar_v = std::is_arithmetic_v<T>;

// ---- 容器 / 视图类型检查 ----

template <typename T>
struct is_ndarray_type : std::false_type {};

template <typename T, std::size_t R, typename L, typename A>
struct is_ndarray_type<ndarray<T, R, L, A>> : std::true_type {};

/// @brief 判断类型是否为 `ndarray`（忽略 cv/ref）
template <typename T>
inline constexpr bool is_ndarray_v = is_ndarray_type<std::decay_t<T>>::value;

template <typename T>
struct is_ndview_type : std::false_type {};

template <typename T, std::size_t R>
struct is_ndview_type<ndview<T, R>> : std::true_type {};

/// @brief 判断类型是否为 `ndview`（忽略 cv/ref）
template <typename T>
inline constexpr bool is_ndview_v = is_ndview_type<std::decay_t<T>>::value;

/// @brief 判断类型是否为数组或视图这类 array-like 对象
template <typename T>
inline constexpr bool is_array_like_v = is_ndarray_v<T> || is_ndview_v<T>;

} // namespace nd
