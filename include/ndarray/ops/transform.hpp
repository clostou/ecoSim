/**
 * @file transform.hpp
 * @brief 自定义 unary 扩展点：transform(array_or_expr, callable)
 */

#pragma once

#include "unary.hpp"

namespace nd {

/**
 * @brief 对表达式执行自定义一元变换
 * @param e 输入表达式
 * @param fn 单输入标量函数对象
 * @return 延迟求值的一元表达式
 */
template <typename E, typename UnaryFn>
auto transform(const expression_base<E>& e, UnaryFn fn) {
    return detail::make_unary(e, std::move(fn));
}

/// @brief 对视图执行自定义一元变换
template <typename T, std::size_t R, typename UnaryFn>
auto transform(ndview<T, R> v, UnaryFn fn) {
    return detail::make_unary(v, std::move(fn));
}

/// @brief 对数组执行自定义一元变换
template <typename T, std::size_t R, typename L, typename A, typename UnaryFn>
auto transform(const ndarray<T, R, L, A>& arr, UnaryFn fn) {
    return detail::make_unary(arr, std::move(fn));
}

} // namespace nd
