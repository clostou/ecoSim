/**
 * @file math.hpp
 * @brief 内建逐元素数学函数：绝对值、平方根、指数、对数、三角函数、
 *        反三角函数、双曲正切与 sigmoid
 */

#pragma once

#include "unary.hpp"

#include <cmath>

namespace nd {

namespace ops {

/// @brief 逐元素绝对值算子
struct abs_op {
    template <typename T>
    T operator()(T x) const { return x < T{0} ? -x : x; }
};

/// @brief 逐元素平方根算子
struct sqrt_op {
    template <typename T>
    auto operator()(T x) const { return std::sqrt(x); }
};

/// @brief 逐元素指数算子
struct exp_op {
    template <typename T>
    auto operator()(T x) const { return std::exp(x); }
};

/// @brief 逐元素自然对数算子
struct log_op {
    template <typename T>
    auto operator()(T x) const { return std::log(x); }
};

/// @brief 逐元素正弦算子
struct sin_op {
    template <typename T>
    auto operator()(T x) const { return std::sin(x); }
};

/// @brief 逐元素余弦算子
struct cos_op {
    template <typename T>
    auto operator()(T x) const { return std::cos(x); }
};

/// @brief 逐元素正切算子
struct tan_op {
    template <typename T>
    auto operator()(T x) const { return std::tan(x); }
};

/// @brief 逐元素反正弦算子
struct asin_op {
    template <typename T>
    auto operator()(T x) const { return std::asin(x); }
};

/// @brief 逐元素反余弦算子
struct acos_op {
    template <typename T>
    auto operator()(T x) const { return std::acos(x); }
};

/// @brief 逐元素反正切算子
struct atan_op {
    template <typename T>
    auto operator()(T x) const { return std::atan(x); }
};

/// @brief 逐元素双曲正切算子
struct tanh_op {
    template <typename T>
    auto operator()(T x) const { return std::tanh(x); }
};

/// @brief 逐元素 sigmoid 算子
struct sigmoid_op {
    template <typename T>
    auto operator()(T x) const {
        return T{1} / (T{1} + std::exp(-x));
    }
};

} // namespace ops

/**
 * @brief 生成逐元素数学函数重载
 *
 * @details
 * 每个数学函数都同时支持表达式、视图与数组输入，并统一返回延迟求值表达式。
 */

#define NDARRAY_DEFINE_MATH_FUNC(FUNC_NAME, OP_STRUCT)                        \
                                                                               \
template <typename E>                                                          \
auto FUNC_NAME(const expression_base<E>& e) {                                 \
    return detail::make_unary(e, ops::OP_STRUCT{});                            \
}                                                                              \
                                                                               \
template <typename T, std::size_t R>                                           \
auto FUNC_NAME(ndview<T, R> v) {                                               \
    return detail::make_unary(v, ops::OP_STRUCT{});                            \
}                                                                              \
                                                                               \
template <typename T, std::size_t R, typename L, typename A>                   \
auto FUNC_NAME(const ndarray<T, R, L, A>& arr) {                              \
    return detail::make_unary(arr, ops::OP_STRUCT{});                          \
}

NDARRAY_DEFINE_MATH_FUNC(abs,     abs_op)
NDARRAY_DEFINE_MATH_FUNC(sqrt,    sqrt_op)
NDARRAY_DEFINE_MATH_FUNC(exp,     exp_op)
NDARRAY_DEFINE_MATH_FUNC(log,     log_op)
NDARRAY_DEFINE_MATH_FUNC(sin,     sin_op)
NDARRAY_DEFINE_MATH_FUNC(cos,     cos_op)
NDARRAY_DEFINE_MATH_FUNC(tan,     tan_op)
NDARRAY_DEFINE_MATH_FUNC(asin,    asin_op)
NDARRAY_DEFINE_MATH_FUNC(acos,    acos_op)
NDARRAY_DEFINE_MATH_FUNC(atan,    atan_op)
NDARRAY_DEFINE_MATH_FUNC(tanh,    tanh_op)
NDARRAY_DEFINE_MATH_FUNC(sigmoid, sigmoid_op)

#undef NDARRAY_DEFINE_MATH_FUNC

} // namespace nd
