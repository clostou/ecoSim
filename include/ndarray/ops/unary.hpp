/**
 * @file unary.hpp
 * @brief 一元表达式模板：unary_expr + 内建一元运算符（取负）
 */

#pragma once

#include "expression.hpp"

namespace nd {

// ================================================================
//  unary_expr<Expr, Op>
// ================================================================

template <typename Expr, typename Op>
class unary_expr : public expression_base<unary_expr<Expr, Op>> {
    Expr expr_;
    Op   op_;
public:
    using value_type = decltype(std::declval<Op>()(
        std::declval<typename Expr::value_type>()));
    static constexpr std::size_t rank = Expr::rank;

    /// @brief 由输入表达式与一元算子构造延迟表达式节点
    unary_expr(Expr expr, Op op)
        : expr_(std::move(expr)), op_(std::move(op)) {}

    /// @brief 返回表达式逻辑形状
    const nd::shape<rank>& shape()         const { return expr_.shape(); }
    /// @brief 返回指定轴长度
    index_t                shape(axis_t a) const { return expr_.shape(a); }
    /// @brief 返回元素总数
    index_t                size()          const { return expr_.size(); }

    /// @brief 对给定逻辑坐标执行一元运算
    template <typename... Indices>
    value_type operator()(Indices... idx) const {
        return op_(expr_(idx...));
    }
};

// ================================================================
//  make_unary：构建 unary_expr 的工厂
// ================================================================

namespace detail {

template <typename Op, typename E>
auto make_unary(const expression_base<E>& e, Op op) {
    return unary_expr<E, Op>(e.self(), std::move(op));
}

/// @brief 由视图构造一元表达式
template <typename Op, typename T, std::size_t Rank>
auto make_unary(ndview<T, Rank> v, Op op) {
    return unary_expr<view_expr<T, Rank>, Op>(view_expr<T, Rank>(v), std::move(op));
}

/// @brief 由数组构造一元表达式
template <typename Op, typename T, std::size_t R, typename L, typename A>
auto make_unary(const ndarray<T, R, L, A>& arr, Op op) {
    return make_unary(arr.cview(), std::move(op));
}

} // namespace detail

// ================================================================
//  operator-（取负）
// ================================================================

namespace ops {
/// @brief 逐元素相反数算子
struct negate_op {
    template <typename T>
    T operator()(T x) const { return -x; }
};
} // namespace ops

/// @brief 表达式上的逐元素取负
template <typename E>
auto operator-(const expression_base<E>& e) {
    return detail::make_unary(e, ops::negate_op{});
}

/// @brief 视图上的逐元素取负
template <typename T, std::size_t R>
auto operator-(ndview<T, R> v) {
    return detail::make_unary(v, ops::negate_op{});
}

/// @brief 数组上的逐元素取负
template <typename T, std::size_t R, typename L, typename A>
auto operator-(const ndarray<T, R, L, A>& arr) {
    return detail::make_unary(arr, ops::negate_op{});
}

} // namespace nd
