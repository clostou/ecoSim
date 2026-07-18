/**
 * @file expression.hpp
 * @brief 表达式模板基础设施：CRTP 基类、求值器、eval / assign 工具
 *
 * 所有延迟求值表达式（unary_expr, binary_expr, scalar_expr）均继承
 * expression_base<Derived>，通过 CRTP 提供 shape()、operator()、eval() 等统一接口。
 *
 * ndview / ndarray 通过 view_expr 适配器融入表达式体系，
 * 使得 `a + b * 2.0f` 中的 a, b 可以是 ndarray、ndview 或其他表达式。
 */

#pragma once

#include "../core/axis.hpp"
#include "../core/shape.hpp"
#include "../core/broadcast.hpp"
#include "../core/traits.hpp"
#include "../view/ndview.hpp"
#include "../view/ndarray.hpp"

#include <cassert>
#include <type_traits>

namespace nd {

// ================================================================
//  表达式 CRTP 基类
// ================================================================

template <typename Derived>
struct expression_base {
    /// @brief 以 CRTP 方式返回派生类引用
    const Derived& self() const { return static_cast<const Derived&>(*this); }
};

// ================================================================
//  表达式萃取
// ================================================================

template <typename T>
struct is_expression_type : std::false_type {};

template <typename D>
struct is_expression_type<expression_base<D>> : std::true_type {};

/// @brief 默认类型不是表达式节点
template <typename T, typename = void>
struct is_expression : std::false_type {};

/// @brief 通过 CRTP 继承关系检测表达式节点
template <typename T>
struct is_expression<T, std::enable_if_t<
    std::is_base_of_v<expression_base<T>, T>>> : std::true_type {};

template <typename T>
inline constexpr bool is_expression_v = is_expression<std::decay_t<T>>::value;

/// @brief 判断类型是否可作为表达式树的操作数
template <typename T>
inline constexpr bool is_operand_v =
    is_expression_v<T> || is_array_like_v<T> || is_scalar_v<T>;

// ================================================================
//  view_expr：将 ndview 包装为表达式
// ================================================================

template <typename T, std::size_t Rank>
class view_expr : public expression_base<view_expr<T, Rank>> {
    ndview<T, Rank> v_;
public:
    using value_type = std::remove_const_t<T>;
    static constexpr std::size_t rank = Rank;

    /// @brief 由视图构造表达式叶节点
    explicit view_expr(ndview<T, Rank> v) : v_(v) {}

    /// @brief 返回表达式逻辑形状
    const nd::shape<Rank>& shape()         const { return v_.shape(); }
    /// @brief 返回指定轴长度
    index_t                shape(axis_t a) const { return v_.shape(a); }
    /// @brief 返回元素总数
    index_t                size()          const { return v_.size(); }

    /// @brief 访问给定逻辑坐标上的元素值
    template <typename... Indices>
    value_type operator()(Indices... idx) const {
        return v_(idx...);
    }
};

// ================================================================
//  scalar_expr：标量广播到任意形状
// ================================================================

template <typename T, std::size_t Rank>
class scalar_expr : public expression_base<scalar_expr<T, Rank>> {
    T              val_;
    nd::shape<Rank> shape_;
public:
    using value_type = T;
    static constexpr std::size_t rank = Rank;

    /// @brief 由标量值与目标形状构造广播表达式
    scalar_expr(const T& val, const nd::shape<Rank>& sh)
        : val_(val), shape_(sh) {}

    /// @brief 返回广播后的逻辑形状
    const nd::shape<Rank>& shape()         const { return shape_; }
    /// @brief 返回指定轴长度
    index_t                shape(axis_t a) const { return shape_[a]; }
    /// @brief 返回逻辑元素总数
    index_t                size()          const { return shape_.total_size(); }

    /// @brief 任意逻辑坐标都返回同一个标量
    template <typename... Indices>
    T operator()(Indices...) const { return val_; }
};

// ================================================================
//  broadcast_expr：将内层表达式广播到目标形状
// ================================================================

/**
 * @brief 广播表达式包装器
 *
 * @details
 * 将内层表达式 `inner_` 的逻辑形状广播到 `out_shape_`。
 * 对于 inner 中长度为 1 的轴，输出坐标映射时始终使用 0；
 * 对于长度一致的轴，直接透传坐标。
 *
 * 这是纯逻辑包装，不产生任何数据拷贝或临时数组。
 */
template <typename Expr>
class broadcast_expr : public expression_base<broadcast_expr<Expr>> {
    Expr                      inner_;
    nd::shape<Expr::rank>     src_shape_;
    nd::shape<Expr::rank>     out_shape_;
public:
    using value_type = typename Expr::value_type;
    static constexpr std::size_t rank = Expr::rank;

    /// @brief 构造广播包装
    /// @param inner 内层表达式
    /// @param out_shape 广播后的目标形状
    broadcast_expr(Expr inner, const nd::shape<rank>& out_shape)
        : inner_(std::move(inner)),
          src_shape_(inner_.shape()),
          out_shape_(out_shape) {}

    /// @brief 返回广播后的逻辑形状
    const nd::shape<rank>& shape()         const { return out_shape_; }
    /// @brief 返回指定轴长度
    index_t                shape(axis_t a) const { return out_shape_[a]; }
    /// @brief 返回广播后的元素总数
    index_t                size()          const { return out_shape_.total_size(); }

    /// @brief 访问给定逻辑坐标上的元素值，长度为 1 的轴自动折叠
    template <typename... Indices>
    value_type operator()(Indices... idx) const {
        return call_inner(std::index_sequence_for<Indices...>{}, idx...);
    }

private:
    /// @brief 将输出坐标映射为源坐标并调用 inner
    template <std::size_t... Is, typename... Indices>
    value_type call_inner(std::index_sequence<Is...>, Indices... idx) const {
        // 将所有 idx 放入数组以便按轴重映射
        sindex_t arr[] = { static_cast<sindex_t>(idx)... };
        return inner_(
            static_cast<sindex_t>(src_shape_[Is] == 1 ? 0 : arr[Is])...
        );
    }
};

// ================================================================
//  as_expr：自动将各类操作数包装为表达式
// ================================================================

namespace detail {

/// @brief 将视图适配为表达式叶节点
template <typename T, std::size_t Rank>
view_expr<T, Rank> as_expr_impl(ndview<T, Rank> v) {
    return view_expr<T, Rank>(v);
}

/// @brief 将常量数组适配为只读表达式叶节点
template <typename T, std::size_t Rank, typename L, typename A>
view_expr<const T, Rank> as_expr_impl(const ndarray<T, Rank, L, A>& arr) {
    return view_expr<const T, Rank>(arr.cview());
}

/// @brief 将可写数组适配为当前实现下的只读表达式叶节点
template <typename T, std::size_t Rank, typename L, typename A>
view_expr<T, Rank> as_expr_impl(ndarray<T, Rank, L, A>& arr) {
    return view_expr<const T, Rank>(arr.cview());
}

/// @brief 对表达式节点直接返回自身
template <typename E>
const E& as_expr_impl(const expression_base<E>& e) {
    return e.self();
}

// ---- expr_rank：获取操作数的秩 ----

template <typename T, typename = void>
struct expr_rank_impl;

template <typename T>
struct expr_rank_impl<T, std::enable_if_t<is_expression_v<T>>> {
    static constexpr std::size_t value = T::rank;
};

template <typename T, std::size_t R>
struct expr_rank_impl<ndview<T, R>, void> {
    static constexpr std::size_t value = R;
};

template <typename T, std::size_t R, typename L, typename A>
struct expr_rank_impl<ndarray<T, R, L, A>, void> {
    static constexpr std::size_t value = R;
};

template <typename T>
inline constexpr std::size_t expr_rank_v =
    expr_rank_impl<std::decay_t<T>>::value;

} // namespace detail

// ================================================================
//  eval：将表达式求值为新 ndarray
// ================================================================

/**
 * @brief 将延迟表达式求值为新的连续数组
 * @param expr 输入表达式
 * @return 求值后的 `ndarray`
 */
template <typename E>
auto eval(const expression_base<E>& expr) {
    const E& e = expr.self();
    using T = typename E::value_type;
    constexpr std::size_t R = E::rank;

    ndarray<T, R> result(e.shape());
    eval_into(result.view(), e);
    return result;
}

// ================================================================
//  eval_into：将表达式写入已有的 ndview
// ================================================================

namespace detail {

template <std::size_t Dim, std::size_t Rank, typename T, typename E, typename... Indices>
void eval_recursive(ndview<T, Rank> dst, const E& expr,
                    const shape<Rank>& sh, Indices... indices) {
    if constexpr (Dim == Rank) {
        dst(indices...) = expr(indices...);
    } else {
        for (index_t i = 0; i < sh[Dim]; ++i) {
            eval_recursive<Dim + 1>(dst, expr, sh, indices...,
                                    static_cast<sindex_t>(i));
        }
    }
}

} // namespace detail

/**
 * @brief 将表达式写入已有视图
 * @param dst 目标视图
 * @param expr 输入表达式
 */
template <typename T, std::size_t Rank, typename E>
void eval_into(ndview<T, Rank> dst, const E& expr) {
    static_assert(E::rank == Rank, "rank mismatch");
    detail::eval_recursive<0>(dst, expr, dst.shape());
}

/// @brief 将表达式写入已有数组
template <typename T, std::size_t Rank, typename L, typename A, typename E>
void eval_into(ndarray<T, Rank, L, A>& dst, const expression_base<E>& expr) {
    eval_into(dst.view(), expr.self());
}

// ================================================================
//  broadcast_to：将 ndview / ndarray 广播到指定形状
// ================================================================

/**
 * @brief 将视图广播到目标形状（返回延迟表达式，不拷贝数据）
 *
 * @param v 源视图
 * @param target 目标形状（必须与源形状广播兼容）
 * @return broadcast_expr 延迟表达式
 */
template <typename T, std::size_t Rank>
auto broadcast_to(ndview<T, Rank> v, const shape<Rank>& target) {
    assert(broadcastable(v.shape(), target) && "cannot broadcast to target shape");
    auto ve = view_expr<T, Rank>(v);
    return broadcast_expr<view_expr<T, Rank>>(std::move(ve), target);
}

/// @brief 将数组广播到目标形状
template <typename T, std::size_t Rank, typename L, typename A>
auto broadcast_to(const ndarray<T, Rank, L, A>& arr, const shape<Rank>& target) {
    return broadcast_to(arr.cview(), target);
}

} // namespace nd
