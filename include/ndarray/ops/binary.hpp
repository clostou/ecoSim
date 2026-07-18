/**
 * @file binary.hpp
 * @brief 二元表达式模板：binary_expr + 四则运算符重载
 *
 * 运算规则：
 *   - 同秩数组/视图/表达式之间的逐元素运算（支持 NumPy 风格广播）
 *   - 数组/视图/表达式与标量之间的逐元素运算（标量广播）
 */

#pragma once

#include "expression.hpp"

namespace nd {

// ================================================================
//  binary_expr<Lhs, Rhs, Op>
// ================================================================

template <typename Lhs, typename Rhs, typename Op>
class binary_expr : public expression_base<binary_expr<Lhs, Rhs, Op>> {
    Lhs lhs_;
    Rhs rhs_;
    Op  op_;
    nd::shape<Lhs::rank> out_shape_;
public:
    using value_type = decltype(std::declval<Op>()(
        std::declval<typename Lhs::value_type>(),
        std::declval<typename Rhs::value_type>()));
    static constexpr std::size_t rank = Lhs::rank;

    /// @brief 由左右操作数与二元算子构造延迟表达式节点
    binary_expr(Lhs lhs, Rhs rhs, Op op)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)), op_(std::move(op)),
          out_shape_(broadcast_shape(lhs_.shape(), rhs_.shape())) {}

    /// @brief 返回表达式逻辑形状（广播后）
    const nd::shape<rank>& shape()         const { return out_shape_; }
    /// @brief 返回指定轴长度
    index_t                shape(axis_t a) const { return out_shape_[a]; }
    /// @brief 返回元素总数
    index_t                size()          const { return out_shape_.total_size(); }

    /// @brief 对给定逻辑坐标执行二元运算
    template <typename... Indices>
    value_type operator()(Indices... idx) const {
        return op_(lhs_(idx...), rhs_(idx...));
    }
};

// ================================================================
//  内建二元算子
// ================================================================

namespace ops {

struct add_op {
    template <typename T, typename U>
    auto operator()(T a, U b) const { return a + b; }
};

struct sub_op {
    template <typename T, typename U>
    auto operator()(T a, U b) const { return a - b; }
};

struct mul_op {
    template <typename T, typename U>
    auto operator()(T a, U b) const { return a * b; }
};

struct div_op {
    template <typename T, typename U>
    auto operator()(T a, U b) const { return a / b; }
};

} // namespace ops

// ================================================================
//  make_binary 工厂
// ================================================================

namespace detail {

// ---- wrap_operand：将操作数统一包装为表达式 ----

/// @brief 表达式节点直接透传
template <typename E>
const E& wrap_operand(const expression_base<E>& e) {
    return e.self();
}

/// @brief 将视图包装成表达式叶节点
template <typename T, std::size_t R>
view_expr<T, R> wrap_operand(ndview<T, R> v) {
    return view_expr<T, R>(v);
}

/// @brief 将数组包装成表达式叶节点
template <typename T, std::size_t R, typename L, typename A>
view_expr<const T, R> wrap_operand(const ndarray<T, R, L, A>& arr) {
    return view_expr<const T, R>(arr.cview());
}

// ---- 获取操作数秩（标量没有秩，设为 0）----

template <typename T, typename = void>
struct operand_rank { static constexpr std::size_t value = 0; };

template <typename E>
struct operand_rank<E, std::enable_if_t<is_expression_v<E>>> {
    static constexpr std::size_t value = E::rank;
};

template <typename T, std::size_t R>
struct operand_rank<ndview<T, R>, void> {
    static constexpr std::size_t value = R;
};

template <typename T, std::size_t R, typename L, typename A>
struct operand_rank<ndarray<T, R, L, A>, void> {
    static constexpr std::size_t value = R;
};

template <typename T>
inline constexpr std::size_t operand_rank_v = operand_rank<std::decay_t<T>>::value;

// ---- 推出两个操作数的共同秩 ----

template <typename L, typename R>
struct common_rank {
    static constexpr std::size_t lv = operand_rank_v<L>;
    static constexpr std::size_t rv = operand_rank_v<R>;
    // 如果两个都有秩，必须相等；否则取非零的那个
    static constexpr std::size_t value =
        (lv != 0 && rv != 0) ? lv : (lv != 0 ? lv : rv);
};

template <typename L, typename R>
inline constexpr std::size_t common_rank_v = common_rank<L, R>::value;

// ---- 获取操作数的 shape（用于标量广播）----

template <typename E>
const auto& get_shape(const expression_base<E>& e) {
    return e.self().shape();
}

template <typename T, std::size_t R>
const shape<R>& get_shape(ndview<T, R> v) {
    return v.shape();
}

template <typename T, std::size_t R, typename L, typename A>
const shape<R>& get_shape(const ndarray<T, R, L, A>& arr) {
    return arr.shape();
}

/// @brief 依据目标形状将标量包装为广播表达式

template <std::size_t Rank, typename T>
scalar_expr<T, Rank> wrap_scalar(const T& val, const shape<Rank>& sh) {
    return scalar_expr<T, Rank>(val, sh);
}

// ---- make_binary_dispatch：统一处理各种组合 ----

/// @brief 构造两侧都为数组表达式的二元节点（支持广播）
template <typename Op, typename L, typename R>
auto make_binary_dispatch(L&& lhs, R&& rhs, Op op, std::false_type /*l_scalar*/,
                          std::false_type /*r_scalar*/) {
    auto le = wrap_operand(std::forward<L>(lhs));
    auto re = wrap_operand(std::forward<R>(rhs));
    using LE = decltype(le);
    using RE = decltype(re);
    static_assert(LE::rank == RE::rank, "rank mismatch in binary op");
    // 计算广播输出形状，用 broadcast_expr 包装两侧
    // 当形状已一致时，broadcast_expr 的索引映射退化为恒等
    auto out_sh = broadcast_shape(le.shape(), re.shape());
    auto ble = broadcast_expr<LE>(std::move(le), out_sh);
    auto bre = broadcast_expr<RE>(std::move(re), out_sh);
    return binary_expr<decltype(ble), decltype(bre), Op>(
        std::move(ble), std::move(bre), std::move(op));
}

/// @brief 构造左侧为标量的二元节点
template <typename Op, typename L, typename R>
auto make_binary_dispatch(const L& lhs, R&& rhs, Op op, std::true_type,
                          std::false_type) {
    auto re = wrap_operand(std::forward<R>(rhs));
    constexpr std::size_t Rank = std::decay_t<decltype(re)>::rank;
    auto le = wrap_scalar<Rank>(lhs, re.shape());
    return binary_expr<decltype(le), decltype(re), Op>(
        std::move(le), std::move(re), std::move(op));
}

/// @brief 构造右侧为标量的二元节点
template <typename Op, typename L, typename R>
auto make_binary_dispatch(L&& lhs, const R& rhs, Op op, std::false_type,
                          std::true_type) {
    auto le = wrap_operand(std::forward<L>(lhs));
    constexpr std::size_t Rank = std::decay_t<decltype(le)>::rank;
    auto re = wrap_scalar<Rank>(rhs, le.shape());
    return binary_expr<decltype(le), decltype(re), Op>(
        std::move(le), std::move(re), std::move(op));
}

template <typename Op, typename L, typename R>
auto make_binary(L&& lhs, R&& rhs, Op op) {
    using Ld = std::decay_t<L>;
    using Rd = std::decay_t<R>;
    return make_binary_dispatch(
        std::forward<L>(lhs), std::forward<R>(rhs), std::move(op),
        std::bool_constant<is_scalar_v<Ld>>{},
        std::bool_constant<is_scalar_v<Rd>>{});
}

} // namespace detail

// ================================================================
//  operator+, -, *, / 重载
// ================================================================

/**
 * @brief 生成逐元素四则运算符重载
 *
 * @details
 * 支持"同秩数组/视图/表达式"之间的逐元素运算（含 NumPy 风格广播），
 * 以及标量参与的逐元素运算。
 */

#define NDARRAY_DEFINE_BINARY_OP(OP_SYMBOL, OP_STRUCT)                        \
                                                                               \
/* expr OP expr */                                                             \
template <typename E1, typename E2>                                            \
auto operator OP_SYMBOL(const expression_base<E1>& lhs,                        \
                        const expression_base<E2>& rhs) {                      \
    return detail::make_binary(lhs.self(), rhs.self(), ops::OP_STRUCT{});       \
}                                                                              \
                                                                               \
/* expr OP scalar */                                                           \
template <typename E, typename S,                                              \
          std::enable_if_t<is_expression_v<E> && is_scalar_v<S>, int> = 0>     \
auto operator OP_SYMBOL(const expression_base<E>& lhs, const S& rhs) {        \
    return detail::make_binary(lhs.self(), rhs, ops::OP_STRUCT{});             \
}                                                                              \
                                                                               \
/* scalar OP expr */                                                           \
template <typename S, typename E,                                              \
          std::enable_if_t<is_scalar_v<S> && is_expression_v<E>, int> = 0>     \
auto operator OP_SYMBOL(const S& lhs, const expression_base<E>& rhs) {        \
    return detail::make_binary(lhs, rhs.self(), ops::OP_STRUCT{});             \
}                                                                              \
                                                                               \
/* ndview OP ndview */                                                         \
template <typename T1, std::size_t R1, typename T2, std::size_t R2>            \
auto operator OP_SYMBOL(ndview<T1, R1> lhs, ndview<T2, R2> rhs) {             \
    static_assert(R1 == R2, "rank mismatch");                                  \
    return detail::make_binary(lhs, rhs, ops::OP_STRUCT{});                    \
}                                                                              \
                                                                               \
/* ndarray OP ndarray */                                                       \
template <typename T1, std::size_t R1, typename L1, typename A1,               \
          typename T2, std::size_t R2, typename L2, typename A2>               \
auto operator OP_SYMBOL(const ndarray<T1, R1, L1, A1>& lhs,                    \
                        const ndarray<T2, R2, L2, A2>& rhs) {                  \
    static_assert(R1 == R2, "rank mismatch");                                  \
    return detail::make_binary(lhs, rhs, ops::OP_STRUCT{});                    \
}                                                                              \
                                                                               \
/* ndview OP scalar */                                                         \
template <typename T, std::size_t R, typename S,                               \
          std::enable_if_t<is_scalar_v<S>, int> = 0>                           \
auto operator OP_SYMBOL(ndview<T, R> lhs, const S& rhs) {                     \
    return detail::make_binary(lhs, rhs, ops::OP_STRUCT{});                    \
}                                                                              \
                                                                               \
/* scalar OP ndview */                                                         \
template <typename S, typename T, std::size_t R,                               \
          std::enable_if_t<is_scalar_v<S>, int> = 0>                           \
auto operator OP_SYMBOL(const S& lhs, ndview<T, R> rhs) {                     \
    return detail::make_binary(lhs, rhs, ops::OP_STRUCT{});                    \
}                                                                              \
                                                                               \
/* ndarray OP scalar */                                                        \
template <typename T, std::size_t R, typename L, typename A, typename S,       \
          std::enable_if_t<is_scalar_v<S>, int> = 0>                           \
auto operator OP_SYMBOL(const ndarray<T, R, L, A>& lhs, const S& rhs) {       \
    return detail::make_binary(lhs, rhs, ops::OP_STRUCT{});                    \
}                                                                              \
                                                                               \
/* scalar OP ndarray */                                                        \
template <typename S, typename T, std::size_t R, typename L, typename A,       \
          std::enable_if_t<is_scalar_v<S>, int> = 0>                           \
auto operator OP_SYMBOL(const S& lhs, const ndarray<T, R, L, A>& rhs) {       \
    return detail::make_binary(lhs, rhs, ops::OP_STRUCT{});                    \
}                                                                              \
                                                                               \
/* expr OP ndview */                                                           \
template <typename E, typename T, std::size_t R>                               \
auto operator OP_SYMBOL(const expression_base<E>& lhs, ndview<T, R> rhs) {    \
    return detail::make_binary(lhs.self(), rhs, ops::OP_STRUCT{});             \
}                                                                              \
                                                                               \
/* ndview OP expr */                                                           \
template <typename T, std::size_t R, typename E>                               \
auto operator OP_SYMBOL(ndview<T, R> lhs, const expression_base<E>& rhs) {    \
    return detail::make_binary(lhs, rhs.self(), ops::OP_STRUCT{});             \
}                                                                              \
                                                                               \
/* expr OP ndarray */                                                          \
template <typename E, typename T, std::size_t R, typename L, typename A>       \
auto operator OP_SYMBOL(const expression_base<E>& lhs,                         \
                        const ndarray<T, R, L, A>& rhs) {                      \
    return detail::make_binary(lhs.self(), rhs, ops::OP_STRUCT{});             \
}                                                                              \
                                                                               \
/* ndarray OP expr */                                                          \
template <typename T, std::size_t R, typename L, typename A, typename E>       \
auto operator OP_SYMBOL(const ndarray<T, R, L, A>& lhs,                        \
                        const expression_base<E>& rhs) {                       \
    return detail::make_binary(lhs, rhs.self(), ops::OP_STRUCT{});             \
}

NDARRAY_DEFINE_BINARY_OP(+, add_op)
NDARRAY_DEFINE_BINARY_OP(-, sub_op)
NDARRAY_DEFINE_BINARY_OP(*, mul_op)
NDARRAY_DEFINE_BINARY_OP(/, div_op)

#undef NDARRAY_DEFINE_BINARY_OP

} // namespace nd
