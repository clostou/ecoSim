/**
 * @file test_m3.cpp
 * @brief M3 逐元素运算框架的功能冒烟测试
 *
 * 覆盖：四则运算、标量广播、取负、内建数学函数、
 *       自定义 transform、表达式组合与延迟求值、eval / eval_into
 */

#include "ndarray/ndarray.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

static constexpr double EPS = 1e-6;

/// @brief 用固定容差比较两个浮点数是否近似相等
static bool near(double a, double b) {
    return std::abs(a - b) < EPS;
}

/**
 * @brief 执行 M1/M2 能力的快速回归，确保表达式系统未破坏已有行为
 */
static void test_regression() {
    auto a = nd::zeros<float>(3, 4);
    a(1, 2) = 42.0f;
    assert(a(-1, -1) == 0.0f);
    assert(a(1, 2) == 42.0f);

    auto v = nd::slice(a, nd::view_all, nd::range(0, 2));
    assert(v.shape(1) == 2);

    std::cout << "[PASS] M1/M2 regression" << std::endl;
}

/**
 * @brief 验证数组之间的逐元素四则运算与求值结果
 */
static void test_arithmetic_array() {
    auto a = nd::arange<float>(0.0f, 6.0f, 1.0f);  // [0,1,2,3,4,5]
    auto b = nd::full<float>(nd::shape<1>(6), 10.0f);

    auto sum_expr = a + b;
    auto sum_arr  = nd::eval(sum_expr);
    assert(sum_arr.size() == 6);
    assert(near(sum_arr(0), 10.0f));
    assert(near(sum_arr(5), 15.0f));

    auto diff = nd::eval(a - b);
    assert(near(diff(0), -10.0f));
    assert(near(diff(5), -5.0f));

    auto prod = nd::eval(a * b);
    assert(near(prod(0), 0.0f));
    assert(near(prod(3), 30.0f));

    auto quot = nd::eval(b / (a + 1.0f));
    assert(near(quot(0), 10.0f));       // 10 / 1
    assert(near(quot(4), 2.0f));        // 10 / 5

    std::cout << "[PASS] arithmetic (array + array)" << std::endl;
}

/**
 * @brief 验证标量参与的逐元素运算与当前标量广播语义
 */
static void test_arithmetic_scalar() {
    auto a = nd::arange<float>(1.0f, 5.0f, 1.0f);  // [1,2,3,4]

    // 数组 + 标量
    auto r1 = nd::eval(a + 10.0f);
    assert(near(r1(0), 11.0f) && near(r1(3), 14.0f));

    // 标量 + 数组
    auto r2 = nd::eval(100.0f - a);
    assert(near(r2(0), 99.0f) && near(r2(3), 96.0f));

    // 标量 * 数组
    auto r3 = nd::eval(2.0f * a);
    assert(near(r3(0), 2.0f) && near(r3(3), 8.0f));

    // 数组 / 标量
    auto r4 = nd::eval(a / 2.0f);
    assert(near(r4(0), 0.5f) && near(r4(3), 2.0f));

    std::cout << "[PASS] arithmetic (scalar broadcast)" << std::endl;
}

/**
 * @brief 验证二维数组表达式组合的逐元素结果
 */
static void test_arithmetic_2d() {
    auto a = nd::ones<float>(3, 4);
    auto b = nd::full<float>(nd::shape<2>(3, 4), 2.0f);

    auto r = nd::eval(a + b * 3.0f);  // 1 + 2*3 = 7
    assert(r.shape(0) == 3 && r.shape(1) == 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            assert(near(r(i, j), 7.0f));

    std::cout << "[PASS] arithmetic 2D" << std::endl;
}

/**
 * @brief 验证一元取负表达式
 */
static void test_negate() {
    auto a = nd::arange<float>(1.0f, 4.0f, 1.0f);
    auto neg = nd::eval(-a);
    assert(near(neg(0), -1.0f));
    assert(near(neg(2), -3.0f));

    // 双重取负
    auto pos = nd::eval(-(-a));
    assert(near(pos(0), 1.0f));

    std::cout << "[PASS] negate" << std::endl;
}

/**
 * @brief 验证内建逐元素数学函数的基础正确性
 */
static void test_math_functions() {
    auto a = nd::full<float>(nd::shape<1>(3), 4.0f);

    // 绝对值
    auto neg_a = nd::eval(-a);
    auto abs_r = nd::eval(nd::abs(neg_a));
    assert(near(abs_r(0), 4.0f));

    // sqrt(4) = 2
    auto sqrt_r = nd::eval(nd::sqrt(a));
    assert(near(sqrt_r(0), 2.0f));

    // exp(0) = 1
    auto z = nd::zeros<float>(2);
    auto exp_r = nd::eval(nd::exp(z));
    assert(near(exp_r(0), 1.0f));

    // log(1) = 0
    auto o = nd::ones<float>(2);
    auto log_r = nd::eval(nd::log(o));
    assert(near(log_r(0), 0.0f));

    // sin(0) = 0, cos(0) = 1
    auto sin_r = nd::eval(nd::sin(z));
    auto cos_r = nd::eval(nd::cos(z));
    assert(near(sin_r(0), 0.0f));
    assert(near(cos_r(0), 1.0f));

    // tan(0) = 0
    auto tan_r = nd::eval(nd::tan(z));
    assert(near(tan_r(0), 0.0f));

    // asin(0) = 0, acos(1) = 0, atan(0) = 0
    auto asin_r = nd::eval(nd::asin(z));
    auto acos_r = nd::eval(nd::acos(o));
    auto atan_r = nd::eval(nd::atan(z));
    assert(near(asin_r(0), 0.0f));
    assert(near(acos_r(0), 0.0f));
    assert(near(atan_r(0), 0.0f));

    // tanh(0) = 0
    auto tanh_r = nd::eval(nd::tanh(z));
    assert(near(tanh_r(0), 0.0f));

    // sigmoid(0) = 0.5
    auto sig_r = nd::eval(nd::sigmoid(z));
    assert(near(sig_r(0), 0.5f));

    std::cout << "[PASS] math functions" << std::endl;
}

/**
 * @brief 验证自定义一元变换扩展点 `transform`
 */
static void test_transform() {
    auto a = nd::arange<float>(-3.0f, 4.0f, 1.0f);  // [-3,-2,-1,0,1,2,3]

    // ReLU
    auto relu = nd::eval(nd::transform(a, [](float x) {
        return x > 0.0f ? x : 0.0f;
    }));
    assert(near(relu(0), 0.0f));   // -3 被截断
    assert(near(relu(3), 0.0f));   // 0
    assert(near(relu(6), 3.0f));   // 3

    // 裁剪到 [0, 1]
    auto clipped = nd::eval(nd::transform(a, [](float x) {
        return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    }));
    assert(near(clipped(0), 0.0f));
    assert(near(clipped(4), 1.0f));
    assert(near(clipped(6), 1.0f));

    std::cout << "[PASS] transform (custom unary)" << std::endl;
}

/**
 * @brief 验证多层表达式组合的延迟求值与数值正确性
 */
static void test_expression_composition() {
    auto a = nd::arange<float>(1.0f, 5.0f, 1.0f);  // [1,2,3,4]

    // (a * a + a) / 2.0  = (x^2 + x)/2
    auto r = nd::eval((a * a + a) / 2.0f);
    // x=1: 1, x=2: 3, x=3: 6, x=4: 10
    assert(near(r(0), 1.0f));
    assert(near(r(1), 3.0f));
    assert(near(r(2), 6.0f));
    assert(near(r(3), 10.0f));

    // sqrt(a * a) == abs(a)
    auto r2 = nd::eval(nd::sqrt(a * a));
    assert(near(r2(0), 1.0f));
    assert(near(r2(3), 4.0f));

    // exp(log(a)) ≈ a
    auto r3 = nd::eval(nd::exp(nd::log(a)));
    for (std::size_t i = 0; i < 4; ++i)
        assert(near(r3(i), a(i)));

    std::cout << "[PASS] expression composition" << std::endl;
}

/**
 * @brief 验证 `eval_into` 可写入数组或视图目标
 */
static void test_eval_into() {
    auto a = nd::arange<float>(0.0f, 6.0f, 1.0f);
    auto b = nd::ones<float>(6);

    auto dst = nd::zeros<float>(6);
    nd::eval_into(dst, a + b);
    assert(near(dst(0), 1.0f));
    assert(near(dst(5), 6.0f));

    // 将表达式写入视图
    auto dst2 = nd::zeros<float>(6);
    nd::eval_into(dst2.view(), a * 2.0f);
    assert(near(dst2(0), 0.0f));
    assert(near(dst2(3), 6.0f));

    std::cout << "[PASS] eval_into" << std::endl;
}

/**
 * @brief 验证视图作为表达式操作数时的正确性
 */
static void test_view_ops() {
    auto a = nd::arange<float>(0.0f, 12.0f, 1.0f);
    auto m = nd::reshape(a, nd::shape<2>(3, 4));

    // 取第 1 行，加上标量
    auto row = nd::slice(m, 1, nd::view_all);  // [4,5,6,7]
    auto r = nd::eval(row + 10.0f);
    assert(r.ndim() == 1 && r.size() == 4);
    assert(near(r(0), 14.0f) && near(r(3), 17.0f));

    // 视图 * 视图
    auto row0 = nd::slice(m, 0, nd::view_all);  // [0,1,2,3]
    auto r2 = nd::eval(row0 * row);
    assert(near(r2(0), 0.0f));   // 0*4
    assert(near(r2(3), 21.0f));  // 3*7

    std::cout << "[PASS] view ops" << std::endl;
}

/**
 * @brief 执行 M3 逐元素表达式系统测试集
 */
void run_test_m3() {
    test_regression();
    test_arithmetic_array();
    test_arithmetic_scalar();
    test_arithmetic_2d();
    test_negate();
    test_math_functions();
    test_transform();
    test_expression_composition();
    test_eval_into();
    test_view_ops();

    std::cout << "\nAll M3 tests passed." << std::endl;
}
