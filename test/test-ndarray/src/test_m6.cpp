/**
 * @file test_m6.cpp
 * @brief M6 广播机制测试
 *
 * 覆盖以下场景：
 *   1. broadcastable / broadcast_shape 元数据判定
 *   2. pad_shape_left 左补 1
 *   3. broadcast_to 显式广播
 *   4. 同秩尾轴广播的逐元素四则运算
 *   5. 行向量 + 列向量广播
 *   6. 矩阵 + 行向量广播
 *   7. 标量广播（回归）
 *   8. broadcast_expr 与数学函数组合
 *   9. 三维张量广播
 *  10. 代表性 Linear bias 加法（广播替代显式循环）
 */

#include "ndarray/ndarray.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

/// @brief 浮点近似比较
static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// ================================================================
//  1. 元数据：broadcastable / broadcast_shape
// ================================================================

void test_broadcast_metadata() {
    using nd::shape;
    using nd::broadcastable;
    using nd::broadcast_shape;

    // 同形状
    shape<2> a{3, 4};
    shape<2> b{3, 4};
    assert((broadcastable(a, b)));
    auto c = broadcast_shape(a, b);
    assert(c[0] == 3 && c[1] == 4);

    // 尾轴广播 (3,4) + (1,4)
    shape<2> d{1, 4};
    assert((broadcastable(a, d)));
    auto e = broadcast_shape(a, d);
    assert(e[0] == 3 && e[1] == 4);

    // 首轴广播 (3,1) + (3,4)
    shape<2> f{3, 1};
    assert((broadcastable(f, a)));
    auto g = broadcast_shape(f, a);
    assert(g[0] == 3 && g[1] == 4);

    // 不兼容
    shape<2> h{2, 3};
    assert((!broadcastable(a, h)));

    // (1,1) + (3,4)
    shape<2> i{1, 1};
    assert((broadcastable(i, a)));
    auto j = broadcast_shape(i, a);
    assert(j[0] == 3 && j[1] == 4);

    // 3D 广播 (2,3,4) + (1,1,4)
    shape<3> s3a{2, 3, 4};
    shape<3> s3b{1, 1, 4};
    assert((broadcastable(s3a, s3b)));
    auto s3c = broadcast_shape(s3a, s3b);
    assert(s3c[0] == 2 && s3c[1] == 3 && s3c[2] == 4);

    std::printf("  test_broadcast_metadata ... OK\n");
}

// ================================================================
//  2. pad_shape_left
// ================================================================

void test_pad_shape_left() {
    using nd::shape;
    using nd::pad_shape_left;

    shape<2> s2{3, 4};
    auto s4 = pad_shape_left<4>(s2);
    assert(s4[0] == 1 && s4[1] == 1 && s4[2] == 3 && s4[3] == 4);

    shape<1> s1{5};
    auto s3 = pad_shape_left<3>(s1);
    assert(s3[0] == 1 && s3[1] == 1 && s3[2] == 5);

    // 同秩不变
    auto same = pad_shape_left<2>(s2);
    assert(same[0] == 3 && same[1] == 4);

    std::printf("  test_pad_shape_left ... OK\n");
}

// ================================================================
//  3. broadcast_to 显式广播
// ================================================================

void test_broadcast_to() {
    // shape (1,4) -> (3,4)
    auto src = nd::ones<float>(nd::shape<2>{1, 4});
    src(0, 0) = 10.f;
    src(0, 1) = 20.f;
    src(0, 2) = 30.f;
    src(0, 3) = 40.f;

    auto expr = nd::broadcast_to(src, nd::shape<2>{3, 4});
    auto result = nd::eval(expr);

    assert((result.shape() == nd::shape<2>{3, 4}));
    for (int r = 0; r < 3; ++r) {
        assert((near(result(r, 0), 10.f)));
        assert((near(result(r, 1), 20.f)));
        assert((near(result(r, 2), 30.f)));
        assert((near(result(r, 3), 40.f)));
    }

    std::printf("  test_broadcast_to ... OK\n");
}

// ================================================================
//  4. 同秩尾轴广播加法：(3,4) + (1,4)
// ================================================================

void test_broadcast_add_tail() {
    auto a = nd::zeros<float>(nd::shape<2>{3, 4});
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            a(r, c) = static_cast<float>(r * 4 + c);

    auto b = nd::zeros<float>(nd::shape<2>{1, 4});
    for (int c = 0; c < 4; ++c)
        b(0, c) = static_cast<float>(c * 100);

    auto result = nd::eval(a.view() + b.view());
    assert((result.shape() == nd::shape<2>{3, 4}));

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            assert((near(result(r, c), static_cast<float>(r * 4 + c + c * 100))));

    std::printf("  test_broadcast_add_tail ... OK\n");
}

// ================================================================
//  5. 行向量 + 列向量广播：(1,4) + (3,1) -> (3,4)
// ================================================================

void test_broadcast_row_col() {
    auto row = nd::zeros<float>(nd::shape<2>{1, 4});
    for (int c = 0; c < 4; ++c) row(0, c) = static_cast<float>(c);

    auto col = nd::zeros<float>(nd::shape<2>{3, 1});
    for (int r = 0; r < 3; ++r) col(r, 0) = static_cast<float>(r * 10);

    auto result = nd::eval(row.view() + col.view());
    assert((result.shape() == nd::shape<2>{3, 4}));

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            assert((near(result(r, c), static_cast<float>(r * 10 + c))));

    std::printf("  test_broadcast_row_col ... OK\n");
}

// ================================================================
//  6. 矩阵 + 行向量广播：ndarray + ndarray
// ================================================================

void test_broadcast_matrix_row() {
    auto mat = nd::zeros<float>(nd::shape<2>{2, 3});
    mat(0, 0) = 1; mat(0, 1) = 2; mat(0, 2) = 3;
    mat(1, 0) = 4; mat(1, 1) = 5; mat(1, 2) = 6;

    auto bias = nd::zeros<float>(nd::shape<2>{1, 3});
    bias(0, 0) = 10; bias(0, 1) = 20; bias(0, 2) = 30;

    // ndarray + ndarray
    auto result = nd::eval(mat.view() + bias.view());
    assert((near(result(0, 0), 11)));
    assert((near(result(0, 1), 22)));
    assert((near(result(0, 2), 33)));
    assert((near(result(1, 0), 14)));
    assert((near(result(1, 1), 25)));
    assert((near(result(1, 2), 36)));

    std::printf("  test_broadcast_matrix_row ... OK\n");
}

// ================================================================
//  7. 标量广播回归
// ================================================================

void test_broadcast_scalar_regression() {
    auto a = nd::ones<float>(nd::shape<2>{2, 3});
    auto result = nd::eval(a.view() + 5.0f);

    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            assert((near(result(r, c), 6.0f)));

    auto result2 = nd::eval(2.0f * a.view());
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            assert((near(result2(r, c), 2.0f)));

    std::printf("  test_broadcast_scalar_regression ... OK\n");
}

// ================================================================
//  8. 广播 + 数学函数组合
// ================================================================

void test_broadcast_with_math() {
    auto a = nd::zeros<float>(nd::shape<2>{2, 3});
    a(0, 0) = 0; a(0, 1) = 1; a(0, 2) = 4;
    a(1, 0) = 9; a(1, 1) = 16; a(1, 2) = 25;

    auto scale = nd::zeros<float>(nd::shape<2>{1, 3});
    scale(0, 0) = 1; scale(0, 1) = 2; scale(0, 2) = 3;

    // sqrt(a) * scale (broadcast)
    auto result = nd::eval(nd::sqrt(a.view()) * scale.view());
    assert((result.shape() == nd::shape<2>{2, 3}));

    assert((near(result(0, 0), 0.f * 1.f)));
    assert((near(result(0, 1), 1.f * 2.f)));
    assert((near(result(0, 2), 2.f * 3.f)));
    assert((near(result(1, 0), 3.f * 1.f)));
    assert((near(result(1, 1), 4.f * 2.f)));
    assert((near(result(1, 2), 5.f * 3.f)));

    std::printf("  test_broadcast_with_math ... OK\n");
}

// ================================================================
//  9. 四则运算全覆盖
// ================================================================

void test_broadcast_all_ops() {
    auto a = nd::full<float>(nd::shape<2>{2, 3}, 6.0f);
    auto b = nd::zeros<float>(nd::shape<2>{1, 3});
    b(0, 0) = 1; b(0, 1) = 2; b(0, 2) = 3;

    // 加法
    auto add_r = nd::eval(a.view() + b.view());
    assert((near(add_r(0, 0), 7) && near(add_r(1, 2), 9)));

    // 减法
    auto sub_r = nd::eval(a.view() - b.view());
    assert((near(sub_r(0, 0), 5) && near(sub_r(1, 2), 3)));

    // 乘法
    auto mul_r = nd::eval(a.view() * b.view());
    assert((near(mul_r(0, 0), 6) && near(mul_r(1, 2), 18)));

    // 除法
    auto div_r = nd::eval(a.view() / b.view());
    assert((near(div_r(0, 0), 6) && near(div_r(1, 1), 3)));

    std::printf("  test_broadcast_all_ops ... OK\n");
}

// ================================================================
//  10. 三维张量广播：(2,3,4) + (1,1,4)
// ================================================================

void test_broadcast_3d() {
    auto a = nd::zeros<float>(nd::shape<3>{2, 3, 4});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                a(i, j, k) = static_cast<float>(i * 12 + j * 4 + k);

    auto b = nd::zeros<float>(nd::shape<3>{1, 1, 4});
    for (int k = 0; k < 4; ++k)
        b(0, 0, k) = static_cast<float>(k * 100);

    auto result = nd::eval(a.view() + b.view());
    assert((result.shape() == nd::shape<3>{2, 3, 4}));

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                assert((near(result(i, j, k),
                            static_cast<float>(i * 12 + j * 4 + k + k * 100))));

    std::printf("  test_broadcast_3d ... OK\n");
}

// ================================================================
//  11. 三维 (2,3,1) + (1,1,4) -> (2,3,4)
// ================================================================

void test_broadcast_3d_mixed() {
    auto a = nd::zeros<float>(nd::shape<3>{2, 3, 1});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            a(i, j, 0) = static_cast<float>(i * 3 + j);

    auto b = nd::zeros<float>(nd::shape<3>{1, 1, 4});
    for (int k = 0; k < 4; ++k)
        b(0, 0, k) = static_cast<float>(k * 10);

    auto result = nd::eval(a.view() + b.view());
    assert((result.shape() == nd::shape<3>{2, 3, 4}));

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                assert((near(result(i, j, k),
                            static_cast<float>(i * 3 + j + k * 10))));

    std::printf("  test_broadcast_3d_mixed ... OK\n");
}

// ================================================================
//  12. 代表性 Linear bias 加法
// ================================================================

void test_linear_bias_broadcast() {
    // y = x @ W + bias
    // x: (4, 8), W: (8, 3), bias: (1, 3)
    // out: (4, 3) + (1, 3) -> (4, 3) (broadcast)

    auto out = nd::zeros<float>(nd::shape<2>{4, 3});
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            out(i, j) = static_cast<float>(i * 3 + j);

    auto bias = nd::zeros<float>(nd::shape<2>{1, 3});
    bias(0, 0) = 100; bias(0, 1) = 200; bias(0, 2) = 300;

    // 之前需要对每行手动加 bias，现在直接广播
    auto result = nd::eval(out.view() + bias.view());
    assert((result.shape() == nd::shape<2>{4, 3}));

    for (int i = 0; i < 4; ++i) {
        assert((near(result(i, 0), static_cast<float>(i * 3 + 0 + 100))));
        assert((near(result(i, 1), static_cast<float>(i * 3 + 1 + 200))));
        assert((near(result(i, 2), static_cast<float>(i * 3 + 2 + 300))));
    }

    std::printf("  test_linear_bias_broadcast ... OK\n");
}

// ================================================================
//  13. 同形状回归（确保广播不破坏已有路径）
// ================================================================

void test_same_shape_regression() {
    auto a = nd::zeros<float>(nd::shape<2>{3, 4});
    auto b = nd::zeros<float>(nd::shape<2>{3, 4});
    for (int i = 0; i < 12; ++i) {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = static_cast<float>(i * 2);
    }

    auto result = nd::eval(a.view() + b.view());
    for (int i = 0; i < 12; ++i)
        assert((near(result.data()[i], static_cast<float>(i * 3))));

    std::printf("  test_same_shape_regression ... OK\n");
}

// ================================================================
//  14. eval_into + 广播
// ================================================================

void test_broadcast_eval_into() {
    auto a = nd::zeros<float>(nd::shape<2>{2, 3});
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    auto b = nd::zeros<float>(nd::shape<2>{1, 3});
    b(0, 0) = 10; b(0, 1) = 20; b(0, 2) = 30;

    auto dst = nd::zeros<float>(nd::shape<2>{2, 3});
    nd::eval_into(dst.view(), a.view() + b.view());

    assert((near(dst(0, 0), 11)));
    assert((near(dst(0, 2), 33)));
    assert((near(dst(1, 1), 25)));

    std::printf("  test_broadcast_eval_into ... OK\n");
}

// ================================================================
//  main
// ================================================================

/**
 * @brief M6 广播机制测试入口
 */
void run_test_m6() {
    std::printf("=== M6 broadcast tests ===\n");

    test_broadcast_metadata();
    test_pad_shape_left();
    test_broadcast_to();
    test_broadcast_add_tail();
    test_broadcast_row_col();
    test_broadcast_matrix_row();
    test_broadcast_scalar_regression();
    test_broadcast_with_math();
    test_broadcast_all_ops();
    test_broadcast_3d();
    test_broadcast_3d_mixed();
    test_linear_bias_broadcast();
    test_same_shape_regression();
    test_broadcast_eval_into();

    std::printf("=== all M6 broadcast tests passed ===\n");
}
