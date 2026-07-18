/**
 * @file test_m4.cpp
 * @brief M4 规约框架的功能测试
 */

#include "ndarray/ndarray.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

/// @brief 用固定容差比较两个浮点数是否近似相等
static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

/**
 * @brief 验证对全部元素求和的规约接口
 */
void test_sum_all() {
    // 1D
    auto v1 = nd::arange(1.0f, 6.0f, 1.0f); // [1,2,3,4,5]
    float s1 = nd::sum_all(v1);
    assert(near(s1, 15.0f));

    // 2D
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;
    // [[1,2,3],[4,5,6]]  sum=21
    assert(near(nd::sum_all(m), 21.0f));

    std::puts("  [PASS] test_sum_all");
}

/**
 * @brief 验证沿指定轴求和的规约接口
 */
void test_sum_axis() {
    // 2x3 矩阵，按轴 0 求和得到 shape(3)，按轴 1 求和得到 shape(2)
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;
    // [[1,2,3],[4,5,6]]

    auto s0 = nd::sum<0>(m); // [5,7,9]
    assert(s0.shape(0) == 3);
    assert(near(s0(0), 5.0f));
    assert(near(s0(1), 7.0f));
    assert(near(s0(2), 9.0f));

    auto s1 = nd::sum<1>(m); // [6,15]
    assert(s1.shape(0) == 2);
    assert(near(s1(0), 6.0f));
    assert(near(s1(1), 15.0f));

    std::puts("  [PASS] test_sum_axis");
}

/**
 * @brief 验证求积规约接口
 */
void test_prod() {
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;
    // [[1,2,3],[4,5,6]]

    float pa = nd::prod_all(m); // 720
    assert(near(pa, 720.0f));

    auto p0 = nd::prod<0>(m); // [4,10,18]
    assert(near(p0(0), 4.0f));
    assert(near(p0(1), 10.0f));
    assert(near(p0(2), 18.0f));

    auto p1 = nd::prod<1>(m); // [6,120]
    assert(near(p1(0), 6.0f));
    assert(near(p1(1), 120.0f));

    std::puts("  [PASS] test_prod");
}

/**
 * @brief 验证最小值与最大值规约接口
 */
void test_min_max() {
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    m(0, 0) = 5; m(0, 1) = 1; m(0, 2) = 3;
    m(1, 0) = 2; m(1, 1) = 8; m(1, 2) = 4;

    assert(near(nd::min_all(m), 1.0f));
    assert(near(nd::max_all(m), 8.0f));

    // min along axis 0 -> [2,1,3]
    auto mn0 = nd::min<0>(m);
    assert(near(mn0(0), 2.0f));
    assert(near(mn0(1), 1.0f));
    assert(near(mn0(2), 3.0f));

    // max along axis 1 -> [5,8]
    auto mx1 = nd::max<1>(m);
    assert(near(mx1(0), 5.0f));
    assert(near(mx1(1), 8.0f));

    std::puts("  [PASS] test_min_max");
}

/**
 * @brief 验证平均值规约接口
 */
void test_mean() {
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;
    // [[1,2,3],[4,5,6]]

    float ma = nd::mean_all(m); // 21/6 = 3.5
    assert(near(ma, 3.5f));

    auto m0 = nd::mean<0>(m); // [2.5, 3.5, 4.5]
    assert(near(m0(0), 2.5f));
    assert(near(m0(1), 3.5f));
    assert(near(m0(2), 4.5f));

    auto m1 = nd::mean<1>(m); // [2.0, 5.0]
    assert(near(m1(0), 2.0f));
    assert(near(m1(1), 5.0f));

    std::puts("  [PASS] test_mean");
}

/**
 * @brief 验证 keepdims 规约接口会保留长度为 1 的轴
 */
void test_keepdims() {
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;

    // sum_keepdims<0> -> shape(1,3)
    auto sk0 = nd::sum_keepdims<0>(m);
    assert(sk0.shape(0) == 1);
    assert(sk0.shape(1) == 3);
    assert(near(sk0(0, 0), 5.0f));
    assert(near(sk0(0, 1), 7.0f));
    assert(near(sk0(0, 2), 9.0f));

    // sum_keepdims<1> -> shape(2,1)
    auto sk1 = nd::sum_keepdims<1>(m);
    assert(sk1.shape(0) == 2);
    assert(sk1.shape(1) == 1);
    assert(near(sk1(0, 0), 6.0f));
    assert(near(sk1(1, 0), 15.0f));

    std::puts("  [PASS] test_keepdims");
}

/**
 * @brief 验证三维数组上的按轴规约行为
 */
void test_3d_reduce() {
    // shape (2,3,4)
    nd::ndarray<float, 3> a(nd::shape<3>(2, 3, 4));
    float v = 0.0f;
    for (nd::index_t i = 0; i < 2; ++i)
        for (nd::index_t j = 0; j < 3; ++j)
            for (nd::index_t k = 0; k < 4; ++k)
                a(i, j, k) = v++;

    // sum_all = 0+1+...+23 = 276
    assert(near(nd::sum_all(a), 276.0f));

    // sum<0> -> shape(3,4)
    auto s0 = nd::sum<0>(a);
    assert(s0.shape(0) == 3);
    assert(s0.shape(1) == 4);
    // s0(0,0) = a(0,0,0)+a(1,0,0) = 0+12 = 12
    assert(near(s0(0, 0), 12.0f));

    // sum<1> -> shape(2,4)
    auto s1 = nd::sum<1>(a);
    assert(s1.shape(0) == 2);
    assert(s1.shape(1) == 4);
    // s1(0,0) = a(0,0,0)+a(0,1,0)+a(0,2,0) = 0+4+8 = 12
    assert(near(s1(0, 0), 12.0f));

    // sum<2> -> shape(2,3)
    auto s2 = nd::sum<2>(a);
    assert(s2.shape(0) == 2);
    assert(s2.shape(1) == 3);
    // s2(0,0) = a(0,0,0)+a(0,0,1)+a(0,0,2)+a(0,0,3) = 0+1+2+3 = 6
    assert(near(s2(0, 0), 6.0f));
    // s2(1,2) = a(1,2,0)+a(1,2,1)+a(1,2,2)+a(1,2,3) = 20+21+22+23 = 86
    assert(near(s2(1, 2), 86.0f));

    std::puts("  [PASS] test_3d_reduce");
}

/**
 * @brief 验证用户自定义归约器可复用通用 reduce 内核
 */
void test_custom_reducer() {
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;
    // [[1,2,3],[4,5,6]]

    // sum of squares along axis 1
    auto ss = nd::reduce_axis<1>(m, 0.0f, [](float acc, float x) {
        return acc + x * x;
    });
    // row 0: 1+4+9=14, row 1: 16+25+36=77
    assert(near(ss(0), 14.0f));
    assert(near(ss(1), 77.0f));

    // reduce_all: sum of squares
    float total_ss = nd::reduce_all(m, 0.0f, [](float acc, float x) {
        return acc + x * x;
    });
    assert(near(total_ss, 91.0f));

    std::puts("  [PASS] test_custom_reducer");
}

/**
 * @brief 验证一维输入上的全量规约与 keepdims 语义
 */
void test_1d_reduce() {
    auto v = nd::arange(1.0f, 5.0f, 1.0f); // [1,2,3,4]
    assert(near(nd::sum_all(v), 10.0f));
    assert(near(nd::prod_all(v), 24.0f));
    assert(near(nd::min_all(v), 1.0f));
    assert(near(nd::max_all(v), 4.0f));
    assert(near(nd::mean_all(v), 2.5f));

    // keepdims on 1D -> shape(1)
    auto sk = nd::sum_keepdims<0>(v);
    assert(sk.shape(0) == 1);
    assert(near(sk(0), 10.0f));

    std::puts("  [PASS] test_1d_reduce");
}

/**
 * @brief 验证平均值的 keepdims 变体
 */
void test_mean_keepdims() {
    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float val = 1.0f;
    for (nd::index_t r = 0; r < 2; ++r)
        for (nd::index_t c = 0; c < 3; ++c)
            m(r, c) = val++;

    auto mk0 = nd::mean_keepdims<0>(m); // shape(1,3), values [2.5,3.5,4.5]
    assert(mk0.shape(0) == 1);
    assert(mk0.shape(1) == 3);
    assert(near(mk0(0, 0), 2.5f));
    assert(near(mk0(0, 1), 3.5f));
    assert(near(mk0(0, 2), 4.5f));

    std::puts("  [PASS] test_mean_keepdims");
}

/**
 * @brief 执行 M4 规约测试集
 */
void run_test_m4() {
    std::puts("===== M4 Reduce Tests =====");
    test_sum_all();
    test_sum_axis();
    test_prod();
    test_min_max();
    test_mean();
    test_keepdims();
    test_3d_reduce();
    test_custom_reducer();
    test_1d_reduce();
    test_mean_keepdims();
    std::puts("===== ALL M4 TESTS PASSED =====");
}
