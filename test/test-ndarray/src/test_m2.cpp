/**
 * @file test_m2.cpp
 * @brief M2 索引与形状变换的功能冒烟测试
 *
 * 覆盖：负下标、slice 描述符、reshape、flatten、
 *       permute / transpose、squeeze_axis、unsqueeze、take_axis
 */

#include "ndarray/ndarray.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

/**
 * @brief 执行 M1 能力的快速回归，确保 M2 新增功能未破坏基础容器行为
 */
static void test_m1_regression() {
    auto a = nd::zeros<float>(2, 3, 4);
    assert(a.size() == 24);
    a(1, 2, 3) = 42.0f;
    assert(a(1, 2, 3) == 42.0f);

    auto r = nd::arange<float>(0.0f, 5.0f, 1.0f);
    assert(r.size() == 5 && r(4) == 4.0f);

    std::cout << "[PASS] M1 regression" << std::endl;
}

/**
 * @brief 验证负下标在数组与视图上的归一化访问
 */
static void test_negative_index() {
    auto a = nd::arange<int>(0, 10, 1);   // [0..9]
    assert(a(-1) == 9);
    assert(a(-2) == 8);
    assert(a(-10) == 0);

    auto m = nd::zeros<int>(3, 4);
    m(0, 0) = 1;
    m(2, 3) = 99;
    assert(m(-1, -1) == 99);
    assert(m(-3, -4) == 1);

    // 通过视图测试
    auto v = a.view();
    assert(v(-1) == 9);
    assert(v(-5) == 5);

    std::cout << "[PASS] Negative index" << std::endl;
}

/**
 * @brief 验证 `view_all` 与 `range` 描述符组成的基础切片语义
 */
static void test_slice_basic() {
    // 一维切片
    auto a = nd::arange<int>(0, 10, 1);

    // range(2, 8, 2) -> [2, 4, 6]
    auto s1 = nd::slice(a, nd::range(2, 8, 2));
    assert(s1.ndim() == 1);
    assert(s1.size() == 3);
    assert(s1(0) == 2 && s1(1) == 4 && s1(2) == 6);

    // 二维切片
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    int raw[] = {0,1,2,3, 4,5,6,7, 8,9,10,11};
    auto m = nd::from_buffer(raw, nd::shape<2>(3, 4));

    // m(view_all, range(1, 3)) -> 第 1、2 列
    auto s2 = nd::slice(m, nd::view_all, nd::range(1, 3));
    assert(s2.shape(0) == 3 && s2.shape(1) == 2);
    assert(s2(0, 0) == 1 && s2(0, 1) == 2);
    assert(s2(2, 0) == 9 && s2(2, 1) == 10);

    // 负下标 range
    auto s3 = nd::slice(a, nd::range(-3, -1, 1));  // [7, 8]
    assert(s3.size() == 2 && s3(0) == 7 && s3(1) == 8);

    std::cout << "[PASS] slice basic" << std::endl;
}

/**
 * @brief 验证标量索引消轴与 `new_axis` 插轴语义
 */
static void test_slice_mixed() {
    int raw[] = {0,1,2,3, 4,5,6,7, 8,9,10,11};
    auto m = nd::from_buffer(raw, nd::shape<2>(3, 4));

    // 标量索引消去一个轴：m(1, view_all) -> [4,5,6,7]
    auto row = nd::slice(m, 1, nd::view_all);
    assert(row.ndim() == 1);
    assert(row.size() == 4);
    assert(row(0) == 4 && row(3) == 7);

    // new_axis 插入轴：从一维变二维
    auto a = nd::arange<int>(0, 5, 1);
    auto expanded = nd::slice(a, nd::new_axis, nd::view_all);  // shape [1, 5]
    assert(expanded.ndim() == 2);
    assert(expanded.shape(0) == 1 && expanded.shape(1) == 5);
    assert(expanded(0, 0) == 0 && expanded(0, 4) == 4);

    // 负标量索引
    auto last_row = nd::slice(m, -1, nd::view_all);
    assert(last_row.ndim() == 1);
    assert(last_row(0) == 8 && last_row(3) == 11);

    std::cout << "[PASS] slice mixed" << std::endl;
}

/**
 * @brief 验证切片视图写入会回写到底层数组
 */
static void test_slice_write() {
    auto a = nd::zeros<int>(3, 4);
    auto row = nd::slice(a, 1, nd::view_all);
    row(0) = 100;
    row(3) = 200;
    assert(a(1, 0) == 100 && a(1, 3) == 200);

    std::cout << "[PASS] slice write-through" << std::endl;
}

/**
 * @brief 验证 reshape 的形状变换与共享存储语义
 */
static void test_reshape() {
    auto a = nd::arange<int>(0, 24, 1);

    // 一维 -> 二维
    auto m = nd::reshape(a, nd::shape<2>(4, 6));
    assert(m.ndim() == 2);
    assert(m.shape(0) == 4 && m.shape(1) == 6);
    assert(m(0, 0) == 0 && m(3, 5) == 23);

    // 二维 -> 三维
    auto t = nd::reshape(a, nd::shape<3>(2, 3, 4));
    assert(t(0, 0, 0) == 0 && t(1, 2, 3) == 23);

    // reshape 返回视图，共享底层存储
    m(0, 0) = 999;
    assert(a(0) == 999);
    a(0) = 0;  // 还原

    // 常量 reshape
    const auto& ca = a;
    auto cm = nd::reshape(ca, nd::shape<2>(6, 4));
    assert(cm(0, 0) == 0);

    std::cout << "[PASS] reshape" << std::endl;
}

/**
 * @brief 验证 flatten 将连续多维视图展平为一维视图
 */
static void test_flatten() {
    auto a = nd::zeros<int>(2, 3, 4);
    // 写入一个已知值
    a(1, 2, 3) = 42;

    auto f = nd::flatten(a);
    assert(f.ndim() == 1);
    assert(f.size() == 24);
    assert(f(23) == 42);  // (1,2,3) = 1*12 + 2*4 + 3 = 23

    // 视图写入
    f(0) = 100;
    assert(a(0, 0, 0) == 100);

    std::cout << "[PASS] flatten" << std::endl;
}

/**
 * @brief 验证轴置换与矩阵转置返回的视图语义
 */
static void test_permute() {
    // 3x4 矩阵
    int raw[] = {0,1,2,3, 4,5,6,7, 8,9,10,11};
    auto m = nd::from_buffer(raw, nd::shape<2>(3, 4));

    // 二维转置
    auto mt = nd::transpose(m);
    assert(mt.shape(0) == 4 && mt.shape(1) == 3);
    assert(mt(0, 0) == 0 && mt(3, 2) == 11);
    assert(mt(1, 0) == 1 && mt(0, 2) == 8);

    // 三维置换
    auto t = nd::zeros<int>(2, 3, 4);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                t(i, j, k) = i * 100 + j * 10 + k;

    // permute<2, 0, 1>: (i,j,k) -> (k,i,j)
    auto tp = nd::permute<2, 0, 1>(t);
    assert(tp.shape(0) == 4 && tp.shape(1) == 2 && tp.shape(2) == 3);
    assert(tp(0, 0, 0) == 0);
    assert(tp(3, 1, 2) == 123);  // original t(1, 2, 3)

    // permute 返回视图
    tp(0, 0, 0) = 999;
    assert(t(0, 0, 0) == 999);

    std::cout << "[PASS] permute / transpose" << std::endl;
}

/**
 * @brief 验证 squeeze 与 unsqueeze 的轴插入/消除语义
 */
static void test_squeeze_unsqueeze() {
    // unsqueeze: [3, 4] -> [1, 3, 4]
    auto m = nd::zeros<int>(3, 4);
    m(1, 2) = 42;

    auto u0 = nd::unsqueeze<0>(m);
    assert(u0.ndim() == 3);
    assert(u0.shape(0) == 1 && u0.shape(1) == 3 && u0.shape(2) == 4);
    assert(u0(0, 1, 2) == 42);

    // unsqueeze: [3, 4] -> [3, 1, 4]
    auto u1 = nd::unsqueeze<1>(m);
    assert(u1.shape(0) == 3 && u1.shape(1) == 1 && u1.shape(2) == 4);
    assert(u1(1, 0, 2) == 42);

    // unsqueeze: [3, 4] -> [3, 4, 1]
    auto u2 = nd::unsqueeze<2>(m);
    assert(u2.shape(0) == 3 && u2.shape(1) == 4 && u2.shape(2) == 1);
    assert(u2(1, 2, 0) == 42);

    // squeeze_axis: [1, 3, 4] -> [3, 4]
    auto sq = nd::squeeze_axis<0>(u0);
    assert(sq.ndim() == 2);
    assert(sq.shape(0) == 3 && sq.shape(1) == 4);
    assert(sq(1, 2) == 42);

    // squeeze 写入穿透
    sq(0, 0) = 99;
    assert(m(0, 0) == 99);

    // 对连续输入，unsqueeze 后仍应被识别为连续
    auto a = nd::arange<int>(0, 6, 1);
    auto au = nd::unsqueeze<0>(a);
    assert(au.is_contiguous());

    std::cout << "[PASS] squeeze_axis / unsqueeze" << std::endl;
}

/**
 * @brief 验证按轴 gather 的复制语义与结果布局
 */
static void test_take_axis() {
    // 二维 gather
    int raw[] = {0,1,2,3, 4,5,6,7, 8,9,10,11};
    auto m = nd::from_buffer(raw, nd::shape<2>(3, 4));

    // take_axis<0>: 取第 0、2 行
    std::array<std::size_t, 2> row_ids{0, 2};
    auto picked_rows = nd::take_axis<0>(m, row_ids);
    assert(picked_rows.shape(0) == 2 && picked_rows.shape(1) == 4);
    assert(picked_rows(0, 0) == 0 && picked_rows(0, 3) == 3);
    assert(picked_rows(1, 0) == 8 && picked_rows(1, 3) == 11);

    // take_axis<1>: 取第 1、3 列
    std::array<std::size_t, 2> col_ids{1, 3};
    auto picked_cols = nd::take_axis<1>(m, col_ids);
    assert(picked_cols.shape(0) == 3 && picked_cols.shape(1) == 2);
    assert(picked_cols(0, 0) == 1 && picked_cols(0, 1) == 3);
    assert(picked_cols(2, 0) == 9 && picked_cols(2, 1) == 11);

    // take_axis 返回的是新数组，而不是视图
    picked_rows(0, 0) = 999;
    assert(m(0, 0) == 0);  // 原数组不受影响

    // 一维 gather
    auto a = nd::arange<int>(0, 10, 1);
    std::array<std::size_t, 4> ids{9, 0, 5, 5};
    auto g = nd::take_axis<0>(a, ids);
    assert(g.size() == 4);
    assert(g(0) == 9 && g(1) == 0 && g(2) == 5 && g(3) == 5);

    std::cout << "[PASS] take_axis" << std::endl;
}

/**
 * @brief 验证多个形状变换与切片组合时的结果一致性
 */
static void test_combined() {
    // arange -> reshape -> permute -> slice
    auto a = nd::arange<int>(0, 24, 1);
    auto t = nd::reshape(a, nd::shape<3>(2, 3, 4));   // (2,3,4)
    auto tp = nd::permute<1, 0, 2>(t);                 // (3,2,4)

    // slice: tp(1, view_all, range(0, 4, 2)) -> (2, 2)
    auto s = nd::slice(tp, 1, nd::view_all, nd::range(0, 4, 2));
    assert(s.ndim() == 2);
    assert(s.shape(0) == 2 && s.shape(1) == 2);
    // tp(1, i, k) = t(i, 1, k)
    // t(0,1,0)=4, t(0,1,2)=6, t(1,1,0)=16, t(1,1,2)=18
    assert(s(0, 0) == 4  && s(0, 1) == 6);
    assert(s(1, 0) == 16 && s(1, 1) == 18);

    // unsqueeze -> squeeze 往返
    auto u = nd::unsqueeze<1>(s);           // (2,1,2)
    auto sq = nd::squeeze_axis<1>(u);       // (2,2)
    assert(sq(0, 0) == 4 && sq(1, 1) == 18);

    std::cout << "[PASS] combined transforms" << std::endl;
}

/**
 * @brief 执行 M2 索引与形状变换测试集
 */
void run_test_m2() {
    test_m1_regression();
    test_negative_index();
    test_slice_basic();
    test_slice_mixed();
    test_slice_write();
    test_reshape();
    test_flatten();
    test_permute();
    test_squeeze_unsqueeze();
    test_take_axis();
    test_combined();

    std::cout << "\nAll M2 tests passed." << std::endl;
}
