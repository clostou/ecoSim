/**
 * @file test_m1.cpp
 * @brief M1 核心容器与元数据的编译 + 基本功能冒烟测试
 */

#include "ndarray/ndarray.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

/**
 * @brief 执行 M1 阶段的核心容器与元数据冒烟测试
 */
void run_test_m1() {
    // 验证 shape 与 stride 元数据计算
    nd::shape<3> sh(2, 3, 4);
    assert(sh[0] == 2 && sh[1] == 3 && sh[2] == 4);
    assert(sh.total_size() == 24);

    auto st = nd::row_major::compute_strides(sh);
    assert(st[0] == 12 && st[1] == 4 && st[2] == 1);

    // 验证 ndarray 构造与基础属性
    auto a = nd::zeros<float>(2, 3, 4);
    assert(a.size() == 24);
    assert(a.ndim() == 3);
    assert(a.is_contiguous());

    // 验证零初始化结果
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                assert(a(i, j, k) == 0.0f);

    // 验证标量索引读写
    a(1, 2, 3) = 42.0f;
    assert(a(1, 2, 3) == 42.0f);

    // 验证 ones 与 full 工厂
    auto b = nd::ones<double>(4, 4);
    assert(b(0, 0) == 1.0 && b(3, 3) == 1.0);

    auto c = nd::full<int>(nd::shape<1>(5), 7);
    for (std::size_t i = 0; i < 5; ++i)
        assert(c(i) == 7);

    // 验证 clone 为深拷贝
    auto d = a.clone();
    assert(d(1, 2, 3) == 42.0f);
    d(1, 2, 3) = 0.0f;
    assert(a(1, 2, 3) == 42.0f);  // 原数组不受影响

    // 验证 fill 与 zero 操作
    d.fill(3.14f);
    assert(d(0, 0, 0) == 3.14f);
    d.zero();
    assert(d(0, 0, 0) == 0.0f);

    // 验证视图读写与共享语义
    auto v = a.view();
    assert(v.size() == 24);
    assert(v.is_contiguous());
    v(0, 0, 0) = 99.0f;
    assert(a(0, 0, 0) == 99.0f);  // 视图写入反映到原数组

    // 验证常量视图访问
    const auto& ca = a;
    auto cv = ca.view();
    assert(cv(0, 0, 0) == 99.0f);

    // 验证 arange 与 linspace 工厂
    auto r = nd::arange<float>(0.0f, 5.0f, 1.0f);
    assert(r.size() == 5);
    assert(r(0) == 0.0f && r(4) == 4.0f);

    auto ls = nd::linspace<double>(0.0, 1.0, 11);
    assert(ls.size() == 11);
    assert(ls(0) == 0.0);
    assert(std::abs(ls(10) - 1.0) < 1e-12);

    // 验证从已有缓冲区复制构造
    int raw[] = {10, 20, 30, 40};
    auto fb = nd::from_buffer(raw, nd::shape<2>(2, 2));
    assert(fb(0, 0) == 10 && fb(0, 1) == 20);
    assert(fb(1, 0) == 30 && fb(1, 1) == 40);

    // 验证 like 系列工厂
    auto zl = nd::zeros_like(b);
    assert(zl.size() == 16 && zl(0, 0) == 0.0);

    // 验证类型别名可正常实例化
    nd::vector<float> vec(nd::shape<1>(8));
    nd::matrix<double> mat(nd::shape<2>(3, 3));

    // 验证 shape 的 initializer_list 构造
    nd::shape<3> sh2{2, 3, 4};
    assert(sh2 == sh);

    std::cout << "All M1 tests passed." << std::endl;
}
