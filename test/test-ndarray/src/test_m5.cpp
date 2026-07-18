/**
 * @file test_m5.cpp
 * @brief M5 einsum 核心子集测试
 */

#include "ndarray/ndarray.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

/// @brief 用固定容差比较两个浮点数是否近似相等
static bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

/**
 * @brief 验证向量点积接口
 */
void test_dot() {
    auto a = nd::arange(1.0f, 4.0f, 1.0f); // [1,2,3]
    auto b = nd::arange(4.0f, 7.0f, 1.0f); // [4,5,6]
    // dot = 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    float d = nd::dot(a, b);
    assert(near(d, 32.0f));
    std::puts("  [PASS] test_dot");
}

/**
 * @brief 验证矩阵乘法接口
 */
void test_matmul() {
    // A(2,3), B(3,2)
    nd::ndarray<float, 2> A(nd::shape<2>(2, 3));
    nd::ndarray<float, 2> B(nd::shape<2>(3, 2));
    // A = [[1,2,3],[4,5,6]]
    float v = 1.0f;
    for (nd::index_t i = 0; i < 2; ++i)
        for (nd::index_t j = 0; j < 3; ++j)
            A(i, j) = v++;
    // B = [[7,8],[9,10],[11,12]]
    v = 7.0f;
    for (nd::index_t i = 0; i < 3; ++i)
        for (nd::index_t j = 0; j < 2; ++j)
            B(i, j) = v++;

    auto C = nd::matmul(A, B);
    // C(0,0) = 1*7+2*9+3*11 = 7+18+33 = 58
    // C(0,1) = 1*8+2*10+3*12 = 8+20+36 = 64
    // C(1,0) = 4*7+5*9+6*11 = 28+45+66 = 139
    // C(1,1) = 4*8+5*10+6*12 = 32+50+72 = 154
    assert(C.shape(0) == 2 && C.shape(1) == 2);
    assert(near(C(0, 0), 58.0f));
    assert(near(C(0, 1), 64.0f));
    assert(near(C(1, 0), 139.0f));
    assert(near(C(1, 1), 154.0f));
    std::puts("  [PASS] test_matmul");
}

/**
 * @brief 验证批矩阵乘法接口
 */
void test_batched_matmul() {
    // shape: (2,2,3) x (2,3,2) -> (2,2,2)
    nd::ndarray<float, 3> A(nd::shape<3>(2, 2, 3));
    nd::ndarray<float, 3> B(nd::shape<3>(2, 3, 2));
    // batch 0: A[0] = [[1,2,3],[4,5,6]], B[0] = [[1,0],[0,1],[1,0]]
    A(0,0,0)=1; A(0,0,1)=2; A(0,0,2)=3;
    A(0,1,0)=4; A(0,1,1)=5; A(0,1,2)=6;
    B(0,0,0)=1; B(0,0,1)=0;
    B(0,1,0)=0; B(0,1,1)=1;
    B(0,2,0)=1; B(0,2,1)=0;
    // C[0] = [[1*1+2*0+3*1, 1*0+2*1+3*0],[4*1+5*0+6*1, 4*0+5*1+6*0]]
    //      = [[4,2],[10,5]]
    // batch 1: identity matrices
    A(1,0,0)=1; A(1,0,1)=0; A(1,0,2)=0;
    A(1,1,0)=0; A(1,1,1)=1; A(1,1,2)=0;
    B(1,0,0)=2; B(1,0,1)=3;
    B(1,1,0)=4; B(1,1,1)=5;
    B(1,2,0)=6; B(1,2,1)=7;
    // C[1] = [[2,3],[4,5]]

    auto C = nd::batched_matmul(A, B);
    assert(C.shape(0) == 2 && C.shape(1) == 2 && C.shape(2) == 2);
    assert(near(C(0,0,0), 4.0f));
    assert(near(C(0,0,1), 2.0f));
    assert(near(C(0,1,0), 10.0f));
    assert(near(C(0,1,1), 5.0f));
    assert(near(C(1,0,0), 2.0f));
    assert(near(C(1,0,1), 3.0f));
    assert(near(C(1,1,0), 4.0f));
    assert(near(C(1,1,1), 5.0f));
    std::puts("  [PASS] test_batched_matmul");
}

/**
 * @brief 验证向量外积接口
 */
void test_outer() {
    auto a = nd::arange(1.0f, 4.0f, 1.0f); // [1,2,3]
    auto b = nd::arange(4.0f, 6.0f, 1.0f); // [4,5]
    auto C = nd::outer(a, b);
    // [[4,5],[8,10],[12,15]]
    assert(C.shape(0) == 3 && C.shape(1) == 2);
    assert(near(C(0,0), 4.0f));
    assert(near(C(0,1), 5.0f));
    assert(near(C(1,0), 8.0f));
    assert(near(C(1,1), 10.0f));
    assert(near(C(2,0), 12.0f));
    assert(near(C(2,1), 15.0f));
    std::puts("  [PASS] test_outer");
}

/**
 * @brief 验证二维矩阵求迹的标量结果
 */
void test_trace_2d() {
    nd::ndarray<float, 2> m(nd::shape<2>(3, 3));
    float v = 1.0f;
    for (nd::index_t i = 0; i < 3; ++i)
        for (nd::index_t j = 0; j < 3; ++j)
            m(i, j) = v++;
    // [[1,2,3],[4,5,6],[7,8,9]]  trace = 1+5+9 = 15
    float tr = nd::trace<0, 1>(m);
    assert(near(tr, 15.0f));
    std::puts("  [PASS] test_trace_2d");
}

/**
 * @brief 验证高阶张量沿两个轴求迹后的降秩结果
 */
void test_trace_3d() {
    // shape (2, 3, 3), trace along axes 1,2 -> shape(2)
    nd::ndarray<float, 3> a(nd::shape<3>(2, 3, 3));
    a.zero();
    // batch 0: identity * 2
    a(0,0,0)=2; a(0,1,1)=2; a(0,2,2)=2;
    // batch 1: identity * 3
    a(1,0,0)=3; a(1,1,1)=3; a(1,2,2)=3;

    auto tr = nd::trace<1, 2>(a);
    assert(tr.shape(0) == 2);
    assert(near(tr(0), 6.0f));
    assert(near(tr(1), 9.0f));
    std::puts("  [PASS] test_trace_3d");
}

/**
 * @brief 验证通用 `contract` 与具名 matmul 的一致性
 */
void test_contract_generic() {
    // ij,jk->ik (same as matmul, but via direct contract call)
    using L = nd::einsum::label_seq<'i','j'>;
    using R = nd::einsum::label_seq<'j','k'>;
    using O = nd::einsum::label_seq<'i','k'>;

    nd::ndarray<float, 2> A(nd::shape<2>(2, 3));
    nd::ndarray<float, 2> B(nd::shape<2>(3, 2));
    float v = 1.0f;
    for (nd::index_t i = 0; i < 2; ++i)
        for (nd::index_t j = 0; j < 3; ++j) A(i, j) = v++;
    v = 7.0f;
    for (nd::index_t i = 0; i < 3; ++i)
        for (nd::index_t j = 0; j < 2; ++j) B(i, j) = v++;

    auto C = nd::einsum::contract<L, R, O>(A, B);
    assert(near(C(0, 0), 58.0f));
    assert(near(C(1, 1), 154.0f));
    std::puts("  [PASS] test_contract_generic");
}

/**
 * @brief 验证输出标签重排时的物理转置逻辑
 */
void test_contract_reorder() {
    // ij,jk->ki (transposed output)
    using L = nd::einsum::label_seq<'i','j'>;
    using R = nd::einsum::label_seq<'j','k'>;
    using O = nd::einsum::label_seq<'k','i'>;

    nd::ndarray<float, 2> A(nd::shape<2>(2, 3));
    nd::ndarray<float, 2> B(nd::shape<2>(3, 2));
    float v = 1.0f;
    for (nd::index_t i = 0; i < 2; ++i)
        for (nd::index_t j = 0; j < 3; ++j) A(i, j) = v++;
    v = 7.0f;
    for (nd::index_t i = 0; i < 3; ++i)
        for (nd::index_t j = 0; j < 2; ++j) B(i, j) = v++;

    auto C = nd::einsum::contract<L, R, O>(A, B);
    // normal matmul: [[58,64],[139,154]]
    // transposed: C(k,i): C(0,0)=58, C(0,1)=139, C(1,0)=64, C(1,1)=154
    assert(C.shape(0) == 2 && C.shape(1) == 2);
    assert(near(C(0, 0), 58.0f));
    assert(near(C(0, 1), 139.0f));
    assert(near(C(1, 0), 64.0f));
    assert(near(C(1, 1), 154.0f));
    std::puts("  [PASS] test_contract_reorder");
}

/**
 * @brief 验证一元标签规约 `reduce_labels`
 */
void test_reduce_labels() {
    // ij->i (sum over j)
    using In = nd::einsum::label_seq<'i','j'>;
    using Out = nd::einsum::label_seq<'i'>;

    nd::ndarray<float, 2> m(nd::shape<2>(2, 3));
    float v = 1.0f;
    for (nd::index_t i = 0; i < 2; ++i)
        for (nd::index_t j = 0; j < 3; ++j) m(i, j) = v++;
    // [[1,2,3],[4,5,6]]
    auto r = nd::einsum::reduce_labels<In, Out>(m);
    assert(r.shape(0) == 2);
    assert(near(r(0), 6.0f));
    assert(near(r(1), 15.0f));
    std::puts("  [PASS] test_reduce_labels");
}

/**
 * @brief 验证编译期标签元编程工具的静态性质
 */
void test_labels_compile_time() {
    using namespace nd::einsum;

    // contains
    using S = label_seq<'a','b','c'>;
    static_assert(contains_v<S, 'a'>, "");
    static_assert(contains_v<S, 'c'>, "");
    static_assert(!contains_v<S, 'd'>, "");

    // index_of
    static_assert(index_of_v<S, 'a'> == 0, "");
    static_assert(index_of_v<S, 'b'> == 1, "");
    static_assert(index_of_v<S, 'c'> == 2, "");

    // intersect
    using A = label_seq<'a','b','c'>;
    using B = label_seq<'b','c','d'>;
    using AB = intersect_t<A, B>;
    static_assert(AB::size == 2, "");
    static_assert(AB::at(0) == 'b', "");
    static_assert(AB::at(1) == 'c', "");

    // diff
    using AmB = diff_t<A, B>;
    static_assert(AmB::size == 1, "");
    static_assert(AmB::at(0) == 'a', "");

    // classify
    using Lhs = label_seq<'i','j'>;
    using Rhs = label_seq<'j','k'>;
    using Out = label_seq<'i','k'>;
    using cl = classify<Lhs, Rhs, Out>;
    static_assert(cl::B::size == 0, "");
    static_assert(cl::M::size == 1 && cl::M::at(0) == 'i', "");
    static_assert(cl::N::size == 1 && cl::N::at(0) == 'k', "");
    static_assert(cl::K::size == 1 && cl::K::at(0) == 'j', "");

    std::puts("  [PASS] test_labels_compile_time");
}

/**
 * @brief 验证较高阶收缩模式可表达注意力中的相似度计算
 */
void test_higher_rank_contract() {
    // Attention-style: qe,ke->qk (Query dot Key^T per-element)
    // Q(4,3), K(5,3) -> S(4,5) = Q @ K^T essentially
    // This is: contract<'q','e'>,<'k','e'>,<'q','k'>
    using QL = nd::einsum::label_seq<'q','e'>;
    using KL = nd::einsum::label_seq<'k','e'>;
    using OL = nd::einsum::label_seq<'q','k'>;

    nd::ndarray<float, 2> Q(nd::shape<2>(2, 3));
    nd::ndarray<float, 2> K(nd::shape<2>(2, 3));
    // Q = [[1,0,0],[0,1,0]]
    Q.zero(); Q(0,0)=1; Q(1,1)=1;
    // K = [[1,0,0],[0,1,0]]
    K.zero(); K(0,0)=1; K(1,1)=1;

    auto S = nd::einsum::contract<QL, KL, OL>(Q, K);
    // S should be identity-like: [[1,0],[0,1]]
    assert(S.shape(0) == 2 && S.shape(1) == 2);
    assert(near(S(0,0), 1.0f));
    assert(near(S(0,1), 0.0f));
    assert(near(S(1,0), 0.0f));
    assert(near(S(1,1), 1.0f));
    std::puts("  [PASS] test_higher_rank_contract");
}

/**
 * @brief 验证较大矩阵乘法的数值正确性
 */
void test_matmul_larger() {
    // 4x3 * 3x5 -> 4x5
    nd::ndarray<float, 2> A(nd::shape<2>(4, 3));
    nd::ndarray<float, 2> B(nd::shape<2>(3, 5));

    // Fill A with row indices, B with column indices
    for (nd::index_t i = 0; i < 4; ++i)
        for (nd::index_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<float>(j + 1); // each row = [1,2,3]
    for (nd::index_t i = 0; i < 3; ++i)
        for (nd::index_t j = 0; j < 5; ++j)
            B(i, j) = static_cast<float>(i + 1); // col j = [1,2,3]^T
    // C(i,j) = sum_k A(i,k)*B(k,j) = 1*1+2*2+3*3 = 14 for all i,j
    auto C = nd::matmul(A, B);
    assert(C.shape(0) == 4 && C.shape(1) == 5);
    for (nd::index_t i = 0; i < 4; ++i)
        for (nd::index_t j = 0; j < 5; ++j)
            assert(near(C(i, j), 14.0f));
    std::puts("  [PASS] test_matmul_larger");
}

/**
 * @brief 执行 M5 einsum 测试集
 */
void run_test_m5() {
    std::puts("===== M5 Einsum Tests =====");
    test_dot();
    test_matmul();
    test_batched_matmul();
    test_outer();
    test_trace_2d();
    test_trace_3d();
    test_contract_generic();
    test_contract_reorder();
    test_reduce_labels();
    test_labels_compile_time();
    test_higher_rank_contract();
    test_matmul_larger();
    std::puts("===== ALL M5 TESTS PASSED =====");
}
