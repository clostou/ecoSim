/**
 * @file test_final.cpp
 * @brief 基于 agent_numpy.py 的最终代表性张量测试
 *
 * 覆盖三类典型计算：
 *   1. Linear.backward 中的矩阵反传与参数梯度
 *   2. 缩放点积注意力（SDPA）前向中的两个核心 contraction
 *   3. 注意力反向中的代表性权重梯度 einsum：kejn, sjn -> kes
 *
 * 说明：
 * 当前库已经支持同秩广播、keepdims reduce 与 einsum / matmul 组合，因此：
 *   - Linear 的 bias 加法可以直接写成 `matmul(w, x) + b`
 *   - SDPA 的稳定 softmax 可以直接写成
 *     `exp(logits - max_keepdims<3>(logits)) / sum_keepdims<3>(...)`
 * 这里保留 NumPy 小尺寸黄金值校验，用于验证最终组合路径的正确性。
 */

#include "ndarray/ndarray.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

static constexpr float EPS = 1e-5f;

/// @brief 用固定容差比较两个浮点数是否近似相等
static bool near(float a, float b, float eps = EPS) {
    return std::fabs(a - b) < eps;
}

/**
 * @brief 沿第 3 轴执行数值稳定的 softmax
 * @param logits 输入 logits 视图
 * @return softmax 概率张量
 */
static nd::ndarray<float, 5> softmax_axis3_stable(nd::ndview<const float, 5> logits) {
    auto row_max = nd::max_keepdims<3>(logits);
    auto shifted = nd::eval(logits - row_max.view());
    auto exp_shifted = nd::eval(nd::exp(shifted.view()));
    auto denom = nd::sum_keepdims<3>(exp_shifted.view());
    return nd::eval(exp_shifted.view() / denom.view());
}

/**
 * @brief 验证 Linear.backward 风格的前向与梯度核心张量公式
 */
static void test_linear_backward_like_agent_numpy() {
    // 对应 agent_numpy.Linear.backward 的核心张量公式：
    //   z = W x + b
    //   a = sigmoid(z)
    //   g1 = sigmoid'(a) * g
    //   g_x = W^T g1
    //   grad_w = g1 x^T
    //   grad_b = sum(g1, axis=1, keepdims=True)

    nd::ndarray<float, 2> w(nd::shape<2>(3, 2));
    nd::ndarray<float, 2> x(nd::shape<2>(2, 2));
    nd::ndarray<float, 2> b(nd::shape<2>(3, 1));
    nd::ndarray<float, 2> upstream(nd::shape<2>(3, 2));

    w(0, 0) = 0.2f;  w(0, 1) = -0.4f;
    w(1, 0) = 0.7f;  w(1, 1) =  0.1f;
    w(2, 0) = -0.3f; w(2, 1) =  0.5f;

    x(0, 0) =  1.0f; x(0, 1) = -1.0f;
    x(1, 0) =  2.0f; x(1, 1) =  0.5f;

    b(0, 0) =  0.1f;
    b(1, 0) = -0.2f;
    b(2, 0) =  0.3f;

    upstream(0, 0) =  0.5f; upstream(0, 1) = -0.1f;
    upstream(1, 0) = -0.3f; upstream(1, 1) =  0.2f;
    upstream(2, 0) =  0.4f; upstream(2, 1) =  0.6f;

    auto z = nd::eval(nd::matmul(w, x) + b);
    auto a = nd::eval(nd::sigmoid(z.view()));
    auto g1 = nd::eval(a.view() * (1.0f - a.view()) * upstream.view());
    auto gx = nd::matmul(nd::transpose(w), g1.view());
    auto gw = nd::matmul(g1.view(), nd::transpose(x));
    auto gb = nd::sum_keepdims<1>(g1.view());

    // NumPy 黄金值，来自与 agent_numpy 同公式的小尺寸例子。
    assert(near(a(0, 0), 0.37754068f));
    assert(near(a(0, 1), 0.42555750f));
    assert(near(a(1, 0), 0.66818780f));
    assert(near(a(1, 1), 0.29943287f));
    assert(near(a(2, 0), 0.73105860f));
    assert(near(a(2, 1), 0.70056720f));

    assert(near(g1(0, 0),  0.11750185f));
    assert(near(g1(0, 1), -0.02444583f));
    assert(near(g1(1, 0), -0.06651386f));
    assert(near(g1(1, 1),  0.04195457f));
    assert(near(g1(2, 0),  0.07864477f));
    assert(near(g1(2, 1),  0.12586369f));

    assert(near(gx(0, 0), -0.04665276f));
    assert(near(gx(0, 1), -0.01328008f));
    assert(near(gx(1, 0), -0.01432974f));
    assert(near(gx(1, 1),  0.07690563f));

    assert(near(gw(0, 0),  0.14194769f));
    assert(near(gw(0, 1),  0.22278080f));
    assert(near(gw(1, 0), -0.10846843f));
    assert(near(gw(1, 1), -0.11205044f));
    assert(near(gw(2, 0), -0.04721891f));
    assert(near(gw(2, 1),  0.22022140f));

    assert(gb.shape(0) == 3 && gb.shape(1) == 1);
    assert(near(gb(0, 0),  0.09305602f));
    assert(near(gb(1, 0), -0.02455929f));
    assert(near(gb(2, 0),  0.20450845f));

    std::puts("  [PASS] test_linear_backward_like_agent_numpy");
}

/**
 * @brief 验证缩放点积注意力前向中的两个代表性 contraction
 */
static void test_sdpa_forward_like_agent_numpy() {
    // 对应 agent_numpy.GQA.forward 中的两个关键 contraction：
    //   attn_raw = einsum('qkein, kejn -> qkijn', q, k) / sqrt(E)
    //   out      = einsum('qkijn, kejn -> qkein', score, v)
    // 这里用单 batch / 单 kv-head / 小尺寸验证最终结果。

    using QL = nd::einsum::label_seq<'q', 'k', 'e', 'i', 'n'>;
    using KL = nd::einsum::label_seq<'k', 'e', 'j', 'n'>;
    using SL = nd::einsum::label_seq<'q', 'k', 'i', 'j', 'n'>;
    using OL = nd::einsum::label_seq<'q', 'k', 'e', 'i', 'n'>;

    nd::ndarray<float, 5> q(nd::shape<5>(1, 1, 2, 2, 1));
    nd::ndarray<float, 4> k(nd::shape<4>(1, 2, 2, 1));
    nd::ndarray<float, 4> v(nd::shape<4>(1, 2, 2, 1));

    q.zero();
    q(0, 0, 0, 0, 0) = 1.0f;
    q(0, 0, 1, 1, 0) = 1.0f;

    k.zero();
    k(0, 0, 0, 0) = 1.0f;
    k(0, 1, 1, 0) = 1.0f;

    v(0, 0, 0, 0) = 2.0f;
    v(0, 0, 1, 0) = 1.0f;
    v(0, 1, 0, 0) = 0.5f;
    v(0, 1, 1, 0) = 3.0f;

    auto raw = nd::einsum::contract<QL, KL, SL>(q, k);
    auto logits = nd::eval(raw / std::sqrt(2.0f));
    auto prob = softmax_axis3_stable(logits.cview());
    auto out = nd::einsum::contract<SL, KL, OL>(prob, v);

    assert(near(logits(0, 0, 0, 0, 0), 0.70710677f));
    assert(near(logits(0, 0, 0, 1, 0), 0.0f));
    assert(near(logits(0, 0, 1, 0, 0), 0.0f));
    assert(near(logits(0, 0, 1, 1, 0), 0.70710677f));

    assert(near(prob(0, 0, 0, 0, 0), 0.66976154f));
    assert(near(prob(0, 0, 0, 1, 0), 0.33023846f));
    assert(near(prob(0, 0, 1, 0, 0), 0.33023846f));
    assert(near(prob(0, 0, 1, 1, 0), 0.66976154f));

    assert(out.shape(0) == 1 && out.shape(1) == 1);
    assert(out.shape(2) == 2 && out.shape(3) == 2 && out.shape(4) == 1);
    assert(near(out(0, 0, 0, 0, 0), 1.66976154f));
    assert(near(out(0, 0, 0, 1, 0), 1.33023846f));
    assert(near(out(0, 0, 1, 0, 0), 1.32559609f));
    assert(near(out(0, 0, 1, 1, 0), 2.17440367f));

    std::puts("  [PASS] test_sdpa_forward_like_agent_numpy");
}

/**
 * @brief 验证注意力反向传播中代表性的权重梯度 contraction
 */
static void test_attention_weight_grad_contract_like_agent_numpy() {
    // 对应 agent_numpy.GQA.backward 中的：
    //   grad_w_v += np.einsum('kejn, sjn -> kes', g_v, x_k)

    using GV = nd::einsum::label_seq<'k', 'e', 'j', 'n'>;
    using XK = nd::einsum::label_seq<'s', 'j', 'n'>;
    using OUT = nd::einsum::label_seq<'k', 'e', 's'>;

    nd::ndarray<float, 4> g_v(nd::shape<4>(2, 2, 2, 1));
    nd::ndarray<float, 3> x_k(nd::shape<3>(3, 2, 1));

    g_v(0, 0, 0, 0) = 1.0f;
    g_v(0, 0, 1, 0) = 2.0f;
    g_v(0, 1, 0, 0) = 3.0f;
    g_v(0, 1, 1, 0) = 4.0f;
    g_v(1, 0, 0, 0) = 0.5f;
    g_v(1, 0, 1, 0) = 1.5f;
    g_v(1, 1, 0, 0) = 2.5f;
    g_v(1, 1, 1, 0) = 3.5f;

    x_k(0, 0, 0) = 1.0f;
    x_k(0, 1, 0) = 2.0f;
    x_k(1, 0, 0) = 0.5f;
    x_k(1, 1, 0) = 1.0f;
    x_k(2, 0, 0) = 2.0f;
    x_k(2, 1, 0) = 1.0f;

    auto grad_w_v = nd::einsum::contract<GV, XK, OUT>(g_v, x_k);

    assert(grad_w_v.shape(0) == 2);
    assert(grad_w_v.shape(1) == 2);
    assert(grad_w_v.shape(2) == 3);

    assert(near(grad_w_v(0, 0, 0),  5.0f));
    assert(near(grad_w_v(0, 0, 1),  2.5f));
    assert(near(grad_w_v(0, 0, 2),  4.0f));
    assert(near(grad_w_v(0, 1, 0), 11.0f));
    assert(near(grad_w_v(0, 1, 1),  5.5f));
    assert(near(grad_w_v(0, 1, 2), 10.0f));

    assert(near(grad_w_v(1, 0, 0),  3.5f));
    assert(near(grad_w_v(1, 0, 1),  1.75f));
    assert(near(grad_w_v(1, 0, 2),  2.5f));
    assert(near(grad_w_v(1, 1, 0),  9.5f));
    assert(near(grad_w_v(1, 1, 1),  4.75f));
    assert(near(grad_w_v(1, 1, 2),  8.5f));

    std::puts("  [PASS] test_attention_weight_grad_contract_like_agent_numpy");
}

/**
 * @brief 执行基于 agent_numpy 公式抽取的最终张量测试
 */
void run_test_final() {
    std::puts("===== final ndarray business tests =====");
    test_linear_backward_like_agent_numpy();
    test_sdpa_forward_like_agent_numpy();
    test_attention_weight_grad_contract_like_agent_numpy();
    std::puts("===== ALL FINAL BUSINESS TESTS PASSED =====");
}