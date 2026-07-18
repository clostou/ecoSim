/**
 * @file net_kernel.cuh
 * @brief AttnLSTM Actor + FCN Critic 的单block内核函数
 *
 * 执行模型：单block=单网络，grid=num_networks
 * 多次前向→一次反向（多cache BPTT累加梯度）
 *
 * 关键设计：
 * - 权重列优先（与 fnn_kernel.cuh 的行优先不同）
 * - 两类并行：GEMM并行（warp分M行）+ 注意力并行（warp分Q头）
 * - 三门融合GEMM：x加载一次，复用三次
 * - BPTT：外层循环逆序遍历cache，d_h_prev在cache间传递
 * - 双缓冲smem：buf_a/buf_b交替，SMEM_PAD消除bank conflict
 *
 * 参考：
 *   doc/attn-lstm-warp-gemm-design.md（GEMM分解）
 *   test/test-cutlass/src/fnn_kernel.cuh（双缓冲smem模式）
 *   python/agent_numpy.py（反向数学参考）
 */

#pragma once

#include <cuda_runtime.h>

#include "net_config.cuh"
#include "warp_gemm.cuh"

namespace agent_gpu {

// ===================================================================
// 数据结构
// ===================================================================

/// @brief AttnLSTM Actor 和 FCN Critic 的权重 + 梯度累加器，列优先+alignas(16)
template <typename Config>
struct alignas(16) AttnLstmWeights {
    // ---- Actor 权重（列优先） ----
    // Wkv [h_kv, d_e, 2*D_obs]：末维 [Wk | Wv] 并列
    float Wkv[2 * Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_DIM];
    // Wq [h_q, d_e, d_h1]
    float Wq[Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM * Config::LSTM_QUERY_DIM];
    // bcat [h_q*d_e]：注意力输出偏置（post-reshape）
    float bcat[Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM];
    // Wico [d_h, 3*d_in]：三门权重（U_* 折叠进 h_prev 段）
    float Wico[3 * Config::LSTM_HIDDEN_DIM * Config::LSTM_INPUT_DIM];
    // bico [3*d_h]：[bi | bc | bo]
    float bico[3 * Config::LSTM_HIDDEN_DIM];
    // Wf [d_o, d_h2] + bf [d_o]
    float Wf[Config::LSTM_OUTPUT_DIM * Config::ACT_DIM];
    float bf[Config::ACT_DIM];

    // ---- Critic 权重（列优先） ----
    float Wc1[Config::CRITIC_HIDDEN_DIM * Config::OBS_DIM];
    float bc1[Config::CRITIC_HIDDEN_DIM];
    float Wc2[Config::CRITIC_HIDDEN_DIM * (Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM)];
    float bc2[Config::CRITIC_HIDDEN_DIM];
    float Wc3[Config::CRITIC_HIDDEN_DIM * Config::CRITIC_HIDDEN_DIM];
    float bc3[Config::CRITIC_HIDDEN_DIM];
    float Wc4[1 * Config::CRITIC_HIDDEN_DIM];
    float bc4[1];

    // ---- 梯度累加器（与权重相同形状，列优先） ----
    float grad_Wkv[2 * Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_DIM];
    float grad_Wq[Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM * Config::LSTM_QUERY_DIM];
    float grad_bcat[Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM];
    float grad_Wico[3 * Config::LSTM_HIDDEN_DIM * Config::LSTM_INPUT_DIM];
    float grad_bico[3 * Config::LSTM_HIDDEN_DIM];
    float grad_Wf[Config::LSTM_OUTPUT_DIM * Config::ACT_DIM];
    float grad_bf[Config::ACT_DIM];

    float grad_Wc1[Config::CRITIC_HIDDEN_DIM * Config::OBS_DIM];
    float grad_bc1[Config::CRITIC_HIDDEN_DIM];
    float grad_Wc2[Config::CRITIC_HIDDEN_DIM * (Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM)];
    float grad_bc2[Config::CRITIC_HIDDEN_DIM];
    float grad_Wc3[Config::CRITIC_HIDDEN_DIM * Config::CRITIC_HIDDEN_DIM];
    float grad_bc3[Config::CRITIC_HIDDEN_DIM];
    float grad_Wc4[1 * Config::CRITIC_HIDDEN_DIM];
    float grad_bc4[1];
};

/// @brief 激活值缓存（行优先），前向写/反向读
template <typename Config>
struct alignas(16) AttnLstmCache {
    // ---- Attention ----
    float x_kv[Config::OBS_DIM * Config::OBS_N];                         // [D_obs, M] 观测矩阵
    float q[Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM];              // [h_q, d_e] Q投影
    float k[Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_N];  // [h_kv, d_e, M] K投影
    float v[Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_N];  // [h_kv, d_e, M] V投影
    float p[Config::ATTN_Q_HEADS * Config::ATTN_KV_HEADS * Config::OBS_N];    // [h_q, h_kv, M] softmax概率

    // ---- LSTM ----
    float x[Config::LSTM_INPUT_DIM];              // [d_in] 拼接输入
    float h_prev[Config::LSTM_HIDDEN_DIM];        // [d_h] 前步隐状态
    float c_prev[Config::LSTM_HIDDEN_DIM];        // [d_h] 前步细胞状态
    float gate_i[Config::LSTM_HIDDEN_DIM];        // [d_h] 输入门（sigmoid）
    float gate_o[Config::LSTM_HIDDEN_DIM];        // [d_h] 输出门（sigmoid）
    float c_alt[Config::LSTM_HIDDEN_DIM];         // [d_h] 候选细胞状态（tanh）
    float tanhc[Config::LSTM_HIDDEN_DIM];         // [d_h] tanh(c_new)

    float lstm_o[Config::LSTM_OUTPUT_DIM];        // [d_h2] LSTM输出切片
    float act[Config::ACT_DIM];                   // [d_o] Actor输出

    // ---- Critic ----
    float c_h1[Config::CRITIC_HIDDEN_DIM];        // [d_c] max-pool后
    int   argmax_c_h1[Config::CRITIC_HIDDEN_DIM]; // [d_c] max-pool的argmax索引
    float c_h2[Config::CRITIC_HIDDEN_DIM];        // [d_c]
    float c_h3[Config::CRITIC_HIDDEN_DIM];        // [d_c]
};

/// @brief 每网络持久状态（存global memory，跨时间步）
template <typename Config>
struct alignas(16) AttnLstmPersistent {
    float h[Config::LSTM_HIDDEN_DIM];  /// 隐状态 [d_h]
    float c[Config::LSTM_HIDDEN_DIM];  /// 细胞状态 [d_h]
};

// ===================================================================
// 共享内存布局（双缓冲 + 注意力工作区）
// ===================================================================
template <typename Config>
struct SmemLayout {
    // 权重区（前向只加载权重，反向加载权重+梯度）
    static constexpr int WEIGHTS_FLOATS = sizeof(AttnLstmWeights<Config>) / sizeof(float);

    // 最大GEMM的输入/输出尺寸
    // 前向最大K*N: Wico K=168 N=1 → 168; Critic L1 K=8 N=16 → 128; K/V K=8 N=16 → 128
    static constexpr int MAX_IN  = (Config::LSTM_INPUT_DIM > Config::OBS_N * Config::OBS_DIM)
                                   ? Config::LSTM_INPUT_DIM : Config::OBS_N * Config::OBS_DIM;
    // 前向最大M*N: K/V M=16 N=16 → 256; Critic L1 M=16 N=16 → 256
    static constexpr int MAX_MN = Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_N;
    // 确保MAX_OUT至少能容纳三门输出（d_h*3=48）和critic各层输出（d_c=16）
    static constexpr int MAX_OUT = (MAX_MN > 3 * Config::LSTM_HIDDEN_DIM)
                                   ? MAX_MN : 3 * Config::LSTM_HIDDEN_DIM;

    static constexpr int BUF_A_SIZE = (MAX_IN > MAX_OUT) ? MAX_IN : MAX_OUT;
    static constexpr int BUF_B_SIZE = BUF_A_SIZE;

    // 注意力工作区：
    //   K[H_KV, D_E, OBS_N] + V[H_KV, D_E, OBS_N] + Q[H_Q, D_E] — 共享
    //   + per-warp: S[OBS_N] + P[OBS_N] + O[D_E]
    static constexpr int ATTN_SHARED =
        Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_N   // K
        + Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_N // V
        + Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM;                 // Q
    static constexpr int ATTN_PER_WARP =
        Config::OBS_N + Config::OBS_N + Config::ATTN_EMBED_DIM;  // S + P + O
    static constexpr int ATTN_WS_SIZE = ATTN_SHARED + Config::WARPS * ATTN_PER_WARP + Config::SMEM_PAD;

    static constexpr int PAD = Config::SMEM_PAD;

    // 持久状态区（h_prev, c_prev, inner，避免被GEMM缓冲覆盖）
    static constexpr int STATE_SIZE = 2 * Config::LSTM_HIDDEN_DIM + Config::INNER_DIM;

    static constexpr int WEIGHTS_OFF = 0;
    static constexpr int BUF_A_OFF   = WEIGHTS_OFF + WEIGHTS_FLOATS + PAD;
    static constexpr int BUF_B_OFF   = BUF_A_OFF + BUF_A_SIZE + PAD;
    static constexpr int STATE_OFF   = BUF_B_OFF + BUF_B_SIZE + PAD;
    static constexpr int ATTN_OFF    = STATE_OFF + STATE_SIZE + PAD;
    static constexpr int TOTAL       = ATTN_OFF + ATTN_WS_SIZE;
};

// ===================================================================
// 工具：协作加载权重到smem
// ===================================================================
template <typename Config>
__device__ void load_weights_to_smem(
    const AttnLstmWeights<Config>& W_global,
    float* __restrict__ W_smem)
{
    const float* src = reinterpret_cast<const float*>(&W_global);
    int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        W_smem[i] = src[i];
    }
}

/// 协作加载梯度区到smem（只加载grad_*字段）
template <typename Config>
__device__ void load_grads_to_smem(
    const AttnLstmWeights<Config>& W_global,
    float* __restrict__ G_smem,
    int grad_offset)
{
    const float* src = reinterpret_cast<const float*>(&W_global);
    int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
    // grad_offset 是第一个梯度字段在struct中的float偏移
    for (int i = threadIdx.x; i < total - grad_offset; i += blockDim.x) {
        G_smem[i] = src[grad_offset + i];
    }
}

/// 协作写回梯度区到global
template <typename Config>
__device__ void store_grads_to_global(
    AttnLstmWeights<Config>& W_global,
    const float* __restrict__ G_smem,
    int grad_offset)
{
    float* dst = reinterpret_cast<float*>(&W_global);
    int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
    for (int i = threadIdx.x; i < total - grad_offset; i += blockDim.x) {
        dst[grad_offset + i] = G_smem[i];
    }
}

// ===================================================================
// 前向KERNEL
// ===================================================================
template <typename Config>
__global__ void attn_lstm_forward_kernel(
    const AttnLstmWeights<Config>* __restrict__ d_weights,        // [num_networks]
    const AttnLstmPersistent<Config>* __restrict__ d_persistent,  // [num_networks]  持久状态（输入）
    AttnLstmPersistent<Config>* __restrict__ d_persistent_out,    // [num_networks]  持久状态（输出）
    const float* __restrict__ d_observe,      // [num_networks][OBS_DIM * OBS_N]  观测矩阵
    const float* __restrict__ d_inner,        // [num_networks][INNER_DIM]         内部状态
    float* __restrict__ d_act,                // [num_networks][ACT_DIM]           Actor输出
    float* __restrict__ d_value,              // [num_networks][1]                 Critic输出
    AttnLstmCache<Config>* __restrict__ d_cache)  // [num_networks]  激活缓存
{
    using Smem = SmemLayout<Config>;

    constexpr int OBS_N    = Config::OBS_N;
    constexpr int OBS_DIM  = Config::OBS_DIM;
    constexpr int INNER_DIM = Config::INNER_DIM;
    constexpr int ACT_DIM  = Config::ACT_DIM;
    constexpr int D_E      = Config::ATTN_EMBED_DIM;
    constexpr int H_KV     = Config::ATTN_KV_HEADS;
    constexpr int H_Q      = Config::ATTN_Q_HEADS;
    constexpr int Q_N      = Config::ATTN_QUERY_N;   // = H_Q / H_KV = 2
    constexpr int D_H      = Config::LSTM_HIDDEN_DIM;
    constexpr int D_H1     = Config::LSTM_QUERY_DIM;
    constexpr int D_H2     = Config::LSTM_OUTPUT_DIM;
    constexpr int D_IN     = Config::LSTM_INPUT_DIM;
    constexpr int D_C      = Config::CRITIC_HIDDEN_DIM;
    constexpr int WARPS    = Config::WARPS;
    constexpr int PAD      = Config::SMEM_PAD;

    // Smem中权重各字段的偏移量
    constexpr int WKV_OFF  = 0;
    constexpr int WQ_OFF   = 2 * H_KV * D_E * OBS_DIM;
    constexpr int BCAT_OFF = WQ_OFF + H_Q * D_E * D_H1;
    constexpr int WICO_OFF = BCAT_OFF + H_Q * D_E;
    constexpr int BICO_OFF = WICO_OFF + 3 * D_H * D_IN;
    constexpr int WF_OFF   = BICO_OFF + 3 * D_H;
    constexpr int BF_OFF   = WF_OFF + D_H2 * ACT_DIM;
    constexpr int WC1_OFF  = BF_OFF + ACT_DIM;
    constexpr int BC1_OFF  = WC1_OFF + D_C * OBS_DIM;
    constexpr int WC2_OFF  = BC1_OFF + D_C;
    constexpr int BC2_OFF  = WC2_OFF + D_C * (D_C + INNER_DIM);
    constexpr int WC3_OFF  = BC2_OFF + D_C;
    constexpr int BC3_OFF  = WC3_OFF + D_C * D_C;
    constexpr int WC4_OFF  = BC3_OFF + D_C;
    constexpr int BC4_OFF  = WC4_OFF + 1 * D_C;

    int net_id = blockIdx.x;
    const AttnLstmWeights<Config>& W_g = d_weights[net_id];
    const AttnLstmPersistent<Config>& P_in = d_persistent[net_id];

    // 输入数据指针
    const float* observe_g = d_observe + net_id * (OBS_DIM * OBS_N);
    const float* inner_g   = d_inner   + net_id * INNER_DIM;

    AttnLstmCache<Config>& cache = d_cache[net_id];

    extern __shared__ float smem_raw[];
    float* W_smem  = smem_raw + Smem::WEIGHTS_OFF;
    float* buf_a   = smem_raw + Smem::BUF_A_OFF;
    float* buf_b   = smem_raw + Smem::BUF_B_OFF;
    float* state_s = smem_raw + Smem::STATE_OFF;  // [h_prev(D_H) | c_prev(D_H) | inner(INNER_DIM)]
    float* attn_ws = smem_raw + Smem::ATTN_OFF;

    float* h_prev_s = state_s;             // [D_H]
    float* c_prev_s = state_s + D_H;       // [D_H]
    float* inner_s  = state_s + 2 * D_H;   // [INNER_DIM]

    // ---- 1. 协作加载所有权重到smem ----
    load_weights_to_smem<Config>(W_g, W_smem);
    __syncthreads();

    // ---- 2. 加载持久状态 h_prev, c_prev → state_s ----
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        h_prev_s[i] = P_in.h[i];
        c_prev_s[i] = P_in.c[i];
    }
    // 加载 inner → state_s
    for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
        inner_s[i] = inner_g[i];
    }
    __syncthreads();

    // ---- 3. 加载 observe → buf_a, 缓存 x_kv ----
    for (int i = threadIdx.x; i < OBS_DIM * OBS_N; i += blockDim.x) {
        buf_a[i] = observe_g[i];
        cache.x_kv[i] = observe_g[i];
    }
    __syncthreads();

    // ---- 4. K/V 投影 ----
    // Wkv 逻辑形状 [H_KV, D_E, 2*OBS_DIM]，末维 [Wk | Wv]
    // K投影：M=H_KV*D_E=16, K=OBS_DIM=8, N=OBS_N=16
    {
        warp_gemm_forward_col<Config>(W_smem + WKV_OFF, buf_a, buf_b,
                                       H_KV * D_E, OBS_DIM, OBS_N);
    }
    __syncthreads();
    // K存入 attn_ws + cache
    {
        float* k_smem = attn_ws;
        for (int i = threadIdx.x; i < H_KV * D_E * OBS_N; i += blockDim.x) {
            k_smem[i] = buf_b[i];
            cache.k[i] = buf_b[i];
        }
    }
    __syncthreads();

    // V投影: Wv在 WKV_OFF + H_KV*D_E*OBS_DIM
    {
        warp_gemm_forward_col<Config>(W_smem + WKV_OFF + H_KV * D_E * OBS_DIM,
                                       buf_a, buf_b,
                                       H_KV * D_E, OBS_DIM, OBS_N);
    }
    __syncthreads();
    {
        float* v_smem = attn_ws + (H_KV * D_E * OBS_N);  // V区
        for (int i = threadIdx.x; i < H_KV * D_E * OBS_N; i += blockDim.x) {
            v_smem[i] = buf_b[i];
            cache.v[i] = buf_b[i];
        }
    }
    __syncthreads();

    // ---- 5. Q 投影: Wq [H_Q, D_E, D_H1] @ h_prev[0:D_H1] ----
    // M = H_Q*D_E = 32, K = D_H1 = 12, N = 1
    for (int i = threadIdx.x; i < D_H1; i += blockDim.x) {
        buf_a[i] = state_s[i];  // h_prev[0:D_H1] from state region
    }
    __syncthreads();
    warp_gemm_forward_col<Config>(W_smem + WQ_OFF, buf_a, buf_b,
                                   H_Q * D_E, D_H1, 1);
    __syncthreads();
    // Q存入smem + cache
    {
        float* q_smem = attn_ws + 2 * (H_KV * D_E * OBS_N);  // Q区（在K,V之后）
        for (int i = threadIdx.x; i < H_Q * D_E; i += blockDim.x) {
            q_smem[i] = buf_b[i];
            cache.q[i] = buf_b[i];
        }
    }
    __syncthreads();

    // ---- 6. 缩放点积注意力（SDPA）per warp ----
    // 每warp处理一对 (h_q, h_kv)
    // warp_id决定: q_head = warp_id, kv_head = warp_id / Q_N
    {
        int warp_id = threadIdx.x / 32;
        int warp_lane = threadIdx.x & 31;
        int q_head   = warp_id;                // 0,1,2,3
        int kv_head  = q_head / Q_N;           // 0,0,1,1

        // 定位本warp的smem数据
        float* k_smem = attn_ws;                                      // K区
        float* v_smem = attn_ws + (H_KV * D_E * OBS_N);              // V区
        float* q_smem = attn_ws + 2 * (H_KV * D_E * OBS_N);          // Q区
        // 本warp专属工作区
        float* warp_ws = attn_ws + 2 * (H_KV * D_E * OBS_N) + (H_Q * D_E)
                         + warp_id * Smem::ATTN_PER_WARP / WARPS;
        // 本warp专属工作区偏移: 共享区(K+V+Q)后 + warp_id * per_warp(S+P+O)
        int ws_base = (H_KV * D_E * OBS_N) * 2 + (H_Q * D_E) + warp_id * (OBS_N * 2 + D_E);
        float* S_smem = attn_ws + ws_base;           // [OBS_N]
        float* P_smem = S_smem + OBS_N;              // [OBS_N]
        float* O_smem = P_smem + OBS_N;              // [D_E]

        // QK点积: S = Q[q_head] @ K[kv_head]^T / sqrt(D_E)
        const float* Q_head = q_smem + q_head * D_E;           // [D_E]
        const float* K_head = k_smem + kv_head * (D_E * OBS_N); // [D_E, OBS_N]
        attn_qk_gemm(Q_head, K_head, S_smem, D_E, OBS_N, warp_lane);

        // Softmax: P = softmax(S)
        // 先将S复制到P
        for (int j = warp_lane; j < OBS_N; j += 32) {
            P_smem[j] = S_smem[j];
        }
        softmax_fwd_warp(P_smem, OBS_N, warp_lane);

        // 缓存P到global（反向需要）
        for (int j = warp_lane; j < OBS_N; j += 32) {
            cache.p[(q_head * H_KV + kv_head) * OBS_N + j] = P_smem[j];
        }

        // PV点积: O = P @ V[kv_head]^T
        const float* V_head = v_smem + kv_head * (D_E * OBS_N);
        attn_pv_gemm(P_smem, V_head, O_smem, D_E, OBS_N, warp_lane);

        // 同步所有warp完成注意力计算
        __syncthreads();

        // 将注意力输出O写入buf_a中对应位置（reshape后加bcat + sigmoid）
        // _observe = sigmoid(concat(O_0, O_1, O_2, O_3) + bcat)
        for (int e = warp_lane; e < D_E; e += 32) {
            int global_e = q_head * D_E + e;
            if (global_e < H_Q * D_E) {
                float pre_sig = O_smem[e] + W_smem[BCAT_OFF + global_e];
                buf_b[global_e] = Sigmoid::fwd(pre_sig);
            }
        }
    }
    __syncthreads();
    // buf_b 现在持有 _observe [H_Q * D_E] = [32]

    // ---- 7. 拼接 LSTM 输入 x = [h_prev(16) | inner(8) | _observe(32)] ----
    // x 存入 buf_a [D_IN=56]
    {
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            buf_a[i] = state_s[i];                      // h_prev[0:16]
        }
        for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
            buf_a[D_H + i] = state_s[2*D_H + i];        // inner[16:24]
        }
        for (int i = threadIdx.x; i < H_Q * D_E; i += blockDim.x) {
            buf_a[D_H + INNER_DIM + i] = buf_b[i];      // _observe[24:56]
        }
    }
    __syncthreads();
    // 缓存 x
    for (int i = threadIdx.x; i < D_IN; i += blockDim.x) {
        cache.x[i] = buf_a[i];
    }
    __syncthreads();

    // ---- 8. 三门LSTM前向 ----
    {
        float* gate_i_s = buf_b;              // [D_H]
        float* c_alt_s  = buf_b + D_H;        // [D_H]
        float* gate_o_s = buf_b + 2 * D_H;    // [D_H]

        warp_gemm_3gate_forward<Config>(
            W_smem + WICO_OFF, buf_a,
            W_smem + BICO_OFF,
            gate_i_s, c_alt_s, gate_o_s,
            D_H, D_IN);
        __syncthreads();

        // 缓存三门激活
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            cache.gate_i[i] = gate_i_s[i];
            cache.c_alt[i]  = c_alt_s[i];
            cache.gate_o[i] = gate_o_s[i];
        }
        __syncthreads();

        // ---- 9. 细胞状态 + 隐状态更新 ----
        // c_new = (1-gate_i)*c_prev + gate_i*c_alt
        // tanhc = tanh(c_new)
        // h_new = gate_o * tanhc
        float* c_new_s = buf_a;       // 复用buf_a [D_H]
        float* h_new_s = buf_a + D_H; // [D_H]
        float* tanhc_s = buf_b + 3 * D_H; // [D_H]

        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            float gi = gate_i_s[i];
            float go = gate_o_s[i];
            float ca = c_alt_s[i];
            float cp = c_prev_s[i];

            float cn = (1.0f - gi) * cp + gi * ca;
            c_new_s[i] = cn;
            float th = tanhf(cn);
            tanhc_s[i] = th;
            h_new_s[i] = go * th;
        }
        __syncthreads();

        // 缓存h_prev, c_prev, tanhc
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            cache.h_prev[i] = state_s[i];        // h_prev from state region
            cache.c_prev[i] = state_s[D_H + i];  // c_prev from state region
            cache.tanhc[i]  = tanhc_s[i];
        }
        // 立即保存 c_new, h_new 到 state_s（避免被后续GEMM覆盖）
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            state_s[i] = c_new_s[i];              // save c_new
            state_s[D_H + i] = h_new_s[i];        // save h_new
        }
        __syncthreads();
    }

    // ---- 10. 输出层: act = sigmoid(Wf @ h_new[8:16] + bf) ----
    // h_new is now in state_s[D_H .. 2*D_H-1]
    // h_new[8:16] = state_s[D_H + D_H2 .. D_H + D_H - 1]
    {
        for (int i = threadIdx.x; i < D_H2; i += blockDim.x) {
            buf_b[i] = state_s[D_H + D_H2 + i];  // h_new[8:16]
            cache.lstm_o[i] = buf_b[i];
        }
    }
    __syncthreads();

    warp_gemm_forward_col<Config>(W_smem + WF_OFF, buf_b, buf_a,
                                   ACT_DIM, D_H2, 1);
    __syncthreads();

    // + bf + sigmoid
    {
        for (int i = threadIdx.x; i < ACT_DIM; i += blockDim.x) {
            float val = buf_a[i] + W_smem[BF_OFF + i];
            buf_a[i] = Sigmoid::fwd(val);
            cache.act[i] = buf_a[i];
        }
    }
    __syncthreads();

    // 写Actor输出到global
    {
        float* act_g = d_act + net_id * ACT_DIM;
        for (int i = threadIdx.x; i < ACT_DIM; i += blockDim.x) {
            act_g[i] = buf_a[i];
        }
    }

    // 写持久状态到global（从state_s读取保存的c_new, h_new）
    {
        AttnLstmPersistent<Config>& P_out = d_persistent_out[net_id];
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            P_out.h[i] = state_s[D_H + i];  // h_new saved in state_s
            P_out.c[i] = state_s[i];        // c_new saved in state_s
        }
    }

    // ---- 11. Critic 前向 ----
    // L1: c_h1[d_c, M] = sigmoid(Wc1[d_c, D_obs] @ observe[D_obs, M] + bc1[d_c])
    //     然后 max-pool over M → c_h1_pooled[d_c]
    // 重新加载 observe → buf_a
    for (int i = threadIdx.x; i < OBS_DIM * OBS_N; i += blockDim.x) {
        buf_a[i] = observe_g[i];
    }
    __syncthreads();

    warp_gemm_forward_col<Config>(W_smem + WC1_OFF, buf_a, buf_b,
                                   D_C, OBS_DIM, OBS_N);
    __syncthreads();

    // + bc1 + sigmoid
    {
        for (int i = threadIdx.x; i < D_C * OBS_N; i += blockDim.x) {
            int r = i / OBS_N;
            float val = buf_b[i] + W_smem[BC1_OFF + r];
            buf_b[i] = Sigmoid::fwd(val);
        }
    }
    __syncthreads();

    // Max-pool over M=OBS_N
    {
        int warp_id = threadIdx.x / 32;
        int warp_lane = threadIdx.x & 31;

        int rows_per_warp = (D_C + WARPS - 1) / WARPS;
        int r_start = warp_id * rows_per_warp;
        int r_end   = min(r_start + rows_per_warp, D_C);

        for (int r = r_start + warp_lane; r < r_end; r += 32) {
            float max_val = -INFINITY;
            int   max_idx = 0;
            for (int j = 0; j < OBS_N; ++j) {
                float val = buf_b[r * OBS_N + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            buf_a[r] = max_val;
            cache.argmax_c_h1[r] = max_idx;
        }
        __syncthreads();
        // 缓存c_h1
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            cache.c_h1[i] = buf_a[i];
        }
    }
    __syncthreads();
    // buf_a[0:D_C] = c_h1 pooled

    // L2: c_h2 = sigmoid(Wc2 @ [c_h1 | inner] + bc2)
    // 拼接 [c_h1(D_C) | inner(INNER_DIM)] → buf_b (作为输入)
    {
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            buf_b[i] = buf_a[i];  // c_h1
        }
        for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
            buf_b[D_C + i] = state_s[2*D_H + i];  // inner from state region
        }
    }
    __syncthreads();

    warp_gemm_forward_col<Config>(W_smem + WC2_OFF, buf_b, buf_a,
                                   D_C, D_C + INNER_DIM, 1);
    __syncthreads();

    {
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            float val = buf_a[i] + W_smem[BC2_OFF + i];
            buf_a[i] = Sigmoid::fwd(val);
            cache.c_h2[i] = buf_a[i];
        }
    }
    __syncthreads();

    // L3: c_h3 = sigmoid(Wc3 @ c_h2 + bc3)
    {
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            buf_b[i] = buf_a[i];  // c_h2 → input buf
        }
    }
    __syncthreads();

    warp_gemm_forward_col<Config>(W_smem + WC3_OFF, buf_b, buf_a,
                                   D_C, D_C, 1);
    __syncthreads();

    {
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            float val = buf_a[i] + W_smem[BC3_OFF + i];
            buf_a[i] = Sigmoid::fwd(val);
            cache.c_h3[i] = buf_a[i];
        }
    }
    __syncthreads();

    // L4: v = Wc4 @ c_h3 + bc4（线性输出，无激活）
    {
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            buf_b[i] = buf_a[i];
        }
    }
    __syncthreads();

    warp_gemm_forward_col<Config>(W_smem + WC4_OFF, buf_b, buf_a, 1, D_C, 1);
    __syncthreads();

    {
        float val = buf_a[0] + W_smem[BC4_OFF];
        d_value[net_id] = val;
    }
}

// ===================================================================
// 反向KERNEL
// ===================================================================
template <typename Config>
__global__ void attn_lstm_backward_kernel(
    AttnLstmWeights<Config>* __restrict__ d_weights,
    const float* __restrict__ d_grad_act,
    const float* __restrict__ d_grad_value,
    const AttnLstmCache<Config>* __restrict__ d_caches,
    int num_steps)
{
    using Smem = SmemLayout<Config>;

    constexpr int OBS_N    = Config::OBS_N;
    constexpr int OBS_DIM  = Config::OBS_DIM;
    constexpr int INNER_DIM = Config::INNER_DIM;
    constexpr int ACT_DIM  = Config::ACT_DIM;
    constexpr int D_E      = Config::ATTN_EMBED_DIM;
    constexpr int H_KV     = Config::ATTN_KV_HEADS;
    constexpr int H_Q      = Config::ATTN_Q_HEADS;
    constexpr int Q_N      = Config::ATTN_QUERY_N;
    constexpr int D_H      = Config::LSTM_HIDDEN_DIM;
    constexpr int D_H1     = Config::LSTM_QUERY_DIM;
    constexpr int D_H2     = Config::LSTM_OUTPUT_DIM;
    constexpr int D_IN     = Config::LSTM_INPUT_DIM;
    constexpr int D_C      = Config::CRITIC_HIDDEN_DIM;

    // ---- 权重场大小 ----
    constexpr int SZ_WKV  = 2 * H_KV * D_E * OBS_DIM;
    constexpr int SZ_WQ   = H_Q * D_E * D_H1;
    constexpr int SZ_BCAT = H_Q * D_E;
    constexpr int SZ_WICO = 3 * D_H * D_IN;
    constexpr int SZ_BICO = 3 * D_H;
    constexpr int SZ_WF   = D_H2 * ACT_DIM;
    constexpr int SZ_BF   = ACT_DIM;
    constexpr int SZ_WC1  = D_C * OBS_DIM;
    constexpr int SZ_BC1  = D_C;
    constexpr int SZ_WC2  = D_C * (D_C + INNER_DIM);
    constexpr int SZ_BC2  = D_C;
    constexpr int SZ_WC3  = D_C * D_C;
    constexpr int SZ_BC3  = D_C;
    constexpr int SZ_WC4  = 1 * D_C;
    constexpr int SZ_BC4  = 1;

    // ---- 权重偏移（梯度偏移相同） ----
    constexpr int WKV_OFF  = 0;
    constexpr int WQ_OFF   = WKV_OFF + SZ_WKV;
    constexpr int BCAT_OFF = WQ_OFF + SZ_WQ;
    constexpr int WICO_OFF = BCAT_OFF + SZ_BCAT;
    constexpr int BICO_OFF = WICO_OFF + SZ_WICO;
    constexpr int WF_OFF   = BICO_OFF + SZ_BICO;
    constexpr int BF_OFF   = WF_OFF + SZ_WF;
    constexpr int WC1_OFF  = BF_OFF + SZ_BF;
    constexpr int BC1_OFF  = WC1_OFF + SZ_WC1;
    constexpr int WC2_OFF  = BC1_OFF + SZ_BC1;
    constexpr int BC2_OFF  = WC2_OFF + SZ_WC2;
    constexpr int WC3_OFF  = BC2_OFF + SZ_BC2;
    constexpr int BC3_OFF  = WC3_OFF + SZ_WC3;
    constexpr int WC4_OFF  = BC3_OFF + SZ_BC3;
    constexpr int BC4_OFF  = WC4_OFF + SZ_WC4;
    constexpr int GRAD_OFFSET = BC4_OFF + SZ_BC4;

    int net_id = blockIdx.x;
    AttnLstmWeights<Config>& W_g = d_weights[net_id];

    extern __shared__ float smem_raw[];
    float* W_smem  = smem_raw + Smem::WEIGHTS_OFF;
    float* buf_a   = smem_raw + Smem::BUF_A_OFF;
    float* buf_b   = smem_raw + Smem::BUF_B_OFF;
    float* attn_ws = smem_raw + Smem::ATTN_OFF;

    // ---- 1. 加载权重到smem ----
    load_weights_to_smem<Config>(W_g, W_smem);
    __syncthreads();

    // 梯度区
    float* G_smem = W_smem + GRAD_OFFSET;
    int grad_total = sizeof(AttnLstmWeights<Config>) / sizeof(float) - GRAD_OFFSET;
    for (int i = threadIdx.x; i < grad_total; i += blockDim.x) {
        G_smem[i] = 0.0f;
    }
    __syncthreads();

    // ---- 2. Critic 反向 ----
    {
        float grad_v = d_grad_value[net_id];
        const AttnLstmCache<Config>& cache = d_caches[net_id * num_steps + (num_steps - 1)];

        // L4: d_c_h3 = Wc4^T @ grad_v
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            buf_a[i] = W_smem[WC4_OFF + i] * grad_v;
        }
        __syncthreads();
        // grad_Wc4 += grad_v * c_h3
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            G_smem[WC4_OFF + i] += grad_v * cache.c_h3[i];
        }
        if (threadIdx.x == 0) { G_smem[BC4_OFF] += grad_v; }
        __syncthreads();

        // L3: sigmoid bwd + grad_Wc3 + d_c_h2
        activation_apply_bwd<Sigmoid>(cache.c_h3, buf_a, D_C);
        __syncthreads();
        {
            for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
                buf_b[i] = cache.c_h2[i];
            }
            __syncthreads();
            warp_gemm_backward_weight_col<Config>(buf_a, buf_b, G_smem + WC3_OFF, D_C, D_C, 1);
            __syncthreads();
            accumulate_bias_grad(buf_a, G_smem + BC3_OFF, D_C, 1);
            warp_gemm_backward_input_col<Config>(W_smem + WC3_OFF, buf_a, buf_b, D_C, D_C, 1);
            __syncthreads();
        }

        // L2: sigmoid bwd + grad_Wc2 + d_c_h2_pooled
        activation_apply_bwd<Sigmoid>(cache.c_h2, buf_b, D_C);
        __syncthreads();
        {
            for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
                buf_a[i] = cache.c_h1[i];
            }
            for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
                buf_a[D_C + i] = cache.x[D_H + i];
            }
            __syncthreads();
            warp_gemm_backward_weight_col<Config>(buf_b, buf_a, G_smem + WC2_OFF, D_C, D_C + INNER_DIM, 1);
            __syncthreads();
            accumulate_bias_grad(buf_b, G_smem + BC2_OFF, D_C, 1);
            warp_gemm_backward_input_col<Config>(W_smem + WC2_OFF, buf_b, buf_a, D_C, D_C + INNER_DIM, 1);
            __syncthreads();
        }

        // L1: max-pool bwd + sigmoid bwd + grad_Wc1
        {
            // Zero d_c1_raw [D_C, OBS_N] in buf_b
            for (int i = threadIdx.x; i < D_C * OBS_N; i += blockDim.x) {
                buf_b[i] = 0.0f;
            }
            __syncthreads();
            // Argmax routing
            for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
                int argmax_j = cache.argmax_c_h1[i];
                buf_b[i * OBS_N + argmax_j] = buf_a[i];
            }
            __syncthreads();
            // Sigmoid bwd
            for (int i = threadIdx.x; i < D_C * OBS_N; i += blockDim.x) {
                int r = i / OBS_N;
                buf_b[i] = Sigmoid::bwd(cache.c_h1[r], buf_b[i]);
            }
            __syncthreads();
            // grad_Wc1
            for (int i = threadIdx.x; i < OBS_DIM * OBS_N; i += blockDim.x) {
                buf_a[i] = cache.x_kv[i];
            }
            __syncthreads();
            warp_gemm_backward_weight_col<Config>(buf_b, buf_a, G_smem + WC1_OFF, D_C, OBS_DIM, OBS_N);
            __syncthreads();
            accumulate_bias_grad(buf_b, G_smem + BC1_OFF, D_C, OBS_N);
            // d_observe_critic = Wc1^T @ d_c1_raw → buf_a
            warp_gemm_backward_input_col<Config>(W_smem + WC1_OFF, buf_b, buf_a, D_C, OBS_DIM, OBS_N);
            __syncthreads();
        }
    }
    // buf_a[0:OBS_N*OBS_DIM] = d_observe_critic

    // ---- 3. Actor 反向 BPTT ----
    // buf_b layout:
    //   [0..15]   d_h_prev_acc  (persistent)
    //   [16..31]  d_h_new temp  (R1-R4, reused per iteration)
    //   [32..47]  d_c_prev_acc  (persistent, SEPARATE from d_h_new!)
    //   [48..103] x_s / d_x_s   (56 floats: R5 loads x, R6 overwrites to d_x)
    //   [104..111] lstm_o temp  (8 floats, R1-R3)
    float* d_h_prev_acc = buf_b;               // [0..15]
    float* d_c_prev_acc = buf_b + 2 * D_H;     // [32..47] — NOT overlapping d_h_new (=buf_b[16..31])
    float* d_x_s        = buf_b + 3 * D_H;     // [48..103] — d_x during R6-R15
    // x_s uses same buffer as d_x_s (buf_b[48..103]) — loaded in R5, overwritten in R6
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        d_h_prev_acc[i] = 0.0f;
        d_c_prev_acc[i] = 0.0f;
    }
    __syncthreads();

    const float* g_act_g = d_grad_act + net_id * ACT_DIM;

    // 仅最后一步有g_act（策略梯度），中间步仅靠BPTT传播
    for (int step = num_steps - 1; step >= 0; --step) {
        const AttnLstmCache<Config>& cache = d_caches[net_id * num_steps + step];

        // --- R1-R3: 输出层反向 ---
        // g_act only for last step; intermediate steps get zero (gradient from BPTT only)
        float g_act_step[ACT_DIM];
        for (int i = threadIdx.x; i < ACT_DIM; i += blockDim.x) {
            g_act_step[i] = (step == num_steps - 1) ? g_act_g[i] : 0.0f;
            buf_a[i] = Sigmoid::bwd(cache.act[i], g_act_step[i]);
        }
        __syncthreads();

        // grad_Wf += d_lstm_o @ lstm_o^T
        // lstm_o temp at buf_b[104..111] (safe: after d_c_prev_acc [32..47], after d_x_s/x_s [48..103])
        constexpr int LSTMO_TMP = 2 * D_H + D_H + D_IN;  // 104
        {
            for (int i = threadIdx.x; i < D_H2; i += blockDim.x) {
                buf_b[LSTMO_TMP + i] = cache.lstm_o[i];
            }
            __syncthreads();
            warp_gemm_backward_weight_col<Config>(buf_a, buf_b + LSTMO_TMP, G_smem + WF_OFF, ACT_DIM, D_H2, 1);
            __syncthreads();
            accumulate_bias_grad(buf_a, G_smem + BF_OFF, ACT_DIM, 1);
        }

        // d_z_o = Wf^T @ d_lstm_o → scatter into d_h_new[8:16]
        float* d_h_new_s = buf_b + D_H;
        {
            float* d_z_o_s = buf_a + ACT_DIM;  // [D_H2]
            warp_gemm_backward_input_col<Config>(W_smem + WF_OFF, buf_a, d_z_o_s, ACT_DIM, D_H2, 1);
            __syncthreads();
            for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
                d_h_new_s[i] = d_h_prev_acc[i];
            }
            for (int i = threadIdx.x; i < D_H2; i += blockDim.x) {
                d_h_new_s[D_H2 + i] += d_z_o_s[i];
            }
            __syncthreads();
        }

        // --- R4: 耦合门导数 ---
        float* d_gate_o_s = buf_a;            // [D_H]
        float* d_c_new_s  = buf_a + D_H;      // [D_H]
        float* d_c_alt_s  = buf_a + 2 * D_H;  // [D_H]
        float* d_gate_i_s = buf_a + 3 * D_H;  // [D_H]

        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            float dhn = d_h_new_s[i];
            float go  = cache.gate_o[i];
            float gi  = cache.gate_i[i];
            float th  = cache.tanhc[i];
            float ca  = cache.c_alt[i];
            float cp  = cache.c_prev[i];

            d_gate_o_s[i] = dhn * th * Sigmoid::bwd(go, 1.0f);
            d_c_new_s[i]  = dhn * go * Tanh::bwd(th, 1.0f) + d_c_prev_acc[i];
            d_c_alt_s[i]  = d_c_new_s[i] * gi * Tanh::bwd(ca, 1.0f);
            d_gate_i_s[i] = d_c_new_s[i] * (ca - cp) * Sigmoid::bwd(gi, 1.0f);
            d_c_prev_acc[i] = d_c_new_s[i] * (1.0f - gi);
        }
        __syncthreads();

        // --- R5: 三门权重梯度 ---
        float* x_s = buf_b + 3 * D_H;  // [D_IN]
        {
            for (int i = threadIdx.x; i < D_IN; i += blockDim.x) {
                x_s[i] = cache.x[i];
            }
            __syncthreads();
            warp_gemm_3gate_backward_weight<Config>(
                d_gate_i_s, d_c_alt_s, d_gate_o_s, x_s,
                G_smem + WICO_OFF, D_H, D_IN);
            __syncthreads();
            for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
                G_smem[BICO_OFF + i]            += d_gate_i_s[i];
                G_smem[BICO_OFF + D_H + i]      += d_c_alt_s[i];
                G_smem[BICO_OFF + 2 * D_H + i]  += d_gate_o_s[i];
            }
            __syncthreads();
        }

        // --- R6: 三门输入梯度 d_x ---
        warp_gemm_3gate_backward_input<Config>(
            W_smem + WICO_OFF, d_gate_i_s, d_c_alt_s, d_gate_o_s,
            d_x_s, D_H, D_IN);
        __syncthreads();

        // --- R7: 拆分 d_x ---
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            d_h_prev_acc[i] += d_x_s[i];
        }

        // --- R7b: sigmoid反向（_observe = sigmoid(O_flat + bcat)） ---
        // d_pre_sigmoid = d_observe_flat ⊙ sigmoid'(_observe)
        // _observe = x[D_H+INNER_DIM:] from forward cache
        // Write result back to d_x_s[D_H+INNER_DIM:] so attention backward uses corrected dO
        for (int i = threadIdx.x; i < H_Q * D_E; i += blockDim.x) {
            float post_sig = cache.x[D_H + INNER_DIM + i];  // _observe (post-sigmoid)
            float d_post   = d_x_s[D_H + INNER_DIM + i];     // d_observe_flat from LSTM
            d_x_s[D_H + INNER_DIM + i] = Sigmoid::bwd(post_sig, d_post); // d_pre_sigmoid
        }
        __syncthreads();

        // --- R8: 累加 bcat 梯度（输入是 d_pre_sigmoid） ---
        for (int i = threadIdx.x; i < H_Q * D_E; i += blockDim.x) {
            G_smem[BCAT_OFF + i] += d_x_s[D_H + INNER_DIM + i];
        }
        __syncthreads();

        // --- R9-R15: 注意力反向（完整链：dV/dP/dS/dK/dQ + Q→h_prev梯度） ---
        {
            // 常量
            constexpr int WKV_STRIDE = H_KV * D_E;  // 16
            constexpr int WQ_STRIDE  = H_Q * D_E;   // 32
            int K_size = D_E * OBS_N;               // 128

            for (int iter = 0; iter < H_Q; iter++) {
                int q_head = iter;
                int kv_head = q_head / Q_N;
                int wkv_row_off = kv_head * D_E;    // row offset in Wk/Wv: kv_h*8
                int wq_row_off  = q_head * D_E;     // row offset in Wq:  q_h*8

                // ---- 加载K到buf_a（后续需要用于dQ = dS @ K^T） ----
                {
                    int kv_off = kv_head * K_size;
                    for (int i = threadIdx.x; i < K_size; i += blockDim.x) {
                        buf_a[i] = cache.k[kv_off + i];          // K [D_E, OBS_N]
                    }
                    // 加载Q, dO到buf_a[K_size..]  (Q:8 + dO:8 = 16 floats)
                    for (int i = threadIdx.x; i < D_E; i += blockDim.x) {
                        buf_a[K_size + i]           = cache.q[q_head * D_E + i];  // Q
                        buf_a[K_size + D_E + i]     = d_x_s[D_H + INNER_DIM + q_head * D_E + i]; // dO
                    }
                }
                __syncthreads();

                float* K_buf   = buf_a;                  // [D_E, OBS_N]  row-major
                float* Q_buf   = buf_a + K_size;         // [D_E]
                float* dO_buf  = buf_a + K_size + D_E;   // [D_E]

                // ---- dP[j] = sum_e dO[e] * V[e, j]  (V from cache) ----
                // 结果存 buf_a 继续复用（K已不需要完整128 floats，但保留用于后续dQ）
                float* dP_buf = buf_a + K_size + 2 * D_E; // [OBS_N]
                for (int j = threadIdx.x; j < OBS_N; j += blockDim.x) {
                    int kv_off = kv_head * K_size;
                    float acc = 0.0f;
                    for (int e = 0; e < D_E; ++e) {
                        acc += dO_buf[e] * cache.v[kv_off + e * OBS_N + j];
                    }
                    dP_buf[j] = acc;
                }
                __syncthreads();

                // ---- Softmax bwd: dS = P * (dP - sum(dP*P)) / sqrt(D_E) ----
                float* dS_buf = dP_buf + OBS_N;  // [OBS_N] — after dP
                {
                    // sum(dP * P): single-thread reduction (OBS_N=16, fast) to avoid overwriting BPTT state
                    if (threadIdx.x == 0) {
                        float sum_local = 0.0f;
                        for (int j = 0; j < OBS_N; j++) {
                            float p_j = cache.p[(q_head * H_KV + kv_head) * OBS_N + j];
                            sum_local += dP_buf[j] * p_j;
                        }
                        dS_buf[0] = sum_local;  // temporary: store sum at dS_buf[0]
                    }
                    __syncthreads();
                    float sum_all = dS_buf[0];  // broadcast to all threads
                    __syncthreads();

                    float inv_sqrt_de = rsqrtf((float)D_E);
                    for (int j = threadIdx.x; j < OBS_N; j += blockDim.x) {
                        float p_j = cache.p[(q_head * H_KV + kv_head) * OBS_N + j];
                        dS_buf[j] = p_j * (dP_buf[j] - sum_all) * inv_sqrt_de;
                    }
                }
                __syncthreads();

                // ---- dK[e, j] = Q[e] * dS[j]  (outer product) ----
                float* dK_buf = buf_a;  // reuse K_buf space [D_E, OBS_N]
                for (int idx = threadIdx.x; idx < K_size; idx += blockDim.x) {
                    int e = idx / OBS_N;
                    int j = idx % OBS_N;
                    dK_buf[e * OBS_N + j] = Q_buf[e] * dS_buf[j];
                }
                __syncthreads();

                // ---- 累加 Wk, Wv 权重梯度（先消费dK_buf，再复用buf_a） ----
                for (int idx = threadIdx.x; idx < D_E * OBS_DIM; idx += blockDim.x) {
                    int e = idx % D_E;
                    int i = idx / D_E;
                    float dO_e = dO_buf[e];
                    float gk = 0.0f, gv = 0.0f;
                    for (int j = 0; j < OBS_N; j++) {
                        float p_j = cache.p[(q_head * H_KV + kv_head) * OBS_N + j];
                        gk += dK_buf[e * OBS_N + j] * cache.x_kv[i * OBS_N + j];
                        gv += dO_e * p_j * cache.x_kv[i * OBS_N + j];
                    }
                    G_smem[WKV_OFF + i * WKV_STRIDE + wkv_row_off + e] += gk;
                    G_smem[WKV_OFF + (i + OBS_DIM) * WKV_STRIDE + wkv_row_off + e] += gv;
                }
                __syncthreads();

                // ---- dQ[e] = sum_j dS[j] * K[e, j] ----
                // Reload K to buf_a (dK consumed, buf_a free)
                {
                    int kv_off = kv_head * K_size;
                    for (int i = threadIdx.x; i < K_size; i += blockDim.x) {
                        buf_a[i] = cache.k[kv_off + i];  // K reload to buf_a
                    }
                    __syncthreads();
                }
                float* K_reload = buf_a;  // [D_E, OBS_N] — safe: dK already consumed
                float* dQ_buf   = buf_a + K_size;  // [D_E] — after K_reload

                for (int e = threadIdx.x; e < D_E; e += blockDim.x) {
                    float acc = 0.0f;
                    for (int j = 0; j < OBS_N; ++j) {
                        acc += dS_buf[j] * K_reload[e * OBS_N + j];
                    }
                    dQ_buf[e] = acc;
                }
                __syncthreads();

                // ---- 累加 Wq 权重梯度 + h_prev 梯度 ----
                // grad_Wq[q_h, e, k] += dQ[e] * h_prev[k]
                // d_h_prev_Q[k] += sum_e Wq[q_h, e, k] * dQ[e]
                for (int idx = threadIdx.x; idx < D_E * D_H1; idx += blockDim.x) {
                    int e = idx % D_E;
                    int k = idx / D_E;
                    float dQ_e = dQ_buf[e];
                    float hp_k = cache.h_prev[k];  // h_prev[:D_H1]
                    G_smem[WQ_OFF + k * WQ_STRIDE + wq_row_off + e] += dQ_e * hp_k;
                }
                // d_h_prev_Q[k] = sum_e Wq[k*32 + q_h*8 + e] * dQ[e]
                for (int k = threadIdx.x; k < D_H1; k += blockDim.x) {
                    float acc = 0.0f;
                    for (int e = 0; e < D_E; ++e) {
                        acc += W_smem[WQ_OFF + k * WQ_STRIDE + wq_row_off + e] * dQ_buf[e];
                    }
                    d_h_prev_acc[k] += acc;
                }

                __syncthreads();  // ensure writes visible before next iter/step
            }
        }
    }

    // ---- 4. 写回梯度到global ----
    store_grads_to_global<Config>(W_g, G_smem, GRAD_OFFSET);
}

} // namespace agent_gpu
