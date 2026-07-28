/**
 * @file net_kernel.cuh
 * @brief AttnLSTM Actor + FCN Critic 的单block内核函数
 *
 * 执行模型：单block=单网络，grid=num_networks
 * 前向每步执行（写 cache 环），反向每步执行（Critic 每步、Actor 触发时遍历窗口）
 *
 * 全局内存四区（见 doc/attn-lstm-implementation.md）：
 *   Gmem1: [STATE, WEIGHT, GRAD] × num_networks  → AttnLstmNetRecord
 *   Gmem2: CACHE × num_networks × A2C_STEPS      → 环形缓存
 *   Gmem3: INPUT  × buf_length                    → CPU→GPU 通信
 *   Gmem4: OUTPUT × buf_length                    → GPU→CPU 通信
 *
 * 共享内存：前向 SmemLayoutFwd / 反向 SmemLayoutBwd（CACHE 常驻，无 ATTN_WS）。
 *
 * 状态：数据布局 + kernel 签名已就位；kernel 计算流程体为空桩，待下一阶段实现。
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

/// @brief AttnLSTM Actor 和 FCN Critic 的权重（列优先 + alignas(16)），仅权重
///        梯度用独立的同型 AttnLstmWeights 实例表示（见 AttnLstmNetRecord.grad）
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

/// @brief 每网络持久状态（Gmem1 记录的 STATE 字段，跨时间步）
template <typename Config>
struct alignas(16) AttnLstmPersistent {
    float h[Config::LSTM_HIDDEN_DIM];  /// 隐状态 [d_h]
    float c[Config::LSTM_HIDDEN_DIM];  /// 细胞状态 [d_h]
};

/// @brief 输入数据结构（Gmem3，CPU→GPU 通信缓冲）
///        前向读 observe/inner/step；反向读 grad_act/grad_value/step
template <typename Config>
struct alignas(16) AttnLstmInput {
    float observe[Config::OBS_DIM * Config::OBS_N];       /// 观测矩阵 [D_obs, M]
    float inner[Config::INNER_DIM];                       /// 内部状态 [D_inn]

    int step;                                             /// 内时间步（前向计数，反向时清零）
    float grad_act[Config::A2C_STEPS * Config::ACT_DIM];  /// Actor梯度 [a2c_steps, d_o]
    float grad_value[Config::A2C_STEPS * 1];              /// Critic梯度 [a2c_steps, 1]
};

/// @brief 输出数据结构（Gmem4，GPU→CPU 通信缓冲）
template <typename Config>
struct alignas(16) AttnLstmOutput {
    float act[Config::ACT_DIM];                           /// Actor输出 [D_o]
    float value[1];                                       /// Critic输出 [1]

    // 其他监控数据...
};

/// @brief Gmem1 单网络记录：[STATE, WEIGHT, GRAD] 连续打包
///        前向原位读写 state、只读 weight；反向只读 weight、读写 grad
template <typename Config>
struct alignas(16) AttnLstmNetRecord {
    AttnLstmPersistent<Config> state;
    AttnLstmWeights<Config>    weight;
    AttnLstmWeights<Config>    grad;
};

// ===================================================================
// 权重场布局（权重与梯度同号偏移）
// ===================================================================
template <typename Config>
struct WeightLayout {
    static constexpr int H_KV    = Config::ATTN_KV_HEADS;
    static constexpr int D_E     = Config::ATTN_EMBED_DIM;
    static constexpr int H_Q     = Config::ATTN_Q_HEADS;
    static constexpr int D_H     = Config::LSTM_HIDDEN_DIM;
    static constexpr int D_H1    = Config::LSTM_QUERY_DIM;
    static constexpr int D_H2    = Config::LSTM_OUTPUT_DIM;
    static constexpr int D_IN    = Config::LSTM_INPUT_DIM;
    static constexpr int D_C     = Config::CRITIC_HIDDEN_DIM;
    static constexpr int OBS_DIM = Config::OBS_DIM;
    static constexpr int OBS_N   = Config::OBS_N;
    static constexpr int INNER_DIM = Config::INNER_DIM;
    static constexpr int ACT_DIM = Config::ACT_DIM;

    static constexpr int SZ_WKV  = 2 * H_KV * D_E * OBS_DIM;
    static constexpr int SZ_WQ   = H_Q * D_E * D_H1;
    static constexpr int SZ_BCAT = H_Q * D_E;
    static constexpr int SZ_WICO = 3 * D_H * D_IN;
    static constexpr int SZ_BICO = 3 * D_H;
    static constexpr int SZ_WF   = D_H2 * ACT_DIM;
    static constexpr int SZ_BF   = ACT_DIM;
    static constexpr int SZ_WC1  = D_C * OBS_DIM;
    static constexpr int SZ_BC1  = D_C;
    static constexpr int SZ_WC2  = D_C * (D_C + INNER_DIM);
    static constexpr int SZ_BC2  = D_C;
    static constexpr int SZ_WC3  = D_C * D_C;
    static constexpr int SZ_BC3  = D_C;
    static constexpr int SZ_WC4  = 1 * D_C;
    static constexpr int SZ_BC4  = 1;

    static constexpr int WKV_OFF  = 0;
    static constexpr int WQ_OFF   = WKV_OFF + SZ_WKV;
    static constexpr int BCAT_OFF = WQ_OFF + SZ_WQ;
    static constexpr int WICO_OFF = BCAT_OFF + SZ_BCAT;
    static constexpr int BICO_OFF = WICO_OFF + SZ_WICO;
    static constexpr int WF_OFF   = BICO_OFF + SZ_BICO;
    static constexpr int BF_OFF   = WF_OFF + SZ_WF;
    static constexpr int WC1_OFF  = BF_OFF + SZ_BF;
    static constexpr int BC1_OFF  = WC1_OFF + SZ_WC1;
    static constexpr int WC2_OFF  = BC1_OFF + SZ_BC1;
    static constexpr int BC2_OFF  = WC2_OFF + SZ_WC2;
    static constexpr int WC3_OFF  = BC2_OFF + SZ_BC2;
    static constexpr int BC3_OFF  = WC3_OFF + SZ_WC3;
    static constexpr int WC4_OFF  = BC3_OFF + SZ_BC3;
    static constexpr int BC4_OFF  = WC4_OFF + SZ_WC4;
    static constexpr int TOTAL    = BC4_OFF + SZ_BC4;  // = sizeof(AttnLstmWeights)/sizeof(float) 去除尾部对齐填充
};

// ===================================================================
// 共享内存布局
//   前向 SmemLayoutFwd：BUF_A/B + STATE(32) + WEIGHTS + CACHE
//   反向 SmemLayoutBwd：BUF_A/B + WEIGHTS_A/B + CACHE×N（批量）
//   双缓冲 + PAD(32) 消除 bank conflict；无 ATTN_WS（CACHE 常驻）
// ===================================================================

/// BUF_A/B 尺寸：K/V 直写 cache 后，最大驻留为 Critic L1 raw [D_C*OBS_N]
template <typename Config>
struct SmemBufSizes {
    static constexpr int CRITIC_L1_RAW = Config::CRITIC_HIDDEN_DIM * Config::OBS_N;  // 256
    static constexpr int X_SIZE        = Config::LSTM_INPUT_DIM;                      // 56
    static constexpr int THREE_GATE    = 3 * Config::LSTM_HIDDEN_DIM;                 // 48
    static constexpr int OBSERVE       = Config::OBS_DIM * Config::OBS_N;             // 128
    static constexpr int BUF_SIZE      = CRITIC_L1_RAW > X_SIZE ? CRITIC_L1_RAW :
                                         (X_SIZE > THREE_GATE ? X_SIZE :
                                          (THREE_GATE > OBSERVE ? THREE_GATE : OBSERVE));  // 256
    static constexpr int BUF_A_SIZE    = BUF_SIZE;
    static constexpr int BUF_B_SIZE    = BUF_SIZE;
};

/// @brief 前向共享内存布局
template <typename Config>
struct SmemLayoutFwd {
    static constexpr int PAD            = Config::SMEM_PAD;
    static constexpr int WEIGHTS_FLOATS = sizeof(AttnLstmWeights<Config>) / sizeof(float);  // 4260
    static constexpr int STATE_FLOATS   = 2 * Config::LSTM_HIDDEN_DIM;                       // 32 (h_prev+c_prev)
    static constexpr int CACHE_FLOATS   = sizeof(AttnLstmCache<Config>) / sizeof(float);

    static constexpr int BUF_A_OFF   = 0;
    static constexpr int BUF_B_OFF   = BUF_A_OFF + SmemBufSizes<Config>::BUF_A_SIZE + PAD;
    static constexpr int STATE_OFF   = BUF_B_OFF + SmemBufSizes<Config>::BUF_B_SIZE + PAD;
    static constexpr int WEIGHTS_OFF = STATE_OFF + STATE_FLOATS + PAD;
    static constexpr int CACHE_OFF   = WEIGHTS_OFF + WEIGHTS_FLOATS + PAD;
    static constexpr int TOTAL       = CACHE_OFF + CACHE_FLOATS;  // 末段无尾 PAD

    static_assert(TOTAL * sizeof(float) <= 49152, "SmemLayoutFwd exceeds 48 KB");
};

/// @brief 反向共享内存布局（WEIGHTS_A=权重，WEIGHTS_B=梯度；CACHE×N 批量）
template <typename Config>
struct SmemLayoutBwd {
    static constexpr int PAD            = Config::SMEM_PAD;
    static constexpr int WEIGHTS_FLOATS = sizeof(AttnLstmWeights<Config>) / sizeof(float);  // 4260
    static constexpr int CACHE_FLOATS   = sizeof(AttnLstmCache<Config>) / sizeof(float);
    static constexpr int SMEM_LIMIT     = 49152 / sizeof(float);  // 12288 floats

    static constexpr int BUF_A_OFF     = 0;
    static constexpr int BUF_B_OFF     = BUF_A_OFF + SmemBufSizes<Config>::BUF_A_SIZE + PAD;
    static constexpr int WEIGHTS_A_OFF = BUF_B_OFF + SmemBufSizes<Config>::BUF_B_SIZE + PAD;
    static constexpr int WEIGHTS_B_OFF = WEIGHTS_A_OFF + WEIGHTS_FLOATS + PAD;
    static constexpr int CACHE_BASE    = WEIGHTS_B_OFF + WEIGHTS_FLOATS + PAD;  // 首块 cache 起点 = 9160
    static constexpr int PER_CACHE     = CACHE_FLOATS;   // cache 块间无 PAD（首块前已有尾部 PAD）

    // 在剩余 smem 预算内可批量载入的 cache 数（Actor 触发时遍历 A2C_STEPS 窗口分块载入）
    static constexpr int FIXED = CACHE_BASE;                       // 9160
    static constexpr int N_MAX = (SMEM_LIMIT - FIXED) / PER_CACHE; // =3（1028 floats/cache，无间 PAD）
    static constexpr int TOTAL = FIXED + PER_CACHE * N_MAX;        // 9160 + 1028*3 = 12244

    static_assert(N_MAX >= 1, "SmemLayoutBwd: no room for even 1 cache");
    static_assert(TOTAL * sizeof(float) <= 49152, "SmemLayoutBwd exceeds 48 KB");
};

// ===================================================================
// 工具：协作加载/写回权重与梯度
// ===================================================================

/// 协作加载权重到 smem（从 Gmem1 记录的 weight 字段）
template <typename Config>
__device__ __forceinline__ void load_weights_to_smem(
    const AttnLstmWeights<Config>& W_global,
    float* __restrict__ W_smem)
{
    const float* src = reinterpret_cast<const float*>(&W_global);
    int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        W_smem[i] = src[i];
    }
}

/// 协作清零 smem 梯度区（WEIGHTS_B，对应独立 grad buffer）
template <typename Config>
__device__ __forceinline__ void zero_grads_in_smem(float* __restrict__ G_smem) {
    int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        G_smem[i] = 0.0f;
    }
}

/// 协作写回 smem 梯度区到 Gmem1 记录的 grad 字段
template <typename Config>
__device__ __forceinline__ void store_grads_to_global(
    AttnLstmWeights<Config>& G_global,
    const float* __restrict__ G_smem)
{
    float* dst = reinterpret_cast<float*>(&G_global);
    int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        dst[i] = G_smem[i];
    }
}

// ===================================================================
// 前向 KERNEL
//
// 语义：每步执行；state 原位读写（读 h_prev/c_prev → 写 h_new/c_new）；
//       weight 只读；cache 环写槽 = d_input.step % A2C_STEPS。
// 读写详见 doc/attn-lstm-implementation.md "前向 kernel 执行流程"。
// ===================================================================
template <typename Config>
__global__ void attn_lstm_forward_kernel(
    AttnLstmNetRecord<Config>*      d_records,  // Gmem1: state(r/w) + weight(r)
    const AttnLstmInput<Config>*    d_input,    // Gmem3: observe + inner + step
    AttnLstmOutput<Config>*         d_output,   // Gmem4: act + value (w)
    AttnLstmCache<Config>*          d_caches)   // Gmem2: 环写 slot = step % A2C_STEPS
{
    using Smem = SmemLayoutFwd<Config>;
    using WL   = WeightLayout<Config>;

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
    constexpr int A2C      = Config::A2C_STEPS;

    int net_id = blockIdx.x;
    AttnLstmNetRecord<Config>& record = d_records[net_id];
    const AttnLstmInput<Config>& in = d_input[net_id];
    int slot = in.step % A2C;

    extern __shared__ float smem_raw[];
    float* W  = smem_raw + Smem::WEIGHTS_OFF;
    float* A  = smem_raw + Smem::BUF_A_OFF;
    float* B  = smem_raw + Smem::BUF_B_OFF;
    float* ST = smem_raw + Smem::STATE_OFF;   // [h_prev(D_H) | c_prev(D_H)]
    AttnLstmCache<Config>* CA = reinterpret_cast<AttnLstmCache<Config>*>(smem_raw + Smem::CACHE_OFF);

    float* h_prev_s = ST;            // [D_H]
    float* c_prev_s = ST + D_H;      // [D_H]

    // ---- 0. 协作载入 ----
    load_weights_to_smem<Config>(record.weight, W);
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        h_prev_s[i] = record.state.h[i];
        c_prev_s[i] = record.state.c[i];
    }
    for (int i = threadIdx.x; i < OBS_DIM * OBS_N; i += blockDim.x) {
        CA->x_kv[i] = in.observe[i];   // observe 直写 cache.x_kv（K/V 投影 + Critic L1 共用）
    }
    __syncthreads();

    // ---- 1. K 投影: Wk @ observe → cache.k ----
    warp_gemm_forward_col<Config>(W + WL::WKV_OFF, CA->x_kv, CA->k,
                                  H_KV * D_E, OBS_DIM, OBS_N);
    __syncthreads();

    // ---- 2. V 投影: Wv @ observe → cache.v ----
    warp_gemm_forward_col<Config>(W + WL::WKV_OFF + H_KV * D_E * OBS_DIM,
                                  CA->x_kv, CA->v,
                                  H_KV * D_E, OBS_DIM, OBS_N);
    __syncthreads();

    // ---- 3. Q 投影: Wq @ h_prev[0:D_H1] → cache.q ----
    // h_prev[0:D_H1] 即 ST[0:D_H1]（h_prev 区前 12 个）
    warp_gemm_forward_col<Config>(W + WL::WQ_OFF, ST, CA->q,
                                  H_Q * D_E, D_H1, 1);
    __syncthreads();

    // ---- 4. SDPA（4 warp 并行, q_head=warp_id, kv_head=warp_id/Q_N）----
    // S scratch = A[warp_id*OBS_N]；O scratch = B[warp_id*D_E]
    {
        int warp_id  = threadIdx.x / 32;
        int lane     = threadIdx.x & 31;
        int q_head   = warp_id;
        int kv_head  = q_head / Q_N;

        float* S_smem = A + warp_id * OBS_N;       // [OBS_N]
        float* O_smem = B + warp_id * D_E;         // [D_E]

        // QK: S = Q[q_head] @ K[kv_head]^T / sqrt(D_E)
        const float* Q_head = CA->q + q_head * D_E;
        const float* K_head = CA->k + kv_head * (D_E * OBS_N);
        attn_qk_gemm(Q_head, K_head, S_smem, D_E, OBS_N, lane);

        // Softmax: P = softmax(S)（原位 S→P）
        softmax_fwd_warp(S_smem, OBS_N, lane);

        // 缓存 P
        for (int j = lane; j < OBS_N; j += 32) {
            CA->p[(q_head * H_KV + kv_head) * OBS_N + j] = S_smem[j];
        }

        // PV: O = P @ V[kv_head]^T
        const float* V_head = CA->v + kv_head * (D_E * OBS_N);
        attn_pv_gemm(S_smem, V_head, O_smem, D_E, OBS_N, lane);
        __syncthreads();

        // +bcat + σ → _observe 直写 cache.x[24:56]
        for (int e = lane; e < D_E; e += 32) {
            int global_e = q_head * D_E + e;
            float pre_sig = O_smem[e] + W[WL::BCAT_OFF + global_e];
            CA->x[D_H + INNER_DIM + global_e] = Sigmoid::fwd(pre_sig);
        }
    }
    __syncthreads();

    // ---- 5. 拼接 x = [h_prev | inner | _observe] → cache.x ----
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        CA->x[i] = h_prev_s[i];                    // h_prev[0:16]
    }
    for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
        CA->x[D_H + i] = in.inner[i];              // inner[16:24]，载入 smem 供 LSTM + Critic L2 复用
    }
    // _observe[24:56] 已就位
    __syncthreads();

    // ---- 6. 三门 LSTM: Wico @ x → gate_i/c_alt/gate_o 直写 cache ----
    warp_gemm_3gate_forward<Config>(
        W + WL::WICO_OFF, CA->x, W + WL::BICO_OFF,
        CA->gate_i, CA->c_alt, CA->gate_o, D_H, D_IN);
    __syncthreads();

    // ---- 7. 细胞/隐状态更新（先缓存 hp/cp 供反向, 再覆写 STATE）----
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        CA->h_prev[i] = h_prev_s[i];    // 供反向
        CA->c_prev[i] = c_prev_s[i];
    }
    // c_new/h_new 临时落 STATE 区（避免被后续 GEMM 覆盖）
    float* c_new_s = ST;            // [D_H]
    float* h_new_s = ST + D_H;      // [D_H]
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        float gi = CA->gate_i[i];
        float go = CA->gate_o[i];
        float ca = CA->c_alt[i];
        float cp = c_prev_s[i];
        float cn = (1.0f - gi) * cp + gi * ca;
        c_new_s[i] = cn;
        float th = tanhf(cn);
        CA->tanhc[i] = th;
        h_new_s[i] = go * th;
    }
    __syncthreads();

    // ---- 8. 输出层: act = σ(Wf @ h_new[8:16] + bf) ----
    // h_new 在 ST[D_H:2*D_H]；h_new[8:16] = ST[D_H + D_H2 : 2*D_H]
    for (int i = threadIdx.x; i < D_H2; i += blockDim.x) {
        CA->lstm_o[i] = h_new_s[D_H2 + i];   // h_new[8:16]
    }
    __syncthreads();
    warp_gemm_forward_col<Config>(W + WL::WF_OFF, CA->lstm_o, A,
                                  ACT_DIM, D_H2, 1);
    __syncthreads();
    for (int i = threadIdx.x; i < ACT_DIM; i += blockDim.x) {
        float val = A[i] + W[WL::BF_OFF + i];
        A[i] = Sigmoid::fwd(val);
        CA->act[i] = A[i];
    }
    __syncthreads();

    // 立即写 act 到 Gmem4（后续 Critic L2 会覆写 A[0:D_C]，必须在覆写前写出）
    if (threadIdx.x == 0) {
        d_output[net_id].act[0] = A[0];
        d_output[net_id].act[1] = A[1];
    }

    // 写持久状态（原位）：h_new / c_new → record.state
    for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
        record.state.h[i] = h_new_s[i];
        record.state.c[i] = c_new_s[i];
    }

    // ---- 9. Critic L1: Wc1 @ observe → raw[256] 驻 BUF_B → maxpool ----
    warp_gemm_forward_col<Config>(W + WL::WC1_OFF, CA->x_kv, B,
                                  D_C, OBS_DIM, OBS_N);
    __syncthreads();
    for (int i = threadIdx.x; i < D_C * OBS_N; i += blockDim.x) {
        int r = i / OBS_N;
        B[i] = Sigmoid::fwd(B[i] + W[WL::BC1_OFF + r]);
    }
    __syncthreads();
    // Max-pool over OBS_N
    {
        int warp_id = threadIdx.x / 32;
        int lane    = threadIdx.x & 31;
        int rows_per_warp = (D_C + WARPS - 1) / WARPS;
        int r_start = warp_id * rows_per_warp;
        int r_end   = min(r_start + rows_per_warp, D_C);
        for (int r = r_start + lane; r < r_end; r += 32) {
            float max_val = -INFINITY;
            int   max_idx = 0;
            for (int j = 0; j < OBS_N; ++j) {
                float val = B[r * OBS_N + j];
                if (val > max_val) { max_val = val; max_idx = j; }
            }
            CA->c_h1[r] = max_val;
            CA->argmax_c_h1[r] = max_idx;
        }
    }
    __syncthreads();

    // ---- 10. L2: σ(Wc2 @ [c_h1 | inner] + bc2) → cache.c_h2 ----
    for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
        A[i] = CA->c_h1[i];
    }
    for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
        A[D_C + i] = CA->x[D_H + i];     // inner 复用自 cache.x
    }
    __syncthreads();
    warp_gemm_forward_col<Config>(W + WL::WC2_OFF, A, CA->c_h2,
                                  D_C, D_C + INNER_DIM, 1);
    __syncthreads();
    for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
        CA->c_h2[i] = Sigmoid::fwd(CA->c_h2[i] + W[WL::BC2_OFF + i]);
    }
    __syncthreads();

    // ---- 11. L3: σ(Wc3 @ c_h2 + bc3) → cache.c_h3 ----
    warp_gemm_forward_col<Config>(W + WL::WC3_OFF, CA->c_h2, CA->c_h3,
                                  D_C, D_C, 1);
    __syncthreads();
    for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
        CA->c_h3[i] = Sigmoid::fwd(CA->c_h3[i] + W[WL::BC3_OFF + i]);
    }
    __syncthreads();

    // ---- 12. L4: v = Wc4 @ c_h3 + bc4 → A[ACT_DIM] ----
    warp_gemm_forward_col<Config>(W + WL::WC4_OFF, CA->c_h3, A + ACT_DIM,
                                  1, D_C, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        A[ACT_DIM] += W[WL::BC4_OFF];   // value ∈ A[ACT_DIM]
    }

    // ---- 13. 写 value 到 Gmem4（act 已在 step 8 末尾写出）----
    if (threadIdx.x == 0) {
        d_output[net_id].value[0] = A[ACT_DIM];
    }

    // ---- 14. 批量协写 cache smem → Gmem2[net_id*A2C + slot] ----
    {
        AttnLstmCache<Config>* dst = d_caches + net_id * A2C + slot;
        int total = sizeof(AttnLstmCache<Config>) / sizeof(float);
        const float* src = reinterpret_cast<const float*>(CA);
        float*       gdst = reinterpret_cast<float*>(dst);
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            gdst[i] = src[i];
        }
    }
}

// ===================================================================
// 反向 KERNEL
//
// 语义：每步执行；weight 只读、grad 读写；Critic 每步更新（单步梯度）；
//       Actor 仅当 step > A2C_STEPS 且 grad_act[cache_i]≠0 时触发，
//       本 launch 内遍历整个 A2C_STEPS 窗口（从 cache_i=step%A2C_STEPS
//       新→旧环绕，smem 分块 N_MAX 载入），累加器仅存 BUF_B（无跨 launch）。
// 末尾流式 SGD: v=β·v+g; w=w·(1−lr·γ)+lr·v。
// 读写详见 doc/attn-lstm-implementation.md "反向 kernel 执行流程"。
// ===================================================================
template <typename Config>
__global__ void attn_lstm_backward_kernel(
    AttnLstmNetRecord<Config>*      d_records,  // Gmem1: weight(r) + grad(r/w, 优化器状态 v)
    const AttnLstmInput<Config>*    d_input,    // Gmem3: grad_act + grad_value + step
    const AttnLstmCache<Config>*    d_caches,   // Gmem2: 环读 (step%A2C_STEPS) 起点
    float lr, float beta, float gamma,
    int   bptt_steps)                           // BPTT 窗口步数（≤ A2C_STEPS，默认 = A2C_STEPS）
{
    using Smem = SmemLayoutBwd<Config>;
    using WL   = WeightLayout<Config>;

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
    constexpr int A2C      = Config::A2C_STEPS;
    constexpr int N_MAX    = Smem::N_MAX;
    constexpr int K_SIZE   = D_E * OBS_N;          // 128
    constexpr int WKV_STRIDE = H_KV * D_E;         // 16
    constexpr int WQ_STRIDE  = H_Q * D_E;          // 32

    int net_id = blockIdx.x;
    AttnLstmNetRecord<Config>& record = d_records[net_id];
    const AttnLstmInput<Config>& in = d_input[net_id];
    int step = in.step;
    int cache_i = ((step - 1) % A2C + A2C) % A2C;   // 最近一次前向的 cache 槽位

    extern __shared__ float smem_raw[];
    float* W = smem_raw + Smem::WEIGHTS_A_OFF;   // 权重（反向只读）
    float* G = smem_raw + Smem::WEIGHTS_B_OFF;   // 梯度累加（清零）
    float* A = smem_raw + Smem::BUF_A_OFF;
    float* B = smem_raw + Smem::BUF_B_OFF;
    AttnLstmCache<Config>* CA = reinterpret_cast<AttnLstmCache<Config>*>(smem_raw + Smem::CACHE_BASE);

    // ---- 阶段 0: 载入权重 + 清零梯度 ----
    load_weights_to_smem<Config>(record.weight, W);
    zero_grads_in_smem<Config>(G);
    __syncthreads();

    // ---- 阶段 1: Critic 反向（每步, 单步梯度）----
    // 读当前步 cache → CA[0]
    {
        const AttnLstmCache<Config>* src = d_caches + net_id * A2C + cache_i;
        const float* s = reinterpret_cast<const float*>(src);
        float*       d = reinterpret_cast<float*>(CA);
        int total = sizeof(AttnLstmCache<Config>) / sizeof(float);
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            d[i] = s[i];
        }
    }
    __syncthreads();
    const AttnLstmCache<Config>& c0 = CA[0];

    {
        float grad_v = in.grad_value[cache_i];

        // L4: d_c_h3 = Wc4^T @ grad_v
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            A[i] = W[WL::WC4_OFF + i] * grad_v;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            G[WL::WC4_OFF + i] += grad_v * c0.c_h3[i];
        }
        if (threadIdx.x == 0) { G[WL::BC4_OFF] += grad_v; }
        __syncthreads();

        // L3: σ'bwd + grad_Wc3 + d_c_h2
        activation_apply_bwd<Sigmoid>(c0.c_h3, A, D_C);
        __syncthreads();
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            B[i] = c0.c_h2[i];
        }
        __syncthreads();
        warp_gemm_backward_weight_col<Config>(A, B, G + WL::WC3_OFF, D_C, D_C, 1);
        __syncthreads();
        accumulate_bias_grad(A, G + WL::BC3_OFF, D_C, 1);
        warp_gemm_backward_input_col<Config>(W + WL::WC3_OFF, A, B, D_C, D_C, 1);
        __syncthreads();

        // L2: σ'bwd + grad_Wc2 + d_c_h1_pooled
        activation_apply_bwd<Sigmoid>(c0.c_h2, B, D_C);
        __syncthreads();
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            A[i] = c0.c_h1[i];
        }
        for (int i = threadIdx.x; i < INNER_DIM; i += blockDim.x) {
            A[D_C + i] = c0.x[D_H + i];   // inner = cache.x[D_H:D_H+INNER]
        }
        __syncthreads();
        warp_gemm_backward_weight_col<Config>(B, A, G + WL::WC2_OFF, D_C, D_C + INNER_DIM, 1);
        __syncthreads();
        accumulate_bias_grad(B, G + WL::BC2_OFF, D_C, 1);
        warp_gemm_backward_input_col<Config>(W + WL::WC2_OFF, B, A, D_C, D_C + INNER_DIM, 1);
        __syncthreads();

        // L1: max-pool bwd + σ'bwd + grad_Wc1
        for (int i = threadIdx.x; i < D_C * OBS_N; i += blockDim.x) {
            B[i] = 0.0f;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < D_C; i += blockDim.x) {
            int argmax_j = c0.argmax_c_h1[i];
            B[i * OBS_N + argmax_j] = A[i];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < D_C * OBS_N; i += blockDim.x) {
            int r = i / OBS_N;
            B[i] = Sigmoid::bwd(c0.c_h1[r], B[i]);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < OBS_DIM * OBS_N; i += blockDim.x) {
            A[i] = c0.x_kv[i];
        }
        __syncthreads();
        warp_gemm_backward_weight_col<Config>(B, A, G + WL::WC1_OFF, D_C, OBS_DIM, OBS_N);
        __syncthreads();
        accumulate_bias_grad(B, G + WL::BC1_OFF, D_C, OBS_N);
        // d_observe_critic 不写出（已知限制）
    }
    __syncthreads();

    // ---- 阶段 2: Actor BPTT（仅 actor_trigger）----
    // 触发判定: step >= bptt_steps 且 grad_act[cache_i] ≠ 0
    bool actor_trigger = false;
    if (threadIdx.x == 0) {
        bool nonzero = false;
        for (int i = 0; i < ACT_DIM; ++i) {
            if (in.grad_act[cache_i * ACT_DIM + i] != 0.0f) { nonzero = true; break; }
        }
        actor_trigger = (step >= bptt_steps) && nonzero;
        A[0] = actor_trigger ? 1.0f : 0.0f;   // 广播标志
    }
    __syncthreads();
    actor_trigger = (A[0] != 0.0f);

    if (actor_trigger) {
        // BPTT 累加器布局（BUF_B 内, 跨块持久）
        float* d_h_prev_acc = B;                   // [0:D_H]
        float* d_h_new_s    = B + D_H;             // [D_H:2*D_H]
        float* d_c_prev_acc = B + 2 * D_H;         // [2*D_H:3*D_H]
        float* d_x_s        = B + 3 * D_H;         // [3*D_H:3*D_H+D_IN]  x_s/d_x_s 复用
        constexpr int LSTMO_TMP = 3 * D_H + D_IN;  // R1-R3 lstm_o temp
        for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
            d_h_prev_acc[i] = 0.0f;
            d_c_prev_acc[i] = 0.0f;
        }
        __syncthreads();

        // 分块遍历窗口（新→旧）: k=0 是最新步（cache_i），k=bptt_steps-1 是最旧
        int remaining = bptt_steps;
        int k_start = 0;   // 已处理的步数（距最新）
        while (remaining > 0) {
            int n = (remaining < N_MAX) ? remaining : N_MAX;

            // 载入 n 个 cache → CA[0..n-1]
            // 块内: slot s 对应 k = k_start + s（新→旧）
            for (int s = 0; s < n; ++s) {
                int k = k_start + s;
                int cur_i = ((step - 1 - k) % A2C + A2C) % A2C;   // 环绕，防负
                const AttnLstmCache<Config>* src = d_caches + net_id * A2C + cur_i;
                const float* sp = reinterpret_cast<const float*>(src);
                float*       dp = reinterpret_cast<float*>(&CA[s]);
                int total = sizeof(AttnLstmCache<Config>) / sizeof(float);
                for (int i = threadIdx.x; i < total; i += blockDim.x) {
                    dp[i] = sp[i];
                }
                __syncthreads();
            }

            // 块内逆序: s = n-1..0（最旧→最新？）注意 BPTT 从旧→新累加 d_h_prev
            // 实际 BPTT 应从最新→最旧遍历（d_h_prev 从最新步向旧步传播）
            // 块内 s=0 是最新（k=k_start），s=n-1 是最旧。逆序 = s=n-1..0 = 旧→新
            // 但 BPTT 需新→旧。因此应正序 s=0..n-1（新→旧），d_h_prev_acc 从新向旧传播。
            for (int s = 0; s < n; ++s) {
                int k = k_start + s;
                int cur_i = ((step - 1 - k) % A2C + A2C) % A2C;
                const AttnLstmCache<Config>& cs = CA[s];

                // --- R1-R3: 输出层反向 ---
                for (int i = threadIdx.x; i < ACT_DIM; i += blockDim.x) {
                    float g_act_k = in.grad_act[cur_i * ACT_DIM + i];
                    A[i] = Sigmoid::bwd(cs.act[i], g_act_k);
                }
                __syncthreads();
                for (int i = threadIdx.x; i < D_H2; i += blockDim.x) {
                    B[LSTMO_TMP + i] = cs.lstm_o[i];
                }
                __syncthreads();
                warp_gemm_backward_weight_col<Config>(A, B + LSTMO_TMP, G + WL::WF_OFF, ACT_DIM, D_H2, 1);
                __syncthreads();
                accumulate_bias_grad(A, G + WL::BF_OFF, ACT_DIM, 1);

                // d_z_o = Wf^T @ d_lstm_o → 散入 d_h_new[8:16]
                {
                    float* d_z_o_s = A + ACT_DIM;   // [D_H2]
                    warp_gemm_backward_input_col<Config>(W + WL::WF_OFF, A, d_z_o_s, ACT_DIM, D_H2, 1);
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
                float* d_gate_o_s = A;
                float* d_c_new_s  = A + D_H;
                float* d_c_alt_s  = A + 2 * D_H;
                float* d_gate_i_s = A + 3 * D_H;
                for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
                    float dhn = d_h_new_s[i];
                    float go  = cs.gate_o[i];
                    float gi  = cs.gate_i[i];
                    float th  = cs.tanhc[i];
                    float ca  = cs.c_alt[i];
                    float cp  = cs.c_prev[i];
                    d_gate_o_s[i] = dhn * th * Sigmoid::bwd(go, 1.0f);
                    d_c_new_s[i]  = dhn * go * Tanh::bwd(th, 1.0f) + d_c_prev_acc[i];
                    d_c_alt_s[i]  = d_c_new_s[i] * gi * Tanh::bwd(ca, 1.0f);
                    d_gate_i_s[i] = d_c_new_s[i] * (ca - cp) * Sigmoid::bwd(gi, 1.0f);
                    d_c_prev_acc[i] = d_c_new_s[i] * (1.0f - gi);
                }
                __syncthreads();

                // --- R5: 三门权梯 ---
                for (int i = threadIdx.x; i < D_IN; i += blockDim.x) {
                    d_x_s[i] = cs.x[i];   // x_s
                }
                __syncthreads();
                warp_gemm_3gate_backward_weight<Config>(
                    d_gate_i_s, d_c_alt_s, d_gate_o_s, d_x_s,
                    G + WL::WICO_OFF, D_H, D_IN);
                __syncthreads();
                for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
                    G[WL::BICO_OFF + i]           += d_gate_i_s[i];
                    G[WL::BICO_OFF + D_H + i]     += d_c_alt_s[i];
                    G[WL::BICO_OFF + 2 * D_H + i] += d_gate_o_s[i];
                }
                __syncthreads();

                // --- R6: d_x = Wico^T @ d_gates ---
                warp_gemm_3gate_backward_input<Config>(
                    W + WL::WICO_OFF, d_gate_i_s, d_c_alt_s, d_gate_o_s,
                    d_x_s, D_H, D_IN);
                __syncthreads();

                // --- R7: 拆分 d_x ---
                for (int i = threadIdx.x; i < D_H; i += blockDim.x) {
                    d_h_prev_acc[i] += d_x_s[i];
                }

                // --- R7b: σ'bwd (_observe) ---
                for (int i = threadIdx.x; i < H_Q * D_E; i += blockDim.x) {
                    float post_sig = cs.x[D_H + INNER_DIM + i];
                    float d_post   = d_x_s[D_H + INNER_DIM + i];
                    d_x_s[D_H + INNER_DIM + i] = Sigmoid::bwd(post_sig, d_post);
                }
                __syncthreads();

                // --- R8: grad_bcat ---
                for (int i = threadIdx.x; i < H_Q * D_E; i += blockDim.x) {
                    G[WL::BCAT_OFF + i] += d_x_s[D_H + INNER_DIM + i];
                }
                __syncthreads();

                // --- R9-R15: 注意力反向（每 q_head 串行）---
                for (int iter = 0; iter < H_Q; iter++) {
                    int q_head = iter;
                    int kv_head = q_head / Q_N;
                    int wkv_row_off = kv_head * D_E;
                    int wq_row_off  = q_head * D_E;

                    // 加载 K 到 A（后续 dQ 需要），加载 Q、dO 到 A[K_SIZE..]
                    {
                        int kv_off = kv_head * K_SIZE;
                        for (int i = threadIdx.x; i < K_SIZE; i += blockDim.x) {
                            A[i] = cs.k[kv_off + i];
                        }
                        for (int i = threadIdx.x; i < D_E; i += blockDim.x) {
                            A[K_SIZE + i]       = cs.q[q_head * D_E + i];
                            A[K_SIZE + D_E + i] = d_x_s[D_H + INNER_DIM + q_head * D_E + i];
                        }
                    }
                    __syncthreads();

                    float* K_buf  = A;
                    float* Q_buf  = A + K_SIZE;
                    float* dO_buf = A + K_SIZE + D_E;

                    // dP[j] = Σ_e dO[e] * V[e,j]
                    float* dP_buf = A + K_SIZE + 2 * D_E;
                    for (int j = threadIdx.x; j < OBS_N; j += blockDim.x) {
                        int kv_off = kv_head * K_SIZE;
                        float acc = 0.0f;
                        for (int e = 0; e < D_E; ++e) {
                            acc += dO_buf[e] * cs.v[kv_off + e * OBS_N + j];
                        }
                        dP_buf[j] = acc;
                    }
                    __syncthreads();

                    // dS = P * (dP - Σ dP·P) / sqrt(D_E)
                    float* dS_buf = dP_buf + OBS_N;
                    {
                        if (threadIdx.x == 0) {
                            float sum_local = 0.0f;
                            for (int j = 0; j < OBS_N; j++) {
                                float p_j = cs.p[(q_head * H_KV + kv_head) * OBS_N + j];
                                sum_local += dP_buf[j] * p_j;
                            }
                            dS_buf[0] = sum_local;
                        }
                        __syncthreads();
                        float sum_all = dS_buf[0];
                        __syncthreads();
                        float inv_sqrt_de = rsqrtf((float)D_E);
                        for (int j = threadIdx.x; j < OBS_N; j += blockDim.x) {
                            float p_j = cs.p[(q_head * H_KV + kv_head) * OBS_N + j];
                            dS_buf[j] = p_j * (dP_buf[j] - sum_all) * inv_sqrt_de;
                        }
                    }
                    __syncthreads();

                    // dK[e,j] = Q[e] * dS[j]
                    float* dK_buf = A;   // 复用 K_buf 区
                    for (int idx = threadIdx.x; idx < K_SIZE; idx += blockDim.x) {
                        int e = idx / OBS_N;
                        int j = idx % OBS_N;
                        dK_buf[e * OBS_N + j] = Q_buf[e] * dS_buf[j];
                    }
                    __syncthreads();

                    // 累加 grad_Wk/Wv
                    for (int idx = threadIdx.x; idx < D_E * OBS_DIM; idx += blockDim.x) {
                        int e = idx % D_E;
                        int i = idx / D_E;
                        float dO_e = dO_buf[e];
                        float gk = 0.0f, gv = 0.0f;
                        for (int j = 0; j < OBS_N; j++) {
                            float p_j = cs.p[(q_head * H_KV + kv_head) * OBS_N + j];
                            gk += dK_buf[e * OBS_N + j] * cs.x_kv[i * OBS_N + j];
                            gv += dO_e * p_j * cs.x_kv[i * OBS_N + j];
                        }
                        G[WL::WKV_OFF + i * WKV_STRIDE + wkv_row_off + e] += gk;
                        G[WL::WKV_OFF + (i + OBS_DIM) * WKV_STRIDE + wkv_row_off + e] += gv;
                    }
                    __syncthreads();

                    // dQ[e] = Σ_j dS[j] * K[e,j]（重载 K）
                    {
                        int kv_off = kv_head * K_SIZE;
                        for (int i = threadIdx.x; i < K_SIZE; i += blockDim.x) {
                            A[i] = cs.k[kv_off + i];
                        }
                        __syncthreads();
                    }
                    float* K_reload = A;
                    float* dQ_buf   = A + K_SIZE;
                    for (int e = threadIdx.x; e < D_E; e += blockDim.x) {
                        float acc = 0.0f;
                        for (int j = 0; j < OBS_N; ++j) {
                            acc += dS_buf[j] * K_reload[e * OBS_N + j];
                        }
                        dQ_buf[e] = acc;
                    }
                    __syncthreads();

                    // grad_Wq + d_h_prev_Q
                    for (int idx = threadIdx.x; idx < D_E * D_H1; idx += blockDim.x) {
                        int e = idx % D_E;
                        int k = idx / D_E;
                        float dQ_e = dQ_buf[e];
                        float hp_k = cs.h_prev[k];
                        G[WL::WQ_OFF + k * WQ_STRIDE + wq_row_off + e] += dQ_e * hp_k;
                    }
                    for (int k = threadIdx.x; k < D_H1; k += blockDim.x) {
                        float acc = 0.0f;
                        for (int e = 0; e < D_E; ++e) {
                            acc += W[WL::WQ_OFF + k * WQ_STRIDE + wq_row_off + e] * dQ_buf[e];
                        }
                        d_h_prev_acc[k] += acc;
                    }
                    __syncthreads();
                }
            }

            k_start += n;
            remaining -= n;
        }
    }
    __syncthreads();

    // ---- 阶段 3: 流式 SGD（v=β·v+g; w=w·(1−lr·γ)+lr·v）----
    // WEIGHTS_B = g（本步梯度累计）；v_old/w_old 从 gmem 流式读，w_new/v_new 流式写
    {
        float* w_gmem = reinterpret_cast<float*>(&record.weight);
        float* v_gmem = reinterpret_cast<float*>(&record.grad);
        int total = sizeof(AttnLstmWeights<Config>) / sizeof(float);
        float one_minus_lr_gamma = 1.0f - lr * gamma;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            float v_old = v_gmem[i];
            float g_i   = G[i];
            float v_new = beta * v_old + g_i;
            float w_old = w_gmem[i];
            float w_new = w_old * one_minus_lr_gamma + lr * v_new;
            w_gmem[i] = w_new;
            v_gmem[i] = v_new;
        }
    }
}

}  // namespace agent_gpu
