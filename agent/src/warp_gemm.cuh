/**
 * @file warp_gemm.cuh
 * @brief 智能体网络的warp级GEMM算子（列优先权重）与注意力并行算子
 *
 * 设计原则：
 * - 所有GEMM使用列优先权重布局（W[k*M + m]），与 fnn_warp_gemm.cuh 的行优先不同
 * - 列优先前向GEMM：每lane独立处理一行M，沿K全遍历，相邻lane读相邻m → coalesced
 * - M ≤ 32 始终成立（最大为 h_q*d_e=32），故无需 warp_reduce_sum
 * - 三门融合GEMM：x加载一次，复用3次计算i/c/o累加器
 * - 注意力并行：每warp处理一对 (h_q, h_kv)，4 warp覆盖全部4对
 *
 * 参考：test/test-cutlass/src/fnn_warp_gemm.cuh（行优先三件套）
 *       python/agent_numpy.py:GQA/LSTM（反向数学）
 */

#pragma once

#include <cuda_runtime.h>

#include "net_config.cuh"

namespace agent_gpu {

// ===================================================================
// Warp级归约原语
// ===================================================================

/// 全warp求和归约（所有32 lane获得相同结果）
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/// 全warp求最大值归约（所有32 lane获得相同结果）
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}

/// Warp级求最大值及其lane索引（lane0持有最终结果）
__device__ __forceinline__ float warp_reduce_max_with_idx(float val, int& idx, int lane) {
    // 用高32位存lane索引（按float位操作）
    // 简化实现：分别归约
    float max_val = val;
    int   max_idx = lane;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        int   other_idx = __shfl_xor_sync(0xFFFFFFFF, lane, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }
    // 广播最终结果
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
    max_idx = __shfl_sync(0xFFFFFFFF, max_idx, 0);
    idx = max_idx;
    return max_val;
}

// ===================================================================
// 激活函数（适配自 fnn_activations.cuh）
// ===================================================================

struct Sigmoid {
    static __device__ __forceinline__ float fwd(float x) {
        return 1.0f / (1.0f + __expf(-x));
    }
    /// dy = y * (1 - y) * grad （y是前向输出）
    static __device__ __forceinline__ float bwd(float y, float grad) {
        return y * (1.0f - y) * grad;
    }
};

struct Tanh {
    static __device__ __forceinline__ float fwd(float x) {
        return tanhf(x);
    }
    /// dy = (1 - y*y) * grad
    static __device__ __forceinline__ float bwd(float y, float grad) {
        return (1.0f - y * y) * grad;
    }
};

struct Identity {
    static __device__ __forceinline__ float fwd(float x) { return x; }
    static __device__ __forceinline__ float bwd(float, float grad) { return grad; }
};

/// 协作式前向激活（所有线程按stride处理连续数组）
template <typename ActFn>
__device__ void activation_apply_fwd(float* data, int count) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < count; i += stride) {
        data[i] = ActFn::fwd(data[i]);
    }
}

/// 协作式反向激活：grad[i] *= act'(out[i])
template <typename ActFn>
__device__ void activation_apply_bwd(const float* __restrict__ out,
                                     float* __restrict__ grad,
                                     int count) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < count; i += stride) {
        grad[i] = ActFn::bwd(out[i], grad[i]);
    }
}

/// 向量加偏置：mat[r*N + c] += bias[r]  （mat在smem中行优先）
__device__ void add_bias(float* __restrict__ mat,
                         const float* __restrict__ bias,
                         int M, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int total = M * N;
    for (int idx = tid; idx < total; idx += stride) {
        int r = idx / N;
        mat[idx] += bias[r];
    }
}

/// 累加偏置梯度：gb[r] += sum_c grad_mat[r*N + c]
__device__ void accumulate_bias_grad(const float* __restrict__ grad_mat,
                                     float* __restrict__ gb,
                                     int M, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int r = tid; r < M; r += stride) {
        float sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            sum += grad_mat[r * N + c];
        }
        gb[r] += sum;
    }
}

/// 逐元素向量加法：dst[i] += src[i]
__device__ void elementwise_add(float* __restrict__ dst,
                                const float* __restrict__ src,
                                int count) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < count; i += stride) {
        dst[i] += src[i];
    }
}

// ===================================================================
// 列优先 GEMM 算子
//
// 权重 W[M,K] 以列优先存储：W[k*M + m]
// 输入 X[K,N] 以行优先存储于smem
// 输出 C[M,N] 以行优先存储于smem
//
// 关键优化：M ≤ 32 时，每lane独立处理一行M，沿K全遍历 →
//   相邻lane读相邻m（同k）→ coalesced ✓
//   无需 warp_reduce_sum（每lane独立产出C[m,:]）
// ===================================================================

/**
 * @brief 列优先前向GEMM：C[M,N] = W[M,K] @ X[K,N]
 *
 * M按warp分组，warp内每lane独立计算一行C（当M≤32）
 *
 * @param W_gmem  权重 [M,K] 列优先（global memory）
 * @param X_smem  输入  [K,N] 行优先（shared memory）
 * @param C_smem  输出  [M,N] 行优先（shared memory，写入）
 * @param M       输出行数
 * @param K       归约维度
 * @param N       输出列数
 */
template <typename Config>
__device__ void warp_gemm_forward_col(
    const float* __restrict__ W_gmem,   // [M, K] 列优先: W[k*M + m]
    const float* __restrict__ X_smem,   // [K, N] 行优先
    float* __restrict__ C_smem,         // [M, N] 行优先
    int M, int K, int N)
{
    constexpr int WARPS = Config::WARPS;
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // M行按warp循环分配
    int rows_per_warp = (M + WARPS - 1) / WARPS;
    int m_start = warp_id * rows_per_warp;
    int m_end   = min(m_start + rows_per_warp, M);

    // 每lane处理一行m（步进32以覆盖m_start..m_end-1）
    for (int m = m_start + lane; m < m_end; m += 32) {
        // 对每列n
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            // 沿K全遍历：固定m，变k → W[k*M+m] ← 同k下相邻lane相邻m，coalesced
            for (int k = 0; k < K; ++k) {
                acc += W_gmem[k * M + m] * X_smem[k * N + n];
            }
            C_smem[m * N + n] = acc;
        }
    }
}

/**
 * @brief 列优先反向输入梯度：GX[K,N] = W[M,K]^T @ GY[M,N]
 *
 * 线程平铺(K,N)对，每线程沿M归约。
 * W读取：固定k，变m → W[k*M+m]连续 → coalesced
 *
 * @param W_gmem   权重 [M,K] 列优先（global memory）
 * @param GY_smem  上游梯度 [M,N] 行优先（shared memory）
 * @param GX_smem  输入梯度 [K,N] 行优先（shared memory，写入）
 */
template <typename Config>
__device__ void warp_gemm_backward_input_col(
    const float* __restrict__ W_gmem,   // [M, K] 列优先
    const float* __restrict__ GY_smem,  // [M, N] 行优先
    float* __restrict__ GX_smem,        // [K, N] 行优先
    int M, int K, int N)
{
    int tid   = threadIdx.x;
    int total = K * N;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int k = idx / N;
        int n = idx % N;
        float acc = 0.0f;
        // 固定k，遍历m → W[k*M+m] 连续，coalesced ✓
        for (int m = 0; m < M; ++m) {
            acc += W_gmem[k * M + m] * GY_smem[m * N + n];
        }
        GX_smem[idx] = acc;
    }
}

/**
 * @brief 列优先反向权重梯度：GW[M,K] += GY[M,N] @ X[K,N]^T
 *
 * 线程平铺(M,K)对，每线程独立累加，无需原子操作。
 * 列优先写：GW[k*M + m]
 *
 * @param GY_smem  上游梯度 [M,N] 行优先（shared memory）
 * @param X_smem   前向输入 [K,N] 行优先（shared memory）
 * @param GW_gmem  权重梯度 [M,K] 列优先（global memory，累加写入）
 */
template <typename Config>
__device__ void warp_gemm_backward_weight_col(
    const float* __restrict__ GY_smem,  // [M, N]
    const float* __restrict__ X_smem,   // [K, N]
    float* __restrict__ GW_gmem,        // [M, K] 列优先: GW[k*M + m]
    int M, int K, int N)
{
    int tid   = threadIdx.x;
    int total = M * K;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int m = idx / K;
        int k = idx % K;
        float acc = 0.0f;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            acc += GY_smem[m * N + n] * X_smem[k * N + n];
        }
        // 列优先索引
        GW_gmem[k * M + m] += acc;
    }
}

// ===================================================================
// 三门融合 GEMM（适配 Wico[d_h, 3*d_in] 列优先，耦合输入/遗忘门）
//
// Wico 列优先布局：[d_h, 3*d_in]，其中 d_in = LSTM_INPUT_DIM
//   列块0 (k=0..d_in-1):     Wi（输入门权重）
//   列块1 (k=d_in..2*d_in-1): Wc（候选门权重）
//   列块2 (k=2*d_in..3*d_in-1): Wo（输出门权重）
// 偏置 bico[3*d_h]：[bi | bc | bo]
//
// 正向：gate_i = sigmoid(Wi@x + bi), c_alt = tanh(Wc@x + bc), gate_o = sigmoid(Wo@x + bo)
// 关键优化：x只加载一次，复用3次计算三个累加器
// ===================================================================

/**
 * @brief 三门融合前向
 *
 * M=d_h ≤ 32，每lane独立处理一行，沿d_in全遍历，三累加器并行
 *
 * @param Wico_gmem Wico权重 [d_h, 3*d_in] 列优先
 * @param x_smem    拼接输入 [d_in] 行优先（shared memory）
 * @param bico_gmem 三门偏置 [3*d_h]（global memory）
 * @param gate_i    输出：输入门激活 [d_h]
 * @param c_alt     输出：候选细胞状态 [d_h]
 * @param gate_o    输出：输出门激活 [d_h]
 * @param M         d_h（LSTM隐维）
 * @param K_in      d_in（LSTM输入维）
 */
template <typename Config>
__device__ void warp_gemm_3gate_forward(
    const float* __restrict__ Wico_gmem,  // [M, 3*K_in] 列优先
    const float* __restrict__ x_smem,     // [K_in]
    const float* __restrict__ bico_gmem,  // [3*M]
    float* __restrict__ gate_i,           // [M]
    float* __restrict__ c_alt,            // [M]
    float* __restrict__ gate_o,           // [M]
    int M, int K_in)
{
    constexpr int WARPS = Config::WARPS;
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    int rows_per_warp = (M + WARPS - 1) / WARPS;
    int m_start = warp_id * rows_per_warp;
    int m_end   = min(m_start + rows_per_warp, M);

    int K_total = 3 * K_in;

    for (int m = m_start + lane; m < m_end; m += 32) {
        float acc_i = 0.0f;
        float acc_c = 0.0f;
        float acc_o = 0.0f;

        // 沿K_in全遍历，x加载一次复用三次
        for (int k = 0; k < K_in; ++k) {
            float xk = x_smem[k];
            // 三个列块：分别对应 Wi, Wc, Wo
            acc_i += Wico_gmem[k * M + m]             * xk;  // 列块0: Wi
            acc_c += Wico_gmem[(K_in + k) * M + m]     * xk;  // 列块1: Wc
            acc_o += Wico_gmem[(2 * K_in + k) * M + m] * xk;  // 列块2: Wo
        }

        // 加偏置 + 激活
        gate_i[m] = Sigmoid::fwd(acc_i + bico_gmem[m]);
        c_alt[m]  = Tanh::fwd(acc_c + bico_gmem[M + m]);
        gate_o[m] = Sigmoid::fwd(acc_o + bico_gmem[2 * M + m]);
    }
}

/**
 * @brief 三门融合反向——权重梯度
 *
 * grad_Wico[k*M + m] += d_gate[m] * x[k]
 * 三门梯度分别累加到对应列块
 *
 * @param d_gate_i  输入门上游梯度 [M]
 * @param d_c_alt    候选门上游梯度 [M]
 * @param d_gate_o  输出门上游梯度 [M]
 * @param x_smem    前向输入 [K_in]
 * @param GW_gmem   权重梯度累加区 [M, 3*K_in] 列优先
 * @param M         d_h
 * @param K_in      d_in
 */
template <typename Config>
__device__ void warp_gemm_3gate_backward_weight(
    const float* __restrict__ d_gate_i,   // [M]
    const float* __restrict__ d_c_alt,    // [M]
    const float* __restrict__ d_gate_o,   // [M]
    const float* __restrict__ x_smem,     // [K_in]
    float* __restrict__ GW_gmem,          // [M, 3*K_in] 列优先
    int M, int K_in)
{
    int tid = threadIdx.x;
    int K_total = 3 * K_in;
    int total = M * K_total;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int m = idx % M;             // 行索引（在三门块内复用）
        int k_block = idx / M;       // 全局列索引 (0..3*K_in-1)

        int gate_idx = k_block / K_in;  // 0→i, 1→c, 2→o
        int k        = k_block % K_in;

        float d_gate = (gate_idx == 0) ? d_gate_i[m] :
                       (gate_idx == 1) ? d_c_alt[m]  :
                                         d_gate_o[m];
        GW_gmem[idx] += d_gate * x_smem[k];
    }
}

/**
 * @brief 三门融合反向——输入梯度
 *
 * d_x[k] = sum_m (Wi^T@d_gate_i + Wc^T@d_c_alt + Wo^T@d_gate_o)
 *        = sum_m Wi[k*M+m]*d_gate_i[m] + Wc[K_in*M+k*M+m]*d_c_alt[m] + Wo[2*K_in*M+k*M+m]*d_gate_o[m]
 *
 * 线程平铺K_in，每线程沿M归约
 *
 * @param Wico_gmem 权重 [M, 3*K_in] 列优先
 * @param d_gate_i  三门上游梯度 [M]
 * @param d_c_alt   [M]
 * @param d_gate_o  [M]
 * @param d_x_smem  输出：输入梯度 [K_in]（shared memory）
 * @param M         d_h
 * @param K_in      d_in
 */
template <typename Config>
__device__ void warp_gemm_3gate_backward_input(
    const float* __restrict__ Wico_gmem,  // [M, 3*K_in] 列优先
    const float* __restrict__ d_gate_i,   // [M]
    const float* __restrict__ d_c_alt,    // [M]
    const float* __restrict__ d_gate_o,   // [M]
    float* __restrict__ d_x_smem,         // [K_in]
    int M, int K_in)
{
    int tid = threadIdx.x;

    for (int k = tid; k < K_in; k += blockDim.x) {
        float acc = 0.0f;
        // 沿M归约，Wico[k*M+m] 变m连续 → coalesced
        for (int m = 0; m < M; ++m) {
            acc += Wico_gmem[k * M + m]             * d_gate_i[m]
                 + Wico_gmem[(K_in + k) * M + m]     * d_c_alt[m]
                 + Wico_gmem[(2 * K_in + k) * M + m] * d_gate_o[m];
        }
        d_x_smem[k] = acc;
    }
}

// ===================================================================
// 注意力并行算子（每warp处理一对 (h_q, h_kv)）
//
// 4 warps 各管一个 Q 头：
//   warp 0 → (q_head=0, kv_head=0)
//   warp 1 → (q_head=1, kv_head=0)
//   warp 2 → (q_head=2, kv_head=1)
//   warp 3 → (q_head=3, kv_head=1)
//
// K/V 在同 h_kv 组内通过 smem 共享（warp 0,1 共享一组 K/V；warp 2,3 共享另一组）
//
// 这些算子假设相关数据已就位于 smem 中对应的 warp 专属区
// ===================================================================

/**
 * @brief 注意力QK点积：S[M] = Q[d_e]^T @ K[d_e, M] / sqrt(d_e)
 *
 * 单warp内：lane按stride-32遍历d_e，warp_reduce_sum归约。
 * 输出S[j]存于smem，lane0写入。
 *
 * @param Q_smem   查询向量 [d_e]（smem，本warp的Q头）
 * @param K_smem   键矩阵 [d_e, M] 行优先（smem，本warp对应的KV头）
 * @param S_smem   输出得分 [M]（smem，lane0写入）
 * @param d_e      嵌入维度
 * @param M        观测实体数
 * @param warp_lane 本线程在warp内的lane索引
 */
__device__ void attn_qk_gemm(
    const float* __restrict__ Q_smem,   // [d_e]
    const float* __restrict__ K_smem,   // [d_e, M] 行优先
    float* __restrict__ S_smem,         // [M]
    int d_e, int M,
    int warp_lane)
{
    float inv_sqrt_d = rsqrtf(static_cast<float>(d_e));

    for (int j = 0; j < M; ++j) {
        float acc = 0.0f;
        for (int e = warp_lane; e < d_e; e += 32) {
            acc += Q_smem[e] * K_smem[e * M + j];
        }
        acc = warp_reduce_sum(acc);
        if (warp_lane == 0) {
            S_smem[j] = acc * inv_sqrt_d;
        }
    }
}

/**
 * @brief Warp级数值稳定softmax前向
 *
 * P[j] = exp(S[j] - max(S)) / sum_j exp(S[j] - max(S))
 *
 * @param S_smem   输入得分 [M]（smem，原位修改为概率P）
 * @param M        序列长度
 * @param warp_lane 本线程的warp lane索引
 */
__device__ void softmax_fwd_warp(
    float* __restrict__ S_smem,   // [M] 输入→输出概率
    int M,
    int warp_lane)
{
    // 1. 找最大值（数值稳定性）
    float max_val = -INFINITY;
    for (int j = warp_lane; j < M; j += 32) {
        if (S_smem[j] > max_val) max_val = S_smem[j];
    }
    max_val = warp_reduce_max(max_val);

    // 2. exp(S - max) 并求和
    float sum_exp = 0.0f;
    for (int j = warp_lane; j < M; j += 32) {
        float val = __expf(S_smem[j] - max_val);
        S_smem[j] = val;
        sum_exp += val;
    }
    sum_exp = warp_reduce_sum(sum_exp);

    // 3. 归一化
    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (int j = warp_lane; j < M; j += 32) {
        S_smem[j] *= inv_sum;
    }
}

/**
 * @brief 注意力PV点积：O[d_e] = P[M] @ V[M, d_e]^T
 *
 * 由于d_e=8≤32：每lane独立计算一个e，迭代所有j，无需warp_reduce
 *
 * @param P_smem   注意力概率 [M]（smem）
 * @param V_smem   值矩阵 [d_e, M] 行优先（smem）
 * @param O_smem   输出注意力向量 [d_e]（smem）
 * @param d_e      嵌入维度
 * @param M        观测实体数
 * @param warp_lane 本线程lane索引
 */
__device__ void attn_pv_gemm(
    const float* __restrict__ P_smem,   // [M]
    const float* __restrict__ V_smem,   // [d_e, M] 行优先
    float* __restrict__ O_smem,         // [d_e]
    int d_e, int M,
    int warp_lane)
{
    // d_e=8 ≤ 32，每lane处理一个e维度
    if (warp_lane < d_e) {
        float acc = 0.0f;
        for (int j = 0; j < M; ++j) {
            acc += P_smem[j] * V_smem[warp_lane * M + j];
        }
        O_smem[warp_lane] = acc;
    }
}

/**
 * @brief Softmax反向：dS = P * (dP - sum_j(dP_j * P_j)) / sqrt(d_e)
 *
 * 参考 agent_numpy.py:GQA.backward 的softmax反向公式
 *
 * @param P_smem   前向softmax概率 [M]（smem，只读）
 * @param dP_smem  dL/dP [M]（smem，输入）
 * @param dS_smem  dL/dS [M]（smem，输出）
 * @param d_e      嵌入维度（用于缩放）
 * @param M        序列长度
 * @param warp_lane
 */
__device__ void softmax_bwd_warp(
    const float* __restrict__ P_smem,   // [M]
    const float* __restrict__ dP_smem,  // [M]
    float* __restrict__ dS_smem,        // [M]
    int d_e, int M,
    int warp_lane)
{
    // 计算 sum_j dP_j * P_j
    float sum_dot = 0.0f;
    for (int j = warp_lane; j < M; j += 32) {
        sum_dot += dP_smem[j] * P_smem[j];
    }
    sum_dot = warp_reduce_sum(sum_dot);

    float inv_sqrt_d = rsqrtf(static_cast<float>(d_e));

    for (int j = warp_lane; j < M; j += 32) {
        // dS_j = P_j * (dP_j - sum_dp_p) / sqrt(d_e)
        dS_smem[j] = P_smem[j] * (dP_smem[j] - sum_dot) * inv_sqrt_d;
    }
}

/**
 * @brief 注意力QK反向：dK[d_e,M] = Q[d_e] @ dS[M]^T,  dQ[d_e] = dS[M] @ K[d_e,M]^T
 *
 * @param Q_smem   前向Q [d_e]（smem）
 * @param K_smem   前向K [d_e, M] 行优先（smem，只读）
 * @param dS_smem  dL/dS [M]（smem）
 * @param dK_smem  dL/dK [d_e, M] 行优先（smem，输出）
 * @param dQ_smem  dL/dQ [d_e]（smem，输出/累加）
 * @param d_e      嵌入维度
 * @param M        序列长度
 * @param warp_lane
 */
__device__ void attn_qk_bwd(
    const float* __restrict__ Q_smem,    // [d_e]
    const float* __restrict__ K_smem,    // [d_e, M] 行优先
    const float* __restrict__ dS_smem,   // [M]
    float* __restrict__ dK_smem,         // [d_e, M] 行优先
    float* __restrict__ dQ_smem,         // [d_e] 累加
    int d_e, int M,
    int warp_lane)
{
    // dK[e,j] = Q[e] * dS[j]  (外积)
    for (int idx = warp_lane; idx < d_e * M; idx += 32) {
        int e = idx / M;
        int j = idx % M;
        dK_smem[idx] = Q_smem[e] * dS_smem[j];
    }

    // dQ[e] += sum_j K[e,j] * dS[j]
    if (warp_lane < d_e) {
        float acc = 0.0f;
        for (int j = 0; j < M; ++j) {
            acc += K_smem[warp_lane * M + j] * dS_smem[j];
        }
        dQ_smem[warp_lane] += acc;
    }
}

/**
 * @brief 注意力PV反向：dV[d_e,M] = P[M] @ dO[d_e]^T,  dP[M] = dO[d_e]^T @ V[d_e,M]
 *
 * dV[e,j] = P[j] * dO[e]  (外积)
 * dP[j] = sum_e dO[e] * V[e,j]
 *
 * @param P_smem   前向概率 [M]（smem）
 * @param V_smem   前向V [d_e, M] 行优先（smem，只读）
 * @param dO_smem  dL/dO [d_e]（smem）
 * @param dV_smem  dL/dV [d_e, M] 行优先（smem，输出）
 * @param dP_smem  dL/dP [M]（smem，输出）
 * @param d_e      嵌入维度
 * @param M        序列长度
 * @param warp_lane
 */
__device__ void attn_pv_bwd(
    const float* __restrict__ P_smem,    // [M]
    const float* __restrict__ V_smem,    // [d_e, M] 行优先
    const float* __restrict__ dO_smem,   // [d_e]
    float* __restrict__ dV_smem,         // [d_e, M] 行优先
    float* __restrict__ dP_smem,         // [M]
    int d_e, int M,
    int warp_lane)
{
    // dV[e,j] = P[j] * dO[e]
    for (int idx = warp_lane; idx < d_e * M; idx += 32) {
        int e = idx / M;
        int j = idx % M;
        dV_smem[idx] = P_smem[j] * dO_smem[e];
    }

    // dP[j] = sum_e dO[e] * V[e,j]
    for (int j = warp_lane; j < M; j += 32) {
        float acc = 0.0f;
        for (int e = 0; e < d_e; ++e) {
            acc += dO_smem[e] * V_smem[e * M + j];
        }
        dP_smem[j] = acc;
    }
}

} // namespace agent_gpu
