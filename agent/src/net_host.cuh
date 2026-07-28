/**
 * @file net_host.cuh
 * @brief AttnLSTM Actor + FCN Critic 的 host 端管理器
 *
 * 管理四区全局内存（见 doc/attn-lstm-implementation.md "全局内存布局"）：
 *   Gmem1: d_records  [num_networks]            = AttnLstmNetRecord{state, weight, grad}
 *   Gmem2: d_caches   [num_networks*A2C_STEPS]  = AttnLstmCache（环形）
 *   Gmem3: d_input    [buf_length]              = AttnLstmInput（CPU→GPU，固定）
 *   Gmem4: d_output   [buf_length]              = AttnLstmOutput（GPU→CPU，固定）
 *
 * 设计要点：
 * - state 在 Gmem1 记录内原位读写（前向），故无 d_persistent_out / advance_state。
 * - cache 环形：前向写槽 = input.step % A2C_STEPS；反向读起点同式（由 kernel 计算）。
 * - 权重/梯度分离：weight 只读，grad 独立同型 AttnLstmWeights。
 *
 * 设计模式参考 test/test-cutlass/src/fnn_host.cuh:FNNHandle
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "net_config.cuh"
#include "net_kernel.cuh"

namespace agent_gpu {

template <typename Config = DefaultConfig>
struct AttnLstmHandle {
    static constexpr int OBS_N     = Config::OBS_N;
    static constexpr int OBS_DIM   = Config::OBS_DIM;
    static constexpr int INNER_DIM = Config::INNER_DIM;
    static constexpr int ACT_DIM   = Config::ACT_DIM;
    static constexpr int D_H       = Config::LSTM_HIDDEN_DIM;
    static constexpr int A2C_STEPS = Config::A2C_STEPS;
    static constexpr int BLK       = Config::BLOCK_DIM;

    int num_networks;
    int buf_length;   // Gmem3/4 通信缓冲容量（固定，≥ num_networks）

    // ---- Device memory（四区） ----
    AttnLstmNetRecord<Config>*  d_records;   // Gmem1 [num_networks]
    AttnLstmCache<Config>*      d_caches;    // Gmem2 [num_networks * A2C_STEPS]
    AttnLstmInput<Config>*      d_input;     // Gmem3 [buf_length]
    AttnLstmOutput<Config>*     d_output;    // Gmem4 [buf_length]

    cudaStream_t stream;

    // 记录内偏移（Gmem1 打包：[state | weight | grad]）
    static constexpr size_t STATE_OFF  = 0;
    static constexpr size_t WEIGHT_OFF = sizeof(AttnLstmPersistent<Config>);
    static constexpr size_t GRAD_OFF   = WEIGHT_OFF + sizeof(AttnLstmWeights<Config>);
    static constexpr size_t RECORD_SZ  = sizeof(AttnLstmNetRecord<Config>);

    /// 分配内存（num_nets 网络；buf_len 通信缓冲容量，默认 = num_nets）
    void alloc(int num_nets, int buf_len = -1) {
        num_networks = num_nets;
        buf_length   = (buf_len < 0) ? num_nets : buf_len;
        if (buf_length < num_networks) buf_length = num_networks;

        cudaStreamCreate(&stream);

        size_t rbytes = num_networks * sizeof(AttnLstmNetRecord<Config>);
        size_t cbytes = num_networks * A2C_STEPS * sizeof(AttnLstmCache<Config>);
        size_t ibytes = buf_length   * sizeof(AttnLstmInput<Config>);
        size_t obytes = buf_length   * sizeof(AttnLstmOutput<Config>);

        auto check = [](cudaError_t e, const char* label) {
            if (e != cudaSuccess) {
                fprintf(stderr, "FATAL alloc %s: %s\n", label, cudaGetErrorString(e));
                fflush(stderr);
            }
        };

        check(cudaMalloc(&d_records, rbytes), "records");
        check(cudaMalloc(&d_caches,  cbytes), "caches");
        check(cudaMalloc(&d_input,   ibytes), "input");
        check(cudaMalloc(&d_output,  obytes), "output");

        check(cudaMemset(d_records, 0, rbytes), "memset records");
    }

    /// 释放内存
    void free() {
        cudaFree(d_records);
        cudaFree(d_caches);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }

    // ================================================================
    // 权重 / 梯度 / 状态 拷贝（Gmem1）
    // ================================================================

    /// Host → Device 权重（写入各 record.weight）
    void copy_weights_from_host(const AttnLstmWeights<Config>* h_weights) {
        for (int n = 0; n < num_networks; ++n) {
            char* dst = reinterpret_cast<char*>(d_records) + n * RECORD_SZ + WEIGHT_OFF;
            cudaMemcpyAsync(dst, h_weights + n, sizeof(AttnLstmWeights<Config>),
                            cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }

    /// Device → Host 权重（从各 record.weight 读）
    void copy_weights_to_host(AttnLstmWeights<Config>* h_weights) {
        for (int n = 0; n < num_networks; ++n) {
            const char* src = reinterpret_cast<const char*>(d_records) + n * RECORD_SZ + WEIGHT_OFF;
            cudaMemcpyAsync(h_weights + n, src, sizeof(AttnLstmWeights<Config>),
                            cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
    }

    /// Device → Host 梯度（从各 record.grad 读）
    void copy_grads_to_host(AttnLstmWeights<Config>* h_grads) {
        for (int n = 0; n < num_networks; ++n) {
            const char* src = reinterpret_cast<const char*>(d_records) + n * RECORD_SZ + GRAD_OFF;
            cudaMemcpyAsync(h_grads + n, src, sizeof(AttnLstmWeights<Config>),
                            cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
    }

    /// Host → Device 持久状态（写入各 record.state）
    void copy_persistent_from_host(const AttnLstmPersistent<Config>* h_persistent) {
        for (int n = 0; n < num_networks; ++n) {
            char* dst = reinterpret_cast<char*>(d_records) + n * RECORD_SZ + STATE_OFF;
            cudaMemcpyAsync(dst, h_persistent + n, sizeof(AttnLstmPersistent<Config>),
                            cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }

    /// Device → Host 持久状态（从各 record.state 读；前向原位写回后的 h_new/c_new）
    void copy_persistent_to_host(AttnLstmPersistent<Config>* h_persistent) {
        for (int n = 0; n < num_networks; ++n) {
            const char* src = reinterpret_cast<const char*>(d_records) + n * RECORD_SZ + STATE_OFF;
            cudaMemcpyAsync(h_persistent + n, src, sizeof(AttnLstmPersistent<Config>),
                            cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
    }

    // ================================================================
    // 输入 / 输出（Gmem3 / Gmem4）
    // ================================================================

    inline char* input_ptr(int net) {
        return reinterpret_cast<char*>(d_input) + net * sizeof(AttnLstmInput<Config>);
    }
    inline const char* output_ptr(int net) const {
        return reinterpret_cast<const char*>(d_output) + net * sizeof(AttnLstmOutput<Config>);
    }

    /// 设置网络 net 的 observe（Gmem3）
    void set_input_observe(int net, const float* host) {
        size_t off = offsetof(AttnLstmInput<Config>, observe);
        cudaMemcpyAsync(input_ptr(net) + off, host,
                        OBS_DIM * OBS_N * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    /// 设置网络 net 的 inner
    void set_input_inner(int net, const float* host) {
        size_t off = offsetof(AttnLstmInput<Config>, inner);
        cudaMemcpyAsync(input_ptr(net) + off, host,
                        INNER_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    /// 设置网络 net 的当前步 Actor 梯度（写入 grad_act[0:ACT_DIM]）
    void set_grad_act(int net, const float* host) {
        size_t off = offsetof(AttnLstmInput<Config>, grad_act);
        cudaMemcpyAsync(input_ptr(net) + off, host,
                        ACT_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    /// 设置网络 net 的 slot 步 Actor 梯度（写入 grad_act[slot*ACT_DIM:(slot+1)*ACT_DIM]）
    void set_grad_act_at_slot(int net, int slot, const float* host) {
        size_t off = offsetof(AttnLstmInput<Config>, grad_act) + slot * ACT_DIM * sizeof(float);
        cudaMemcpyAsync(input_ptr(net) + off, host,
                        ACT_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    /// 设置网络 net 的当前步 Critic 梯度（写入 grad_value[0:1]）
    void set_grad_value(int net, const float* host) {
        size_t off = offsetof(AttnLstmInput<Config>, grad_value);
        cudaMemcpyAsync(input_ptr(net) + off, host,
                        1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    /// 设置网络 net 的 slot 步 Critic 梯度（写入 grad_value[slot]）
    void set_grad_value_at_slot(int net, int slot, const float* host) {
        size_t off = offsetof(AttnLstmInput<Config>, grad_value) + slot * sizeof(float);
        cudaMemcpyAsync(input_ptr(net) + off, host,
                        1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    /// 设置网络 net 的内时间步
    void set_input_step(int net, int step) {
        size_t off = offsetof(AttnLstmInput<Config>, step);
        cudaMemcpyAsync(input_ptr(net) + off, &step, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }
    /// 整块设置网络 net 的输入
    void copy_input_from_host(int net, const AttnLstmInput<Config>* host) {
        cudaMemcpyAsync(input_ptr(net), host, sizeof(AttnLstmInput<Config>),
                        cudaMemcpyHostToDevice, stream);
    }

    /// 读取网络 net 的 Actor 输出 act
    void get_output_act(int net, float* host) {
        size_t off = offsetof(AttnLstmOutput<Config>, act);
        cudaMemcpyAsync(host, output_ptr(net) + off,
                        ACT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    /// 读取网络 net 的 Critic 输出 value
    void get_output_value(int net, float* host) {
        size_t off = offsetof(AttnLstmOutput<Config>, value);
        cudaMemcpyAsync(host, output_ptr(net) + off,
                        1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    /// 整块读取网络 net 的输出
    void copy_output_to_host(int net, AttnLstmOutput<Config>* host) {
        cudaMemcpyAsync(host, output_ptr(net), sizeof(AttnLstmOutput<Config>),
                        cudaMemcpyDeviceToHost, stream);
    }

    // ================================================================
    // 梯度清零 / SGD
    // ================================================================

    /// 梯度清零（各 record.grad）
    void zero_gradients() {
        for (int n = 0; n < num_networks; ++n) {
            char* base = reinterpret_cast<char*>(d_records) + n * RECORD_SZ + GRAD_OFF;
            cudaMemsetAsync(base, 0, sizeof(AttnLstmWeights<Config>), stream);
        }
        cudaStreamSynchronize(stream);
    }

    /// 同步
    void sync() { cudaStreamSynchronize(stream); }

    // 注：SGD 更新（带动量 + 解耦 L2）已在反向 kernel 末尾流式完成，
    //     故 host 端不再提供 apply_gradients_sgd。

    // ================================================================
    // 权重初始化（Xavier normal）
    // ================================================================
    void init_weights_xavier(unsigned int seed = 42) {
        AttnLstmWeights<Config>* h_weights =
            new AttnLstmWeights<Config>[num_networks];

        srand(seed);
        for (int n = 0; n < num_networks; ++n) {
            auto& w = h_weights[n];

            auto xavier_fill = [](float* arr, int size, int fan_in) {
                float std = 1.0f / sqrtf((float)fan_in);
                for (int i = 0; i < size; ++i) {
                    float u1 = (float)rand() / (float)RAND_MAX;
                    float u2 = (float)rand() / (float)RAND_MAX;
                    if (u1 < 1e-8f) u1 = 1e-8f;
                    arr[i] = std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                }
            };

            auto small_rand = [](float* arr, int size) {
                for (int i = 0; i < size; ++i) {
                    arr[i] = 0.01f * (float)rand() / (float)RAND_MAX;
                }
            };

            xavier_fill(w.Wkv,  2 * Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_DIM, Config::OBS_DIM);
            xavier_fill(w.Wq,   Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM * Config::LSTM_QUERY_DIM, Config::LSTM_QUERY_DIM);
            xavier_fill(w.Wico, 3 * Config::LSTM_HIDDEN_DIM * Config::LSTM_INPUT_DIM, Config::LSTM_INPUT_DIM);
            xavier_fill(w.Wf,   Config::LSTM_OUTPUT_DIM * Config::ACT_DIM, Config::LSTM_OUTPUT_DIM);
            xavier_fill(w.Wc1,  Config::CRITIC_HIDDEN_DIM * Config::OBS_DIM, Config::OBS_DIM);
            xavier_fill(w.Wc2,  Config::CRITIC_HIDDEN_DIM * (Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM),
                        Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM);
            xavier_fill(w.Wc3,  Config::CRITIC_HIDDEN_DIM * Config::CRITIC_HIDDEN_DIM, Config::CRITIC_HIDDEN_DIM);
            xavier_fill(w.Wc4,  1 * Config::CRITIC_HIDDEN_DIM, Config::CRITIC_HIDDEN_DIM);

            small_rand(w.bcat, Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM);
            small_rand(w.bico, 3 * Config::LSTM_HIDDEN_DIM);
            small_rand(w.bf,   Config::ACT_DIM);
            small_rand(w.bc1,  Config::CRITIC_HIDDEN_DIM);
            small_rand(w.bc2,  Config::CRITIC_HIDDEN_DIM);
            small_rand(w.bc3,  Config::CRITIC_HIDDEN_DIM);
            small_rand(w.bc4,  1);
        }

        copy_weights_from_host(h_weights);
        delete[] h_weights;
    }

    // ================================================================
    // Kernel 启动
    // ================================================================

    /// 前向：每步执行；cache 环写槽 = d_input.step % A2C_STEPS（kernel 内计算）
    void forward() {
        dim3 block(BLK);
        dim3 grid(num_networks);
        size_t smem = SmemLayoutFwd<Config>::TOTAL * sizeof(float);
        attn_lstm_forward_kernel<Config>
            <<<grid, block, smem, stream>>>(d_records, d_input, d_output, d_caches);
    }

    /// 反向：每步执行；Critic 每步，Actor 触发时遍历窗口（kernel 内）；
    /// 末尾流式 SGD（v=β·v+g; w=w·(1−lr·γ)+lr·v）
    /// @param bptt_steps BPTT 回传步数（-1 表示使用 A2C_STEPS）
    void backward(float lr = 1e-5f, float beta = 0.9f, float gamma = 0.0f,
                  int bptt_steps = -1) {
        if (bptt_steps < 0) bptt_steps = A2C_STEPS;
        dim3 block(BLK);
        dim3 grid(num_networks);
        size_t smem = SmemLayoutBwd<Config>::TOTAL * sizeof(float);
        attn_lstm_backward_kernel<Config>
            <<<grid, block, smem, stream>>>(d_records, d_input, d_caches,
                                            lr, beta, gamma, bptt_steps);
    }
};

}  // namespace agent_gpu
