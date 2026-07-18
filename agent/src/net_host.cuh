/**
 * @file net_host.cuh
 * @brief AttnLSTM Actor + FCN Critic 的host端管理器
 *
 * 管理device内存分配/释放、权重初始化、kernel启动、梯度更新。
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
    static constexpr int OBS_N    = Config::OBS_N;
    static constexpr int OBS_DIM  = Config::OBS_DIM;
    static constexpr int INNER_DIM = Config::INNER_DIM;
    static constexpr int ACT_DIM  = Config::ACT_DIM;
    static constexpr int D_H      = Config::LSTM_HIDDEN_DIM;
    static constexpr int BLK      = Config::BLOCK_DIM;

    int num_networks;
    int num_caches_per_net;

    // Device memory
    AttnLstmWeights<Config>*  d_weights;
    AttnLstmPersistent<Config>* d_persistent;
    AttnLstmPersistent<Config>* d_persistent_out;
    float* d_observe;
    float* d_inner;
    float* d_act;
    float* d_value;
    float* d_grad_act;
    float* d_grad_value;
    AttnLstmCache<Config>* d_caches;

    cudaStream_t stream;

    // ================================================================
    // 分配
    // ================================================================
    void alloc(int num_nets, int caches_per_net = 1) {
        num_networks = num_nets;
        num_caches_per_net = caches_per_net;

        cudaStreamCreate(&stream);

        size_t wbytes   = num_nets * sizeof(AttnLstmWeights<Config>);
        size_t pbytes   = num_nets * sizeof(AttnLstmPersistent<Config>);
        size_t obytes   = num_nets * OBS_DIM * OBS_N * sizeof(float);
        size_t ibytes   = num_nets * INNER_DIM * sizeof(float);
        size_t abytes   = num_nets * ACT_DIM * sizeof(float);
        size_t vbytes   = num_nets * 1 * sizeof(float);
        size_t cbytes   = num_nets * caches_per_net * sizeof(AttnLstmCache<Config>);

        auto check = [](cudaError_t e, const char* label) {
            if (e != cudaSuccess) {
                fprintf(stderr, "FATAL alloc %s: %s\n", label, cudaGetErrorString(e));
                fflush(stderr);
            }
        };

        check(cudaMalloc(&d_weights,        wbytes), "weights");
        check(cudaMalloc(&d_persistent,     pbytes), "persistent");
        check(cudaMalloc(&d_persistent_out, pbytes), "persistent_out");
        check(cudaMalloc(&d_observe,        obytes), "observe");
        check(cudaMalloc(&d_inner,          ibytes), "inner");
        check(cudaMalloc(&d_act,            abytes), "act");
        check(cudaMalloc(&d_value,          vbytes), "value");
        check(cudaMalloc(&d_grad_act,       abytes), "grad_act");
        check(cudaMalloc(&d_grad_value,     vbytes), "grad_value");
        check(cudaMalloc(&d_caches,         cbytes), "caches");

        // Zero-initialize weights, gradients, persistent state
        check(cudaMemset(d_weights,    0, wbytes), "memset weights");
        check(cudaMemset(d_persistent, 0, pbytes), "memset persistent");
    }

    void free() {
        cudaFree(d_weights);
        cudaFree(d_persistent);
        cudaFree(d_persistent_out);
        cudaFree(d_observe);
        cudaFree(d_inner);
        cudaFree(d_act);
        cudaFree(d_value);
        cudaFree(d_grad_act);
        cudaFree(d_grad_value);
        cudaFree(d_caches);
        cudaStreamDestroy(stream);
    }

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

            int Wkv_size  = 2 * Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_DIM;
            int Wq_size   = Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM * Config::LSTM_QUERY_DIM;
            int bcat_size = Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM;
            int Wico_size = 3 * Config::LSTM_HIDDEN_DIM * Config::LSTM_INPUT_DIM;
            int bico_size = 3 * Config::LSTM_HIDDEN_DIM;
            int Wf_size   = Config::LSTM_OUTPUT_DIM * Config::ACT_DIM;
            int Wc1_size  = Config::CRITIC_HIDDEN_DIM * Config::OBS_DIM;
            int Wc2_size  = Config::CRITIC_HIDDEN_DIM * (Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM);
            int Wc3_size  = Config::CRITIC_HIDDEN_DIM * Config::CRITIC_HIDDEN_DIM;
            int Wc4_size  = 1 * Config::CRITIC_HIDDEN_DIM;

            xavier_fill(w.Wkv,  Wkv_size,  Config::OBS_DIM);
            xavier_fill(w.Wq,   Wq_size,   Config::LSTM_QUERY_DIM);
            xavier_fill(w.Wico, Wico_size, Config::LSTM_INPUT_DIM);
            xavier_fill(w.Wf,   Wf_size,   Config::LSTM_OUTPUT_DIM);
            xavier_fill(w.Wc1,  Wc1_size,  Config::OBS_DIM);
            xavier_fill(w.Wc2,  Wc2_size,  Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM);
            xavier_fill(w.Wc3,  Wc3_size,  Config::CRITIC_HIDDEN_DIM);
            xavier_fill(w.Wc4,  Wc4_size,  Config::CRITIC_HIDDEN_DIM);

            small_rand(w.bcat, bcat_size);
            small_rand(w.bico, bico_size);
            small_rand(w.bf,   Config::ACT_DIM);
            small_rand(w.bc1,  Config::CRITIC_HIDDEN_DIM);
            small_rand(w.bc2,  Config::CRITIC_HIDDEN_DIM);
            small_rand(w.bc3,  Config::CRITIC_HIDDEN_DIM);
            small_rand(w.bc4,  1);
        }

        cudaMemcpyAsync(d_weights, h_weights,
                        num_networks * sizeof(AttnLstmWeights<Config>),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        delete[] h_weights;
    }

    // ================================================================
    // Host ↔ Device 权重拷贝
    // ================================================================
    void copy_weights_from_host(const AttnLstmWeights<Config>* h_weights) {
        cudaMemcpyAsync(d_weights, h_weights,
                        num_networks * sizeof(AttnLstmWeights<Config>),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    void copy_weights_to_host(AttnLstmWeights<Config>* h_weights) {
        cudaMemcpyAsync(h_weights, d_weights,
                        num_networks * sizeof(AttnLstmWeights<Config>),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ================================================================
    // 持久状态操作
    // ================================================================
    void copy_persistent_from_host(const AttnLstmPersistent<Config>* h_persistent) {
        cudaMemcpyAsync(d_persistent, h_persistent,
                        num_networks * sizeof(AttnLstmPersistent<Config>),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    void copy_persistent_to_host(AttnLstmPersistent<Config>* h_persistent) {
        cudaMemcpyAsync(h_persistent, d_persistent_out,
                        num_networks * sizeof(AttnLstmPersistent<Config>),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ================================================================
    // 零梯度
    // ================================================================
    void zero_gradients() {
        // 计算梯度区偏移
        constexpr int GRAD_OFFSET_FLOATS =
            2 * Config::ATTN_KV_HEADS * Config::ATTN_EMBED_DIM * Config::OBS_DIM
            + Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM * Config::LSTM_QUERY_DIM
            + Config::ATTN_Q_HEADS * Config::ATTN_EMBED_DIM
            + 3 * Config::LSTM_HIDDEN_DIM * Config::LSTM_INPUT_DIM
            + 3 * Config::LSTM_HIDDEN_DIM
            + Config::LSTM_OUTPUT_DIM * Config::ACT_DIM
            + Config::ACT_DIM
            + Config::CRITIC_HIDDEN_DIM * Config::OBS_DIM
            + Config::CRITIC_HIDDEN_DIM
            + Config::CRITIC_HIDDEN_DIM * (Config::CRITIC_HIDDEN_DIM + Config::INNER_DIM)
            + Config::CRITIC_HIDDEN_DIM
            + Config::CRITIC_HIDDEN_DIM * Config::CRITIC_HIDDEN_DIM
            + Config::CRITIC_HIDDEN_DIM
            + 1 * Config::CRITIC_HIDDEN_DIM
            + 1;

        size_t grad_offset_bytes = GRAD_OFFSET_FLOATS * sizeof(float);
        size_t grad_bytes = sizeof(AttnLstmWeights<Config>) - grad_offset_bytes;

        for (int n = 0; n < num_networks; ++n) {
            char* base = reinterpret_cast<char*>(d_weights) +
                         n * sizeof(AttnLstmWeights<Config>) + grad_offset_bytes;
            cudaMemsetAsync(base, 0, grad_bytes, stream);
        }
        cudaStreamSynchronize(stream);
    }

    // ================================================================
    // 前向启动
    // ================================================================
    void forward(int cache_step = 0) {
        dim3 block(BLK);
        dim3 grid(num_networks);
        size_t smem = SmemLayout<Config>::TOTAL * sizeof(float);

        attn_lstm_forward_kernel<Config>
            <<<grid, block, smem, stream>>>(
                d_weights, d_persistent, d_persistent_out,
                d_observe, d_inner,
                d_act, d_value,
                d_caches + cache_step);
    }

    // ================================================================
    // 反向启动
    // ================================================================
    void backward(int num_steps = 1) {
        dim3 block(BLK);
        dim3 grid(num_networks);
        size_t smem = SmemLayout<Config>::TOTAL * sizeof(float);

        attn_lstm_backward_kernel<Config>
            <<<grid, block, smem, stream>>>(
                d_weights, d_grad_act, d_grad_value,
                d_caches, num_steps);
    }

    // ================================================================
    // 推进持久状态（多步forward时调用：d_persistent_out → d_persistent）
    // ================================================================
    void advance_state() {
        size_t pbytes = num_networks * sizeof(AttnLstmPersistent<Config>);
        cudaMemcpyAsync(d_persistent, d_persistent_out, pbytes,
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    // ================================================================
    // 同步
    // ================================================================
    void sync() {
        cudaStreamSynchronize(stream);
    }

    // ================================================================
    // SGD更新（host端实现）
    // ================================================================
    void apply_gradients_sgd(float lr) {
        AttnLstmWeights<Config>* h = new AttnLstmWeights<Config>[num_networks];
        copy_weights_to_host(h);

        for (int n = 0; n < num_networks; ++n) {
            auto& w = h[n];

            // 所有权重数组成员（不含梯度区）的迭代更新
            int num_weight_floats = sizeof(AttnLstmWeights<Config>) / sizeof(float) / 2;
            float* weights = reinterpret_cast<float*>(&w);
            float* grads   = weights + num_weight_floats;

            for (int i = 0; i < num_weight_floats; ++i) {
                weights[i] -= lr * grads[i];
            }
        }

        copy_weights_from_host(h);
        delete[] h;
    }
};

} // namespace agent_gpu
