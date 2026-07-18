/*
 * fnn_host.cuh — Host-side management for FNN networks.
 *
 * Provides initialization, weight randomization, kernel launching,
 * and gradient-based weight updates.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "fnn_config.cuh"
#include "fnn_kernel.cuh"

namespace fnn {

// ===================================================================
// FNNHandle — manages device memory for a collection of networks
// ===================================================================
template <typename Config = DefaultConfig>
struct FNNHandle {
    static constexpr int IN  = Config::IN_DIM;
    static constexpr int H1  = Config::H1_DIM;
    static constexpr int H2  = Config::H2_DIM;
    static constexpr int OUT = Config::OUT_DIM;
    static constexpr int B   = Config::BATCH;
    static constexpr int BLK = Config::BLOCK_DIM;

    int num_networks;

    NetworkWeights<Config>*  d_weights;
    float*                   d_inputs;
    float*                   d_outputs;
    float*                   d_loss_grad;
    float*                   d_grad_inputs;
    ActivationCache<Config>* d_cache;

    cudaStream_t stream;

    // ---- Construction / destruction ----

    void alloc(int num_nets) {
        num_networks = num_nets;
        cudaError_t e;
        e = cudaStreamCreate(&stream);

        size_t wbytes  = num_nets * sizeof(NetworkWeights<Config>);
        size_t ibytes  = num_nets * IN  * B * sizeof(float);
        size_t obytes  = num_nets * OUT * B * sizeof(float);
        size_t cbytes  = num_nets * sizeof(ActivationCache<Config>);

        e = cudaMalloc(&d_weights,    wbytes);
        if (e != cudaSuccess) { fprintf(stderr, "FATAL alloc weights: %s\n", cudaGetErrorString(e)); fflush(stderr); }
        e = cudaMalloc(&d_inputs,     ibytes);
        e = cudaMalloc(&d_outputs,    obytes);
        e = cudaMalloc(&d_loss_grad,  obytes);
        e = cudaMalloc(&d_grad_inputs, ibytes);
        e = cudaMalloc(&d_cache,      cbytes);

        // Zero-initialize weights and grads
        e = cudaMemset(d_weights, 0, wbytes);
    }

    void free() {
        cudaFree(d_weights);
        cudaFree(d_inputs);
        cudaFree(d_outputs);
        cudaFree(d_loss_grad);
        cudaFree(d_grad_inputs);
        cudaFree(d_cache);
        cudaStreamDestroy(stream);
    }

    // ---- Weight initialization (Xavier normal) ----

    void init_weights_xavier(unsigned int seed = 42) {
        // Allocate and fill on host, then copy
        NetworkWeights<Config>* h_weights =
            new NetworkWeights<Config>[num_networks];

        srand(seed);
        for (int n = 0; n < num_networks; ++n) {
            auto& w = h_weights[n];

            // Xavier normal: std = sqrt(2 / (fan_in + fan_out)) → simplified: 1/sqrt(fan_in)
            auto xavier_fill = [&](float* arr, int size, int fan_in) {
                float std = 1.0f / sqrtf((float)fan_in);
                for (int i = 0; i < size; ++i) {
                    // Box-Muller using rand()
                    float u1 = (float)rand() / (float)RAND_MAX;
                    float u2 = (float)rand() / (float)RAND_MAX;
                    // Avoid log(0)
                    if (u1 < 1e-8f) u1 = 1e-8f;
                    arr[i] = std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                }
            };

            xavier_fill(w.W1, H1 * IN, IN);
            xavier_fill(w.W2, H2 * H1, H1);
            xavier_fill(w.W3, OUT * H2, H2);

            // Biases: small random
            for (int i = 0; i < H1; ++i) w.b1[i] = 0.01f * (float)rand() / RAND_MAX;
            for (int i = 0; i < H2; ++i) w.b2[i] = 0.01f * (float)rand() / RAND_MAX;
            for (int i = 0; i < OUT; ++i) w.b3[i] = 0.01f * (float)rand() / RAND_MAX;
        }

        cudaMemcpyAsync(d_weights, h_weights,
                        num_networks * sizeof(NetworkWeights<Config>),
                        cudaMemcpyHostToDevice, stream);
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { fprintf(stderr, "FATAL init_weights: %s\n", cudaGetErrorString(e)); fflush(stderr); }
        delete[] h_weights;
    }

    // ---- Copy weights from host buffer (for golden data testing) ----

    void copy_weights_from_host(const NetworkWeights<Config>* h_weights) {
        cudaMemcpyAsync(d_weights, h_weights,
                        num_networks * sizeof(NetworkWeights<Config>),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    void copy_weights_to_host(NetworkWeights<Config>* h_weights) {
        cudaMemcpyAsync(h_weights, d_weights,
                        num_networks * sizeof(NetworkWeights<Config>),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // ---- Zero gradients ----

    void zero_gradients() {
        // Zero the gradient fields in each NetworkWeights
        // We launch a kernel for this
        for (int n = 0; n < num_networks; ++n) {
            size_t offset = offsetof(NetworkWeights<Config>, grad_W1);
            size_t grad_bytes = sizeof(NetworkWeights<Config>) - offset;
            char* base = reinterpret_cast<char*>(d_weights) +
                         n * sizeof(NetworkWeights<Config>) + offset;
            cudaMemsetAsync(base, 0, grad_bytes, stream);
        }
        cudaStreamSynchronize(stream);
    }

    // ---- Launch forward ----

    void forward() {
        dim3 block(BLK);
        dim3 grid(num_networks);
        size_t smem = SmemLayout<Config>::TOTAL_FLOATS * sizeof(float);

        fnn_forward_kernel<Config>
            <<<grid, block, smem, stream>>>
            (d_weights, d_inputs, d_outputs, d_cache);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "FATAL forward kernel launch: %s\n", cudaGetErrorString(err));
        }
    }

    // ---- Launch backward ----

    void backward() {
        dim3 block(BLK);
        dim3 grid(num_networks);
        size_t smem = SmemLayout<Config>::TOTAL_FLOATS * sizeof(float);

        fnn_backward_kernel<Config>
            <<<grid, block, smem, stream>>>
            (d_weights, d_loss_grad, d_grad_inputs, d_cache);
    }

    // ---- Synchronize ----

    void sync() {
        cudaStreamSynchronize(stream);
    }

    // ---- Apply gradients with SGD ----

    void apply_gradients_sgd(float lr) {
        // Launched as a simple per-network kernel later, or done on host.
        // For now, copy to host, apply, copy back.
        NetworkWeights<Config>* h = new NetworkWeights<Config>[num_networks];
        copy_weights_to_host(h);
        for (int n = 0; n < num_networks; ++n) {
            auto& w = h[n];
            for (int i = 0; i < H1 * IN; ++i) w.W1[i] -= lr * w.grad_W1[i];
            for (int i = 0; i < H1;      ++i) w.b1[i] -= lr * w.grad_b1[i];
            for (int i = 0; i < H2 * H1; ++i) w.W2[i] -= lr * w.grad_W2[i];
            for (int i = 0; i < H2;      ++i) w.b2[i] -= lr * w.grad_b2[i];
            for (int i = 0; i < OUT * H2;++i) w.W3[i] -= lr * w.grad_W3[i];
            for (int i = 0; i < OUT;     ++i) w.b3[i] -= lr * w.grad_b3[i];
        }
        copy_weights_from_host(h);
        delete[] h;
    }
};

} // namespace fnn
