/*
 * fnn_kernel.cuh — Single-block feedforward neural network kernels.
 *
 * One CUDA block = one network instance.
 * The grid dimension = number of independent networks.
 *
 * Forward kernel:  input → sigmoid(W1@x+b1) → sigmoid(W2@h1+b2) → W3@h2+b3
 * Backward kernel: output_grad → ... → input_grad + accumulated weight grads
 */

#pragma once

#include <cuda_runtime.h>

// NOTE: CUTLASS 4.5.1 is linked via CMake and its design patterns are followed
// (warp-level tiled GEMM, shared memory staging, AlignedBuffer conventions).
// Direct inclusion of cutlass/cutlass.h is avoided because CuTe requires
// C++20 features (auto NTTP) not available with MSVC 2019 host compiler.
// The warp GEMM implementation follows CUTLASS MmaSimt tiling strategy.

#include "fnn_config.cuh"
#include "fnn_activations.cuh"
#include "fnn_warp_gemm.cuh"

namespace fnn {

// ===================================================================
// NetworkWeights — per-network parameters and gradient accumulators
//
// Stored in global memory as an array: d_weights[num_networks].
// Each block accesses d_weights[blockIdx.x].
// ===================================================================
template <typename Config>
struct alignas(16) NetworkWeights {
    static constexpr int IN  = Config::IN_DIM;
    static constexpr int H1  = Config::H1_DIM;
    static constexpr int H2  = Config::H2_DIM;
    static constexpr int OUT = Config::OUT_DIM;

    // -- Weights (row-major) --
    float W1[H1 * IN];
    float b1[H1];
    float W2[H2 * H1];
    float b2[H2];
    float W3[OUT * H2];
    float b3[OUT];

    // -- Gradient accumulators (zeroed before backward, applied by host) --
    float grad_W1[H1 * IN];
    float grad_b1[H1];
    float grad_W2[H2 * H1];
    float grad_b2[H2];
    float grad_W3[OUT * H2];
    float grad_b3[OUT];
};

// ===================================================================
// Activation cache — forward activations needed by backward pass
//
// Stored in global memory as an array: d_cache[num_networks].
// Layout: [input_x | h1_out | h2_out]  (row-major within each segment)
// ===================================================================
template <typename Config>
struct alignas(16) ActivationCache {
    static constexpr int BYTES =
        (Config::IN_DIM + Config::H1_DIM + Config::H2_DIM)
        * Config::BATCH * sizeof(float);
    // Raw bytes — interpreted as [x | h1 | h2] in kernel
    char data[BYTES];
};

// ===================================================================
// Shared memory layout helper
//
// Dynamic shared memory is reinterpreted as float[] in the kernel.
// We name regions within it via offset/size pairs.
// ===================================================================
template <typename Config>
struct SmemLayout {
    static constexpr int IN  = Config::IN_DIM;
    static constexpr int H1  = Config::H1_DIM;
    static constexpr int H2  = Config::H2_DIM;
    static constexpr int OUT = Config::OUT_DIM;
    static constexpr int B   = Config::BATCH;

    // Maximum needed: max(input+output per layer) = max(IN+H1, H1+H2, H2+OUT) * B
    // We use two buffers: buf_a (input to layer), buf_b (output of layer)
    // and swap roles between layers.
    static constexpr int MAX_IN   = (IN > H1) ? IN : H1;   // largest layer input
    static constexpr int MAX_OUT  = (H1 > H2) ? ((H1 > OUT) ? H1 : OUT)
                                               : ((H2 > OUT) ? H2 : OUT);
    static constexpr int BUF_A_SIZE = MAX_IN  * B;
    static constexpr int BUF_B_SIZE = MAX_OUT * B;

    static constexpr int BUF_A_OFFSET = 0;
    static constexpr int BUF_B_OFFSET = BUF_A_SIZE;

    static constexpr int TOTAL_FLOATS = BUF_A_SIZE + BUF_B_SIZE;
};

// ===================================================================
// Initialize weights with Xavier-like normal distribution (host only)
// ===================================================================

// ===================================================================
// FORWARD KERNEL
// ===================================================================
template <typename Config>
__global__ void fnn_forward_kernel(
    const NetworkWeights<Config>* __restrict__ d_weights,   // [num_networks]
    const float*                   __restrict__ d_inputs,    // [num_networks][IN][BATCH]
    float*                         __restrict__ d_outputs,   // [num_networks][OUT][BATCH]
    ActivationCache<Config>*       __restrict__ d_cache)     // [num_networks]  intermediate values for backward
{
    constexpr int IN  = Config::IN_DIM;
    constexpr int H1  = Config::H1_DIM;
    constexpr int H2  = Config::H2_DIM;
    constexpr int OUT = Config::OUT_DIM;
    constexpr int B   = Config::BATCH;

    int net_id = blockIdx.x;
    const NetworkWeights<Config>& W = d_weights[net_id];
    ActivationCache<Config>& cache  = d_cache[net_id];

    // Dynamic shared memory
    extern __shared__ float smem[];
    using Smem = SmemLayout<Config>;
    float* buf_a = smem + Smem::BUF_A_OFFSET;  // input to current layer
    float* buf_b = smem + Smem::BUF_B_OFFSET;  // output of current layer

    // =================================================================
    // Layer 1:  h1 = sigmoid(W1 @ x + b1)
    // =================================================================
    // Load input x from global → shared memory (buf_a)
    {
        const float* x_gmem = d_inputs + net_id * (IN * B);
        for (int i = threadIdx.x; i < IN * B; i += blockDim.x) {
            buf_a[i] = x_gmem[i];
        }
    }
    __syncthreads();

    // Cache input x for backward pass (needed by grad_W1 = ... @ x^T)
    {
        float* cache_x = reinterpret_cast<float*>(cache.data);
        for (int i = threadIdx.x; i < IN * B; i += blockDim.x) {
            cache_x[i] = buf_a[i];
        }
    }

    // W1 @ x → buf_b
    warp_gemm_forward<Config>(W.W1, buf_a, buf_b, H1, IN, B);
    __syncthreads();

    // + b1
    add_bias(buf_b, W.b1, H1, B);
    __syncthreads();

    // sigmoid in-place on buf_b
    activation_apply_fwd<Sigmoid>(buf_b, H1 * B);
    __syncthreads();

    // Cache h1 output for backward pass
    {
        float* cache_h1 = reinterpret_cast<float*>(cache.data) + IN * B;
        for (int i = threadIdx.x; i < H1 * B; i += blockDim.x) {
            cache_h1[i] = buf_b[i];
        }
    }

    // =================================================================
    // Layer 2:  h2 = sigmoid(W2 @ h1 + b2)
    //   h1 is in buf_b ; copy to buf_a as input for layer 2
    // =================================================================
    // Swap: h1 (buf_b) → buf_a
    for (int i = threadIdx.x; i < H1 * B; i += blockDim.x) {
        buf_a[i] = buf_b[i];
    }
    __syncthreads();

    // W2 @ h1 → buf_b
    warp_gemm_forward<Config>(W.W2, buf_a, buf_b, H2, H1, B);
    __syncthreads();

    // + b2
    add_bias(buf_b, W.b2, H2, B);
    __syncthreads();

    // sigmoid in-place
    activation_apply_fwd<Sigmoid>(buf_b, H2 * B);
    __syncthreads();

    // Cache h2 output for backward pass
    {
        float* cache_h2 = reinterpret_cast<float*>(cache.data) + (IN + H1) * B;
        for (int i = threadIdx.x; i < H2 * B; i += blockDim.x) {
            cache_h2[i] = buf_b[i];
        }
    }

    // =================================================================
    // Layer 3:  y = W3 @ h2 + b3  (linear output, no activation)
    // =================================================================
    // h2 (buf_b) → buf_a
    for (int i = threadIdx.x; i < H2 * B; i += blockDim.x) {
        buf_a[i] = buf_b[i];
    }
    __syncthreads();

    // W3 @ h2 → buf_b
    warp_gemm_forward<Config>(W.W3, buf_a, buf_b, OUT, H2, B);
    __syncthreads();

    // + b3
    add_bias(buf_b, W.b3, OUT, B);
    __syncthreads();

    // Write output to global memory
    {
        float* y_gmem = d_outputs + net_id * (OUT * B);
        for (int i = threadIdx.x; i < OUT * B; i += blockDim.x) {
            y_gmem[i] = buf_b[i];
        }
    }
}

// ===================================================================
// BACKWARD KERNEL
// ===================================================================
template <typename Config>
__global__ void fnn_backward_kernel(
    NetworkWeights<Config>* __restrict__ d_weights,      // [num_networks]  (grad fields updated)
    const float*            __restrict__ d_loss_grad,     // [num_networks][OUT][BATCH]  dL/dy
    float*                  __restrict__ d_grad_inputs,   // [num_networks][IN][BATCH]   dL/dx  (output)
    const ActivationCache<Config>* __restrict__ d_cache)  // [num_networks]  cached activations from forward
{
    constexpr int IN  = Config::IN_DIM;
    constexpr int H1  = Config::H1_DIM;
    constexpr int H2  = Config::H2_DIM;
    constexpr int OUT = Config::OUT_DIM;
    constexpr int B   = Config::BATCH;

    int net_id = blockIdx.x;
    NetworkWeights<Config>& W           = d_weights[net_id];
    const ActivationCache<Config>& cache = d_cache[net_id];

    extern __shared__ float smem[];
    using Smem = SmemLayout<Config>;
    float* buf_a = smem + Smem::BUF_A_OFFSET;  // used for grad of next layer back
    float* buf_b = smem + Smem::BUF_B_OFFSET;  // used for current layer's grad

    // Load cached forward activations from cache
    const float* cache_x  = reinterpret_cast<const float*>(cache.data);
    const float* cache_h1 = reinterpret_cast<const float*>(cache.data) + IN * B;
    const float* cache_h2 = reinterpret_cast<const float*>(cache.data) + (IN + H1) * B;

    // =================================================================
    // Layer 3 backward (linear output, no activation)
    //   GX3[K=H2,N=B] = W3[M=OUT,K=H2]^T @ GY[M=OUT,N=B]
    //   GW3[M=OUT,K=H2] += GY[M,N] @ X3[K,N]^T    (X3 = h2 = cache_h2)
    //   Gb3 += sum over N of GY
    // =================================================================
    // Load loss gradient dL/dy → buf_b
    {
        const float* gy_gmem = d_loss_grad + net_id * (OUT * B);
        for (int i = threadIdx.x; i < OUT * B; i += blockDim.x) {
            buf_b[i] = gy_gmem[i];
        }
    }
    __syncthreads();

    // Load h2 from cache → buf_a  (needed for weight gradient)
    for (int i = threadIdx.x; i < H2 * B; i += blockDim.x) {
        buf_a[i] = cache_h2[i];
    }
    __syncthreads();

    // dW3 += GY @ h2^T
    warp_gemm_backward_weight<Config>(buf_b, buf_a, W.grad_W3, OUT, H2, B);
    // db3 += sum_n GY[m,n]
    accumulate_bias_grad(buf_b, W.grad_b3, OUT, B);
    __syncthreads();

    // G_h2 = W3^T @ GY  (result in buf_a, overwriting h2 which is no longer needed)
    // We compute into buf_a after weight grad uses h2
    warp_gemm_backward_input<Config>(W.W3, buf_b, buf_a, OUT, H2, B);
    __syncthreads();

    // buf_a now holds the upstream gradient for layer 2: G_h2
    // buf_b is free for reuse

    // =================================================================
    // Layer 2 backward (sigmoid)
    //   G_a2_raw = G_h2 * sigmoid'(h2_out)    element-wise
    //   G_h1     = W2^T @ G_a2_raw
    //   GW2      += G_a2_raw @ h1^T
    // =================================================================
    // Apply sigmoid'(h2) to the incoming gradient
    // cache_h2 holds h2 forward output (after sigmoid)
    activation_apply_bwd<Sigmoid>(cache_h2, buf_a, H2 * B);
    __syncthreads();
    // buf_a now = G_a2_raw

    // Load h1 from cache → buf_b  (needed for weight gradient)
    for (int i = threadIdx.x; i < H1 * B; i += blockDim.x) {
        buf_b[i] = cache_h1[i];
    }
    __syncthreads();

    // dW2 += G_a2_raw @ h1^T
    warp_gemm_backward_weight<Config>(buf_a, buf_b, W.grad_W2, H2, H1, B);
    // db2
    accumulate_bias_grad(buf_a, W.grad_b2, H2, B);
    __syncthreads();

    // G_h1 = W2^T @ G_a2_raw  → buf_b (overwrite h1 which was consumed)
    warp_gemm_backward_input<Config>(W.W2, buf_a, buf_b, H2, H1, B);
    __syncthreads();

    // buf_b now = G_h1  (upstream gradient for layer 1)
    // buf_a is free

    // =================================================================
    // Layer 1 backward (sigmoid)
    //   G_x_raw = G_h1 * sigmoid'(h1_out)
    //   G_x     = W1^T @ G_x_raw
    //   GW1     += G_x_raw @ x^T
    // =================================================================
    // Apply sigmoid'(h1) ...
    activation_apply_bwd<Sigmoid>(cache_h1, buf_b, H1 * B);
    __syncthreads();
    // buf_b now = G_x_raw

    // Load x from cache → buf_a  (needed for weight gradient)
    for (int i = threadIdx.x; i < IN * B; i += blockDim.x) {
        buf_a[i] = cache_x[i];
    }
    __syncthreads();

    // dW1 += G_x_raw @ x^T
    warp_gemm_backward_weight<Config>(buf_b, buf_a, W.grad_W1, H1, IN, B);
    // db1
    accumulate_bias_grad(buf_b, W.grad_b1, H1, B);
    __syncthreads();

    // G_x = W1^T @ G_x_raw  → buf_a (overwrite x)
    warp_gemm_backward_input<Config>(W.W1, buf_b, buf_a, H1, IN, B);
    __syncthreads();

    // buf_a now = G_x  (final input gradient)
    // Write to global memory
    {
        float* gx_gmem = d_grad_inputs + net_id * (IN * B);
        for (int i = threadIdx.x; i < IN * B; i += blockDim.x) {
            gx_gmem[i] = buf_a[i];
        }
    }
}

} // namespace fnn
