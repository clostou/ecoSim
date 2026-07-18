/*
 * fnn_activations.cuh — Activation functions for the feedforward neural network.
 *
 * Each activation provides a forward device function (apply in-place)
 * and a backward device function (derivative multiplied by upstream gradient).
 */

#pragma once

#include <cuda_runtime.h>

namespace fnn {

// ===================================================================
// Sigmoid
// ===================================================================
struct Sigmoid {
    /// y = 1 / (1 + exp(-x))
    static __device__ __forceinline__ float fwd(float x) {
        return 1.0f / (1.0f + __expf(-x));
    }

    /// dy = y * (1 - y) * grad   (y is the forward output)
    static __device__ __forceinline__ float bwd(float y, float grad) {
        return y * (1.0f - y) * grad;
    }
};

// ===================================================================
// Tanh
// ===================================================================
struct Tanh {
    /// y = tanh(x)
    static __device__ __forceinline__ float fwd(float x) {
        return tanhf(x);
    }

    /// dy = (1 - y*y) * grad
    static __device__ __forceinline__ float bwd(float y, float grad) {
        return (1.0f - y * y) * grad;
    }
};

// ===================================================================
// ReLU
// ===================================================================
struct ReLU {
    /// y = max(0, x)
    static __device__ __forceinline__ float fwd(float x) {
        return (x > 0.0f) ? x : 0.0f;
    }

    /// dy = (x > 0 ? 1 : 0) * grad
    static __device__ __forceinline__ float bwd(float y, float grad) {
        return (y > 0.0f) ? grad : 0.0f;
    }
};

// ===================================================================
// Identity (linear)
// ===================================================================
struct Identity {
    static __device__ __forceinline__ float fwd(float x) { return x; }
    static __device__ __forceinline__ float bwd(float, float grad) { return grad; }
};

// ===================================================================
// Cooperative apply — all threads in block process a contiguous array
// ===================================================================

/// Apply activation forward on an array of `count` elements in shared memory.
/// Each thread processes `count / blockDim.x` elements (round up).
template <typename ActFn>
__device__ void activation_apply_fwd(float* data, int count) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < count; i += stride) {
        data[i] = ActFn::fwd(data[i]);
    }
}

/// Apply activation backward (multiply derivative by upstream gradient).
/// `out` contains the forward activation values; `grad` contains the upstream
/// gradient. On exit, `grad` holds grad * act'(out).
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

// ===================================================================
// Add bias to an [M, N] matrix stored in shared memory.
// `mat` is row-major: mat[r * N + c].
// `bias` is in global memory, length M.
// ===================================================================
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

// ===================================================================
// Accumulate bias gradient: gb[r] += sum_c grad_mat[r * N + c]
// grad_mat is in shared memory, gb is in global memory.
// ===================================================================
__device__ void accumulate_bias_grad(const float* __restrict__ grad_mat,
                                     float* __restrict__ gb,
                                     int M, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // Each thread reduces one or more rows (when M < blockDim)
    for (int r = tid; r < M; r += stride) {
        float sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            sum += grad_mat[r * N + c];
        }
        // Plain add — each row is processed by exactly one thread
        gb[r] += sum;
    }
}

} // namespace fnn
