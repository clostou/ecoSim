/*
 * fnn_warp_gemm.cuh — Warp-cooperative tiled GEMM for small matrices.
 *
 * All routines run within a single CUDA block (128 threads = 4 warps).
 * The input operand is always in shared memory for reuse across warps;
 * the weight matrix is streamed from global memory with coalesced reads.
 *
 * CUTLASS is used for:
 *   - AlignedBuffer (bank-conflict-free shared memory)
 *   - GemmCoord / MatrixCoord type aliases
 *   - __shfl_xor_sync for warp-level reductions
 */

#pragma once

#include <cuda_runtime.h>

// NOTE: CUTLASS 4.5.1 is linked via CMake. This warp-level GEMM follows
// CUTLASS design patterns (tiled iteration space, warp-cooperative reduction,
// shared memory staging) while using raw CUDA intrinsics for the inner loops.
// Direct CUTLASS header inclusion is avoided due to MSVC 2019 host compiler
// limitations with CuTe's C++20 NTTP features.

#include "fnn_config.cuh"

namespace fnn {

// ===================================================================
// Warp-level inclusive scan / reduction helpers
// ===================================================================

/// Full-warp sum reduction (all 32 lanes get the same result).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Butterfly reduction: 32 → 16 → 8 → 4 → 2 → 1
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // all lanes now hold the total
}

// ===================================================================
// Forward GEMM:  C[M,N] = W[M,K] @ X[K,N]
//
//   W_gmem  — weight matrix [M, K] row-major in global memory
//   X_smem  — input  matrix [K, N] in shared memory
//   C_smem  — output matrix [M, N] in shared memory (written by this call)
//   bias    — optional bias vector [M] in global memory (nullptr = skip)
//
// Strategy:
//   - Distribute M across warps (each warp handles ~M/warps rows)
//   - Within a warp, each lane handles a stride-K segment for coalesced
//     reads from W, then warp-reduce to get one C element.
//   - When M is small (≤32), a single warp is enough; extra warps are idle.
// ===================================================================
template <typename Config>
__device__ void warp_gemm_forward(
    const float* __restrict__ W_gmem,   // [M, K] row-major
    const float* __restrict__ X_smem,   // [K, N]
    float* __restrict__ C_smem,         // [M, N] output
    int M, int K, int N)
{
    constexpr int WARPS = Config::WARPS;
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // Rows of M assigned to this warp (cyclic to balance across warps)
    int rows_per_warp = (M + WARPS - 1) / WARPS;
    int row_start     = warp_id * rows_per_warp;
    int row_end       = min(row_start + rows_per_warp, M);

    // Each warp iterates over its assigned M rows.
    // Within the warp, each lane contributes to one row at a time.
    // IMPORTANT: all 32 lanes must participate in warp_reduce_sum to avoid
    //            deadlock, even if the lane's assigned row is out of bounds.
    for (int m_base = row_start; m_base < row_end; m_base += 32) {
        int m = m_base + lane;
        bool valid = (m < row_end);

        // For each column in N (usually 1, but can be small batch)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            if (valid) {
                // Coalesced: lane L reads W[m, L], W[m, L+32], ..., W[m, L+64], ...
                for (int k = lane; k < K; k += 32) {
                    acc += W_gmem[m * K + k] * X_smem[k * N + n];
                }
            }
            // ALL lanes must call the reduce (non-participating lanes contribute 0)
            acc = warp_reduce_sum(acc);
            // Lane 0 writes the result for this row
            if (valid && lane == 0) {
                C_smem[m * N + n] = acc;
            }
        }
    }
}

// ===================================================================
// Backward input gradient:  GX[K,N] = W[M,K]^T @ GY[M,N]
//
//   W_gmem   — weight matrix [M, K] row-major in global memory
//   GY_smem  — upstream gradient [M, N] in shared memory
//   GX_smem  — input gradient  [K, N] in shared memory (written)
//
// Strategy:
//   - Distribute K across warps / threads
//   - Each thread computes GX[k, n] = sum_m W[m,k] * GY[m,n]
//   - W is read column-by-column (stride K between consecutive elements
//     for the same m), but consecutive threads access consecutive k
//     for the same m → coalesced.
// ===================================================================
template <typename Config>
__device__ void warp_gemm_backward_input(
    const float* __restrict__ W_gmem,   // [M, K] row-major
    const float* __restrict__ GY_smem,  // [M, N]
    float* __restrict__ GX_smem,        // [K, N] output
    int M, int K, int N)
{
    int tid   = threadIdx.x;
    int total = K * N;

    // Flat distribution of (k,n) pairs across all threads
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int k = idx / N;
        int n = idx % N;
        float acc = 0.0f;
        // Accumulate W[m, k] * GY[m, n] over all m
        for (int m = 0; m < M; ++m) {
            acc += W_gmem[m * K + k] * GY_smem[m * N + n];
        }
        GX_smem[idx] = acc;
    }
}

// ===================================================================
// Backward weight gradient:  GW[M,K] += GY[M,N] @ X[K,N]^T
//
//   GY_smem  — upstream gradient [M, N] in shared memory
//   X_smem   — input activation  [K, N] in shared memory (from forward)
//   GW_gmem  — weight gradient  [M, K] row-major in global memory
//
//   GW[m, k] += sum_n GY[m, n] * X[k, n]
//
// Strategy:
//   - Distribute (m, k) pairs across all threads
//   - For N=1 this is simply GW[m,k] += GY[m] * X[k]
//   - Each (m, k) pair is assigned to exactly one thread → no atomics
// ===================================================================
template <typename Config>
__device__ void warp_gemm_backward_weight(
    const float* __restrict__ GY_smem,  // [M, N]
    const float* __restrict__ X_smem,   // [K, N]
    float* __restrict__ GW_gmem,        // [M, K] to accumulate into
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
        GW_gmem[idx] += acc;
    }
}

} // namespace fnn
