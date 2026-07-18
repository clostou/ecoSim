/*
 * fnn_config.cuh — Compile-time configuration for the feedforward neural network.
 *
 * All network dimensions are template parameters so the compiler can
 * optimize inner loops and allocate register/shared memory precisely.
 */

#pragma once

namespace fnn {

// ---------------------------------------------------------------------------
// FNNConfig — template struct holding all network hyperparameters
// ---------------------------------------------------------------------------
template <
    int IN_DIM_      = 32,   ///< Input feature dimension
    int H1_DIM_      = 64,   ///< Hidden layer 1 dimension
    int H2_DIM_      = 64,   ///< Hidden layer 2 dimension
    int OUT_DIM_     = 4,    ///< Output dimension
    int BATCH_       = 1,    ///< Mini-batch size (1 = online SGD)
    int BLOCK_DIM_   = 128   ///< Threads per block (= 4 warps)
>
struct FNNConfig {
    // -- dimensions --
    static constexpr int IN_DIM      = IN_DIM_;
    static constexpr int H1_DIM      = H1_DIM_;
    static constexpr int H2_DIM      = H2_DIM_;
    static constexpr int OUT_DIM     = OUT_DIM_;
    static constexpr int BATCH       = BATCH_;

    // -- execution --
    static constexpr int BLOCK_DIM   = BLOCK_DIM_;
    static constexpr int WARPS       = BLOCK_DIM / 32;
    static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM must be a multiple of 32");

    // -- derived: rows per warp for each layer's output (M dimension) --
    static constexpr int H1_ROWS_PER_WARP = (H1_DIM + WARPS - 1) / WARPS;
    static constexpr int H2_ROWS_PER_WARP = (H2_DIM + WARPS - 1) / WARPS;
    static constexpr int OUT_ROWS_PER_WARP = (OUT_DIM + WARPS - 1) / WARPS;

    // -- derived: shared memory budget for the largest simultaneous buffer pair --
    //    At most we need [X_smem + C_smem] where X is the input to a layer
    //    and C is its output. Largest combo is H2→OUT: (H2 + OUT) * BATCH.
    static constexpr int MAX_SMEM_FLOATS =
        ((H1_DIM > IN_DIM ? H1_DIM : IN_DIM) +
         (H2_DIM > H1_DIM ? H2_DIM : H1_DIM)) * BATCH;
    // Pad by WARP_SIZE to mitigate bank conflicts
    static constexpr int SMEM_PAD      = 32;
    static constexpr int SMEM_CAPACITY = MAX_SMEM_FLOATS + SMEM_PAD;
};

// ---------------------------------------------------------------------------
// Default configuration used by the test harness
// ---------------------------------------------------------------------------
using DefaultConfig = FNNConfig<32, 64, 64, 4, 1, 128>;

// ---------------------------------------------------------------------------
// Phase tag for kernel dispatch
// ---------------------------------------------------------------------------
enum class Phase : int {
    FORWARD  = 0,
    BACKWARD = 1
};

// ---------------------------------------------------------------------------
// Activation type
// ---------------------------------------------------------------------------
enum class Activation : int {
    IDENTITY = 0,
    SIGMOID  = 1,
    TANH     = 2,
    RELU     = 3
};

} // namespace fnn
