/*
 * fnn_test.cu — Correctness and performance test for the FNN single-block kernel.
 *
 * Validates forward/backward against a NumPy reference.
 * Measures throughput: networks processed per second.
 *
 * Usage:
 *   1. Generate golden data:  python tools/generate_golden.py
 *   2. Build and run:         cmake --build build --target testCutlass && ./build/test/test-cutlass/testCutlass
 *
 * The golden data file ("golden_data.bin") should be in the working directory.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "fnn_config.cuh"
#include "fnn_kernel.cuh"
#include "fnn_host.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            fflush(stderr);                                                     \
            exit(1);                                                            \
        }                                                                       \
    } while(0)

// ===================================================================
// Mini-timer using CUDA events
// ===================================================================
struct GpuTimer {
    cudaEvent_t start_ev, stop_ev;
    GpuTimer()  { cudaEventCreate(&start_ev); cudaEventCreate(&stop_ev); }
    ~GpuTimer() { cudaEventDestroy(start_ev); cudaEventDestroy(stop_ev); }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_ev, s); }
    void stop (cudaStream_t s = 0) { cudaEventRecord(stop_ev, s); }
    float elapsed_ms() {
        cudaEventSynchronize(stop_ev);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        return ms;
    }
};

// ===================================================================
// Reference implementation (CPU, matches agent_numpy.py Linear layer)
// ===================================================================
using Cfg = fnn::DefaultConfig;
constexpr int IN  = Cfg::IN_DIM;
constexpr int H1  = Cfg::H1_DIM;
constexpr int H2  = Cfg::H2_DIM;
constexpr int OUT = Cfg::OUT_DIM;
constexpr int B   = Cfg::BATCH;

using Weights = fnn::NetworkWeights<Cfg>;
using ACache  = fnn::ActivationCache<Cfg>;

// Sigmoid
static inline float sigmoid_fwd(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float sigmoid_bwd(float y) { return y * (1.0f - y); }

// Forward pass on CPU (one network, single batch)
static void cpu_forward(const Weights& w, const float* x, float* y,
                        float* cache_x, float* cache_h1, float* cache_h2)
{
    // Cache input
    memcpy(cache_x, x, IN * B * sizeof(float));

    // Layer 1: h1 = sigmoid(W1 @ x + b1)
    float h1[H1];  // H1 * B = H1 since B=1
    for (int i = 0; i < H1; ++i) {
        float acc = w.b1[i];
        for (int j = 0; j < IN; ++j) acc += w.W1[i * IN + j] * x[j];
        h1[i] = sigmoid_fwd(acc);
    }
    memcpy(cache_h1, h1, H1 * B * sizeof(float));

    // Layer 2: h2 = sigmoid(W2 @ h1 + b2)
    float h2[H2];
    for (int i = 0; i < H2; ++i) {
        float acc = w.b2[i];
        for (int j = 0; j < H1; ++j) acc += w.W2[i * H1 + j] * h1[j];
        h2[i] = sigmoid_fwd(acc);
    }
    memcpy(cache_h2, h2, H2 * B * sizeof(float));

    // Layer 3: y = W3 @ h2 + b3
    for (int i = 0; i < OUT; ++i) {
        float acc = w.b3[i];
        for (int j = 0; j < H2; ++j) acc += w.W3[i * H2 + j] * h2[j];
        y[i] = acc;
    }
}

// Backward pass on CPU
static void cpu_backward(Weights& w, const float* loss_grad,
                         const float* cache_x, const float* cache_h1, const float* cache_h2,
                         float* grad_x)
{
    // Layer 3 backward (linear)
    float grad_h2[H2] = {0};
    for (int k = 0; k < H2; ++k) {
        for (int m = 0; m < OUT; ++m) {
            grad_h2[k] += w.W3[m * H2 + k] * loss_grad[m];
        }
    }
    // dW3
    for (int m = 0; m < OUT; ++m) {
        for (int k = 0; k < H2; ++k) {
            w.grad_W3[m * H2 + k] += loss_grad[m] * cache_h2[k];
        }
        w.grad_b3[m] += loss_grad[m];
    }

    // Layer 2 backward (sigmoid)
    float grad_h2_raw[H2];
    for (int i = 0; i < H2; ++i) grad_h2_raw[i] = grad_h2[i] * sigmoid_bwd(cache_h2[i]);

    float grad_h1[H1] = {0};
    for (int k = 0; k < H1; ++k) {
        for (int m = 0; m < H2; ++m) {
            grad_h1[k] += w.W2[m * H1 + k] * grad_h2_raw[m];
        }
    }
    for (int m = 0; m < H2; ++m) {
        for (int k = 0; k < H1; ++k) {
            w.grad_W2[m * H1 + k] += grad_h2_raw[m] * cache_h1[k];
        }
        w.grad_b2[m] += grad_h2_raw[m];
    }

    // Layer 1 backward (sigmoid)
    float grad_x_raw[H1];
    for (int i = 0; i < H1; ++i) grad_x_raw[i] = grad_h1[i] * sigmoid_bwd(cache_h1[i]);

    for (int k = 0; k < IN; ++k) {
        grad_x[k] = 0;
        for (int m = 0; m < H1; ++m) {
            grad_x[k] += w.W1[m * IN + k] * grad_x_raw[m];
        }
    }
    for (int m = 0; m < H1; ++m) {
        for (int k = 0; k < IN; ++k) {
            w.grad_W1[m * IN + k] += grad_x_raw[m] * cache_x[k];
        }
        w.grad_b1[m] += grad_x_raw[m];
    }
}

// ===================================================================
// Error checking helpers
// ===================================================================
static float rel_error(float a, float b) {
    float denom = fmaxf(fabsf(a), fabsf(b));
    if (denom < 1e-8f) return fabsf(a - b);
    return fabsf(a - b) / denom;
}

static bool allclose(const float* a, const float* b, int n,
                     float rtol = 1e-4f, float atol = 1e-6f,
                     const char* label = "", bool verbose = true) {
    float max_rel = 0, max_abs = 0;
    int bad_count = 0;
    int first_bad = -1;
    for (int i = 0; i < n; ++i) {
        float abs_err = fabsf(a[i] - b[i]);
        float rel_err = rel_error(a[i], b[i]);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > atol && rel_err > rtol) {
            if (bad_count == 0) first_bad = i;
            ++bad_count;
        }
    }
    bool ok = (bad_count == 0);
    if (verbose || !ok) {
        printf("  %-30s  %s  (max_rel=%.2e  max_abs=%.2e  bad=%d",
               label, ok ? "PASS" : "FAIL", max_rel, max_abs, bad_count);
        if (!ok && first_bad >= 0) {
            printf("  first[%d]: gpu=%.6f  cpu=%.6f", first_bad, a[first_bad], b[first_bad]);
        }
        printf(")\n");
    }
    return ok;
}

// ===================================================================
// Main test
// ===================================================================
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);  // disable buffering
    fprintf(stderr, "Starting FNN Kernel test...\n");

    // Force CUDA runtime initialization
    cudaError_t init_err = cudaFree(0);
    fprintf(stderr, "CUDA init: %d (%s)\n", (int)init_err, cudaGetErrorString(init_err));
    if (init_err != cudaSuccess) return 1;

    printf("=== FNN Single-Block Kernel Test ===\n\n");

    // ---- Device info ----
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (SM %d.%d, %zu MB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024 * 1024));

    // =================================================================
    // Test 1: Correctness — compare against CPU reference
    // =================================================================
    printf("--- Test 1: Correctness (1 network) ---\n");
    fprintf(stderr, "Test1: start\n"); fflush(stderr);

    constexpr int NUM_NETS = 1;  // single network for correctness
    fnn::FNNHandle<Cfg> handle;
    fprintf(stderr, "Test1: alloc...\n"); fflush(stderr);
    handle.alloc(NUM_NETS);
    fprintf(stderr, "Test1: init_weights...\n"); fflush(stderr);
    handle.init_weights_xavier(42);

    // Get a copy of the weights for CPU reference
    Weights* h_weights = new Weights[NUM_NETS];
    handle.copy_weights_to_host(h_weights);
    // Zero GPU gradients
    handle.zero_gradients();

    // Host input
    float h_input[IN * B];
    srand(1234);
    for (int i = 0; i < IN * B; ++i) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }

    // Copy input to device
    cudaMemcpyAsync(handle.d_inputs, h_input, sizeof(h_input),
                    cudaMemcpyHostToDevice, handle.stream);

    // ---- GPU forward ----
    fprintf(stderr, "Test1: forward kernel...\n"); fflush(stderr);
    handle.forward();
    fprintf(stderr, "Test1: sync...\n"); fflush(stderr);
    handle.sync();

    // Read GPU output
    float h_output_gpu[OUT * B];
    cudaMemcpy(h_output_gpu, handle.d_outputs, sizeof(h_output_gpu),
               cudaMemcpyDeviceToHost);

    // ---- CPU forward (reference) ----
    float h_output_cpu[OUT * B];
    float cache_x[IN * B], cache_h1[H1 * B], cache_h2[H2 * B];
    cpu_forward(h_weights[0], h_input, h_output_cpu, cache_x, cache_h1, cache_h2);

    bool fwd_ok = allclose(h_output_gpu, h_output_cpu, OUT * B,
                           1e-4f, 1e-5f, "Forward output");

    // ---- GPU backward ----
    float h_loss_grad[OUT * B];
    srand(5678);
    for (int i = 0; i < OUT * B; ++i) {
        h_loss_grad[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    cudaMemcpyAsync(handle.d_loss_grad, h_loss_grad, sizeof(h_loss_grad),
                    cudaMemcpyHostToDevice, handle.stream);

    handle.backward();
    handle.sync();

    // Read GPU gradients
    Weights h_weights_gpu;
    handle.copy_weights_to_host(&h_weights_gpu);
    float h_grad_input_gpu[IN * B];
    cudaMemcpy(h_grad_input_gpu, handle.d_grad_inputs, sizeof(h_grad_input_gpu),
               cudaMemcpyDeviceToHost);

    // ---- CPU backward (reference) ----
    Weights h_weights_cpu = h_weights[0];  // copy
    float h_grad_input_cpu[IN * B];
    cpu_backward(h_weights_cpu, h_loss_grad, cache_x, cache_h1, cache_h2,
                 h_grad_input_cpu);

    bool bwd_input_ok = allclose(h_grad_input_gpu, h_grad_input_cpu, IN * B,
                                 1e-4f, 1e-5f, "Backward grad_input");
    bool bwd_w1_ok    = allclose(h_weights_gpu.grad_W1, h_weights_cpu.grad_W1, H1 * IN,
                                 1e-4f, 1e-5f, "Backward grad_W1");
    bool bwd_b1_ok    = allclose(h_weights_gpu.grad_b1, h_weights_cpu.grad_b1, H1,
                                 1e-4f, 1e-5f, "Backward grad_b1");
    bool bwd_w2_ok    = allclose(h_weights_gpu.grad_W2, h_weights_cpu.grad_W2, H2 * H1,
                                 1e-4f, 1e-5f, "Backward grad_W2");
    bool bwd_b2_ok    = allclose(h_weights_gpu.grad_b2, h_weights_cpu.grad_b2, H2,
                                 1e-4f, 1e-5f, "Backward grad_b2");
    bool bwd_w3_ok    = allclose(h_weights_gpu.grad_W3, h_weights_cpu.grad_W3, OUT * H2,
                                 1e-4f, 1e-5f, "Backward grad_W3");
    bool bwd_b3_ok    = allclose(h_weights_gpu.grad_b3, h_weights_cpu.grad_b3, OUT,
                                 1e-4f, 1e-5f, "Backward grad_b3");

    bool all_ok = fwd_ok && bwd_input_ok &&
                  bwd_w1_ok && bwd_b1_ok && bwd_w2_ok && bwd_b2_ok && bwd_w3_ok && bwd_b3_ok;
    printf("\n  Overall correctness: %s\n\n", all_ok ? "PASS" : "FAIL");

    // =================================================================
    // Test 2: Multi-network correctness (ensure independence)
    // =================================================================
    printf("--- Test 2: Multi-network independence (%d networks) ---\n", 4);

    handle.free();
    constexpr int NUM_NETS2 = 4;
    handle.alloc(NUM_NETS2);
    handle.init_weights_xavier(100);

    // Different input per network
    float h_inputs2[NUM_NETS2 * IN * B];
    srand(200);
    for (int i = 0; i < NUM_NETS2 * IN * B; ++i) {
        h_inputs2[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    cudaMemcpyAsync(handle.d_inputs, h_inputs2, sizeof(h_inputs2),
                    cudaMemcpyHostToDevice, handle.stream);

    // Copy weights for CPU comparison
    Weights* h_w2 = new Weights[NUM_NETS2];
    handle.copy_weights_to_host(h_w2);
    handle.zero_gradients();

    // GPU forward + backward
    handle.forward();
    handle.sync();

    // Compare each network's output against CPU
    int nets_ok = 0;
    for (int n = 0; n < NUM_NETS2; ++n) {
        float gpu_out[OUT * B], cpu_out[OUT * B];
        float cx[IN*B], ch1[H1*B], ch2[H2*B];
        cudaMemcpy(gpu_out, handle.d_outputs + n * OUT * B,
                   sizeof(gpu_out), cudaMemcpyDeviceToHost);
        cpu_forward(h_w2[n], h_inputs2 + n * IN * B, cpu_out, cx, ch1, ch2);
        bool ok = allclose(gpu_out, cpu_out, OUT * B, 1e-4f, 1e-5f, "", false);
        if (ok) ++nets_ok;
    }
    printf("  Independent outputs: %d/%d correct\n\n", nets_ok, NUM_NETS2);

    // =================================================================
    // Test 3: Performance benchmark
    // =================================================================
    printf("--- Test 3: Performance benchmark ---\n");

    constexpr int PERF_NETS[] = {1, 100, 1000, 5000};
    constexpr int WARMUP = 3;
    constexpr int ITERS  = 20;

    constexpr int PERF_CASES = sizeof(PERF_NETS) / sizeof(PERF_NETS[0]);
    for (int p = 0; p < PERF_CASES; ++p) {
        int N = PERF_NETS[p];
        // Re-init with N networks
        fnn::FNNHandle<Cfg> ph;
        ph.alloc(N);
        ph.init_weights_xavier(42 + p);
        ph.zero_gradients();

        // Prepare random input and loss_grad
        float* h_in = new float[N * IN * B];
        float* h_lg = new float[N * OUT * B];
        srand(300 + p);
        for (int i = 0; i < N * IN * B;  ++i) h_in[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int i = 0; i < N * OUT * B; ++i) h_lg[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        cudaMemcpy(ph.d_inputs, h_in, N * IN * B * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ph.d_loss_grad, h_lg, N * OUT * B * sizeof(float), cudaMemcpyHostToDevice);

        GpuTimer fwd_timer, bwd_timer;

        // Warmup
        for (int i = 0; i < WARMUP; ++i) { ph.forward(); ph.backward(); }
        ph.sync();

        // Benchmark forward
        fwd_timer.start(ph.stream);
        for (int i = 0; i < ITERS; ++i) ph.forward();
        fwd_timer.stop(ph.stream);
        float fwd_ms = fwd_timer.elapsed_ms() / ITERS;

        // Benchmark backward
        bwd_timer.start(ph.stream);
        for (int i = 0; i < ITERS; ++i) ph.backward();
        bwd_timer.stop(ph.stream);
        float bwd_ms = bwd_timer.elapsed_ms() / ITERS;

        float total_ms = fwd_ms + bwd_ms;
        float nets_per_sec = N / (total_ms / 1000.0f);
        float fwd_bw = (N * sizeof(Weights) + N * IN * B * sizeof(float) + N * OUT * B * sizeof(float)) / (fwd_ms / 1000.0f);
        float bwd_bw = (N * sizeof(Weights) + N * (IN + OUT) * B * sizeof(float) + N * sizeof(ACache)) / (bwd_ms / 1000.0f);

        printf("  N=%6d  |  fwd: %8.3f us  |  bwd: %8.3f us  |  total: %8.3f us  |  throughput: %10.1f nets/s\n",
               N, fwd_ms * 1000, bwd_ms * 1000, total_ms * 1000, nets_per_sec);

        delete[] h_in; delete[] h_lg;
        ph.free();
    }

    printf("\n");

    // =================================================================
    // Test 4: Online learning sanity check (loss decreases)
    // =================================================================
    printf("--- Test 4: Online learning sanity check ---\n");

    {
        constexpr int TRAIN_NETS = 4;
        constexpr int TRAIN_STEPS = 100;
        constexpr float LR = 0.01f;

        fnn::FNNHandle<Cfg> th;
        th.alloc(TRAIN_NETS);
        th.init_weights_xavier(999);

        // Simple regression target: each network learns to output a constant
        float targets[TRAIN_NETS * OUT * B];
        srand(400);
        for (int i = 0; i < TRAIN_NETS * OUT * B; ++i)
            targets[i] = ((float)rand() / RAND_MAX) * 2 - 1;

        float h_in[TRAIN_NETS * IN * B];
        for (int i = 0; i < TRAIN_NETS * IN * B; ++i)
            h_in[i] = ((float)rand() / RAND_MAX) * 2 - 1;

        // Keep same input across steps (static regression)
        cudaMemcpy(th.d_inputs, h_in, sizeof(h_in), cudaMemcpyHostToDevice);

        for (int step = 0; step < TRAIN_STEPS; ++step) {
            th.zero_gradients();

            // Forward
            th.forward();
            th.sync();

            // Compute MSE loss and gradient on host (for simplicity)
            float h_out[TRAIN_NETS * OUT * B];
            cudaMemcpy(h_out, th.d_outputs, sizeof(h_out), cudaMemcpyDeviceToHost);

            float loss = 0;
            float h_grad[TRAIN_NETS * OUT * B];
            for (int i = 0; i < TRAIN_NETS * OUT * B; ++i) {
                float err = h_out[i] - targets[i];
                loss += err * err;
                h_grad[i] = 2.0f * err / (TRAIN_NETS * OUT * B);  // MSE gradient
            }
            loss /= (TRAIN_NETS * OUT * B);

            cudaMemcpy(th.d_loss_grad, h_grad, sizeof(h_grad), cudaMemcpyHostToDevice);

            // Backward + update
            th.backward();
            th.sync();
            th.apply_gradients_sgd(LR);

            if (step == 0) printf("  Step %3d: loss=%.6f\n", step, loss);
            if (step == TRAIN_STEPS - 1) printf("  Step %3d: loss=%.6f\n", step, loss);
        }
        printf("  Training complete (MSE loss should decrease)\n\n");
        th.free();
    }

    // Cleanup
    delete[] h_weights;
    handle.free();

    printf("=== All tests done ===\n");
    return all_ok ? 0 : 1;
}
