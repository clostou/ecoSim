/**
 * attnlstm_bench.cu — AttnLSTM 性能基准测试
 *
 * 测量不同网络数量下的前向/反向吞吐量和延迟。
 *
 * Usage:
 *   cmake --build build --config Release --target testAttnLstmBench
 *   ./build/test/test-attnlstm/Release/testAttnLstmBench
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "net_config.cuh"
#include "net_host.cuh"

using Cfg = agent_gpu::DefaultConfig;

constexpr int WARMUP_ITERS = 10;
constexpr int TIMED_ITERS  = 50;

// 参数总量：权重 4259 + 梯度 4259 = 8518 floats = ~34KB
constexpr int TOTAL_PARAM_FLOATS = sizeof(agent_gpu::AttnLstmWeights<Cfg>) / sizeof(float);
constexpr int PARAM_BYTES        = TOTAL_PARAM_FLOATS * sizeof(float);

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); \
    } } while(0)

// ================================================================
// 单次测量：分配 → 初始化 → 预热 → 计时
// ================================================================
struct BenchResult {
    float fwd_us_per_net;
    float bwd_us_per_net;
    float fwd_nets_per_sec;
    float bwd_nets_per_sec;
    float fwd_bw_gb_s;  // effective bandwidth
    float bwd_bw_gb_s;
    float combined_us_per_net;
};

BenchResult measure(int num_nets) {
    BenchResult r = {0};

    agent_gpu::AttnLstmHandle<Cfg> handle;
    handle.alloc(num_nets, 1);
    handle.init_weights_xavier(42);

    // 随机输入
    int obs_sz  = num_nets * Cfg::OBS_DIM * Cfg::OBS_N;
    int inn_sz  = num_nets * Cfg::INNER_DIM;
    int st_sz   = num_nets * Cfg::LSTM_HIDDEN_DIM;
    int act_sz  = num_nets * Cfg::ACT_DIM;

    float* h_obs  = new float[obs_sz];
    float* h_inn  = new float[inn_sz];
    float* h_hp   = new float[st_sz];
    float* h_cp   = new float[st_sz];
    float* h_gact = new float[act_sz];
    float* h_gval = new float[num_nets];

    srand(1234);
    for (int i = 0; i < obs_sz; i++)  h_obs[i]  = (float)rand() / RAND_MAX * 0.5f;
    for (int i = 0; i < inn_sz; i++)  h_inn[i]  = (float)rand() / RAND_MAX * 0.5f;
    for (int i = 0; i < st_sz;  i++) { h_hp[i] = 0.0f; h_cp[i] = 0.0f; }
    for (int i = 0; i < act_sz; i++)  h_gact[i] = 0.5f;
    for (int i = 0; i < num_nets; i++) h_gval[i] = 0.5f;

    CUDA_CHECK(cudaMemcpy(handle.d_observe,    h_obs,  obs_sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(handle.d_inner,      h_inn,  inn_sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(handle.d_grad_act,   h_gact, act_sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(handle.d_grad_value, h_gval, num_nets * sizeof(float), cudaMemcpyHostToDevice));

    agent_gpu::AttnLstmPersistent<Cfg>* h_p =
        new agent_gpu::AttnLstmPersistent<Cfg>[num_nets];
    memset(h_p, 0, num_nets * sizeof(agent_gpu::AttnLstmPersistent<Cfg>));
    handle.copy_persistent_from_host(h_p);

    // ---- Forward benchmark ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        handle.forward(0);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_ITERS; i++) {
        handle.forward(0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fwd_ms;
    cudaEventElapsedTime(&fwd_ms, start, stop);

    r.fwd_us_per_net   = (fwd_ms / TIMED_ITERS) * 1000.0f / num_nets;
    r.fwd_nets_per_sec = (num_nets * TIMED_ITERS) / (fwd_ms / 1000.0f);
    // BW: 读取权重(34KB) + 读取输入 + 写输出 + 写cache(~7KB) ≈ 50KB/network forward
    r.fwd_bw_gb_s = (r.fwd_nets_per_sec * 50.0f * 1024.0f) / 1e9f;

    // ---- Backward benchmark ----
    // 先做一次前向填充缓存
    handle.forward(0);
    cudaDeviceSynchronize();
    handle.zero_gradients();

    for (int i = 0; i < WARMUP_ITERS; i++) {
        handle.backward(1);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_ITERS; i++) {
        handle.backward(1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bwd_ms;
    cudaEventElapsedTime(&bwd_ms, start, stop);

    r.bwd_us_per_net   = (bwd_ms / TIMED_ITERS) * 1000.0f / num_nets;
    r.bwd_nets_per_sec = (num_nets * TIMED_ITERS) / (bwd_ms / 1000.0f);
    // BW: 读权重+梯度(~68KB) + 读cache(~7KB) + 写梯度(~34KB) ≈ 110KB
    r.bwd_bw_gb_s = (r.bwd_nets_per_sec * 110.0f * 1024.0f) / 1e9f;

    r.combined_us_per_net = r.fwd_us_per_net + r.bwd_us_per_net;

    cudaEventDestroy(start); cudaEventDestroy(stop);
    delete[] h_obs; delete[] h_inn; delete[] h_hp; delete[] h_cp;
    delete[] h_gact; delete[] h_gval; delete[] h_p;
    handle.free();

    return r;
}

// ================================================================
// Main
// ================================================================
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    CUDA_CHECK(cudaFree(0));

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("GPU: %s\n", prop.name);
    printf("SMs: %d, Max blocks/SM: %d, Max warps/SM: %d\n",
           prop.multiProcessorCount,
           prop.maxBlocksPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / 32);
    printf("Smem/block: %zu KB, used: %.1f KB\n",
           prop.sharedMemPerBlock / 1024,
           agent_gpu::SmemLayout<Cfg>::TOTAL * sizeof(float) / 1024.0f);
    printf("Params: %d weights + %d grads = %d floats (%.1f KB/network)\n",
           4259, 4259, TOTAL_PARAM_FLOATS, PARAM_BYTES / 1024.0f);
    printf("Warmup: %d, Timed: %d\n\n", WARMUP_ITERS, TIMED_ITERS);

    // 测试不同网络数量
    int net_counts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_tests = sizeof(net_counts) / sizeof(net_counts[0]);

    printf("%6s  %10s  %10s  %12s  %12s  %8s  %8s\n",
           "Nets", "Fwd μs", "Bwd μs", "Fwd net/s", "Bwd net/s",
           "Fwd BW", "Bwd BW");
    printf("%6s  %10s  %10s  %12s  %12s  %8s  %8s\n",
           "------", "------", "------", "--------", "--------", "------", "------");

    for (int i = 0; i < num_tests; i++) {
        int n = net_counts[i];
        BenchResult r = measure(n);

        printf("%6d  %8.1f   %8.1f   %10.0f   %10.0f   %5.1f   %5.1f\n",
               n,
               r.fwd_us_per_net,
               r.bwd_us_per_net,
               r.fwd_nets_per_sec,
               r.bwd_nets_per_sec,
               r.fwd_bw_gb_s,
               r.bwd_bw_gb_s);
    }

    printf("\n=== Summary ===\n");
    // 找到最佳吞吐量的网络数
    printf("(Peak throughput typically at ~SM-count × max-blocks-per-SM networks)\n");
    printf("For A2C with 10 forward + 1 backward per env step:\n");
    printf("  per-env-step = 10 × fwd + 1 × bwd\n\n");

    printf("Done.\n");

    // ================================================================
    // A2C workload simulation: 10 forward + 1 backward per cycle
    // ================================================================
    printf("\n========================================\n");
    printf("  A2C Workload Simulation\n");
    printf("  Pattern: 10× forward → advance_state() → ... → 1× backward\n");
    printf("========================================\n\n");

    int a2c_nets[]   = {1024, 2048, 4096, 8192};
    int a2c_fwd_total = 1000;   // total forward calls
    int a2c_steps     = 10;     // forward steps per backward

    printf("%6s  %7s  %7s  %10s  %10s  %15s\n",
           "Nets", "FwdIters", "BwdIters", "Total ms", "μs/fwd/net", "μs/fwd+bwd/net");
    printf("%6s  %7s  %7s  %10s  %10s  %15s\n",
           "------", "-------", "-------", "--------", "----------", "---------------");

    for (int test_i = 0; test_i < 4; test_i++) {
        int N = a2c_nets[test_i];
        int bwd_total = a2c_fwd_total / a2c_steps;  // 100 backward calls
        int cycles = bwd_total;  // 100 cycles of (10×fwd + 1×bwd)

        // Allocate with enough caches for a2c_steps
        agent_gpu::AttnLstmHandle<Cfg> a2c_handle;
        a2c_handle.alloc(N, a2c_steps);
        a2c_handle.init_weights_xavier(42);

        // Random inputs
        int obs_sz = N * Cfg::OBS_DIM * Cfg::OBS_N;
        int inn_sz = N * Cfg::INNER_DIM;
        int st_sz  = N * Cfg::LSTM_HIDDEN_DIM;
        float* h_obs  = new float[obs_sz];
        float* h_inn  = new float[inn_sz];
        float* h_hp   = new float[st_sz];
        float* h_cp   = new float[st_sz];
        float* h_gact = new float[N * Cfg::ACT_DIM];
        float* h_gval = new float[N];

        srand(1234);
        for (int i = 0; i < obs_sz; i++) h_obs[i] = (float)rand() / RAND_MAX * 0.5f;
        for (int i = 0; i < inn_sz; i++) h_inn[i] = (float)rand() / RAND_MAX * 0.5f;
        for (int i = 0; i < st_sz;  i++) { h_hp[i] = 0.0f; h_cp[i] = 0.0f; }
        for (int i = 0; i < N * Cfg::ACT_DIM; i++) h_gact[i] = 0.5f;
        for (int i = 0; i < N; i++) h_gval[i] = 0.5f;

        CUDA_CHECK(cudaMemcpy(a2c_handle.d_observe,    h_obs,  obs_sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(a2c_handle.d_inner,      h_inn,  inn_sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(a2c_handle.d_grad_act,   h_gact, N * Cfg::ACT_DIM * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(a2c_handle.d_grad_value, h_gval, N * sizeof(float), cudaMemcpyHostToDevice));

        agent_gpu::AttnLstmPersistent<Cfg>* h_persist =
            new agent_gpu::AttnLstmPersistent<Cfg>[N];
        memset(h_persist, 0, N * sizeof(agent_gpu::AttnLstmPersistent<Cfg>));
        a2c_handle.copy_persistent_from_host(h_persist);

        // Warmup: 2 cycles
        for (int w = 0; w < 2; w++) {
            for (int t = 0; t < a2c_steps; t++) {
                a2c_handle.forward(t);
                if (t < a2c_steps - 1) a2c_handle.advance_state();
            }
            a2c_handle.zero_gradients();
            a2c_handle.backward(a2c_steps);
            a2c_handle.sync();
            // Reset state for next cycle
            memset(h_persist, 0, N * sizeof(agent_gpu::AttnLstmPersistent<Cfg>));
            a2c_handle.copy_persistent_from_host(h_persist);
        }

        // Timed
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);

        for (int cycle = 0; cycle < cycles; cycle++) {
            // 10 forward steps
            for (int t = 0; t < a2c_steps; t++) {
                a2c_handle.forward(t);
                if (t < a2c_steps - 1) a2c_handle.advance_state();
            }
            // 1 backward
            a2c_handle.zero_gradients();
            a2c_handle.backward(a2c_steps);
            a2c_handle.sync();
            // Reset state for next cycle
            memset(h_persist, 0, N * sizeof(agent_gpu::AttnLstmPersistent<Cfg>));
            a2c_handle.copy_persistent_from_host(h_persist);
        }

        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float total_ms;
        cudaEventElapsedTime(&total_ms, s, e);

        float us_per_fwd_net  = (total_ms / a2c_fwd_total) * 1000.0f / N;
        float us_per_cycle    = (total_ms / cycles) * 1000.0f / N;  // per (10fwd+1bwd) per net

        printf("%6d  %7d  %7d  %10.1f  %10.2f  %15.2f\n",
               N, a2c_fwd_total, bwd_total, total_ms, us_per_fwd_net, us_per_cycle);

        cudaEventDestroy(s); cudaEventDestroy(e);
        delete[] h_obs; delete[] h_inn; delete[] h_hp; delete[] h_cp;
        delete[] h_gact; delete[] h_gval; delete[] h_persist;
        a2c_handle.free();
    }

    printf("\n=== A2C Notes ===\n");
    printf("Each cycle = %d× forward (state advancing) + 1× backward (BPTT)\n", a2c_steps);
    printf("Total: %d forward + %d backward calls\n", a2c_fwd_total, a2c_fwd_total / a2c_steps);
    printf("μs/cycle/net = time for one agent to complete %d fwd + 1 bwd\n", a2c_steps);
    printf("For 10,000 agents × 1000 env steps: ~%.0f ms\n",
           (10.0f) * 1000.0f);  // rough placeholder

    printf("\nDone.\n");
    return 0;
}
