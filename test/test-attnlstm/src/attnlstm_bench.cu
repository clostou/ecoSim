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

constexpr int WARMUP_ITERS = 10;     // 预热迭代数
constexpr int TIMED_ITERS  = 50;    // 测试迭代数
constexpr int A2C_STEPS    = 10;    // (10×fwd + 1×bwd) per cycles

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); \
    } } while(0)

struct BenchResult {
    float fwd_us_per_net;
    float bwd_us_per_net;
    float combined_us_per_net;      // 10 forward + 1 backward cycle latency
    float cycle_per_s;              // cycle throughput
    float bw_gb_s;                  // effective bandwidth
};

/// 单次测量：分配 → 初始化 → 预热 → 计时
BenchResult measure(int num_nets) {
    BenchResult r = {0};

    // ================================================================
    // Network benchmark: forward + backward in different network counts
    // ================================================================
    agent_gpu::AttnLstmHandle<Cfg> handle;
    handle.alloc(num_nets);
    handle.init_weights_xavier(42);

    // Random inputs and gradients
    int obs_sz  = num_nets * Cfg::OBS_DIM * Cfg::OBS_N;
    int inn_sz  = num_nets * Cfg::INNER_DIM;
    int st_sz   = num_nets * Cfg::LSTM_HIDDEN_DIM;
    int act_sz  = num_nets * Cfg::ACT_DIM;
    int per_obs = Cfg::OBS_DIM * Cfg::OBS_N;

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

    for (int n = 0; n < num_nets; n++) {
        handle.set_input_observe(n, h_obs + n * per_obs);
        handle.set_input_inner(n,   h_inn + n * Cfg::INNER_DIM);
        handle.set_grad_act(n,     h_gact + n * Cfg::ACT_DIM);
        handle.set_grad_value(n,  h_gval + n);
        handle.set_input_step(n, 0);
    }
    handle.sync();

    agent_gpu::AttnLstmPersistent<Cfg>* h_p =
        new agent_gpu::AttnLstmPersistent<Cfg>[num_nets];
    memset(h_p, 0, num_nets * sizeof(agent_gpu::AttnLstmPersistent<Cfg>));
    handle.copy_persistent_from_host(h_p);

    // ---- Forward benchmark ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        handle.forward();
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_ITERS; i++) {
        handle.forward();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fwd_ms;
    cudaEventElapsedTime(&fwd_ms, start, stop);

    r.fwd_us_per_net   = (fwd_ms / TIMED_ITERS) * 1000.0f / num_nets;

    // ---- Backward benchmark ----
    // 先做一次前向填充缓存（step=0 → cache[0]）
    handle.forward();
    cudaDeviceSynchronize();
    handle.zero_gradients();
    handle.set_input_step(0, 1);   // backward step=1 → cache_i=0, bptt_steps=1

    for (int i = 0; i < WARMUP_ITERS; i++) {
        handle.backward(0.0f, 0.0f, 0.0f, 1);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_ITERS; i++) {
        handle.backward(0.0f, 0.0f, 0.0f, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bwd_ms;
    cudaEventElapsedTime(&bwd_ms, start, stop);

    r.bwd_us_per_net   = (bwd_ms / TIMED_ITERS) * 1000.0f / num_nets;

    cudaEventDestroy(start); cudaEventDestroy(stop);
    handle.free();

    // ================================================================
    // A2C workload simulation: 10 forward + 1 backward per cycle
    // ================================================================
    // Allocate with enough caches for a2c_step
    agent_gpu::AttnLstmHandle<Cfg> a2c_handle;
    a2c_handle.alloc(num_nets);
    a2c_handle.init_weights_xavier(42);

    for (int n = 0; n < num_nets; n++) {
        a2c_handle.set_input_observe(n, h_obs + n * per_obs);
        a2c_handle.set_input_inner(n,   h_inn + n * Cfg::INNER_DIM);
        a2c_handle.set_grad_act(n,     h_gact + n * Cfg::ACT_DIM);
        a2c_handle.set_grad_value(n,  h_gval + n);
    }
    a2c_handle.sync();

    agent_gpu::AttnLstmPersistent<Cfg>* h_persist =
        new agent_gpu::AttnLstmPersistent<Cfg>[num_nets];
    memset(h_persist, 0, num_nets * sizeof(agent_gpu::AttnLstmPersistent<Cfg>));
    a2c_handle.copy_persistent_from_host(h_persist);

    // Warmup
    for (int w = 0; w < WARMUP_ITERS; w++) {
        for (int t = 0; t < A2C_STEPS; t++) {
            for (int n = 0; n < num_nets; n++) a2c_handle.set_input_step(n, t);
            a2c_handle.forward();
            a2c_handle.copy_persistent_to_host(h_persist);
        }
        for (int n = 0; n < num_nets; n++) a2c_handle.set_input_step(n, A2C_STEPS);
        a2c_handle.zero_gradients();
        a2c_handle.backward(1e-5f, 0.9f, 0.0f, A2C_STEPS);   // SGD 在 kernel 内
        a2c_handle.sync();
    }

    // Timed
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);

    for (int cycle = 0; cycle < TIMED_ITERS; cycle++) {
        // 10 forward steps (state 原位推进，无 advance_state)
        for (int t = 0; t < A2C_STEPS; t++) {
            for (int n = 0; n < num_nets; n++) a2c_handle.set_input_step(n, t);
            a2c_handle.forward();
            a2c_handle.copy_persistent_to_host(h_persist);
        }
        // 1 backward (step = A2C_STEPS 使得 cache_i = 最新槽 = A2C_STEPS-1)
        for (int n = 0; n < num_nets; n++) a2c_handle.set_input_step(n, A2C_STEPS);
        a2c_handle.zero_gradients();
        a2c_handle.backward(1e-5f, 0.9f, 0.0f, A2C_STEPS);
        a2c_handle.sync();
    }

    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float total_ms;
    cudaEventElapsedTime(&total_ms, s, e);

    r.combined_us_per_net = (total_ms / TIMED_ITERS) * 1000.0f / num_nets;
    r.cycle_per_s = (float)TIMED_ITERS / (total_ms / 1000.0f);
    r.bw_gb_s = r.cycle_per_s * num_nets * (A2C_STEPS * (Cfg::OBS_DIM * Cfg::OBS_N + Cfg::INNER_DIM + Cfg::ACT_DIM + 1) * sizeof(float)) / 1e6f;

    cudaEventDestroy(s); cudaEventDestroy(e);
    delete[] h_obs; delete[] h_inn; delete[] h_hp; delete[] h_cp;
    delete[] h_gact; delete[] h_gval; delete[] h_persist;
    a2c_handle.free();

    return r;
}

/// 主函数
int main() {
    system("chcp 65001 > nul");  // Windows: UTF-8
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
    printf("Smem/block: %zu KB, used: %.1f KB (fwd) / %.1f KB (bwd, N_MAX=%d)\n",
           prop.sharedMemPerBlock / 1024,
           agent_gpu::SmemLayoutFwd<Cfg>::TOTAL * sizeof(float) / 1024.0f,
           agent_gpu::SmemLayoutBwd<Cfg>::TOTAL * sizeof(float) / 1024.0f,
           agent_gpu::SmemLayoutBwd<Cfg>::N_MAX);
    printf("Params: %d weights + %d cache = %d floats (%.1f KB/network)\n",
           Cfg::NET_SIZE, Cfg::TOTAL_BYTES,
           Cfg::NET_SIZE + Cfg::TOTAL_BYTES,
           (sizeof(agent_gpu::AttnLstmWeights<Cfg>) + sizeof(agent_gpu::AttnLstmCache<Cfg>)) / 1024.0f);

    // Occupancy estimation
    {
        size_t smem_per_block = agent_gpu::SmemLayoutFwd<Cfg>::TOTAL * sizeof(float);
        int max_blocks_by_smem = (int)(prop.sharedMemPerBlock * prop.multiProcessorCount / smem_per_block);
        if (smem_per_block == 0) max_blocks_by_smem = 9999;
        int max_blocks_by_hw = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;
        int max_warps_by_sm = prop.maxThreadsPerMultiProcessor / 32;
        printf("Occupancy (fwd smem): %.1f KB → max %d blocks across %d SMs "
               "(hw limit: %d)\n",
               smem_per_block / 1024.0f,
               max_blocks_by_smem, prop.multiProcessorCount,
               max_blocks_by_hw);
        printf("  Per SM: %.1f KB smem/block → %d blocks/SM "
               "(max %d blocks/SM, max %d warps/SM)\n",
               smem_per_block / 1024.0f,
               (int)(prop.sharedMemPerBlock / smem_per_block),
               prop.maxBlocksPerMultiProcessor,
               max_warps_by_sm);
    }
    printf("\n");
    printf("Warmup: %d, Timed: %d\n", WARMUP_ITERS, TIMED_ITERS);
    printf("Each cycle: %d forward (state advancing) + 1 backward (BPTT, %d steps)\n", A2C_STEPS, A2C_STEPS);

    printf("\n========================================================\n");
    printf("Network Benchmark & A2C Workload Simulation:\n\n");

    // 测试不同网络数量
    int net_counts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int num_tests = sizeof(net_counts) / sizeof(net_counts[0]);

    printf("%10s  %18s  %18s  %18s  %16s  %16s\n",
           "Nets", "Fwd Latency", "Bwd Latency", "Workload Latency", "Throughput", "Bandwidth");
    printf("%10s  %18s  %18s  %18s  %16s  %16s\n",
           "", "μs", "μs", "μs", "cycle/s", "MB/s");
    printf("%10s  %18s  %18s  %18s  %16s  %16s\n",
           "------", "-------------", "-------------", "------------------", "------------", "-----------");

    for (int i = 0; i < num_tests; i++) {
        int n = net_counts[i];
        BenchResult r = measure(n);

        printf("%10d  %18.2f  %18.2f  %18.2f  %16.1f  %16.1f\n",
               n,
               r.fwd_us_per_net,
               r.bwd_us_per_net,
               r.combined_us_per_net,
               r.cycle_per_s,
               r.bw_gb_s);
    }
    printf("\nDone.\n");

    return 0;
}
