/**
 * attnlstm_test.cu — AttnLSTM Actor + FCN Critic 正确性测试
 *
 * 验证 GPU 前向/反向与 NumPy golden 数据的一致性。
 *
 * Usage:
 *   1. python python/tools/generate_golden_attnlstm.py
 *   2. cmake --build build --config Release --target testAttnLstm
 *   3. ./build/test/test-attnlstm/Release/testAttnLstm
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "net_config.cuh"
#include "net_kernel.cuh"
#include "net_host.cuh"

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

using Cfg = agent_gpu::DefaultConfig;

constexpr int OBS_N    = Cfg::OBS_N;
constexpr int OBS_DIM  = Cfg::OBS_DIM;
constexpr int INNER_DIM = Cfg::INNER_DIM;
constexpr int ACT_DIM  = Cfg::ACT_DIM;
constexpr int D_E      = Cfg::ATTN_EMBED_DIM;
constexpr int H_KV     = Cfg::ATTN_KV_HEADS;
constexpr int H_Q      = Cfg::ATTN_Q_HEADS;
constexpr int D_H      = Cfg::LSTM_HIDDEN_DIM;
constexpr int D_H1     = Cfg::LSTM_QUERY_DIM;
constexpr int D_H2     = Cfg::LSTM_OUTPUT_DIM;
constexpr int D_IN     = Cfg::LSTM_INPUT_DIM;
constexpr int D_C      = Cfg::CRITIC_HIDDEN_DIM;

// 权重区总大小（与 net_kernel.cuh 一致）
constexpr int WEIGHT_FLOATS =
    2 * H_KV * D_E * OBS_DIM        // Wkv
    + H_Q * D_E * D_H1              // Wq
    + H_Q * D_E                     // bcat
    + 3 * D_H * D_IN                // Wico
    + 3 * D_H                       // bico
    + D_H2 * ACT_DIM                // Wf
    + ACT_DIM                       // bf
    + D_C * OBS_DIM                 // Wc1
    + D_C                           // bc1
    + D_C * (D_C + INNER_DIM)       // Wc2
    + D_C                           // bc2
    + D_C * D_C                     // Wc3
    + D_C                           // bc3
    + 1 * D_C                       // Wc4
    + 1;                            // bc4

constexpr int GRAD_FLOATS = WEIGHT_FLOATS;

// ================================================================
// 各权重场的字节偏移（用于分场验证）
// ================================================================
constexpr int OFF_WKV  = 0;
constexpr int SZ_WKV   = 2 * H_KV * D_E * OBS_DIM;
constexpr int OFF_WQ   = OFF_WKV + SZ_WKV;
constexpr int SZ_WQ    = H_Q * D_E * D_H1;
constexpr int OFF_BCAT = OFF_WQ + SZ_WQ;
constexpr int SZ_BCAT  = H_Q * D_E;
constexpr int OFF_WICO = OFF_BCAT + SZ_BCAT;
constexpr int SZ_WICO  = 3 * D_H * D_IN;
constexpr int OFF_BICO = OFF_WICO + SZ_WICO;
constexpr int SZ_BICO  = 3 * D_H;
constexpr int OFF_WF   = OFF_BICO + SZ_BICO;
constexpr int SZ_WF    = D_H2 * ACT_DIM;
constexpr int OFF_BF   = OFF_WF + SZ_WF;
constexpr int SZ_BF    = ACT_DIM;
constexpr int OFF_WC1  = OFF_BF + SZ_BF;
constexpr int SZ_WC1   = D_C * OBS_DIM;
constexpr int OFF_BC1  = OFF_WC1 + SZ_WC1;
constexpr int SZ_BC1   = D_C;
constexpr int OFF_WC2  = OFF_BC1 + SZ_BC1;
constexpr int SZ_WC2   = D_C * (D_C + INNER_DIM);
constexpr int OFF_BC2  = OFF_WC2 + SZ_WC2;
constexpr int SZ_BC2   = D_C;
constexpr int OFF_WC3  = OFF_BC2 + SZ_BC2;
constexpr int SZ_WC3   = D_C * D_C;
constexpr int OFF_BC3  = OFF_WC3 + SZ_WC3;
constexpr int SZ_BC3   = D_C;
constexpr int OFF_WC4  = OFF_BC3 + SZ_BC3;
constexpr int SZ_WC4   = D_C;
constexpr int OFF_BC4  = OFF_WC4 + SZ_WC4;
constexpr int SZ_BC4   = 1;

// ================================================================
// 误差检查
// ================================================================
static float rel_error(float a, float b) {
    float denom = fmaxf(fabsf(a), fabsf(b));
    if (denom < 1e-8f) return fabsf(a - b);
    return fabsf(a - b) / denom;
}

static bool allclose(const float* a, const float* b, int n,
                     float rtol = 1e-3f, float atol = 1e-4f,
                     const char* label = "", bool verbose = true) {
    float max_rel = 0, max_abs = 0;
    int bad_count = 0, first_bad = -1;
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
        printf("  %-35s  %s  (max_rel=%.2e  max_abs=%.2e  bad=%d",
               label, ok ? "PASS" : "FAIL", max_rel, max_abs, bad_count);
        if (!ok && first_bad >= 0) {
            printf("  first[%d]: gpu=%.6f  golden=%.6f", first_bad, a[first_bad], b[first_bad]);
        }
        printf(")\n");
    }
    return ok;
}

// ================================================================
// Binary golden data reader
// ================================================================
struct GoldenData {
    float* weights;     // [WEIGHT_FLOATS]
    float* observe;     // [OBS_DIM * OBS_N]
    float* inner;       // [INNER_DIM]
    float* h_prev;      // [D_H]
    float* c_prev;      // [D_H]
    float* act_out;     // [ACT_DIM]
    float* value_out;   // [1]
    float* h_new;       // [D_H]
    float* c_new;       // [D_H]
    float* g_act;       // [ACT_DIM]
    float* g_value;     // [1]
    float* grads;       // [GRAD_FLOATS]
    float* g_observe;   // [OBS_DIM * OBS_N]
    float* g_h_prev;    // [D_H]
    float* g_c_prev;    // [D_H]

    GoldenData() {
        weights   = new float[WEIGHT_FLOATS];
        observe   = new float[OBS_DIM * OBS_N];
        inner     = new float[INNER_DIM];
        h_prev    = new float[D_H];
        c_prev    = new float[D_H];
        act_out   = new float[ACT_DIM];
        value_out = new float[1];
        h_new     = new float[D_H];
        c_new     = new float[D_H];
        g_act     = new float[ACT_DIM];
        g_value   = new float[1];
        grads     = new float[GRAD_FLOATS];
        g_observe = new float[OBS_DIM * OBS_N];
        g_h_prev  = new float[D_H];
        g_c_prev  = new float[D_H];
    }

    ~GoldenData() {
        delete[] weights; delete[] observe; delete[] inner;
        delete[] h_prev; delete[] c_prev;
        delete[] act_out; delete[] value_out;
        delete[] h_new; delete[] c_new;
        delete[] g_act; delete[] g_value;
        delete[] grads; delete[] g_observe;
        delete[] g_h_prev; delete[] g_c_prev;
    }

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

        char magic[5] = {0};
        int version;
        fread(magic, 1, 4, f);
        fread(&version, sizeof(int), 1, f);
        printf("Golden data: magic=%.4s version=%d\n", magic, version);

        // 跳过维度头（13个int）
        int dims[13];
        fread(dims, sizeof(int), 13, f);

        auto read_arr = [&](float* dst) {
            int count;
            fread(&count, sizeof(int), 1, f);
            fread(dst, sizeof(float), count, f);
        };

        read_arr(weights);
        read_arr(observe);
        read_arr(inner);
        read_arr(h_prev);
        read_arr(c_prev);
        read_arr(act_out);
        read_arr(value_out);
        read_arr(h_new);
        read_arr(c_new);
        read_arr(g_act);
        read_arr(g_value);
        read_arr(grads);
        read_arr(g_observe);
        read_arr(g_h_prev);
        read_arr(g_c_prev);

        fclose(f);
        return true;
    }
};

// ================================================================
// Main test
// ================================================================
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr, "Starting AttnLSTM Kernel test...\n");

    cudaError_t init_err = cudaFree(0);
    fprintf(stderr, "CUDA init: %d (%s)\n", (int)init_err, cudaGetErrorString(init_err));
    if (init_err != cudaSuccess) return 1;

    printf("=== AttnLSTM Single-Block Kernel Test ===\n\n");

    // Device info
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (SM %d.%d, %zu MB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024 * 1024));

    // Load golden data
    GoldenData golden;
    if (!golden.load("golden_attnlstm.bin")) {
        fprintf(stderr, "FATAL: Cannot load golden data. Run generate_golden_attnlstm.py first.\n");
        return 1;
    }
    printf("  First 5 golden weights: %.6f %.6f %.6f %.6f %.6f\n\n",
           golden.weights[0], golden.weights[1], golden.weights[2],
           golden.weights[3], golden.weights[4]);

    constexpr int NUM_NETS = 1;
    agent_gpu::AttnLstmHandle<Cfg> handle;
    handle.alloc(NUM_NETS);

    // ================================================================
    // Test 1: Forward correctness
    // ================================================================
    printf("--- Test 1: Forward Correctness (single step) ---\n");

    // 加载权重到GPU
    agent_gpu::AttnLstmWeights<Cfg>* h_w = new agent_gpu::AttnLstmWeights<Cfg>[NUM_NETS];
    memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
    memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
    handle.copy_weights_from_host(h_w);

    // 加载输入
    CUDA_CHECK(cudaMemcpy(handle.d_observe, golden.observe,
                          OBS_DIM * OBS_N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(handle.d_inner, golden.inner,
                          INNER_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // 加载持久状态
    agent_gpu::AttnLstmPersistent<Cfg>* h_p = new agent_gpu::AttnLstmPersistent<Cfg>[NUM_NETS];
    memcpy(h_p[0].h, golden.h_prev, D_H * sizeof(float));
    memcpy(h_p[0].c, golden.c_prev, D_H * sizeof(float));
    handle.copy_persistent_from_host(h_p);

    // 前向
    fprintf(stderr, "Launching forward kernel...\n");
    handle.forward(0);
    cudaError_t kern_err = cudaDeviceSynchronize();
    if (kern_err != cudaSuccess) {
        fprintf(stderr, "FORWARD KERNEL ERROR: %s\n", cudaGetErrorString(kern_err));
        return 1;
    }
    fprintf(stderr, "Forward kernel completed.\n");

    // 读取输出
    float h_act[ACT_DIM], h_value[1];
    CUDA_CHECK(cudaMemcpy(h_act, handle.d_act, ACT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_value, handle.d_value, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    agent_gpu::AttnLstmPersistent<Cfg> h_p_out[NUM_NETS];
    handle.copy_persistent_to_host(h_p_out);

    bool fwd_act_ok   = allclose(h_act, golden.act_out, ACT_DIM, 1e-4f, 1e-5f, "Actor output (act)");
    bool fwd_value_ok = allclose(h_value, golden.value_out, 1, 1e-4f, 1e-5f, "Critic output (value)");
    bool fwd_h_ok     = allclose(h_p_out[0].h, golden.h_new, D_H, 1e-4f, 1e-5f, "Hidden state (h_new)");
    bool fwd_c_ok     = allclose(h_p_out[0].c, golden.c_new, D_H, 1e-4f, 1e-5f, "Cell state (c_new)");

    printf("  GPU act:   [%.6f, %.6f]  golden: [%.6f, %.6f]\n",
           h_act[0], h_act[1], golden.act_out[0], golden.act_out[1]);
    printf("  GPU value: %.6f  golden: %.6f\n", h_value[0], golden.value_out[0]);

    // Diagnostic: read caches from GPU and print key intermediates
    {
        agent_gpu::AttnLstmCache<Cfg> h_cache;
        CUDA_CHECK(cudaMemcpy(&h_cache, handle.d_caches,
                              sizeof(agent_gpu::AttnLstmCache<Cfg>),
                              cudaMemcpyDeviceToHost));

        // Compare K, Q, gate_i, h_new
        float cache_diff;

        // Check a few K values
        printf("\n  --- GPU Cache Diag ---\n");
        printf("  K[0,0,0:4]:  gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.k[0], h_cache.k[1], h_cache.k[2], h_cache.k[3]);
        printf("  K[1,0,0:4]:  gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.k[128], h_cache.k[129], h_cache.k[130], h_cache.k[131]);
        printf("  Q[0,0:4]:    gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.q[0], h_cache.q[1], h_cache.q[2], h_cache.q[3]);
        printf("  P[0,0:4]:    gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.p[0], h_cache.p[1], h_cache.p[2], h_cache.p[3]);
        printf("  gate_i[0:4]: gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.gate_i[0], h_cache.gate_i[1], h_cache.gate_i[2], h_cache.gate_i[3]);
        printf("  c_alt[0:4]:  gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.c_alt[0], h_cache.c_alt[1], h_cache.c_alt[2], h_cache.c_alt[3]);
        printf("  gate_o[0:4]: gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.gate_o[0], h_cache.gate_o[1], h_cache.gate_o[2], h_cache.gate_o[3]);
        // x contains h_prev + inner + _observe
        printf("  x[0:4]  (hp):  gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.x[0], h_cache.x[1], h_cache.x[2], h_cache.x[3]);
        printf("  x[24:28](obs): gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.x[24], h_cache.x[25], h_cache.x[26], h_cache.x[27]);
        printf("  h_new[0:4]:   gpu=[%.6f %.6f %.6f %.6f]\n",
               h_p_out[0].h[0], h_p_out[0].h[1], h_p_out[0].h[2], h_p_out[0].h[3]);
        printf("  h_new[8:12]:  gpu=[%.6f %.6f %.6f %.6f]\n",
               h_p_out[0].h[8], h_p_out[0].h[9], h_p_out[0].h[10], h_p_out[0].h[11]);
        printf("  c_new[0:4]:   gpu=[%.6f %.6f %.6f %.6f]\n",
               h_p_out[0].c[0], h_p_out[0].c[1], h_p_out[0].c[2], h_p_out[0].c[3]);
        printf("  lstm_o[0:4]:  gpu=[%.6f %.6f %.6f %.6f]\n",
               h_cache.lstm_o[0], h_cache.lstm_o[1], h_cache.lstm_o[2], h_cache.lstm_o[3]);
    }

    bool fwd_all = fwd_act_ok && fwd_value_ok && fwd_h_ok && fwd_c_ok;
    printf("  Forward overall: %s\n\n", fwd_all ? "PASS" : "FAIL");

    // ================================================================
    // Test 2: Backward correctness
    // ================================================================
    printf("--- Test 2: Backward Correctness (single step) ---\n");

    // 重新加载权重（清除前向可能残留的梯度）
    memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
    memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
    handle.copy_weights_from_host(h_w);
    handle.zero_gradients();

    // 传递上游梯度（使用golden中的g_act和g_value）
    CUDA_CHECK(cudaMemcpy(handle.d_grad_act, golden.g_act,
                          ACT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(handle.d_grad_value, golden.g_value,
                          1 * sizeof(float), cudaMemcpyHostToDevice));

    // 重新前向以填充缓存
    handle.forward(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 反向
    fprintf(stderr, "Launching backward kernel...\n");
    handle.backward(1);
    handle.sync();
    fprintf(stderr, "Backward kernel completed.\n");

    // 读取权重梯度
    agent_gpu::AttnLstmWeights<Cfg> h_w_gpu;
    handle.copy_weights_to_host(&h_w_gpu);
    const float* gpu_grads = h_w_gpu.grad_Wkv;  // first gradient field

    printf("  Upstream: g_act=[%.6f, %.6f] g_value=%.6f\n",
           golden.g_act[0], golden.g_act[1], golden.g_value[0]);

    // 分场验证所有权重梯度
    bool all_bwd_ok = true;
    #define CHECK_BWD(name, off, sz, rtol, atol) \
        do { \
            bool ok = allclose(gpu_grads + (off), golden.grads + (off), (sz), rtol, atol, "grad_" #name); \
            all_bwd_ok = all_bwd_ok && ok; \
        } while(0)

    CHECK_BWD(Wkv,  OFF_WKV,  SZ_WKV,  1e-2f, 1e-3f);
    CHECK_BWD(Wq,   OFF_WQ,   SZ_WQ,   1e-2f, 1e-3f);
    CHECK_BWD(bcat, OFF_BCAT, SZ_BCAT, 1e-2f, 1e-3f);
    CHECK_BWD(Wico, OFF_WICO, SZ_WICO, 1e-2f, 1e-3f);
    CHECK_BWD(bico, OFF_BICO, SZ_BICO, 1e-2f, 1e-3f);
    CHECK_BWD(Wf,   OFF_WF,   SZ_WF,   1e-2f, 1e-3f);
    CHECK_BWD(bf,   OFF_BF,   SZ_BF,   1e-2f, 1e-3f);
    CHECK_BWD(Wc1,  OFF_WC1,  SZ_WC1,  1e-2f, 1e-3f);
    CHECK_BWD(bc1,  OFF_BC1,  SZ_BC1,  1e-2f, 1e-3f);
    CHECK_BWD(Wc2,  OFF_WC2,  SZ_WC2,  1e-2f, 1e-3f);
    CHECK_BWD(bc2,  OFF_BC2,  SZ_BC2,  1e-2f, 1e-3f);
    CHECK_BWD(Wc3,  OFF_WC3,  SZ_WC3,  1e-2f, 1e-3f);
    CHECK_BWD(bc3,  OFF_BC3,  SZ_BC3,  1e-2f, 1e-3f);
    CHECK_BWD(Wc4,  OFF_WC4,  SZ_WC4,  1e-2f, 1e-3f);
    CHECK_BWD(bc4,  OFF_BC4,  SZ_BC4,  1e-2f, 1e-3f);

    #undef CHECK_BWD

    printf("  Backward overall: %s\n\n", all_bwd_ok ? "PASS" : "FAIL");

    // ================================================================
    // Test 3: Smem budget check
    // ================================================================
    printf("--- Test 3: Smem budget check ---\n");
    size_t smem_total = agent_gpu::SmemLayout<Cfg>::TOTAL * sizeof(float);
    printf("  Required smem: %zu bytes (%.1f KB)\n", smem_total, smem_total / 1024.0f);
    printf("  Available smem per block: %zu bytes\n", prop.sharedMemPerBlock);
    if (smem_total <= prop.sharedMemPerBlock) {
        printf("  Smem budget: PASS\n\n");
    } else {
        printf("  Smem budget: FAIL — exceeds device limit!\n\n");
    }

    // Cleanup
    delete[] h_w;
    delete[] h_p;
    handle.free();

    bool overall = fwd_all && all_bwd_ok;
    printf("=== Overall: %s ===\n", overall ? "PASS" : "FAIL");
    return overall ? 0 : 1;
}
