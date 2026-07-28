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
    handle.set_input_observe(0, golden.observe);
    handle.set_input_inner(0, golden.inner);
    handle.set_input_step(0, 0);

    // 加载持久状态
    agent_gpu::AttnLstmPersistent<Cfg>* h_p = new agent_gpu::AttnLstmPersistent<Cfg>[NUM_NETS];
    memcpy(h_p[0].h, golden.h_prev, D_H * sizeof(float));
    memcpy(h_p[0].c, golden.c_prev, D_H * sizeof(float));
    handle.copy_persistent_from_host(h_p);

    // 前向
    fprintf(stderr, "Launching forward kernel...\n");
    handle.forward();
    cudaError_t kern_err = cudaDeviceSynchronize();
    if (kern_err != cudaSuccess) {
        fprintf(stderr, "FORWARD KERNEL ERROR: %s\n", cudaGetErrorString(kern_err));
        return 1;
    }
    fprintf(stderr, "Forward kernel completed.\n");

    // 读取输出
    float h_act[ACT_DIM], h_value[1];
    handle.get_output_act(0, h_act);
    handle.get_output_value(0, h_value);
    handle.sync();

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
    // Test 2: Backward correctness (single step)
    // ================================================================
    printf("--- Test 2: Backward Correctness (single step) ---\n");

    // 重新加载权重（清除前向可能残留的梯度）
    memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
    memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
    handle.copy_weights_from_host(h_w);
    handle.zero_gradients();

    // 传递上游梯度到 slot 0（前向 step=0 → cache slot 0）
    handle.set_grad_act_at_slot(0, 0, golden.g_act);
    handle.set_grad_value_at_slot(0, 0, golden.g_value);

    // 重新前向以填充缓存（step=0 → cache[0]）
    handle.set_input_step(0, 0);
    handle.forward();
    CUDA_CHECK(cudaDeviceSynchronize());

    // step=1 使 cache_i = (1-1)%A2C = 0，bptt_steps=1
    // 必须在前向之后设置step，避免被前向的 set_input_step 覆盖
    handle.set_input_step(0, 1);

    // 反向：lr=0, beta=0: v_new = g, w unchanged（可验证 g 正确性）
    //       bptt_steps=1: 仅遍历 1 个 cache
    fprintf(stderr, "Launching backward kernel...\n");
    handle.backward(0.0f, 0.0f, 0.0f, 1);
    handle.sync();
    fprintf(stderr, "Backward kernel completed.\n");

    // 读取权重梯度（梯度为独立 AttnLstmWeights，字段布局与权重相同）
    agent_gpu::AttnLstmWeights<Cfg> h_w_grad;
    handle.copy_grads_to_host(&h_w_grad);
    const float* gpu_grads = h_w_grad.Wkv;  // first field (grad_Wkv)

    printf("  Upstream: g_act=[%.6f, %.6f] g_value=%.6f\n",
           golden.g_act[0], golden.g_act[1], golden.g_value[0]);

    // 分场验证所有权重梯度
    // Actor 注意力/LSTM 字段有 fp32 BPTT 数值累积误差（见 doc/attn-lstm-implementation.md "误差来源分析"）
    // Critic 和输出层应有机器精度
    bool all_bwd_ok = true;
    #define CHECK_BWD(name, off, sz, rtol, atol) \
        do { \
            bool ok = allclose(gpu_grads + (off), golden.grads + (off), (sz), rtol, atol, "grad_" #name); \
            all_bwd_ok = all_bwd_ok && ok; \
        } while(0)

    CHECK_BWD(Wkv,  OFF_WKV,  SZ_WKV,  5e-2f, 8e-3f);
    CHECK_BWD(Wq,   OFF_WQ,   SZ_WQ,   5e-2f, 8e-3f);
    CHECK_BWD(bcat, OFF_BCAT, SZ_BCAT, 5e-2f, 8e-3f);
    CHECK_BWD(Wico, OFF_WICO, SZ_WICO, 5e-2f, 8e-3f);
    CHECK_BWD(bico, OFF_BICO, SZ_BICO, 5e-2f, 8e-3f);
    CHECK_BWD(Wf,   OFF_WF,   SZ_WF,   5e-2f, 8e-3f);
    CHECK_BWD(bf,   OFF_BF,   SZ_BF,   5e-2f, 8e-3f);
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
    // Test 3: Weight integrity (verify weight loading)
    // ================================================================
    printf("--- Test 3: Weight Integrity Check ---\n");
    {
        agent_gpu::AttnLstmWeights<Cfg> h_w_chk;
        handle.copy_weights_to_host(&h_w_chk);
        const float* gpu_w = h_w_chk.Wkv;
        bool wf_ok = allclose(gpu_w + OFF_WF, golden.weights + OFF_WF, SZ_WF,
                              1e-6f, 1e-6f, "Weight: Wf");
        bool bf_ok = allclose(gpu_w + OFF_BF, golden.weights + OFF_BF, SZ_BF,
                              1e-6f, 1e-6f, "Weight: bf");
        bool wkv_ok = allclose(gpu_w + OFF_WKV, golden.weights + OFF_WKV,
                               fminf(SZ_WKV, 16), 1e-6f, 1e-6f, "Weight: Wkv[0:16]");
        printf("  Weight integrity: %s\n\n",
               (wf_ok && bf_ok && wkv_ok) ? "PASS" : "FAIL");
    }

    // ================================================================
    // Test 4: Cache integrity (verify GPU cache vs golden)
    // ================================================================
    printf("--- Test 4: Cache Integrity Check ---\n");
    {
        agent_gpu::AttnLstmCache<Cfg> h_cache;
        CUDA_CHECK(cudaMemcpy(&h_cache, handle.d_caches,
                              sizeof(agent_gpu::AttnLstmCache<Cfg>),
                              cudaMemcpyDeviceToHost));

        // Compute expected values from golden data using NumPy reference
        // For now, verify key cache fields are non-trivial (not all zeros)
        auto chk_nonzero = [](const float* arr, int n, const char* label) {
            float sum_abs = 0.0f;
            for (int i = 0; i < n; ++i) sum_abs += fabsf(arr[i]);
            bool ok = (sum_abs > 1e-6f);
            printf("  %-35s  %s  (sum_abs=%.4f)\n", label, ok ? "PASS" : "FAIL", sum_abs);
            return ok;
        };
        printf("  (Cache fields populated check)\n");
        chk_nonzero(h_cache.k, H_KV * D_E * OBS_N, "cache.k");
        chk_nonzero(h_cache.v, H_KV * D_E * OBS_N, "cache.v");
        chk_nonzero(h_cache.q, H_Q * D_E, "cache.q");
        chk_nonzero(h_cache.p, H_Q * H_KV * OBS_N, "cache.p");
        chk_nonzero(h_cache.x, D_IN, "cache.x");
        chk_nonzero(h_cache.gate_i, D_H, "cache.gate_i");
        chk_nonzero(h_cache.gate_o, D_H, "cache.gate_o");
        chk_nonzero(h_cache.c_alt, D_H, "cache.c_alt");
        chk_nonzero(h_cache.tanhc, D_H, "cache.tanhc");
        chk_nonzero(h_cache.lstm_o, D_H2, "cache.lstm_o");
        chk_nonzero(h_cache.act, ACT_DIM, "cache.act");
        chk_nonzero(h_cache.c_h1, D_C, "cache.c_h1");
        chk_nonzero(h_cache.c_h2, D_C, "cache.c_h2");
        chk_nonzero(h_cache.c_h3, D_C, "cache.c_h3");
    }
    printf("\n");

    // ================================================================
    // Test 5: Gradient accumulation (double backward, beta=1.0 for accumulation)
    // ================================================================
    printf("--- Test 5: Gradient Accumulation (double backward) ---\n");
    {
        // Reset weights, zero grads
        memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
        memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
        handle.copy_weights_from_host(h_w);
        handle.zero_gradients();

        // Forward
        handle.set_input_step(0, 0);
        handle.forward();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Backward x2: lr=0 (weights unchanged), beta=1.0 (accumulate v)
        // v_new = 1.0 * v_old + g → v accumulates g on each call
        handle.set_input_step(0, 1);
        handle.backward(0.0f, 1.0f, 0.0f, 1);   // v = 0 + g = g
        handle.backward(0.0f, 1.0f, 0.0f, 1);   // v = g + g = 2g
        handle.sync();

        agent_gpu::AttnLstmWeights<Cfg> h_grad2;
        handle.copy_grads_to_host(&h_grad2);
        const float* g2 = h_grad2.Wkv;

        // Compare: grad_accumulated should be ~2x golden grads
        // (within fp32 BPTT accumulation tolerance)
        bool acc_ok = true;
        float sum_single = 0.0f, sum_double = 0.0f;
        for (int i = 0; i < WEIGHT_FLOATS; i++) {
            sum_single += fabsf(golden.grads[i]);
            sum_double += fabsf(g2[i]);
        }
        // Double backward should roughly double the gradient magnitude
        float ratio = sum_double / fmaxf(sum_single, 1e-8f);
        printf("  Single grad L1: %.4f, Double grad L1: %.4f, ratio: %.2f\n",
               sum_single, sum_double, ratio);
        if (ratio < 1.5f || ratio > 2.5f) acc_ok = false;
        printf("  Gradient accumulation: %s (ratio=%.2f)\n\n",
               acc_ok ? "PASS" : "FAIL", ratio);
    }

    // ================================================================
    // Test 6: SGD update verification (non-zero lr, beta)
    // ================================================================
    printf("--- Test 6: SGD Update Verification ---\n");
    {
        // Reset weights
        memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
        memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
        handle.copy_weights_from_host(h_w);
        handle.zero_gradients();

        // Forward
        handle.set_input_step(0, 0);
        handle.forward();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Backward with SGD: lr=0.01, beta=0.9, gamma=0 (no weight decay)
        handle.set_input_step(0, 1);
        handle.backward(1e-2f, 0.9f, 0.0f, 1);
        handle.sync();

        // Read updated weights + grads
        agent_gpu::AttnLstmWeights<Cfg> h_w_new, h_g_new;
        handle.copy_weights_to_host(&h_w_new);
        handle.copy_grads_to_host(&h_g_new);

        // Check weights changed (Wkv first few elements)
        const float* w_new = h_w_new.Wkv;
        bool changed = false;
        for (int i = 0; i < 8; ++i) {
            if (fabsf(w_new[OFF_WKV + i] - golden.weights[OFF_WKV + i]) > 1e-8f) {
                changed = true;
                break;
            }
        }
        // Check momentum v = beta*0 + g = g (first iteration, v_old=0)
        bool mom_ok = true;
        const float* v_new = h_g_new.Wkv;
        for (int i = 0; i < fminf(8, SZ_WKV); ++i) {
            float expected_v = golden.grads[OFF_WKV + i];  // v_new = 0.9*0 + g = g
            if (fabsf(v_new[OFF_WKV + i] - expected_v) > 1e-3f) {
                mom_ok = false;
                break;
            }
        }
        printf("  Weight changed: %s\n", changed ? "PASS" : "FAIL (no update)");
        printf("  Momentum init:  %s\n\n", mom_ok ? "PASS" : "FAIL");

        printf("  Note: w_new = w_old*(1-lr*gamma) + lr*v_new (gamma=0)\n");
        printf("        v_new = beta*v_old + g  (v_old=0, beta=0.9)\n");
    }

    // ================================================================
    // Test 3: Smem budget check
    // ================================================================
    printf("--- Test 3: Smem budget check ---\n");
    size_t smem_fwd = agent_gpu::SmemLayoutFwd<Cfg>::TOTAL * sizeof(float);
    size_t smem_bwd = agent_gpu::SmemLayoutBwd<Cfg>::TOTAL * sizeof(float);
    printf("  Forward smem: %zu bytes (%.1f KB)\n", smem_fwd, smem_fwd / 1024.0f);
    printf("  Backward smem: %zu bytes (%.1f KB) (N_MAX=%d)\n", smem_bwd, smem_bwd / 1024.0f,
           agent_gpu::SmemLayoutBwd<Cfg>::N_MAX);
    size_t smem_total = smem_fwd > smem_bwd ? smem_fwd : smem_bwd;
    printf("  Required smem (max): %zu bytes (%.1f KB)\n", smem_total, smem_total / 1024.0f);
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
    if (!overall) {
        printf("  Forward:  %s\n", fwd_all ? "PASS" : "FAIL");
        printf("  Backward: %s\n", all_bwd_ok ? "PASS" : "FAIL");
    }
    return overall ? 0 : 1;
}
