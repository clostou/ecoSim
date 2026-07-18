/**
 * attnlstm_multi_test.cu — AttnLSTM 多步 BPTT 正确性测试
 *
 * 验证 GPU 多步前向（状态推进）+ 多步反向（BPTT梯度累加）。
 *
 * Usage:
 *   1. python python/tools/generate_golden_attnlstm_multi.py 2
 *   2. cmake --build build --config Release --target testAttnLstmMulti
 *   3. ./build/test/test-attnlstm/Release/testAttnLstmMulti
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

constexpr int OBS_N     = Cfg::OBS_N;
constexpr int OBS_DIM   = Cfg::OBS_DIM;
constexpr int INNER_DIM = Cfg::INNER_DIM;
constexpr int ACT_DIM   = Cfg::ACT_DIM;
constexpr int D_E       = Cfg::ATTN_EMBED_DIM;
constexpr int H_KV      = Cfg::ATTN_KV_HEADS;
constexpr int H_Q       = Cfg::ATTN_Q_HEADS;
constexpr int D_H       = Cfg::LSTM_HIDDEN_DIM;
constexpr int D_H1      = Cfg::LSTM_QUERY_DIM;
constexpr int D_H2      = Cfg::LSTM_OUTPUT_DIM;
constexpr int D_IN      = Cfg::LSTM_INPUT_DIM;
constexpr int D_C       = Cfg::CRITIC_HIDDEN_DIM;

constexpr int WEIGHT_FLOATS =
    2 * H_KV * D_E * OBS_DIM + H_Q * D_E * D_H1 + H_Q * D_E
    + 3 * D_H * D_IN + 3 * D_H + D_H2 * ACT_DIM + ACT_DIM
    + D_C * OBS_DIM + D_C + D_C * (D_C + INNER_DIM) + D_C
    + D_C * D_C + D_C + 1 * D_C + 1;

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
// Multi-step golden data
// ================================================================
struct MultiStepGolden {
    int num_steps;
    float* weights;

    // Per-step inputs
    float** observe;
    float** inner;
    float** h_prev;
    float** c_prev;

    // Per-step caches (read by backward kernel)
    float** x_kv;  float** k;    float** v;   float** q;
    float** p;     float** x;    float** h_prev_c; float** c_prev_c;
    float** gate_i; float** gate_o; float** c_alt; float** tanhc;
    float** lstm_o; float** act_cache;
    float** c_h1; int** argmax_c_h1; float** c_h2; float** c_h3;

    // Per-step outputs
    float** act_out; float** value_out;
    float** h_new;   float** c_new;

    // Upstream grads (last step only)
    float* g_act; float* g_value;

    // Accumulated weight grads
    float* grads;

    // Per-step input grads (BPTT)
    float** g_observe; float** g_h_prev; float** g_c_prev;

    MultiStepGolden(int ns) : num_steps(ns) {
        weights = new float[WEIGHT_FLOATS];
        observe = new float*[ns]; inner = new float*[ns]; h_prev = new float*[ns]; c_prev = new float*[ns];
        x_kv = new float*[ns]; k = new float*[ns]; v = new float*[ns]; q = new float*[ns];
        p = new float*[ns]; x = new float*[ns]; h_prev_c = new float*[ns]; c_prev_c = new float*[ns];
        gate_i = new float*[ns]; gate_o = new float*[ns]; c_alt = new float*[ns]; tanhc = new float*[ns];
        lstm_o = new float*[ns]; act_cache = new float*[ns];
        c_h1 = new float*[ns]; argmax_c_h1 = new int*[ns]; c_h2 = new float*[ns]; c_h3 = new float*[ns];
        act_out = new float*[ns]; value_out = new float*[ns]; h_new = new float*[ns]; c_new = new float*[ns];
        g_observe = new float*[ns]; g_h_prev = new float*[ns]; g_c_prev = new float*[ns];
        g_act = new float[ACT_DIM]; g_value = new float[1];
        grads = new float[WEIGHT_FLOATS];
        memset(grads, 0, WEIGHT_FLOATS * sizeof(float));
    }

    ~MultiStepGolden() {
        delete[] weights; delete[] g_act; delete[] g_value; delete[] grads;
        for (int t = 0; t < num_steps; t++) {
            delete[] observe[t]; delete[] inner[t]; delete[] h_prev[t]; delete[] c_prev[t];
            delete[] x_kv[t]; delete[] k[t]; delete[] v[t]; delete[] q[t];
            delete[] p[t]; delete[] x[t]; delete[] h_prev_c[t]; delete[] c_prev_c[t];
            delete[] gate_i[t]; delete[] gate_o[t]; delete[] c_alt[t]; delete[] tanhc[t];
            delete[] lstm_o[t]; delete[] act_cache[t];
            delete[] c_h1[t]; delete[] argmax_c_h1[t]; delete[] c_h2[t]; delete[] c_h3[t];
            delete[] act_out[t]; delete[] value_out[t]; delete[] h_new[t]; delete[] c_new[t];
            delete[] g_observe[t]; delete[] g_h_prev[t]; delete[] g_c_prev[t];
        }
        delete[] observe; delete[] inner; delete[] h_prev; delete[] c_prev;
        delete[] x_kv; delete[] k; delete[] v; delete[] q;
        delete[] p; delete[] x; delete[] h_prev_c; delete[] c_prev_c;
        delete[] gate_i; delete[] gate_o; delete[] c_alt; delete[] tanhc;
        delete[] lstm_o; delete[] act_cache;
        delete[] c_h1; delete[] argmax_c_h1; delete[] c_h2; delete[] c_h3;
        delete[] act_out; delete[] value_out; delete[] h_new; delete[] c_new;
        delete[] g_observe; delete[] g_h_prev; delete[] g_c_prev;
    }

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

        char magic[5] = {0};
        int version;
        fread(magic, 1, 4, f);
        fread(&version, sizeof(int), 1, f);
        printf("Multi-step golden: magic=%.4s version=%d\n", magic, version);

        int dims[14];
        fread(dims, sizeof(int), 14, f);
        int ns = dims[13];
        if (ns != num_steps) {
            fprintf(stderr, "Expected %d steps, got %d\n", num_steps, ns);
            fclose(f); return false;
        }

        auto read_arr_f = [&](float* dst, int expected) {
            int count;
            fread(&count, sizeof(int), 1, f);
            if (count != expected) {
                fprintf(stderr, "Size mismatch: expected %d, got %d\n", expected, count);
            }
            fread(dst, sizeof(float), count, f);
        };

        auto read_arr_i = [&](int* dst, int expected) {
            int count;
            fread(&count, sizeof(int), 1, f);
            fread(dst, sizeof(int), count, f);
        };

        read_arr_f(weights, WEIGHT_FLOATS);

        auto alloc_step = [&](float*& ptr, int sz) { ptr = new float[sz]; };

        for (int t = 0; t < num_steps; t++) {
            alloc_step(observe[t], OBS_DIM * OBS_N); read_arr_f(observe[t], OBS_DIM * OBS_N);
            alloc_step(inner[t], INNER_DIM);          read_arr_f(inner[t], INNER_DIM);
            alloc_step(h_prev[t], D_H);               read_arr_f(h_prev[t], D_H);
            alloc_step(c_prev[t], D_H);               read_arr_f(c_prev[t], D_H);

            alloc_step(x_kv[t], OBS_DIM * OBS_N);     read_arr_f(x_kv[t], OBS_DIM * OBS_N);
            alloc_step(k[t], H_KV * D_E * OBS_N);     read_arr_f(k[t], H_KV * D_E * OBS_N);
            alloc_step(v[t], H_KV * D_E * OBS_N);     read_arr_f(v[t], H_KV * D_E * OBS_N);
            alloc_step(q[t], H_Q * D_E);              read_arr_f(q[t], H_Q * D_E);
            alloc_step(p[t], H_Q * H_KV * OBS_N);     read_arr_f(p[t], H_Q * H_KV * OBS_N);
            alloc_step(x[t], D_IN);                   read_arr_f(x[t], D_IN);
            alloc_step(h_prev_c[t], D_H);             read_arr_f(h_prev_c[t], D_H);
            alloc_step(c_prev_c[t], D_H);             read_arr_f(c_prev_c[t], D_H);
            alloc_step(gate_i[t], D_H);               read_arr_f(gate_i[t], D_H);
            alloc_step(gate_o[t], D_H);               read_arr_f(gate_o[t], D_H);
            alloc_step(c_alt[t], D_H);                read_arr_f(c_alt[t], D_H);
            alloc_step(tanhc[t], D_H);                read_arr_f(tanhc[t], D_H);
            alloc_step(lstm_o[t], D_H2);              read_arr_f(lstm_o[t], D_H2);
            alloc_step(act_cache[t], ACT_DIM);        read_arr_f(act_cache[t], ACT_DIM);
            alloc_step(c_h1[t], D_C);                 read_arr_f(c_h1[t], D_C);
            argmax_c_h1[t] = new int[D_C];            read_arr_i(argmax_c_h1[t], D_C);
            alloc_step(c_h2[t], D_C);                 read_arr_f(c_h2[t], D_C);
            alloc_step(c_h3[t], D_C);                 read_arr_f(c_h3[t], D_C);
        }

        for (int t = 0; t < num_steps; t++) {
            alloc_step(act_out[t], ACT_DIM);  read_arr_f(act_out[t], ACT_DIM);
            alloc_step(value_out[t], 1);      read_arr_f(value_out[t], 1);
            alloc_step(h_new[t], D_H);        read_arr_f(h_new[t], D_H);
            alloc_step(c_new[t], D_H);        read_arr_f(c_new[t], D_H);
        }

        read_arr_f(g_act, ACT_DIM);
        read_arr_f(g_value, 1);
        read_arr_f(grads, WEIGHT_FLOATS);

        for (int t = 0; t < num_steps; t++) {
            alloc_step(g_observe[t], OBS_DIM * OBS_N); read_arr_f(g_observe[t], OBS_DIM * OBS_N);
            alloc_step(g_h_prev[t], D_H);              read_arr_f(g_h_prev[t], D_H);
            alloc_step(g_c_prev[t], D_H);              read_arr_f(g_c_prev[t], D_H);
        }

        fclose(f);
        return true;
    }
};

// ================================================================
// 手动填充GPU缓存（绕过前向kernel，直接测试反向BPTT）
// ================================================================
template<typename Cfg>
void fill_cache_from_golden(agent_gpu::AttnLstmCache<Cfg>& cache,
                            const MultiStepGolden& g, int step) {
    memcpy(cache.x_kv,   g.x_kv[step],   OBS_DIM * OBS_N * sizeof(float));
    memcpy(cache.k,      g.k[step],      H_KV * D_E * OBS_N * sizeof(float));
    memcpy(cache.v,      g.v[step],      H_KV * D_E * OBS_N * sizeof(float));
    memcpy(cache.q,      g.q[step],      H_Q * D_E * sizeof(float));
    memcpy(cache.p,      g.p[step],      H_Q * H_KV * OBS_N * sizeof(float));
    memcpy(cache.x,      g.x[step],      D_IN * sizeof(float));
    memcpy(cache.h_prev, g.h_prev_c[step], D_H * sizeof(float));
    memcpy(cache.c_prev, g.c_prev_c[step], D_H * sizeof(float));
    memcpy(cache.gate_i, g.gate_i[step], D_H * sizeof(float));
    memcpy(cache.gate_o, g.gate_o[step], D_H * sizeof(float));
    memcpy(cache.c_alt,  g.c_alt[step],  D_H * sizeof(float));
    memcpy(cache.tanhc,  g.tanhc[step],  D_H * sizeof(float));
    memcpy(cache.lstm_o, g.lstm_o[step], D_H2 * sizeof(float));
    memcpy(cache.act,    g.act_cache[step], ACT_DIM * sizeof(float));
    memcpy(cache.c_h1,   g.c_h1[step],   D_C * sizeof(float));
    memcpy(cache.argmax_c_h1, g.argmax_c_h1[step], D_C * sizeof(int));
    memcpy(cache.c_h2,   g.c_h2[step],   D_C * sizeof(float));
    memcpy(cache.c_h3,   g.c_h3[step],   D_C * sizeof(float));
}

// ================================================================
// Main test
// ================================================================
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr, "Starting multi-step BPTT test...\n");

    cudaError_t init_err = cudaFree(0);
    if (init_err != cudaSuccess) return 1;

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Load golden (num_steps from file)
    // Run: testAttnLstmMulti [golden_file_path]
    const char* golden_path = "golden_attnlstm_multi.bin";

    // Read num_steps from file header
    int NUM_STEPS = 2;
    {
        FILE* f = fopen(golden_path, "rb");
        if (!f) { fprintf(stderr, "FATAL: Cannot open %s\n", golden_path); return 1; }
        fseek(f, 8, SEEK_SET);  // skip magic(4) + version(4)
        int dims[14];
        fread(dims, sizeof(int), 14, f);
        NUM_STEPS = dims[13];  // last element is num_steps
        fclose(f);
        printf("Golden file: %d steps\n", NUM_STEPS);
    }

    MultiStepGolden golden(NUM_STEPS);
    if (!golden.load(golden_path)) {
        fprintf(stderr, "FATAL: Cannot load golden data.\n");
        return 1;
    }
    printf("Loaded %d-step golden data\n\n", NUM_STEPS);

    constexpr int NUM_NETS = 1;
    agent_gpu::AttnLstmHandle<Cfg> handle;
    handle.alloc(NUM_NETS, NUM_STEPS);

    // Upload weights
    agent_gpu::AttnLstmWeights<Cfg>* h_w = new agent_gpu::AttnLstmWeights<Cfg>[NUM_NETS];
    memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
    memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
    handle.copy_weights_from_host(h_w);

    // ================================================================
    // Test 1: Multi-step forward (state advancing)
    // ================================================================
    printf("--- Test 1: Multi-step Forward (%d steps) ---\n", NUM_STEPS);

    // Init state to golden h_prev[0], c_prev[0] (= zeros)
    agent_gpu::AttnLstmPersistent<Cfg>* h_p = new agent_gpu::AttnLstmPersistent<Cfg>[NUM_NETS];
    memcpy(h_p[0].h, golden.h_prev[0], D_H * sizeof(float));
    memcpy(h_p[0].c, golden.c_prev[0], D_H * sizeof(float));
    handle.copy_persistent_from_host(h_p);

    bool all_fwd_ok = true;
    for (int t = 0; t < NUM_STEPS; t++) {
        // Upload observe and inner for this step
        CUDA_CHECK(cudaMemcpy(handle.d_observe, golden.observe[t],
                              OBS_DIM * OBS_N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(handle.d_inner, golden.inner[t],
                              INNER_DIM * sizeof(float), cudaMemcpyHostToDevice));

        // Forward
        fprintf(stderr, "  Forward step %d...\n", t);
        handle.forward(t);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Advance state for next step
        if (t < NUM_STEPS - 1) {
            handle.advance_state();
        }

        // Read outputs
        float h_act[ACT_DIM], h_value[1];
        CUDA_CHECK(cudaMemcpy(h_act, handle.d_act, ACT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_value, handle.d_value, 1 * sizeof(float), cudaMemcpyDeviceToHost));

        agent_gpu::AttnLstmPersistent<Cfg> h_p_out[NUM_NETS];
        handle.copy_persistent_to_host(h_p_out);

        char label[64];
        snprintf(label, sizeof(label), "Step %d act", t);
        bool ok_act = allclose(h_act, golden.act_out[t], ACT_DIM, 1e-4f, 1e-5f, label);
        snprintf(label, sizeof(label), "Step %d value", t);
        bool ok_val = allclose(h_value, golden.value_out[t], 1, 1e-4f, 1e-5f, label);
        snprintf(label, sizeof(label), "Step %d h_new", t);
        bool ok_h   = allclose(h_p_out[0].h, golden.h_new[t], D_H, 1e-4f, 1e-5f, label);
        snprintf(label, sizeof(label), "Step %d c_new", t);
        bool ok_c   = allclose(h_p_out[0].c, golden.c_new[t], D_H, 1e-4f, 1e-5f, label);

        if (!ok_act || !ok_val || !ok_h || !ok_c) {
            printf("  Step %d: GPU act=[%.6f,%.6f] golden=[%.6f,%.6f] val=%.6f/%.6f\n",
                   t, h_act[0], h_act[1], golden.act_out[t][0], golden.act_out[t][1],
                   h_value[0], golden.value_out[t][0]);
            all_fwd_ok = false;
        }
    }
    printf("  Multi-step Forward: %s\n\n", all_fwd_ok ? "PASS" : "FAIL");

    // ================================================================
    // Test 2: Multi-step backward (BPTT) — use forward-generated caches
    // ================================================================
    printf("--- Test 2: Multi-step Backward (BPTT, %d steps) ---\n", NUM_STEPS);

    // Use caches from Test 1 forward passes (already on GPU in d_caches).
    // Re-load weights and zero gradients for a clean backward pass.
    memset(h_w, 0, sizeof(agent_gpu::AttnLstmWeights<Cfg>));
    memcpy(h_w, golden.weights, WEIGHT_FLOATS * sizeof(float));
    handle.copy_weights_from_host(h_w);
    handle.zero_gradients();

    // Upload upstream gradients (only last step gets g_act/g_value)
    CUDA_CHECK(cudaMemcpy(handle.d_grad_act, golden.g_act, ACT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(handle.d_grad_value, golden.g_value, 1 * sizeof(float), cudaMemcpyHostToDevice));

    // Run BPTT backward (reads from d_caches populated by forward kernel)
    fprintf(stderr, "  Launching backward kernel (num_steps=%d)...\n", NUM_STEPS);
    handle.backward(NUM_STEPS);
    handle.sync();
    fprintf(stderr, "  Backward kernel completed.\n");

    // Read gradients
    agent_gpu::AttnLstmWeights<Cfg> h_w_gpu;
    handle.copy_weights_to_host(&h_w_gpu);
    const float* gpu_grads = h_w_gpu.grad_Wkv;

    // Offsets
    constexpr int SZ_WKV  = 2*H_KV*D_E*OBS_DIM; constexpr int SZ_WQ  = H_Q*D_E*D_H1;
    constexpr int SZ_BCAT = H_Q*D_E;              constexpr int SZ_WICO= 3*D_H*D_IN;
    constexpr int SZ_BICO = 3*D_H;                constexpr int SZ_WF  = D_H2*ACT_DIM;
    constexpr int SZ_BF   = ACT_DIM;              constexpr int SZ_WC1 = D_C*OBS_DIM;
    constexpr int SZ_BC1  = D_C;                  constexpr int SZ_WC2 = D_C*(D_C+INNER_DIM);
    constexpr int SZ_BC2  = D_C;                  constexpr int SZ_WC3 = D_C*D_C;
    constexpr int SZ_BC3  = D_C;                  constexpr int SZ_WC4 = D_C;
    constexpr int SZ_BC4  = 1;
    constexpr int OFF_WKV=0, OFF_WQ=SZ_WKV, OFF_BCAT=OFF_WQ+SZ_WQ;
    constexpr int OFF_WICO=OFF_BCAT+SZ_BCAT, OFF_BICO=OFF_WICO+SZ_WICO;
    constexpr int OFF_WF=OFF_BICO+SZ_BICO, OFF_BF=OFF_WF+SZ_WF;
    constexpr int OFF_WC1=OFF_BF+SZ_BF, OFF_BC1=OFF_WC1+SZ_WC1;
    constexpr int OFF_WC2=OFF_BC1+SZ_BC1, OFF_BC2=OFF_WC2+SZ_WC2;
    constexpr int OFF_WC3=OFF_BC2+SZ_BC2, OFF_BC3=OFF_WC3+SZ_WC3;
    constexpr int OFF_WC4=OFF_BC3+SZ_BC3, OFF_BC4=OFF_WC4+SZ_WC4;

    bool all_bwd_ok = true;
    #define CHK(n,o,s) { bool ok=allclose(gpu_grads+(o),golden.grads+(o),(s),5e-2f,5e-4f,"grad_"#n); all_bwd_ok=all_bwd_ok&&ok; }
    CHK(Wkv,  OFF_WKV,  SZ_WKV);  CHK(Wq,   OFF_WQ,   SZ_WQ);
    CHK(bcat, OFF_BCAT, SZ_BCAT); CHK(Wico, OFF_WICO, SZ_WICO);
    CHK(bico, OFF_BICO, SZ_BICO); CHK(Wf,   OFF_WF,   SZ_WF);
    CHK(bf,   OFF_BF,   SZ_BF);   CHK(Wc1,  OFF_WC1,  SZ_WC1);
    CHK(bc1,  OFF_BC1,  SZ_BC1);  CHK(Wc2,  OFF_WC2,  SZ_WC2);
    CHK(bc2,  OFF_BC2,  SZ_BC2);  CHK(Wc3,  OFF_WC3,  SZ_WC3);
    CHK(bc3,  OFF_BC3,  SZ_BC3);  CHK(Wc4,  OFF_WC4,  SZ_WC4);
    CHK(bc4,  OFF_BC4,  SZ_BC4);
    #undef CHK
    printf("  Multi-step Backward overall: %s\n\n", all_bwd_ok ? "PASS" : "FAIL");

    // ================================================================
    // Test 3: Compare forward cache h_prev/c_prev across steps
    // ================================================================
    printf("--- Test 3: State consistency across steps ---\n");
    {
        // State at step 0: h_prev=0, c_prev=0 (init)
        // State at step 1: h_prev = h_new_0, c_prev = c_new_0
        printf("  Step 0: h_prev=[%.6f ...]  c_prev=[%.6f ...]\n",
               golden.h_prev[0][0], golden.c_prev[0][0]);
        printf("  Step 1: h_prev=[%.6f ...]  c_prev=[%.6f ...]\n",
               golden.h_prev[1][0], golden.c_prev[1][0]);
        printf("  Step 0 h_new=[%.6f ...] (should feed to step1 h_prev): %s\n",
               golden.h_new[0][0],
               fabsf(golden.h_new[0][0] - golden.h_prev[1][0]) < 1e-6f ? "OK" : "MISMATCH");
    }

    // Cleanup
    delete[] h_w; delete[] h_p;
    handle.free();

    bool overall = all_fwd_ok && all_bwd_ok;
    printf("=== Overall: %s ===\n", overall ? "PASS" : "FAIL");
    return overall ? 0 : 1;
}
