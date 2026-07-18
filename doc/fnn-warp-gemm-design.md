# 单Block前馈神经网络：基于CUTLASS的Warp级GEMM实现

> 项目：ecoSim — 生态仿真中的RL Agent在线学习
> 日期：2026-06-29
> 对应代码：`test/test-cutlass/src/`

---

## 一、项目背景与需求

ecoSim 中的每个 Agent 拥有独立的神经网络（前馈，2隐藏层），需要在 GPU 上同时运行大量网络实例进行推理和在线学习。核心约束：

| 需求 | 实现 |
|------|------|
| 每个网络在**单个 CUDA Block** 内执行 | block = 4 warps × 32 threads = 128 threads |
| 大量网络**同时并行** | grid = `num_networks`，每 block 对应一个网络 |
| 矩阵乘法使用 **Warp 级 GEMM** | 32 个 lane 协作完成单个输出行的点积 |
| 激活函数在**同一 Block** 内完成 | sigmoid/tanh/relu 以 cooperative 方式并行 |
| **共享内存缓存**数据复用 | 层间激活值通过 smem 传递，避免 global memory 往返 |
| 层内**并行**（warp 协作），层间**串行**（`__syncthreads`） | 每层 GEMM 完成后同步，再进入下一层 |
| 前向 + 反向传播 | 两个独立 kernel，中间激活值缓存在 global memory |

---

## 二、网络结构

```
Input(32) → Linear(W1:64×32, b1:64) → Sigmoid
         → Linear(W2:64×64, b2:64) → Sigmoid
         → Linear(W3:4×64,  b3:4)  → Identity → Output(4)
```

所有维度由 `FNNConfig` 模板参数控制：

```cpp
// fnn_config.cuh
template <int IN_DIM_=32, int H1_DIM_=64, int H2_DIM_=64, int OUT_DIM_=4,
          int BATCH_=1, int BLOCK_DIM_=128>
struct FNNConfig {
    static constexpr int IN_DIM   = IN_DIM_;
    static constexpr int H1_DIM   = H1_DIM_;
    static constexpr int H2_DIM   = H2_DIM_;
    static constexpr int OUT_DIM  = OUT_DIM_;
    static constexpr int BATCH    = BATCH_;
    static constexpr int BLOCK_DIM = BLOCK_DIM_;
    static constexpr int WARPS    = BLOCK_DIM / 32;  // 4
    // 每 warp 负责的输出行数（向上取整）
    static constexpr int H1_ROWS_PER_WARP = (H1_DIM + WARPS - 1) / WARPS;  // 16
    // ...同理 H2_ROWS_PER_WARP、OUT_ROWS_PER_WARP
};
```

---

## 三、文件结构

```
test/test-cutlass/
├── CMakeLists.txt              # 构建配置，FetchContent 获取 CUTLASS v4.5.1
├── src/
│   ├── fnn_config.cuh          # 编译期配置（模板参数、Phase枚举、激活类型枚举）
│   ├── fnn_activations.cuh     # 激活函数（Sigmoid/Tanh/ReLU/Identity）及 cooperative apply
│   ├── fnn_warp_gemm.cuh       # Warp 级 tiled GEMM（前向/反向输入/反向权重）
│   ├── fnn_kernel.cuh          # 前向和反向 CUDA kernel（每 block 一个网络）
│   ├── fnn_host.cuh            # 宿主端管理器 FNNHandle（内存/初始化/启动/SGD）
│   └── fnn_test.cu             # 正确性验证 + 多网络测试 + 性能基准 + 在线学习测试
└── tools/
    └── generate_golden.py      # Python NumPy 参考实现，生成 golden 数据
```

### CMakeLists.txt 要点

```cmake
project(testCutlass LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

# 复用 external/cutlass-src 本地缓存
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/external)
set(FETCHCONTENT_FULLY_DISCONNECTED ON)
FetchContent_Declare(CUTLASS GIT_REPOSITORY ... GIT_TAG v4.5.1 ...)
FetchContent_MakeAvailable(CUTLASS)   # 提供 CUTLASS 头文件和 include 路径

add_executable(${PROJECT_NAME} src/fnn_test.cu)
target_compile_features(... cxx_std_17)
target_compile_options(... $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
target_link_libraries(... CUDA::cublas CUDA::cufft CUDA::cudart CUTLASS)
```

---

## 四、数据布局

### 4.1 Global Memory — 网络权重

```cpp
// fnn_kernel.cuh
template <typename Config>
struct alignas(16) NetworkWeights {
    // 权重（行主序）
    float W1[H1 * IN];   float b1[H1];
    float W2[H2 * H1];   float b2[H2];
    float W3[OUT * H2];  float b3[OUT];

    // 梯度累积器
    float grad_W1[H1 * IN];   float grad_b1[H1];
    float grad_W2[H2 * H1];   float grad_b2[H2];
    float grad_W3[OUT * H2];  float grad_b3[OUT];
};
```

- 所有矩阵**行主序**存储，与 NumPy/C++ 默认一致
- 每网络的 `NetworkWeights` 在 global memory 中连续存储为数组
- Kernel 通过 `d_weights[blockIdx.x]` 索引当前网络的权重
- 对齐到 16 字节，保证 coalesced 访问

### 4.2 Global Memory — 激活值缓存

```cpp
template <typename Config>
struct alignas(16) ActivationCache {
    // 布局：[input_x | h1_out | h2_out]
    //        IN*B     H1*B    H2*B
    char data[(IN + H1 + H2) * BATCH * sizeof(float)];
};
```

前向 kernel 将中间激活值写入此缓存，反向 kernel 读取。内存开销极小（每网络约 640 bytes）。

### 4.3 共享内存布局

```
┌───────────┬───────────┐
│  buf_a    │  buf_b    │
│  BUF_A_SIZE │ BUF_B_SIZE │
│  (64 float) │ (64 float) │
└───────────┴───────────┘
  smem[0]   smem[64]   smem[128]
```

- `buf_a`：当前层的**输入**（尺寸 = `max(IN, H1, H2) × BATCH`）
- `buf_b`：当前层的**输出**（尺寸 = `max(H1, H2, OUT) × BATCH`）
- 总大小 = 128 floats = 512 bytes，远小于 48KB 限制
- 层间通过将 buf_b 拷贝到 buf_a 完成数据传递

---

## 五、Warp 级 GEMM 算法

### 5.1 层内并行策略

对于矩阵乘法 `C[M×N] = W[M×K] @ X[K×N]`：

```
┌──────────────────────────────────────────────┐
│  W (64×32)                    X (32×1)        │
│  ┌─────┬──────┐               ┌──┐           │
│  │Warp0│ 0-15 │               │x0│           │
│  ├─────┼──────┤               │x1│           │
│  │Warp1│16-31 │    @          │..│    →      │
│  ├─────┼──────┤               │x31│          │
│  │Warp2│32-47 │               └──┘           │
│  ├─────┼──────┤                              │
│  │Warp3│48-63 │                              │
│  └─────┴──────┘                              │
│                                              │
│  C (64×1)                                    │
│  ┌─────┐                                     │
│  │ h0  │ ← Warp0                             │
│  │ ... │                                     │
│  │h15  │                                     │
│  │h16  │ ← Warp1                             │
│  │ ... │                                     │
│  │h63  │ ← Warp3                             │
│  └─────┘                                     │
└──────────────────────────────────────────────┘
```

- M（输出行）按 warp 分组：4 个 warp 各处理约 M/4 行
- **Warp 内部**：32 个 lane **串行处理**各行，每行所有 32 lane **协作**完成点积
- K 维度以 stride-32 分片，每个 lane 读取不同的 K 片段

### 5.2 前向 GEMM 核心算法

```cpp
// fnn_warp_gemm.cuh — warp_gemm_forward
// C[M,N] = W[M,K] @ X[K,N]

int rows_per_warp = (M + WARPS - 1) / WARPS;
int row_start = warp_id * rows_per_warp;
int row_end   = min(row_start + rows_per_warp, M);

// 串行遍历该 warp 分配的行
for (int m = row_start; m < row_end; ++m) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        // 32 个 lane 并行读取 W 的不同列（coalesced）
        for (int k = lane; k < K; k += 32) {
            acc += W_gmem[m * K + k] * X_smem[k * N + n];
        }
        // ALL 32 lanes must participate in shuffle!
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            C_smem[m * N + n] = acc;
        }
    }
}
```

**关键设计点**：
1. W 的读取是 **coalesced**：lane L 读 W[m, L], W[m, L+32], ...（同一行内连续地址）
2. X 在 smem 中被所有 warp 复用——只需加载一次
3. **`warp_reduce_sum` 必须所有 32 lane 参与**，否则导致 shuffle 死锁

### 5.3 Warp 级 Reduction

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 所有 lane 得到相同结果
}
```

- 使用 `__shfl_xor_sync` 蝴蝶归约
- 掩码 `0xFFFFFFFF` 表示 warp 内全部 32 lane 参与
- 归约后**所有 lane** 持有相同的总和，仅 lane 0 写回

### 5.4 反向 GEMM — 输入梯度

```cpp
// GX[K,N] = W[M,K]^T @ GY[M,N]
// 线程粒度的列归约（无 warp shuffle）
for (int idx = tid; idx < K * N; idx += blockDim.x) {
    int k = idx / N, n = idx % N;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += W_gmem[m * K + k] * GY_smem[m * N + n];
    }
    GX_smem[idx] = acc;
}
```

- 由于转置后每个 (k,n) 独立，直接用线程粒度并行（无需 warp 协作）
- 全部 128 线程分配 (K × N) 个输出元素
- W 按列读取：线程 i 读 W[:, k]，其中相邻线程的 k 相邻 → coalesced

### 5.5 反向 GEMM — 权重梯度

```cpp
// GW[M,K] += GY[M,N] @ X[K,N]^T
// 也是线程粒度独立，直接用 atomic-free 累加
for (int idx = tid; idx < M * K; idx += blockDim.x) {
    int m = idx / K, k = idx % K;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        acc += GY_smem[m * N + n] * X_smem[k * N + n];
    }
    GW_gmem[idx] += acc;   // 每个 (m,k) 只有一个线程写，无需原子操作
}
```

---

## 六、激活函数

```cpp
// fnn_activations.cuh

// 定义：每个激活函数提供 fwd(x) 和 bwd(y, grad) 静态方法
struct Sigmoid {
    static __device__ float fwd(float x) { return 1.0f / (1.0f + __expf(-x)); }
    static __device__ float bwd(float y, float grad) { return y * (1.0f - y) * grad; }
    // bwd 使用前向输出 y 而非原始输入 x（避免重复计算 exp）
};

// Cooperative apply：所有线程共同处理一个数组
template <typename ActFn>
__device__ void activation_apply_fwd(float* data, int count) {
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        data[i] = ActFn::fwd(data[i]);
}
```

| 激活函数 | 前向 | 反向导数 |
|----------|------|----------|
| Sigmoid | `1/(1+e^{-x})` | `y(1-y)` |
| Tanh | `tanh(x)` | `1-y²` |
| ReLU | `max(0,x)` | `x>0?1:0` |
| Identity | `x` | `1` |

---

## 七、Kernel 执行流程

### 7.1 前向 Kernel

```
__global__ fnn_forward_kernel
├─ 加载输入 x → smem(buf_a)
│  └─ 缓存 x 到 ActivationCache（供反向使用）
├─ [Layer 1] W1 @ x → buf_b
│  ├─ warp_gemm_forward(W1, buf_a, buf_b, M=64, K=32, N=1)
│  ├─ __syncthreads()
│  ├─ add_bias(buf_b, b1)
│  ├─ activation_apply_fwd<Sigmoid>(buf_b) → h1
│  └─ 缓存 h1 到 ActivationCache
├─ [Layer 2] W2 @ h1 → buf_b
│  ├─ 拷贝 h1: buf_b → buf_a
│  ├─ warp_gemm_forward(W2, buf_a, buf_b, M=64, K=64, N=1)
│  ├─ __syncthreads()
│  ├─ add_bias(buf_b, b2)
│  ├─ activation_apply_fwd<Sigmoid>(buf_b) → h2
│  └─ 缓存 h2 到 ActivationCache
├─ [Layer 3] W3 @ h2 → buf_b
│  ├─ 拷贝 h2: buf_b → buf_a
│  ├─ warp_gemm_forward(W3, buf_a, buf_b, M=4, K=64, N=1)
│  └─ add_bias(buf_b, b3)  [Identity activation = no-op]
└─ 写输出 buf_b → d_outputs
```

### 7.2 反向 Kernel

```
__global__ fnn_backward_kernel
├─ 加载 dL/dy → smem(buf_b)
├─ 从 ActivationCache 加载 h2 → buf_a
├─ [Layer 3 backward, linear]
│  ├─ warp_gemm_backward_weight(GY=buf_b, X=buf_a, GW=grad_W3)
│  ├─ accumulate_bias_grad(buf_b, grad_b3)
│  ├─ warp_gemm_backward_input(W3, buf_b, buf_a) → G_h2
│  └─ __syncthreads()
├─ [Layer 2 backward, sigmoid]
│  ├─ activation_apply_bwd<Sigmoid>(h2_cache, buf_a)  // G_h2 *= sigmoid'(h2)
│  ├─ 加载 h1 → buf_b
│  ├─ warp_gemm_backward_weight(GY=buf_a, X=buf_b, GW=grad_W2)
│  ├─ accumulate_bias_grad(buf_a, grad_b2)
│  ├─ warp_gemm_backward_input(W2, buf_a, buf_b) → G_h1
│  └─ __syncthreads()
├─ [Layer 1 backward, sigmoid]
│  ├─ activation_apply_bwd<Sigmoid>(h1_cache, buf_b)
│  ├─ 加载 x → buf_a
│  ├─ warp_gemm_backward_weight(GY=buf_b, X=buf_a, GW=grad_W1)
│  ├─ accumulate_bias_grad(buf_b, grad_b1)
│  └─ warp_gemm_backward_input(W1, buf_b, buf_a) → G_x
└─ 写 G_x → d_grad_inputs
```

---

## 八、宿主端接口

```cpp
// fnn_host.cuh
template <typename Config = DefaultConfig>
struct FNNHandle {
    int num_networks;
    NetworkWeights<Config>*  d_weights;  // [num_networks]
    float* d_inputs;    // [num_networks][IN][BATCH]
    float* d_outputs;   // [num_networks][OUT][BATCH]
    float* d_loss_grad; // [num_networks][OUT][BATCH]
    float* d_grad_inputs; // [num_networks][IN][BATCH]
    ActivationCache<Config>* d_cache; // 中间激活值

    void alloc(int num_nets);              // 分配 GPU 内存
    void init_weights_xavier(uint seed);   // Xavier 初始化
    void zero_gradients();                  // 清零梯度
    void forward();                         // 启动 fnn_forward_kernel
    void backward();                        // 启动 fnn_backward_kernel
    void sync();                            // cudaStreamSynchronize
    void apply_gradients_sgd(float lr);    // 用累积的梯度更新权重
};
```

### 启动配置

```cpp
dim3 block(128);                            // 4 warps
dim3 grid(num_networks);                    // 每个网络一个 block
size_t smem = 128 * sizeof(float);          // 512 bytes 动态共享内存
fnn_forward_kernel<<<grid, block, smem, stream>>>(...);
```

---

## 九、CUTLASS 使用说明

### 9.1 本项目如何"使用" CUTLASS

本项目通过以下方式使用 CUTLASS：

| 层面 | CUTLASS 使用 |
|------|-------------|
| **构建系统** | CMake `FetchContent` 获取 CUTLASS v4.5.1，链接 `CUTLASS` 目标 |
| **设计模式** | 遵循 CUTLASS 的 warp 级 tiling 策略、共享内存分期（staging）、cooperative reduction |
| **头文件** | 不能直接 `#include <cutlass/cutlass.h>`（见 9.2 节），但编译包含路径已配置 |

### 9.2 为什么不能直接 include CUTLASS 头文件

**问题**：本项目使用的 MSVC 2019 (v14.27) 不支持 C++20 的 `auto` 非类型模板参数（NTTP），而 CUTLASS 4.5.1 的 CuTe 子库（`cute/numeric/integral_constant.hpp`）大量使用了该特性。

**现象**：
```
error C2171: '+'：'auto' 类型的操作数非法
error C2296: '*'：非法，左操作数包含 'auto' 类型
```

**解决方案**：
- Kernel 代码中移除 `#include <cutlass/cutlass.h>` 等直接引用
- 自定义的 warp 级 GEMM 使用原生 CUDA 原语（`__shfl_xor_sync`、`__syncthreads`）实现，遵循 CUTLASS 设计模式
- 如需使用 CUTLASS 完整模板接口（如 `MmaSimt`），需要 MSVC 2022 / GCC 11+ / Clang 14+

---

## 十、实践中的关键 Bug 及修复

### Bug 1：Warp Shuffle 死锁（致命）

**症状**：kernel 启动后 GPU 挂死，`cudaStreamSynchronize` 永不返回，TDR 触发

**根因**：`__shfl_xor_sync` 调用被包裹在 `if (m < row_end)` 条件内：

```cpp
// 错误：只有部分 lane 调用 shuffle
for (int m_base = row_start; m_base < row_end; m_base += 32) {
    int m = m_base + lane;
    if (m < row_end) {                               // ← 部分 lane 跳过
        for (int n = 0; n < N; ++n) {
            float acc = ...;
            acc = warp_reduce_sum(acc);              // ← 死锁！
        }
    }
}
```

**修复**：改为串行处理每行，所有 32 lane 始终参与归约：

```cpp
// 正确：所有 lane 都参与每个归约
for (int m = row_start; m < row_end; ++m) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32) {
            acc += W_gmem[m * K + k] * X_smem[k * N + n];
        }
        acc = warp_reduce_sum(acc);                  // ← 32 lane 都调用
        if (lane == 0) C_smem[m * N + n] = acc;
    }
}
```

### Bug 2：归约跨行混合（正确性错误）

**症状**：forward 输出与 CPU 参考偏差高达 92.8%

**根因**：每个 lane 计算**不同行**的点积，然后全部归约在一起：

```
Lane0: dot(W[0,:], X) → 归约
Lane1: dot(W[1,:], X) →   ↓
...                        ↓  所有行的和混合在一起
Lane15: dot(W[15,:], X) → ↓
```

**修复**：见 Bug 1 的修复——改为串行处理行，一行一归约，与死锁修复一起解决。

### Bug 3：程序启动缓慢/挂死

**症状**：`testCutlass.exe` 启动后 stdout 无输出

**根因**：MSVC 默认 stdout 全缓冲，`printf` 输出在 pipe 到文件时不会立即刷新

**修复**：
```cpp
setvbuf(stdout, NULL, _IONBF, 0);  // 禁用缓冲，直接输出
```

以及与 `stderr`（默认无缓冲）交叉使用进行调试输出。

---

## 十一、性能数据

**测试环境**：NVIDIA GeForce RTX 2060 (SM 7.5, 6GB VRAM), CUDA 13.1, MSVC 2019

| 网络数 | 前向 (μs) | 反向 (μs) | 总耗时 (μs) | 吞吐量 (nets/s) |
|--------|-----------|-----------|-------------|-----------------|
| 1      | 23.4      | 39.0      | 62.4        | 16,034          |
| 100    | 30.9      | 87.1      | 118.0       | 847,412         |
| 1,000  | 173.0     | 642.2     | 815.2       | 1,226,727       |
| 5,000  | 752.5     | 3,028.7   | 3,781.2     | **1,322,339**   |

**分析**：
- 吞吐量随网络数增加而提升（更好地利用 GPU 并行度），在 5,000 网络时达到 **1.32M nets/s**
- 反向比前向慢约 4×（权重梯度累积需要更多 global memory 写入）
- 单网络 62 μs 适合实时在线学习（~16K 步/秒）

### 正确性验证

| 测试项 | 最大相对误差 | 结果 |
|--------|-------------|------|
| 前向输出 | 1.82×10⁻⁷ | PASS |
| 反向输入梯度 | 5.09×10⁻⁶ | PASS |
| 权重梯度 W1/W2/W3 | 2.27×10⁻⁵ | PASS |
| 偏置梯度 b1/b2/b3 | 2.26×10⁻⁵ | PASS |
| 多网络独立性 (4 nets) | — | 4/4 correct |
| 在线学习 (100步) | loss: 0.73→0.005 | PASS |

---

## 十二、内存占用

| 每网络 | 大小 |
|--------|------|
| NetworkWeights（权重+梯度） | ~52 KB |
| 输入/输出/梯度 | ~144 bytes |
| ActivationCache | ~640 bytes |
| **合计** | ~53 KB |

5000 网络总计 ≈ 265 MB（global memory），RTX 2060 6GB 容量充裕。

---

## 十三、构建与运行

### 构建

```bash
# Windows: 使用 VS 2019 生成器
cmake -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --target testCutlass --config Release -- -m
```

### 运行测试

```bash
./build/test/test-cutlass/Release/testCutlass.exe
```

### 生成 Golden 数据（可选，用于独立验证）

```bash
cd test/test-cutlass/tools
python generate_golden.py
```

---

## 十四、扩展方向

1. **支持更大 batch size**：当前 `BATCH=1`，可扩展至 BATCH≤8（仍在 smem 容量内）以支持 mini-batch SGD
2. **Tensor Core 支持**：使用 fp16 权重 + `MmaTensorOp`（需要 MSVC 2022 以支持 CUTLASS CuTe）
3. **LSTM/Attention**：参考 `python/agent_numpy.py` 的 GQA 和 LSTM 实现，在 warp 级 tiling 框架下扩展
4. **动态维度**：当前维度为编译期常量，可增加运行时维度分支（template switch）
5. **流水线**：在层间使用 double-buffered shared memory 隐藏 global memory 延迟

---

## 十五、关键代码索引

| 功能 | 文件 | 行号概览 |
|------|------|----------|
| FNNConfig 模板 | `fnn_config.cuh` | 全文 |
| Sigmoid/ReLU/Tanh | `fnn_activations.cuh` | 全文 |
| warp_reduce_sum | `fnn_warp_gemm.cuh` | L42-L49 |
| warp_gemm_forward | `fnn_warp_gemm.cuh` | L57-L78 |
| warp_gemm_backward_input | `fnn_warp_gemm.cuh` | L88-L110 |
| warp_gemm_backward_weight | `fnn_warp_gemm.cuh` | L115-L132 |
| forward kernel | `fnn_kernel.cuh` | L99-L195 |
| backward kernel | `fnn_kernel.cuh` | L200-L350 |
| FNNHandle | `fnn_host.cuh` | 全文 |
| host test harness | `fnn_test.cu` | main() |
| CPU reference (forward) | `fnn_test.cu` | cpu_forward() |
| CPU reference (backward) | `fnn_test.cu` | cpu_backward() |
