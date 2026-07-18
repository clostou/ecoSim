# CUTLASS

## 基本概念

在 CUDA 领域，‌**MMA**‌ 是 ‌**Matrix Multiply-Accumulate**‌（矩阵乘加）的缩写，指代利用 GPU ‌**Tensor Core**‌ 执行的高效底层指令，用于加速形如 $D=A×B+C$ 的运算。‌‌

### 核心定义

- ‌**含义**‌：MMA 指令是 PTX（Parallel Thread Execution）汇编层面的原语，直接映射到硬件 Tensor Core 单元，执行定点或浮点矩阵的乘累加操作。
- ‌**执行单元**‌：以 ‌**Warp**‌（线程束，通常 32 个线程）为最小执行单位，要求 Warp 内所有线程同步协作，无分支发散。
- ‌**应用场景**‌：主要用于深度学习推理/训练中的 GEMM（通用矩阵乘法）、卷积等算子加速，是 CUTLASS 等高性能库的核心构建块。‌‌

### 与 WMMA 的区别

- ‌**MMA (PTX)**‌：底层汇编接口，灵活性极高，需手动管理寄存器布局和线程协作，性能上限高但开发复杂。
- ‌**WMMA (C++ API)**‌：NVIDIA 提供的高层 C++ 接口（`#include <cuda_fp16.h>` 等），封装了寄存器细节，开发简便但优化空间受限。‌‌

简言之，MMA 是 CUDA 中调用 Tensor Core 进行‌**硬件级矩阵加速**‌的最底层编程接口。

### NVIDIA GPU 架构

| 架构名称  | 计算能力         | 关键技术                                            | 性能特性                             | 代表产品                                                     | 应用场景                      |
| --------- | ---------------- | --------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ | ----------------------------- |
| Pascal    | sm60、sm61、sm62 | CUDA 核心优化、GDDR5X/HBM2 显存、NVLink             | 性能与能效提升、支持 VR              | GeForce GTX 10 系列、Quadro P 系列、Tesla P 系列             | 游戏、VR 开发、初级 AI        |
| Volta     | sm70、sm72       | Tensor Core、HBM2 显存、NVLink 2.0                  | AI 加速、FP16/INT8 运算优化          | Titan V、Tesla V100                                          | 深度学习、HPC                 |
| Turing    | sm75             | RT Core、第二代 Tensor Core、DLSS                   | 实时光线追踪、混合渲染               | GeForce RTX 20 系列、Quadro RTX 系列、Tesla T4               | 游戏、视觉效果制作、AI 推理   |
| Ampere    | sm80、sm86、sm87 | 第三代 Tensor Core、第二代 RT Core、MIG、PCIe Gen 4 | 高效 AI 和光线追踪性能、稀疏矩阵运算 | GeForce RTX 30 系列、NVIDIA A 系列、A100                     | 游戏、AI 训练和推理、数据中心 |
| Ada       | sm89             | 第四代 Tensor Core、第三代 RT Core、DLSS 3          | 极致光线追踪、高效 AI 加速           | GeForce RTX 40 系列、L40                                     | 高端游戏、内容创作、AI 推理   |
| Hopper    | sm90、sm90a      | Transformer Engine、第四代 NVLink、HBM3 显存        | 针对大模型优化、更高互联带宽         | H100                                                         | 大规模 AI、科学计算           |
| Blackwell | sm95             | 第五代 Tensor Core、第四代 RT Core、新一代显存      | 更强 AI 和光线追踪性能、更高能效比   | GeForce RTX 50 系列（预计）、B 系列（预计）、下一代数据中心 GPU（预计） | 下一代游戏、高级 AI、数据中心 |



## 文件结构

```
include/                     # Top-level include directory. Client applications should target this path.
  cutlass/                   # CUDA Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)
      *
    gemm/                    # code specialized for general matrix product computations
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      collective/            #   3.x API operators for all threads a tiled mma/copy are built over
      threadblock/           #   CTA-level operators
      kernel/                #   CUDA kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions for GEMM

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory
      *

    reduction/               # bandwidth-limited reduction kernels that do not fit the "gemm" models
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      threadblock/           #   CTA-level operators
      kernel/                #   CUDA kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions

    transform/               # code specialized for layout, type, and domain transformations
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      threadblock/           #   CTA-level operators
      kernel/                #   CUDA kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions

    util/                    # miscellaneous CUTLASS components
      *
    *                        # core vocabulary types and fundamental arithmetic operators

  cute /                     # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy
    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples
    arch/                    # Bare bones PTX wrapper structs for copy and math instructions
    atom/                    # Meta-information either link to or built from arch/ operators
      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma
      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy
      *sm*.hpp               # Arch specific meta-information for copy and math operations
    container/               # Core container types used across CuTe, namely, cute::tuple
    numeric/                 # CuTe's internal numerics implementation
    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations
```

