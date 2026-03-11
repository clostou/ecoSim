#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief CUDA核函数：每个线程打印自己的全局编号
 * 设计思路：
 * 1. 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
 * 2. 严格分配20个线程（block=20，grid=1），确保总线程数精准为20
 * 3. 打印格式清晰，包含block/thread索引和全局编号，便于理解
 */
__global__ void print_thread_id() {
    // 计算当前线程的全局唯一编号（核心：区分每个线程的标识）
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 打印线程信息：threadIdx(线程在块内编号) + global_thread_id(全局编号)
    printf(
        "blockIdx.x = %d, threadIdx.x = %d | global_thread_id = %d\n",
        blockIdx.x, threadIdx.x, global_thread_id
    );
}

int main() {
    // 1. 配置线程：总线程数=20（block=20，grid=1）
    int block_size = 20;  // 每个block分配20个线程
    int grid_size = 1;    // 仅1个block（确保总线程数=1×20=20）

    // 2. 启动核函数
    print_thread_id<<<grid_size, block_size>>>();

    // 3. 同步GPU：确保所有线程的打印输出完整显示（必须加，否则可能看不到输出）
    cudaDeviceSynchronize();

    // 检查核函数执行错误（可选，便于调试）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to run kernal !" << std::endl;
    }

    return 0;
}