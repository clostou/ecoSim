#include <iostream>

__global__ void testKernel() {
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);
}

// CUDA kernel function: vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Print GPU device information
void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    printf("Found %d CUDA device(s):\n", deviceCount);
    printf("==================================================\n");
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Streaming Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Constant Memory: %lu KB\n", deviceProp.totalConstMem / 1024);
        printf("  Shared Memory per Block: %lu KB\n", deviceProp.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("  Memory Bus Width: %d bit\n", deviceProp.memoryBusWidth);
        printf("  Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
        
        // Use cudaDeviceGetAttribute for version-independent attributes
        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        printf("  Clock Rate: %.2f MHz\n", clockRate / 1000.0);
        
        int memoryClockRate;
        if (cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev) == cudaSuccess) {
            printf("  Memory Clock Rate: %.2f MHz\n", memoryClockRate / 1000.0);
        } else {
            printf("  Memory Clock Rate: N/A\n");
        }
        
        int l2CacheSize;
        if (cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, dev) == cudaSuccess) {
            printf("  L2 Cache Size: %d KB\n", l2CacheSize / 1024);
        } else {
            printf("  L2 Cache Size: N/A\n");
        }
        
        // Calculate theoretical memory bandwidth if memory clock rate is available
        if (cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev) == cudaSuccess) {
            float memoryBandwidth = 2.0 * memoryClockRate * (deviceProp.memoryBusWidth / 8.0) / 1.0e6;
            printf("  Theoretical Memory Bandwidth: %.2f GB/s\n", memoryBandwidth);
        } else {
            printf("  Theoretical Memory Bandwidth: N/A\n");
        }
        
        printf("==================================================\n");
    }
}

int main_old() {
    std::cout << "Testing GPU...\n";

    // Launch the kernel with 1 block and 10 threads
    testKernel<<<1, 10>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    std::cout << "GPU test completed.\n";
    return 0;
}

int main() {
    printf("CUDA GPU Benchmark Program\n");
    printf("==================================================\n");
    
    // 1. Print device information
    printDeviceInfo();
    
    // 2. Select current device
    int currentDevice = 0;
    cudaSetDevice(currentDevice);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, currentDevice);
    printf("\nUsing Device %d: %s for testing\n", currentDevice, prop.name);
    
    // 3. Test data size (adjustable)
    int numElements = 1 << 20; // 1,048,576 elements
    printf("Vector size: %d elements\n", numElements);
    printf("Total data size: %.2f MB\n", numElements * sizeof(float) / (1024.0 * 1024.0));
    
    // 4. Allocate host memory
    size_t size = numElements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        printf("Host memory allocation failed!\n");
        return -1;
    }
    
    // 5. Initialize data
    printf("Initializing data...\n");
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // 6. Allocate device memory
    printf("Allocating device memory...\n");
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    cudaError_t err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1;
    }
    
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }
    
    // 7. Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 8. Copy data from host to device
    printf("Copying data to device...\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Host to device copy failed: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Host to device copy failed: %s\n", cudaGetErrorString(err));
    }
    
    // 9. Set thread and block counts
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Grid configuration: %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    
    // 10. Launch kernel and time it
    printf("\nLaunching kernel...\n");
    cudaEventRecord(start);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
    
    // 11. Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 12. Copy results back to host
    printf("Copying results back to host...\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Device to host copy failed: %s\n", cudaGetErrorString(err));
    }
    
    // 13. Verify results (optional)
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            errors++;
            if (errors < 10) {
                printf("Error at index %d: A=%.6f, B=%.6f, Expected=%.6f, Got=%.6f\n",
                       i, h_A[i], h_B[i], h_A[i] + h_B[i], h_C[i]);
            }
        }
    }
    
    // 14. Print performance results
    printf("\n==================================================\n");
    printf("Performance Results:\n");
    printf("  Total execution time: %.3f ms\n", milliseconds);
    printf("  Compute throughput: %.3f GFLOPS\n", 
           (2.0 * numElements) / (milliseconds * 1e6));
    printf("  Effective memory bandwidth: %.3f GB/s\n", 
           (3.0 * size) / (milliseconds * 1e6));
    printf("  Validation: %d errors out of %d elements\n", errors, numElements);
    
    // 15. Multiple runs for stability
    printf("\nRunning multiple iterations for stability test...\n");
    int runs = 10;
    float totalTime = 0;
    float minTime = 1e9;
    float maxTime = 0;
    
    for (int run = 0; run < runs; run++) {
        cudaEventRecord(start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float runTime = 0;
        cudaEventElapsedTime(&runTime, start, stop);
        totalTime += runTime;
        
        if (runTime < minTime) minTime = runTime;
        if (runTime > maxTime) maxTime = runTime;
        
        if (run < 3) {
            printf("  Run %d: %.3f ms\n", run + 1, runTime);
        }
    }
    
    float avgTime = totalTime / runs;
    printf("  Performance Summary (%d runs):\n", runs);
    printf("    Best time: %.3f ms (%.3f GFLOPS)\n", 
           minTime, (2.0 * numElements) / (minTime * 1e6));
    printf("    Worst time: %.3f ms (%.3f GFLOPS)\n", 
           maxTime, (2.0 * numElements) / (maxTime * 1e6));
    printf("    Average time: %.3f ms (%.3f GFLOPS)\n", 
           avgTime, (2.0 * numElements) / (avgTime * 1e6));
    
    // 16. Test different block sizes
    printf("\nTesting different block sizes...\n");
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    for (int i = 0; i < numBlockSizes; i++) {
        int bs = blockSizes[i];
        if (bs <= prop.maxThreadsPerBlock) {
            int blocks = (numElements + bs - 1) / bs;
            
            cudaEventRecord(start);
            vectorAdd<<<blocks, bs>>>(d_A, d_B, d_C, numElements);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            printf("  Block size %4d: %.3f ms (%.3f GFLOPS)\n", 
                   bs, time, (2.0 * numElements) / (time * 1e6));
        }
    }
    
    // 17. Clean up resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 18. Reset device
    cudaDeviceReset();
    
    printf("\n==================================================\n");
    printf("Benchmark completed successfully!\n");
    return 0;
}

