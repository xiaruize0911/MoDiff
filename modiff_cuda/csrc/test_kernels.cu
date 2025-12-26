#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Test 1: Just write to global memory (simple 1D indexing)
__global__ void test_global_write(half *C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        C[idx] = __float2half(42.0f);
    }
}

// Test 2: Use shared memory with same allocation size as Conv kernel
__global__ void test_shared_memory(half *C, int M, int N) {
    extern __shared__ int8_t smem[];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    
    // Write to shared memory
    if (tid < 256) {
        for (int i = 0; i < 100; i++) {
            smem[tid * 100 + i] = (int8_t)tid;
        }
    }
    __syncthreads();
    
    // Read from shared memory and write to global
    int idx = bid * 256 + tid;
    if (idx < M * N && tid < 256) {
        C[idx] = __float2half((float)smem[tid * 100]);
    }
}

// Test 3: Exactly replicate Conv kernel's shared memory usage pattern
__global__ void test_conv_smem_pattern(half *C, int M, int N) {
    extern __shared__ int8_t smem_buf[][64];  // Renamed to avoid collision
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;
    
    // Write pattern similar to A/B loading
    if (warp_id < 4) {  // First 4 warps
        for (int i = 0; i < 16; i++) {
            int row = warp_id * 32 + i;
            if (row < 128) {
                for (int j = lane_id; j < 64; j += 32) {
                    smem_buf[row][j] = (int8_t)(row + j);
                }
            }
        }
    }
    __syncthreads();
    
    // Read and write to global
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = bid * 256 + tid;
    if (idx < M * N) {
        int8_t val = smem_buf[warp_id * 16 + lane_id % 16][lane_id % 64];
        C[idx] = __float2half((float)val);
    }
}

// Launcher functions
void test_global_write_run(half *C, int M, int N) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    test_global_write<<<blocks, threads>>>(C, M, N);
}

void test_shared_memory_run(half *C, int M, int N, int smem_size) {
    dim3 grid(N / 128, M / 128, 1);
    dim3 block(32, 8, 1);
    test_shared_memory<<<grid, block, smem_size>>>(C, M, N);
}

void test_conv_smem_pattern_run(half *C, int M, int N, int smem_size) {
    dim3 grid(N / 128, M / 128, 1);
    dim3 block(32, 8, 1);
    
    auto kernel_func = test_conv_smem_pattern;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    kernel_func<<<grid, block, smem_size>>>(C, M, N);
}
