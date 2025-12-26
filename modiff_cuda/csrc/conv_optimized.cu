#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimized INT8 convolution kernel with tiling and shared memory
// Strategy: Tile-based computation with shared memory for weight reuse

// Version 1: Shared memory for weights
template<int TILE_H, int TILE_W, int TILE_C_OUT>
__global__ void conv2d_optimized_v1(
    const int8_t* __restrict__ input,    // [N, H, W, C_in]
    const int8_t* __restrict__ weight,   // [C_out, K, K, C_in]
    int32_t* __restrict__ output,        // [N, H_out, W_out, C_out]
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    // Block processes TILE_C_OUT output channels
    int c_out_base = blockIdx.z * TILE_C_OUT;
    int h_out_base = blockIdx.y * TILE_H;
    int w_out_base = blockIdx.x * TILE_W;
    int n = 0;  // Assume batch size 1 for now
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Shared memory for weights: [TILE_C_OUT, K, K, C_in]
    extern __shared__ int8_t smem[];
    int8_t* weight_smem = smem;
    
    // Load weights into shared memory cooperatively
    int weight_size = TILE_C_OUT * K * K * C_in;
    for (int i = tx + ty * blockDim.x; i < weight_size; i += blockDim.x * blockDim.y) {
        if (c_out_base * K * K * C_in + i < C_out * K * K * C_in) {
            weight_smem[i] = weight[c_out_base * K * K * C_in + i];
        }
    }
    __syncthreads();
    
    // Each thread computes one spatial position across all output channels in tile
    int h_out = h_out_base + ty;
    int w_out = w_out_base + tx;
    
    if (h_out >= H_out || w_out >= W_out || n >= N) return;
    
    // Compute for each output channel in tile
    for (int c_out_offset = 0; c_out_offset < TILE_C_OUT; c_out_offset++) {
        int c_out = c_out_base + c_out_offset;
        if (c_out >= C_out) break;
        
        int32_t sum = 0;
        
        // Convolve
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                for (int c_in = 0; c_in < C_in; c_in++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = ((n * H + h_in) * W + w_in) * C_in + c_in;
                        int weight_smem_idx = ((c_out_offset * K + kh) * K + kw) * C_in + c_in;
                        
                        int8_t in_val = input[input_idx];
                        int8_t w_val = weight_smem[weight_smem_idx];
                        sum += (int32_t)in_val * (int32_t)w_val;
                    }
                }
            }
        }
        
        int output_idx = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out;
        output[output_idx] = sum;
    }
}

// Version 2: Vectorized loads with int4
__global__ void conv2d_optimized_v2(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    // Each block processes BLOCK_C output channels and BLOCK_H x BLOCK_W spatial positions
    constexpr int BLOCK_C = 64;
    constexpr int BLOCK_H = 4;
    constexpr int BLOCK_W = 4;
    
    int c_out_base = blockIdx.z * BLOCK_C;
    int h_out_base = blockIdx.y * BLOCK_H;
    int w_out_base = blockIdx.x * BLOCK_W;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Shared memory for weights
    __shared__ int8_t weight_smem[BLOCK_C * 9 * 64];  // Assuming K=3, C_in=64
    
    // Load weights using vectorized access where possible
    int weight_size = BLOCK_C * K * K * C_in;
    for (int i = tid; i < weight_size; i += blockDim.x * blockDim.y) {
        int global_idx = c_out_base * K * K * C_in + i;
        if (global_idx < C_out * K * K * C_in) {
            weight_smem[i] = weight[global_idx];
        }
    }
    __syncthreads();
    
    // Each thread processes one output position
    int h_out = h_out_base + ty;
    int w_out = w_out_base + tx;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    // Process multiple output channels per thread
    for (int n = 0; n < N; n++) {
        for (int c_out_offset = 0; c_out_offset < BLOCK_C; c_out_offset++) {
            int c_out = c_out_base + c_out_offset;
            if (c_out >= C_out) break;
            
            int32_t sum = 0;
            
            // Vectorized inner loop over C_in (process 4 channels at a time)
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_base = ((n * H + h_in) * W + w_in) * C_in;
                        int weight_base = ((c_out_offset * K + kh) * K + kw) * C_in;
                        
                        // Vectorized dot product over C_in
                        for (int c_in = 0; c_in < C_in; c_in += 4) {
                            if (c_in + 3 < C_in) {
                                // Load 4 values at once
                                int4 in_vec = *reinterpret_cast<const int4*>(&input[input_base + c_in]);
                                int4 w_vec = *reinterpret_cast<const int4*>(&weight_smem[weight_base + c_in]);
                                
                                int8_t* in_ptr = reinterpret_cast<int8_t*>(&in_vec);
                                int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_vec);
                                
                                #pragma unroll
                                for (int i = 0; i < 16; i++) {
                                    sum += (int32_t)in_ptr[i] * (int32_t)w_ptr[i];
                                }
                            } else {
                                // Handle remainder
                                for (int i = c_in; i < C_in; i++) {
                                    sum += (int32_t)input[input_base + i] * (int32_t)weight_smem[weight_base + i];
                                }
                                break;
                            }
                        }
                    }
                }
            }
            
            int output_idx = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out;
            output[output_idx] = sum;
        }
    }
}

// Version 3: Output stationary with register tiling
template<int BLOCK_H, int BLOCK_W, int BLOCK_C, int THREAD_H, int THREAD_W, int THREAD_C>
__global__ void conv2d_optimized_v3(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    int32_t* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    // Register tiling: each thread accumulates THREAD_H x THREAD_W x THREAD_C outputs
    int32_t accum[THREAD_H][THREAD_W][THREAD_C] = {0};
    
    int c_out_base = blockIdx.z * BLOCK_C;
    int h_out_base = blockIdx.y * BLOCK_H;
    int w_out_base = blockIdx.x * BLOCK_W;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // This thread handles output positions
    int h_start = h_out_base + ty * THREAD_H;
    int w_start = w_out_base + tx * THREAD_W;
    
    // Shared memory for weights
    __shared__ int8_t weight_smem[BLOCK_C * 9 * 128];  // K=3, C_in up to 128
    
    // Load weights
    int tid = ty * blockDim.x + tx;
    int weight_size = min(BLOCK_C, C_out - c_out_base) * K * K * C_in;
    for (int i = tid; i < weight_size; i += blockDim.x * blockDim.y) {
        weight_smem[i] = weight[c_out_base * K * K * C_in + i];
    }
    __syncthreads();
    
    // Compute
    for (int n = 0; n < N; n++) {
        // Convolve
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                for (int c_in = 0; c_in < C_in; c_in++) {
                    // Load input values for this thread's tile
                    int8_t in_tile[THREAD_H][THREAD_W];
                    
                    #pragma unroll
                    for (int th = 0; th < THREAD_H; th++) {
                        #pragma unroll
                        for (int tw = 0; tw < THREAD_W; tw++) {
                            int h_out = h_start + th;
                            int w_out = w_start + tw;
                            
                            if (h_out < H_out && w_out < W_out) {
                                int h_in = h_out * stride - padding + kh;
                                int w_in = w_out * stride - padding + kw;
                                
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    in_tile[th][tw] = input[((n * H + h_in) * W + w_in) * C_in + c_in];
                                } else {
                                    in_tile[th][tw] = 0;
                                }
                            }
                        }
                    }
                    
                    // Accumulate for all output channels
                    #pragma unroll
                    for (int tc = 0; tc < THREAD_C; tc++) {
                        int c_out = c_out_base + tc;
                        if (c_out < C_out) {
                            int w_idx = ((tc * K + kh) * K + kw) * C_in + c_in;
                            int8_t w_val = weight_smem[w_idx];
                            
                            #pragma unroll
                            for (int th = 0; th < THREAD_H; th++) {
                                #pragma unroll
                                for (int tw = 0; tw < THREAD_W; tw++) {
                                    accum[th][tw][tc] += (int32_t)in_tile[th][tw] * (int32_t)w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Write results
        #pragma unroll
        for (int th = 0; th < THREAD_H; th++) {
            #pragma unroll
            for (int tw = 0; tw < THREAD_W; tw++) {
                int h_out = h_start + th;
                int w_out = w_start + tw;
                
                if (h_out < H_out && w_out < W_out) {
                    #pragma unroll
                    for (int tc = 0; tc < THREAD_C; tc++) {
                        int c_out = c_out_base + tc;
                        if (c_out < C_out) {
                            int out_idx = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out;
                            output[out_idx] = accum[th][tw][tc];
                        }
                    }
                }
            }
        }
    }
}

// Launcher functions
void conv2d_optimized_v1_cuda(
    const int8_t* input, const int8_t* weight, int32_t* output,
    int N, int C_in, int H, int W, int C_out, int K,
    int stride, int padding
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;
    constexpr int TILE_C_OUT = 32;
    
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((W_out + TILE_W - 1) / TILE_W,
              (H_out + TILE_H - 1) / TILE_H,
              (C_out + TILE_C_OUT - 1) / TILE_C_OUT);
    
    size_t smem_size = TILE_C_OUT * K * K * C_in * sizeof(int8_t);
    
    conv2d_optimized_v1<TILE_H, TILE_W, TILE_C_OUT><<<grid, block, smem_size>>>(
        input, weight, output, N, C_in, H, W, C_out, K, stride, padding, H_out, W_out
    );
}

void conv2d_optimized_v2_cuda(
    const int8_t* input, const int8_t* weight, int32_t* output,
    int N, int C_in, int H, int W, int C_out, int K,
    int stride, int padding
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    constexpr int BLOCK_H = 8;
    constexpr int BLOCK_W = 8;
    constexpr int BLOCK_C = 64;
    
    dim3 block(BLOCK_W, BLOCK_H, 1);
    dim3 grid((W_out + BLOCK_W - 1) / BLOCK_W,
              (H_out + BLOCK_H - 1) / BLOCK_H,
              (C_out + BLOCK_C - 1) / BLOCK_C);
    
    conv2d_optimized_v2<<<grid, block>>>(
        input, weight, output, N, C_in, H, W, C_out, K, stride, padding, H_out, W_out
    );
}

void conv2d_optimized_v3_cuda(
    const int8_t* input, const int8_t* weight, int32_t* output,
    int N, int C_in, int H, int W, int C_out, int K,
    int stride, int padding
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    constexpr int BLOCK_H = 16;
    constexpr int BLOCK_W = 16;
    constexpr int BLOCK_C = 64;
    constexpr int THREAD_H = 2;
    constexpr int THREAD_W = 2;
    constexpr int THREAD_C = 8;
    
    dim3 block(BLOCK_W / THREAD_W, BLOCK_H / THREAD_H, 1);
    dim3 grid((W_out + BLOCK_W - 1) / BLOCK_W,
              (H_out + BLOCK_H - 1) / BLOCK_H,
              (C_out + BLOCK_C - 1) / BLOCK_C);
    
    conv2d_optimized_v3<BLOCK_H, BLOCK_W, BLOCK_C, THREAD_H, THREAD_W, THREAD_C><<<grid, block>>>(
        input, weight, output, N, C_in, H, W, C_out, K, stride, padding, H_out, W_out
    );
}
