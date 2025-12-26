#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <algorithm>
#include "cp_async.cuh"
#include "permuted_smem.cuh"
#include "mma.cuh"

// Optimized W8A8 Convolution Kernel
// Implicit GEMM with Tensor Core path (1x1) and dp4a path (3x3).
// For 1x1, use mma/ldmatrix with swizzled shared memory.
// Fuses dequantization and FP16 conversion.

#define TILE_M 128
#define TILE_N 64
#define TILE_K 64

__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template<typename T>
__global__ void find_max_abs_kernel(const T* __restrict__ input, float* __restrict__ max_val, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float local_max = 0.0f;
    for (int i = idx; i < size; i += stride) {
        float val;
        if constexpr (std::is_same<T, half>::value) {
            val = __half2float(input[i]);
        } else {
            val = (float)input[i];
        }
        local_max = fmaxf(local_max, fabsf(val));
    }
    
    sdata[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int* addr_as_int = (int*)max_val;
        int val_as_int = __float_as_int(sdata[0]);
        atomicMax(addr_as_int, val_as_int);
    }
}

// Quantization Kernel (NHWC -> NHWC)
template<typename T>
__global__ void quantize_kernel(const T* __restrict__ input, int8_t* __restrict__ output, const float* __restrict__ scale_ptr, int size) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float val;
        if constexpr (std::is_same<T, half>::value) {
            val = __half2float(input[i]);
        } else {
            val = (float)input[i];
        }
        val = roundf(val / scale);
        output[i] = (int8_t)fmaxf(-128.0f, fminf(127.0f, val));
    }
}

// Fused Permute + Quantize Kernel (NCHW -> NHWC)
// Transposes [C, H*W] to [H*W, C] for each N.
// Block 32x32.
template<typename T>
__global__ void quantize_permute_kernel(
    const T* __restrict__ input, // NCHW
    int8_t* __restrict__ output,     // NHWC
    const float* __restrict__ scale_ptr,
    int N, int C, int H, int W)
{
    float scale = *scale_ptr;
    
    // Shared memory for tile
    __shared__ float tile[32][33]; // Padding to avoid bank conflicts
    
    int hw = H * W;
    int n_idx = blockIdx.z;
    
    // Global coordinates
    int x = blockIdx.x * 32 + threadIdx.x; // H*W dimension
    int y = blockIdx.y * 32 + threadIdx.y; // C dimension
    
    // Input is [N, C, H, W]. Flattened: n*C*HW + c*HW + hw_idx
    // We read [y, x] from current N.
    
    if (x < hw && y < C) {
        int in_idx = n_idx * C * hw + y * hw + x;
        float val;
        if constexpr (std::is_same<T, half>::value) {
            val = __half2float(input[in_idx]);
        } else {
            val = (float)input[in_idx];
        }
        tile[threadIdx.y][threadIdx.x] = val;
    }
    
    __syncthreads();
    
    // Transpose: read from tile[x][y] (swapped)
    // We want to write to Output [N, H, W, C]. Flattened: n*HW*C + hw_idx*C + c
    // Output coords: x (HW), y (C).
    // We want to write [x, y].
    // But to coalesce writes, we want threadIdx.x to map to inner dim (C).
    // So we swap thread roles for writing.
    
    int out_x = blockIdx.x * 32 + threadIdx.y; // H*W
    int out_y = blockIdx.y * 32 + threadIdx.x; // C
    
    if (out_x < hw && out_y < C) {
        float val = tile[threadIdx.x][threadIdx.y]; // Transposed read from shared
        
        // Quantize
        val = roundf(val / scale);
        int8_t q_val = (int8_t)fmaxf(-128.0f, fminf(127.0f, val));
        
        int out_idx = n_idx * hw * C + out_x * C + out_y;
        output[out_idx] = q_val;
    }
}

// Permute Kernel (NHWC -> NCHW) for Half
// Transposes [H*W, C] to [C, H*W] for each N.
// Also computes max(abs(input)) for the next layer.
__global__ void permute_half_nhwc_nchw_kernel(
    const half* __restrict__ input, // NHWC
    half* __restrict__ output,      // NCHW
    float* __restrict__ max_val,    // Output max value
    int N, int C, int H, int W)
{
    // Shared memory for tile
    __shared__ half tile[32][33]; 
    __shared__ float sdata[1024];
    
    int hw = H * W;
    int n_idx = blockIdx.z;
    
    int x = blockIdx.x * 32 + threadIdx.x; // C
    int y = blockIdx.y * 32 + threadIdx.y; // HW
    
    float local_max = 0.0f;
    
    if (x < C && y < hw) {
        int in_idx = n_idx * hw * C + y * C + x;
        half val = input[in_idx];
        tile[threadIdx.y][threadIdx.x] = val;
        if (max_val) {
            local_max = fabsf(__half2float(val));
        }
    }
    
    __syncthreads();
    
    int out_x = blockIdx.x * 32 + threadIdx.y; // C
    int out_y = blockIdx.y * 32 + threadIdx.x; // HW
    
    if (out_x < C && out_y < hw) {
        half val = tile[threadIdx.x][threadIdx.y];
        int out_idx = n_idx * C * hw + out_x * hw + out_y;
        output[out_idx] = val;
    }
    
    if (max_val) {
        int tid = threadIdx.y * 32 + threadIdx.x;
        sdata[tid] = local_max;
        __syncthreads();
        
        for (int s = 512; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            int* addr_as_int = (int*)max_val;
            int val_as_int = __float_as_int(sdata[0]);
            atomicMax(addr_as_int, val_as_int);
        }
    }
}

template<bool IS_1X1, bool ACCUM>
__global__ void __launch_bounds__(256, 2) conv2d_fast_w8a8_kernel_impl(
    const int8_t* __restrict__ A,      // [Batch, H, W, C_in]
    const int8_t* __restrict__ B,      // [C_out, R, S, C_in]
    half* __restrict__ C,              // [Batch, H_out, W_out, C_out]
    const half* __restrict__ PrevC,    // [Batch, H_out, W_out, C_out] (Optional)
    const float* __restrict__ act_scale_ptr,
    const float* __restrict__ weight_scales, // [C_out]
    float* __restrict__ max_output,    // [Optional] Output max value
    int M, int N, int K,
    int Batch, int H, int W, int C_in,
    int C_out, int R, int S,
    int H_out, int W_out,
    int stride, int padding) 
{
    // Shared memory for tiling
    // Shared memory for tiling (Triple Buffered for 3-stage cp.async pipeline)
    __shared__ __align__(16) int8_t sh_a[3][TILE_M][TILE_K + 16];
    __shared__ __align__(16) int8_t sh_b[3][TILE_N][TILE_K + 16];
    // Tensor Core staging (single buffer, reused per compute tile)
    __shared__ __align__(16) half sh_a_tc[1][TILE_M][TILE_K];
    __shared__ __align__(16) half sh_b_tc[1][TILE_K][TILE_N];
    
    // Shared memory for cached indices (only for non-1x1)
    __shared__ int16_t sh_m_b[TILE_M];
    __shared__ int16_t sh_m_h[TILE_M];
    __shared__ int16_t sh_m_w[TILE_M];
    
    // Triple buffer K indices too
    __shared__ int16_t sh_k_r[3][TILE_K];
    __shared__ int16_t sh_k_s[3][TILE_K];
    __shared__ int16_t sh_k_c[3][TILE_K];

    int thread_id = threadIdx.x;
    
    // Precompute M indices
    if (!IS_1X1) {
        if (thread_id < TILE_M) {
            int m_global = blockIdx.x * TILE_M + thread_id;
            if (m_global < M) {
                sh_m_b[thread_id] = m_global / (H_out * W_out);
                int rem = m_global % (H_out * W_out);
                sh_m_h[thread_id] = rem / W_out;
                sh_m_w[thread_id] = rem % W_out;
            }
        }
    }
    
    int ty = thread_id / 16;
    int tx = thread_id % 16;
    int m_start = ty * 8; // Each thread computes 8 rows
    int n_start = tx * 4; // Each thread computes 4 cols

    int32_t acc[8][4] = {0};

    bool constant_rs = (!IS_1X1) && (C_in % TILE_K == 0);

    // Helper lambda to load one tile with cp.async into a given buffer
    auto load_tile = [&](int k_outer, int buf_idx) {
        // Precompute K indices for this tile
        if (!IS_1X1 && !constant_rs) {
            if (thread_id < TILE_K) {
                int k_curr = k_outer + thread_id;
                if (k_curr < K) {
                    sh_k_r[buf_idx][thread_id] = k_curr / (S * C_in);
                    int rem = k_curr % (S * C_in);
                    sh_k_s[buf_idx][thread_id] = rem / C_in;
                    sh_k_c[buf_idx][thread_id] = rem % C_in;
                }
            }
        }

        // Ensure K indices are visible before they are consumed below
        __syncthreads();

        // Load A -> sh_a[buf_idx]
        int tid = thread_id;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int chunk_idx = tid * 2 + i;
            int m_idx = chunk_idx / 4;
            int k_sub = (chunk_idx % 4) * 16;

            int m_global = blockIdx.x * TILE_M + m_idx;
            int k_global = k_outer + k_sub;

            if (m_global < M) {
                if (IS_1X1) {
                    if (k_global + 15 < K) {
                        cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                            &sh_a[buf_idx][m_idx][k_sub],
                            &A[m_global * K + k_global],
                            true);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            sh_a[buf_idx][m_idx][k_sub + j] = (k_global + j < K) ? A[m_global * K + k_global + j] : 0;
                        }
                    }
                } else {
                    bool use_vectorized = (C_in % 16 == 0);

                    if (use_vectorized) {
                        int b = sh_m_b[m_idx];
                        int h_out = sh_m_h[m_idx];
                        int w_out = sh_m_w[m_idx];

                        int r, s, c;
                        if (constant_rs) {
                            r = k_outer / (S * C_in);
                            int rem = k_outer % (S * C_in);
                            s = rem / C_in;
                            c = (rem % C_in) + k_sub;
                        } else {
                            r = sh_k_r[buf_idx][k_sub];
                            s = sh_k_s[buf_idx][k_sub];
                            c = sh_k_c[buf_idx][k_sub];
                        }

                        int h_in = h_out * stride + r - padding;
                        int w_in = w_out * stride + s - padding;

                        int h_clamped = max(0, min(H - 1, h_in));
                        int w_clamped = max(0, min(W - 1, w_in));

                        bool in_bounds = (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W);

                        cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                            &sh_a[buf_idx][m_idx][k_sub],
                            &A[((b * H + h_clamped) * W + w_clamped) * C_in + c],
                            in_bounds);
                    } else {
                        int b = sh_m_b[m_idx];
                        int h_out = sh_m_h[m_idx];
                        int w_out = sh_m_w[m_idx];
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            int k_idx = k_sub + j;
                            int8_t val = 0;
                            if (k_outer + k_idx < K) {
                                int r, s, c;
                                if (constant_rs) {
                                    r = k_outer / (S * C_in);
                                    int rem = k_outer % (S * C_in);
                                    s = rem / C_in;
                                    c = (rem % C_in) + k_idx;
                                } else {
                                    r = sh_k_r[buf_idx][k_idx];
                                    s = sh_k_s[buf_idx][k_idx];
                                    c = sh_k_c[buf_idx][k_idx];
                                }

                                int h_in = h_out * stride + r - padding;
                                int w_in = w_out * stride + s - padding;

                                int h_clamped = max(0, min(H - 1, h_in));
                                int w_clamped = max(0, min(W - 1, w_in));

                                val = A[((b * H + h_clamped) * W + w_clamped) * C_in + c];

                                if (!(h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)) val = 0;
                            }
                            sh_a[buf_idx][m_idx][k_idx] = val;
                        }
                    }
                }
            } else {
                cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                    &sh_a[buf_idx][m_idx][k_sub],
                    (const int8_t*)nullptr,
                    false);
            }
        }

        // Load B -> sh_b[buf_idx]
        tid = thread_id;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int chunk_idx = tid * 2 + i;
            int n_idx = chunk_idx / 4;
            int k_sub = (chunk_idx % 4) * 16;

            if (n_idx < TILE_N) {
                int n_global = blockIdx.y * TILE_N + n_idx;
                int k_global = k_outer + k_sub;

                bool b_vectorized = (K % 16 == 0);

                if (n_global < N) {
                    if (b_vectorized && k_global + 15 < K) {
                        cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                            &sh_b[buf_idx][n_idx][k_sub],
                            &B[n_global * K + k_global],
                            true);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            sh_b[buf_idx][n_idx][k_sub + j] = (k_global + j < K) ? B[n_global * K + k_global + j] : 0;
                        }
                    }
                } else {
                    cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                        &sh_b[buf_idx][n_idx][k_sub],
                        (const int8_t*)nullptr,
                        false);
                }
            }
        }

        cp_async::commit_group();
    };

    // Prologue: prefetch first two tiles for a 3-stage pipeline
    int buf_compute = 0;
    int buf_prefetch = 1;
    int k_prefetch = TILE_K;

    load_tile(0, buf_compute);
    cp_async::wait_group<0>();
    __syncthreads();

    if (k_prefetch < K) {
        load_tile(k_prefetch, buf_prefetch);
    }

#if __CUDA_ARCH__ >= 800
    // Tensor Core path (any kernel size)
    int warp_id = thread_id >> 5;
    int warp_m = warp_id >> 1; // 0..3 (rows)
    int warp_n = warp_id & 1;  // 0..1 (cols)

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    // Main Loop
    for (int k_outer = 0; k_outer < K; k_outer += TILE_K) {
        // Ensure the tile to compute is ready (except the very first which is already waited)
        if (k_outer != 0) {
            cp_async::wait_group<0>();
            __syncthreads();
        }

        // Convert int8 tiles to half for Tensor Core MMA
        int idx = thread_id;
        int total_a = TILE_M * TILE_K;
        while (idx < total_a) {
            int m = idx / TILE_K;
            int k = idx - m * TILE_K;
            sh_a_tc[0][m][k] = __int2half_rn((int)sh_a[buf_compute][m][k]);
            idx += blockDim.x;
        }
        idx = thread_id;
        int total_b = TILE_N * TILE_K;
        while (idx < total_b) {
            int n = idx / TILE_K;
            int k = idx - n * TILE_K;
            // Store B in column-major layout for wmma matrix_b (K major)
            sh_b_tc[0][k][n] = __int2half_rn((int)sh_b[buf_compute][n][k]);
            idx += blockDim.x;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];

        #pragma unroll
        for (int k = 0; k < TILE_K; k += 16) {
            // Load B fragments for current K slice
            #pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                int n_base = warp_n * 32 + ni * 16;
                wmma::load_matrix_sync(b_frag[ni], &sh_b_tc[0][k][n_base], TILE_K);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int m_base = warp_m * 32 + mi * 16;
                wmma::load_matrix_sync(a_frag[mi], &sh_a_tc[0][m_base][k], TILE_K);
                #pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }

        // Prefetch next tile (3-stage ring buffer)
        int next_k = k_prefetch + TILE_K;
        int buf_next = (buf_prefetch + 1) % 3;
        if (next_k < K) {
            load_tile(next_k, buf_next);
        }

        // Advance buffers
        buf_compute = buf_prefetch;
        buf_prefetch = buf_next;
        k_prefetch = next_k;
    }

    // Write back once after full K accumulation
    float act_scale = *act_scale_ptr;
    float local_max_tc = 0.0f;

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        int m_base = blockIdx.x * TILE_M + warp_m * 32 + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            int n_base = blockIdx.y * TILE_N + warp_n * 32 + ni * 16;
            #pragma unroll
            for (int t = 0; t < (int)c_frag[mi][ni].num_elements; ++t) {
                int row = t / 16;
                int col = t - row * 16;
                int m_global = m_base + row;
                int n_global = n_base + col;
                if (m_global < M && n_global < N) {
                    float w_scale = weight_scales[n_global];
                    float val = c_frag[mi][ni].x[t] * act_scale * w_scale;
                    if (ACCUM) {
                        float prev = __half2float(PrevC[m_global * N + n_global]);
                        val += prev;
                    }
                    C[m_global * N + n_global] = __float2half(val);
                    if (max_output) {
                        local_max_tc = fmaxf(local_max_tc, fabsf(val));
                    }
                }
            }
        }
    }

    if (max_output) {
        // Block reduction for max
        __shared__ float sdata_tc[256];
        sdata_tc[threadIdx.x] = local_max_tc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_tc[threadIdx.x] = fmaxf(sdata_tc[threadIdx.x], sdata_tc[threadIdx.x + s]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int* addr_as_int = (int*)max_output;
            int val_as_int = __float_as_int(sdata_tc[0]);
            atomicMax(addr_as_int, val_as_int);
        }
    }

#else
    // Main Loop (dp4a path)
    for (int k_outer = 0; k_outer < K; k_outer += TILE_K) {
        if (k_outer != 0) {
            cp_async::wait_group<0>();
            __syncthreads();
        }

        // Compute Tile (buf_compute) using dp4a
        #pragma unroll
        for (int k = 0; k < TILE_K; k += 4) {
            uint32_t a_vals[8];
            uint32_t b_vals[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_vals[i] = *((uint32_t*)&sh_a[buf_compute][m_start + i][k]);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                b_vals[i] = *((uint32_t*)&sh_b[buf_compute][n_start + i][k]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __dp4a((int)a_vals[i], (int)b_vals[j], (int)acc[i][j]);
                }
            }
        }

        // Prefetch next tile (3-stage ring buffer)
        int next_k = k_prefetch + TILE_K;
        int buf_next = (buf_prefetch + 1) % 3;
        if (next_k < K) {
            load_tile(next_k, buf_next);
        }

        // Advance buffers
        buf_compute = buf_prefetch;
        buf_prefetch = buf_next;
        k_prefetch = next_k;
    }
#endif

#if __CUDA_ARCH__ < 800
    {
        // Write back with Dequantization and FP16 conversion (dp4a path)
        int n_global_base = blockIdx.y * TILE_N + n_start;
        float act_scale = *act_scale_ptr;
        float local_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int m_idx = m_start + i;
            int m_global = blockIdx.x * TILE_M + m_idx;
            if (m_global < M) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int n_global = n_global_base + j;
                    if (n_global < N) {
                        float w_scale = weight_scales[n_global];
                        float val = (float)acc[i][j] * act_scale * w_scale;
                        
                        if (ACCUM) {
                            float prev = __half2float(PrevC[m_global * N + n_global]);
                            val += prev;
                        }
                        C[m_global * N + n_global] = __float2half(val);
                        
                        if (max_output) {
                            local_max = fmaxf(local_max, fabsf(val));
                        }
                    }
                }
            }
        }

        if (max_output) {
            // Block reduction
            __shared__ float sdata[256];
            int tid = threadIdx.x;
            sdata[tid] = local_max;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                int* addr_as_int = (int*)max_output;
                int val_as_int = __float_as_int(sdata[0]);
                atomicMax(addr_as_int, val_as_int);
            }
        }
    }
#endif
}


extern "C" {

void quantize_tensor_cuda(const float* input, int8_t* output, const float* scale_ptr, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads * 4 - 1) / (threads * 4); 
    quantize_kernel<float><<<blocks, threads, 0, stream>>>(input, output, scale_ptr, size);
}

void quantize_tensor_half_cuda(const half* input, int8_t* output, const float* scale_ptr, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads * 4 - 1) / (threads * 4); 
    quantize_kernel<half><<<blocks, threads, 0, stream>>>(input, output, scale_ptr, size);
}

void quantize_permute_cuda(const float* input, int8_t* output, const float* scale_ptr, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 32);
    dim3 blocks((H * W + 31) / 32, (C + 31) / 32, N);
    quantize_permute_kernel<float><<<blocks, threads, 0, stream>>>(input, output, scale_ptr, N, C, H, W);
}

void quantize_permute_half_cuda(const half* input, int8_t* output, const float* scale_ptr, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 32);
    dim3 blocks((H * W + 31) / 32, (C + 31) / 32, N);
    quantize_permute_kernel<half><<<blocks, threads, 0, stream>>>(input, output, scale_ptr, N, C, H, W);
}

void permute_half_nhwc_nchw_cuda(const half* input, half* output, float* max_val, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 32);
    // Grid dimensions match the reading pattern: x->C, y->HW
    dim3 blocks((C + 31) / 32, (H * W + 31) / 32, N);
    
    if (max_val) {
        cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    }
    
    permute_half_nhwc_nchw_kernel<<<blocks, threads, 0, stream>>>(input, output, max_val, N, C, H, W);
}

void find_max_abs_cuda(const float* input, float* max_val, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads * 8 - 1) / (threads * 8);
    blocks = std::min(blocks, 1024);
    
    // Initialize max_val to 0
    cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    
    find_max_abs_kernel<float><<<blocks, threads, threads * sizeof(float), stream>>>(input, max_val, size);
}

void find_max_abs_half_cuda(const half* input, float* max_val, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads * 8 - 1) / (threads * 8);
    blocks = std::min(blocks, 1024);
    
    // Initialize max_val to 0
    cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    
    find_max_abs_kernel<half><<<blocks, threads, threads * sizeof(float), stream>>>(input, max_val, size);
}

void conv2d_fast_w8a8_cuda(
    const int8_t* input, const int8_t* weight, half* output,
    const float* act_scale, const float* weight_scales,
    float* max_output,
    int N, int C_in, int H, int W, int C_out, int K_size,
    int stride, int padding, cudaStream_t stream) 
{
    int R = K_size;
    int S = K_size;
    int H_out = (H + 2 * padding - R) / stride + 1;
    int W_out = (W + 2 * padding - S) / stride + 1;
    
    int M = N * H_out * W_out;
    int K = R * S * C_in;
    // N in GEMM is C_out

    dim3 block(256);
    dim3 grid((M + TILE_M - 1) / TILE_M, (C_out + TILE_N - 1) / TILE_N);
    
    if (max_output) {
        cudaMemsetAsync(max_output, 0, sizeof(float), stream);
    }

    if (R == 1 && S == 1 && stride == 1 && padding == 0) {
        conv2d_fast_w8a8_kernel_impl<true, false><<<grid, block, 0, stream>>>(
            input, weight, output, nullptr,
            act_scale, weight_scales,
            max_output,
            M, C_out, K,
            N, H, W, C_in,
            C_out, R, S,
            H_out, W_out,
            stride, padding
        );
    } else {
        conv2d_fast_w8a8_kernel_impl<false, false><<<grid, block, 0, stream>>>(
            input, weight, output, nullptr,
            act_scale, weight_scales,
            max_output,
            M, C_out, K,
            N, H, W, C_in,
            C_out, R, S,
            H_out, W_out,
            stride, padding
        );
    }
}

void conv2d_fast_w8a8_accum_cuda(
    const int8_t* input, const int8_t* weight, half* output, const half* prev_output,
    const float* act_scale, const float* weight_scales,
    float* max_output,
    int N, int C_in, int H, int W, int C_out, int K_size,
    int stride, int padding, cudaStream_t stream) 
{
    int R = K_size;
    int S = K_size;
    int H_out = (H + 2 * padding - R) / stride + 1;
    int W_out = (W + 2 * padding - S) / stride + 1;
    
    int M = N * H_out * W_out;
    int K = R * S * C_in;

    dim3 block(256);
    dim3 grid((M + TILE_M - 1) / TILE_M, (C_out + TILE_N - 1) / TILE_N);
    
    if (max_output) {
        cudaMemsetAsync(max_output, 0, sizeof(float), stream);
    }

    if (R == 1 && S == 1 && stride == 1 && padding == 0) {
        conv2d_fast_w8a8_kernel_impl<true, true><<<grid, block, 0, stream>>>(
            input, weight, output, prev_output,
            act_scale, weight_scales,
            max_output,
            M, C_out, K,
            N, H, W, C_in,
            C_out, R, S,
            H_out, W_out,
            stride, padding
        );
    } else {
        conv2d_fast_w8a8_kernel_impl<false, true><<<grid, block, 0, stream>>>(
            input, weight, output, prev_output,
            act_scale, weight_scales,
            max_output,
            M, C_out, K,
            N, H, W, C_in,
            C_out, R, S,
            H_out, W_out,
            stride, padding
        );
    }
}

} // extern "C"

// --------------------------------------------------------------------------
// MoDiff Fused Kernels
// --------------------------------------------------------------------------

// Kernel 1: Compute Max Abs Diff
// Computes max(|x - prev|)
template<typename T>
__global__ void find_max_abs_diff_kernel(const T* __restrict__ x, const T* __restrict__ prev, float* __restrict__ max_val, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float local_max = 0.0f;
    for (int i = idx; i < size; i += stride) {
        float val_x, val_prev;
        if constexpr (std::is_same<T, half>::value) {
            val_x = __half2float(x[i]);
            val_prev = __half2float(prev[i]);
        } else {
            val_x = (float)x[i];
            val_prev = (float)prev[i];
        }
        local_max = fmaxf(local_max, fabsf(val_x - val_prev));
    }
    
    sdata[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int* addr_as_int = (int*)max_val;
        int val_as_int = __float_as_int(sdata[0]);
        atomicMax(addr_as_int, val_as_int);
    }
}

// Kernel 2: Update and Quantize
// res = x - prev
// scale = max_val / 127.0
// q = round(res / scale)
// prev += q * scale
// out_q = q
template<typename T>
__global__ void modiff_update_kernel(
    const T* __restrict__ x,
    T* __restrict__ prev, // In-place update
    int8_t* __restrict__ out_q,
    const float* __restrict__ max_val_ptr,
    int size)
{
    float max_val = *max_val_ptr;
    float scale = max_val / 127.0f;
    
    // Avoid division by zero
    if (scale < 1e-8f) scale = 1e-8f;
    float inv_scale = 1.0f / scale;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float val_x, val_prev;
        if constexpr (std::is_same<T, half>::value) {
            val_x = __half2float(x[i]);
            val_prev = __half2float(prev[i]);
        } else {
            val_x = (float)x[i];
            val_prev = (float)prev[i];
        }
        
        float residual = val_x - val_prev;
        float q_float = roundf(residual * inv_scale);
        int8_t q_int8 = (int8_t)fmaxf(-128.0f, fminf(127.0f, q_float));
        
        // Update prev
        float recon_res = (float)q_int8 * scale;
        float new_prev = val_prev + recon_res;
        
        if constexpr (std::is_same<T, half>::value) {
            prev[i] = __float2half(new_prev);
        } else {
            prev[i] = new_prev;
        }
        
        out_q[i] = q_int8;
    }
}

// Kernel 3: Compute Max Abs Diff with Permute (NCHW x, NHWC prev)
// Optimized with 64x64 Tiled Shared Memory Transpose
template<typename T>
__global__ void find_max_abs_diff_permute_kernel(
    const T* __restrict__ x,      // NCHW
    const T* __restrict__ prev,   // NHWC
    float* __restrict__ max_val,
    int N, int C, int H, int W)
{
    // Tile size: 64x64
    // Block size: 32x8 (256 threads)
    // Each thread loads 16 elements
    
    __shared__ T smem[64][65]; // [C_tile][S_tile] with padding
    
    int HW = H * W;
    int C_HW = C * HW;
    
    float local_max = 0.0f;
    
    // Grid Stride Loop over N
    for (int n = blockIdx.z; n < N; n += gridDim.z) {
        // Grid Stride Loop over HW (S)
        for (int s_base = blockIdx.y * 64; s_base < HW; s_base += gridDim.y * 64) {
            // Grid Stride Loop over C
            for (int c_base = blockIdx.x * 64; c_base < C; c_base += gridDim.x * 64) {
                
                // 1. Load X (NCHW) -> SMEM (Transposed)
                // We want to load x[c, s] coalesced in s.
                // Threads map to s.
                // tid.x (0..31) -> s offset (needs to cover 0..63)
                // tid.y (0..7) -> c offset (needs to cover 0..63)
                
                // We loop 2 times in X (s) and 8 times in Y (c) -> 16 loads
                
                #pragma unroll
                for (int ky = 0; ky < 8; ++ky) {
                    int c_load = c_base + threadIdx.y + ky * 8;
                    if (c_load < C) {
                        #pragma unroll
                        for (int kx = 0; kx < 2; ++kx) {
                            int s_load = s_base + threadIdx.x + kx * 32;
                            if (s_load < HW) {
                                int idx_x = n * C_HW + c_load * HW + s_load;
                                smem[threadIdx.y + ky * 8][threadIdx.x + kx * 32] = x[idx_x];
                            }
                        }
                    }
                }
                
                __syncthreads();
                
                // 2. Load Prev (NHWC) and Compute
                // We want to load prev[s, c] coalesced in c.
                // Threads map to c.
                // tid.x (0..31) -> c offset (needs 0..63)
                // tid.y (0..7) -> s offset (needs 0..63)
                
                #pragma unroll
                for (int ky = 0; ky < 8; ++ky) {
                    int s_compute = s_base + threadIdx.y + ky * 8;
                    if (s_compute < HW) {
                        #pragma unroll
                        for (int kx = 0; kx < 2; ++kx) {
                            int c_compute = c_base + threadIdx.x + kx * 32;
                            if (c_compute < C) {
                                int idx_prev = n * HW * C + s_compute * C + c_compute;
                                
                                float val_x, val_prev;
                                if constexpr (std::is_same<T, half>::value) {
                                    val_x = __half2float(smem[threadIdx.x + kx * 32][threadIdx.y + ky * 8]); // Transposed read
                                    val_prev = __half2float(prev[idx_prev]);
                                } else {
                                    val_x = (float)smem[threadIdx.x + kx * 32][threadIdx.y + ky * 8];
                                    val_prev = (float)prev[idx_prev];
                                }
                                
                                local_max = fmaxf(local_max, fabsf(val_x - val_prev));
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    
    // Warp Reduction
    local_max = warpReduceMax(local_max);
    
    // Block Reduction
    float* sdata = (float*)smem;
    int lane = threadIdx.x;
    int warpId = threadIdx.y; // 0..7
    
    if (lane == 0) {
        sdata[warpId] = local_max;
    }
    __syncthreads();
    
    if (warpId == 0 && lane < 8) {
        local_max = sdata[lane];
        local_max = fmaxf(local_max, __shfl_down_sync(0xff, local_max, 4));
        local_max = fmaxf(local_max, __shfl_down_sync(0xff, local_max, 2));
        local_max = fmaxf(local_max, __shfl_down_sync(0xff, local_max, 1));
        
        if (lane == 0) {
            int* addr_as_int = (int*)max_val;
            int val_as_int = __float_as_int(local_max);
            atomicMax(addr_as_int, val_as_int);
        }
    }
}

// Kernel 4: Update and Quantize with Permute (NCHW x, NHWC prev -> NHWC out)
// Optimized with 64x64 Tiled Shared Memory Transpose
template<typename T>
__global__ void modiff_update_permute_kernel(
    const T* __restrict__ x,      // NCHW
    T* __restrict__ prev,         // NHWC (In-place update)
    int8_t* __restrict__ out_q,   // NHWC
    const float* __restrict__ max_val_ptr,
    int N, int C, int H, int W)
{
    float max_val = *max_val_ptr;
    float scale = max_val / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv_scale = 1.0f / scale;

    // Tile size: 64x64
    // Block size: 32x8 (256 threads)
    
    __shared__ T smem[64][65]; // [C_tile][S_tile]
    
    int HW = H * W;
    int C_HW = C * HW;
    
    // Grid Stride Loop over N
    for (int n = blockIdx.z; n < N; n += gridDim.z) {
        // Grid Stride Loop over HW (S)
        for (int s_base = blockIdx.y * 64; s_base < HW; s_base += gridDim.y * 64) {
            // Grid Stride Loop over C
            for (int c_base = blockIdx.x * 64; c_base < C; c_base += gridDim.x * 64) {
                
                // 1. Load X (NCHW) -> SMEM (Transposed)
                #pragma unroll
                for (int ky = 0; ky < 8; ++ky) {
                    int c_load = c_base + threadIdx.y + ky * 8;
                    if (c_load < C) {
                        #pragma unroll
                        for (int kx = 0; kx < 2; ++kx) {
                            int s_load = s_base + threadIdx.x + kx * 32;
                            if (s_load < HW) {
                                int idx_x = n * C_HW + c_load * HW + s_load;
                                smem[threadIdx.y + ky * 8][threadIdx.x + kx * 32] = x[idx_x];
                            }
                        }
                    }
                }
                
                __syncthreads();
                
                // 2. Load Prev, Compute, Update
                #pragma unroll
                for (int ky = 0; ky < 8; ++ky) {
                    int s_compute = s_base + threadIdx.y + ky * 8;
                    if (s_compute < HW) {
                        #pragma unroll
                        for (int kx = 0; kx < 2; ++kx) {
                            int c_compute = c_base + threadIdx.x + kx * 32;
                            if (c_compute < C) {
                                int idx_prev = n * HW * C + s_compute * C + c_compute;
                                
                                float val_x, val_prev;
                                if constexpr (std::is_same<T, half>::value) {
                                    val_x = __half2float(smem[threadIdx.x + kx * 32][threadIdx.y + ky * 8]);
                                    val_prev = __half2float(prev[idx_prev]);
                                } else {
                                    val_x = (float)smem[threadIdx.x + kx * 32][threadIdx.y + ky * 8];
                                    val_prev = (float)prev[idx_prev];
                                }
                                
                                float residual = val_x - val_prev;
                                float q_float = roundf(residual * inv_scale);
                                int8_t q_int8 = (int8_t)fmaxf(-128.0f, fminf(127.0f, q_float));
                                
                                // Update prev (NHWC)
                                float recon_res = (float)q_int8 * scale;
                                float new_prev = val_prev + recon_res;
                                
                                if constexpr (std::is_same<T, half>::value) {
                                    prev[idx_prev] = __float2half(new_prev);
                                } else {
                                    prev[idx_prev] = new_prev;
                                }
                                
                                out_q[idx_prev] = q_int8;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

extern "C" {

void find_max_abs_diff_cuda(const float* x, const float* prev, float* max_val, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    if (blocks > 128) blocks = 128;
    
    cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    find_max_abs_diff_kernel<float><<<blocks, threads, threads * sizeof(float), stream>>>(x, prev, max_val, size);
}

void find_max_abs_diff_half_cuda(const half* x, const half* prev, float* max_val, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    if (blocks > 128) blocks = 128;
    
    cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    find_max_abs_diff_kernel<half><<<blocks, threads, threads * sizeof(float), stream>>>(x, prev, max_val, size);
}

void modiff_update_cuda(const float* x, float* prev, int8_t* out_q, const float* max_val_ptr, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    modiff_update_kernel<float><<<blocks, threads, 0, stream>>>(x, prev, out_q, max_val_ptr, size);
}

void modiff_update_half_cuda(const half* x, half* prev, int8_t* out_q, const float* max_val_ptr, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    modiff_update_kernel<half><<<blocks, threads, 0, stream>>>(x, prev, out_q, max_val_ptr, size);
}

void find_max_abs_diff_permute_cuda(const float* x, const float* prev, float* max_val, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 8);
    dim3 blocks(
        std::min((C + 63) / 64, 32),
        std::min((H * W + 63) / 64, 128),
        std::min(N, 64)
    );
    cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    find_max_abs_diff_permute_kernel<float><<<blocks, threads, 0, stream>>>(x, prev, max_val, N, C, H, W);
}

void find_max_abs_diff_permute_half_cuda(const half* x, const half* prev, float* max_val, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 8);
    dim3 blocks(
        std::min((C + 63) / 64, 32),
        std::min((H * W + 63) / 64, 128),
        std::min(N, 64)
    );
    cudaMemsetAsync(max_val, 0, sizeof(float), stream);
    find_max_abs_diff_permute_kernel<half><<<blocks, threads, 0, stream>>>(x, prev, max_val, N, C, H, W);
}

void modiff_update_permute_cuda(const float* x, float* prev, int8_t* out_q, const float* max_val_ptr, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 8);
    dim3 blocks(
        std::min((C + 63) / 64, 32),
        std::min((H * W + 63) / 64, 128),
        std::min(N, 64)
    );
    modiff_update_permute_kernel<float><<<blocks, threads, 0, stream>>>(x, prev, out_q, max_val_ptr, N, C, H, W);
}

void modiff_update_permute_half_cuda(const half* x, half* prev, int8_t* out_q, const float* max_val_ptr, int N, int C, int H, int W, cudaStream_t stream) {
    dim3 threads(32, 8);
    dim3 blocks(
        std::min((C + 63) / 64, 32),
        std::min((H * W + 63) / 64, 128),
        std::min(N, 64)
    );
    modiff_update_permute_kernel<half><<<blocks, threads, 0, stream>>>(x, prev, out_q, max_val_ptr, N, C, H, W);
}

} // extern "C"
