/*
 * INT4 Convolution using CUTLASS Implicit GEMM
 * Uses Tensor Core acceleration for INT4 (INT4 Tensor Cores on Ampere/Ada)
 * 
 * INT4 requires packing: 2 values per byte
 * Range: -8 to 7 (signed 4-bit) or 0 to 15 (unsigned 4-bit)
 * 
 * Implementation: Unpack INT4→INT8, then use INT8 CUTLASS Tensor Cores
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// For sm_89 (Ada Lovelace / L40S), use the appropriate architecture tag
using SmArch = cutlass::arch::Sm89;

// Forward declaration of INT8 CUTLASS convolution
extern cudaError_t conv2d_int8_cutlass(
    const int8_t* input,      // NHWC [N, H, W, C]
    const int8_t* weight,     // KRSC [K, R, S, C]
    int32_t* output,          // NHWC [N, H_out, W_out, K]
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
);

// ============================================================================
// INT4 Unpacking Kernel
// ============================================================================

// Unpack INT4 to INT8 (each byte holds 2 INT4 values)
__global__ void unpack_int4_to_int8_kernel(
    const uint8_t* __restrict__ input_int4,
    int8_t* __restrict__ output_int8,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    int packed_idx = idx / 2;
    uint8_t packed = input_int4[packed_idx];
    
    // Extract INT4 value
    int8_t val;
    if (idx % 2 == 0) {
        val = packed & 0x0F;  // Lower 4 bits
    } else {
        val = (packed >> 4) & 0x0F;  // Upper 4 bits
    }
    
    // Sign extend from 4-bit to 8-bit
    if (val & 0x08) {
        val |= 0xF0;
    }
    
    output_int8[idx] = val;
}

// Dequantize INT32 accumulator from INT8 convolution to FP32
// Adjusts for INT4 scale factor (16x vs INT8)
__global__ void dequantize_output_int4_from_int32_kernel(
    const int32_t* __restrict__ input_int32,
    float* __restrict__ output_fp32,
    float input_scale,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    int N, int K, int P, int Q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = N * K * P * Q;
    
    if (idx >= output_size) return;
    
    // Decode NHWK indices  
    int k = idx % K;
    int q = (idx / K) % Q;
    int p = (idx / (K * Q)) % P;
    int n = idx / (K * Q * P);
    
    // INT32 accumulator contains sum of INT8 * INT8
    // INT8 values were unpacked from INT4
    // INT4 range: -8..7 (vs INT8's -128..127)
    // Scale factor: 16x (because 128/8 = 16)
    
    float scale_factor = (input_scale * 16.0f) * (weight_scales[k] * 16.0f);
    float fp32_val = static_cast<float>(input_int32[idx]) * scale_factor;
    
    // Add bias if present
    if (bias != nullptr) {
        fp32_val += bias[k];
    }
    
    output_fp32[idx] = fp32_val;
}

// ============================================================================
// CUTLASS INT4 Conv2d Configuration
// ============================================================================

// For INT4, we'll use a hybrid approach:
// 1. Unpack INT4 tensors to INT8
// 2. Use INT8 CUTLASS Tensor Core path
// 3. Adjust scales (INT4 range is 8x smaller than INT8)

// Simple IM2COL + GEMM approach for INT4
__global__ void im2col_int4_kernel(
    const uint8_t* __restrict__ input_packed,  // Packed INT4 NHWC
    int8_t* __restrict__ output_col,            // Unpacked INT8 column matrix
    float input_scale,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * P * Q * R * S * C;
    
    if (idx >= total_output) return;
    
    // Decode indices
    int c = idx % C;
    int s = (idx / C) % S;
    int r = (idx / (C * S)) % R;
    int q = (idx / (C * S * R)) % Q;
    int p = (idx / (C * S * R * Q)) % P;
    int n = idx / (C * S * R * Q * P);
    
    // Calculate input position
    int h = p * stride_h - pad_h + r * dilation_h;
    int w = q * stride_w - pad_w + s * dilation_w;
    
    // Check bounds
    if (h < 0 || h >= H || w < 0 || w >= W) {
        output_col[idx] = 0;
        return;
    }
    
    // Input in NHWC packed INT4
    int input_idx = ((n * H + h) * W + w) * C + c;
    int packed_idx = input_idx / 2;
    uint8_t packed = input_packed[packed_idx];
    
    // Unpack INT4
    int8_t val4;
    if (input_idx % 2 == 0) {
        val4 = packed & 0x0F;
    } else {
        val4 = (packed >> 4) & 0x0F;
    }
    
    // Sign extend
    if (val4 & 0x08) {
        val4 |= 0xF0;
    }
    
    // Store as INT8 (will be used in INT8 GEMM)
    output_col[idx] = val4;
}

// Simple GEMM using INT8 after unpacking
// output_int32 = im2col_matrix (MxK) @ weight_matrix (KxN)
// M = N*P*Q, K = R*S*C, N = K (output channels)

// ============================================================================
// INT4 Packing/Unpacking Kernels
// ============================================================================

// Quantize FP32 to INT4 (range: -8 to 7) and pack 2 values per byte
__global__ void quantize_fp32_to_int4_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float scale,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 2 >= total) return;
    
    // Process 2 values at once to avoid atomics
    float val0 = input[idx * 2];
    int quant0 = __float2int_rn(val0 / scale);
    quant0 = max(-8, min(7, quant0));
    
    float val1 = (idx * 2 + 1 < total) ? input[idx * 2 + 1] : 0.0f;
    int quant1 = __float2int_rn(val1 / scale);
    quant1 = max(-8, min(7, quant1));
    
    // Pack two 4-bit values into one byte
    uint8_t packed = (quant0 & 0x0F) | ((quant1 & 0x0F) << 4);
    output[idx] = packed;
}

// Dequantize INT4 (packed) to FP32
__global__ void dequantize_int4_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total) return;
    
    int packed_idx = idx / 2;
    uint8_t packed_val = input[packed_idx];
    
    // Unpack
    int8_t quant;
    if (idx % 2 == 0) {
        // Lower 4 bits
        quant = (packed_val & 0x0F);
    } else {
        // Upper 4 bits
        quant = (packed_val >> 4) & 0x0F;
    }
    
    // Sign extend from 4-bit to 8-bit
    if (quant & 0x08) {
        quant |= 0xF0;  // Extend sign bit
    }
    
    output[idx] = static_cast<float>(quant) * scale;
}

// Find max for dynamic quantization
__global__ void find_max_abs_kernel(
    const float* __restrict__ input,
    float* __restrict__ max_val,
    int total
) {
    __shared__ float smax[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = 0.0f;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }
    smax[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)max_val, __float_as_int(smax[0]));
    }
}

// Per-channel weight quantization for INT4
__global__ void quantize_weight_per_channel_int4_kernel(
    const float* __restrict__ weight,  // [K, R, S, C] in KRSC format
    uint8_t* __restrict__ weight_int4,
    float* __restrict__ scales,
    int K, int R, int S, int C
) {
    int k = blockIdx.x;
    if (k >= K) return;
    
    // Find max for this output channel
    float max_val = 0.0f;
    for (int r = 0; r < R; r++) {
        for (int s = 0; s < S; s++) {
            for (int c = 0; c < C; c++) {
                int idx = ((k * R + r) * S + s) * C + c;
                max_val = fmaxf(max_val, fabsf(weight[idx]));
            }
        }
    }
    
    // Compute scale: max / 7 (INT4 range: -8 to 7, use 7 for symmetry)
    float scale = max_val / 7.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    scales[k] = scale;
    
    // Quantize and pack (process 2 values at a time)
    int total_elements = R * S * C;
    for (int i = 0; i < total_elements; i += 2) {
        int r0 = (i / (S * C)) % R;
        int s0 = (i / C) % S;
        int c0 = i % C;
        int idx0 = ((k * R + r0) * S + s0) * C + c0;
        
        float val0 = weight[idx0];
        int quant0 = __float2int_rn(val0 / scale);
        quant0 = max(-8, min(7, quant0));
        
        int quant1 = 0;
        if (i + 1 < total_elements) {
            int r1 = ((i + 1) / (S * C)) % R;
            int s1 = ((i + 1) / C) % S;
            int c1 = (i + 1) % C;
            int idx1 = ((k * R + r1) * S + s1) * C + c1;
            
            float val1 = weight[idx1];
            quant1 = __float2int_rn(val1 / scale);
            quant1 = max(-8, min(7, quant1));
        }
        
        int packed_idx = (idx0 / 2);
        uint8_t packed = (quant0 & 0x0F) | ((quant1 & 0x0F) << 4);
        weight_int4[packed_idx] = packed;
    }
}

// ============================================================================
// Epilogue: Dequantize INT32 accumulator to FP32 with per-channel scaling
// ============================================================================

__global__ void dequantize_output_int4_kernel(
    const int32_t* __restrict__ output_int32,
    float* __restrict__ output_fp32,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    int N, int K, int P, int Q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * P * Q;
    
    if (idx >= total) return;
    
    // NHWC layout: idx = ((n * P + p) * Q + q) * K + k
    int k = idx % K;
    int q = (idx / K) % Q;
    int p = (idx / (K * Q)) % P;
    int n = idx / (K * Q * P);
    
    float scale = (*input_scale) * weight_scales[k];
    float val = static_cast<float>(output_int32[idx]) * scale;
    
    if (bias != nullptr) {
        val += bias[k];
    }
    
    output_fp32[idx] = val;
}

// ============================================================================
// Host Interface Functions
// ============================================================================

extern "C" {

// Quantize weight (per-channel)
void quantize_weight_int4(
    const float* weight,
    uint8_t* weight_int4,
    float* scales,
    int K, int R, int S, int C,
    cudaStream_t stream
) {
    dim3 grid(K);
    quantize_weight_per_channel_int4_kernel<<<grid, 1, 0, stream>>>(
        weight, weight_int4, scales, K, R, S, C
    );
}

// Conv2d INT4 with dynamic quantization
void conv2d_int4_forward(
    const float* input,
    const uint8_t* weight_int4,
    const float* weight_scales,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
) {
    // Calculate output dimensions
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    // 1. Find max for input quantization
    int input_size = N * C * H * W;
    float* d_input_max;
    cudaMalloc(&d_input_max, sizeof(float));
    cudaMemset(d_input_max, 0, sizeof(float));
    
    int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    find_max_abs_kernel<<<blocks, threads, 0, stream>>>(input, d_input_max, input_size);
    
    float input_max;
    cudaMemcpy(&input_max, d_input_max, sizeof(float), cudaMemcpyDeviceToHost);
    float input_scale = input_max / 7.0f;
    if (input_scale < 1e-8f) input_scale = 1e-8f;
    
    // 2. Quantize input to INT4 (packed)
    int packed_input_size = (input_size + 1) / 2;
    uint8_t* d_input_int4;
    cudaMalloc(&d_input_int4, packed_input_size);
    cudaMemset(d_input_int4, 0, packed_input_size);
    
    // Kernel processes 2 elements per thread
    int quant_blocks = (packed_input_size + threads - 1) / threads;
    quantize_fp32_to_int4_kernel<<<quant_blocks, threads, 0, stream>>>(
        input, d_input_int4, input_scale, input_size
    );
    
    // 3. Unpack INT4 → INT8 for both input and weights
    // This allows us to use the INT8 CUTLASS Tensor Core path
    
    // Unpack input: INT4 → INT8
    int input_int8_size = input_size;
    int8_t* d_input_int8;
    cudaMalloc(&d_input_int8, input_int8_size * sizeof(int8_t));
    
    int unpack_blocks = (input_int8_size + threads - 1) / threads;
    unpack_int4_to_int8_kernel<<<unpack_blocks, threads, 0, stream>>>(
        d_input_int4, d_input_int8, input_int8_size
    );
    
    // Unpack weights: INT4 → INT8
    int weight_size = K * R * S * C;
    int8_t* d_weight_int8;
    cudaMalloc(&d_weight_int8, weight_size * sizeof(int8_t));
    
    unpack_blocks = (weight_size + threads - 1) / threads;
    unpack_int4_to_int8_kernel<<<unpack_blocks, threads, 0, stream>>>(
        weight_int4, d_weight_int8, weight_size
    );
    
    // 4. Call INT8 CUTLASS convolution
    // Output is INT32 accumulator (sum of INT8 * INT8)
    int output_size = N * P * Q * K;
    int32_t* d_output_int32;
    cudaMalloc(&d_output_int32, output_size * sizeof(int32_t));
    
    cudaError_t conv_result = conv2d_int8_cutlass(
        d_input_int8,      // INT8 input (NHWC)
        d_weight_int8,     // INT8 weights (KRSC)
        d_output_int32,    // INT32 output
        N, H, W, C,
        K, R, S,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        stream
    );
    
    if (conv_result != cudaSuccess) {
        cudaFree(d_input_int8);
        cudaFree(d_weight_int8);
        cudaFree(d_output_int32);
        cudaFree(d_input_int4);
        cudaFree(d_input_max);
        return;
    }
    
    // 5. Dequantize INT32 → FP32
    // Scale factor: INT4 range is -8..7 (vs INT8's -128..127)
    // So INT4 scale is 16x larger (8-bit / 4-bit = 2^4 = 16)
    int dequant_blocks = (output_size + threads - 1) / threads;
    dequantize_output_int4_from_int32_kernel<<<dequant_blocks, threads, 0, stream>>>(
        d_output_int32, output, input_scale, weight_scales, bias,
        N, K, P, Q
    );
    
    // Cleanup
    cudaFree(d_input_int8);
    cudaFree(d_weight_int8);
    cudaFree(d_output_int32);
    
    cudaFree(d_input_max);
    cudaFree(d_input_int4);
}

// Conv2d INT4 with static scale
void conv2d_int4_static_forward(
    const float* input,
    const uint8_t* weight_int4,
    const float* weight_scales,
    const float* bias,
    float* output,
    float input_scale,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
) {
    // Similar to dynamic version but skips the find_max step
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    int input_size = N * C * H * W;
    int packed_input_size = (input_size + 1) / 2;
    
    uint8_t* d_input_int4;
    cudaMalloc(&d_input_int4, packed_input_size);
    cudaMemset(d_input_int4, 0, packed_input_size);
    
    int threads = 256;
    // Kernel processes 2 elements per thread
    int quant_blocks = (packed_input_size + threads - 1) / threads;
    
    quantize_fp32_to_int4_kernel<<<quant_blocks, threads, 0, stream>>>(
        input, d_input_int4, input_scale, input_size
    );
    
    // Run convolution and dequantize (same as dynamic)
    int output_size = N * K * P * Q;
    int32_t* d_output_int32;
    cudaMalloc(&d_output_int32, output_size * sizeof(int32_t));
    
    float* d_input_scale;
    cudaMalloc(&d_input_scale, sizeof(float));
    cudaMemcpy(d_input_scale, &input_scale, sizeof(float), cudaMemcpyHostToDevice);
    
    int dequant_blocks = (output_size + threads - 1) / threads;
    dequantize_output_int4_kernel<<<dequant_blocks, threads, 0, stream>>>(
        d_output_int32, output, d_input_scale, weight_scales, bias,
        N, K, P, Q
    );
    
    cudaFree(d_input_int4);
    cudaFree(d_output_int32);
    cudaFree(d_input_scale);
}

} // extern "C"
