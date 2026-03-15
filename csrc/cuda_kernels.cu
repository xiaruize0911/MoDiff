#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

// Macro for error checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::ChannelsLast), #x " must be channels_last contiguous")

// Architecture: Ampere (Sm80)
using Arch = cutlass::arch::Sm80;

// =========================================================================
// INT8 Kernel Definition
// =========================================================================

// Fused scale + quantize to INT8 kernel (vectorized float4 loads).
// Replaces: x_scaled = x * scale; x_int8 = x_scaled.round().clamp(-127, 127).to(int8)
// Scale is read from device memory (no CPU sync needed).
__global__ void scale_quantize_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 in_v = reinterpret_cast<const float4*>(input)[idx4];
        int8_t o0 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.x * scale)));
        int8_t o1 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.y * scale)));
        int8_t o2 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.z * scale)));
        int8_t o3 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.w * scale)));
        // 32-bit coalesced write for 4 int8 values
        reinterpret_cast<int32_t*>(output)[idx4] =
            ((unsigned char)o0) | ((unsigned char)o1 << 8) |
            ((unsigned char)o2 << 16) | ((unsigned char)o3 << 24);
    } else {
        // Scalar tail
        for (int i = base; i < num_elements; i++) {
            output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(input[i] * scale)));
        }
    }
}

// Fused dequant + accumulate for INT4 cache update (vectorized float4).
// Replaces: a_hat_cache += round(clamp(residual * scale, -7, 7)) / scale
// 5 separate kernel launches → 1
__global__ void dequant_accumulate_int4_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];
        float4 c_v = reinterpret_cast<float4*>(a_hat_cache)[idx4];
        c_v.x += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.x * scale))) * inv_scale;
        c_v.y += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.y * scale))) * inv_scale;
        c_v.z += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.z * scale))) * inv_scale;
        c_v.w += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.w * scale))) * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            a_hat_cache[i] += fmaxf(-7.0f, fminf(7.0f, roundf(r * scale))) * inv_scale;
        }
    }
}

// Fused dequant + accumulate for INT8 cache update (vectorized float4).
// Replaces: a_hat_cache += round(clamp(residual * scale, -127, 127)) / scale
__global__ void dequant_accumulate_int8_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];
        float4 c_v = reinterpret_cast<float4*>(a_hat_cache)[idx4];
        c_v.x += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.x * scale))) * inv_scale;
        c_v.y += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.y * scale))) * inv_scale;
        c_v.z += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.z * scale))) * inv_scale;
        c_v.w += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.w * scale))) * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            a_hat_cache[i] += fmaxf(-127.0f, fminf(127.0f, roundf(r * scale))) * inv_scale;
        }
    }
}

// Fused step 1: quantize residual and accumulate into a_hat_cache simultaneously
__global__ void quantize_and_update_ahat_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    
    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];
        float4 c_v = reinterpret_cast<const float4*>(a_hat_cache)[idx4];
        
        float q0 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.x * scale)));
        float q1 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.y * scale)));
        float q2 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.z * scale)));
        float q3 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.w * scale)));
        
        c_v.x += q0 * inv_scale;
        c_v.y += q1 * inv_scale;
        c_v.z += q2 * inv_scale;
        c_v.w += q3 * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
        
        reinterpret_cast<int32_t*>(output_int8)[idx4] =
            ((unsigned char)(int8_t)q0) | ((unsigned char)(int8_t)q1 << 8) |
            ((unsigned char)(int8_t)q2 << 16) | ((unsigned char)(int8_t)q3 << 24);
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            float q = fmaxf(-127.0f, fminf(127.0f, roundf(r * scale)));
            a_hat_cache[i] += q * inv_scale;
            output_int8[i] = (int8_t)q;
        }
    }
}

// =========================================================================
// Fused: o_hat_cache += conv_output * weight_scale (per-channel broadcast)
// Vectorized float4 loads for better memory bandwidth.
// =========================================================================

__global__ void scale_accumulate_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    float* __restrict__ o_hat_cache,
    int num_elements,
    int num_channels  // K (must be divisible by 4 for vectorized path)
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 conv_v = reinterpret_cast<const float4*>(conv_output)[idx4];
        float4 cache_v = reinterpret_cast<float4*>(o_hat_cache)[idx4];
        int ch_base = base % num_channels;
        float s0 = weight_scale[ch_base];
        float s1 = weight_scale[ch_base + 1];
        float s2 = weight_scale[ch_base + 2];
        float s3 = weight_scale[ch_base + 3];
        cache_v.x += conv_v.x * s0;
        cache_v.y += conv_v.y * s1;
        cache_v.z += conv_v.z * s2;
        cache_v.w += conv_v.w * s3;
        reinterpret_cast<float4*>(o_hat_cache)[idx4] = cache_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            int ch = i % num_channels;
            o_hat_cache[i] += conv_output[i] * weight_scale[ch];
        }
    }
}

void scale_accumulate(
    torch::Tensor conv_output,
    torch::Tensor weight_scale,
    torch::Tensor o_hat_cache
) {
    int num_elements = conv_output.numel();
    int num_channels = weight_scale.numel();
    int block_size = 256;
    // Each thread processes 4 elements (float4)
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    scale_accumulate_kernel<<<grid_size, block_size>>>(
        conv_output.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        o_hat_cache.data_ptr<float>(),
        num_elements,
        num_channels
    );
}

// =========================================================================
// Fused: residual = x - cache, absmax reduction, scale + inv_scale output
// Replaces 6 separate kernel launches (sub, abs, amax, clamp, div, div)
// with 1 cooperative kernel using atomic float max + block counting.
// =========================================================================

__global__ void sub_absmax_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ a_hat_cache,
    float* __restrict__ residual,
    float* __restrict__ absmax_buf,     // Must be 0 on entry (self-resetting)
    float* __restrict__ scale_out,      // Q_level / max(absmax, eps)
    float* __restrict__ inv_scale_out,  // max(absmax, eps) / Q_level  (CUTLASS alpha)
    unsigned int* __restrict__ retire_count, // Must be 0 on entry (self-resetting)
    const float* __restrict__ smooth_inv,  // Per-channel SmoothQuant inverse (NULL=skip)
    int num_channels,                    // C for NHWC channel indexing
    float Q_level,                      // 7.0 for INT4, 127.0 for INT8
    int num_elements
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    
    // Step 1: Compute residual (with optional SmoothQuant) and find thread-local abs max
    float local_max = 0.0f;
    int num_elements4 = num_elements / 4;
    // Vectorized loop: process 4 elements at a time
    for (int i4 = blockIdx.x * blockDim.x + tid; i4 < num_elements4; i4 += num_threads) {
        float4 x_v = reinterpret_cast<const float4*>(x)[i4];
        float4 c_v = reinterpret_cast<const float4*>(a_hat_cache)[i4];
        // Fused SmoothQuant: x_smooth = x * smooth_inv[channel]
        if (smooth_inv != nullptr) {
            int ch = (i4 * 4) % num_channels;
            x_v.x *= smooth_inv[ch];
            x_v.y *= smooth_inv[ch + 1];
            x_v.z *= smooth_inv[ch + 2];
            x_v.w *= smooth_inv[ch + 3];
        }
        float4 r_v;
        r_v.x = x_v.x - c_v.x;
        r_v.y = x_v.y - c_v.y;
        r_v.z = x_v.z - c_v.z;
        r_v.w = x_v.w - c_v.w;
        reinterpret_cast<float4*>(residual)[i4] = r_v;
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(r_v.x), fabsf(r_v.y)),
                                            fmaxf(fabsf(r_v.z), fabsf(r_v.w))));
    }
    // Scalar tail for remaining elements
    int tail_start = num_elements4 * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + tid; i < num_elements; i += num_threads) {
        float xval = x[i];
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        float r = xval - a_hat_cache[i];
        residual[i] = r;
        local_max = fmaxf(local_max, fabsf(r));
    }
    
    // Step 2: Block-level reduction in shared memory
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    
    if (tid == 0) {
        // Step 3: Atomic float max across blocks (CAS-based)
        float val = sdata[0];
        unsigned int* addr = (unsigned int*)absmax_buf;
        unsigned int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
        } while (assumed != old);
        
        // Step 4: Count completed blocks; last block computes + writes scale
        __threadfence();  // Ensure atomicCAS is visible before counting
        unsigned int ticket = atomicAdd(retire_count, 1u);
        if (ticket == gridDim.x - 1) {
            float absmax = *absmax_buf;
            float am = fmaxf(absmax, 1e-6f);
            *scale_out = Q_level / am;
            *inv_scale_out = am / Q_level;
            // Self-reset for next invocation (safe: next kernel starts after this one)
            *absmax_buf = 0.0f;
            *retire_count = 0;
        }
    }
}

// C++ wrapper: performs subtraction + absmax reduction + scale computation in one kernel.
// Outputs: residual, scale_out (Q/max), inv_scale_out (max/Q = CUTLASS alpha).
// absmax_buf and retire_count must be persistent, initialized to 0 (kernel self-resets).
void sub_absmax_scale(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual,
    torch::Tensor absmax_buf,      // 1-element float32, init 0
    torch::Tensor scale_out,       // 1-element float32 output
    torch::Tensor inv_scale_out,   // 1-element float32 output
    torch::Tensor retire_count,    // 1-element int32, init 0
    float Q_level,
    torch::Tensor smooth_inv       // Per-channel SmoothQuant inverse (empty = skip)
) {
    int num_elements = x.numel();
    int block_size = 256;
    int grid_size = min((num_elements + block_size - 1) / block_size, 1024);
    
    // Determine smooth_inv pointer and num_channels
    const float* smooth_ptr = nullptr;
    int num_channels = 0;
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }
    
    sub_absmax_scale_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        residual.data_ptr<float>(),
        absmax_buf.data_ptr<float>(),
        scale_out.data_ptr<float>(),
        inv_scale_out.data_ptr<float>(),
        (unsigned int*)retire_count.data_ptr<int>(),
        smooth_ptr,
        num_channels,
        Q_level,
        num_elements
    );
}

// =========================================================================
// Helper: Fast Quantization + Packing Kernel
// =========================================================================

__global__ void quantize_pack_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    int num_elements // Number of OUTPUT packed bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    // Use float2 vectorized load for 2 adjacent floats
    float2 vals = reinterpret_cast<const float2*>(input)[idx];

    int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(vals.x)));
    int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(vals.y)));
    
    output[idx] = (i0 & 0x0F) | ((i1 & 0x0F) << 4);
}

// Fused version: applies per-tensor scale before quantizing and packing.
// Eliminates a separate x*scale kernel launch + global memory round-trip.
// Scale is read from device memory (no CPU sync needed).
__global__ void scale_quantize_pack_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements // Number of OUTPUT packed bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    float scale = *scale_ptr;
    float2 vals = reinterpret_cast<const float2*>(input)[idx];
    float v0 = vals.x * scale;
    float v1 = vals.y * scale;

    int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v0)));
    int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v1)));
    
    output[idx] = (i0 & 0x0F) | ((i1 & 0x0F) << 4);
}

torch::Tensor quantize_and_pack(torch::Tensor input) {
    // Input: FP32 [N, H, W, C]
    // Output: Int8 [N, H, W, C/2] (Packed Int4)
    
    int num_input = input.numel();
    int num_output = num_input / 2;
    
    auto output = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    
    int block_size = 256;
    int grid_size = (num_output + block_size - 1) / block_size;
    
    quantize_pack_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        num_output
    );
    
    // Reshape logic: [N, C, H, W] -> [N, H, W, C/2]
    // Input is NCHW logical, but NHWC physical.
    // Result treated as NHWC-like packed tensor.
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    return output.view({N, H, W, C/2});
}

torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale) {
    // Fused version: input * scale -> quantize -> pack
    // Input: FP32 [N, C, H, W] (NHWC physical)
    // Scale: FP32 scalar tensor on GPU (read by kernel, no CPU sync)
    // Output: Int8 [N, H, W, C/2] (Packed Int4)
    
    int num_input = input.numel();
    int num_output = num_input / 2;
    
    auto output = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    
    int block_size = 256;
    int grid_size = (num_output + block_size - 1) / block_size;
    
    scale_quantize_pack_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        num_output
    );
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    return output.view({N, H, W, C/2});
}

torch::Tensor scale_quantize_int8(torch::Tensor input, torch::Tensor scale) {
    // Fused: input * scale -> round -> clamp(-127,127) -> int8 (vectorized float4)
    auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    
    int num_elements = input.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    scale_quantize_int8_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        num_elements
    );
    
    return output;
}

void dequant_accumulate_int4(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale) {
    // In-place: a_hat_cache += round(clamp(residual * scale, -7, 7)) / scale
    int num_elements = residual.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    dequant_accumulate_int4_kernel<<<grid_size, block_size>>>(
        residual.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        scale.data_ptr<float>(),
        num_elements
    );
}

void dequant_accumulate_int8(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale) {
    // In-place: a_hat_cache += round(clamp(residual * scale, -127, 127)) / scale
    int num_elements = residual.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    dequant_accumulate_int8_kernel<<<grid_size, block_size>>>(
        residual.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        scale.data_ptr<float>(),
        num_elements
    );
}

// 1. Define the Kernel using DefaultConv2dFprop
using Conv2dInt8Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  int8_t, cutlass::layout::TensorNHWC,
  int8_t, cutlass::layout::TensorNHWC,
  float, cutlass::layout::TensorNHWC,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  Arch,
  cutlass::gemm::GemmShape<128, 128, 128>,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<16, 8, 32>,
  cutlass::epilogue::thread::LinearCombination<
    float, 
    1, // ElementCount for vector load from Accum/C
    int32_t, 
    float,
    cutlass::epilogue::thread::ScaleType::Default
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3, // Stages (was 2, 3 improves memory latency pipelining)
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

// 2. Define the Device Operator
using Conv2dInt8Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt8Kernel>;

// =========================================================================
// INT4 Kernel Definition
// =========================================================================

using Conv2dInt4Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  cutlass::int4b_t, cutlass::layout::TensorNHWC,
  cutlass::int4b_t, cutlass::layout::TensorNHWC,
  float, cutlass::layout::TensorNHWC,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  Arch,
  cutlass::gemm::GemmShape<128, 128, 128>, 
  cutlass::gemm::GemmShape<64, 64, 128>,
  cutlass::gemm::GemmShape<16, 8, 64>, // INT4 often uses K=64 for instruction shape on Ampere
  cutlass::epilogue::thread::LinearCombination<
    float, 1, int32_t, float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3, // Stages (was 2, 3 improves memory latency pipelining)
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

using Conv2dInt4Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt4Kernel>;

// =========================================================================
// Implementation: INT8
// =========================================================================

torch::Tensor conv2d_int8_fprop(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,     
    torch::Tensor bias,       
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    // Permuted weight (K, R, S, C)
    int R = weight.size(1);
    int S = weight.size(2);
    // int C = weight.size(3); // C from input

    
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto output = torch::empty({N, K, H_out, W_out}, options);
    
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C},
        {K, R, S, C},
        {padding_h, padding_h, padding_w, padding_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {N, H_out, W_out, K},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );
    
    int8_t* input_ptr = reinterpret_cast<int8_t*>(input.data_ptr());
    int8_t* weight_ptr = reinterpret_cast<int8_t*>(weight.data_ptr());
    float* output_ptr = output.data_ptr<float>();
    float* bias_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    
    // Build epilogue params: use device pointer when scales is on GPU to avoid D2H sync
    using Int8EpilogueParams = typename Conv2dInt8Op::EpilogueOutputOp::Params;
    Int8EpilogueParams ep;
    bool use_device_alpha = (scales.numel() == 1 && scales.is_cuda() && bias.numel() == 0);
    if (use_device_alpha) {
        // Device pointer path: CUTLASS reads alpha from GPU memory, beta=0 (no bias)
        ep = Int8EpilogueParams(scales.data_ptr<float>());
    } else {
        float alpha = 1.0f;
        if (scales.numel() == 1) alpha = scales.item<float>();
        float beta = (bias.numel() > 0) ? 1.0f : 0.0f;
        ep = Int8EpilogueParams(alpha, beta);
    }
    
    Conv2dInt8Op op;
    Conv2dInt8Op::Arguments args(
        problem_size,
        {input_ptr, {C, W * C, H * W * C}},
        {weight_ptr, {C, S * C, R * S * C}},
        {bias_ptr, {0,0,0}}, // Broadcast bias
        {output_ptr, {K, W_out * K, H_out * W_out * K}},
        ep
    );
    
    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    
    cutlass::Status status = op(args, workspace.data_ptr());
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT8 Kernel Failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS Kernel failed");
    }

    return output;
}

// =========================================================================
// Implementation: INT4
// =========================================================================

torch::Tensor conv2d_int4_fprop(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    // Previous Code Assumed NCHW input and performed check
    // Now we accept packed input so we disable these generic checks
    // as we handle specific checks below.
    CHECK_CUDA(input);
    // CHECK_CONTIGUOUS(input);
    
    // Correct Dimension Extraction for Packed (NHWC-like) Tensors
    // Input is (N, H, W, C/2) contiguous
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    // input.size(3) is C_packed (C/2)
    
    // Weight is (K, R, S, C/2) contiguous
    int K_packed = weight_packed.size(0);
    int R = weight_packed.size(1);
    int S = weight_packed.size(2);
    int C_packed = weight_packed.size(3); 
    
    int C_logical = C_packed * 2; 

    // Safety checks
    // TORCH_CHECK(input.is_contiguous(), "Input must be contiguous (N, H, W, C/2)");
    // TORCH_CHECK(weight_packed.is_contiguous(), "Weight must be contiguous (K, R, S, C/2)");
    TORCH_CHECK(input.size(3) == C_packed, "Input/Weight channel mismatch");

    void* input_ptr_raw = input.data_ptr();
    
    cutlass::int4b_t* input_ptr = (cutlass::int4b_t*)input_ptr_raw;
    cutlass::int4b_t* weight_ptr = (cutlass::int4b_t*)weight_packed.data_ptr();
    
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    int K_logical = K_packed; 
    
    auto out_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto output = torch::empty({N, K_logical, H_out, W_out}, out_options);
    
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C_logical},
        {K_packed, R, S, C_logical},
        {padding_h, padding_h, padding_w, padding_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {N, H_out, W_out, K_packed},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );

    // Build epilogue params: use device pointer when scales is on GPU to avoid D2H sync
    using Int4EpilogueParams = typename Conv2dInt4Op::EpilogueOutputOp::Params;
    Int4EpilogueParams ep;
    bool use_device_alpha = (scales.numel() == 1 && scales.is_cuda() && bias.numel() == 0);
    if (use_device_alpha) {
        ep = Int4EpilogueParams(scales.data_ptr<float>());
    } else {
        float alpha = 1.0f;
        if (scales.numel() == 1) alpha = scales.item<float>();
        float beta = (bias.numel() > 0) ? 1.0f : 0.0f;
        ep = Int4EpilogueParams(alpha, beta);
    }

    Conv2dInt4Op op;
    Conv2dInt4Op::Arguments args(
        problem_size,
        {input_ptr, {C_logical, W * C_logical, H * W * C_logical}},
        {weight_ptr, {C_logical, S * C_logical, R * S * C_logical}}, 
        {(float*)((bias.numel() > 0) ? bias.data_ptr() : nullptr), {0,0,0}},
        {output.data_ptr<float>(), {K_packed, W_out * K_packed, H_out * W_out * K_packed}},
        ep
    );
    
    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    
    cutlass::Status status = op(args, workspace.data_ptr());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT4 Kernel Failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT4 Kernel failed");
    }

    return output;
}

// =========================================================================
// Implementation: Fused Conv + O_Hat Accumulate
// =========================================================================

torch::Tensor conv2d_int8_fprop_o_hat(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,      // scalar tensor for CUTLASS alpha
    torch::Tensor weight_scales,  // per-channel vector
    torch::Tensor o_hat_cache,    // in-place output accumulate target
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    // 1) Empty bias
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    
    // 2) Run CUTLASS Conv (outputs int32 accumulator scaled by inv_scale)
    auto conv_out = conv2d_int8_fprop(
        input, weight, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );
    
    // 3) Setup thread grid for scale_accumulate
    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    // 4) Launch native CUDA kernel for per-channel scale + accumulation into o_hat_cache
    // this directly overwrites o_hat_cache (C/D matrix) in global memory.
    scale_accumulate_kernel<<<grid_size, block_size>>>(
        conv_out.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        o_hat_cache.data_ptr<float>(),
        num_elements,
        num_channels
    );
    
    // Return o_hat_cache itself for identical graph tracking
    return o_hat_cache;
}


__global__ void quantize_pack_and_update_ahat_kernel_int4(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    
    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];
        float4 c_v = reinterpret_cast<const float4*>(a_hat_cache)[idx4];
        
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.x * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.y * scale)));
        float q2 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.z * scale)));
        float q3 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.w * scale)));
        
        c_v.x += q0 * inv_scale;
        c_v.y += q1 * inv_scale;
        c_v.z += q2 * inv_scale;
        c_v.w += q3 * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
        
        int8_t b0 = ((int8_t)q0 & 0x0F) | (((int8_t)q1 & 0x0F) << 4);
        int8_t b1 = ((int8_t)q2 & 0x0F) | (((int8_t)q3 & 0x0F) << 4);
        
        // Write 2 bytes (1 int16)
        reinterpret_cast<int16_t*>(output_packed)[idx4] = (uint16_t)(uint8_t)b0 | ((uint16_t)(uint8_t)b1 << 8);
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            float q = fmaxf(-7.0f, fminf(7.0f, roundf(r * scale)));
            a_hat_cache[i] += q * inv_scale;
            // The tail quantization is technically not beautifully packed for output_packed 
            // since one byte holds 2 elements. Assuming num_elements is always even.
            if (i % 2 == 0) {
                float r_next = (i + 1 < num_elements) ? residual[i + 1] : 0.0f;
                float q_next = fmaxf(-7.0f, fminf(7.0f, roundf(r_next * scale)));
                
                int8_t b_curr = ((int8_t)q & 0x0F) | (((int8_t)q_next & 0x0F) << 4);
                output_packed[i / 2] = b_curr;
            }
        }
    }
}

torch::Tensor step1_quantize_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
) {
    // 1) Compute sub_absmax_scale globally
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);
    
    // 2) Allocate int8 output
    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    
    // 3) Fused quantize + accumulate into a_hat_cache
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    quantize_and_update_ahat_kernel<<<grid_size, block_size>>>(
        residual_buf.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        x_int8.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_elements
    );
    
    return x_int8;
}


torch::Tensor step1_quantize_pack_int4_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
) {
    // 1) Compute sub_absmax_scale globally
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);
    
    // 2) Allocate int4 output
    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    
    // 3) Fused quantize + accumulate into a_hat_cache
    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    quantize_pack_and_update_ahat_kernel_int4<<<grid_size, block_size>>>(
        residual_buf.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        x_packed.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_input
    );
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    return x_packed.view({N, H, W, C/2});
}

torch::Tensor conv2d_int4_fprop_o_hat(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,      // scalar tensor for CUTLASS alpha
    torch::Tensor weight_scales,  // per-channel vector
    torch::Tensor o_hat_cache,    // in-place output accumulate target
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    
    auto conv_out = conv2d_int4_fprop(
        input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );
    
    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    
    scale_accumulate_kernel<<<grid_size, block_size>>>(
        conv_out.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        o_hat_cache.data_ptr<float>(),
        num_elements,
        num_channels
    );
    
    return o_hat_cache;
}

