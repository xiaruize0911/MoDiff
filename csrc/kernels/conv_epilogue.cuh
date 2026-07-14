// Internal header shared by conv_epilogue.cu, conv2d_int8.cu and conv2d_int4.cu.
//
// The CUTLASS convolutions in this project run in two phases: first the raw
// INT8xINT8 (or INT4xINT4) matmul accumulates into an int32/float32 buffer,
// then one of these kernels applies the per-output-channel dequant scale
// (and optionally bias) and writes/accumulates into the final buffer. This
// second phase is small and layout-simple enough to hand-write instead of
// using a CUTLASS epilogue, and is shared between the INT8 and INT4 paths.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__device__ __forceinline__ float bias_value(const float* bias, int ch) {
    return bias[ch];
}

__device__ __forceinline__ float bias_value(const __half* bias, int ch) {
    return __half2float(bias[ch]);
}

// Defined in conv_epilogue.cu; declared here so conv2d_int8.cu / conv2d_int4.cu
// can launch them directly (kernel launches only need a __global__ declaration
// visible at the call site — this does not require relocatable device code).
__global__ void scale_accumulate_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    float* __restrict__ o_hat_cache,
    int num_elements,
    int num_channels
);

__global__ void scale_accumulate_half_cache_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ o_hat_cache,
    int num_elements,
    int num_channels
);

__global__ void scale_store_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    float* __restrict__ output,
    int num_elements,
    int num_channels
);

__global__ void scale_store_half_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
);

// Templates must be fully defined here (not just declared) so each including
// .cu instantiates its own copy for BiasT in {float, __half}.
//
// Vectorized float4 path is only taken when the 4-lane group doesn't straddle
// a channel boundary (`(base % num_channels) <= num_channels - 4`); otherwise
// it falls back to a scalar loop over exactly this thread's own elements
// (`min(base + 4, num_elements)` — never scans past this thread's chunk, so
// concurrent threads never write overlapping ranges).
template <typename BiasT>
__global__ void scale_bias_store_half_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    const BiasT* __restrict__ bias,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float value = conv_output[i] * weight_scale[ch] + bias_value(bias, ch);
        output[i] = __float2half_rn(value);
    }
}

// Same as scale_bias_store_half_kernel, but also adds a per-element residual
// (channels_last FP16, same layout/shape as output) -- fuses a ResBlock's skip
// connection add into the conv's dequant/bias store, so the conv writes
// conv*weight_scale[ch] + bias[ch] + residual[i] in one pass instead of a
// separate aten::add. `bias` may be nullptr (skips the bias term). Accumulates in
// FP32 and rounds once (slightly more accurate than the fp16-accumulated add it
// replaces). Scalar loop: residual[i] aligns with output[i] under identical
// channels_last layout.
template <typename BiasT>
__global__ void scale_bias_residual_store_half_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    const BiasT* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float value = conv_output[i] * weight_scale[ch]
                    + (bias != nullptr ? bias_value(bias, ch) : 0.0f)
                    + __half2float(residual[i]);
        output[i] = __float2half_rn(value);
    }
}

// INT8-OUTPUT (requantizing) epilogue for conv->conv int8 chaining. Dequantizes
// the accumulator (conv*weight_scale[ch] + bias[ch]) to real units, optionally
// applies ReLU (folds the activation into the requantize since ReLU commutes with
// positive scaling), then re-quantizes by `*requant_scale_ptr` (= the NEXT conv's
// 127/absmax input scale) straight to int8 -- so the following conv reads int8
// directly with no fp16 round-trip + separate quantize. `bias` may be nullptr.
// Post-ReLU values land in [0,127]; the negative codes go unused (no zero-point in
// the s8xs8 GEMM). Reuses the clamp/round-to-int8 idiom from quantize.cu.
template <typename BiasT>
__global__ void scale_bias_relu_requant_store_int8_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    const BiasT* __restrict__ bias,
    const float* __restrict__ requant_scale_ptr,
    int8_t* __restrict__ output,
    int num_elements,
    int num_channels,
    bool apply_relu
) {
    const float rq = *requant_scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float value = conv_output[i] * weight_scale[ch]
                    + (bias != nullptr ? bias_value(bias, ch) : 0.0f);
        if (apply_relu) value = fmaxf(value, 0.0f);
        output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(value * rq)));
    }
}

// INT8-output store from an ALREADY-dequantized fp16 conv output (the CUTLASS
// deep-fuse epilogue Int8DequantScaleSource already applied per-channel
// weight_scale, writing fp16 with no fp32 temporary). Adds per-channel bias,
// optional ReLU, and requantizes to int8 by *requant_scale_ptr. This is the
// deep-fuse int8 chaining store: GEMM->fp16(dequant) then this, avoiding the
// fp32 intermediate the scale_bias_relu_requant_store_int8_kernel path pays.
template <typename BiasT>
__global__ void bias_relu_requant_store_int8_from_half_kernel(
    const __half* __restrict__ deq,        // fp16, = acc * alpha * weight_scale[ch]
    const BiasT* __restrict__ bias,
    const float* __restrict__ requant_scale_ptr,
    int8_t* __restrict__ output,
    int num_elements,
    int num_channels,
    bool apply_relu
) {
    const float rq = *requant_scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float value = __half2float(deq[i]) + (bias != nullptr ? bias_value(bias, ch) : 0.0f);
        if (apply_relu) value = fmaxf(value, 0.0f);
        output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(value * rq)));
    }
}

// FP16-output store from an already-dequantized fp16 conv output (deep-fuse GEMM,
// no fp32 temp): adds per-channel bias + a per-element fp16 residual (the ResBlock
// skip). The FP16-out counterpart of bias_relu_requant_store_int8_from_half_kernel,
// used to deep-fuse + tune the ResNet conv3 residual path (was fp32-temp).
template <typename BiasT>
__global__ void bias_residual_store_half_from_half_kernel(
    const __half* __restrict__ deq,
    const BiasT* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float value = __half2float(deq[i])
                    + (bias != nullptr ? bias_value(bias, ch) : 0.0f)
                    + __half2float(residual[i]);
        output[i] = __float2half_rn(value);
    }
}

// DUAL-OUTPUT store for cross-block int8 chaining (ResNet "block-entry-quantize
// fusion"). From an already-dequantized fp16 conv output (deep-fuse GEMM, no fp32
// temp), computes v = deq + bias[ch] + residual (the ResBlock skip add), applies
// ReLU, then writes BOTH:
//   out_half[i]  = fp16 v            (the post-ReLU block output x_{N+1}, needed as
//                                     the next block's fp16 identity/residual), and
//   out_int8[i]  = round(clamp(v * rq))  (v requantized by the NEXT block conv1's
//                                     127/absmax input scale).
// This lets the previous block's conv3 emit the next block's conv1 input int8 in
// the same pass that already materializes the fp16 output -- eliminating the
// standalone per-block-entry quantize kernel. `bias` may be nullptr. ReLU is folded
// in (commutes with positive scaling), so the outer F.relu is also dropped for
// chained-boundary blocks. Post-ReLU int8 codes land in [0,127].
template <typename BiasT>
__global__ void bias_residual_relu_dual_store_from_half_kernel(
    const __half* __restrict__ deq,        // fp16, = acc * alpha * weight_scale[ch]
    const BiasT* __restrict__ bias,
    const __half* __restrict__ residual,
    const float* __restrict__ requant_scale_ptr,
    __half* __restrict__ out_half,
    int8_t* __restrict__ out_int8,
    int num_elements,
    int num_channels,
    bool apply_relu
) {
    const float rq = *requant_scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float value = __half2float(deq[i])
                    + (bias != nullptr ? bias_value(bias, ch) : 0.0f)
                    + __half2float(residual[i]);
        if (apply_relu) value = fmaxf(value, 0.0f);
        out_half[i] = __float2half_rn(value);
        out_int8[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(value * rq)));
    }
}

template <typename BiasT>
__global__ void scale_bias_store_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    const BiasT* __restrict__ bias,
    float* __restrict__ output,
    int num_elements,
    int num_channels
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements && (base % num_channels) <= num_channels - 4) {
        float4 conv_v = reinterpret_cast<const float4*>(conv_output)[idx4];
        int ch_base = base % num_channels;
        float4 out_v;
        out_v.x = conv_v.x * weight_scale[ch_base] + bias_value(bias, ch_base);
        out_v.y = conv_v.y * weight_scale[ch_base + 1] + bias_value(bias, ch_base + 1);
        out_v.z = conv_v.z * weight_scale[ch_base + 2] + bias_value(bias, ch_base + 2);
        out_v.w = conv_v.w * weight_scale[ch_base + 3] + bias_value(bias, ch_base + 3);
        reinterpret_cast<float4*>(output)[idx4] = out_v;
    } else {
        int end = min(base + 4, num_elements);
        for (int i = base; i < end; i++) {
            int ch = i % num_channels;
            output[i] = conv_output[i] * weight_scale[ch] + bias_value(bias, ch);
        }
    }
}

