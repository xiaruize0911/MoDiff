// =========================================================================
// Post-conv per-channel scale/bias epilogue kernels.
//
// conv2d_int8.cu / conv2d_int4.cu run the raw INT8xINT8 / INT4xINT4 CUTLASS
// matmul into an intermediate float32 accumulator; these kernels then apply
// the per-output-channel weight dequant scale (o[i] = acc[i] * weight_scale[ch])
// and either accumulate into a running o_hat cache (scale_accumulate*) or
// store into a fresh output buffer (scale_store*), optionally adding bias
// (scale_bias_store*, in conv_epilogue.cuh since it's a template).
//
// Correctness note: the float4-vectorized kernels below process 4 elements
// per thread. Since the tensor is NHWC-physical (channels_last), 4 elements
// starting at flat index `base` only share a clean `base % num_channels`
// pattern when `num_channels % 4 == 0` (true for every layer in this
// project's models: 64/128/192/256/320/384/512/640/768/1280 are all
// multiples of 4). For other channel counts, a 4-wide group can straddle a
// channel boundary, so every kernel here explicitly checks
// `(base % num_channels) <= num_channels - 4` before taking the vectorized
// path, falling back to a scalar loop bounded to `min(base+4, num_elements)`
// (this thread's own elements only) otherwise. This matters especially for
// scale_accumulate_kernel: without the bound, an out-of-range vectorized
// read/write would corrupt neighboring channels' data, and an unbounded
// scalar fallback would double-accumulate elements touched by more than one
// misaligned thread.
// =========================================================================

#include <ATen/cuda/CUDAContext.h>

#include "conv_epilogue.cuh"

__global__ void scale_accumulate_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    float* __restrict__ o_hat_cache,
    int num_elements,
    int num_channels
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements && (base % num_channels) <= num_channels - 4) {
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
        int end = min(base + 4, num_elements);
        for (int i = base; i < end; i++) {
            int ch = i % num_channels;
            o_hat_cache[i] += conv_output[i] * weight_scale[ch];
        }
    }
}

// Per-element (no vectorization needed), so there is no channel-boundary
// hazard here regardless of num_channels.
__global__ void scale_accumulate_half_cache_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ o_hat_cache,
    int num_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float cache = __half2float(o_hat_cache[i]);
        o_hat_cache[i] = __float2half_rn(cache + conv_output[i] * weight_scale[ch]);
    }
}

// Vectorized (float2/half2) counterpart of scale_accumulate_half_cache_kernel. Reads
// conv_output (fp32) and weight_scale as float2, reads/writes o_hat_cache (fp16) as half2.
// Requires num_channels % 2 == 0 so a pair never straddles a channel boundary (true for
// every real channel count in this project, per this file's header comment); the caller
// only dispatches here when that holds, with a scalar fallback (and odd-tail epilogue)
// otherwise.
__global__ void scale_accumulate_half_cache_vec2_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ o_hat_cache,
    int num_elements,
    int num_channels
) {
    const int stride = blockDim.x * gridDim.x;
    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            int ch = base % num_channels;
            float2 conv_v = *reinterpret_cast<const float2*>(&conv_output[base]);
            float2 scale_v = *reinterpret_cast<const float2*>(&weight_scale[ch]);
            float2 cache_v = __half22float2(*reinterpret_cast<const __half2*>(&o_hat_cache[base]));
            cache_v.x += conv_v.x * scale_v.x;
            cache_v.y += conv_v.y * scale_v.y;
            *reinterpret_cast<__half2*>(&o_hat_cache[base]) = __float22half2_rn(cache_v);
        } else {
            int ch = base % num_channels;
            float cache = __half2float(o_hat_cache[base]);
            o_hat_cache[base] = __float2half_rn(cache + conv_output[base] * weight_scale[ch]);
        }
    }
}

// Same o_hat accumulate as scale_accumulate_half_cache_kernel, but ALSO writes a
// separate `output` = (updated o_hat) + residual -- fusing the ResBlock skip-add
// into the accumulate pass so the modiff conv doesn't pay a trailing aten::add.
// The o_hat_cache write is byte-identical to the plain accumulate kernel (the
// temporal cache must NOT include the residual, or the next step's accumulate
// corrupts), so the cache evolution is unchanged; only the returned `output`
// carries the skip. Following scale_bias_residual_store_half_kernel's precedent,
// the residual add is done in fp32 and rounded once (slightly more accurate than
// the fp16-accumulated aten::add it replaces). residual/output are fp16
// channels_last, same layout/shape as o_hat_cache.
__global__ void scale_accumulate_residual_half_cache_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ o_hat_cache,      // in-place accumulate (NO residual)
    const __half* __restrict__ residual,   // per-element skip
    __half* __restrict__ output,           // = o_hat_new + residual (separate buffer)
    int num_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        float new_val = __half2float(o_hat_cache[i]) + conv_output[i] * weight_scale[ch];
        o_hat_cache[i] = __float2half_rn(new_val);
        output[i] = __float2half_rn(new_val + __half2float(residual[i]));
    }
}

// Vectorized (float2/half2) counterpart of scale_accumulate_residual_half_cache_kernel.
// Same num_channels % 2 == 0 gate rationale as scale_accumulate_half_cache_vec2_kernel above.
__global__ void scale_accumulate_residual_half_cache_vec2_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ o_hat_cache,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
) {
    const int stride = blockDim.x * gridDim.x;
    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            int ch = base % num_channels;
            float2 conv_v = *reinterpret_cast<const float2*>(&conv_output[base]);
            float2 scale_v = *reinterpret_cast<const float2*>(&weight_scale[ch]);
            float2 cache_v = __half22float2(*reinterpret_cast<const __half2*>(&o_hat_cache[base]));
            float2 res_v = __half22float2(*reinterpret_cast<const __half2*>(&residual[base]));
            float nv0 = cache_v.x + conv_v.x * scale_v.x;
            float nv1 = cache_v.y + conv_v.y * scale_v.y;
            *reinterpret_cast<__half2*>(&o_hat_cache[base]) = __float22half2_rn(make_float2(nv0, nv1));
            *reinterpret_cast<__half2*>(&output[base]) =
                __float22half2_rn(make_float2(nv0 + res_v.x, nv1 + res_v.y));
        } else {
            int ch = base % num_channels;
            float new_val = __half2float(o_hat_cache[base]) + conv_output[base] * weight_scale[ch];
            o_hat_cache[base] = __float2half_rn(new_val);
            output[base] = __float2half_rn(new_val + __half2float(residual[base]));
        }
    }
}

__global__ void scale_store_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    float* __restrict__ output,
    int num_elements,
    int num_channels
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements && (base % num_channels) <= num_channels - 4) {
        float4 conv_v = reinterpret_cast<const float4*>(conv_output)[idx4];
        float4 out_v;
        int ch_base = base % num_channels;
        float s0 = weight_scale[ch_base];
        float s1 = weight_scale[ch_base + 1];
        float s2 = weight_scale[ch_base + 2];
        float s3 = weight_scale[ch_base + 3];
        out_v.x = conv_v.x * s0;
        out_v.y = conv_v.y * s1;
        out_v.z = conv_v.z * s2;
        out_v.w = conv_v.w * s3;
        reinterpret_cast<float4*>(output)[idx4] = out_v;
    } else {
        int end = min(base + 4, num_elements);
        for (int i = base; i < end; i++) {
            int ch = i % num_channels;
            output[i] = conv_output[i] * weight_scale[ch];
        }
    }
}

// Per-element (no vectorization needed), so there is no channel-boundary
// hazard here regardless of num_channels.
__global__ void scale_store_half_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        int ch = i % num_channels;
        output[i] = __float2half_rn(conv_output[i] * weight_scale[ch]);
    }
}

// Vectorized (float2/half2) counterpart of scale_store_half_kernel. Same num_channels % 2
// == 0 gate rationale as scale_accumulate_half_cache_vec2_kernel above.
__global__ void scale_store_half_vec2_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ weight_scale,
    __half* __restrict__ output,
    int num_elements,
    int num_channels
) {
    const int stride = blockDim.x * gridDim.x;
    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            int ch = base % num_channels;
            float2 conv_v = *reinterpret_cast<const float2*>(&conv_output[base]);
            float2 scale_v = *reinterpret_cast<const float2*>(&weight_scale[ch]);
            float2 out_v = make_float2(conv_v.x * scale_v.x, conv_v.y * scale_v.y);
            *reinterpret_cast<__half2*>(&output[base]) = __float22half2_rn(out_v);
        } else {
            int ch = base % num_channels;
            output[base] = __float2half_rn(conv_output[base] * weight_scale[ch]);
        }
    }
}

// Note: there used to be a standalone `scale_accumulate` host wrapper here
// (FP32-only) exposed to Python, but it had zero callers -- every real
// accumulate path goes through conv2d_int8_fprop_o_hat/conv2d_int4_fprop_o_hat,
// which dispatch to scale_accumulate_kernel/scale_accumulate_half_cache_kernel
// directly. Removed; the two __global__ kernels above remain in use.
