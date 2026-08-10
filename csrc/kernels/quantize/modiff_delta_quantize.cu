// =========================================================================
// MoDiff temporal-delta cache-update kernels.
//
// This file implements the core "modulated quantization" step from the
// MoDiff paper: instead of quantizing an activation x directly, we quantize
// the residual (x - a_hat_cache) against a running FP32/FP16 approximation
// a_hat_cache of x, then fold the quantized residual back into a_hat_cache
// so the next call sees an even better approximation. Because consecutive
// diffusion timesteps are highly correlated, the residual has a much smaller
// dynamic range than x itself, so quantizing it (down to 3-4 bits) loses much
// less precision than quantizing x directly.
//
// Two scale modes:
//   - "dynamic" (sub_absmax_scale + quantize_and_update_ahat_kernel family):
//     the per-call scale is derived from this call's residual absmax.
//   - "static" (static_quantize_and_update_ahat_kernel family): the scale is
//     a precomputed calibration constant passed in by the caller; the kernel
//     only computes the residual, quantizes, and updates the cache.
// An optional per-channel `smooth_inv` (SmoothQuant) factor can be folded in:
// when present, x is multiplied by smooth_inv[channel] before the residual is
// formed, so a_hat_cache always tracks the *smoothed* activation, consistently
// across both the dynamic and static paths.
//
// All of the vectorized (float4) paths below assume the "quad" of 4 elements
// being processed by one thread never straddles a channel boundary when
// smooth_inv is used, i.e. num_channels % 4 == 0 (true for every conv/linear
// layer in this project's models). The scalar tail loops have no such
// requirement.
// =========================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// For scale_quantize_int8 / scale_quantize_and_pack (quantize.cu), used by
// dynamic_quantize_int8_fprop / dynamic_quantize_pack_int4_fprop below.
#include "modiff_kernels_api.h"

#include "common.cuh"

// Scalar-load helper so the static-scale half-cache kernels below can be
// templated on the input activation's dtype (fp32 or fp16) instead of always
// requiring a pre-cast to fp32. The MoDiff modulated hot path calls these
// once per quantized conv layer per sampling step; forcing every caller to
// materialize a full fp32 copy of x first (a separate elementwise kernel
// launch touching the whole activation tensor) was costing more GPU time
// than the quantized conv itself. Reading fp16 directly and upconverting
// per-element in registers removes that extra full-tensor pass.
__device__ __forceinline__ float load_as_float(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float load_as_float(const __half* p, int i) { return __half2float(p[i]); }

// 2-wide (vectorized) counterparts, for the half-cache kernels' vec2 variants below.
// Caller guarantees `i` is even relative to `p`'s base pointer (see the fast-path gate
// on num_channels % 2 == 0 at each vec2 kernel's call site).
__device__ __forceinline__ float2 load_as_float2(const float* p, int i) {
    return reinterpret_cast<const float2*>(p)[i >> 1];
}
__device__ __forceinline__ float2 load_as_float2(const __half* p, int i) {
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}
__device__ __forceinline__ float2 half_cache_load2(const __half* p, int i) {
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}
__device__ __forceinline__ void half_cache_store2(__half* p, int i, float2 v) {
    reinterpret_cast<__half2*>(p)[i >> 1] = __float22half2_rn(v);
}

// SiLU(v) = v * sigmoid(v), used by the "_silu" kernel variants below to fuse
// a ResBlock's activation function directly into the quantize step instead of
// materializing a separate F.silu(x) pass over the same full-size tensor
// beforehand (see step1_static_quantize_fprop_silu / _pack_int4_fprop_silu).
__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }

// ---- Dynamic path: quantize + update a_hat, scale derived elsewhere ----

// Fused step 1: quantize residual and accumulate into a_hat_cache simultaneously.
// residual is expected to already be (x - a_hat_cache), e.g. from sub_absmax_scale.
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

// Same quantization as above, but does NOT touch a_hat_cache. Used by the
// "_no_ahat" step1 wrappers where the caller manages the cache update itself
// (or intentionally skips it, e.g. for a one-off/no-caching baseline run).
__global__ void quantize_only_int8_kernel(
    const float* __restrict__ residual,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];

        float q0 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.x * scale)));
        float q1 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.y * scale)));
        float q2 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.z * scale)));
        float q3 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.w * scale)));

        reinterpret_cast<int32_t*>(output_int8)[idx4] =
            ((unsigned char)(int8_t)q0) | ((unsigned char)(int8_t)q1 << 8) |
            ((unsigned char)(int8_t)q2 << 16) | ((unsigned char)(int8_t)q3 << 24);
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            float q = fmaxf(-127.0f, fminf(127.0f, roundf(r * scale)));
            output_int8[i] = (int8_t)q;
        }
    }
}

// INT4 variant of quantize_and_update_ahat_kernel: quantizes to [-7,7], updates
// a_hat_cache, and packs 2 elements per output byte (low nibble = element 2i,
// high nibble = element 2i+1).
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
            // Only the even element of each pair emits a packed byte; it also
            // computes (without storing) its odd neighbor's quantized value so
            // the byte can be written in one shot. The odd iteration then just
            // updates a_hat_cache[i] for that neighbor (recomputing the same q).
            if (i % 2 == 0) {
                float r_next = (i + 1 < num_elements) ? residual[i + 1] : 0.0f;
                float q_next = fmaxf(-7.0f, fminf(7.0f, roundf(r_next * scale)));

                int8_t b_curr = ((int8_t)q & 0x0F) | (((int8_t)q_next & 0x0F) << 4);
                output_packed[i / 2] = b_curr;
            }
        }
    }
}

// INT4 variant of quantize_only_int8_kernel: quantize + pack, no cache update.
__global__ void quantize_pack_only_kernel_int4(
    const float* __restrict__ residual,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];

        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.x * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.y * scale)));
        float q2 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.z * scale)));
        float q3 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.w * scale)));

        int8_t b0 = ((int8_t)q0 & 0x0F) | (((int8_t)q1 & 0x0F) << 4);
        int8_t b1 = ((int8_t)q2 & 0x0F) | (((int8_t)q3 & 0x0F) << 4);
        reinterpret_cast<int16_t*>(output_packed)[idx4] = (uint16_t)(uint8_t)b0 | ((uint16_t)(uint8_t)b1 << 8);
    } else {
        for (int i = base; i < num_elements; i += 2) {
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(residual[i] * scale)));
            float q1 = 0.0f;
            if (i + 1 < num_elements) {
                q1 = fmaxf(-7.0f, fminf(7.0f, roundf(residual[i + 1] * scale)));
            }
            output_packed[i / 2] = (((int8_t)q0 & 0x0F) | ((((int8_t)q1 & 0x0F) << 4)));
        }
    }
}

// ---- Static path: scale is a precomputed calibration constant ----
// These kernels compute the residual (with optional SmoothQuant) themselves
// instead of taking a precomputed residual buffer, since there is no absmax
// reduction step to share the residual with.

__global__ void static_quantize_and_update_ahat_kernel_int8(
    const float* __restrict__ x,
    float* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. A code above it must SATURATE -- which is
    // what a delta that outgrew a reused scale (MODIFF_DELTA_REFRESH > 1) depends on. The clip
    // ratio that used to be the other reason is retired -- see OptimizedInt8Conv2d.
    const float lim = a4 ? 7.0f : 127.0f;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements) {
        float4 x_v = reinterpret_cast<const float4*>(x)[idx4];
        float4 c_v = reinterpret_cast<float4*>(a_hat_cache)[idx4];

        if (smooth_inv != nullptr) {
            int ch = base % num_channels;
            x_v.x *= smooth_inv[ch];
            x_v.y *= smooth_inv[ch + 1];
            x_v.z *= smooth_inv[ch + 2];
            x_v.w *= smooth_inv[ch + 3];
        }

        float r0 = x_v.x - c_v.x;
        float r1 = x_v.y - c_v.y;
        float r2 = x_v.z - c_v.z;
        float r3 = x_v.w - c_v.w;

        float q0 = fmaxf(-lim, fminf(lim, roundf(r0 * scale)));
        float q1 = fmaxf(-lim, fminf(lim, roundf(r1 * scale)));
        float q2 = fmaxf(-lim, fminf(lim, roundf(r2 * scale)));
        float q3 = fmaxf(-lim, fminf(lim, roundf(r3 * scale)));

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
            float xval = x[i];
            if (smooth_inv != nullptr) {
                xval *= smooth_inv[i % num_channels];
            }
            float r = xval - a_hat_cache[i];
            float q = fmaxf(-lim, fminf(lim, roundf(r * scale)));
            a_hat_cache[i] += q * inv_scale;
            output_int8[i] = (int8_t)q;
        }
    }
}

// Same as above but a_hat_cache is stored as FP16 (halves resident cache memory
// for calibrated/static-scale layers). Uses a simple grid-stride scalar loop
// since __half doesn't have a convenient 4-wide vector load like float4.
template <typename T_IN>
__global__ void static_quantize_and_update_ahat_kernel_int8_half_cache(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. A code above it must SATURATE -- which is
    // what a delta that outgrew a reused scale (MODIFF_DELTA_REFRESH > 1) depends on. The clip
    // ratio that used to be the other reason is retired -- see OptimizedInt8Conv2d.
    const float lim = a4 ? 7.0f : 127.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float xval = load_as_float(x, i);
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        float cache = __half2float(a_hat_cache[i]);
        float q = fmaxf(-lim, fminf(lim, roundf((xval - cache) * scale)));
        a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        output_int8[i] = static_cast<int8_t>(q);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_and_update_ahat_kernel_int8_half_cache.
// Pair-major grid-stride loop with an inline single-element epilogue for the last leftover
// element when num_elements is odd (mirrors this file's grid-stride-loop style rather than
// switching to the separate idx4-no-loop idiom the fp32-cache sibling above uses). Requires
// num_channels % 2 == 0 for the smooth_inv fast path (same pre-existing, unenforced
// channels-last-contiguity assumption this file's float4 kernels already rely on for %4==0 --
// not a new risk); the caller only dispatches here when that holds.
template <typename T_IN>
__global__ void static_quantize_and_update_ahat_kernel_int8_half_cache_vec2(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. A code above it must SATURATE -- which is
    // what a delta that outgrew a reused scale (MODIFF_DELTA_REFRESH > 1) depends on. The clip
    // ratio that used to be the other reason is retired -- see OptimizedInt8Conv2d.
    const float lim = a4 ? 7.0f : 127.0f;
    const int stride = blockDim.x * gridDim.x;

    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            float2 cache = half_cache_load2(a_hat_cache, base);
            float q0 = fmaxf(-lim, fminf(lim, roundf((xv.x - cache.x) * scale)));
            float q1 = fmaxf(-lim, fminf(lim, roundf((xv.y - cache.y) * scale)));
            half_cache_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
            output_int8[base] = static_cast<int8_t>(q0);
            output_int8[base + 1] = static_cast<int8_t>(q1);
        } else {
            // odd num_elements: single leftover element, same math as the scalar kernel.
            float xval = load_as_float(x, base);
            if (smooth_inv != nullptr) xval *= smooth_inv[base % num_channels];
            float cache = __half2float(a_hat_cache[base]);
            float q = fmaxf(-lim, fminf(lim, roundf((xval - cache) * scale)));
            a_hat_cache[base] = __float2half_rn(cache + q * inv_scale);
            output_int8[base] = static_cast<int8_t>(q);
        }
    }
}

// Same as static_quantize_and_update_ahat_kernel_int8_half_cache above, but
// `x` is the ResBlock's GroupNorm output *before* SiLU: applies SiLU inline
// to each element before forming the residual, instead of requiring a
// separate F.silu(x) kernel pass over the same full-size activation first
// (see step1_static_quantize_fprop_silu, ResBlock's fused_in_norm_silu /
// fused_out_norm_silu -> in_conv / out_conv hot path in int8_optimized.py).
template <typename T_IN>
__global__ void static_quantize_and_update_ahat_kernel_int8_half_cache_silu(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. A code above it must SATURATE -- which is
    // what a delta that outgrew a reused scale (MODIFF_DELTA_REFRESH > 1) depends on. The clip
    // ratio that used to be the other reason is retired -- see OptimizedInt8Conv2d.
    const float lim = a4 ? 7.0f : 127.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float xval = silu_f(load_as_float(x, i));
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        float cache = __half2float(a_hat_cache[i]);
        float q = fmaxf(-lim, fminf(lim, roundf((xval - cache) * scale)));
        a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        output_int8[i] = static_cast<int8_t>(q);
    }
}

// Vectorized (half2/float2) counterpart of
// static_quantize_and_update_ahat_kernel_int8_half_cache_silu. Same pair-major
// grid-stride loop + odd-leftover epilogue as the non-silu vec2 kernel above.
template <typename T_IN>
__global__ void static_quantize_and_update_ahat_kernel_int8_half_cache_silu_vec2(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. A code above it must SATURATE -- which is
    // what a delta that outgrew a reused scale (MODIFF_DELTA_REFRESH > 1) depends on. The clip
    // ratio that used to be the other reason is retired -- see OptimizedInt8Conv2d.
    const float lim = a4 ? 7.0f : 127.0f;
    const int stride = blockDim.x * gridDim.x;

    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            xv.x = silu_f(xv.x); xv.y = silu_f(xv.y);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            float2 cache = half_cache_load2(a_hat_cache, base);
            float q0 = fmaxf(-lim, fminf(lim, roundf((xv.x - cache.x) * scale)));
            float q1 = fmaxf(-lim, fminf(lim, roundf((xv.y - cache.y) * scale)));
            half_cache_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
            output_int8[base] = static_cast<int8_t>(q0);
            output_int8[base + 1] = static_cast<int8_t>(q1);
        } else {
            float xval = silu_f(load_as_float(x, base));
            if (smooth_inv != nullptr) xval *= smooth_inv[base % num_channels];
            float cache = __half2float(a_hat_cache[base]);
            float q = fmaxf(-lim, fminf(lim, roundf((xval - cache) * scale)));
            a_hat_cache[base] = __float2half_rn(cache + q * inv_scale);
            output_int8[base] = static_cast<int8_t>(q);
        }
    }
}

// INT4 static-scale variant of static_quantize_and_update_ahat_kernel_int8,
// packing 2 elements per output byte.
__global__ void static_quantize_pack_and_update_ahat_kernel_int4(
    const float* __restrict__ x,
    float* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements) {
        float4 x_v = reinterpret_cast<const float4*>(x)[idx4];
        float4 c_v = reinterpret_cast<float4*>(a_hat_cache)[idx4];

        if (smooth_inv != nullptr) {
            int ch = base % num_channels;
            x_v.x *= smooth_inv[ch];
            x_v.y *= smooth_inv[ch + 1];
            x_v.z *= smooth_inv[ch + 2];
            x_v.w *= smooth_inv[ch + 3];
        }

        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x_v.x - c_v.x) * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf((x_v.y - c_v.y) * scale)));
        float q2 = fmaxf(-7.0f, fminf(7.0f, roundf((x_v.z - c_v.z) * scale)));
        float q3 = fmaxf(-7.0f, fminf(7.0f, roundf((x_v.w - c_v.w) * scale)));

        c_v.x += q0 * inv_scale;
        c_v.y += q1 * inv_scale;
        c_v.z += q2 * inv_scale;
        c_v.w += q3 * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;

        int8_t b0 = ((int8_t)q0 & 0x0F) | (((int8_t)q1 & 0x0F) << 4);
        int8_t b1 = ((int8_t)q2 & 0x0F) | (((int8_t)q3 & 0x0F) << 4);
        reinterpret_cast<int16_t*>(output_packed)[idx4] = (uint16_t)(uint8_t)b0 | ((uint16_t)(uint8_t)b1 << 8);
    } else {
        for (int i = base; i < num_elements; i += 2) {
            float x0 = x[i];
            if (smooth_inv != nullptr) {
                x0 *= smooth_inv[i % num_channels];
            }
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - a_hat_cache[i]) * scale)));
            a_hat_cache[i] += q0 * inv_scale;

            float q1 = 0.0f;
            if (i + 1 < num_elements) {
                float x1 = x[i + 1];
                if (smooth_inv != nullptr) {
                    x1 *= smooth_inv[(i + 1) % num_channels];
                }
                q1 = fmaxf(-7.0f, fminf(7.0f, roundf((x1 - a_hat_cache[i + 1]) * scale)));
                a_hat_cache[i + 1] += q1 * inv_scale;
            }
            output_packed[i / 2] = (((int8_t)q0 & 0x0F) | ((((int8_t)q1 & 0x0F) << 4)));
        }
    }
}

// FP16-cache version of the INT4 static packer.
//
// Grid-stride loop (2 elements/iteration): the host wrapper sizes its launch
// grid for the *fp32*-cache sibling kernel below, which processes 4
// elements/thread via float4 vectorization. Without a loop here, a thread
// that only advances 2 elements once would leave roughly the back half of
// a_hat_cache/output_packed (in flat index order) untouched -- this was a
// real bug (fixed): stale/uninitialized cache values and garbage packed
// output for that tail half of every tensor on the static INT4 + FP16-cache
// path (int4_optimized.py's calibrated OptimizedInt4Conv2d._forward_modulated).
// Matches the loop structure already used by the INT8 sibling,
// static_quantize_and_update_ahat_kernel_int8_half_cache, above.
template <typename T_IN>
__global__ void static_quantize_pack_and_update_ahat_kernel_int4_half_cache(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        float x0 = load_as_float(x, base);
        if (smooth_inv != nullptr) {
            x0 *= smooth_inv[base % num_channels];
        }
        float c0 = __half2float(a_hat_cache[base]);
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - c0) * scale)));
        a_hat_cache[base] = __float2half_rn(c0 + q0 * inv_scale);

        float q1 = 0.0f;
        if (base + 1 < num_elements) {
            float x1 = load_as_float(x, base + 1);
            if (smooth_inv != nullptr) {
                x1 *= smooth_inv[(base + 1) % num_channels];
            }
            float c1 = __half2float(a_hat_cache[base + 1]);
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf((x1 - c1) * scale)));
            a_hat_cache[base + 1] = __float2half_rn(c1 + q1 * inv_scale);
        }

        output_packed[base / 2] =
            (static_cast<int8_t>(q0) & 0x0F) |
            ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_pack_and_update_ahat_kernel_int4_half_cache.
// The scalar kernel above is already pair-major (base = idx*2); this only widens the per-pair
// load/store from two independent scalar accesses to one vectorized half2/float2 access -- same
// technique, same safety argument, as the INT8 sibling's _vec2 kernel above. Requires
// num_channels % 2 == 0 for the smooth_inv fast path (not a new risk: base is always even here,
// so base % num_channels is even too when num_channels is even, so the pair never straddles a
// channel boundary); the caller only dispatches here when that holds.
template <typename T_IN>
__global__ void static_quantize_pack_and_update_ahat_kernel_int4_half_cache_vec2(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            float2 cache = half_cache_load2(a_hat_cache, base);
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((xv.x - cache.x) * scale)));
            float q1 = fmaxf(-7.0f, fminf(7.0f, roundf((xv.y - cache.y) * scale)));
            half_cache_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
            output_packed[base / 2] =
                (static_cast<int8_t>(q0) & 0x0F) |
                ((static_cast<int8_t>(q1) & 0x0F) << 4);
        } else {
            // odd num_elements: single leftover element, same math as the scalar kernel.
            float x0 = load_as_float(x, base);
            if (smooth_inv != nullptr) x0 *= smooth_inv[base % num_channels];
            float c0 = __half2float(a_hat_cache[base]);
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - c0) * scale)));
            a_hat_cache[base] = __float2half_rn(c0 + q0 * inv_scale);
            output_packed[base / 2] = static_cast<int8_t>(q0) & 0x0F;
        }
    }
}

// ---- TRUE cache-free static quantize (baseline, NO a_hat): same as the *_update_ahat kernels
// above but without the a_hat read/subtract/write. Baseline conv doesn't do temporal caching, so
// residual = x - a_hat collapses to x; dropping a_hat removes the a_hat read+write + the zero-fill.
template <typename T_IN>
__global__ void static_quantize_int8_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float xval = load_as_float(x, i);
        if (smooth_inv != nullptr) xval *= smooth_inv[i % num_channels];
        float q = fmaxf(-127.0f, fminf(127.0f, roundf(xval * scale)));
        output_int8[i] = static_cast<int8_t>(q);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_int8_noahat_kernel. Pair-major
// grid-stride loop with an inline single-element epilogue for odd num_elements, same technique
// as static_quantize_and_update_ahat_kernel_int8_half_cache_vec2 above. Requires
// num_channels % 2 == 0 for the smooth_inv fast path (not a new risk -- same argument as that
// kernel: base is always even here, so base % num_channels is even too when num_channels is
// even, so a pair never straddles a channel boundary); the caller only dispatches here when
// that holds. Packs the 2 output int8 codes into one int16_t store.
template <typename T_IN>
__global__ void static_quantize_int8_noahat_vec2_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    const int stride = blockDim.x * gridDim.x;

    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            int8_t q0 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(xv.x * scale)));
            int8_t q1 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(xv.y * scale)));
            reinterpret_cast<int16_t*>(output_int8)[base >> 1] =
                (int16_t)(((unsigned char)q0) | (((unsigned char)q1) << 8));
        } else {
            float xval = load_as_float(x, base);
            if (smooth_inv != nullptr) xval *= smooth_inv[base % num_channels];
            float q = fmaxf(-127.0f, fminf(127.0f, roundf(xval * scale)));
            output_int8[base] = static_cast<int8_t>(q);
        }
    }
}

template <typename T_IN>
__global__ void static_quantize_pack_int4_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        float x0 = load_as_float(x, base);
        if (smooth_inv != nullptr) x0 *= smooth_inv[base % num_channels];
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(x0 * scale)));
        float q1 = 0.0f;
        if (base + 1 < num_elements) {
            float x1 = load_as_float(x, base + 1);
            if (smooth_inv != nullptr) x1 *= smooth_inv[(base + 1) % num_channels];
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf(x1 * scale)));
        }
        output_packed[base / 2] =
            (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_pack_int4_noahat_kernel. The scalar
// kernel above is already pair-major (base = idx*2); this only widens the per-pair load from
// two independent scalar accesses to one vectorized half2/float2 access -- same technique as
// static_quantize_pack_and_update_ahat_kernel_int4_half_cache_vec2 above. Same num_channels % 2
// == 0 gate rationale.
template <typename T_IN>
__global__ void static_quantize_pack_int4_noahat_vec2_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(xv.x * scale)));
            float q1 = fmaxf(-7.0f, fminf(7.0f, roundf(xv.y * scale)));
            output_packed[base / 2] =
                (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
        } else {
            float x0 = load_as_float(x, base);
            if (smooth_inv != nullptr) x0 *= smooth_inv[base % num_channels];
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(x0 * scale)));
            output_packed[base / 2] = static_cast<int8_t>(q0) & 0x0F;
        }
    }
}

// Same as static_quantize_pack_and_update_ahat_kernel_int4_half_cache above,
// but applies SiLU inline to each element of `x` before forming the residual
// -- see static_quantize_and_update_ahat_kernel_int8_half_cache_silu's
// comment for the rationale (fuses ResBlock's activation into the quantize
// step instead of a separate full-tensor F.silu(x) pass).
template <typename T_IN>
__global__ void static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        float x0 = silu_f(load_as_float(x, base));
        if (smooth_inv != nullptr) {
            x0 *= smooth_inv[base % num_channels];
        }
        float c0 = __half2float(a_hat_cache[base]);
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - c0) * scale)));
        a_hat_cache[base] = __float2half_rn(c0 + q0 * inv_scale);

        float q1 = 0.0f;
        if (base + 1 < num_elements) {
            float x1 = silu_f(load_as_float(x, base + 1));
            if (smooth_inv != nullptr) {
                x1 *= smooth_inv[(base + 1) % num_channels];
            }
            float c1 = __half2float(a_hat_cache[base + 1]);
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf((x1 - c1) * scale)));
            a_hat_cache[base + 1] = __float2half_rn(c1 + q1 * inv_scale);
        }

        output_packed[base / 2] =
            (static_cast<int8_t>(q0) & 0x0F) |
            ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu.
// Same pair-major vec2 treatment as static_quantize_pack_and_update_ahat_kernel_int4_half_cache_vec2
// above, with SiLU applied per-lane before the residual/quantize math.
template <typename T_IN>
__global__ void static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu_vec2(
    const T_IN* __restrict__ x,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int num_channels,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            xv.x = silu_f(xv.x); xv.y = silu_f(xv.y);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            float2 cache = half_cache_load2(a_hat_cache, base);
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((xv.x - cache.x) * scale)));
            float q1 = fmaxf(-7.0f, fminf(7.0f, roundf((xv.y - cache.y) * scale)));
            half_cache_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
            output_packed[base / 2] =
                (static_cast<int8_t>(q0) & 0x0F) |
                ((static_cast<int8_t>(q1) & 0x0F) << 4);
        } else {
            float x0 = silu_f(load_as_float(x, base));
            if (smooth_inv != nullptr) x0 *= smooth_inv[base % num_channels];
            float c0 = __half2float(a_hat_cache[base]);
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - c0) * scale)));
            a_hat_cache[base] = __float2half_rn(c0 + q0 * inv_scale);
            output_packed[base / 2] = static_cast<int8_t>(q0) & 0x0F;
        }
    }
}

// ---- sub_absmax_scale: shared "dynamic scale" reduction used by the dynamic step1_* wrappers ----

// Fused: residual = x - a_hat_cache (with optional SmoothQuant), absmax reduction
// over the whole tensor, then scale = Q_level/max(absmax,eps) and its inverse.
// Replaces 6 separate kernel launches (sub, abs, amax, clamp, div, div) with one
// cooperative kernel using atomic float-max + a block-retirement counter to elect
// a single "last" block to finalize the scale.
//
// a_hat_cache and residual are independently nullable, subsuming what used to be
// a separate near-duplicate kernel (quantize.cu's absmax_scale_kernel) for the
// cache-free baseline case:
//   a_hat_cache == nullptr -> no cache to subtract, "residual" is just x itself
//                             (still smoothed if smooth_inv is set)
//   residual == nullptr    -> pure reduction, nothing to materialize (this is
//                             exactly what the baseline path needs: it only wants
//                             scale/inv_scale, and quantizing straight from x
//                             costs nothing extra since x is already resident)
// Both null is the baseline case (compute_dynamic_scale); both non-null is the
// original MoDiff dynamic case (sub_absmax_scale).
__global__ void sub_absmax_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ a_hat_cache,   // nullptr => baseline, no cache to subtract
    float* __restrict__ residual,            // nullptr => don't materialize, reduction only
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
        // Fused SmoothQuant: x_smooth = x * smooth_inv[channel]
        if (smooth_inv != nullptr) {
            int ch = (i4 * 4) % num_channels;
            x_v.x *= smooth_inv[ch];
            x_v.y *= smooth_inv[ch + 1];
            x_v.z *= smooth_inv[ch + 2];
            x_v.w *= smooth_inv[ch + 3];
        }
        float4 r_v;
        if (a_hat_cache != nullptr) {
            float4 c_v = reinterpret_cast<const float4*>(a_hat_cache)[i4];
            r_v.x = x_v.x - c_v.x;
            r_v.y = x_v.y - c_v.y;
            r_v.z = x_v.z - c_v.z;
            r_v.w = x_v.w - c_v.w;
        } else {
            r_v = x_v;
        }
        if (residual != nullptr) {
            reinterpret_cast<float4*>(residual)[i4] = r_v;
        }
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
        float r = (a_hat_cache != nullptr) ? (xval - a_hat_cache[i]) : xval;
        if (residual != nullptr) {
            residual[i] = r;
        }
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
//   Op:       MoDiff temporal-delta residual + absmax + dynamic-scale (shared reduction)
//   Inputs:   x FP32 [N,C,H,W]; a_hat_cache FP32 [same] MoDiff cache; residual FP32 [same]
//             out buffer; absmax_buf FP32 [1] (init 0, self-reset); scale_out FP32 [1] out;
//             inv_scale_out FP32 [1] out; retire_count int32 [1] (init 0, self-reset);
//             Q_level float (7.0 int4 / 127.0 int8); smooth_inv FP32 [C] per-channel
//             SmoothQuant inverse (empty = skip)
//   Outputs:  residual = (x*smooth_inv) - a_hat_cache; scale_out = Q_level/max(absmax,1e-6);
//             inv_scale_out = max(absmax,1e-6)/Q_level (CUTLASS alpha)
//   Computes: temporal delta of x against the cache, then the per-tensor dynamic scale
//             from the delta's absmax
//   Fuses:    6 elementwise launches (sub, abs, amax, clamp, div, div) into one cooperative
//             kernel (atomic float-max + block-retirement counter elects the last block to
//             finalize the scale); no host sync
//   Constraints: absmax_buf & retire_count must be 0 on entry (self-resetting); with
//                smooth_inv the float4 quads assume num_channels % 4 == 0
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
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

    sub_absmax_scale_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
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
// FP16 twin of sub_absmax_scale: reduction ONLY, no residual materialization.
//
// Why this exists as a separate kernel rather than a template on the one above.
// sub_absmax_scale is fp32-in/fp32-cache and writes a full fp32 residual tensor,
// because it serves the *uncalibrated* path where a_hat/o_hat are fp32. Once a
// layer is calibrated the caches are fp16 (see int8_optimized.py
// _ensure_state_buffers) and the conv is the CUTLASS-EVT in-place o_hat RMW,
// which requires an fp16 o_hat. So the fp32 kernel cannot be reused there
// without demoting the whole layer off the EVT path -- the opposite of what we
// want.
//
// What it buys: the *dynamic* MoDiff delta scale. Measured 2026-08-04 on the
// real LSUN-churches checkpoint, the production static setting (delta quantized
// on the activation's grid, scale = static_input_scale) CLIPS on 49 of 70
// observed conv layers -- median max|q| lands exactly on the 127 ceiling. A
// per-call scale of Q/max|delta| cannot clip by construction, which is also the
// regime the paper's Theorem 4.3 assumes ("dynamic quantizers ... to avoid
// clipping error").
//
// It writes into the same scale_out/inv_scale_out 1-element buffers that
// step1_static_quantize_fprop and conv2d_int8_evt_o_hat already read through
// device pointers, so the entire existing fused chain downstream is untouched:
//   delta_absmax_fp16(...)  ->  step1_static_quantize_fprop(x, a_hat, scale_out, ...)
//                           ->  conv2d_int8_evt_o_hat(..., inv_scale_out, ...)
// No residual is stored because step1_static_quantize_fprop recomputes the
// delta itself from x and a_hat; materializing it here would add a full-tensor
// fp32 write + read for nothing. The cost is therefore one extra *read* pass
// over x and a_hat -- the price of a non-clipping scale.
//
// Q_level parameterizes int8 (127) vs int4 (7), so this one kernel serves both
// W8A8 and W4A4.
//
// TIn is templated because the calibrated MoDiff path does NOT guarantee an fp16
// *input* -- only an fp16 *cache*. step1_static_quantize_fprop reads x through
// load_as_float and is instantiated for both float and __half, and the ResBlock
// feeds fp32 activations into some of those layers. The "fp16" in the name refers
// to the cache dtype, which is what distinguishes this from sub_absmax_scale.
template <typename TIn>
__global__ void delta_absmax_fp16_kernel(
    const TIn* __restrict__ x,
    const __half* __restrict__ a_hat_cache,   // nullptr => absmax(x) with no cache to subtract
    float* __restrict__ absmax_buf,      // Must be 0 on entry (self-resetting)
    float* __restrict__ scale_out,       // Q_level / max(absmax, eps)
    float* __restrict__ inv_scale_out,   // max(absmax, eps) / Q_level  (CUTLASS alpha)
    unsigned int* __restrict__ retire_count, // Must be 0 on entry (self-resetting)
    const float* __restrict__ smooth_inv,  // Per-channel SmoothQuant inverse (NULL=skip)
    int num_channels,                    // C for NHWC channel indexing
    float Q_level,                       // 7.0 for INT4, 127.0 for INT8
    bool fused_silu,                     // apply SiLU to x first (see below)
    int num_elements
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    // Pairwise loop. For fp16 input a __half2 load keeps the access fully coalesced
    // across a warp (128 B/warp) while avoiding the num_channels % 8 constraint an
    // 8-wide float4 load would impose; every channel count in this UNet is even, and
    // the scalar tail covers the odd case. For fp32 input the pair is a float2.
    //
    // Operation order is silu -> smooth_inv -> subtract cache, matching
    // static_quantize_and_update_ahat_kernel_int8_half_cache_silu exactly. The
    // order is not interchangeable: SiLU is nonlinear, so silu(x*s) != silu(x)*s,
    // and reducing a different expression than the quantizer evaluates would
    // give a scale that clips.
    float local_max = 0.0f;
    int num_elements2 = num_elements / 2;
    for (int i2 = blockIdx.x * blockDim.x + tid; i2 < num_elements2; i2 += num_threads) {
        // load_as_float2 / half_cache_load2 take an ELEMENT index and pair internally,
        // and are overloaded for both input dtypes -- same helpers the vec2 quantize
        // kernels in this file use.
        float2 x_v = load_as_float2(x, i2 * 2);
        if (fused_silu) {
            x_v.x = silu_f(x_v.x);
            x_v.y = silu_f(x_v.y);
        }
        if (smooth_inv != nullptr) {
            int ch = (i2 * 2) % num_channels;
            x_v.x *= smooth_inv[ch];
            x_v.y *= smooth_inv[ch + 1];
        }
        if (a_hat_cache != nullptr) {
            float2 c_v = half_cache_load2(a_hat_cache, i2 * 2);
            x_v.x -= c_v.x;
            x_v.y -= c_v.y;
        }
        local_max = fmaxf(local_max, fmaxf(fabsf(x_v.x), fabsf(x_v.y)));
    }
    // Scalar tail (odd num_elements only).
    for (int i = num_elements2 * 2 + blockIdx.x * blockDim.x + tid;
         i < num_elements; i += num_threads) {
        float xval = load_as_float(x, i);
        if (fused_silu) {
            xval = silu_f(xval);
        }
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        if (a_hat_cache != nullptr) {
            xval -= __half2float(a_hat_cache[i]);
        }
        local_max = fmaxf(local_max, fabsf(xval));
    }

    // Block reduction, then the same atomic-max + last-block-retires election as
    // sub_absmax_scale_kernel above.
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        float val = sdata[0];
        unsigned int* addr = (unsigned int*)absmax_buf;
        unsigned int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
        } while (assumed != old);

        __threadfence();
        unsigned int ticket = atomicAdd(retire_count, 1u);
        if (ticket == gridDim.x - 1) {
            float am = fmaxf(*absmax_buf, 1e-6f);
            *scale_out = Q_level / am;
            *inv_scale_out = am / Q_level;
            *absmax_buf = 0.0f;
            *retire_count = 0;
        }
    }
}

// C++ wrapper for the FP16 dynamic delta scale. See the kernel comment for why
// this is not a template on sub_absmax_scale.
//   Op:       MoDiff temporal-delta absmax + dynamic scale (fp16 caches, reduction only)
//   Inputs:   x FP16 [N,C,H,W] (channels_last); a_hat_cache FP16 [same] MoDiff cache, or an
//             empty tensor to reduce absmax(x) with no subtraction; absmax_buf FP32 [1]
//             (init 0, self-reset); scale_out FP32 [1] out; inv_scale_out FP32 [1] out;
//             retire_count int32 [1] (init 0, self-reset); Q_level float (7.0 int4 /
//             127.0 int8); smooth_inv FP32 [C] per-channel SmoothQuant inverse (empty = skip);
//             fused_silu bool -- set it iff the paired quantize kernel is a _silu variant, so
//             both reduce and quantize the same expression
//   Outputs:  scale_out = Q_level/max(absmax,1e-6); inv_scale_out = its reciprocal
//             (CUTLASS alpha). No residual tensor is written.
//   Computes: max |silu?(x)*smooth_inv - a_hat| over the tensor, then the dynamic scale that
//             maps it to exactly +-Q_level -- so the subsequent delta quantize cannot clip.
//   Fuses:    sub + abs + amax + div into one cooperative kernel (atomic float-max +
//             block-retirement counter elects the last block); no host sync, so the result
//             stays a device pointer and the static-quantize + EVT-conv chain is unchanged.
//   Constraints: absmax_buf & retire_count must be 0 on entry (self-resetting); x and
//                a_hat_cache must both be FP16 and the same shape; with smooth_inv the
//                __half2 pairs assume num_channels % 2 == 0
//   vs fp16:  n/a (quantization support op — this is the overhead a dynamic quantizer
//             costs, traded against the clipping a static one suffers)
void delta_absmax_fp16(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor absmax_buf,
    torch::Tensor scale_out,
    torch::Tensor inv_scale_out,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv,
    bool fused_silu
) {
    TORCH_CHECK(x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kFloat32,
                "delta_absmax_fp16: x must be FP16 or FP32");
    const bool has_cache = a_hat_cache.numel() > 0;
    if (has_cache) {
        TORCH_CHECK(a_hat_cache.scalar_type() == torch::kHalf,
                    "delta_absmax_fp16: a_hat_cache must be FP16 (the calibrated MoDiff path); "
                    "use sub_absmax_scale for the FP32-cache uncalibrated path");
        TORCH_CHECK(a_hat_cache.numel() == x.numel(),
                    "delta_absmax_fp16: a_hat_cache must match x element count");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = x.numel();
    int block_size = 256;
    int grid_size = min((num_elements / 2 + block_size - 1) / block_size, 1024);
    grid_size = max(grid_size, 1);

    const float* smooth_ptr = nullptr;
    int num_channels = 0;
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }

    const __half* cache_ptr = has_cache
        ? reinterpret_cast<const __half*>(a_hat_cache.data_ptr<at::Half>()) : nullptr;
    if (x.scalar_type() == torch::kHalf) {
        delta_absmax_fp16_kernel<__half><<<grid_size, block_size, block_size * sizeof(float), stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
            absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
            inv_scale_out.data_ptr<float>(), (unsigned int*)retire_count.data_ptr<int>(),
            smooth_ptr, num_channels, Q_level, fused_silu, num_elements);
    } else {
        delta_absmax_fp16_kernel<float><<<grid_size, block_size, block_size * sizeof(float), stream>>>(
            x.data_ptr<float>(), cache_ptr,
            absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
            inv_scale_out.data_ptr<float>(), (unsigned int*)retire_count.data_ptr<int>(),
            smooth_ptr, num_channels, Q_level, fused_silu, num_elements);
    }
}

// =========================================================================
// Dynamic per-tensor scale discovery for the plain (non-MoDiff) baseline.
//
// Cache-free counterpart of sub_absmax_scale above: reduces absmax(x) over
// the whole tensor and writes scale = Q_level/max(absmax,eps) and
// inv_scale = 1/scale, entirely on GPU (no host sync), by calling
// sub_absmax_scale_kernel with a_hat_cache=nullptr and residual=nullptr
// (formerly a separate near-duplicate kernel, absmax_scale_kernel in
// quantize.cu; merged here since the only difference was the nullable
// subtract/materialize this kernel already supports).
//
// Before this existed, callers that wanted a fused dynamic-scale baseline had
// to fake it by passing a permanently-zero a_hat_cache into the MoDiff
// kernels (sub_absmax_scale / step1_quantize_no_ahat_fprop), which still paid
// for reading and subtracting that zero tensor on every call. This kernel
// does only the work a baseline actually needs.
// =========================================================================
void compute_dynamic_scale(
    torch::Tensor x,
    torch::Tensor absmax_buf,      // 1-element float32, init 0
    torch::Tensor scale_out,       // 1-element float32 output
    torch::Tensor inv_scale_out,   // 1-element float32 output
    torch::Tensor retire_count,    // 1-element int32, init 0
    float Q_level
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = x.numel();
    int block_size = 256;
    int grid_size = min((num_elements + block_size - 1) / block_size, 1024);

    sub_absmax_scale_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        nullptr,  // a_hat_cache: no cache in the baseline
        nullptr,  // residual: nothing to materialize, pure reduction
        absmax_buf.data_ptr<float>(),
        scale_out.data_ptr<float>(),
        inv_scale_out.data_ptr<float>(),
        (unsigned int*)retire_count.data_ptr<int>(),
        nullptr,  // smooth_inv: baseline callers apply smoothing in Python beforehand
        0,
        Q_level,
        num_elements
    );
}

// Fused compute_dynamic_scale + scale_quantize_int8: plain dynamic INT8
// quantization of x with no MoDiff cache involved anywhere.
//   Op:       Quantize (activation int8, dynamic scale) — cache-free baseline (no MoDiff)
//   Inputs:   x FP32 [any shape]; absmax_buf FP32 [1] (init 0); scale_buf FP32 [1] out;
//             inv_scale_buf FP32 [1] out; retire_count int32 [1] (init 0)
//   Outputs:  INT8 [same shape as x]
//   Computes: scale = 127/max(absmax(x),1e-6); out = clamp(round(x*scale), -127, 127)
//   Fuses:    compute_dynamic_scale (on-GPU absmax reduction, no host sync) + scale_quantize_int8
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset); NO a_hat cache (true
//                cache-free baseline, unlike step1_quantize_no_ahat_fprop which still reads it)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor dynamic_quantize_int8_fprop(
    torch::Tensor x,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count
) {
    compute_dynamic_scale(x, absmax_buf, scale_buf, inv_scale_buf, retire_count, 127.0f);
    return scale_quantize_int8(x, scale_buf);
}

// Fused compute_dynamic_scale + scale_quantize_and_pack: plain dynamic INT4
// quantization of x (packed 2-per-byte) with no MoDiff cache involved anywhere.
//   Op:       Quantize (activation int4, dynamic scale) + pack — cache-free baseline (no MoDiff)
//   Inputs:   x FP32 [N,C,H,W] channels_last-contiguous; absmax_buf FP32 [1] (init 0);
//             scale_buf FP32 [1] out; inv_scale_buf FP32 [1] out; retire_count int32 [1] (init 0)
//   Outputs:  INT8 [N,H,W,C/2] packed int4
//   Computes: scale = 7/max(absmax(x),1e-6); out = pack(clamp(round(x*scale), -7, 7))
//   Fuses:    compute_dynamic_scale (on-GPU absmax reduction) + scale_quantize_and_pack
//   Constraints: channels_last-contiguous (from scale_quantize_and_pack); absmax_buf &
//                retire_count 0 on entry; NO a_hat cache (true cache-free baseline)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor dynamic_quantize_pack_int4_fprop(
    torch::Tensor x,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count
) {
    compute_dynamic_scale(x, absmax_buf, scale_buf, inv_scale_buf, retire_count, 7.0f);
    return scale_quantize_and_pack(x, scale_buf);
}

// ---- step1_* : the Python-facing pipelines that compose the kernels above ----

// Dynamic INT8: compute scale from this call's residual, quantize, update a_hat_cache.
//   Op:       MoDiff temporal-delta quantize (dynamic int8) + cache update
//   Inputs:   x FP32 [N,C,H,W]; a_hat_cache FP32 [same] (in-place); residual_buf FP32 [same]
//             scratch; absmax_buf FP32 [1] (init 0); scale_buf FP32 [1]; inv_scale_buf FP32 [1];
//             retire_count int32 [1] (init 0); Q_level float (127 here); smooth_inv FP32 [C]
//             (empty = skip)
//   Outputs:  INT8 [same shape as x] (quantized residual); a_hat_cache updated in place
//   Computes: residual = (x*smooth_inv) - a_hat_cache; scale = 127/absmax(residual);
//             q = clamp(round(residual*scale), -127, 127); a_hat_cache += q/scale; out = q
//   Fuses:    sub_absmax_scale (residual + absmax + dynamic scale) + quantize + cache-update
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);

    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));

    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    quantize_and_update_ahat_kernel<<<grid_size, block_size, 0, stream>>>(
        residual_buf.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        x_int8.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_elements
    );

    return x_int8;
}

// Dynamic INT8, but a_hat_cache is left untouched (caller manages the cache itself).
//   Op:       MoDiff temporal-delta quantize (dynamic int8), NO cache write — benchmark-only
//   Inputs:   same as step1_quantize_fprop (x, a_hat_cache, residual_buf, absmax_buf,
//             scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv)
//   Outputs:  INT8 [same shape as x]; a_hat_cache NOT modified
//   Computes: residual = (x*smooth_inv) - a_hat_cache; scale = 127/absmax(residual);
//             out = clamp(round(residual*scale), -127, 127)
//   Fuses:    sub_absmax_scale + quantize (skips the a_hat write to isolate its cost in
//             microbenchmarks). Still READS a_hat_cache to form the residual — NOT a
//             cache-free substitute; for that use dynamic_quantize_int8_fprop
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_quantize_no_ahat_fprop(
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);

    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    quantize_only_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        residual_buf.data_ptr<float>(),
        x_int8.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_elements
    );

    return x_int8;
}

// Static-scale INT8: scale_buf is a precomputed calibration constant; updates a_hat_cache.
// Supports either an FP32 or FP16 a_hat_cache.
//   Op:       MoDiff temporal-delta quantize (static/calibrated int8) + cache update
//   Inputs:   x FP32 or FP16 [N,C,H,W]; a_hat_cache FP32 or FP16 [same] (in-place);
//             scale_buf FP32 [1] precomputed calibration scale; smooth_inv FP32 [C] (empty = skip)
//   Outputs:  INT8 [same shape as x]; a_hat_cache updated in place
//   Computes: residual = (x*smooth_inv) - a_hat_cache; q = clamp(round(residual*scale), -Q_b, Q_b);
//             a_hat_cache += q/scale; out = q, with Q_b = 7 when a4 else 127.
//             The saturation is what makes a CLIP ratio possible: with scale = Q_b/(r*absmax) the
//             top (1-r) of the range must saturate at +-Q_b, and clamping at 127 on a 4-bit layer
//             lets it through instead (docs/delta_clip_2026-08-06). `a4` names the datapath, so
//             there is no ceiling VALUE a caller can get wrong -- only which datapath this is
//   Fuses:    subtract + quantize + cache-update in one launch (no absmax reduction — the scale
//             is a constant). FP16-cache path reads fp16 x directly, upconverting in registers
//             (avoids a full fp32 copy of x)
//   Constraints: num_channels from smooth_inv.numel() (else x.size(1)); float4 path assumes
//                num_channels % 4 == 0 when smooth_inv is set
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_static_quantize_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    const float* smooth_ptr = nullptr;
    int num_channels = x.size(1);
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }

    if (a_hat_cache.scalar_type() == torch::kFloat16) {
        // Vec2 fast path requires num_channels % 2 == 0 for the smooth_inv half2 load
        // (same pre-existing, unenforced channels-last assumption the float4 kernels
        // already rely on for %4==0 -- not a new risk). Right-size the grid for the
        // 2-wide step rather than reusing the float4-sized `grid_size` above.
        const bool use_vec2 = (num_channels % 2 == 0);
        const int num_work_items_vec2 = (num_elements + 1) / 2;
        const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
        if (x.scalar_type() == torch::kHalf) {
            if (use_vec2) {
                static_quantize_and_update_ahat_kernel_int8_half_cache_vec2<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_int8.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_elements,
                    a4
                );
            } else {
                static_quantize_and_update_ahat_kernel_int8_half_cache<__half><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_int8.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_elements,
                    a4
                );
            }
        } else {
            if (use_vec2) {
                static_quantize_and_update_ahat_kernel_int8_half_cache_vec2<float><<<grid_size_vec2, block_size, 0, stream>>>(
                    x.data_ptr<float>(),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_int8.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_elements,
                    a4
                );
            } else {
                static_quantize_and_update_ahat_kernel_int8_half_cache<float><<<grid_size, block_size, 0, stream>>>(
                    x.data_ptr<float>(),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_int8.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_elements,
                    a4
                );
            }
        }
    } else {
        static_quantize_and_update_ahat_kernel_int8<<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(),
            a_hat_cache.data_ptr<float>(),
            x_int8.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(),
            smooth_ptr,
            num_channels,
            num_elements,
            a4
        );
    }

    return x_int8;
}

// Same as step1_static_quantize_fprop, but `x` is the pre-activation (ResBlock
// GroupNorm output *before* SiLU) -- applies SiLU inline in the kernel instead
// of requiring a separate F.silu(x) elementwise pass over the whole tensor
// first. Only implemented for the FP16-cache path: that's the only case the
// calibrated (production) modulated hot path actually uses -- see cache_dtype
// in int8_optimized.py, always FP16 once a layer is calibrated.
//   Op:       MoDiff temporal-delta quantize (static int8) + inline SiLU + cache update
//   Inputs:   x FP32 or FP16 [N,C,H,W] (ResBlock GroupNorm output BEFORE SiLU);
//             a_hat_cache FP16 [same] (in-place); scale_buf FP32 [1] calibration scale;
//             smooth_inv FP32 [C] (empty = skip)
//   Outputs:  INT8 [same shape as x]; a_hat_cache updated in place
//   Computes: xs = SiLU(x)*smooth_inv; residual = xs - a_hat_cache;
//             q = clamp(round(residual*scale), -Q_b, Q_b); a_hat_cache += q/scale; out = q
//             (Q_b = 7 when a4 else 127, derived from the datapath rather than passed as a value)
//   Fuses:    SiLU + subtract + quantize + cache-update (removes a separate full-tensor
//             F.silu(x) pass)
//   Constraints: FP16 a_hat_cache only (TORCH_CHECK) — the calibrated production path
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_static_quantize_fprop_silu(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv,
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument) on a 4-bit layer and it silently stayed 8-bit, which is what
    // the updown resize kernel did for months. A bool has no such value to get wrong, and
    // the saturation limit below is now a property of the datapath rather than an argument.
    bool a4
) {
    TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16,
                "step1_static_quantize_fprop_silu: only implemented for FP16 a_hat_cache");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    const float* smooth_ptr = nullptr;
    int num_channels = x.size(1);
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }

    // See step1_static_quantize_fprop for the use_vec2 gate rationale.
    const bool use_vec2 = (num_channels % 2 == 0);
    const int num_work_items_vec2 = (num_elements + 1) / 2;
    const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;

    if (x.scalar_type() == torch::kHalf) {
        if (use_vec2) {
            static_quantize_and_update_ahat_kernel_int8_half_cache_silu_vec2<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_elements,
                a4
            );
        } else {
            static_quantize_and_update_ahat_kernel_int8_half_cache_silu<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_elements,
                a4
            );
        }
    } else {
        if (use_vec2) {
            static_quantize_and_update_ahat_kernel_int8_half_cache_silu_vec2<float><<<grid_size_vec2, block_size, 0, stream>>>(
                x.data_ptr<float>(),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_elements,
                a4
            );
        } else {
            static_quantize_and_update_ahat_kernel_int8_half_cache_silu<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_elements,
                a4
            );
        }
    }

    return x_int8;
}

// Dynamic INT4: like step1_quantize_fprop but quantizes to [-7,7] and packs 2
// elements per byte; updates a_hat_cache.
//   Op:       MoDiff temporal-delta quantize (dynamic int4) + pack + cache update
//   Inputs:   x FP32 [N,C,H,W]; a_hat_cache FP32 [same] (in-place); residual_buf FP32 [same];
//             absmax_buf FP32 [1] (init 0); scale_buf FP32 [1]; inv_scale_buf FP32 [1];
//             retire_count int32 [1] (init 0); Q_level float (7 here); smooth_inv FP32 [C]
//   Outputs:  INT8 [N,H,W,C/2] packed int4; a_hat_cache updated in place
//   Computes: residual = (x*smooth_inv) - a_hat_cache; scale = 7/absmax(residual);
//             q = clamp(round(residual*scale), -7, 7); a_hat_cache += q/scale; out = pack(q)
//   Fuses:    sub_absmax_scale + quantize + nibble-pack + cache-update
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);

    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    quantize_pack_and_update_ahat_kernel_int4<<<grid_size, block_size, 0, stream>>>(
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

    return x_packed.view({N, H, W, C / 2});
}

// Static-scale INT4: precomputed scale, packs 2 elements per byte, updates a_hat_cache
// (FP32 or FP16).
//   Op:       MoDiff temporal-delta quantize (static/calibrated int4) + pack + cache update
//   Inputs:   x FP32 or FP16 [N,C,H,W]; a_hat_cache FP32 or FP16 [same] (in-place);
//             scale_buf FP32 [1] precomputed calibration scale; smooth_inv FP32 [C] (empty = skip)
//   Outputs:  INT8 [N,H,W,C/2] packed int4; a_hat_cache updated in place
//   Computes: residual = (x*smooth_inv) - a_hat_cache; q = clamp(round(residual*scale), -7, 7);
//             a_hat_cache += q/scale; out = pack(q)
//   Fuses:    subtract + quantize + nibble-pack + cache-update (constant scale, no absmax)
//   Constraints: num_channels from smooth_inv.numel() (else x.size(1))
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_static_quantize_pack_int4_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    const float* smooth_ptr = nullptr;
    int num_channels = x.size(1);
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }

    if (a_hat_cache.scalar_type() == torch::kFloat16) {
        // Vec2 fast path requires num_channels % 2 == 0 for the smooth_inv half2 load -- see
        // step1_static_quantize_fprop's identical gate for the INT8 sibling. Right-size the grid
        // for the 2-wide step rather than reusing the float4-sized `grid_size` above.
        const bool use_vec2 = (num_channels % 2 == 0);
        const int num_work_items_vec2 = (num_input + 1) / 2;
        const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
        if (x.scalar_type() == torch::kHalf) {
            if (use_vec2) {
                static_quantize_pack_and_update_ahat_kernel_int4_half_cache_vec2<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_packed.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_input
                );
            } else {
                static_quantize_pack_and_update_ahat_kernel_int4_half_cache<__half><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_packed.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_input
                );
            }
        } else {
            if (use_vec2) {
                static_quantize_pack_and_update_ahat_kernel_int4_half_cache_vec2<float><<<grid_size_vec2, block_size, 0, stream>>>(
                    x.data_ptr<float>(),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_packed.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_input
                );
            } else {
                static_quantize_pack_and_update_ahat_kernel_int4_half_cache<float><<<grid_size, block_size, 0, stream>>>(
                    x.data_ptr<float>(),
                    reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                    x_packed.data_ptr<int8_t>(),
                    scale_buf.data_ptr<float>(),
                    smooth_ptr,
                    num_channels,
                    num_input
                );
            }
        }
    } else {
        static_quantize_pack_and_update_ahat_kernel_int4<<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(),
            a_hat_cache.data_ptr<float>(),
            x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(),
            smooth_ptr,
            num_channels,
            num_input
        );
    }

    int N = x.size(0);
    int C = x.size(1);
    // 2D is a first-class case: the MoDiff Linear path quantizes a [M, K] activation with this same
    // kernel (the elementwise body only walks numel), so the output reshape must not assume NCHW.
    // Without this the Linear path died with "Dimension out of range ... but got 2".
    if (x.dim() == 2) {
        return x_packed.view({x.size(0), x.size(1) / 2});
    }
    int H = x.size(2);
    int W = x.size(3);

    return x_packed.view({N, H, W, C / 2});
}

// Cache-free static int8 quantize (baseline conv): fp16/fp32 x + static scale (+optional smooth),
// NO a_hat. Bit-identical output to step1_static_quantize_fprop(x, a_hat=0, scale) but without the
// a_hat read/write + the caller's per-call a_hat zero-fill. Output int8, same shape/layout as x.
torch::Tensor step1_static_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : (int)x.size(1);
    // Vec2 fast path requires num_channels % 2 == 0 for the smooth_inv half2 load -- see
    // step1_static_quantize_fprop's identical gate. Right-size the grid for the 2-wide step.
    const bool use_vec2 = (num_channels % 2 == 0);
    const int num_work_items_vec2 = (num_elements + 1) / 2;
    const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
    if (x.scalar_type() == torch::kHalf) {
        if (use_vec2) {
            static_quantize_int8_noahat_vec2_kernel<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        } else {
            static_quantize_int8_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        }
    } else {
        if (use_vec2) {
            static_quantize_int8_noahat_vec2_kernel<float><<<grid_size_vec2, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        } else {
            static_quantize_int8_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        }
    }
    return x_int8;
}

// Cache-free static int4 quantize+pack (baseline conv): fp16/fp32 x + static scale (+optional
// smooth), NO a_hat. Output packed int8 [N,H,W,C/2] (same layout as step1_static_quantize_pack_int4_fprop).
torch::Tensor step1_static_quantize_pack_int4_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : (int)x.size(1);
    // See step1_static_quantize_pack_int4_fprop for the use_vec2 gate rationale.
    const bool use_vec2 = (num_channels % 2 == 0);
    const int num_work_items_vec2 = (num_input + 1) / 2;
    const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
    if (x.scalar_type() == torch::kHalf) {
        if (use_vec2) {
            static_quantize_pack_int4_noahat_vec2_kernel<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        } else {
            static_quantize_pack_int4_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        }
    } else {
        if (use_vec2) {
            static_quantize_pack_int4_noahat_vec2_kernel<float><<<grid_size_vec2, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        } else {
            static_quantize_pack_int4_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        }
    }
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    return x_packed.view({N, H, W, C / 2});
}

// ---- Upsample(nearest, 2x) + static quantize fusion (baseline conv, NO a_hat): fold the
// Upsample.forward -> F.interpolate(scale_factor=2, mode="nearest") pass into the following
// conv's quantize prologue. Nearest 2x upsample repeats each source pixel into a 2x2 block, so
// this reads the SMALL pre-upsample [N,C,H,W] tensor once and writes the LARGE quantized
// [N,C,2H,2W] output directly with 2x2-repeat addressing -- the intermediate fp16
// [N,C,2H,2W] upsampled tensor (what F.interpolate would materialize, then the plain
// static_quantize_*_noahat_kernel would re-read) is never allocated. Per-element quantize math
// is copy-pasted from static_quantize_int8_noahat_kernel / static_quantize_pack_int4_noahat_kernel
// (same scale/smooth_inv/clamp), just reading through the repeat-by-2 address instead of `x[i]`
// directly -- so output is bit-identical to F.interpolate(x) -> static_quantize_*_noahat_kernel.
template <typename T_IN>
__global__ void upsample2x_quantize_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out,
    // MoDiff, opt-in: subtract a_hat before quantizing and advance it in place. nullptr keeps the
    // baseline behaviour BIT-IDENTICALLY (cache reads as 0 and nothing is stored), so one kernel
    // serves both modes -- no cloned MoDiff twin. The loop already grids over OUTPUT elements,
    // which is what makes this work: nearest-2x upsample gives each output position its own a_hat
    // entry, and a clone gridding over input positions would have had to quantize four times.
    __half* __restrict__ a_hat_cache) {
    float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const int Wo = W * 2, Ho = H * 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long i = idx; i < num_elements_out; i += (long)blockDim.x * gridDim.x) {
        int c = (int)(i % C);
        long rest = i / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho >> 1, wi = wo >> 1;
        long i_in = ((n * H + hi) * (long)W + wi) * C + c;
        float xval = load_as_float(x, (int)i_in);
        if (smooth_inv != nullptr) xval *= smooth_inv[c % num_channels];
        const float cache = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[i]) : 0.0f;
        float q = fmaxf(-127.0f, fminf(127.0f, roundf((xval - cache) * scale)));
        if (a_hat_cache != nullptr) a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        output_int8[i] = static_cast<int8_t>(q);
    }
}

template <typename T_IN>
__global__ void upsample2x_quantize_pack_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out,
    __half* __restrict__ a_hat_cache) {   // nullptr => baseline, bit-identical (see int8 sibling)
    float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const int Wo = W * 2, Ho = H * 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    for (long base = idx * 2; base < num_elements_out; base += stride * 2) {
        int c0 = (int)(base % C);
        long rest = base / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho >> 1, wi = wo >> 1;
        long pix_in = (n * H + hi) * (long)W + wi;
        float x0 = load_as_float(x, (int)(pix_in * C + c0));
        if (smooth_inv != nullptr) x0 *= smooth_inv[c0 % num_channels];
        const float c_0 = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[base]) : 0.0f;
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - c_0) * scale)));
        if (a_hat_cache != nullptr) a_hat_cache[base] = __float2half_rn(c_0 + q0 * inv_scale);
        float q1 = 0.0f;
        if (base + 1 < num_elements_out) {
            float x1 = load_as_float(x, (int)(pix_in * C + c0 + 1));
            if (smooth_inv != nullptr) x1 *= smooth_inv[(c0 + 1) % num_channels];
            const float c_1 = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[base + 1]) : 0.0f;
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf((x1 - c_1) * scale)));
            if (a_hat_cache != nullptr)
                a_hat_cache[base + 1] = __float2half_rn(c_1 + q1 * inv_scale);
        }
        output_packed[base / 2] = (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Op: fused Upsample(nearest,2x) + cache-free static int8 quantize (baseline conv, no a_hat).
// Inputs: x FP16/FP32 [N,C,H,W] channels_last (pre-upsample); scale_buf FP32 [1]; smooth_inv FP32 [C] (empty = skip).
// Output: INT8 [N,C,2H,2W] channels_last -- feeds the Upsample.conv's conv2d_int*_evt_* directly.
torch::Tensor upsample2x_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache) {
    // Name kept for pybind/API compatibility (csrc/modiff_kernels_api.h notes the _noahat spelling
    // is load-bearing); an EMPTY a_hat_cache is the no-a_hat baseline it was named for.
    TORCH_CHECK(x.dim() == 4, "upsample2x_quantize_noahat_fprop: x must be [N,C,H,W]");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto y = torch::empty({N, C, H * 2, W * 2}, options);
    long num_elements_out = (long)N * C * H * 2 * W * 2;
    int block_size = 256;
    int grid_size = (int)std::min<long>((num_elements_out + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    __half* cache_ptr = nullptr;
    if (a_hat_cache.numel() > 0) {
        TORCH_CHECK(a_hat_cache.scalar_type() == torch::kHalf,
                    "upsample2x_quantize_noahat_fprop: a_hat_cache must be fp16");
        TORCH_CHECK(a_hat_cache.numel() == num_elements_out,
                    "upsample2x_quantize_noahat_fprop: a_hat_cache must be POST-upsample sized "
                    "(N*C*2H*2W)");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
    if (x.scalar_type() == torch::kHalf) {
        upsample2x_quantize_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out,
            cache_ptr);
    } else {
        upsample2x_quantize_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out,
            cache_ptr);
    }
    return y;
}

// int4-packed counterpart of upsample2x_quantize_noahat_fprop. Output packed int8 [N,2H,2W,C/2]
// (same layout convention as step1_static_quantize_pack_int4_noahat_fprop). Requires C%2==0
// (channel-pair packing), matching every other int4 quantize kernel in this file.
torch::Tensor upsample2x_quantize_pack_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache) {
    TORCH_CHECK(x.dim() == 4, "upsample2x_quantize_pack_noahat_fprop: x must be [N,C,H,W]");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % 2 == 0, "upsample2x_quantize_pack_noahat_fprop: C must be even for int4 packing");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    long num_elements_out = (long)N * C * H * 2 * W * 2;
    long num_output = num_elements_out / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    int block_size = 256;
    long num_work_items = (num_elements_out + 3) / 4;
    int grid_size = (int)std::min<long>((num_work_items + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    __half* cache_ptr = nullptr;
    if (a_hat_cache.numel() > 0) {
        TORCH_CHECK(a_hat_cache.scalar_type() == torch::kHalf,
                    "upsample2x_quantize_pack_noahat_fprop: a_hat_cache must be fp16");
        TORCH_CHECK(a_hat_cache.numel() == num_elements_out,
                    "upsample2x_quantize_pack_noahat_fprop: a_hat_cache must be POST-upsample sized");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
    if (x.scalar_type() == torch::kHalf) {
        upsample2x_quantize_pack_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out, cache_ptr);
    } else {
        upsample2x_quantize_pack_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out, cache_ptr);
    }
    return x_packed.view({N, H * 2, W * 2, C / 2});
}

// Bit-exact reproduction of ATen's avg_pool2d(kernel_size=2, stride=2, padding=0) forward for a
// single 2x2 non-overlapping window: sums the 4 window elements as float in (kh,kw) row-major
// order, scales by 0.25f, then -- ONLY when T_IN is half-precision storage -- rounds that sum
// through __half (round-to-nearest-even, matching what avg_pool2d actually writes to its fp16
// output tensor) before handing it back out as float. Verified against a real
// nn.AvgPool2d(2,2).cuda() fp16 forward: this reproduces it bit-for-bit (torch.equal), not just
// approximately, which is the same bar every other kernel in this file is held to.
template <typename T_IN> __device__ __forceinline__ float avgpool4_as_stored(float sum);
template <> __device__ __forceinline__ float avgpool4_as_stored<__half>(float sum) {
    return __half2float(__float2half_rn(sum));
}
template <> __device__ __forceinline__ float avgpool4_as_stored<float>(float sum) { return sum; }

// Op: fused Downsample(avg_pool,2x2,stride2) + cache-free static int8 quantize (baseline conv, no
// a_hat). Mirrors upsample2x_quantize_noahat_kernel's structure and the fusion rationale in its
// header comment above, but for the *down* transition: reads the LARGE pre-pool [N,C,H,W] tensor
// once, computes each output pixel's 2x2 average directly (via avgpool4_as_stored, bit-exact to
// nn.AvgPool2d's fp16 output), and quantizes -- never materializing the fp16 pooled [N,C,H/2,W/2]
// intermediate that Downsample.forward would otherwise write and the plain
// static_quantize_*_noahat_kernel would re-read.
template <typename T_IN>
__global__ void avgpool2x_quantize_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out) {
    float scale = *scale_ptr;
    const int Wo = W / 2, Ho = H / 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long i = idx; i < num_elements_out; i += (long)blockDim.x * gridDim.x) {
        int c = (int)(i % C);
        long rest = i / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho * 2, wi = wo * 2;
        long row0 = (n * H + hi) * (long)W + wi;
        long row1 = (n * H + hi + 1) * (long)W + wi;
        float sum = load_as_float(x, (int)(row0 * C + c)) + load_as_float(x, (int)((row0 + 1) * C + c))
                  + load_as_float(x, (int)(row1 * C + c)) + load_as_float(x, (int)((row1 + 1) * C + c));
        float xval = avgpool4_as_stored<T_IN>(sum * 0.25f);
        if (smooth_inv != nullptr) xval *= smooth_inv[c % num_channels];
        float q = fmaxf(-127.0f, fminf(127.0f, roundf(xval * scale)));
        output_int8[i] = static_cast<int8_t>(q);
    }
}

template <typename T_IN>
__global__ void avgpool2x_quantize_pack_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out) {
    float scale = *scale_ptr;
    const int Wo = W / 2, Ho = H / 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    for (long base = idx * 2; base < num_elements_out; base += stride * 2) {
        int c0 = (int)(base % C);
        long rest = base / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho * 2, wi = wo * 2;
        long row0 = (n * H + hi) * (long)W + wi;
        long row1 = (n * H + hi + 1) * (long)W + wi;
        float sum0 = load_as_float(x, (int)(row0 * C + c0)) + load_as_float(x, (int)((row0 + 1) * C + c0))
                   + load_as_float(x, (int)(row1 * C + c0)) + load_as_float(x, (int)((row1 + 1) * C + c0));
        float x0 = avgpool4_as_stored<T_IN>(sum0 * 0.25f);
        if (smooth_inv != nullptr) x0 *= smooth_inv[c0 % num_channels];
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(x0 * scale)));
        float q1 = 0.0f;
        if (base + 1 < num_elements_out) {
            int c1 = c0 + 1;
            float sum1 = load_as_float(x, (int)(row0 * C + c1)) + load_as_float(x, (int)((row0 + 1) * C + c1))
                       + load_as_float(x, (int)(row1 * C + c1)) + load_as_float(x, (int)((row1 + 1) * C + c1));
            float x1 = avgpool4_as_stored<T_IN>(sum1 * 0.25f);
            if (smooth_inv != nullptr) x1 *= smooth_inv[c1 % num_channels];
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf(x1 * scale)));
        }
        output_packed[base / 2] = (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Op: fused Downsample(avg_pool,2x2) + cache-free static int8 quantize (baseline conv, no a_hat).
// Inputs: x FP16/FP32 [N,C,H,W] channels_last (pre-pool, LARGE); scale_buf FP32 [1]; smooth_inv FP32 [C] (empty = skip).
// Output: INT8 [N,C,H/2,W/2] channels_last -- feeds the downsample ResBlock's in_conv directly.
torch::Tensor avgpool2x_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    TORCH_CHECK(x.dim() == 4, "avgpool2x_quantize_noahat_fprop: x must be [N,C,H,W]");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "avgpool2x_quantize_noahat_fprop: H,W must be even");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto y = torch::empty({N, C, H / 2, W / 2}, options);
    long num_elements_out = (long)N * C * (H / 2) * (W / 2);
    int block_size = 256;
    int grid_size = (int)std::min<long>((num_elements_out + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    if (x.scalar_type() == torch::kHalf) {
        avgpool2x_quantize_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    } else {
        avgpool2x_quantize_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    }
    return y;
}

// int4-packed counterpart of avgpool2x_quantize_noahat_fprop. Output packed int8 [N,H/2,W/2,C/2]
// (same layout convention as upsample2x_quantize_pack_noahat_fprop). Requires C%2==0.
torch::Tensor avgpool2x_quantize_pack_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    TORCH_CHECK(x.dim() == 4, "avgpool2x_quantize_pack_noahat_fprop: x must be [N,C,H,W]");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "avgpool2x_quantize_pack_noahat_fprop: H,W must be even");
    TORCH_CHECK(C % 2 == 0, "avgpool2x_quantize_pack_noahat_fprop: C must be even for int4 packing");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    long num_elements_out = (long)N * C * (H / 2) * (W / 2);
    long num_output = num_elements_out / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    int block_size = 256;
    long num_work_items = (num_elements_out + 3) / 4;
    int grid_size = (int)std::min<long>((num_work_items + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    if (x.scalar_type() == torch::kHalf) {
        avgpool2x_quantize_pack_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    } else {
        avgpool2x_quantize_pack_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    }
    return x_packed.view({N, H / 2, W / 2, C / 2});
}

// Same as step1_static_quantize_pack_int4_fprop, but `x` is the pre-activation
// (ResBlock GroupNorm output *before* SiLU) -- applies SiLU inline in the
// kernel. See step1_static_quantize_fprop_silu's comment; only implemented
// for the FP16-cache path used by the calibrated modulated hot path.
//   Op:       MoDiff temporal-delta quantize (static int4) + inline SiLU + pack + cache update
//   Inputs:   x FP32 or FP16 [N,C,H,W] (ResBlock GroupNorm output BEFORE SiLU);
//             a_hat_cache FP16 [same] (in-place); scale_buf FP32 [1] calibration scale;
//             smooth_inv FP32 [C] (empty = skip)
//   Outputs:  INT8 [N,H,W,C/2] packed int4; a_hat_cache updated in place
//   Computes: xs = SiLU(x)*smooth_inv; residual = xs - a_hat_cache;
//             q = clamp(round(residual*scale), -7, 7); a_hat_cache += q/scale; out = pack(q)
//   Fuses:    SiLU + subtract + quantize + nibble-pack + cache-update
//   Constraints: FP16 a_hat_cache only (TORCH_CHECK) — the calibrated production path
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_static_quantize_pack_int4_fprop_silu(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv
) {
    TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16,
                "step1_static_quantize_pack_int4_fprop_silu: only implemented for FP16 a_hat_cache");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    const float* smooth_ptr = nullptr;
    int num_channels = x.size(1);
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }

    // See step1_static_quantize_pack_int4_fprop for the use_vec2 gate rationale.
    const bool use_vec2 = (num_channels % 2 == 0);
    const int num_work_items_vec2 = (num_input + 1) / 2;
    const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
    if (x.scalar_type() == torch::kHalf) {
        if (use_vec2) {
            static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu_vec2<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_input
            );
        } else {
            static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_input
            );
        }
    } else {
        if (use_vec2) {
            static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu_vec2<float><<<grid_size_vec2, block_size, 0, stream>>>(
                x.data_ptr<float>(),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_input
            );
        } else {
            static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_input
            );
        }
    }

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    return x_packed.view({N, H, W, C / 2});
}

// Dynamic INT4 without a_hat_cache update (caller manages the cache itself).
//   Op:       MoDiff temporal-delta quantize (dynamic int4) + pack, NO cache write — benchmark-only
//   Inputs:   same as step1_quantize_pack_int4_fprop (x, a_hat_cache, residual_buf, absmax_buf,
//             scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv)
//   Outputs:  INT8 [N,H,W,C/2] packed int4; a_hat_cache NOT modified
//   Computes: residual = (x*smooth_inv) - a_hat_cache; scale = 7/absmax(residual);
//             out = pack(clamp(round(residual*scale), -7, 7))
//   Fuses:    sub_absmax_scale + quantize + nibble-pack (skips the a_hat write). Still READS
//             a_hat_cache — NOT a cache-free substitute; for that use dynamic_quantize_pack_int4_fprop
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_quantize_pack_int4_no_ahat_fprop(
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);

    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    quantize_pack_only_kernel_int4<<<grid_size, block_size, 0, stream>>>(
        residual_buf.data_ptr<float>(),
        x_packed.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_input
    );

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    return x_packed.view({N, H, W, C / 2});
}
