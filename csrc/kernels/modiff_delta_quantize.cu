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
#include "../modiff_kernels_api.h"

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

        float r0 = x_v.x - c_v.x;
        float r1 = x_v.y - c_v.y;
        float r2 = x_v.z - c_v.z;
        float r3 = x_v.w - c_v.w;

        float q0 = fmaxf(-127.0f, fminf(127.0f, roundf(r0 * scale)));
        float q1 = fmaxf(-127.0f, fminf(127.0f, roundf(r1 * scale)));
        float q2 = fmaxf(-127.0f, fminf(127.0f, roundf(r2 * scale)));
        float q3 = fmaxf(-127.0f, fminf(127.0f, roundf(r3 * scale)));

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
            float q = fmaxf(-127.0f, fminf(127.0f, roundf(r * scale)));
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
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float xval = load_as_float(x, i);
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        float cache = __half2float(a_hat_cache[i]);
        float q = fmaxf(-127.0f, fminf(127.0f, roundf((xval - cache) * scale)));
        a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        output_int8[i] = static_cast<int8_t>(q);
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
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float xval = silu_f(load_as_float(x, i));
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        float cache = __half2float(a_hat_cache[i]);
        float q = fmaxf(-127.0f, fminf(127.0f, roundf((xval - cache) * scale)));
        a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        output_int8[i] = static_cast<int8_t>(q);
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
torch::Tensor step1_static_quantize_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv
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
        if (x.scalar_type() == torch::kHalf) {
            static_quantize_and_update_ahat_kernel_int8_half_cache<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_elements
            );
        } else {
            static_quantize_and_update_ahat_kernel_int8_half_cache<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(),
                reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
                x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(),
                smooth_ptr,
                num_channels,
                num_elements
            );
        }
    } else {
        static_quantize_and_update_ahat_kernel_int8<<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(),
            a_hat_cache.data_ptr<float>(),
            x_int8.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(),
            smooth_ptr,
            num_channels,
            num_elements
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
torch::Tensor step1_static_quantize_fprop_silu(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv
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

    if (x.scalar_type() == torch::kHalf) {
        static_quantize_and_update_ahat_kernel_int8_half_cache_silu<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
            x_int8.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(),
            smooth_ptr,
            num_channels,
            num_elements
        );
    } else {
        static_quantize_and_update_ahat_kernel_int8_half_cache_silu<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>()),
            x_int8.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(),
            smooth_ptr,
            num_channels,
            num_elements
        );
    }

    return x_int8;
}

// Dynamic INT4: like step1_quantize_fprop but quantizes to [-7,7] and packs 2
// elements per byte; updates a_hat_cache.
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
        if (x.scalar_type() == torch::kHalf) {
            static_quantize_pack_and_update_ahat_kernel_int4_half_cache<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
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
    int H = x.size(2);
    int W = x.size(3);

    return x_packed.view({N, H, W, C / 2});
}

// Same as step1_static_quantize_pack_int4_fprop, but `x` is the pre-activation
// (ResBlock GroupNorm output *before* SiLU) -- applies SiLU inline in the
// kernel. See step1_static_quantize_fprop_silu's comment; only implemented
// for the FP16-cache path used by the calibrated modulated hot path.
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

    if (x.scalar_type() == torch::kHalf) {
        static_quantize_pack_and_update_ahat_kernel_int4_half_cache_silu<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
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

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    return x_packed.view({N, H, W, C / 2});
}

// Dynamic INT4 without a_hat_cache update (caller manages the cache itself).
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
