// Shared int8/int4-code a_hat load/store. a_hat_cache is either fp16 or int8 codes
// of the same NHWC shape; int8 codes dequant as code * scale[0] and clamp to ±scale[1]
// (127 for MODIFF_AHAT_BITS=8, 7 for bits=4). Int4 a_hat is stored unpacked in int8
// (1 byte/elem) so scalar kernels cannot nibble-race; the grid is 4-bit, the footprint
// matches int8. Scale is a 2-element fp32 tensor [scale, qmax].
#pragma once

#include <torch/extension.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float ahat_load_i8(const int8_t* p, long i, float s) {
    return (float)p[i] * s;
}
__device__ __forceinline__ float2 ahat_load2_i8(const int8_t* p, long i, float s) {
    const char2 v = *reinterpret_cast<const char2*>(p + i);
    return make_float2((float)v.x * s, (float)v.y * s);
}
__device__ __forceinline__ void ahat_store_i8(int8_t* p, long i, float v, float inv_s, float lim) {
    p[i] = (int8_t)fmaxf(-lim, fminf(lim, roundf(v * inv_s)));
}
__device__ __forceinline__ void ahat_store2_i8(int8_t* p, long i, float2 v, float inv_s, float lim) {
    char2 o;
    o.x = (signed char)(int)fmaxf(-lim, fminf(lim, roundf(v.x * inv_s)));
    o.y = (signed char)(int)fmaxf(-lim, fminf(lim, roundf(v.y * inv_s)));
    *reinterpret_cast<char2*>(p + i) = o;
}

__device__ __forceinline__ void ahat_qparams(bool i8, const float* q,
                                            float& s, float& inv, float& lim) {
    if (!i8) { s = 1.0f; inv = 1.0f; lim = 127.0f; return; }
    s = q[0];
    inv = 1.0f / s;
    lim = q[1];
}

__device__ __forceinline__ float ahat_load(const __half* p, long i, bool i8, float s) {
    if (i8) return ahat_load_i8(reinterpret_cast<const int8_t*>(p), i, s);
    return __half2float(p[i]);
}
__device__ __forceinline__ float2 ahat_load2(const __half* p, long i, bool i8, float s) {
    if (i8) return ahat_load2_i8(reinterpret_cast<const int8_t*>(p), i, s);
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}
__device__ __forceinline__ void ahat_store(__half* p, long i, float v, bool i8, float inv_s, float lim) {
    if (i8) { ahat_store_i8(reinterpret_cast<int8_t*>(p), i, v, inv_s, lim); return; }
    p[i] = __float2half_rn(v);
}
__device__ __forceinline__ void ahat_store2(__half* p, long i, float2 v, bool i8, float inv_s, float lim) {
    if (i8) { ahat_store2_i8(reinterpret_cast<int8_t*>(p), i, v, inv_s, lim); return; }
    reinterpret_cast<__half2*>(p)[i >> 1] = __float22half2_rn(v);
}

// Host: fp16 a_hat, or int8 codes + fp32 [scale, qmax]. The __half* is a type pun.
static void bind_ahat_cache(const torch::Tensor& a_hat_cache, const torch::Tensor& ahat_scale,
                            __half*& cache_ptr, bool& ahat_i8, const float*& ahat_qscale_ptr,
                            const char* where) {
    ahat_i8 = a_hat_cache.scalar_type() == torch::kInt8;
    ahat_qscale_ptr = nullptr;
    if (ahat_i8) {
        TORCH_CHECK(ahat_scale.defined() && ahat_scale.numel() >= 2
                    && ahat_scale.scalar_type() == torch::kFloat32,
                    where, ": int8 a_hat_cache requires a 2-element fp32 ahat_scale [scale, qmax]");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<int8_t>());
        ahat_qscale_ptr = ahat_scale.data_ptr<float>();
    } else {
        TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16,
                    where, ": a_hat_cache must be fp16 or int8");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
}
