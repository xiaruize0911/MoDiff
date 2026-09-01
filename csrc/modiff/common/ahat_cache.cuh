// Shared a_hat load/store + I-MoDiff integer residual step.
//
// Storage kinds:
//   fp16          — float residual (shipped). ahat_i8=false.
//   int8 dequant  — held-int8: load code*scale[0], store round(v/scale). ahat_i8=true, scale[0]>0.
//                   This is MODIFF_AHAT_BITS=8/4 with IMODE off (FID 121 path).
//   I-MoDiff      — integer sub/add, no dequant. Signalled by scale[0]==0.
//                   bits=16: int16 buffer, lim=32767.
//                   bits=8/4: int8 buffer, lim=127 or 7 (unpacked; no nibble packing).
//
// I-MoDiff formula (same grid s* = 1/δ for a_hat and q):
//   x_i = sat(round(x * δ), ±ahat_lim)
//   q   = sat(x_i - a, ±code_lim)
//   a  += q   (sat to ±ahat_lim)
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

// ahat_s==0 means I-MoDiff (integer). inv is unused on that path.
__device__ __forceinline__ void ahat_qparams(bool i8, const float* q,
                                            float& s, float& inv, float& lim) {
    if (!i8) { s = 1.0f; inv = 1.0f; lim = 127.0f; return; }
    s = q[0];
    lim = q[1];
    inv = (s == 0.0f) ? 0.0f : (1.0f / s);
}

__device__ __forceinline__ bool ahat_is_imode(bool i8, float s) {
    return i8 && s == 0.0f;
}

__device__ __forceinline__ int ahat_load_int(const __half* p, long i, float lim) {
    if (lim > 127.5f)
        return (int)reinterpret_cast<const int16_t*>(p)[i];
    return (int)reinterpret_cast<const int8_t*>(p)[i];
}
__device__ __forceinline__ void ahat_store_int(__half* p, long i, int v, float lim) {
    const int lo = -(int)lim, hi = (int)lim;
    if (v < lo) v = lo;
    else if (v > hi) v = hi;
    if (lim > 127.5f)
        reinterpret_cast<int16_t*>(p)[i] = (int16_t)v;
    else
        reinterpret_cast<int8_t*>(p)[i] = (int8_t)v;
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

// One-element residual quantize + optional a_hat commit. Returns q as float (fits int8).
// WriteAhat is a compile-time flag so skip-K DCE's the stores instead of predicating them.
template <bool WriteAhat>
__device__ __forceinline__ float ahat_quant_update_w(
    __half* cache, long i, float xval,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim)
{
    if (ahat_is_imode(ahat_i8, ahat_s)) {
        const int a = ahat_load_int(cache, i, ahat_lim);
        const int xi = (int)fmaxf(-ahat_lim, fminf(ahat_lim, roundf(xval * scale)));
        const int q = (int)fmaxf(-code_lim, fminf(code_lim, (float)(xi - a)));
        if constexpr (WriteAhat) ahat_store_int(cache, i, a + q, ahat_lim);
        return (float)q;
    }
    const float c = ahat_load(cache, i, ahat_i8, ahat_s);
    const float q = fmaxf(-code_lim, fminf(code_lim, roundf((xval - c) * scale)));
    if constexpr (WriteAhat) ahat_store(cache, i, c + q * inv_scale, ahat_i8, ahat_inv, ahat_lim);
    return q;
}
__device__ __forceinline__ float ahat_quant_update(
    __half* cache, long i, float xval,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    bool write_ahat)
{
    return write_ahat
        ? ahat_quant_update_w<true>(cache, i, xval, scale, inv_scale, code_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim)
        : ahat_quant_update_w<false>(cache, i, xval, scale, inv_scale, code_lim,
                                     ahat_i8, ahat_s, ahat_inv, ahat_lim);
}

// Pair version. Also writes the pre-clamp float residual into d0/d1 (for absmax reporting).
template <bool WriteAhat>
__device__ __forceinline__ void ahat_quant_update2_w(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    float& q0, float& q1, float& d0, float& d1)
{
    if (ahat_is_imode(ahat_i8, ahat_s)) {
        int a0, a1;
        if (ahat_lim > 127.5f) {
            const short2 v = *reinterpret_cast<const short2*>(
                reinterpret_cast<const int16_t*>(cache) + i);
            a0 = (int)v.x; a1 = (int)v.y;
        } else {
            const char2 v = *reinterpret_cast<const char2*>(
                reinterpret_cast<const int8_t*>(cache) + i);
            a0 = (int)v.x; a1 = (int)v.y;
        }
        const int xi0 = (int)fmaxf(-ahat_lim, fminf(ahat_lim, roundf(x0 * scale)));
        const int xi1 = (int)fmaxf(-ahat_lim, fminf(ahat_lim, roundf(x1 * scale)));
        const int iq0 = (int)fmaxf(-code_lim, fminf(code_lim, (float)(xi0 - a0)));
        const int iq1 = (int)fmaxf(-code_lim, fminf(code_lim, (float)(xi1 - a1)));
        d0 = x0 - (float)a0 * inv_scale;
        d1 = x1 - (float)a1 * inv_scale;
        q0 = (float)iq0;
        q1 = (float)iq1;
        if constexpr (WriteAhat) {
            ahat_store_int(cache, i, a0 + iq0, ahat_lim);
            ahat_store_int(cache, i + 1, a1 + iq1, ahat_lim);
        }
        return;
    }
    const float2 c = ahat_load2(cache, i, ahat_i8, ahat_s);
    d0 = x0 - c.x;
    d1 = x1 - c.y;
    q0 = fmaxf(-code_lim, fminf(code_lim, roundf(d0 * scale)));
    q1 = fmaxf(-code_lim, fminf(code_lim, roundf(d1 * scale)));
    if constexpr (WriteAhat)
        ahat_store2(cache, i, make_float2(c.x + q0 * inv_scale, c.y + q1 * inv_scale),
                    ahat_i8, ahat_inv, ahat_lim);
}
__device__ __forceinline__ void ahat_quant_update2(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    bool write_ahat, float& q0, float& q1, float& d0, float& d1)
{
    if (write_ahat)
        ahat_quant_update2_w<true>(cache, i, x0, x1, scale, inv_scale, code_lim,
                                   ahat_i8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1);
    else
        ahat_quant_update2_w<false>(cache, i, x0, x1, scale, inv_scale, code_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1);
}

template <bool WriteAhat>
__device__ __forceinline__ void ahat_quant_update2_w(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    float& q0, float& q1)
{
    float d0, d1;
    ahat_quant_update2_w<WriteAhat>(cache, i, x0, x1, scale, inv_scale, code_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1);
}

__device__ __forceinline__ void ahat_quant_update2(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    bool write_ahat, float& q0, float& q1)
{
    float d0, d1;
    ahat_quant_update2(cache, i, x0, x1, scale, inv_scale, code_lim,
                       ahat_i8, ahat_s, ahat_inv, ahat_lim, write_ahat, q0, q1, d0, d1);
}

// Host: fp16, int8 (held or I-MoDiff), or int16 (I-MoDiff). The __half* is a type pun.
// I-MoDiff: ahat_scale = [0, qmax]. Held int8: ahat_scale = [dequant_scale, qmax] with scale>0.
static void bind_ahat_cache(const torch::Tensor& a_hat_cache, const torch::Tensor& ahat_scale,
                            __half*& cache_ptr, bool& ahat_i8, const float*& ahat_qscale_ptr,
                            const char* where) {
    ahat_i8 = false;
    ahat_qscale_ptr = nullptr;
    const auto st = a_hat_cache.scalar_type();
    if (st == torch::kInt16) {
        TORCH_CHECK(ahat_scale.defined() && ahat_scale.numel() >= 2
                    && ahat_scale.scalar_type() == torch::kFloat32,
                    where, ": int16 a_hat_cache requires a 2-element fp32 ahat_scale [0, qmax]");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<int16_t>());
        ahat_i8 = true;
        ahat_qscale_ptr = ahat_scale.data_ptr<float>();
    } else if (st == torch::kInt8) {
        TORCH_CHECK(ahat_scale.defined() && ahat_scale.numel() >= 2
                    && ahat_scale.scalar_type() == torch::kFloat32,
                    where, ": int8 a_hat_cache requires a 2-element fp32 ahat_scale [scale, qmax]");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<int8_t>());
        ahat_i8 = true;
        ahat_qscale_ptr = ahat_scale.data_ptr<float>();
    } else {
        TORCH_CHECK(st == torch::kFloat16,
                    where, ": a_hat_cache must be fp16, int8, or int16");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
}
