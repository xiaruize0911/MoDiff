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
// int8 <-> float without the conversion pipe. I2F/F2I issue on the XU pipe at an
// eighth of the FMA rate on GA10x and the fused B=32 update does four of them per
// pair; PRMT/FADD are full rate. Both directions use 1.5*2^23, where a float's ulp
// is exactly 1, so the low mantissa byte of the sum IS the two's-complement integer
// and the rounding is round-to-nearest-even, same as cvt.rni.
#define AHAT_F2I_MAGIC 12582912.0f  // 1.5 * 2^23 == 0x4B400000

// Byte `sel` of `packed` as a float. `packed` must already be XORed with 0x8080 so
// the codes are unsigned; the subtraction is exact (both operands sit next to 2^23).
__device__ __forceinline__ float ahat_byte_to_f(unsigned packed, unsigned sel) {
    return __uint_as_float(__byte_perm(packed, 0x4B400000u, sel)) - (AHAT_F2I_MAGIC + 128.0f);
}
__device__ __forceinline__ unsigned ahat_f_to_byte(float v) {
    return __float_as_uint(fminf(127.0f, fmaxf(-127.0f, v)) + AHAT_F2I_MAGIC);
}

// Two int8 channels -> float2, without the conversion pipe: one PRMT + one FADD per channel
// instead of I2F, which issues on the XU pipe at an eighth of the FMA rate on GA10x. This is the
// same decode ahat_b32_read2 has used since 2026-09-01; the generic path never got it, and it is
// the remaining half of the B != 32 gap after the index fix.
__device__ __forceinline__ float2 ahat_load2_i8(const int8_t* p, long i, float s) {
    const unsigned a = (unsigned)*reinterpret_cast<const unsigned short*>(p + i) ^ 0x8080u;
    return make_float2(ahat_byte_to_f(a, 0x7640u) * s, ahat_byte_to_f(a, 0x7641u) * s);
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
// ng>0: blockwise scales live in q as [N,H,W,ng]; do not read q[0]/q[1] as
// [scale, lim] (those slots are two neighboring block scales).
__device__ __forceinline__ void ahat_qparams(bool i8, const float* q,
                                            float& s, float& inv, float& lim,
                                            int ng = 0) {
    if (!i8 || ng > 0) { s = 1.0f; inv = 1.0f; lim = 127.0f; return; }
    s = q[0];
    lim = q[1];
    inv = (s == 0.0f) ? 0.0f : (1.0f / s);
}

__device__ __forceinline__ bool ahat_is_imode(bool i8, float s) {
    return i8 && s == 0.0f;
}

// True when along-C groups are B=32 (qscale[i>>5] == q[nhw, c/32]). Churches always.
__host__ __device__ inline bool ahat_is_b32(int C, int ng) {
    return ng > 0 && C > 0 && (C & 31) == 0 && ng == (C >> 5);
}

// Along-C block scale index, for ANY block size, in one shift.
//   B = C/ng divides C, and i = nhw*C + c with 0 <= c < C, so
//     nhw*ng + c/B = nhw*(C/B) + c/B = (nhw*C + c)/B = i/B
//   exactly -- the [N,H,W,ng] scale index IS i/B. B is a power of two on every path that reaches
// here (ahat_block_shuffle_ok), so it is `i >> log2(B)`. The generic path used to spend three
// 64-bit integer div/mods (i%C, i/C, c/B) computing this, and Ampere has no integer divide unit:
// that alone made every B != 32 run 1.32-1.75x slower than B=32, flat in B (measured).
__device__ __forceinline__ long ahat_block_qindex(long i, int C, int ng) {
    const int B = C / ng;                              // uniform in i, hoisted out of the loop
    if ((B & (B - 1)) == 0) return (long)((unsigned long long)i >> (__ffs(B) - 1));
    return i / (long)B;                                // non-power-of-two B: still one divide
}

// Blockwise int8: q is fp32 NHWC [N,H,W,ng], ng = C/B. Per-tensor: ng==0, use s0 from q[0].
// B=32: q[i>>5] because i = nhw*C + c and C%32==0 ⇒ nhw*(C/32) + c/32.
__device__ __forceinline__ void ahat_resolve(
    bool i8, const float* q, long i, int C, int ng,
    float s0, float inv0, float lim0,
    float& s, float& inv, float& lim)
{
    if (ng > 0 && i8 && q != nullptr && C > 0) {
        s = q[ahat_block_qindex(i, C, ng)];
        inv = __fdividef(1.0f, fmaxf(s, 1e-12f));
        lim = 127.0f;
        return;
    }
    s = s0; inv = inv0; lim = lim0;
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

// Along-C groups whose size is a power of two (B in {2,4,8,16,32}) can resnap
// in the pair-major vec2 kernel:  B/2 consecutive threads own one group and a
// xor-shuffle amax stays inside that group. Host uses this to skip ahat_commit_block.
__host__ __device__ inline bool ahat_block_shuffle_ok(int C, int ng) {
    if (ng <= 0 || C <= 0 || (C % ng) != 0) return false;
    const int B = C / ng;
    return B >= 2 && B <= 64 && (B & (B - 1)) == 0;
}

// Vec2 (2 ch / thread) dynamic along-C resnap. Call only from live lanes; uses
// __activemask so a tail warp with a dead half-warp still reduces correctly.
// B=32: scale index is i>>5 (no C/ng integer div).
__device__ __forceinline__ void ahat_block_resnap2(
    __half* cache, float* qscale, long i, int C, int ng,
    float nc0, float nc1)
{
    if (qscale == nullptr) return;
    const unsigned am = __activemask();
    float gmax = fmaxf(fabsf(nc0), fabsf(nc1));
    const int np = (C / ng) >> 1;                      // threads owning one along-C block
    // gmax >= 0, so IEEE bit order is magnitude order and REDUX.MAX on the raw bits is exact --
    // the same trick ahat_group16_amax uses, generalised to any group width. One instruction
    // instead of five predicated shuffles.
    // REDUX is warp-wide, so it only pays when the whole warp shares ONE mask (np == 32). With
    // 32/16/8 distinct sub-warp masks the hardware serialises it: measured 1.54x SLOWER than a
    // shuffle loop at np == 1 and no gain at all at np == 2. Below a full warp, take the loop --
    // it is log2(np) shuffles (0 to 4), not the 5 predicated ones the old code always issued.
    const unsigned gm = (np >= 32) ? 0xFFFFFFFFu
                                   : (((1u << np) - 1u) << (threadIdx.x & 31u & ~(unsigned)(np - 1)));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (np >= 32) gmax = __uint_as_float(__reduce_max_sync(am, __float_as_uint(gmax)));
    else
#endif
    for (int off = 1; off < np; off <<= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(am & gm, gmax, off));
    const float g = fmaxf(gmax, 1e-12f);
    const float s = g * (1.0f / 127.f);
    ahat_store2_i8(reinterpret_cast<int8_t*>(cache), i, make_float2(nc0, nc1),
                   __fdividef(127.0f, g), 127.f);
    if ((threadIdx.x & (unsigned)(np - 1)) == 0) qscale[ahat_block_qindex(i, C, ng)] = s;
}

// ---------------------------------------------------------------------------
// B=32 fast path. The generic helpers above pay three divisions per pair --
// 1/s in ahat_resolve (dead when the caller resnaps), C/ng in resnap2, and
// 1/s_new -- plus roundf, which is not a single instruction. At 62 GN launches
// per step that was the whole gap to the fp16 a_hat. These do the same math
// with one reciprocal and no integer division.
// ---------------------------------------------------------------------------

// round-to-nearest-even + saturate to +-127. cvt.rni is one instruction where
// roundf (half-away-from-zero) is a sequence; the tie difference is far below
// the quantization noise this path already carries.
__device__ __forceinline__ int ahat_q8(float v) {
    return min(127, max(-127, __float2int_rn(v)));
}

// Same, with the saturation limit as a parameter -- int4 symmetric needs +-7.
__device__ __forceinline__ int ahat_qn(float v, int lim) {
    return min(lim, max(-lim, __float2int_rn(v)));
}



// amax over the 16 lanes that own one B=32 along-C group. IEEE bit order equals
// magnitude order for non-negative floats, so REDUX.MAX on the raw bits is exact.
// Signed 4-bit nibble decoders for the packed a_hat / packed delta-code layouts:
// one byte holds two consecutive channels, EVEN channel in the low nibble.
__device__ __forceinline__ int ahat_nib_lo(int b) { int v = b & 0xF; return v > 7 ? v - 16 : v; }
__device__ __forceinline__ int ahat_nib_hi(int b) { int v = (b >> 4) & 0xF; return v > 7 ? v - 16 : v; }

__device__ __forceinline__ float ahat_group16_amax(float v) {
    const unsigned half_mask = 0xFFFFu << (threadIdx.x & 16);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __uint_as_float(__reduce_max_sync(__activemask() & half_mask,
                                             __float_as_uint(v)));
#else
    const unsigned m = __activemask() & half_mask;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) v = fmaxf(v, __shfl_xor_sync(m, v, off));
    return v;
#endif
}

// amax over a full warp == one B=32 group (1 channel / thread).
__device__ __forceinline__ float ahat_group32_amax(float v) {
    const unsigned m = __activemask();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __uint_as_float(__reduce_max_sync(m, __float_as_uint(v)));
#else
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) v = fmaxf(v, __shfl_xor_sync(m, v, off));
    return v;
#endif
}

// Fused B=32 dequant -> residual quantize -> resnap -> store, for the pair-major
// vec2 kernels. 16 consecutive threads own one group (requires blockDim.x % 16 == 0
// and a 2*blockDim.x*gridDim.x grid stride, which every caller has).
// Call only from live lanes.
__device__ __forceinline__ void ahat_b32_update2(
    int8_t* __restrict__ cache, float* __restrict__ qscale, long i,
    float x0, float x1, float scale, float inv_scale, float code_lim,
    float& q0, float& q1, float& d0, float& d1)
{
    const long gi = i >> 5;
    const float s = qscale[gi];
    const unsigned a = (unsigned)*reinterpret_cast<const unsigned short*>(cache + i) ^ 0x8080u;
    d0 = x0 - ahat_byte_to_f(a, 0x7640u) * s;
    d1 = x1 - ahat_byte_to_f(a, 0x7641u) * s;
    q0 = fmaxf(-code_lim, fminf(code_lim, roundf(d0 * scale)));
    q1 = fmaxf(-code_lim, fminf(code_lim, roundf(d1 * scale)));
    const float nc0 = x0 - d0 + q0 * inv_scale;
    const float nc1 = x1 - d1 + q1 * inv_scale;
    const float g = fmaxf(ahat_group16_amax(fmaxf(fabsf(nc0), fabsf(nc1))), 1e-12f);
    const float inv = __fdividef(127.0f, g);
    *reinterpret_cast<short*>(cache + i) = (short)__byte_perm(
        ahat_f_to_byte(nc0 * inv), ahat_f_to_byte(nc1 * inv), 0x4040u);
    if ((threadIdx.x & 15) == 0) qscale[gi] = g * (1.0f / 127.0f);
}

// amax over the 8 lanes that own one B=32 group at 4 channels / thread.
__device__ __forceinline__ float ahat_group8_amax(float v) {
    const unsigned eighth = 0xFFu << (threadIdx.x & 24);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __uint_as_float(__reduce_max_sync(__activemask() & eighth, __float_as_uint(v)));
#else
    const unsigned m = __activemask() & eighth;
#pragma unroll
    for (int off = 1; off < 8; off <<= 1) v = fmaxf(v, __shfl_xor_sync(m, v, off));
    return v;
#endif
}

// Four codes as one 32-bit word: inner perms interleave {b0,b1} and {b2,b3}, the
// outer takes bytes 0,1 of each.
__device__ __forceinline__ unsigned ahat_pack4(const float (&v)[4], float inv) {
    return __byte_perm(
        __byte_perm(ahat_f_to_byte(v[0] * inv), ahat_f_to_byte(v[1] * inv), 0x4040u),
        __byte_perm(ahat_f_to_byte(v[2] * inv), ahat_f_to_byte(v[3] * inv), 0x4040u),
        0x5410u);
}

// 4 channels / thread: 8 lanes own one B=32 group. Against the vec2 form this halves
// the group reduce, the scale load and the reciprocal per element, and makes the a_hat
// access one 4-byte word instead of two 2-byte ones. Standalone probe at 128x384x16x16:
// vec2 int8 0.871x of the fp16 epilogue, vec4 int8 0.779x. Widening the FP16 epilogue the
// same way makes it *slower* (1.049x), so this is specific to int8 a_hat being half as
// wide per lane, not vectorization in general.
__device__ __forceinline__ void ahat_b32_update4(
    int8_t* __restrict__ cache, float* __restrict__ qscale, long i,
    const float (&x)[4], float scale, float inv_scale, float code_lim,
    float (&q)[4], float (&d)[4])
{
    const long gi = i >> 5;
    const float s = qscale[gi];
    const unsigned a = *reinterpret_cast<const unsigned*>(cache + i) ^ 0x80808080u;
    float nc[4];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        d[k] = x[k] - ahat_byte_to_f(a, 0x7640u + (unsigned)k) * s;
        q[k] = fmaxf(-code_lim, fminf(code_lim, roundf(d[k] * scale)));
        nc[k] = x[k] - d[k] + q[k] * inv_scale;
        amax = fmaxf(amax, fabsf(nc[k]));
    }
    const float g = fmaxf(ahat_group8_amax(amax), 1e-12f);
    *reinterpret_cast<unsigned*>(cache + i) = ahat_pack4(nc, __fdividef(127.0f, g));
    if ((threadIdx.x & 7) == 0) qscale[gi] = g * (1.0f / 127.0f);
}

__device__ __forceinline__ void ahat_b32_read4(
    const int8_t* __restrict__ cache, const float* __restrict__ qscale, long i,
    const float (&x)[4], float scale, float code_lim, float (&q)[4], float (&d)[4])
{
    const float s = qscale[i >> 5];
    const unsigned a = *reinterpret_cast<const unsigned*>(cache + i) ^ 0x80808080u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        d[k] = x[k] - ahat_byte_to_f(a, 0x7640u + (unsigned)k) * s;
        q[k] = fmaxf(-code_lim, fminf(code_lim, roundf(d[k] * scale)));
    }
}

// Read-only B=32 counterpart (skip-K replay steps): no 1/s, no store.
__device__ __forceinline__ void ahat_b32_read2(
    const int8_t* __restrict__ cache, const float* __restrict__ qscale, long i,
    float x0, float x1, float scale, float code_lim,
    float& q0, float& q1, float& d0, float& d1)
{
    const float s = qscale[i >> 5];
    const unsigned a = (unsigned)*reinterpret_cast<const unsigned short*>(cache + i) ^ 0x8080u;
    d0 = x0 - ahat_byte_to_f(a, 0x7640u) * s;
    d1 = x1 - ahat_byte_to_f(a, 0x7641u) * s;
    q0 = fmaxf(-code_lim, fminf(code_lim, roundf(d0 * scale)));
    q1 = fmaxf(-code_lim, fminf(code_lim, roundf(d1 * scale)));
}

// ---------------------------------------------------------------------------
// PACKED INT4 a_hat, B=32. Byte-for-byte twin of ahat_b32_update2/read2 above,
// except a_hat is stored 4 bits per channel: one thread owns 2 consecutive
// channels == exactly ONE byte, so the load and the store are single-byte and
// the 32-channel group is 16 bytes over 16 lanes -- the same warp geometry
// ahat_group16_amax already reduces over. Storage is 0.5 B/elem + 4 B per 32
// channels = 0.625 B/elem, against 1.125 for int8 B=32 and 2.0 for fp16.
//
// The a_hat CODE limit is 7 (4-bit), which is NOT the delta code limit passed
// as `code_lim` -- those are different quantizers and conflating them is how a
// 4-bit path silently stays 8-bit. See the note on `bool a4` in
// gn_apply_delta_quantize_flat_vec2_kernel.
// ---------------------------------------------------------------------------
#define MODIFF_AHAT_I4_LIM 7.0f
__device__ __forceinline__ void ahat_b32_update2_i4(
    int8_t* __restrict__ cache, float* __restrict__ qscale, long i,
    float x0, float x1, float scale, float inv_scale, float code_lim,
    float& q0, float& q1, float& d0, float& d1)
{
    const long gi = i >> 5;
    const float s = qscale[gi];
    const int b = (int)(unsigned char)cache[i >> 1];
    d0 = x0 - (float)ahat_nib_lo(b) * s;
    d1 = x1 - (float)ahat_nib_hi(b) * s;
    q0 = fmaxf(-code_lim, fminf(code_lim, roundf(d0 * scale)));
    q1 = fmaxf(-code_lim, fminf(code_lim, roundf(d1 * scale)));
    const float nc0 = x0 - d0 + q0 * inv_scale;
    const float nc1 = x1 - d1 + q1 * inv_scale;
    const float g = fmaxf(ahat_group16_amax(fmaxf(fabsf(nc0), fabsf(nc1))), 1e-12f);
    const float inv = __fdividef(MODIFF_AHAT_I4_LIM, g);
    const int n0 = (int)fmaxf(-MODIFF_AHAT_I4_LIM,
                              fminf(MODIFF_AHAT_I4_LIM, roundf(nc0 * inv)));
    const int n1 = (int)fmaxf(-MODIFF_AHAT_I4_LIM,
                              fminf(MODIFF_AHAT_I4_LIM, roundf(nc1 * inv)));
    cache[i >> 1] = (int8_t)((n0 & 0xF) | ((n1 & 0xF) << 4));
    if ((threadIdx.x & 15) == 0) qscale[gi] = g * (1.0f / MODIFF_AHAT_I4_LIM);
}

__device__ __forceinline__ void ahat_b32_read2_i4(
    const int8_t* __restrict__ cache, const float* __restrict__ qscale, long i,
    float x0, float x1, float scale, float code_lim,
    float& q0, float& q1, float& d0, float& d1)
{
    const float s = qscale[i >> 5];
    const int b = (int)(unsigned char)cache[i >> 1];
    d0 = x0 - (float)ahat_nib_lo(b) * s;
    d1 = x1 - (float)ahat_nib_hi(b) * s;
    q0 = fmaxf(-code_lim, fminf(code_lim, roundf(d0 * scale)));
    q1 = fmaxf(-code_lim, fminf(code_lim, roundf(d1 * scale)));
}

// Scalar (1 ch / thread) B=32 resnap: one warp == one along-C group.
__device__ __forceinline__ void ahat_block_resnap1_b32(
    __half* cache, float* qscale, long i, float nc)
{
    if (qscale == nullptr) return;
    const float g = fmaxf(ahat_group32_amax(fabsf(nc)), 1e-12f);
    reinterpret_cast<int8_t*>(cache)[i] = (signed char)ahat_q8(nc * __fdividef(127.0f, g));
    if ((threadIdx.x & 31) == 0)
        qscale[(unsigned long long)i >> 5] = g * (1.0f / 127.0f);
}

// One-element residual quantize + optional a_hat commit. Returns q as float (fits int8).
// WriteAhat is a compile-time flag so skip-K DCE's the stores instead of predicating them.
template <bool WriteAhat>
__device__ __forceinline__ float ahat_quant_update_w(
    __half* cache, long i, float xval,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    const float* ahat_q = nullptr, int ahat_C = 0, int ahat_ng = 0)
{
    ahat_resolve(ahat_i8, ahat_q, i, ahat_C, ahat_ng, ahat_s, ahat_inv, ahat_lim,
                 ahat_s, ahat_inv, ahat_lim);
    if (ahat_is_imode(ahat_i8, ahat_s) && ahat_ng <= 0) {
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
    bool write_ahat,
    const float* ahat_q = nullptr, int ahat_C = 0, int ahat_ng = 0)
{
    return write_ahat
        ? ahat_quant_update_w<true>(cache, i, xval, scale, inv_scale, code_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim, ahat_q, ahat_C, ahat_ng)
        : ahat_quant_update_w<false>(cache, i, xval, scale, inv_scale, code_lim,
                                     ahat_i8, ahat_s, ahat_inv, ahat_lim, ahat_q, ahat_C, ahat_ng);
}

// Pair version. Also writes the pre-clamp float residual into d0/d1 (for absmax reporting).
template <bool WriteAhat>
__device__ __forceinline__ void ahat_quant_update2_w(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    float& q0, float& q1, float& d0, float& d1,
    const float* ahat_q = nullptr, int ahat_C = 0, int ahat_ng = 0)
{
    ahat_resolve(ahat_i8, ahat_q, i, ahat_C, ahat_ng, ahat_s, ahat_inv, ahat_lim,
                 ahat_s, ahat_inv, ahat_lim);
    if (ahat_is_imode(ahat_i8, ahat_s) && ahat_ng <= 0) {
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
    bool write_ahat, float& q0, float& q1, float& d0, float& d1,
    const float* ahat_q = nullptr, int ahat_C = 0, int ahat_ng = 0)
{
    if (write_ahat)
        ahat_quant_update2_w<true>(cache, i, x0, x1, scale, inv_scale, code_lim,
                                   ahat_i8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1,
                                   ahat_q, ahat_C, ahat_ng);
    else
        ahat_quant_update2_w<false>(cache, i, x0, x1, scale, inv_scale, code_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1,
                                    ahat_q, ahat_C, ahat_ng);
}

template <bool WriteAhat>
__device__ __forceinline__ void ahat_quant_update2_w(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    float& q0, float& q1,
    const float* ahat_q = nullptr, int ahat_C = 0, int ahat_ng = 0)
{
    float d0, d1;
    ahat_quant_update2_w<WriteAhat>(cache, i, x0, x1, scale, inv_scale, code_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1,
                                    ahat_q, ahat_C, ahat_ng);
}

__device__ __forceinline__ void ahat_quant_update2(
    __half* cache, long i, float x0, float x1,
    float scale, float inv_scale, float code_lim,
    bool ahat_i8, float ahat_s, float ahat_inv, float ahat_lim,
    bool write_ahat, float& q0, float& q1,
    const float* ahat_q = nullptr, int ahat_C = 0, int ahat_ng = 0)
{
    float d0, d1;
    ahat_quant_update2(cache, i, x0, x1, scale, inv_scale, code_lim,
                       ahat_i8, ahat_s, ahat_inv, ahat_lim, write_ahat, q0, q1, d0, d1,
                       ahat_q, ahat_C, ahat_ng);
}

// Host: fp16, int8 (held or I-MoDiff), or int16 (I-MoDiff). The __half* is a type pun.
// I-MoDiff: ahat_scale = [0, qmax]. Held int8: ahat_scale = [dequant_scale, qmax] with scale>0.
static void bind_ahat_cache(const torch::Tensor& a_hat_cache, const torch::Tensor& ahat_scale,
                            __half*& cache_ptr, bool& ahat_i8, const float*& ahat_qscale_ptr,
                            const char* where, int* ahat_ng_out = nullptr) {
    ahat_i8 = false;
    ahat_qscale_ptr = nullptr;
    if (ahat_ng_out) *ahat_ng_out = 0;
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
                    where, ": int8 a_hat_cache requires fp32 ahat_scale");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<int8_t>());
        ahat_i8 = true;
        ahat_qscale_ptr = ahat_scale.data_ptr<float>();
        if (ahat_scale.dim() == 4) {
            const int C = (int)a_hat_cache.size(1);
            const int ng = (int)ahat_scale.size(3);
            TORCH_CHECK(ahat_scale.is_contiguous(), where, ": block ahat_scale must be contiguous");
            TORCH_CHECK(ahat_scale.size(0) == a_hat_cache.size(0)
                        && ahat_scale.size(1) == a_hat_cache.size(2)
                        && ahat_scale.size(2) == a_hat_cache.size(3),
                        where, ": block ahat_scale must be [N,H,W,C/B]");
            TORCH_CHECK(ng > 0 && C % ng == 0, where, ": C must divide ahat block groups");
            const int B = C / ng;
            TORCH_CHECK(B >= 2 && B <= 64 && (B % 2) == 0,
                        where, ": along-C block size C/ng must be even and in [2,64]");
            if (ahat_ng_out) *ahat_ng_out = ng;
        }
    } else {
        TORCH_CHECK(st == torch::kFloat16,
                    where, ": a_hat_cache must be fp16, int8, or int16");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
}

// Fold delta codes into int8 a_hat with a FRESH along-C amax (same math as the
// Python fake-quant snap). Used after a quantize kernel that left a_hat unchanged.
// cache/yq: NHWC int8, numel = N*C*H*W. qscale: [N,H,W,ng] contiguous.
// B=32: group gi owns cache[gi*32 .. gi*32+32) — no C/ng addressing.
template <int Dummy = 0>
__global__ void ahat_commit_block_kernel_b32(
    int8_t* __restrict__ cache, float* __restrict__ qscale,
    const int8_t* __restrict__ yq, const float* __restrict__ delta_scale_ptr,
    long ngrp)
{
    const float inv_delta = 1.0f / *delta_scale_ptr;
    for (long gi = (long)blockIdx.x * blockDim.x + threadIdx.x; gi < ngrp;
         gi += (long)blockDim.x * gridDim.x) {
        const long base = gi << 5;
        const float old_s = qscale[gi];
        float amax = 0.f;
        float nc[32];
#pragma unroll
        for (int k = 0; k < 32; k += 4) {
            const char4 a = *reinterpret_cast<const char4*>(cache + base + k);
            const char4 y = *reinterpret_cast<const char4*>(yq + base + k);
            float v0 = (float)a.x * old_s + (float)y.x * inv_delta;
            float v1 = (float)a.y * old_s + (float)y.y * inv_delta;
            float v2 = (float)a.z * old_s + (float)y.z * inv_delta;
            float v3 = (float)a.w * old_s + (float)y.w * inv_delta;
            nc[k] = v0; nc[k + 1] = v1; nc[k + 2] = v2; nc[k + 3] = v3;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
        }
        const float s = fmaxf(amax, 1e-12f) / 127.f;
        qscale[gi] = s;
        const float inv = 1.0f / s;
#pragma unroll
        for (int k = 0; k < 32; k += 4) {
            char4 o;
            o.x = (signed char)(int)fmaxf(-127.f, fminf(127.f, roundf(nc[k] * inv)));
            o.y = (signed char)(int)fmaxf(-127.f, fminf(127.f, roundf(nc[k + 1] * inv)));
            o.z = (signed char)(int)fmaxf(-127.f, fminf(127.f, roundf(nc[k + 2] * inv)));
            o.w = (signed char)(int)fmaxf(-127.f, fminf(127.f, roundf(nc[k + 3] * inv)));
            *reinterpret_cast<char4*>(cache + base + k) = o;
        }
    }
}

template <int Dummy = 0>
__global__ void ahat_commit_block_kernel(
    int8_t* __restrict__ cache, float* __restrict__ qscale,
    const int8_t* __restrict__ yq, const float* __restrict__ delta_scale_ptr,
    int C, int ng, long ngrp)
{
    const float inv_delta = 1.0f / *delta_scale_ptr;
    const int B = C / ng;
    for (long gi = (long)blockIdx.x * blockDim.x + threadIdx.x; gi < ngrp;
         gi += (long)blockDim.x * gridDim.x) {
        const float old_s = qscale[gi];
        const long base = (gi / (long)ng) * (long)C + (gi % (long)ng) * (long)B;
        float amax = 0.f;
        float nc[32];
        for (int k = 0; k < B; ++k) {
            float v = (float)cache[base + k] * old_s + (float)yq[base + k] * inv_delta;
            nc[k] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float s = fmaxf(amax, 1e-12f) / 127.f;
        qscale[gi] = s;
        const float inv = 1.0f / s;
        for (int k = 0; k < B; ++k) {
            cache[base + k] = (int8_t)fmaxf(-127.f, fminf(127.f, roundf(nc[k] * inv)));
        }
    }
}

// ---------------------------------------------------------------------------
// PACKED INT4 twin of the two kernels above. `yqp` holds two channels per byte,
// EVEN channel in the low nibble -- the layout every int4 producer here writes
// ((i0 & 0x0F) | ((i1 & 0x0F) << 4)), signed 4-bit so >7 means negative.
//
// This exists so group_norm_silu_delta_quantize_resize_nhwc can accept blockwise
// a_hat with pack=true. It previously TORCH_CHECK'd that combination out, which
// forced _prequant_gn_resize_conv_modiff to decline the fusion for the 8 updown
// ResBlocks -- measured at 2.22 ms/step on W4A4.
//
// Addressing: a_hat element i = nhw*C + c; the packed byte is nhw*(Kpad/2) + c/2.
// Kpad == C unless the caller asked for GEMM-alignment padding, hence the explicit
// argument rather than i>>1.
// ---------------------------------------------------------------------------
template <int Dummy = 0>
__global__ void ahat_commit_block_pack4_kernel_b32(
    int8_t* __restrict__ cache, float* __restrict__ qscale,
    const int8_t* __restrict__ yqp, const float* __restrict__ delta_scale_ptr,
    int C, int Kpad, long ngrp)
{
    const float inv_delta = 1.0f / *delta_scale_ptr;
    const long kp2 = (long)(Kpad >> 1);
    for (long gi = (long)blockIdx.x * blockDim.x + threadIdx.x; gi < ngrp;
         gi += (long)blockDim.x * gridDim.x) {
        const long base = gi << 5;                     // a_hat element base
        // C % 32 == 0 on this path, so a 32-channel group never straddles a pixel row.
        const long nhw = base / (long)C;
        const long pb = nhw * kp2 + ((base - nhw * (long)C) >> 1);
        const float old_s = qscale[gi];
        float amax = 0.f;
        float nc[32];
#pragma unroll
        for (int k = 0; k < 32; k += 2) {
            const int b = (int)(unsigned char)yqp[pb + (k >> 1)];
            const float v0 = (float)cache[base + k]     * old_s + (float)ahat_nib_lo(b) * inv_delta;
            const float v1 = (float)cache[base + k + 1] * old_s + (float)ahat_nib_hi(b) * inv_delta;
            nc[k] = v0; nc[k + 1] = v1;
            amax = fmaxf(amax, fmaxf(fabsf(v0), fabsf(v1)));
        }
        const float s = fmaxf(amax, 1e-12f) / 127.f;
        qscale[gi] = s;
        const float inv = 1.0f / s;
#pragma unroll
        for (int k = 0; k < 32; ++k)
            cache[base + k] = (int8_t)(int)fmaxf(-127.f, fminf(127.f, roundf(nc[k] * inv)));
    }
}

template <int Dummy = 0>
__global__ void ahat_commit_block_pack4_kernel(
    int8_t* __restrict__ cache, float* __restrict__ qscale,
    const int8_t* __restrict__ yqp, const float* __restrict__ delta_scale_ptr,
    int C, int Kpad, int ng, long ngrp)
{
    const float inv_delta = 1.0f / *delta_scale_ptr;
    const int B = C / ng;
    const long kp2 = (long)(Kpad >> 1);
    for (long gi = (long)blockIdx.x * blockDim.x + threadIdx.x; gi < ngrp;
         gi += (long)blockDim.x * gridDim.x) {
        const float old_s = qscale[gi];
        const long nhw = gi / (long)ng;
        const long c0 = (gi % (long)ng) * (long)B;
        const long base = nhw * (long)C + c0;
        const long pb = nhw * kp2 + (c0 >> 1);
        float amax = 0.f;
        float nc[64];
        for (int k = 0; k < B; k += 2) {
            const int b = (int)(unsigned char)yqp[pb + (k >> 1)];
            const float v0 = (float)cache[base + k]     * old_s + (float)ahat_nib_lo(b) * inv_delta;
            const float v1 = (float)cache[base + k + 1] * old_s + (float)ahat_nib_hi(b) * inv_delta;
            nc[k] = v0; nc[k + 1] = v1;
            amax = fmaxf(amax, fmaxf(fabsf(v0), fabsf(v1)));
        }
        const float s = fmaxf(amax, 1e-12f) / 127.f;
        qscale[gi] = s;
        const float inv = 1.0f / s;
        for (int k = 0; k < B; ++k)
            cache[base + k] = (int8_t)(int)fmaxf(-127.f, fminf(127.f, roundf(nc[k] * inv)));
    }
}

static void ahat_commit_block_pack4(
    __half* cache_ptr, float* qscale, const int8_t* yqp,
    const float* delta_scale, int C, int Kpad, int ng, long numel, cudaStream_t stream)
{
    if (cache_ptr == nullptr || qscale == nullptr || yqp == nullptr || ng <= 0) return;
    const long ngrp = (numel / (long)C) * (long)ng;
    const int t = 256;
    int g = (int)((ngrp + (t - 1)) / t);
    if (g < 1) g = 1;
    if (g > 65535) g = 65535;
    auto* i8 = reinterpret_cast<int8_t*>(cache_ptr);
    if (ahat_is_b32(C, ng)) {
        ahat_commit_block_pack4_kernel_b32<0><<<g, t, 0, stream>>>(
            i8, qscale, yqp, delta_scale, C, Kpad, ngrp);
    } else {
        ahat_commit_block_pack4_kernel<0><<<g, t, 0, stream>>>(
            i8, qscale, yqp, delta_scale, C, Kpad, ng, ngrp);
    }
}

static void ahat_commit_block(
    __half* cache_ptr, float* qscale, const int8_t* yq,
    const float* delta_scale, int C, int ng, long numel, cudaStream_t stream)
{
    if (cache_ptr == nullptr || qscale == nullptr || yq == nullptr || ng <= 0) return;
    const long ngrp = (numel / (long)C) * (long)ng;
    const int t = 256;
    int g = (int)((ngrp + (t - 1)) / t);
    if (g < 1) g = 1;
    if (g > 65535) g = 65535;
    auto* i8 = reinterpret_cast<int8_t*>(cache_ptr);
    if (ahat_is_b32(C, ng)) {
        ahat_commit_block_kernel_b32<0><<<g, t, 0, stream>>>(i8, qscale, yq, delta_scale, ngrp);
    } else {
        ahat_commit_block_kernel<0><<<g, t, 0, stream>>>(
            i8, qscale, yq, delta_scale, C, ng, ngrp);
    }
}

