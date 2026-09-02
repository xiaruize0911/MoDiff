// Standalone probe for the a_hat epilogue of gn_apply_delta_quantize_flat_vec2_kernel.
//
// Rebuilding group_norm_silu.cu is ~4 minutes and ncu has no counter permission on
// this box, so this replicates the kernel's loop body (same loads, same GN math,
// same Yq store) with swappable a_hat epilogues and times them against each other.
// It is a diagnostic, not the shipped path -- whatever wins here gets applied to the
// real kernel and re-verified end to end.
//
//   nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 \
//        -I/workspace/MoDiff/csrc ahat_variants.cu -o /tmp/ahat_variants && /tmp/ahat_variants
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

#include "modiff/common/ahat_cache.cuh"

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + __expf(-v)); }

// ---- variant 0: fp16 a_hat (the baseline this has to beat) -----------------
// ---- variant 1: int8 B=32, current shipped epilogue ------------------------
// ---- variant 2: int8 B=32, held scale (no group reduce) -- reduce cost only -
// ---- variant 3: int8 B=32, reduce but no scale store -----------------------
template <int V>
__global__ void probe_vec2(
    const __half* __restrict__ X, __half* __restrict__ cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    float scale_v, int C, int CPG, long sample_stride, long numel, float* qscale)
{
    const float scale = scale_v, inv_scale = 1.0f / scale;
    const long stride = (long)blockDim.x * gridDim.x;
    for (long base = 2 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < numel; base += 2 * stride) {
        const int c0 = (int)(base % C);
        const long n = base / sample_stride;
        const long si = n * (C / CPG) + (c0 / CPG);
        const float mean = mean_in[si], inv_std = inv_std_in[si];
        const float2 v = __half22float2(*reinterpret_cast<const __half2*>(X + base));
        const float2 w = __half22float2(*reinterpret_cast<const __half2*>(gamma + c0));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(beta + c0));
        const float o0 = silu_f((v.x - mean) * inv_std * w.x + b.x);
        const float o1 = silu_f((v.y - mean) * inv_std * w.y + b.y);
        float q0, q1, d0, d1;
        if constexpr (V == 0) {
            const float2 c = __half22float2(*reinterpret_cast<const __half2*>(cache + base));
            d0 = o0 - c.x; d1 = o1 - c.y;
            q0 = fmaxf(-127.f, fminf(127.f, roundf(d0 * scale)));
            q1 = fmaxf(-127.f, fminf(127.f, roundf(d1 * scale)));
            *reinterpret_cast<__half2*>(cache + base) =
                __float22half2_rn(make_float2(c.x + q0 * inv_scale, c.y + q1 * inv_scale));
        } else {
            int8_t* i8 = reinterpret_cast<int8_t*>(cache);
            if constexpr (V == 1) {
                ahat_b32_update2(i8, qscale, base, o0, o1, scale, inv_scale, 127.f,
                                 q0, q1, d0, d1);
            } else {
                const long gi = base >> 5;
                const float s = qscale[gi];
                const unsigned a =
                    (unsigned)*reinterpret_cast<const unsigned short*>(i8 + base) ^ 0x8080u;
                d0 = o0 - ahat_byte_to_f(a, 0x7640u) * s;
                d1 = o1 - ahat_byte_to_f(a, 0x7641u) * s;
                q0 = fmaxf(-127.f, fminf(127.f, roundf(d0 * scale)));
                q1 = fmaxf(-127.f, fminf(127.f, roundf(d1 * scale)));
                const float nc0 = o0 - d0 + q0 * inv_scale;
                const float nc1 = o1 - d1 + q1 * inv_scale;
                float inv;
                if constexpr (V == 2) {
                    inv = __fdividef(1.0f, s);           // held scale: no group reduce
                } else {
                    const float g = fmaxf(ahat_group16_amax(fmaxf(fabsf(nc0), fabsf(nc1))),
                                          1e-12f);
                    inv = __fdividef(127.0f, g);
                }
                *reinterpret_cast<short*>(i8 + base) = (short)__byte_perm(
                    ahat_f_to_byte(nc0 * inv), ahat_f_to_byte(nc1 * inv), 0x4040u);
                if constexpr (V == 2) {
                    if ((threadIdx.x & 15) == 0) qscale[gi] = s;
                }
            }
        }
        reinterpret_cast<short*>(Yq)[base >> 1] =
            (short)__byte_perm(ahat_f_to_byte(q0), ahat_f_to_byte(q1), 0x4040u);
    }
}

// ---- variant 4: int8 B=32, 4 channels per thread (8 lanes == one group) ----
// Two pairs' loads issue before the single group reduce, and the a_hat store is
// one 4-byte STG instead of two 2-byte ones.
__global__ void probe_vec4(
    const __half* __restrict__ X, __half* __restrict__ cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    float scale_v, int C, int CPG, long sample_stride, long numel, float* qscale)
{
    const float scale = scale_v, inv_scale = 1.0f / scale;
    const long stride = (long)blockDim.x * gridDim.x;
    int8_t* i8 = reinterpret_cast<int8_t*>(cache);
    for (long base = 4 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < numel; base += 4 * stride) {
        const int c0 = (int)(base % C);
        const long n = base / sample_stride;
        const long si = n * (C / CPG) + (c0 / CPG);
        const float mean = mean_in[si], inv_std = inv_std_in[si];
        const float2 vr = *reinterpret_cast<const float2*>(X + base);  // 4 halves == 8 B
        const __half2* xh = reinterpret_cast<const __half2*>(&vr);
        const float2 v0 = __half22float2(xh[0]), v1 = __half22float2(xh[1]);
        const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(gamma + c0));
        const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(gamma + c0 + 2));
        const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(beta + c0));
        const float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(beta + c0 + 2));
        float o[4];
        o[0] = silu_f((v0.x - mean) * inv_std * w0.x + b0.x);
        o[1] = silu_f((v0.y - mean) * inv_std * w0.y + b0.y);
        o[2] = silu_f((v1.x - mean) * inv_std * w1.x + b1.x);
        o[3] = silu_f((v1.y - mean) * inv_std * w1.y + b1.y);

        const long gi = base >> 5;
        const float s = qscale[gi];
        const unsigned a = *reinterpret_cast<const unsigned*>(i8 + base) ^ 0x80808080u;
        float q[4], nc[4], amax = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float d = o[k] - ahat_byte_to_f(a, 0x7640u + (unsigned)k) * s;
            q[k] = fmaxf(-127.f, fminf(127.f, roundf(d * scale)));
            nc[k] = o[k] - d + q[k] * inv_scale;
            amax = fmaxf(amax, fabsf(nc[k]));
        }
        const unsigned m = __activemask() & (0xFFu << (threadIdx.x & 24));
        const float g = fmaxf(__uint_as_float(__reduce_max_sync(m, __float_as_uint(amax))),
                              1e-12f);
        const float inv = __fdividef(127.0f, g);
        *reinterpret_cast<unsigned*>(i8 + base) = __byte_perm(
            __byte_perm(ahat_f_to_byte(nc[0] * inv), ahat_f_to_byte(nc[1] * inv), 0x4040u),
            __byte_perm(ahat_f_to_byte(nc[2] * inv), ahat_f_to_byte(nc[3] * inv), 0x4040u),
            0x5410u);
        if ((threadIdx.x & 7) == 0) qscale[gi] = g * (1.0f / 127.0f);
        *reinterpret_cast<unsigned*>(Yq + base) = __byte_perm(
            __byte_perm(ahat_f_to_byte(q[0]), ahat_f_to_byte(q[1]), 0x4040u),
            __byte_perm(ahat_f_to_byte(q[2]), ahat_f_to_byte(q[3]), 0x4040u), 0x5410u);
    }
}

// ---- variant 5: fp16 a_hat, 4 channels per thread -------------------------
// The control for variant 4: if fp16 gains as much from the wider vector then
// vec4 is just vectorization, not something the int8 path specifically needs.
__global__ void probe_vec4_fp16(
    const __half* __restrict__ X, __half* __restrict__ cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    float scale_v, int C, int CPG, long sample_stride, long numel, float* qscale)
{
    const float scale = scale_v, inv_scale = 1.0f / scale;
    const long stride = (long)blockDim.x * gridDim.x;
    for (long base = 4 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < numel; base += 4 * stride) {
        const int c0 = (int)(base % C);
        const long n = base / sample_stride;
        const long si = n * (C / CPG) + (c0 / CPG);
        const float mean = mean_in[si], inv_std = inv_std_in[si];
        const float2 vr = *reinterpret_cast<const float2*>(X + base);  // 4 halves == 8 B
        const __half2* xh = reinterpret_cast<const __half2*>(&vr);
        const float2 v0 = __half22float2(xh[0]), v1 = __half22float2(xh[1]);
        const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(gamma + c0));
        const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(gamma + c0 + 2));
        const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(beta + c0));
        const float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(beta + c0 + 2));
        float o[4];
        o[0] = silu_f((v0.x - mean) * inv_std * w0.x + b0.x);
        o[1] = silu_f((v0.y - mean) * inv_std * w0.y + b0.y);
        o[2] = silu_f((v1.x - mean) * inv_std * w1.x + b1.x);
        o[3] = silu_f((v1.y - mean) * inv_std * w1.y + b1.y);
        const float2 cr = *reinterpret_cast<const float2*>(cache + base);
        const __half2* ch = reinterpret_cast<const __half2*>(&cr);
        const float2 c0v = __half22float2(ch[0]), c1v = __half22float2(ch[1]);
        const float cv[4] = {c0v.x, c0v.y, c1v.x, c1v.y};
        float q[4], nc[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float d = o[k] - cv[k];
            q[k] = fmaxf(-127.f, fminf(127.f, roundf(d * scale)));
            nc[k] = cv[k] + q[k] * inv_scale;
        }
        __half2 out[2] = {__float22half2_rn(make_float2(nc[0], nc[1])),
                          __float22half2_rn(make_float2(nc[2], nc[3]))};
        *reinterpret_cast<float2*>(cache + base) = *reinterpret_cast<const float2*>(out);
        *reinterpret_cast<unsigned*>(Yq + base) = __byte_perm(
            __byte_perm(ahat_f_to_byte(q[0]), ahat_f_to_byte(q[1]), 0x4040u),
            __byte_perm(ahat_f_to_byte(q[2]), ahat_f_to_byte(q[3]), 0x4040u), 0x5410u);
    }
}

// ---- variant 6: int8 B=32, 8 channels per thread (4 lanes == one group) ----
__global__ void probe_vec8(
    const __half* __restrict__ X, __half* __restrict__ cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    float scale_v, int C, int CPG, long sample_stride, long numel, float* qscale)
{
    const float scale = scale_v, inv_scale = 1.0f / scale;
    const long stride = (long)blockDim.x * gridDim.x;
    int8_t* i8 = reinterpret_cast<int8_t*>(cache);
    for (long base = 8 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < numel; base += 8 * stride) {
        const int c0 = (int)(base % C);
        const long n = base / sample_stride;
        const long si = n * (C / CPG) + (c0 / CPG);
        const float mean = mean_in[si], inv_std = inv_std_in[si];
        const float4 xr = *reinterpret_cast<const float4*>(X + base);
        const float4 wr = *reinterpret_cast<const float4*>(gamma + c0);
        const float4 br = *reinterpret_cast<const float4*>(beta + c0);
        const __half2* xh = reinterpret_cast<const __half2*>(&xr);
        const __half2* wh = reinterpret_cast<const __half2*>(&wr);
        const __half2* bh = reinterpret_cast<const __half2*>(&br);
        float o[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 v = __half22float2(xh[k]);
            const float2 w = __half22float2(wh[k]);
            const float2 b = __half22float2(bh[k]);
            o[2 * k] = silu_f((v.x - mean) * inv_std * w.x + b.x);
            o[2 * k + 1] = silu_f((v.y - mean) * inv_std * w.y + b.y);
        }
        const long gi = base >> 5;
        const float s = qscale[gi];
        const uint2 ar = *reinterpret_cast<const uint2*>(i8 + base);
        const unsigned aw[2] = {ar.x ^ 0x80808080u, ar.y ^ 0x80808080u};
        float q[8], nc[8], amax = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float d = o[k] - ahat_byte_to_f(aw[k >> 2], 0x7640u + (unsigned)(k & 3)) * s;
            q[k] = fmaxf(-127.f, fminf(127.f, roundf(d * scale)));
            nc[k] = o[k] - d + q[k] * inv_scale;
            amax = fmaxf(amax, fabsf(nc[k]));
        }
        const unsigned m = __activemask() & (0xFu << (threadIdx.x & 28));
        const float g = fmaxf(__uint_as_float(__reduce_max_sync(m, __float_as_uint(amax))),
                              1e-12f);
        const float inv = __fdividef(127.0f, g);
        uint2 oc, oq;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned* pc = h ? &oc.y : &oc.x;
            unsigned* pq = h ? &oq.y : &oq.x;
            *pc = __byte_perm(
                __byte_perm(ahat_f_to_byte(nc[4 * h] * inv), ahat_f_to_byte(nc[4 * h + 1] * inv), 0x4040u),
                __byte_perm(ahat_f_to_byte(nc[4 * h + 2] * inv), ahat_f_to_byte(nc[4 * h + 3] * inv), 0x4040u),
                0x5410u);
            *pq = __byte_perm(
                __byte_perm(ahat_f_to_byte(q[4 * h]), ahat_f_to_byte(q[4 * h + 1]), 0x4040u),
                __byte_perm(ahat_f_to_byte(q[4 * h + 2]), ahat_f_to_byte(q[4 * h + 3]), 0x4040u),
                0x5410u);
        }
        *reinterpret_cast<uint2*>(i8 + base) = oc;
        if ((threadIdx.x & 3) == 0) qscale[gi] = g * (1.0f / 127.0f);
        *reinterpret_cast<uint2*>(Yq + base) = oq;
    }
}

#define CK(e) do { cudaError_t r = (e); if (r) { printf("cuda %s @%d\n", cudaGetErrorString(r), __LINE__); return 1; } } while (0)

int main() {
    const int N = 128, C = 384, H = 16, W = 16, G = 32;
    const long numel = (long)N * C * H * W;
    const long ss = (long)C * H * W;
    __half *X, *cache;
    int8_t* Yq;
    __half *gamma, *beta;
    float *mean, *istd, *qscale;
    CK(cudaMalloc(&X, numel * 2));
    CK(cudaMalloc(&cache, numel * 2));
    CK(cudaMalloc(&Yq, numel));
    CK(cudaMalloc(&gamma, C * 2));
    CK(cudaMalloc(&beta, C * 2));
    CK(cudaMalloc(&mean, (long)N * G * 4));
    CK(cudaMalloc(&istd, (long)N * G * 4));
    CK(cudaMalloc(&qscale, numel / 32 * 4));
    CK(cudaMemset(X, 0x11, numel * 2));
    CK(cudaMemset(cache, 0x11, numel * 2));
    CK(cudaMemset(gamma, 0x11, C * 2));
    CK(cudaMemset(beta, 0x11, C * 2));
    CK(cudaMemset(mean, 0, (long)N * G * 4));
    CK(cudaMemset(istd, 0x3f, (long)N * G * 4));
    { std::vector<float> h(numel / 32, 0.01f);
      CK(cudaMemcpy(qscale, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }

    const int blk = 256;
    const int grid2 = (int)((numel / 2 + blk - 1) / blk);
    const int grid4 = (int)((numel / 4 + blk - 1) / blk);
    const int grid8 = (int)((numel / 8 + blk - 1) / blk);
    const char* names[] = {"fp16 a_hat", "int8 B=32 (shipped)", "int8 held scale",
                           "int8 reduce, no scale store", "int8 B=32 vec4",
                           "fp16 a_hat vec4 (control)", "int8 B=32 vec8"};
    float base_ms = 0.f;
    for (int v = 0; v < 7; ++v) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto launch = [&]() {
            switch (v) {
            case 0: probe_vec2<0><<<grid2, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            case 1: probe_vec2<1><<<grid2, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            case 2: probe_vec2<2><<<grid2, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            case 3: probe_vec2<3><<<grid2, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            case 4: probe_vec4<<<grid4, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            case 5: probe_vec4_fp16<<<grid4, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            default: probe_vec8<<<grid8, blk>>>(X, cache, Yq, gamma, beta, mean, istd, 8.f, C, C / G, ss, numel, qscale); break;
            }
        };
        for (int i = 0; i < 20; ++i) launch();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 100; ++i) launch();
        CK(cudaEventRecord(b));
        CK(cudaDeviceSynchronize());
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        ms /= 100.f;
        if (v == 0) base_ms = ms;
        printf("  %-30s %7.1f us   %.3fx vs fp16\n", names[v], ms * 1e3, ms / base_ms);
    }
    return 0;
}
