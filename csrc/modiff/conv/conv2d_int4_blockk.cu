// ============================================================================================
// int4 NHWC conv2d fprop with a BLOCKWISE-ALONG-C activation scale, plus a matched scalar-alpha
// control. Port of conv2d_int8_blockk.cu -- read that file's header first. Only three things
// change; everything about the gather, the smem swizzle and the scale staging is identical.
//
//   1. mma.m16n8k64.s4 instead of mma.m16n8k32.s8. One mma reduces 64 K, not 32.
//   2. Operands are PACKED, 2 codes per byte, so a 64-BYTE smem row holds 128 elements. The
//      loaders are byte arithmetic and are therefore unchanged; only the element<->byte
//      conversions in the host and in the A gather differ.
//   3. Overflow is a non-issue. A block accumulates BLK products of int4 codes, peaking at
//      BLK*7*7 = 49*BLK, so bk4_i2f's 2^22 limit is not reached until BLK ~ 85000. Compare int8,
//      where BLK=256 already sits at 1.5% margin.
//
// WHY int4 IS THE POINT. docs/wa_budget_2026-09-02: at 4 bits the activation quantizer IS the
// error budget -- per-tensor 0.5181 vs blockwise B=64 0.0415, a 12.5x reduction, against 8 bits
// where blockwise merely takes an already-small term to the floor. Weights stay
// per-output-channel: blocking them measured 1.29x at 4 bits and 1.0x at 8, not worth moving the
// weight scale off the free axis.
//
// COST WARNING. One mma covers 64 K here against 32 for int8, so at the same BLK the flush rate
// per mma DOUBLES. B=64 int4 has the flush cadence of B=32 int8 (which measured +73%), and B=128
// int4 matches B=64 int8 (+29%). Both are built; pick by measurement.
//
// CONSTRAINTS: C % BK4_CTA_KE == 0 (a 128-element K tile must not straddle two (r,s) taps), which
// holds for C=384 and 768 but NOT 192 or 576 -- those fall back to the CUTLASS int4 conv. Full
// coverage needs a 32-byte-row variant with a 1-bit swizzle. dilation 1, groups 1.
// ============================================================================================
#include <algorithm>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "../common/mma_int8.cuh"

#define BK4_CTA_M 128
#define BK4_CTA_N 128
#define BK4_CTA_K 64       // BYTES per smem row (unchanged from int8)
#define BK4_CTA_KE 128     // ELEMENTS per K tile (2 codes/byte)
#define BK4_NUM_WARPS 8
#define BK4_WARP_N 16
#define BK4_STAGES 2
#define BK4_INTRIN_M 16
#define BK4_INTRIN_N 16
#define BK4_INTRIN_K 64    // ELEMENTS per mma (m16n8k64.s4)
#define BK4_INTRIN_KB 32   // ...which is 32 BYTES; the smem loaders are byte-indexed
#define BK4_PACK 16
#define BK4_MI (BK4_CTA_M / BK4_INTRIN_M)   // 8
#define BK4_NJ (BK4_WARP_N / BK4_INTRIN_N)  // 1

#define BK4_MAGIC_I 0x4B400000
#define BK4_MAGIC_F 12582912.0f
__device__ __forceinline__ float bk4_i2f(int v) {
    return __int_as_float(v + BK4_MAGIC_I) - BK4_MAGIC_F;
}

// cp.async with zero-fill. `src` must stay in-bounds even when !pred (src-size 0 reads nothing,
// but the address is still formed), so callers clamp it.
__device__ __forceinline__ void bk4_cp_async_zfill(uint32_t smem, const void* src, bool pred) {
    const int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem), "l"(src),
                 "n"(16), "r"(src_size));
}

// Per-CTA-row gather state, computed once. `pix0` is the linear NHWC pixel index of the (r=0,s=0)
// tap; h0/w0 are that tap's spatial coords, so a tap (ro,so) is at pix0 + ro*W + so and is valid
// iff h0+ro and w0+so are both in range.
struct BkcRow {
    int pix0, h0, w0;
    bool m_ok;
};
__device__ __forceinline__ BkcRow bk4_row(int m, int M, int H, int W, int P, int Q,
                                          int stride, int pad) {
    BkcRow r;
    r.m_ok = m < M;
    const int mm = r.m_ok ? m : 0;
    const int n = mm / (P * Q), rem = mm - n * P * Q;
    const int p = rem / Q, q = rem - p * Q;
    r.h0 = p * stride - pad;
    r.w0 = q * stride - pad;
    r.pix0 = (n * H + r.h0) * W + r.w0;
    return r;
}

__device__ __forceinline__ void bk4_s2r_A(const int8_t* src, int8_t* dst, int lane, int k01) {
    const int ld_col = (k01 * BK4_INTRIN_KB + (lane / 16) * 16) / BK4_PACK;
#pragma unroll
    for (int si = 0; si < BK4_MI; ++si) {
        const int ld_row = si * BK4_INTRIN_M + (lane % 16);
        const int swz = ld_col ^ ((ld_row / 2) & 3);
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * BK4_CTA_K + swz * BK4_PACK));
    }
}
__device__ __forceinline__ void bk4_s2r_B(const int8_t* src, int8_t* dst, int lane,
                                          int warp_off_n, int k01) {
    const int ld_col = (k01 * BK4_INTRIN_KB + ((lane / 8) % 2) * 16) / BK4_PACK;
#pragma unroll
    for (int si = 0; si < BK4_NJ; ++si) {
        const int ld_row = warp_off_n + si * BK4_INTRIN_N + ((lane / 8 / 2) * 8 + lane % 8);
        const int swz = ld_col ^ ((ld_row / 2) & 3);
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * BK4_CTA_K + swz * BK4_PACK));
    }
}

// a_scale_blk: [N, H, W, C/BLK] fp32, i.e. the same NHWC-with-a-block-axis layout the a_hat cache
// already uses. Empty -> scalar-alpha control.
// ACCUM is the MoDiff arm: o_hat += A(Q(delta)) instead of out = A(Q(a_t)). It is a template
// parameter, not a runtime flag, for the reason docs/ahat_blockwise_2026-09-01 found the hard way
// -- a runtime branch leaves the unused datapath in the instance and the dead code costs registers,
// which costs occupancy, which was the whole reason B=32 a_hat started out 4.3% SLOWER than fp16.
template <int BLK, bool BLOCKWISE, bool ACCUM, bool RESID>
// min-2-blocks-per-SM in the launch bounds is load-bearing, not a hint. At BK4_STAGES 3 the
// B=32 instance took 130 regs, which sm_86 rounds up to 136 -> 34816 regs per 8-warp CTA ->
// 1 CTA/SM, while the scalar control at 125 (-> 128) got 2. Half the occupancy for 2
// registers, and B=32 also crossed the smem half at 51.0 KiB. STAGES 2 fixed the smem half;
// this fixes the register half by making the compiler budget for 2 CTAs.
// min-blocks is 2 only for BLK <= CTA_K. For BLK > CTA_K the int32 accumulator lives across
// tile boundaries, and forcing 128 regs then spills 160 B and costs 3.5x -- there, take 1
// CTA/SM and the registers instead.
static __global__ __launch_bounds__(BK4_NUM_WARPS * 32, (BLK <= BK4_CTA_K ? 2 : 1))
void conv2d_int4_blockk_kernel(
    const int8_t* __restrict__ X, const int8_t* __restrict__ Wt,
    const float* __restrict__ w_scale, const float* __restrict__ a_scale_blk, float a_scale,
    __half* __restrict__ Out, const __half* __restrict__ bias,
    const __half* __restrict__ resid,
    int Nn, int H, int W, int C, int Kout, int R, int S, int P, int Q,
    int stride, int pad, int M, int Kg)
{
    // BLK <= CTA_K: NB blocks fit inside one CTA-K tile, flush NB times per tile.
    // BLK >  CTA_K: one block SPANS TPB tiles, so the int32 accumulator is carried across
    // tiles and flushed once every TPB. That is the whole point of a larger B -- the flush
    // is what costs (measured: 0 flushes/tile 25.77 ms, 1 -> 33.21, 2 -> 44.56), so halving
    // the flush rate is the direct lever on it.
    constexpr int NB  = (BK4_CTA_KE / BLK) > 0 ? (BK4_CTA_KE / BLK) : 1;
    constexpr int TPB = (BLK / BK4_CTA_KE) > 0 ? (BLK / BK4_CTA_KE) : 1;
    extern __shared__ char bk4_smem[];
    int8_t* As = reinterpret_cast<int8_t*>(bk4_smem);
    int8_t* Bs = As + BK4_STAGES * BK4_CTA_M * BK4_CTA_K;
    float* Ss = reinterpret_cast<float*>(Bs + BK4_STAGES * BK4_CTA_N * BK4_CTA_K);

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.y * BK4_CTA_M, n0 = blockIdx.x * BK4_CTA_N;
    const int warp_off_n = warp * BK4_WARP_N;
    const int nb_c = C / BLK;                        // scale blocks per pixel
    const size_t pix_max = (size_t)Nn * H * W;

    // Hoisted gather state. The A loader visits CTA rows t/4 and t/4+64 (256 threads, 4 chunks of
    // 16 B per row); the scale stager owns CTA row t%128. Two divisions each, once.
    const BkcRow ra0 = bk4_row(m0 + (t >> 2), M, H, W, P, Q, stride, pad);
    const BkcRow ra1 = bk4_row(m0 + (t >> 2) + 64, M, H, W, P, Q, stride, pad);
    const BkcRow rs0 = bk4_row(m0 + (t & 127), M, H, W, P, Q, stride, pad);

    int acc[BK4_MI][BK4_NJ][8];
    float accf[BK4_MI][BK4_NJ][8];
#pragma unroll
    for (int i = 0; i < BK4_MI; ++i)
#pragma unroll
        for (int j = 0; j < BK4_NJ; ++j)
#pragma unroll
            for (int k = 0; k < 8; ++k) { acc[i][j][k] = 0; accf[i][j][k] = 0.0f; }

    const int nkt = Kg / BK4_CTA_KE;

    // Stage `kt` (a K_gemm offset) of A, B and the block scales into buffer `buf`.
    auto load_tile = [&](int kt, int buf) {
        const int rs = kt / C, c0 = kt - rs * C;
        const int ro = rs / S, so = rs - ro * S;
        // ---- A: gathered, zero-filled outside the input ----
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const BkcRow& rr = half ? ra1 : ra0;
            const int r = (t >> 2) + half * 64;
            const int off16 = t & 3;
            const int h = rr.h0 + ro, w = rr.w0 + so;
            const bool ok = rr.m_ok && (unsigned)h < (unsigned)H && (unsigned)w < (unsigned)W;
            const size_t pix = ok ? (size_t)(rr.pix0 + ro * W + so) : 0;
            const int8_t* src = X + pix * (C >> 1) + (c0 >> 1) + off16 * 16;   // packed bytes
            const int swz = (off16 ^ ((r / 2) & 3)) * 16;
            bk4_cp_async_zfill(modiff_smem_ptr(&As[buf * BK4_CTA_M * BK4_CTA_K + r * BK4_CTA_K + swz]),
                               src, ok);
        }
        // ---- B: weight rows are contiguous over (r,s,c), same as the GEMM ----
        for (int c = t; c < BK4_CTA_N * (BK4_CTA_K / 16); c += BK4_NUM_WARPS * 32) {
            const int r = c / (BK4_CTA_K / 16), off16 = c % (BK4_CTA_K / 16);
            const int swz = (off16 ^ ((r / 2) & 3)) * 16;
            modiff_cp_async_cg(
                modiff_smem_ptr(&Bs[buf * BK4_CTA_N * BK4_CTA_K + r * BK4_CTA_K + swz]),
                (const uint4*)(Wt + (size_t)(n0 + r) * (Kg >> 1) + (kt >> 1) + off16 * 16), (n0 + r) < Kout);
        }
        // ---- block scales for this tile's NB blocks, one CTA row per thread ----
        if (BLOCKWISE) {
            const int r = t & 127;
            const int h = rs0.h0 + ro, w = rs0.w0 + so;
            const bool ok = rs0.m_ok && (unsigned)h < (unsigned)H && (unsigned)w < (unsigned)W;
            const size_t pix = ok ? (size_t)(rs0.pix0 + ro * W + so) : 0;
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                if (NB == 1 && t >= 128) break;
                if (NB == 2 && (t >> 7) != b) continue;
                Ss[(buf * NB + b) * BK4_CTA_M + r] =
                    (ok && pix < pix_max) ? a_scale_blk[pix * nb_c + (c0 / BLK) + b] : 0.0f;
            }
        }
    };

#pragma unroll
    for (int s = 0; s < BK4_STAGES - 1; ++s) {
        if (s < nkt) load_tile(s * BK4_CTA_KE, s);
        __pipeline_commit();
    }
    __pipeline_wait_prior(BK4_STAGES - 2);
    __syncthreads();

    for (int i = 0; i < nkt; ++i) {
        const int buf = i % BK4_STAGES;
        const int8_t* Ab = &As[buf * BK4_CTA_M * BK4_CTA_K];
        const int8_t* Bb = &Bs[buf * BK4_CTA_N * BK4_CTA_K];
#pragma unroll
        for (int k01 = 0; k01 < BK4_CTA_KE / BK4_INTRIN_K; ++k01) {
            int8_t Afrag[BK4_MI * 16], Bfrag[BK4_NJ * 16];
            bk4_s2r_A(Ab, Afrag, lane, k01);
            bk4_s2r_B(Bb, Bfrag, lane, warp_off_n, k01);
#pragma unroll
            for (int ii = 0; ii < BK4_MI; ++ii)
#pragma unroll
                for (int jj = 0; jj < BK4_NJ; ++jj) {
                    modiff_mma_m16n8k64_s4(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
                    modiff_mma_m16n8k64_s4(acc[ii][jj] + 4, Afrag + ii * 16, Bfrag + jj * 16 + 8);
                }
            // Kg % BLK == 0 (host-checked via C % BLK == 0), so blocks never straddle the
            // end of the k-loop and no trailing partial-block flush is needed.
            if (BLOCKWISE && (BLK == BK4_INTRIN_K || k01 == 1)
                && (TPB == 1 || (i % TPB) == TPB - 1)) {
                const float* sb = &Ss[(buf * NB + (NB == 2 ? k01 : 0)) * BK4_CTA_M];
#pragma unroll
                for (int ii = 0; ii < BK4_MI; ++ii) {
                    const int lr0 = ii * BK4_INTRIN_M + gid;   // CTA-local rows of this fragment
                    const float s0 = sb[lr0], s1 = sb[lr0 + 8];
#pragma unroll
                    for (int jj = 0; jj < BK4_NJ; ++jj) {
                        int* a = acc[ii][jj];
                        float* f = accf[ii][jj];
                        f[0] = fmaf(bk4_i2f(a[0]), s0, f[0]);
                        f[1] = fmaf(bk4_i2f(a[1]), s0, f[1]);
                        f[2] = fmaf(bk4_i2f(a[2]), s1, f[2]);
                        f[3] = fmaf(bk4_i2f(a[3]), s1, f[3]);
                        f[4] = fmaf(bk4_i2f(a[4]), s0, f[4]);
                        f[5] = fmaf(bk4_i2f(a[5]), s0, f[5]);
                        f[6] = fmaf(bk4_i2f(a[6]), s1, f[6]);
                        f[7] = fmaf(bk4_i2f(a[7]), s1, f[7]);
#pragma unroll
                        for (int k = 0; k < 8; ++k) a[k] = 0;
                    }
                }
            }
        }
        const int li = i + BK4_STAGES - 1;
        if (li < nkt) load_tile(li * BK4_CTA_KE, li % BK4_STAGES);
        __pipeline_commit();
        __pipeline_wait_prior(BK4_STAGES - 2);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < BK4_MI; ++i) {
        const int row0 = m0 + i * BK4_INTRIN_M + gid, row1 = row0 + 8;
#pragma unroll
        for (int j = 0; j < BK4_NJ; ++j) {
            const int col0 = n0 + warp_off_n + j * BK4_INTRIN_N + tig * 2, col1 = col0 + 8;
            const float as = BLOCKWISE ? 1.0f : a_scale;
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = BLOCKWISE ? accf[i][j][k] : (float)acc[i][j][k];
            const bool c0ok = col0 + 1 < Kout, c1ok = col1 + 1 < Kout;
            const float s00 = c0ok ? as * w_scale[col0] : 0.f, s01 = c0ok ? as * w_scale[col0 + 1] : 0.f;
            const float s10 = c1ok ? as * w_scale[col1] : 0.f, s11 = c1ok ? as * w_scale[col1 + 1] : 0.f;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = half ? row1 : row0;
                if (row >= M) continue;
                const int i0 = half ? 2 : 0, i1 = half ? 6 : 4;
                if (c0ok) {
                    float a0 = v[i0] * s00, a1 = v[i0 + 1] * s01;
                    if (bias) { a0 += __half2float(bias[col0]); a1 += __half2float(bias[col0 + 1]); }
                    __half2* dst = (__half2*)&Out[(size_t)row * Kout + col0];
                    if (ACCUM) {
                        const __half2 prev = *dst;
                        a0 += __half2float(__low2half(prev));
                        a1 += __half2float(__high2half(prev));
                    }
                    if (RESID) {
                        const __half2 rv =
                            *(const __half2*)&resid[(size_t)row * Kout + col0];
                        a0 += __half2float(__low2half(rv));
                        a1 += __half2float(__high2half(rv));
                    }
                    *dst = __halves2half2(__float2half(a0), __float2half(a1));
                }
                if (c1ok) {
                    float a0 = v[i1] * s10, a1 = v[i1 + 1] * s11;
                    if (bias) { a0 += __half2float(bias[col1]); a1 += __half2float(bias[col1 + 1]); }
                    __half2* dst = (__half2*)&Out[(size_t)row * Kout + col1];
                    if (ACCUM) {
                        const __half2 prev = *dst;
                        a0 += __half2float(__low2half(prev));
                        a1 += __half2float(__high2half(prev));
                    }
                    if (RESID) {
                        const __half2 rv =
                            *(const __half2*)&resid[(size_t)row * Kout + col1];
                        a0 += __half2float(__low2half(rv));
                        a1 += __half2float(__high2half(rv));
                    }
                    *dst = __halves2half2(__float2half(a0), __float2half(a1));
                }
            }
        }
    }
}

torch::Tensor conv2d_int4_blockk(torch::Tensor x, torch::Tensor weight, torch::Tensor w_scale,
                                 torch::Tensor a_scale_blk, double a_scale, int64_t blk,
                                 int64_t stride, int64_t pad,
                                 c10::optional<torch::Tensor> bias_opt,
                                 c10::optional<torch::Tensor> o_hat_opt,
                                 c10::optional<torch::Tensor> resid_opt)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kInt8 && weight.dtype() == torch::kInt8,
                "packed int4 operands are carried in int8 storage");
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4,
                "x packed [N,H,W,C/2] contiguous, w packed [K,R,S,C/2] -- same convention as "
                "conv2d_int4_fprop");
    const int Nn = x.size(0), H = x.size(1), W = x.size(2);
    const int Kout = weight.size(0), R = weight.size(1), S = weight.size(2);
    const int C = weight.size(3) * 2;
    TORCH_CHECK(x.size(3) == C / 2, "x/weight packed channel mismatch");
    TORCH_CHECK(C % BK4_CTA_KE == 0, "C must be a multiple of ", BK4_CTA_KE,
                " (got ", C, "); C=192/576 need the 32-byte-row variant");
    TORCH_CHECK(C % 2 == 0 && Kout % 2 == 0, "packed int4 needs even C and Kout");
    TORCH_CHECK(Kout % 2 == 0, "Kout must be even (the epilogue stores column pairs)");
    // 256 is the ceiling: a block accumulates BLK int8 products, peaking at BLK*127*127,
    // and bk4_i2f (add into the mantissa of 1.5*2^23) is exact only below 2^22 = 4194304.
    // BLK=256 -> 4129024, a 1.5% margin. BLK=512 would overflow it silently.
    TORCH_CHECK(blk == 64 || blk == 128 || blk == 256,
                "int4 blk must be 64, 128 or 256 -- one mma reduces 64 K, so a smaller block cannot "
                "be read out of a single mma result (got ", blk, ")");
    TORCH_CHECK(C % blk == 0, "blk must divide C");
    TORCH_CHECK(x.is_contiguous(), "packed x must be contiguous [N,H,W,C/2]");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous [K,R,S,C]");

    const int P = (H + 2 * pad - R) / stride + 1, Q = (W + 2 * pad - S) / stride + 1;
    const int M = Nn * P * Q, Kg = R * S * C;

    const bool blockwise = a_scale_blk.numel() > 0;
    if (blockwise) {
        TORCH_CHECK(a_scale_blk.dtype() == torch::kFloat32, "a_scale_blk must be fp32");
        TORCH_CHECK(a_scale_blk.numel() == (int64_t)Nn * H * W * (C / blk),
                    "a_scale_blk must be [N,H,W,C/blk]");
    }
    // Fold the ResBlock skip-add into the store epilogue, which is what the shipped EVT conv
    // does. Doing it eagerly cost ~9 ms/step of elementwise. READ-ONLY, unlike o_hat_opt, so
    // the caller's skip tensor is not mutated.
    const __half* resid_p = nullptr;
    if (resid_opt.has_value() && resid_opt->numel() > 0) {
        TORCH_CHECK(resid_opt->dtype() == torch::kFloat16, "residual must be fp16");
        TORCH_CHECK(resid_opt->is_contiguous(at::MemoryFormat::ChannelsLast),
                    "residual must be channels_last");
        resid_p = reinterpret_cast<const __half*>(resid_opt->data_ptr<at::Half>());
    }
    const __half* bias_p = nullptr;
    if (bias_opt.has_value() && bias_opt->numel() > 0)
        bias_p = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());

    // MoDiff arm: accumulate into the caller's o_hat in place. Baseline arm: fresh output.
    const bool accum = o_hat_opt.has_value() && o_hat_opt->numel() > 0;
    torch::Tensor out;
    if (accum) {
        out = *o_hat_opt;
        TORCH_CHECK(out.dtype() == torch::kFloat16, "o_hat must be fp16");
        TORCH_CHECK(out.is_contiguous(at::MemoryFormat::ChannelsLast),
                    "o_hat must be channels_last");
        TORCH_CHECK(out.size(0) == Nn && out.size(1) == Kout && out.size(2) == P
                        && out.size(3) == Q,
                    "o_hat must be [N,Kout,P,Q]");
    } else {
        // The memory format MUST be in the allocation. `torch::empty(...)` gives an
        // NCHW-contiguous tensor and a trailing .contiguous(ChannelsLast) then COPIES it --
        // a full-size copy of uninitialised data on every conv call, measured at +6.7 ms/step
        // of aten::copy_ across the UNet.
        out = torch::empty({Nn, Kout, P, Q},
                           x.options().dtype(torch::kFloat16)
                               .memory_format(at::MemoryFormat::ChannelsLast));
    }

    const int nb = std::max<int>(1, BK4_CTA_KE / (int)blk);
    const size_t smem = (size_t)BK4_STAGES * (BK4_CTA_M + BK4_CTA_N) * BK4_CTA_K
                        + (blockwise ? (size_t)BK4_STAGES * nb * BK4_CTA_M * sizeof(float) : 0);
    dim3 grid((Kout + BK4_CTA_N - 1) / BK4_CTA_N, (M + BK4_CTA_M - 1) / BK4_CTA_M);
    dim3 block(BK4_NUM_WARPS * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

#define BK4_LAUNCH(BLKV, BW, AC, RS)                                                                \
    do {                                                                                        \
        auto kern = conv2d_int4_blockk_kernel<BLKV, BW, AC, RS>;                                    \
        C10_CUDA_CHECK(cudaFuncSetAttribute(                                                    \
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));                     \
        kern<<<grid, block, smem, stream>>>(                                                    \
            x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), w_scale.data_ptr<float>(),         \
            blockwise ? a_scale_blk.data_ptr<float>() : nullptr, (float)a_scale,                \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), bias_p, resid_p,                        \
            Nn, H, W, C, Kout, R, S, P, Q, (int)stride, (int)pad, M, Kg);                       \
    } while (0)
#define BK4_DISPATCH_B(BW, AC, RS)                                                            \
    do {                                                                                      \
        switch (blk) {                                                                        \
            case 32:  BK4_LAUNCH(32,  BW, AC, RS); break;                                     \
            case 64:  BK4_LAUNCH(64,  BW, AC, RS); break;                                     \
            case 128: BK4_LAUNCH(128, BW, AC, RS); break;                                     \
            default:  BK4_LAUNCH(256, BW, AC, RS); break;                                     \
        }                                                                                     \
    } while (0)
#define BK4_DISPATCH(BW, AC)                                                                  \
    do {                                                                                      \
        if (resid_p) { BK4_DISPATCH_B(BW, AC, true); }                                        \
        else         { BK4_DISPATCH_B(BW, AC, false); }                                       \
    } while (0)
    if (accum) {
        if (blockwise) { BK4_DISPATCH(true, true); } else { BK4_DISPATCH(false, true); }
    } else {
        if (blockwise) { BK4_DISPATCH(true, false); } else { BK4_DISPATCH(false, false); }
    }
#undef BK4_DISPATCH
#undef BK4_DISPATCH_B
#undef BK4_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
