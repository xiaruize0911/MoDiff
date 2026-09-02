// ============================================================================================
// int8 NHWC conv2d fprop with a BLOCKWISE-ALONG-C activation scale, plus a matched scalar-alpha
// control. The conv twin of csrc/modiff/linear/gemm_blockk.cu -- read that file's header first;
// the mainloop arithmetic, the register argument and the I2F-avoiding conversion are identical.
//
// What the conv adds over the GEMM
// --------------------------------
// In implicit-GEMM terms M = N*P*Q, N_gemm = K_out, K_gemm = R*S*C. The weight is already
// [K_out, R, S, C] contiguous, so B loads exactly as in the GEMM. Two things change:
//
//   1. A is gathered. Row m = (n,p,q), column (r,s,c) reads x[n, p*stride+r-pad, q*stride+s-pad, c].
//      Out-of-bounds taps must contribute ZERO, so the A loader uses cp.async's src-size form
//      (src-size 0 zero-fills the destination) rather than the predicated form the GEMM uses --
//      there a masked row is simply never read, here a masked tap is part of the sum.
//
//   2. The block scale is indexed by the INPUT pixel, not by m. This is the whole reason the
//      scale cannot live in the epilogue: for R,S>1 a single output pixel reads R*S different
//      input pixels, so the scale depends on the reduction index (r,s) as well as on m. Nothing
//      finer than per-tensor is an epilogue broadcast for a 3x3 conv.
//
// Recomputing that gather per flush costs ~7 integer ops + 1 LDG per accumulator row, on top of
// the 3 ops per accumulator the flush already does -- about +60%. Instead the CTA's BK_CTA_M row
// scales are staged into smem on the SAME pipeline stage as the A/B cp.async, so the existing
// per-tile __syncthreads covers them and the flush just does LDS. The (n,p,q) decomposition each
// thread needs is hoisted out of the k-loop: a thread always stages the same CTA row, so the two
// integer divisions happen once per kernel instead of once per tile.
//
// CONSTRAINTS: C % BK_CTA_K == 0 (so a 64-wide K_gemm tile never straddles two (r,s) taps),
// dilation 1, groups 1. blk in {32, 64}.
// ============================================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "../common/mma_int8.cuh"

#define BKC_CTA_M 128
#define BKC_CTA_N 128
#define BKC_CTA_K 64
#define BKC_NUM_WARPS 8
#define BKC_WARP_N 16
#define BKC_STAGES 3
#define BKC_INTRIN_M 16
#define BKC_INTRIN_N 16
#define BKC_INTRIN_K 32
#define BKC_PACK 16
#define BKC_MI (BKC_CTA_M / BKC_INTRIN_M)   // 8
#define BKC_NJ (BKC_WARP_N / BKC_INTRIN_N)  // 1

#define BKC_MAGIC_I 0x4B400000
#define BKC_MAGIC_F 12582912.0f
__device__ __forceinline__ float bkc_i2f(int v) {
    return __int_as_float(v + BKC_MAGIC_I) - BKC_MAGIC_F;
}

// cp.async with zero-fill. `src` must stay in-bounds even when !pred (src-size 0 reads nothing,
// but the address is still formed), so callers clamp it.
__device__ __forceinline__ void bkc_cp_async_zfill(uint32_t smem, const void* src, bool pred) {
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
__device__ __forceinline__ BkcRow bkc_row(int m, int M, int H, int W, int P, int Q,
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

__device__ __forceinline__ void bkc_s2r_A(const int8_t* src, int8_t* dst, int lane, int k01) {
    const int ld_col = (k01 * BKC_INTRIN_K + (lane / 16) * 16) / BKC_PACK;
#pragma unroll
    for (int si = 0; si < BKC_MI; ++si) {
        const int ld_row = si * BKC_INTRIN_M + (lane % 16);
        const int swz = ld_col ^ ((ld_row / 2) & 3);
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * BKC_CTA_K + swz * BKC_PACK));
    }
}
__device__ __forceinline__ void bkc_s2r_B(const int8_t* src, int8_t* dst, int lane,
                                          int warp_off_n, int k01) {
    const int ld_col = (k01 * BKC_INTRIN_K + ((lane / 8) % 2) * 16) / BKC_PACK;
#pragma unroll
    for (int si = 0; si < BKC_NJ; ++si) {
        const int ld_row = warp_off_n + si * BKC_INTRIN_N + ((lane / 8 / 2) * 8 + lane % 8);
        const int swz = ld_col ^ ((ld_row / 2) & 3);
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * BKC_CTA_K + swz * BKC_PACK));
    }
}

// a_scale_blk: [N, H, W, C/BLK] fp32, i.e. the same NHWC-with-a-block-axis layout the a_hat cache
// already uses. Empty -> scalar-alpha control.
template <int BLK, bool BLOCKWISE>
static __global__ __launch_bounds__(BKC_NUM_WARPS * 32) void conv2d_int8_blockk_kernel(
    const int8_t* __restrict__ X, const int8_t* __restrict__ Wt,
    const float* __restrict__ w_scale, const float* __restrict__ a_scale_blk, float a_scale,
    __half* __restrict__ Out, const __half* __restrict__ bias,
    int Nn, int H, int W, int C, int Kout, int R, int S, int P, int Q,
    int stride, int pad, int M, int Kg)
{
    constexpr int NB = BKC_CTA_K / BLK;             // scale blocks per CTA-K tile (2 or 1)
    extern __shared__ char bkc_smem[];
    int8_t* As = reinterpret_cast<int8_t*>(bkc_smem);
    int8_t* Bs = As + BKC_STAGES * BKC_CTA_M * BKC_CTA_K;
    float* Ss = reinterpret_cast<float*>(Bs + BKC_STAGES * BKC_CTA_N * BKC_CTA_K);

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.y * BKC_CTA_M, n0 = blockIdx.x * BKC_CTA_N;
    const int warp_off_n = warp * BKC_WARP_N;
    const int nb_c = C / BLK;                        // scale blocks per pixel
    const size_t pix_max = (size_t)Nn * H * W;

    // Hoisted gather state. The A loader visits CTA rows t/4 and t/4+64 (256 threads, 4 chunks of
    // 16 B per row); the scale stager owns CTA row t%128. Two divisions each, once.
    const BkcRow ra0 = bkc_row(m0 + (t >> 2), M, H, W, P, Q, stride, pad);
    const BkcRow ra1 = bkc_row(m0 + (t >> 2) + 64, M, H, W, P, Q, stride, pad);
    const BkcRow rs0 = bkc_row(m0 + (t & 127), M, H, W, P, Q, stride, pad);

    int acc[BKC_MI][BKC_NJ][8];
    float accf[BKC_MI][BKC_NJ][8];
#pragma unroll
    for (int i = 0; i < BKC_MI; ++i)
#pragma unroll
        for (int j = 0; j < BKC_NJ; ++j)
#pragma unroll
            for (int k = 0; k < 8; ++k) { acc[i][j][k] = 0; accf[i][j][k] = 0.0f; }

    const int nkt = Kg / BKC_CTA_K;

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
            const int8_t* src = X + pix * C + c0 + off16 * 16;
            const int swz = (off16 ^ ((r / 2) & 3)) * 16;
            bkc_cp_async_zfill(modiff_smem_ptr(&As[buf * BKC_CTA_M * BKC_CTA_K + r * BKC_CTA_K + swz]),
                               src, ok);
        }
        // ---- B: weight rows are contiguous over (r,s,c), same as the GEMM ----
        for (int c = t; c < BKC_CTA_N * (BKC_CTA_K / 16); c += BKC_NUM_WARPS * 32) {
            const int r = c / (BKC_CTA_K / 16), off16 = c % (BKC_CTA_K / 16);
            const int swz = (off16 ^ ((r / 2) & 3)) * 16;
            modiff_cp_async_cg(
                modiff_smem_ptr(&Bs[buf * BKC_CTA_N * BKC_CTA_K + r * BKC_CTA_K + swz]),
                (const uint4*)(Wt + (size_t)(n0 + r) * Kg + kt + off16 * 16), (n0 + r) < Kout);
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
                Ss[(buf * NB + b) * BKC_CTA_M + r] =
                    (ok && pix < pix_max) ? a_scale_blk[pix * nb_c + (c0 / BLK) + b] : 0.0f;
            }
        }
    };

#pragma unroll
    for (int s = 0; s < BKC_STAGES - 1; ++s) {
        if (s < nkt) load_tile(s * BKC_CTA_K, s);
        __pipeline_commit();
    }
    __pipeline_wait_prior(BKC_STAGES - 2);
    __syncthreads();

    for (int i = 0; i < nkt; ++i) {
        const int buf = i % BKC_STAGES;
        const int8_t* Ab = &As[buf * BKC_CTA_M * BKC_CTA_K];
        const int8_t* Bb = &Bs[buf * BKC_CTA_N * BKC_CTA_K];
#pragma unroll
        for (int k01 = 0; k01 < BKC_CTA_K / BKC_INTRIN_K; ++k01) {
            int8_t Afrag[BKC_MI * 16], Bfrag[BKC_NJ * 16];
            bkc_s2r_A(Ab, Afrag, lane, k01);
            bkc_s2r_B(Bb, Bfrag, lane, warp_off_n, k01);
#pragma unroll
            for (int ii = 0; ii < BKC_MI; ++ii)
#pragma unroll
                for (int jj = 0; jj < BKC_NJ; ++jj) {
                    modiff_mma_m16n8k32(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
                    modiff_mma_m16n8k32(acc[ii][jj] + 4, Afrag + ii * 16, Bfrag + jj * 16 + 8);
                }
            if (BLOCKWISE && (BLK == BKC_INTRIN_K || k01 == 1)) {
                const float* sb = &Ss[(buf * NB + (NB == 2 ? k01 : 0)) * BKC_CTA_M];
#pragma unroll
                for (int ii = 0; ii < BKC_MI; ++ii) {
                    const int lr0 = ii * BKC_INTRIN_M + gid;   // CTA-local rows of this fragment
                    const float s0 = sb[lr0], s1 = sb[lr0 + 8];
#pragma unroll
                    for (int jj = 0; jj < BKC_NJ; ++jj) {
                        int* a = acc[ii][jj];
                        float* f = accf[ii][jj];
                        f[0] = fmaf(bkc_i2f(a[0]), s0, f[0]);
                        f[1] = fmaf(bkc_i2f(a[1]), s0, f[1]);
                        f[2] = fmaf(bkc_i2f(a[2]), s1, f[2]);
                        f[3] = fmaf(bkc_i2f(a[3]), s1, f[3]);
                        f[4] = fmaf(bkc_i2f(a[4]), s0, f[4]);
                        f[5] = fmaf(bkc_i2f(a[5]), s0, f[5]);
                        f[6] = fmaf(bkc_i2f(a[6]), s1, f[6]);
                        f[7] = fmaf(bkc_i2f(a[7]), s1, f[7]);
#pragma unroll
                        for (int k = 0; k < 8; ++k) a[k] = 0;
                    }
                }
            }
        }
        const int li = i + BKC_STAGES - 1;
        if (li < nkt) load_tile(li * BKC_CTA_K, li % BKC_STAGES);
        __pipeline_commit();
        __pipeline_wait_prior(BKC_STAGES - 2);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < BKC_MI; ++i) {
        const int row0 = m0 + i * BKC_INTRIN_M + gid, row1 = row0 + 8;
#pragma unroll
        for (int j = 0; j < BKC_NJ; ++j) {
            const int col0 = n0 + warp_off_n + j * BKC_INTRIN_N + tig * 2, col1 = col0 + 8;
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
                    *(__half2*)&Out[(size_t)row * Kout + col0] =
                        __halves2half2(__float2half(a0), __float2half(a1));
                }
                if (c1ok) {
                    float a0 = v[i1] * s10, a1 = v[i1 + 1] * s11;
                    if (bias) { a0 += __half2float(bias[col1]); a1 += __half2float(bias[col1 + 1]); }
                    *(__half2*)&Out[(size_t)row * Kout + col1] =
                        __halves2half2(__float2half(a0), __float2half(a1));
                }
            }
        }
    }
}

torch::Tensor conv2d_int8_blockk(torch::Tensor x, torch::Tensor weight, torch::Tensor w_scale,
                                 torch::Tensor a_scale_blk, double a_scale, int64_t blk,
                                 int64_t stride, int64_t pad,
                                 c10::optional<torch::Tensor> bias_opt)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kInt8 && weight.dtype() == torch::kInt8, "x,w must be int8");
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4, "x [N,C,H,W] channels_last, w [K,R,S,C]");
    const int Nn = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int Kout = weight.size(0), R = weight.size(1), S = weight.size(2);
    TORCH_CHECK(weight.size(3) == C, "weight C mismatch");
    TORCH_CHECK(C % BKC_CTA_K == 0, "C must be a multiple of ", BKC_CTA_K, " (got ", C, ")");
    TORCH_CHECK(Kout % 2 == 0, "Kout must be even (the epilogue stores column pairs)");
    TORCH_CHECK(blk == 32 || blk == 64, "blk must be 32 or 64");
    TORCH_CHECK(C % blk == 0, "blk must divide C");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x must be channels_last");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous [K,R,S,C]");

    const int P = (H + 2 * pad - R) / stride + 1, Q = (W + 2 * pad - S) / stride + 1;
    const int M = Nn * P * Q, Kg = R * S * C;

    const bool blockwise = a_scale_blk.numel() > 0;
    if (blockwise) {
        TORCH_CHECK(a_scale_blk.dtype() == torch::kFloat32, "a_scale_blk must be fp32");
        TORCH_CHECK(a_scale_blk.numel() == (int64_t)Nn * H * W * (C / blk),
                    "a_scale_blk must be [N,H,W,C/blk]");
    }
    const __half* bias_p = nullptr;
    if (bias_opt.has_value() && bias_opt->numel() > 0)
        bias_p = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());

    auto out = torch::empty({Nn, Kout, P, Q}, x.options().dtype(torch::kFloat16))
                   .contiguous(at::MemoryFormat::ChannelsLast);

    const int nb = BKC_CTA_K / (int)blk;
    const size_t smem = (size_t)BKC_STAGES * (BKC_CTA_M + BKC_CTA_N) * BKC_CTA_K
                        + (blockwise ? (size_t)BKC_STAGES * nb * BKC_CTA_M * sizeof(float) : 0);
    dim3 grid((Kout + BKC_CTA_N - 1) / BKC_CTA_N, (M + BKC_CTA_M - 1) / BKC_CTA_M);
    dim3 block(BKC_NUM_WARPS * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

#define BKC_LAUNCH(BLKV, BW)                                                                    \
    do {                                                                                        \
        auto kern = conv2d_int8_blockk_kernel<BLKV, BW>;                                        \
        C10_CUDA_CHECK(cudaFuncSetAttribute(                                                    \
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));                     \
        kern<<<grid, block, smem, stream>>>(                                                    \
            x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), w_scale.data_ptr<float>(),         \
            blockwise ? a_scale_blk.data_ptr<float>() : nullptr, (float)a_scale,                \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), bias_p,                        \
            Nn, H, W, C, Kout, R, S, P, Q, (int)stride, (int)pad, M, Kg);                       \
    } while (0)
    if (blockwise) {
        if (blk == 32) { BKC_LAUNCH(32, true); } else { BKC_LAUNCH(64, true); }
    } else {
        if (blk == 32) { BKC_LAUNCH(32, false); } else { BKC_LAUNCH(64, false); }
    }
#undef BKC_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
