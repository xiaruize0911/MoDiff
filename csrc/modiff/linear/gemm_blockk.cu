// ============================================================================================
// W8A8 int8 GEMM with a BLOCKWISE-ALONG-K activation scale, and a matched scalar-alpha control.
//
// Why this kernel exists
// ----------------------
// Every other int8 GEMM/conv in this tree takes ONE scalar activation scale, because that scale
// is the epilogue's `alpha` and the epilogue only ever sees the finished int32 accumulator. A
// scale that varies along K cannot be applied there: by then the K-blocks have been summed.
// docs/act_blockwise_2026-09-01 measures what that costs in accuracy -- the shipped per-tensor
// static scale carries ~20x the quantization error of a B=32 along-C scale -- so this kernel
// implements the other side: dequantize inside the mainloop, once per K-block.
//
// The shape fits Ampere well. mma.m16n8k32.s8 reduces exactly 32 K per instruction, so one mma
// IS one B=32 block: its int32 result is already the block's partial sum, with no extra
// bookkeeping. B=64 flushes once per CTA-K tile.
//
// The two costs, both real and both measured by the BLOCKWISE=false twin below:
//
//   REGISTERS. A blockwise mainloop must hold an int32 block partial AND an fp32 running sum at
//   the same time, so the accumulator register count doubles. This is independent of B. The
//   shipped gemm_w8a8_kernel_awq carries acc[8][2][8] = 128 int32/thread, which would become 256
//   and spill, so this kernel runs 8 warps with WARP_N=16 (NJ=1) instead of 4 with WARP_N=32:
//   acc[8][1][8] = 64 int32 + 64 fp32. Same CTA tile, same 48 KiB of smem, same 8 warps/SM.
//
//   ALU. Per accumulator per flush: one IADD + one FADD (the int32->float conversion) and one
//   FFMA. Scales as 1/B, which is why B is a template parameter and not 32.
//
// The int32->float conversion deliberately avoids I2F, which issues on the XU pipe at a quarter
// of the FMA rate on GA10x -- the same reason ahat_cache.cuh converts int8 with PRMT. Adding an
// integer to the mantissa of 1.5*2^23 (where a float's ulp is exactly 1) and subtracting it back
// is exact and full rate. Valid for |v| < 2^22; a K=128 block of int8 products maxes out at
// 128*127*127 = 2064512, so B up to 128 is safe with room to spare.
//
// SCOPE: this is the measurement vehicle for the mainloop question, not a drop-in for the shipped
// GEMM. It has no o_hat accumulate, no residual, no int8 output, and no QKV layouts.
// ============================================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "../common/mma_int8.cuh"

#define BK_CTA_M 128
#define BK_CTA_N 128
#define BK_CTA_K 64
#define BK_NUM_WARPS 8
#define BK_WARP_N 16          // CTA_N / NUM_WARPS -> NJ = 1, half the accumulators of the 4-warp twin
#define BK_STAGES 3           // smem = 3*(128*64 + 128*64) = 48 KiB, the static limit
#define BK_INTRIN_M 16
#define BK_INTRIN_N 16
#define BK_INTRIN_K 32
#define BK_PACK_SIZE 16

#define BK_MI (BK_CTA_M / BK_INTRIN_M)   // 8
#define BK_NJ (BK_WARP_N / BK_INTRIN_N)  // 1

// int32 -> float without the conversion pipe. Exact for |v| < 2^22.
#define BK_MAGIC_I 0x4B400000
#define BK_MAGIC_F 12582912.0f           // 1.5 * 2^23
__device__ __forceinline__ float bk_i2f(int v) {
    return __int_as_float(v + BK_MAGIC_I) - BK_MAGIC_F;
}

// Shared->register fragment loads. Identical formulas to gwq_s2r_A/B in gemm_wxax.cu (AWQ's
// share_to_reg_one_stage_A/B), re-derived here against the BK_* tile so the two files can be
// tuned independently; the XOR swizzle must match the writer in BK_LOAD below.
__device__ __forceinline__ void bk_s2r_A(const int8_t* src, int8_t* dst, int lane, int k_0_1) {
    const int ld_col = (k_0_1 * BK_INTRIN_K + (lane / 16) * 16) / BK_PACK_SIZE;
#pragma unroll
    for (int si = 0; si < BK_MI; ++si) {
        const int ld_row = si * BK_INTRIN_M + (lane % 16);
        const int ld_col_swz = ld_col ^ ((ld_row / 2) & 3);
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * BK_CTA_K + ld_col_swz * BK_PACK_SIZE));
    }
}

__device__ __forceinline__ void bk_s2r_B(const int8_t* src, int8_t* dst, int lane,
                                         int warp_offset_n, int k_0_1) {
    const int ld_col = (k_0_1 * BK_INTRIN_K + ((lane / 8) % 2) * 16) / BK_PACK_SIZE;
#pragma unroll
    for (int si = 0; si < BK_NJ; ++si) {
        const int ld_row = warp_offset_n + si * BK_INTRIN_N + ((lane / 8 / 2) * 8 + lane % 8);
        const int ld_col_swz = ld_col ^ ((ld_row / 2) & 3);
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * BK_CTA_K + ld_col_swz * BK_PACK_SIZE));
    }
}

// BLK: K-elements per activation scale block (32 or 64; both divide BK_CTA_K).
// BLOCKWISE=false is the control: one scalar a_scale, pure int32 accumulate, everything else
// identical -- so a timing difference between the two instantiations is the mainloop dequant and
// nothing else.
//
// a_scale_blk is [K/BLK, M] fp32, K-block major. Block-major so that for a fixed block the M
// values a warp needs are adjacent; each thread owns rows {gid + 8j} and reads 2*BK_MI of them
// per flush, and all 8 warps read the same BK_CTA_M floats, so these land in L1.
template <int BLK, bool BLOCKWISE>
static __global__ __launch_bounds__(BK_NUM_WARPS * 32) void gemm_w8a8_blockk_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ Bm,
    const float* __restrict__ w_scale, const float* __restrict__ a_scale_blk,
    float a_scale, __half* __restrict__ C, int M, int N, int K, int n_out,
    const __half* __restrict__ bias)
{
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.y * BK_CTA_M, n0 = blockIdx.x * BK_CTA_N;
    const int warp_offset_n = warp * BK_WARP_N;

    __shared__ int8_t As[BK_STAGES][BK_CTA_M * BK_CTA_K];
    __shared__ int8_t Bs[BK_STAGES][BK_CTA_N * BK_CTA_K];

    int acc[BK_MI][BK_NJ][8];
    float accf[BK_MI][BK_NJ][8];
#pragma unroll
    for (int i = 0; i < BK_MI; ++i)
#pragma unroll
        for (int j = 0; j < BK_NJ; ++j)
#pragma unroll
            for (int k = 0; k < 8; ++k) { acc[i][j][k] = 0; accf[i][j][k] = 0.0f; }

    const int nkt = K / BK_CTA_K;
    const int nkb = K / BLK;

#define BK_LOAD(kt, buf)                                                                        \
    for (int c = t; c < BK_CTA_M * (BK_CTA_K / 16); c += blockDim.x) {                          \
        int r = c / (BK_CTA_K / 16), off16 = c % (BK_CTA_K / 16);                               \
        int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                             \
        modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * BK_CTA_K + off_swz]),                   \
                           (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off16 * 16),        \
                           (m0 + r) < M);                                                       \
    }                                                                                           \
    for (int c = t; c < BK_CTA_N * (BK_CTA_K / 16); c += blockDim.x) {                          \
        int r = c / (BK_CTA_K / 16), off16 = c % (BK_CTA_K / 16);                               \
        int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                             \
        modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * BK_CTA_K + off_swz]),                   \
                           (const uint4*)(Bm + (size_t)(n0 + r) * K + (kt) + off16 * 16),       \
                           (n0 + r) < N);                                                       \
    }

#pragma unroll
    for (int s = 0; s < BK_STAGES - 1; ++s) {
        if (s < nkt) { BK_LOAD(s * BK_CTA_K, s); }
        __pipeline_commit();
    }
    __pipeline_wait_prior(BK_STAGES - 2);
    __syncthreads();

    for (int i = 0; i < nkt; ++i) {
        const int buf = i % BK_STAGES;
#pragma unroll
        for (int k01 = 0; k01 < BK_CTA_K / BK_INTRIN_K; ++k01) {
            int8_t Afrag[BK_MI * 16], Bfrag[BK_NJ * 16];
            bk_s2r_A(&As[buf][0], Afrag, lane, k01);
            bk_s2r_B(&Bs[buf][0], Bfrag, lane, warp_offset_n, k01);
#pragma unroll
            for (int ii = 0; ii < BK_MI; ++ii)
#pragma unroll
                for (int jj = 0; jj < BK_NJ; ++jj) {
                    modiff_mma_m16n8k32(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
                    modiff_mma_m16n8k32(acc[ii][jj] + 4, Afrag + ii * 16, Bfrag + jj * 16 + 8);
                }

            if (BLOCKWISE) {
                // One flush per BLK of K. mma.m16n8k32 already reduced 32, so BLK==32 flushes
                // every k01 and BLK==64 flushes on the second.
                const int kstep = i * (BK_CTA_K / BK_INTRIN_K) + k01;   // which 32-wide step
                if (BLK == BK_INTRIN_K || (kstep & ((BLK / BK_INTRIN_K) - 1))
                        == (BLK / BK_INTRIN_K) - 1) {
                    const int kb = kstep / (BLK / BK_INTRIN_K);
                    const float* sb = a_scale_blk + (size_t)kb * M;
#pragma unroll
                    for (int ii = 0; ii < BK_MI; ++ii) {
                        // mma m16n8 fragment rows: acc[0,1,4,5] -> row0, acc[2,3,6,7] -> row0+8.
                        const int r0 = m0 + ii * BK_INTRIN_M + gid, r1 = r0 + 8;
                        const float s0 = (r0 < M) ? sb[r0] : 0.0f;
                        const float s1 = (r1 < M) ? sb[r1] : 0.0f;
#pragma unroll
                        for (int jj = 0; jj < BK_NJ; ++jj) {
                            int* a = acc[ii][jj];
                            float* f = accf[ii][jj];
                            f[0] = fmaf(bk_i2f(a[0]), s0, f[0]);
                            f[1] = fmaf(bk_i2f(a[1]), s0, f[1]);
                            f[2] = fmaf(bk_i2f(a[2]), s1, f[2]);
                            f[3] = fmaf(bk_i2f(a[3]), s1, f[3]);
                            f[4] = fmaf(bk_i2f(a[4]), s0, f[4]);
                            f[5] = fmaf(bk_i2f(a[5]), s0, f[5]);
                            f[6] = fmaf(bk_i2f(a[6]), s1, f[6]);
                            f[7] = fmaf(bk_i2f(a[7]), s1, f[7]);
#pragma unroll
                            for (int k = 0; k < 8; ++k) a[k] = 0;
                        }
                    }
                }
            }
        }
        const int li = i + BK_STAGES - 1;
        if (li < nkt) { BK_LOAD(li * BK_CTA_K, li % BK_STAGES); }
        __pipeline_commit();
        __pipeline_wait_prior(BK_STAGES - 2);
        __syncthreads();
    }
#undef BK_LOAD
    (void)nkb;

#pragma unroll
    for (int i = 0; i < BK_MI; ++i) {
        const int row0 = m0 + i * BK_INTRIN_M + gid, row1 = row0 + 8;
#pragma unroll
        for (int j = 0; j < BK_NJ; ++j) {
            const int col0 = n0 + warp_offset_n + j * BK_INTRIN_N + tig * 2, col1 = col0 + 8;
            // BLOCKWISE already applied the activation scale inside the mainloop.
            const float as = BLOCKWISE ? 1.0f : a_scale;
            const bool c0 = col0 < n_out, c1 = col1 < n_out;
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = BLOCKWISE ? accf[i][j][k] : (float)acc[i][j][k];
            const float s00 = as * w_scale[col0], s01 = as * w_scale[col0 + 1];
            const float s10 = as * w_scale[col1], s11 = as * w_scale[col1 + 1];
            if (row0 < M) {
                if (c0) {
                    float a0 = v[0] * s00, a1 = v[1] * s01;
                    if (bias) { a0 += __half2float(bias[col0]); a1 += __half2float(bias[col0 + 1]); }
                    *(__half2*)&C[(size_t)row0 * n_out + col0] =
                        __halves2half2(__float2half(a0), __float2half(a1));
                }
                if (c1) {
                    float a0 = v[4] * s10, a1 = v[5] * s11;
                    if (bias) { a0 += __half2float(bias[col1]); a1 += __half2float(bias[col1 + 1]); }
                    *(__half2*)&C[(size_t)row0 * n_out + col1] =
                        __halves2half2(__float2half(a0), __float2half(a1));
                }
            }
            if (row1 < M) {
                if (c0) {
                    float a0 = v[2] * s00, a1 = v[3] * s01;
                    if (bias) { a0 += __half2float(bias[col0]); a1 += __half2float(bias[col0 + 1]); }
                    *(__half2*)&C[(size_t)row1 * n_out + col0] =
                        __halves2half2(__float2half(a0), __float2half(a1));
                }
                if (c1) {
                    float a0 = v[6] * s10, a1 = v[7] * s11;
                    if (bias) { a0 += __half2float(bias[col1]); a1 += __half2float(bias[col1 + 1]); }
                    *(__half2*)&C[(size_t)row1 * n_out + col1] =
                        __halves2half2(__float2half(a0), __float2half(a1));
                }
            }
        }
    }
}

// `a_scale_blk`: [K/blk, M] fp32, or an empty tensor for the scalar-alpha control (then
// `a_scale` is used). `blk` must be 32 or 64.
torch::Tensor gemm_w8a8_blockk(torch::Tensor A, torch::Tensor Bm, torch::Tensor w_scale,
                               torch::Tensor a_scale_blk, double a_scale, int64_t blk,
                               c10::optional<torch::Tensor> bias_opt)
{
    TORCH_CHECK(A.is_cuda() && Bm.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8 && Bm.dtype() == torch::kInt8, "A,B must be int8");
    TORCH_CHECK(A.dim() == 2 && Bm.dim() == 2, "A [M,K], B [N,K]");
    const int M = A.size(0), K = A.size(1), N = Bm.size(0);
    TORCH_CHECK(Bm.size(1) == K, "K mismatch");
    TORCH_CHECK(K % BK_CTA_K == 0, "K must be a multiple of ", BK_CTA_K);
    TORCH_CHECK(blk == 32 || blk == 64, "blk must be 32 or 64");
    TORCH_CHECK(K % blk == 0, "blk must divide K");

    const bool blockwise = a_scale_blk.numel() > 0;
    if (blockwise) {
        TORCH_CHECK(a_scale_blk.dtype() == torch::kFloat32, "a_scale_blk must be fp32");
        TORCH_CHECK(a_scale_blk.dim() == 2 && a_scale_blk.size(0) == K / blk
                        && a_scale_blk.size(1) == M,
                    "a_scale_blk must be [K/blk, M]");
    }
    const __half* bias_p = nullptr;
    if (bias_opt.has_value() && bias_opt->numel() > 0)
        bias_p = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());

    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat16));
    dim3 grid((N + BK_CTA_N - 1) / BK_CTA_N, (M + BK_CTA_M - 1) / BK_CTA_M);
    dim3 block(BK_NUM_WARPS * 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int8_t* Ap = A.data_ptr<int8_t>();
    const int8_t* Bp = Bm.data_ptr<int8_t>();
    const float* wsp = w_scale.data_ptr<float>();
    const float* asp = blockwise ? a_scale_blk.data_ptr<float>() : nullptr;
    __half* Cp = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

#define BK_LAUNCH(BLKV, BW)                                                                     \
    gemm_w8a8_blockk_kernel<BLKV, BW><<<grid, block, 0, stream>>>(                              \
        Ap, Bp, wsp, asp, (float)a_scale, Cp, M, N, K, N, bias_p)
    if (blockwise) {
        if (blk == 32) { BK_LAUNCH(32, true); } else { BK_LAUNCH(64, true); }
    } else {
        if (blk == 32) { BK_LAUNCH(32, false); } else { BK_LAUNCH(64, false); }
    }
#undef BK_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
