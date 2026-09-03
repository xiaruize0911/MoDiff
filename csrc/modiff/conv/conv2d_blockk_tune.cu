// ============================================================================================
// TUNABLE blockwise-along-C conv, int8 and int4, for the tile/B parameter sweep.
//
// Separate file on purpose: conv2d_int8_blockk.cu is wired into the model and must not move
// while we sweep. Once a config wins it gets promoted there.
//
// WHY A SWEEP. The flush is 3 CUDA-core ops per accumulator per block against B MACs of tensor
// work, so the predicted tax is 1 + (3/B)*(tensor_rate/fp32_rate). On A40 that is
//     B      32     64    128    256
//   int8   1.75x  1.38x  1.19x  1.09x        (measured at B=64: 1.45x)
//   int4   2.50x  1.75x  1.38x  1.19x        (measured at B=64: 1.45x)
// so hitting 1.25x needs B>=128 (int8) / B>=256 (int4). Large B previously lost to REGISTERS,
// not ALU: a block spanning several K tiles keeps the int32 accumulator live across them.
// CTA_M is the counter-lever -- accumulators per thread are MI*NJ*8 = (CTA_M/16)*NJ*8, so
// halving CTA_M halves them. That is the axis this file exists to sweep.
//
// Fixed here: CTA_N=128, CTA_K=64 BYTES, 8 warps, WARP_N=16 (so the warp tiling and the
// smem->register path are byte-identical to the production kernel). Swept: CTA_M, STAGES, BLK.
// int4 shares the byte layout exactly -- one mma.m16n8k64.s4 and one mma.m16n8k32.s8 both
// consume 32 bytes per operand row -- so only the element accounting and the mma differ.
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

#define TU_NW 8              // warps (fixed: (TM/WM)*(TN/WN) must equal 8)
#define TU_INTRIN_M 16
#define TU_INTRIN_N 16
#define TU_INTRIN_KB 32      // BYTES per mma per operand row (same for s8-k32 and s4-k64)
#define TU_PACK 16
#define TU_THREADS (TU_NW * 32)            // 256

#define TU_MAGIC_I 0x4B400000
#define TU_MAGIC_F 12582912.0f
__device__ __forceinline__ float tu_i2f(int v) {
    return __int_as_float(v + TU_MAGIC_I) - TU_MAGIC_F;
}

__device__ __forceinline__ void tu_cp_async_zfill(uint32_t smem, const void* src, bool pred) {
    const int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem), "l"(src),
                 "n"(16), "r"(src_size));
}

struct TuRow { int pix0, h0, w0; bool m_ok; };
__device__ __forceinline__ TuRow tu_row(int m, int M, int H, int W, int P, int Q,
                                        int stride, int pad) {
    TuRow r;
    r.m_ok = m < M;
    const int mm = r.m_ok ? m : 0;
    const int n = mm / (P * Q), rem = mm - n * P * Q;
    const int p = rem / Q, q = rem - p * Q;
    r.h0 = p * stride - pad;
    r.w0 = q * stride - pad;
    r.pix0 = (n * H + r.h0) * W + r.w0;
    return r;
}

// smem -> registers. Byte arithmetic, so shared by int8 and int4. TK is the row stride in
// BYTES; CPR = TK/16 chunks per row, and the swizzle xors over CPR (not a hardcoded 4).
template <int TK, int MI>
__device__ __forceinline__ void tu_s2r_A(const int8_t* src, int8_t* dst, int lane, int k01,
                                         int warp_m_off) {
    constexpr int CPR = TK / TU_PACK;
    const int ld_col = (k01 * TU_INTRIN_KB + (lane / 16) * 16) / TU_PACK;
#pragma unroll
    for (int si = 0; si < MI; ++si) {
        const int ld_row = warp_m_off + si * TU_INTRIN_M + (lane % 16);
        const int swz = ld_col ^ ((ld_row / 2) & (CPR - 1));
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * TK + swz * TU_PACK));
    }
}
template <int TK, int NJ>
__device__ __forceinline__ void tu_s2r_B(const int8_t* src, int8_t* dst, int lane,
                                         int warp_off_n, int k01) {
    constexpr int CPR = TK / TU_PACK;
    const int ld_col = (k01 * TU_INTRIN_KB + ((lane / 8) % 2) * 16) / TU_PACK;
#pragma unroll
    for (int si = 0; si < NJ; ++si) {
        const int ld_row = warp_off_n + si * TU_INTRIN_N + ((lane / 8 / 2) * 8 + lane % 8);
        const int swz = ld_col ^ ((ld_row / 2) & (CPR - 1));
        modiff_ldmatrix_x4(dst + si * 16,
                           modiff_smem_ptr(src + ld_row * TK + swz * TU_PACK));
    }
}

// BW is a TEMPLATE parameter, not a runtime `a_scale_blk != nullptr` test. With a runtime test
// both datapaths (int32 acc and fp32 accf) stay live and the flush branch sits in the innermost
// loop; measured 6x slower than the production kernel at the identical config. Same lesson as
// docs/ahat_blockwise_2026-09-01: dead code in the instance costs registers costs occupancy.
template <int TM, int TN, int TK, int WM, int WN, int SA, int SB, int BLK, bool I4, bool BW>
static __global__ __launch_bounds__(TU_THREADS, 2) void blockk_tune_kernel(
    const int8_t* __restrict__ X, const int8_t* __restrict__ Wt,
    const float* __restrict__ w_scale, const float* __restrict__ a_scale_blk, float a_scale,
    __half* __restrict__ Out, int Nn, int H, int W, int C, int Kout, int R, int S,
    int P, int Q, int stride, int pad, int M, int Kg)
{
    constexpr int CPR = TK / TU_PACK;               // 16B chunks per smem row
    constexpr int MI  = WM / TU_INTRIN_M;           // M fragments per warp
    constexpr int NJ  = WN / TU_INTRIN_N;           // N fragments per warp
    constexpr int WPN = TN / WN;                    // warps along N
    constexpr int EPB = I4 ? 2 : 1;                 // elements per byte
    constexpr int EPM = I4 ? 64 : 32;               // elements per mma
    constexpr int CTA_KE = TK * EPB;                // elements per K tile
    constexpr int SPT = TK / TU_INTRIN_KB;          // mma steps per tile
    // int4 cannot express BLK < 64: one mma.m16n8k64.s4 reduces 64 elements, so a smaller
    // block's partial sum is not separable from it. Clamped to keep the instantiation legal;
    // the host refuses the combination.
    constexpr int SPB = (BLK / EPM) > 0 ? (BLK / EPM) : 1;
    constexpr int NB  = (SPT >= SPB) ? (SPT / SPB) : 1;   // scale slots to stage per tile
    constexpr int SLOTS = TM * CPR / TU_THREADS;
    static_assert(SA >= 1 && SB >= 1, "stage counts >= 1");          // A-loader chunk slots per thread
    static_assert((TM / WM) * WPN == TU_NW, "warp tiling must cover the CTA with 8 warps");
    static_assert(TM * CPR % TU_THREADS == 0, "A loader needs whole slots per thread");

    extern __shared__ char tu_smem[];
    int8_t* As = reinterpret_cast<int8_t*>(tu_smem);
    // Asymmetric rings. smem = SA*TM*TK + SB*TN*TK, so SA=2/SB=1 fits a FULL 128x128 tile at
    // TK=128 in 50176 B -> 2 CTA/SM, which SA=SB=2 (66560 B) cannot. That is the one combination
    // the sweep was missing: the tile quality lives at 128x128 and the low tax needs TK==BLK.
    // The price is that B loses its load/compute overlap and needs a barrier before each reload.
    int8_t* Bs = As + SA * TM * TK;
    float* Ss = reinterpret_cast<float*>(Bs + SB * TN * TK);

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
    const int warp_m = warp / WPN, warp_n = warp % WPN;
    const int warp_off_m = warp_m * WM, warp_off_n = warp_n * WN;
    const int nb_c = C / BLK;
    const size_t pix_max = (size_t)Nn * H * W;
    const int nkt = Kg / CTA_KE;

    // Hoisted gather state: each thread always loads the same A rows and stages the same
    // scale rows, so the (n,p,q) divisions happen once per kernel rather than once per tile.
    TuRow ra[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s)
        ra[s] = tu_row(m0 + ((t + s * TU_THREADS) / CPR), M, H, W, P, Q, stride, pad);
    const TuRow rs0 = tu_row(m0 + (t % TM), M, H, W, P, Q, stride, pad);

    auto load_A = [&](int kt, int buf) {
        const int rs = kt / C, c0 = kt - rs * C;      // kt, c0 in ELEMENTS
        const int ro = rs / S, so = rs - ro * S;
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int idx = t + s * TU_THREADS;
            const int r = idx / CPR, off16 = idx % CPR;
            const TuRow& rr = ra[s];
            const int h = rr.h0 + ro, w = rr.w0 + so;
            const bool ok = rr.m_ok && (unsigned)h < (unsigned)H && (unsigned)w < (unsigned)W;
            const size_t pix = ok ? (size_t)(rr.pix0 + ro * W + so) : 0;
            const int8_t* src = X + pix * (C / EPB) + (c0 / EPB) + off16 * 16;
            const int swz = (off16 ^ ((r / 2) & (CPR - 1))) * 16;
            tu_cp_async_zfill(modiff_smem_ptr(&As[buf * TM * TK + r * TK + swz]),
                              src, ok);
        }
    };
    auto load_B = [&](int kt, int buf) {
        for (int c = t; c < TN * CPR; c += TU_THREADS) {
            const int r = c / CPR, off16 = c % CPR;
            const int swz = (off16 ^ ((r / 2) & (CPR - 1))) * 16;
            modiff_cp_async_cg(
                modiff_smem_ptr(&Bs[buf * TN * TK + r * TK + swz]),
                (const uint4*)(Wt + (size_t)(n0 + r) * (Kg / EPB) + (kt / EPB) + off16 * 16),
                (n0 + r) < Kout);
        }
    };
    auto load_S = [&](int kt, int buf) {
        const int rs = kt / C, c0 = kt - rs * C;
        const int ro = rs / S, so = rs - ro * S;
        if (BW) {
            const int h = rs0.h0 + ro, w = rs0.w0 + so;
            const bool ok = rs0.m_ok && (unsigned)h < (unsigned)H && (unsigned)w < (unsigned)W;
            const size_t pix = ok ? (size_t)(rs0.pix0 + ro * W + so) : 0;
            const int r = t % TM;
#pragma unroll
            for (int b = 0; b < NB; ++b)
                if ((t / TM) == b || TU_THREADS / TM <= b)
                    Ss[(buf * NB + b) * TM + r] =
                        (ok && pix < pix_max)
                            ? a_scale_blk[pix * nb_c + (c0 / BLK) + b] : 0.0f;
        }
    };
    auto load_tile = [&](int kt, int bufA, int bufB) {
        load_A(kt, bufA); load_B(kt, bufB); load_S(kt, bufA);
    };

    int acc[MI][NJ][8];
    float accf[MI][NJ][8];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int k = 0; k < 8; ++k) { acc[i][j][k] = 0; accf[i][j][k] = 0.f; }

    // Prologue: fill tile 0 in both rings, plus SA-1 further A tiles.
    if (nkt > 0) { load_tile(0, 0, 0); __pipeline_commit(); }
#pragma unroll
    for (int s = 1; s < SA; ++s) {
        if (s < nkt) { load_A(s * CTA_KE, s % SA); load_S(s * CTA_KE, s % SA); }
        __pipeline_commit();
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int i = 0; i < nkt; ++i) {
        const int buf = i % SA;
        const int8_t* Ab = &As[(i % SA) * TM * TK];
        const int8_t* Bb = &Bs[(i % SB) * TN * TK];
#pragma unroll
        for (int k01 = 0; k01 < SPT; ++k01) {
            int8_t Afrag[MI * 16], Bfrag[NJ * 16];
            tu_s2r_A<TK, MI>(Ab, Afrag, lane, k01, warp_off_m);
            tu_s2r_B<TK, NJ>(Bb, Bfrag, lane, warp_off_n, k01);
#pragma unroll
            for (int ii = 0; ii < MI; ++ii)
#pragma unroll
                for (int jj = 0; jj < NJ; ++jj) {
                    if (I4) {
                        modiff_mma_m16n8k64_s4(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
                        modiff_mma_m16n8k64_s4(acc[ii][jj] + 4, Afrag + ii * 16,
                                               Bfrag + jj * 16 + 8);
                    } else {
                        modiff_mma_m16n8k32(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
                        modiff_mma_m16n8k32(acc[ii][jj] + 4, Afrag + ii * 16,
                                            Bfrag + jj * 16 + 8);
                    }
                }
            // Flush at the end of every block. One global step counter handles both
            // "several blocks inside a tile" and "one block spanning several tiles".
            if (BW && (((i * SPT + k01) + 1) % SPB == 0)) {
                const int slot = (SPT >= SPB) ? (k01 / SPB) : 0;
                const float* sb = &Ss[(buf * NB + slot) * TM];
#pragma unroll
                for (int ii = 0; ii < MI; ++ii) {
                    const int lr0 = warp_off_m + ii * TU_INTRIN_M + gid;
                    const float s0 = sb[lr0], s1 = sb[lr0 + 8];
#pragma unroll
                    for (int jj = 0; jj < NJ; ++jj) {
                        int* a = acc[ii][jj];
                        float* f = accf[ii][jj];
                        f[0] = fmaf(tu_i2f(a[0]), s0, f[0]);
                        f[1] = fmaf(tu_i2f(a[1]), s0, f[1]);
                        f[2] = fmaf(tu_i2f(a[2]), s1, f[2]);
                        f[3] = fmaf(tu_i2f(a[3]), s1, f[3]);
                        f[4] = fmaf(tu_i2f(a[4]), s0, f[4]);
                        f[5] = fmaf(tu_i2f(a[5]), s0, f[5]);
                        f[6] = fmaf(tu_i2f(a[6]), s1, f[6]);
                        f[7] = fmaf(tu_i2f(a[7]), s1, f[7]);
#pragma unroll
                        for (int k = 0; k < 8; ++k) a[k] = 0;
                    }
                }
            }
        }
        // A prefetches SA-1 ahead into a buffer nobody is reading. B, when single-buffered,
        // can only be reloaded after every warp has finished reading it -- hence the barrier.
        const int la = i + SA - 1;
        if (SA > 1) { if (la < nkt) { load_A(la * CTA_KE, la % SA); load_S(la * CTA_KE, la % SA); } }
        if (SB > 1) {
            const int lb = i + SB - 1;
            if (lb < nkt) load_B(lb * CTA_KE, lb % SB);
        } else {
            __syncthreads();
            if (i + 1 < nkt) load_B((i + 1) * CTA_KE, 0);
            if (SA == 1) { load_A((i + 1) * CTA_KE, 0); load_S((i + 1) * CTA_KE, 0); }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MI; ++i) {
        const int row0 = m0 + warp_off_m + i * TU_INTRIN_M + gid, row1 = row0 + 8;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int col0 = n0 + warp_off_n + j * TU_INTRIN_N + tig * 2, col1 = col0 + 8;
            const float as = BW ? 1.0f : a_scale;
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k)
                v[k] = BW ? accf[i][j][k] : (float)acc[i][j][k];
            const bool c0ok = col0 + 1 < Kout, c1ok = col1 + 1 < Kout;
            const float s00 = c0ok ? as * w_scale[col0] : 0.f;
            const float s01 = c0ok ? as * w_scale[col0 + 1] : 0.f;
            const float s10 = c1ok ? as * w_scale[col1] : 0.f;
            const float s11 = c1ok ? as * w_scale[col1 + 1] : 0.f;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = half ? row1 : row0;
                if (row >= M) continue;
                const int i0 = half ? 2 : 0, i1 = half ? 6 : 4;
                if (c0ok)
                    *(__half2*)&Out[(size_t)row * Kout + col0] = __halves2half2(
                        __float2half(v[i0] * s00), __float2half(v[i0 + 1] * s01));
                if (c1ok)
                    *(__half2*)&Out[(size_t)row * Kout + col1] = __halves2half2(
                        __float2half(v[i1] * s10), __float2half(v[i1 + 1] * s11));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Curated config table. Each entry is a distinct kernel instantiation, so this is deliberately
// short. smem = TS*(TM+TN)*TK + TS*NB*TM*4 must stay <= 50688 B for 2 CTA/SM on an A40.
//
// The important axis is TK vs BLK. A block spanning several K tiles keeps the int32 accumulator
// live across them, which is what made large BLK lose on registers. Setting TK*EPB == BLK gives
// exactly one flush per tile and no carry -- that is configs 3/4/5 (int8 B=128) and, for int4
// where one byte is two elements, config 4 reaches B=256 with no carry.
struct TuCfg { int tm, tn, tk, wm, wn, sa, sb, blk; const char* name; };
static const TuCfg TU_CFGS[] = {
    {128, 128, 64, 128, 16, 2, 2, 64, "M128N128K64_W128x16_S2_B64"},   // 0 production equivalent
    {128, 128, 64, 128, 16, 3, 3, 64, "M128N128K64_W128x16_S3_B64"},   // 1
    {128, 128, 64, 64, 32, 2, 2, 64, "M128N128K64_W64x32_S2_B64"},    // 2 warp tiling variant
    {128, 64, 128, 64, 16, 2, 2, 128, "M128N64K128_W64x16_S2_B128"},   // 3 TK==BLK, no carry
    {64, 128, 128, 64, 16, 2, 2, 128, "M64N128K128_W64x16_S2_B128"},   // 4 TK==BLK, no carry
    {128, 64, 128, 32, 32, 2, 2, 128, "M128N64K128_W32x32_S2_B128"},   // 5 TK==BLK, other warps
    {128, 128, 64, 128, 16, 2, 2, 128, "M128N128K64_W128x16_S2_B128"},  // 6 carry over 2 tiles
    {128, 64, 128, 64, 16, 2, 2, 256, "M128N64K128_W64x16_S2_B256"},   // 7 int8: carry; int4: none
    {64, 128, 128, 64, 16, 2, 2, 256, "M64N128K128_W64x16_S2_B256"},   // 8
    {128, 128, 64, 128, 16, 2, 2, 32, "M128N128K64_W128x16_S2_B32"},   // 9 int8 only
    {128, 64, 128, 64, 16, 3, 3, 128, "M128N64K128_W64x16_S3_B128"},   // 10 smem 75264 -> 1 CTA/SM
    {64, 64, 128, 32, 16, 2, 2, 128, "M64N64K128_W32x16_S2_B128"},    // 11 small tile
    // The combination missing above: keep the FULL 128x128 tile (which is where the tile
    // quality is -- cfgs 0/1/2/6 have the best scalar numbers) AND set TK==BLK so there is no
    // cross-tile carry. That costs smem: 2*(128+128)*128 = 64 KiB -> 1 CTA/SM. cfg 13 buys the
    // occupancy back by dropping to a single stage instead.
    {128, 128, 128, 64, 32, 2, 2, 128, "M128N128K128_W64x32_S2_B128"},  // 12 full tile, 1 CTA/SM
    {128, 128, 128, 128, 16, 1, 1, 128, "M128N128K128_W128x16_S1_B128"}, // 13 S1 -> 3 CTA/SM
    {128, 128, 64, 64, 32, 3, 3, 64, "M128N128K64_W64x32_S3_B64"},    // 14 best-tile candidate
    {128, 128, 128, 64, 32, 2, 2, 256, "M128N128K128_W64x32_S2_B256"},  // 15 int4: TK*2==BLK
    // Asymmetric rings: SA=2 / SB=1 fits the FULL 128x128 tile at TK=128 into 50176 B, i.e.
    // 2 CTA/SM, which SA=SB=2 cannot (66560 B -> 1 CTA/SM, cfg 12). This is the combination the
    // rest of the table cannot express: 128x128 tile quality AND TK==BLK (no accumulator carry).
    // The price is that B loses load/compute overlap and needs a barrier before each reload.
    {128, 128, 128, 128, 16, 2, 1, 128, "M128N128K128_W128x16_SA2SB1_B128"},  // 16
    {128, 128, 128, 64, 32, 2, 1, 128, "M128N128K128_W64x32_SA2SB1_B128"},    // 17
    {128, 128, 128, 64, 32, 2, 1, 256, "M128N128K128_W64x32_SA2SB1_B256"},    // 18 int4: no carry
    // ---- TK=32 BYTES. The point is ELIGIBILITY, not tax. int4 at TK=128 needs C%256==0 and
    // at TK=64 needs C%128==0, which 12 of the UNet's 70 conv layers (C=192, C=576) fail --
    // they fall back to per-tensor and make any W4A4 blockwise attribution impure. TK=32 needs
    // only C%64==0, which ALL six channel counts (192/384/576/768/1152/1536) satisfy.
    // TK*EPB == BLK at BLK=64, so there is still exactly one flush per tile and no carry.
    // Cost: SPT=1, i.e. one mma step per tile, so there is much less compute to hide the tile
    // load behind -- which is why the deeper rings (SA=SB=3/4) are swept here and not above.
    // smem is tiny (17408 B at S2), so depth is affordable: S4 is still 34816 -> 2 CTA/SM.
    {128, 128, 32, 128, 16, 2, 2, 64, "M128N128K32_W128x16_S2_B64"},      // 19 int4: all C
    {128, 128, 32, 64, 32, 2, 2, 64, "M128N128K32_W64x32_S2_B64"},       // 20 warp variant
    {128, 128, 32, 128, 16, 3, 3, 64, "M128N128K32_W128x16_S3_B64"},      // 21 deeper ring
    {128, 128, 32, 128, 16, 4, 4, 64, "M128N128K32_W128x16_S4_B64"},      // 22 deeper still
    {128, 128, 32, 128, 16, 4, 4, 128, "M128N128K32_W128x16_S4_B128"},     // 23 carry over 2 tiles
};
int64_t blockk_tune_num_cfgs() { return (int64_t)(sizeof(TU_CFGS) / sizeof(TuCfg)); }
std::string blockk_tune_cfg_name(int64_t i) {
    TORCH_CHECK(i >= 0 && i < blockk_tune_num_cfgs(), "cfg out of range");
    return TU_CFGS[i].name;
}

torch::Tensor conv2d_blockk_tune(torch::Tensor x, torch::Tensor weight, torch::Tensor w_scale,
                                 torch::Tensor a_scale_blk, double a_scale,
                                 int64_t cfg, bool int4, int64_t stride, int64_t pad)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "cuda only");
    TORCH_CHECK(cfg >= 0 && cfg < blockk_tune_num_cfgs(), "cfg out of range");
    const TuCfg& K = TU_CFGS[cfg];
    const int Nn = x.size(0);
    int H, W, C, Kout, R, S;
    if (int4) {
        H = x.size(1); W = x.size(2);
        Kout = weight.size(0); R = weight.size(1); S = weight.size(2);
        C = weight.size(3) * 2;
        TORCH_CHECK(x.size(3) == C / 2, "packed channel mismatch");
        TORCH_CHECK(x.is_contiguous(), "packed x must be contiguous [N,H,W,C/2]");
    } else {
        C = x.size(1); H = x.size(2); W = x.size(3);
        Kout = weight.size(0); R = weight.size(1); S = weight.size(2);
        TORCH_CHECK(weight.size(3) == C, "weight C mismatch");
        TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x channels_last");
    }
    const int EPB = int4 ? 2 : 1, EPM = int4 ? 64 : 32;
    const int CTA_KE = K.tk * EPB;
    TORCH_CHECK(C % CTA_KE == 0, "C=", C, " must be a multiple of ", CTA_KE);
    TORCH_CHECK(C % K.blk == 0, "blk must divide C");
    TORCH_CHECK(K.blk >= EPM, int4 ? "int4 needs blk >= 64" : "blk >= 32");
    TORCH_CHECK(Kout % 2 == 0 && Kout % K.tn == 0 ? true : true, "");
    const int P = (H + 2 * pad - R) / stride + 1, Q = (W + 2 * pad - S) / stride + 1;
    const int M = Nn * P * Q, Kg = R * S * C;
    const bool bw = a_scale_blk.numel() > 0;
    auto out = torch::empty({Nn, Kout, P, Q},
                            x.options().dtype(torch::kFloat16)
                                .memory_format(at::MemoryFormat::ChannelsLast));
    const int SPT = K.tk / TU_INTRIN_KB, SPB = K.blk / EPM;
    const int NB = (SPT >= SPB) ? (SPT / SPB) : 1;
    const size_t smem = (size_t)K.sa * K.tm * K.tk + (size_t)K.sb * K.tn * K.tk
                        + (bw ? (size_t)K.sa * NB * K.tm * sizeof(float) : 0);
    dim3 grid((Kout + K.tn - 1) / K.tn, (M + K.tm - 1) / K.tm);
    dim3 block(TU_THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* sbp = bw ? a_scale_blk.data_ptr<float>() : nullptr;

#define TU_LAUNCH(TM, TN, TK, WM, WN, SA, SB, BL, I4)                                          \
    do {                                                                                    \
        auto kf = bw ? blockk_tune_kernel<TM, TN, TK, WM, WN, SA, SB, BL, I4, true>              \
                     : blockk_tune_kernel<TM, TN, TK, WM, WN, SA, SB, BL, I4, false>;            \
        C10_CUDA_CHECK(cudaFuncSetAttribute(                                                \
            kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));                    \
        kf<<<grid, block, smem, stream>>>(                                                  \
            x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), w_scale.data_ptr<float>(),     \
            sbp, (float)a_scale,                                                            \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),                            \
            Nn, H, W, C, Kout, R, S, P, Q, (int)stride, (int)pad, M, Kg);                    \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                     \
        return out;                                                                          \
    } while (0)
#define TU_CASE(IDX, TM, TN, TK, WM, WN, SA, SB, BL)                                           \
    if (cfg == IDX) { if (int4) TU_LAUNCH(TM, TN, TK, WM, WN, SA, SB, BL, true);            \
                      else      TU_LAUNCH(TM, TN, TK, WM, WN, SA, SB, BL, false); }

    TU_CASE(0,128, 128, 64, 128, 16, 2, 2, 64)
    TU_CASE(1,128, 128, 64, 128, 16, 3, 3, 64)
    TU_CASE(2,128, 128, 64, 64, 32, 2, 2, 64)
    TU_CASE(3,128, 64, 128, 64, 16, 2, 2, 128)
    TU_CASE(4,64, 128, 128, 64, 16, 2, 2, 128)
    TU_CASE(5,128, 64, 128, 32, 32, 2, 2, 128)
    TU_CASE(6,128, 128, 64, 128, 16, 2, 2, 128)
    TU_CASE(7,128, 64, 128, 64, 16, 2, 2, 256)
    TU_CASE(8,64, 128, 128, 64, 16, 2, 2, 256)
    TU_CASE(9,128, 128, 64, 128, 16, 2, 2, 32)
    TU_CASE(10,128, 64, 128, 64, 16, 3, 3, 128)
    TU_CASE(11,64, 64, 128, 32, 16, 2, 2, 128)
    TU_CASE(12,128, 128, 128, 64, 32, 2, 2, 128)
    TU_CASE(13,128, 128, 128, 128, 16, 1, 1, 128)
    TU_CASE(14,128, 128, 64, 64, 32, 3, 3, 64)
    TU_CASE(15,128, 128, 128, 64, 32, 2, 2, 256)
    TU_CASE(16,128, 128, 128, 128, 16, 2, 1, 128)
    TU_CASE(17,128, 128, 128, 64, 32, 2, 1, 128)
    TU_CASE(18,128, 128, 128, 64, 32, 2, 1, 256)
    TU_CASE(19,128, 128, 32, 128, 16, 2, 2, 64)
    TU_CASE(20,128, 128, 32, 64, 32, 2, 2, 64)
    TU_CASE(21,128, 128, 32, 128, 16, 3, 3, 64)
    TU_CASE(22,128, 128, 32, 128, 16, 4, 4, 64)
    TU_CASE(23,128, 128, 32, 128, 16, 4, 4, 128)
#undef TU_CASE
#undef TU_LAUNCH
    TORCH_CHECK(false, "unreachable cfg ", cfg);
}
