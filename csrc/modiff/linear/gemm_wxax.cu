// =========================================================================
// MoDiff temporal-delta variants of the Linear GEMM backend. The baseline twin is
// csrc/baseline/linear/gemm_wxax.cu; same AWQ-tiling scheme (CTA_M=CTA_N=128, CTA_K=64,
// WARP_N=32, 4 warps, GWQ_STAGES=3, ldmatrix + XOR bank swizzle). The three host
// entry points here additionally accumulate the cross-timestep o_hat state in the
// GEMM epilogue: o_hat_t = A(Q(delta)) + o_hat_{t+1} (Eq 9), with bias and residual
// applied to the SEPARATE output C so the temporal state stays bias-free.
//
// Family 2 of the csrc/ datapath split (2026-08-12). Everything above the
// "MoDiff host entry points" marker is COPIED from the baseline file, because the
// GEMM kernels are genuinely dual-purpose: the baseline wrappers launch the SAME
// kernels with the o_hat pointer left null (the `o_hat` parameter defaults to
// nullptr and the epilogue skips the accumulate). The copies are `static` so they
// cannot collide with the exported originals at link time.
//
// KEEP THE COPIES IDENTICAL to their twins. Every A/B in docs/ compares the two
// datapaths; a numerical edit applied here and not there invalidates them. Check with
//   diff <(sed -n '88,182p;184,579p;895,999p' csrc/baseline/linear/gemm_wxax.cu) ...
// or just re-read csrc/README.md's divergence note.
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

// Explicit relative path, NOT bare: a bare include resolves through the global -I csrc
// and would pick a different tree's copy. See csrc/README.md.
#include "../common/common.cuh"
#include "../common/mma_int8.cuh"

// ==== COPIED shared tiling constants, s2r/store helpers, and GEMM kernels ====

// ---- COPY of GWQ_constants ----
// =========================================================================
// AWQ-tiling-scheme W8A8 GEMM (the production int8 Linear backend).
//
// AWQ's large-M config (w8a8_gemm_cuda.cu, num_out_feats>128 branch:
// CTA_M=128,CTA_N=128,CTA_K=64,WARP_M=128,WARP_N=32,STAGES=3) partitions N across
// warps and shares A redundantly (WARP_M==CTA_M -> every warp reads the FULL
// M-range of A; WARP_N=32, CTA_N/WARP_N=4 warps split N into 4x32 slices). This is
// a from-scratch port of that scheme (same tile shape, same swizzle formula
// `col ^ ((row/2)&3)`, same ldmatrix.m8n8.x4 reads).
//
// Deliberately NOT a full transcription of dense_kernel0: this version loads both
// INTRIN_K=32 sub-halves of a CTA_K=64 tile back-to-back before issuing the next
// global prefetch (AWQ's register-ping-pong overlaps that differently) -- simpler,
// with a small cost only at long K. Requires N%128==0 (pad at the call site,
// exactly like AWQ's own kernel requires callers to do).
// =========================================================================
#define GWQ_CTA_M 128
#define GWQ_CTA_N 128
#define GWQ_CTA_K 64
#define GWQ_WARP_N 32            // CTA_N/WARP_N = 4 warps, each owns a distinct 32-wide N slice
#define GWQ_NUM_WARPS 4
#define GWQ_STAGES 3             // matches AWQ's own STAGES for this config; smem = 2*3*128*64 = 48KiB exactly
#define GWQ_INTRIN_M 16
#define GWQ_INTRIN_N 16
#define GWQ_INTRIN_K 32
#define GWQ_PACK_SIZE 16

// ---- COPY of gwq_s2r_A ----
// Shared->register fragment loads, ported from AWQ's share_to_reg_one_stage_A/B
// (w8a8_gemm_cuda.cu) with threadIdx.x/.y replaced by lane/warp_offset_n (this
// kernel uses a flat 128-thread block like our other kernels, not (32,4)).
// warp_offset_m is always 0: WARP_M==CTA_M means every warp reads the full M-range.
__device__ __forceinline__ void gwq_s2r_A(const int8_t* src, int8_t* dst, int lane, int k_0_1) {
  constexpr int kSmemCol = GWQ_CTA_K;
  int ld_col = (k_0_1 * GWQ_INTRIN_K + (lane / 16) * 16) / GWQ_PACK_SIZE;
#pragma unroll
  for (int si = 0; si < GWQ_CTA_M / GWQ_INTRIN_M; ++si) {
    int ld_row = si * GWQ_INTRIN_M + (lane % 16);
    int ld_col_swz = ld_col ^ ((ld_row / 2) & 3);
    uint32_t addr = modiff_smem_ptr(src + ld_row * kSmemCol + ld_col_swz * GWQ_PACK_SIZE);
    modiff_ldmatrix_x4(dst + si * 16, addr);
  }
}

// ---- COPY of gwq_s2r_B ----
__device__ __forceinline__ void gwq_s2r_B(const int8_t* src, int8_t* dst, int lane, int warp_offset_n, int k_0_1) {
  constexpr int kSmemCol = GWQ_CTA_K;
  int ld_col = (k_0_1 * GWQ_INTRIN_K + ((lane / 8) % 2) * 16) / GWQ_PACK_SIZE;
#pragma unroll
  for (int si = 0; si < GWQ_WARP_N / GWQ_INTRIN_N; ++si) {
    int ld_row = warp_offset_n + si * GWQ_INTRIN_N + ((lane / 8 / 2) * 8 + lane % 8);
    int ld_col_swz = ld_col ^ ((ld_row / 2) & 3);
    uint32_t addr = modiff_smem_ptr(src + ld_row * kSmemCol + ld_col_swz * GWQ_PACK_SIZE);
    modiff_ldmatrix_x4(dst + si * 16, addr);
  }
}

// ---- COPY of gwq_store2 ----
// `n_out` is the output row stride / valid column count. When n_out==N the store is dense (original
// behavior); when n_out<N (e.g. N padded up to CTA_N=128 but the real out_features is smaller) the
// kernel writes the unpadded [M, n_out] result directly, skipping the padded columns -- this removes
// the downstream slice+`.contiguous()` copy on padded qkv/proj GEMM outputs.
// Fused epilogue store: writes two adjacent fp16 columns (col,col+1) at C[idx], optionally adding a
// per-column bias[col] and an elementwise residual[idx] in fp32 before the half cast. bias/residual
// are nullptr for the plain dequant path (behavior identical to the original store).
// MoDiff Eq 9 in the GEMM epilogue: o_hat_t = A(Q(delta)) + o_hat_{t+1}, accumulated in place.
//
// `o_hat` nullable keeps this ONE helper serving both modes -- with o_hat == nullptr the function is
// byte-for-byte the baseline it always was. That is the same pattern used for the upsample/avgpool
// delta quantize, and it is why the Linear MoDiff path needs no cloned GEMM.
//
// Two contract details that are easy to get wrong:
//   * BIAS MUST NOT be added on a modulated step. Per Eq 9 the bias belongs to o_hat_T only, so the
//     increment A(Q(delta)) carries no bias. The caller passes bias=nullptr on modulated steps; this
//     function does not silently drop it, because a non-null bias with a non-null o_hat is a caller
//     bug worth reproducing rather than hiding.
//   * the accumulate reads o_hat BEFORE adding, and the residual is added only to the SEPARATE
//     output C, never into o_hat. Folding the ResBlock skip into o_hat would corrupt the temporal
//     state for every remaining timestep.
__device__ __forceinline__ void gwq_store2(__half* C, const __half* bias, const __half* residual,
                                           size_t idx, int col, float v0, float v1,
                                           __half* o_hat = nullptr) {
  // ORDER IS THE CONTRACT. The o_hat accumulate happens FIRST, so bias and residual land on the
  // OUTPUT only and never enter the temporal state:
  //     o_hat_t = A(Q(delta)) + o_hat_{t+1}          <- Eq 9, no bias, no residual
  //     out_t   = o_hat_t + bias + residual          <- what the next layer consumes
  // Adding bias before the accumulate would compound it once per step, and folding the residual
  // into o_hat would corrupt the state for every remaining timestep. With o_hat == nullptr the
  // ordering is unobservable, so the baseline path is unchanged.
  if (o_hat) {
    float2 h = __half22float2(*(const __half2*)&o_hat[idx]);
    v0 += h.x; v1 += h.y;
    *(__half2*)&o_hat[idx] = __halves2half2(__float2half(v0), __float2half(v1));
  }
  if (bias) { v0 += __half2float(bias[col]); v1 += __half2float(bias[col + 1]); }
  if (residual) { float2 r = __half22float2(*(const __half2*)&residual[idx]); v0 += r.x; v1 += r.y; }
  if (C) *(__half2*)&C[idx] = __halves2half2(__float2half(v0), __float2half(v1));
}

// ---- COPY of gemm_w8a8_kernel_awq ----
static __global__ void gemm_w8a8_kernel_awq(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     const float* __restrict__ w_scale, float a_scale,
                                     __half* __restrict__ C, int M, int N, int K, int n_out,
                                     const __half* __restrict__ bias, const __half* __restrict__ residual,
                                     __half* __restrict__ o_hat = nullptr,
                                     // Optional DEVICE a_scale. The MoDiff path's delta scale is
                                     // computed on device per call; passing it by value would need a
                                     // .item() sync per linear per step (42 x 200 = 8400 syncs, which
                                     // cost the conv path ~5 ms/step when it made the same mistake).
                                     const float* __restrict__ a_scale_ptr = nullptr) {
  if (a_scale_ptr) a_scale = *a_scale_ptr;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.y * GWQ_CTA_M, n0 = blockIdx.x * GWQ_CTA_N;
  const int warp_offset_n = warp * GWQ_WARP_N;
  __shared__ int8_t As[GWQ_STAGES][GWQ_CTA_M * GWQ_CTA_K];
  __shared__ int8_t Bs[GWQ_STAGES][GWQ_CTA_N * GWQ_CTA_K];
  constexpr int MI = GWQ_CTA_M / GWQ_INTRIN_M, NJ = GWQ_WARP_N / GWQ_INTRIN_N;
  int acc[MI][NJ][8];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[i][j][k] = 0;
  const int nkt = K / GWQ_CTA_K;
  // Write-side swizzle mirrors AWQ's A_hoisted_col_swizzled: col chunk (0..3, PACK_SIZE=16
  // wide) XORed with (row/2)&3 before placing in shared mem; read-side (gwq_s2r_*) applies
  // the identical formula, so writer and reader always agree on physical bank.
#define GWQ_LOAD(kt, buf)                                                                          \
  for (int c = t; c < GWQ_CTA_M * (GWQ_CTA_K / 16); c += blockDim.x) {                             \
    int r = c / (GWQ_CTA_K / 16), off16 = c % (GWQ_CTA_K / 16);                                    \
    int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                                     \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * GWQ_CTA_K + off_swz]),                          \
                       (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off16 * 16), (m0 + r) < M); \
  }                                                                                                  \
  for (int c = t; c < GWQ_CTA_N * (GWQ_CTA_K / 16); c += blockDim.x) {                              \
    int r = c / (GWQ_CTA_K / 16), off16 = c % (GWQ_CTA_K / 16);                                    \
    int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                                     \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * GWQ_CTA_K + off_swz]),                          \
                       (const uint4*)(B + (size_t)(n0 + r) * K + (kt) + off16 * 16), (n0 + r) < N); \
  }
#pragma unroll
  for (int s = 0; s < GWQ_STAGES - 1; ++s) { if (s < nkt) { GWQ_LOAD(s * GWQ_CTA_K, s); } __pipeline_commit(); }
  __pipeline_wait_prior(GWQ_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % GWQ_STAGES;
#pragma unroll
    for (int k01 = 0; k01 < GWQ_CTA_K / GWQ_INTRIN_K; ++k01) {
      int8_t Afrag[MI * 16], Bfrag[NJ * 16];
      gwq_s2r_A(&As[buf][0], Afrag, lane, k01);
      gwq_s2r_B(&Bs[buf][0], Bfrag, lane, warp_offset_n, k01);
#pragma unroll
      for (int ii = 0; ii < MI; ++ii)
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
          modiff_mma_m16n8k32(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
          modiff_mma_m16n8k32(acc[ii][jj] + 4, Afrag + ii * 16, Bfrag + jj * 16 + 8);
        }
    }
    int li = i + GWQ_STAGES - 1;
    if (li < nkt) { GWQ_LOAD(li * GWQ_CTA_K, li % GWQ_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(GWQ_STAGES - 2);
    __syncthreads();
  }
#undef GWQ_LOAD
#pragma unroll
  for (int i = 0; i < MI; ++i) {
    int row0 = m0 + i * GWQ_INTRIN_M + gid, row1 = row0 + 8;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int col0 = n0 + warp_offset_n + j * GWQ_INTRIN_N + tig * 2, col1 = col0 + 8;
      float s00 = a_scale * w_scale[col0], s01 = a_scale * w_scale[col0 + 1];
      float s10 = a_scale * w_scale[col1], s11 = a_scale * w_scale[col1 + 1];
      int* accv = acc[i][j];
      bool c0 = col0 < n_out, c1 = col1 < n_out;  // n_out even -> col0<n_out guards [col0,col0+1]
      if (row0 < M) {
        if (c0) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col0, col0, accv[0] * s00, accv[1] * s01, o_hat);
        if (c1) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col1, col1, accv[4] * s10, accv[5] * s11, o_hat);
      }
      if (row1 < M) {
        if (c0) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col0, col0, accv[2] * s00, accv[3] * s01, o_hat);
        if (c1) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col1, col1, accv[6] * s10, accv[7] * s11, o_hat);
      }
    }
  }
}

// ---- COPY of gemm_w8a8_kernel_awq_out_i8 ----
// ---- int8-OUTPUT variant (output-fusion + bandwidth fix) ----
// Identical (validated) mainloop to gemm_w8a8_kernel_awq, but the epilogue requantizes the fp32
// accumulator to INT8 instead of fp16 -- halving the dominant output write (M*N*2 -> M*N*1 bytes),
// which the roofline profile showed is ~86% of this GEMM's memory traffic and the reason it sits at
// 21-68% of the fp16-output roofline. `inv_out_scale[N]` = 127/per-column-absmax (calibrated); the
// downstream op dequants with out_scale = 1/inv_out_scale (or consumes int8 directly). Stores are
// packed to 16-bit (two adjacent int8 columns per store) so the write stays coalesced.
// QKV_LAYOUT: 0=ordinary contiguous output, 1=per-head padded direct layouts,
// 2=compact QKV columns with padded K/VT only in the destination.
template <int QKV_LAYOUT = 0>
static __global__ void gemm_w8a8_kernel_awq_out_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                            const float* __restrict__ w_scale, float a_scale,
                                            const float* __restrict__ inv_out_scale,
                                            const __half* __restrict__ bias,
                                            int8_t* __restrict__ C,
                                            int M, int N, int K, int n_out,
                                            int8_t* __restrict__ qout = nullptr,
                                            int8_t* __restrict__ kout = nullptr,
                                            int8_t* __restrict__ vtout = nullptr,
                                            int nh = 0, int T = 0, int hd = 0,
                                            int hp = 0,
                                            // MoDiff DUAL output. When non-null this kernel becomes
                                            // Eq 9 plus a requantize: it accumulates the GEMM result
                                            // into o_hat (fp16 state, bias-free) and emits int8 codes
                                            // of o_hat+bias for the next consumer, in ONE pass. With
                                            // o_hat == nullptr every line below is bit-identical to
                                            // before -- the fast path folds inv_out_scale into the
                                            // dequant multiply, which is only valid when there is no
                                            // state to read, so the two branches are separate.
                                            __half* __restrict__ o_hat = nullptr,
                                            // Optional DEVICE a_scale, for the same reason gemm_w8a8_kernel_awq has one: the
                                            // MoDiff delta scale is produced on device per call, so copying it to the host to
                                            // pass by value would be one sync per linear per step (42 x 200 = 8400 per sample;
                                            // that mistake cost the conv path ~5 ms/step). Overrides a_scale when non-null.
                                            const float* __restrict__ a_scale_ptr = nullptr) {
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.y * GWQ_CTA_M, n0 = blockIdx.x * GWQ_CTA_N;
  const int warp_offset_n = warp * GWQ_WARP_N;
  if (a_scale_ptr) a_scale = *a_scale_ptr;
  __shared__ int8_t As[GWQ_STAGES][GWQ_CTA_M * GWQ_CTA_K];
  __shared__ int8_t Bs[GWQ_STAGES][GWQ_CTA_N * GWQ_CTA_K];
  constexpr int MI = GWQ_CTA_M / GWQ_INTRIN_M, NJ = GWQ_WARP_N / GWQ_INTRIN_N;
  int acc[MI][NJ][8];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[i][j][k] = 0;
  const int nkt = K / GWQ_CTA_K;
#define GWQ_LOAD_O(kt, buf)                                                                         \
  for (int c = t; c < GWQ_CTA_M * (GWQ_CTA_K / 16); c += blockDim.x) {                             \
    int r = c / (GWQ_CTA_K / 16), off16 = c % (GWQ_CTA_K / 16);                                    \
    int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                                     \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * GWQ_CTA_K + off_swz]),                          \
                       (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off16 * 16), (m0 + r) < M); \
  }                                                                                                  \
  for (int c = t; c < GWQ_CTA_N * (GWQ_CTA_K / 16); c += blockDim.x) {                              \
    int r = c / (GWQ_CTA_K / 16), off16 = c % (GWQ_CTA_K / 16);                                    \
    int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                                     \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * GWQ_CTA_K + off_swz]),                          \
                       (const uint4*)(B + (size_t)(n0 + r) * K + (kt) + off16 * 16), (n0 + r) < N); \
  }
#pragma unroll
  for (int s = 0; s < GWQ_STAGES - 1; ++s) { if (s < nkt) { GWQ_LOAD_O(s * GWQ_CTA_K, s); } __pipeline_commit(); }
  __pipeline_wait_prior(GWQ_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % GWQ_STAGES;
#pragma unroll
    for (int k01 = 0; k01 < GWQ_CTA_K / GWQ_INTRIN_K; ++k01) {
      int8_t Afrag[MI * 16], Bfrag[NJ * 16];
      gwq_s2r_A(&As[buf][0], Afrag, lane, k01);
      gwq_s2r_B(&Bs[buf][0], Bfrag, lane, warp_offset_n, k01);
#pragma unroll
      for (int ii = 0; ii < MI; ++ii)
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
          modiff_mma_m16n8k32(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
          modiff_mma_m16n8k32(acc[ii][jj] + 4, Afrag + ii * 16, Bfrag + jj * 16 + 8);
        }
    }
    int li = i + GWQ_STAGES - 1;
    if (li < nkt) { GWQ_LOAD_O(li * GWQ_CTA_K, li % GWQ_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(GWQ_STAGES - 2);
    __syncthreads();
  }
#undef GWQ_LOAD_O
  // ---- shared-memory-staged epilogue: scatter int8 results into a [CTA_M][CTA_N] smem tile (reusing
  // the As mainloop buffer), then store to global with COALESCED 128-bit (uint4 = 16 int8) writes.
  // A direct per-thread int8 store is only 16-bit and poorly coalesced (undoes the byte saving); the
  // smem stage turns it into full-width coalesced stores so the halved output actually saturates BW. ----
  auto q8 = [](float v) -> int { int x = __float2int_rn(v); return x > 127 ? 127 : (x < -127 ? -127 : x); };
  __syncthreads();                                   // mainloop done reading As/Bs
  int8_t* Cs = &As[0][0];                            // reuse As (GWQ_STAGES*CTA_M*CTA_K = 24KB >= CTA_M*CTA_N = 16KB)
#pragma unroll
  for (int i = 0; i < MI; ++i) {
    int r0 = i * GWQ_INTRIN_M + gid, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int c0 = warp_offset_n + j * GWQ_INTRIN_N + tig * 2, c1 = c0 + 8;   // local cols
      int gc0 = n0 + c0, gc1 = n0 + c1;                                   // global cols (for scales)
      float s00 = a_scale * w_scale[gc0] * inv_out_scale[gc0], s01 = a_scale * w_scale[gc0 + 1] * inv_out_scale[gc0 + 1];
      float s10 = a_scale * w_scale[gc1] * inv_out_scale[gc1], s11 = a_scale * w_scale[gc1 + 1] * inv_out_scale[gc1 + 1];
      float b00 = bias && gc0 < n_out ? __half2float(bias[gc0]) * inv_out_scale[gc0] : 0.f;
      float b01 = bias && gc0 + 1 < n_out ? __half2float(bias[gc0 + 1]) * inv_out_scale[gc0 + 1] : 0.f;
      float b10 = bias && gc1 < n_out ? __half2float(bias[gc1]) * inv_out_scale[gc1] : 0.f;
      float b11 = bias && gc1 + 1 < n_out ? __half2float(bias[gc1 + 1]) * inv_out_scale[gc1 + 1] : 0.f;
      int* accv = acc[i][j];
      if (o_hat) {
        // Dequant WITHOUT inv_out_scale (that is an output-domain scale), accumulate into the state,
        // then bias, then requantize. ORDER IS THE CONTRACT, same as gwq_store2: bias must never
        // enter o_hat or it compounds once per diffusion step.
        const float d00 = a_scale * w_scale[gc0], d01 = a_scale * w_scale[gc0 + 1];
        const float d10 = a_scale * w_scale[gc1], d11 = a_scale * w_scale[gc1 + 1];
        const float hb00 = bias && gc0 < n_out ? __half2float(bias[gc0]) : 0.f;
        const float hb01 = bias && gc0 + 1 < n_out ? __half2float(bias[gc0 + 1]) : 0.f;
        const float hb10 = bias && gc1 < n_out ? __half2float(bias[gc1]) : 0.f;
        const float hb11 = bias && gc1 + 1 < n_out ? __half2float(bias[gc1 + 1]) : 0.f;
        // Adjacent columns are contiguous in o_hat, so each pair is one __half2 access -- the same
        // pairing gwq_store2 relies on. Guarded on gc+1 < n_out because n_out can be unpadded.
        #define GWQ_OH2(RL, RG, CL, CG, V0, V1, D0, D1, B0, B1)                                  \
          if ((RG) < M && (CG) + 1 < n_out) {                                                    \
            const size_t oi = (size_t)(RG) * n_out + (CG);                                       \
            float2 h = __half22float2(*(const __half2*)&o_hat[oi]);                              \
            const float o0 = h.x + (V0) * (D0), o1 = h.y + (V1) * (D1);                          \
            *(__half2*)&o_hat[oi] = __halves2half2(__float2half(o0), __float2half(o1));           \
            Cs[(RL) * GWQ_CTA_N + (CL)] = q8((o0 + (B0)) * inv_out_scale[CG]);                   \
            Cs[(RL) * GWQ_CTA_N + (CL) + 1] = q8((o1 + (B1)) * inv_out_scale[(CG) + 1]);         \
          }
        GWQ_OH2(r0, m0 + r0, c0, gc0, accv[0], accv[1], d00, d01, hb00, hb01)
        GWQ_OH2(r0, m0 + r0, c1, gc1, accv[4], accv[5], d10, d11, hb10, hb11)
        GWQ_OH2(r1, m0 + r1, c0, gc0, accv[2], accv[3], d00, d01, hb00, hb01)
        GWQ_OH2(r1, m0 + r1, c1, gc1, accv[6], accv[7], d10, d11, hb10, hb11)
        #undef GWQ_OH2
        continue;
      }
      Cs[r0 * GWQ_CTA_N + c0] = q8(accv[0] * s00 + b00); Cs[r0 * GWQ_CTA_N + c0 + 1] = q8(accv[1] * s01 + b01);
      Cs[r0 * GWQ_CTA_N + c1] = q8(accv[4] * s10 + b10); Cs[r0 * GWQ_CTA_N + c1 + 1] = q8(accv[5] * s11 + b11);
      Cs[r1 * GWQ_CTA_N + c0] = q8(accv[2] * s00 + b00); Cs[r1 * GWQ_CTA_N + c0 + 1] = q8(accv[3] * s01 + b01);
      Cs[r1 * GWQ_CTA_N + c1] = q8(accv[6] * s10 + b10); Cs[r1 * GWQ_CTA_N + c1 + 1] = q8(accv[7] * s11 + b11);
    }
  }
  __syncthreads();
  if constexpr (QKV_LAYOUT == 0) {
    for (int idx = t; idx < GWQ_CTA_M * GWQ_CTA_N / 16; idx += blockDim.x) {
      int rc = idx * 16, row = rc / GWQ_CTA_N, col = rc % GWQ_CTA_N;
      if (m0 + row < M && n0 + col + 15 < n_out)
        *(uint4*)&C[(size_t)(m0 + row) * n_out + n0 + col] =
            *(const uint4*)&Cs[row * GWQ_CTA_N + col];
    }
  } else if constexpr (QKV_LAYOUT == 1) {
    // The offline weight layout pads every (head,Q/K/V) segment to hp=32/64.
    // A 128-column CTA therefore contains exactly 4/2 aligned segments.
    const int nseg = GWQ_CTA_N / hp, chunks_d = hp / 16;
    const int seg0 = n0 / hp;
    // Q/K: full-width coalesced 16-byte copies from each row/segment.
    const int qk_work = GWQ_CTA_M * nseg * chunks_d;
    for (int idx = t; idx < qk_work; idx += blockDim.x) {
      const int row = idx / (nseg * chunks_d);
      const int r = idx % (nseg * chunks_d);
      const int sl = r / chunks_d, dc = r % chunks_d;
      const int seg = seg0 + sl, h = seg / 3, sel = seg % 3;
      const int gm = m0 + row, d0 = dc * 16;
      if (gm < M && sel != 2) {
        const uint4 val =
            *reinterpret_cast<const uint4*>(&Cs[row * GWQ_CTA_N + sl * hp + d0]);
        if (sel == 0) {
          *reinterpret_cast<uint4*>(
              &qout[((size_t)gm * nh + h) * hp + d0]) = val;
        } else {
          const int b = gm / T, tok = gm - b * T;
          *reinterpret_cast<uint4*>(
              &kout[((size_t)(b * nh + h) * T + tok) * hp + d0]) = val;
        }
      }
    }
    // V: gather a 16-token shared-memory column into one coalesced uint4
    // global store. T is a multiple of 64 for every eligible shape.
    const int token_chunks = GWQ_CTA_M / 16;
    const int v_work = nseg * hp * token_chunks;
    for (int idx = t; idx < v_work; idx += blockDim.x) {
      const int sl = idx / (hp * token_chunks);
      const int r = idx % (hp * token_chunks);
      const int d = r / token_chunks, tc = r % token_chunks;
      const int seg = seg0 + sl, h = seg / 3, sel = seg % 3;
      const int row0 = tc * 16, gm = m0 + row0;
      if (sel == 2 && gm < M) {
        const int b = gm / T, tok = gm - b * T;
        const int col = sl * hp + d;
        uint32_t p[4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int rr = row0 + g * 4;
          p[g] = (uint8_t)Cs[(rr + 0) * GWQ_CTA_N + col]
               | ((uint32_t)(uint8_t)Cs[(rr + 1) * GWQ_CTA_N + col] << 8)
               | ((uint32_t)(uint8_t)Cs[(rr + 2) * GWQ_CTA_N + col] << 16)
               | ((uint32_t)(uint8_t)Cs[(rr + 3) * GWQ_CTA_N + col] << 24);
        }
        if (tok + 15 < T && gm + 15 < M) {
          *reinterpret_cast<uint4*>(
              &vtout[((size_t)(b * nh + h) * hp + d) * T + tok]) =
                  make_uint4(p[0], p[1], p[2], p[3]);
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i)
            if (gm + i < M && tok + i < T)
              vtout[((size_t)(b * nh + h) * hp + d) * T + tok + i] =
                  Cs[(row0 + i) * GWQ_CTA_N + col];
        }
      }
    }
  } else {
    // Compact physical QKV columns: hd=24 uses aligned 8-byte chunks and
    // hd=48 uses aligned 16-byte chunks. Segment boundaries and CTA boundaries
    // are multiples of this unit, so no vector crosses a Q/K/V segment.
    const int unit = (hd % 16 == 0) ? 16 : 8;
    const int chunks = GWQ_CTA_N / unit;
    const int qk_work = GWQ_CTA_M * chunks;
    for (int idx = t; idx < qk_work; idx += blockDim.x) {
      const int row = idx / chunks, ch = idx % chunks;
      const int gm = m0 + row, gc = n0 + ch * unit;
      if (gm < M && gc < n_out) {
        const int seg = gc / hd, h = seg / 3, sel = seg % 3;
        const int d = gc - seg * hd;
        const int b = gm / T, tok = gm - b * T;
        if (sel == 0) {
          if (unit == 16)
            *reinterpret_cast<uint4*>(&qout[((size_t)gm * nh + h) * hd + d]) =
                *reinterpret_cast<const uint4*>(&Cs[row * GWQ_CTA_N + ch * unit]);
          else
            *reinterpret_cast<int2*>(&qout[((size_t)gm * nh + h) * hd + d]) =
                *reinterpret_cast<const int2*>(&Cs[row * GWQ_CTA_N + ch * unit]);
        } else if (sel == 1) {
          if (unit == 16)
            *reinterpret_cast<uint4*>(
                &kout[((size_t)(b * nh + h) * T + tok) * hp + d]) =
                    *reinterpret_cast<const uint4*>(
                        &Cs[row * GWQ_CTA_N + ch * unit]);
          else
            *reinterpret_cast<int2*>(
                &kout[((size_t)(b * nh + h) * T + tok) * hp + d]) =
                    *reinterpret_cast<const int2*>(
                        &Cs[row * GWQ_CTA_N + ch * unit]);
        }
      }
    }
    // V transpose: one uint4 global store per 16-token column fragment.
    constexpr int TOKEN_CHUNKS = GWQ_CTA_M / 16;
    const int v_work = GWQ_CTA_N * TOKEN_CHUNKS;
    for (int idx = t; idx < v_work; idx += blockDim.x) {
      const int col = idx / TOKEN_CHUNKS, tc = idx % TOKEN_CHUNKS;
      const int gc = n0 + col, row0 = tc * 16, gm = m0 + row0;
      if (gc < n_out && gm < M) {
        const int seg = gc / hd, h = seg / 3, sel = seg % 3;
        const int d = gc - seg * hd;
        if (sel == 2) {
          const int b = gm / T, tok = gm - b * T;
          uint32_t p[4];
#pragma unroll
          for (int g = 0; g < 4; ++g) {
            const int rr = row0 + g * 4;
            p[g] = (uint8_t)Cs[(rr + 0) * GWQ_CTA_N + col]
                 | ((uint32_t)(uint8_t)Cs[(rr + 1) * GWQ_CTA_N + col] << 8)
                 | ((uint32_t)(uint8_t)Cs[(rr + 2) * GWQ_CTA_N + col] << 16)
                 | ((uint32_t)(uint8_t)Cs[(rr + 3) * GWQ_CTA_N + col] << 24);
          }
          if (tok + 15 < T && gm + 15 < M)
            *reinterpret_cast<uint4*>(
                &vtout[((size_t)(b * nh + h) * hp + d) * T + tok]) =
                    make_uint4(p[0], p[1], p[2], p[3]);
        }
      }
    }
    // Clear only K/VT destination tails; compact Q has no padding.
    if (blockIdx.x == 0 && hp > hd) {
      const int pad = hp - hd, pad_chunks = pad / 8;
      for (int idx = t; idx < GWQ_CTA_M * nh * pad_chunks;
           idx += blockDim.x) {
        const int row = idx / (nh * pad_chunks);
        const int r = idx % (nh * pad_chunks), h = r / pad_chunks;
        const int pc = r % pad_chunks, gm = m0 + row;
        if (gm < M) {
          const int b = gm / T, tok = gm - b * T;
          *reinterpret_cast<int2*>(
              &kout[((size_t)(b * nh + h) * T + tok) * hp + hd + pc * 8]) =
                  make_int2(0, 0);
        }
      }
      const int zwork = nh * pad * TOKEN_CHUNKS;
      for (int idx = t; idx < zwork; idx += blockDim.x) {
        const int h = idx / (pad * TOKEN_CHUNKS);
        const int r = idx % (pad * TOKEN_CHUNKS);
        const int pd = r / TOKEN_CHUNKS, tc = r % TOKEN_CHUNKS;
        const int gm = m0 + tc * 16;
        if (gm < M) {
          const int b = gm / T, tok = gm - b * T;
          if (tok + 15 < T && gm + 15 < M)
            *reinterpret_cast<uint4*>(
                &vtout[((size_t)(b * nh + h) * hp + hd + pd) * T + tok]) =
                    make_uint4(0, 0, 0, 0);
        }
      }
    }
  }
}

// ---- COPY of gemm_w4a4_kernel_awq ----
// =========================================================================
// int4 port of the validated Stage-3 mechanism above (gemm_w8a8_kernel_awq passed
// its int8 validation -- beat AWQ's own kernel at 4/6 real shapes, see NEXT_STEPS.md).
// No true AWQ W4A4 kernel exists to reference (llm-awq only ships W4A16, weight-only)
// -- this is generalization-by-analogy, not reference-matching, so it carries more
// risk than the int8 port. The key insight making it low(er)-risk anyway: ldmatrix
// and the swizzle formula operate purely on 16-byte chunks, dtype-agnostic; and
// m16n8k64.s4's A/B operands are exactly the same total byte footprint per warp as
// m16n8k32.s8's (512B/256B -- NVIDIA scales K inversely with element bit-width so the
// per-instruction data volume, and hence the ldmatrix footprint, stays constant
// across dtypes). So the *same* gwq_s2r_A/gwq_s2r_B helpers, the *same* swizzle, and
// the *same* GWQ_CTA_M/N/WARP_N/NUM_WARPS/STAGES/INTRIN_M/N/PACK_SIZE constants
// (all byte-space or M/N-space, unaffected by K packing) carry over unchanged. Only
// three things differ: the mma primitive (m16n8k64.s4 instead of m16n8k32.s8), the
// global row stride (Kb=K/2 packed bytes instead of K), and nkt (Kb/GWQ_CTA_K instead
// of K/GWQ_CTA_K -- GWQ_CTA_K=64 is already a BYTE quantity, so this just falls out).
// Correctness is NOT assumed from this reasoning -- verified against golden gemm_w4a4
// same as every other change in this file (bit-identical rel_err check required).
// =========================================================================
// `n_out`: unpadded output width/stride (see gemm_w8a8_kernel_awq). n_out==N -> dense store.
static __global__ void gemm_w4a4_kernel_awq(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     const float* __restrict__ w_scale, float a_scale,
                                     __half* __restrict__ C, int M, int N, int Kb, int n_out,
                                     const __half* __restrict__ bias, const __half* __restrict__ residual,
                                     __half* __restrict__ o_hat = nullptr,
                                     // Optional DEVICE a_scale. The MoDiff path's delta scale is
                                     // computed on device per call; passing it by value would need a
                                     // .item() sync per linear per step (42 x 200 = 8400 syncs, which
                                     // cost the conv path ~5 ms/step when it made the same mistake).
                                     const float* __restrict__ a_scale_ptr = nullptr) {
  if (a_scale_ptr) a_scale = *a_scale_ptr;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.y * GWQ_CTA_M, n0 = blockIdx.x * GWQ_CTA_N;
  const int warp_offset_n = warp * GWQ_WARP_N;
  __shared__ int8_t As[GWQ_STAGES][GWQ_CTA_M * GWQ_CTA_K];
  __shared__ int8_t Bs[GWQ_STAGES][GWQ_CTA_N * GWQ_CTA_K];
  constexpr int MI = GWQ_CTA_M / GWQ_INTRIN_M, NJ = GWQ_WARP_N / GWQ_INTRIN_N;
  int acc[MI][NJ][8];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[i][j][k] = 0;
  const int nkt = Kb / GWQ_CTA_K;   // GWQ_CTA_K is a BYTE quantity (64) -- Kb is packed-byte K/2
#define GWQ4_LOAD(kt, buf)                                                                          \
  for (int c = t; c < GWQ_CTA_M * (GWQ_CTA_K / 16); c += blockDim.x) {                              \
    int r = c / (GWQ_CTA_K / 16), off16 = c % (GWQ_CTA_K / 16);                                     \
    int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                                      \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * GWQ_CTA_K + off_swz]),                           \
                       (const uint4*)(A + (size_t)(m0 + r) * Kb + (kt) + off16 * 16), (m0 + r) < M); \
  }                                                                                                   \
  for (int c = t; c < GWQ_CTA_N * (GWQ_CTA_K / 16); c += blockDim.x) {                               \
    int r = c / (GWQ_CTA_K / 16), off16 = c % (GWQ_CTA_K / 16);                                     \
    int off_swz = (off16 ^ ((r / 2) & 3)) * 16;                                                      \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * GWQ_CTA_K + off_swz]),                           \
                       (const uint4*)(B + (size_t)(n0 + r) * Kb + (kt) + off16 * 16), (n0 + r) < N); \
  }
#pragma unroll
  for (int s = 0; s < GWQ_STAGES - 1; ++s) { if (s < nkt) { GWQ4_LOAD(s * GWQ_CTA_K, s); } __pipeline_commit(); }
  __pipeline_wait_prior(GWQ_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % GWQ_STAGES;
#pragma unroll
    for (int k01 = 0; k01 < GWQ_CTA_K / GWQ_INTRIN_K; ++k01) {
      int8_t Afrag[MI * 16], Bfrag[NJ * 16];
      gwq_s2r_A(&As[buf][0], Afrag, lane, k01);
      gwq_s2r_B(&Bs[buf][0], Bfrag, lane, warp_offset_n, k01);
#pragma unroll
      for (int ii = 0; ii < MI; ++ii)
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
          modiff_mma_m16n8k64_s4(acc[ii][jj], Afrag + ii * 16, Bfrag + jj * 16);
          modiff_mma_m16n8k64_s4(acc[ii][jj] + 4, Afrag + ii * 16, Bfrag + jj * 16 + 8);
        }
    }
    int li = i + GWQ_STAGES - 1;
    if (li < nkt) { GWQ4_LOAD(li * GWQ_CTA_K, li % GWQ_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(GWQ_STAGES - 2);
    __syncthreads();
  }
#undef GWQ4_LOAD
#pragma unroll
  for (int i = 0; i < MI; ++i) {
    int row0 = m0 + i * GWQ_INTRIN_M + gid, row1 = row0 + 8;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int col0 = n0 + warp_offset_n + j * GWQ_INTRIN_N + tig * 2, col1 = col0 + 8;
      float s00 = a_scale * w_scale[col0], s01 = a_scale * w_scale[col0 + 1];
      float s10 = a_scale * w_scale[col1], s11 = a_scale * w_scale[col1 + 1];
      int* accv = acc[i][j];
      bool c0 = col0 < n_out, c1 = col1 < n_out;  // n_out even -> col0<n_out guards [col0,col0+1]
      if (row0 < M) {
        if (c0) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col0, col0, accv[0] * s00, accv[1] * s01, o_hat);
        if (c1) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col1, col1, accv[4] * s10, accv[5] * s11, o_hat);
      }
      if (row1 < M) {
        if (c0) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col0, col0, accv[2] * s00, accv[3] * s01, o_hat);
        if (c1) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col1, col1, accv[6] * s10, accv[7] * s11, o_hat);
      }
    }
  }
}

// ==== MoDiff o_hat-accumulating host entry points (moved from the baseline file) ====

// MoDiff dual output: o_hat_t = A(Q(delta)) + o_hat_{t+1} (fp16 state, Eq 9), and the returned int8
// is Q_out(o_hat_t + bias) for the next consumer -- one pass instead of an fp16 materialize plus a
// separate quantize. Written for the attention qkv, whose consumer (flash) wants int8 while MoDiff
// wants the fp16 state kept: the two used to be mutually exclusive, which is what forced the whole
// GN->qkv->flash chain to fall back (docs/delta_clip_2026-08-06).
torch::Tensor gemm_w8a8_awq_o_hat_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                        torch::Tensor a_scale, int64_t n_out, torch::Tensor o_hat,
                                        torch::Tensor bias, torch::Tensor inv_out_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  TORCH_CHECK(o_hat.dtype() == torch::kFloat16, "o_hat must be fp16");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K && N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0, "shape/pad");
  TORCH_CHECK(o_hat.numel() == (int64_t)M * n_out, "o_hat must be [M, n_out]");
  // [M, n_out], NOT [M, N]. The store loop writes `C[(m0+row) * n_out + n0 + col]` -- an UNPADDED row
  // stride -- so allocating at the padded width N made the two disagree whenever n_out != N, and the
  // codes scattered. Measured before the fix (integration/tests/test_qkv_o_hat_out_i8.py): at
  // n_out=576 padded to 640, 81.6% of codes wrong with max|diff| 254, while the two 128-aligned
  // shapes were correct to within 1. o_hat was right throughout, because it already indexed at
  // n_out. Allocating unpadded also matches gemm_w8a8_awq_o_hat's return shape, so the caller needs
  // no slice and there is no uninitialised tail to slice off.
  auto C = torch::empty({M, n_out}, torch::TensorOptions().dtype(torch::kChar).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  // a_scale is a 1-element DEVICE tensor here, as on the o_hat path: taking it by value would cost a
  // .item() sync per linear per step. The kernel signature takes a float, so read it on device via
  // the same trick the o_hat GEMM uses -- pass it through the existing float parameter is NOT
  // possible, so this wrapper requires the caller to have it on device and dereferences once.
  TORCH_CHECK(a_scale.numel() == 1 && a_scale.dtype() == torch::kFloat32,
              "a_scale must be a 1-element float32 device tensor");
  gemm_w8a8_kernel_awq_out_i8<0><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      0.f, inv_out_scale.contiguous().data_ptr<float>(),
      bias.numel() > 0 ? (const __half*)bias.contiguous().data_ptr<at::Half>() : (const __half*)nullptr,
      C.data_ptr<int8_t>(), M, N, K, (int)n_out,
      nullptr, nullptr, nullptr, 0, 0, 0, 0,
      (__half*)o_hat.data_ptr<at::Half>(),
      a_scale.contiguous().data_ptr<float>());
  return C;
}

//   Op:       Linear W8A8 GEMM + fused bias + optional residual (attention proj / qkv path)
//   Inputs:   A int8 [M,K]; B int8 [N,K] (N%128); w_scale f32 [N]; a_scale f64; n_out i64 (even, <=N);
//             bias fp16 [n_out] or empty; residual fp16 [M,n_out] or empty
//   Outputs:  C fp16 [M,n_out] = dequant(A.B)*a_scale*w_scale[n] + bias[n] + residual[m,n]
//   Fuses:    dequant + bias add + residual add in the GEMM epilogue -> removes the separate
//             `out + bias` and `x + proj(out)` elementwise-add kernels (the residual-add glue).
// -----------------------------------------------------------------------------
//   Op:       Linear W8A8 GEMM + MoDiff o_hat accumulate (attention qkv / proj on the modulated path)
//   Inputs:   A int8 [M,K] the quantized DELTA codes; B int8 [N,K]; w_scale fp32 [N]; a_scale the
//             reciprocal of the scale that quantized A; n_out; o_hat fp16 [M,n_out] modified in
//             place; residual fp16 [M,n_out] or empty
//   Outputs:  o_hat advanced to o_hat_t (Eq 9, bias- and residual-free), and RETURNS
//             o_hat_t + bias + residual -- what the next layer consumes.
//   Computes: o_hat_t = A(Q(a_t - a_hat_{t+1})) * a_scale * w_scale[n] + o_hat_{t+1}
//   Fuses:    the accumulate into the GEMM epilogue, replacing three full-tensor PyTorch launches
//             per linear per step (dequant, add, store). That overhead -- measured +10.9 ms/step at
//             batch 8 -- is the ONLY reason MoDiff on the Linear layers was off by default; the
//             method itself is correct there since Bug 2 was fixed (int4 latent relL2 0.4571 ->
//             0.4220 with it on).
//   Constraints: bias and residual are applied to the RETURNED tensor only, never to o_hat -- see
//             gwq_store2, where the accumulate deliberately precedes them.
// -----------------------------------------------------------------------------
torch::Tensor gemm_w8a8_awq_o_hat(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                  torch::Tensor a_scale, int64_t n_out, torch::Tensor o_hat,
                                  torch::Tensor residual, torch::Tensor bias) {
  TORCH_CHECK(a_scale.numel() == 1 && a_scale.dtype() == torch::kFloat32,
              "a_scale must be a 1-element fp32 DEVICE tensor (no host sync on the hot path)");
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0, "N%128==0, K%64==0");
  TORCH_CHECK(n_out > 0 && n_out <= N && n_out % 2 == 0, "n_out even in (0,N]");
  TORCH_CHECK(o_hat.dtype() == torch::kHalf && o_hat.numel() == (int64_t)M * n_out,
              "o_hat must be fp16 [M,n_out]");
  __half* op = reinterpret_cast<__half*>(o_hat.contiguous().data_ptr<at::Half>());
  const __half* rp = nullptr;
  const __half* bp = nullptr;
  if (residual.numel()) {
    TORCH_CHECK(residual.numel() == (int64_t)M * n_out && residual.dtype() == torch::kHalf,
                "residual fp16 [M,n_out]");
    rp = reinterpret_cast<const __half*>(residual.contiguous().data_ptr<at::Half>());
  }
  if (bias.numel()) {
    TORCH_CHECK(bias.numel() == n_out && bias.dtype() == torch::kHalf, "bias fp16 [n_out]");
    bp = reinterpret_cast<const __half*>(bias.contiguous().data_ptr<at::Half>());
  }
  // C is ALWAYS produced: it is o_hat_t + bias + residual, i.e. what the next layer consumes.
  // Returning o_hat itself would force the caller to add bias to the state tensor in place.
  auto C = torch::empty({M, (int)n_out},
                        torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  __half* cp = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      0.f, cp, M, N, K, (int)n_out, bp, rp, op,
      a_scale.contiguous().data_ptr<float>());
  C10_CUDA_CHECK(cudaGetLastError());
  return C;
}

//   Op:       Linear W4A4 GEMM (production int4 Linear entry point)
//   Inputs:   A int8 [M, K/2] packed int4 (2 per byte, quantized activation); B int8 [N, K/2] packed
//             int4 (weight, one row per output channel); w_scale f32 [N] (per-channel); a_scale f64
//             (per-tensor scalar); K i64 = logical (unpacked) K
//   Outputs:  C fp16 [M, N]
//   Computes: C[m,n] = (A[m,:].B[n,:] over K int4 elems) * a_scale * w_scale[n]
//   Fuses:    fp16 dequant epilogue (int32 accumulator * a_scale * w_scale[n] -> fp16)
//   Constraints: A/B packed-int4 CUDA (kChar); N%128==0, K%128==0 (pad B/w_scale at the call site)
//   vs fp16:  W8A8 GEMM ~1.46x / W4A4 ~1.83x vs fp16 F.linear on churches qkv/proj shapes (b128),
//             GEMM-only i.e. activation-quantize fused upstream. With a standalone activation quantize
//             the win is erased (int8 ~0.99x, int4 ~0.78x). Wins biggest at K>=384 (int4 up to 2.66x);
//             weakest at K=192.
// INT4 twin. Same contract; Kb is the packed K (two int4 per byte), handled by the kernel.
torch::Tensor gemm_w4a4_awq_o_hat(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                  torch::Tensor a_scale, int64_t n_out, torch::Tensor o_hat,
                                  torch::Tensor residual, torch::Tensor bias) {
  TORCH_CHECK(a_scale.numel() == 1 && a_scale.dtype() == torch::kFloat32,
              "a_scale must be a 1-element fp32 DEVICE tensor (no host sync on the hot path)");
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), Kb = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == Kb, "Kb mismatch");
  TORCH_CHECK(n_out > 0 && n_out <= N && n_out % 2 == 0, "n_out even in (0,N]");
  TORCH_CHECK(o_hat.dtype() == torch::kHalf && o_hat.numel() == (int64_t)M * n_out,
              "o_hat must be fp16 [M,n_out]");
  __half* op = reinterpret_cast<__half*>(o_hat.contiguous().data_ptr<at::Half>());
  const __half* rp = nullptr;
  const __half* bp = nullptr;
  if (residual.numel()) {
    TORCH_CHECK(residual.numel() == (int64_t)M * n_out && residual.dtype() == torch::kHalf,
                "residual fp16 [M,n_out]");
    rp = reinterpret_cast<const __half*>(residual.contiguous().data_ptr<at::Half>());
  }
  if (bias.numel()) {
    TORCH_CHECK(bias.numel() == n_out && bias.dtype() == torch::kHalf, "bias fp16 [n_out]");
    bp = reinterpret_cast<const __half*>(bias.contiguous().data_ptr<at::Half>());
  }
  // C is ALWAYS produced: it is o_hat_t + bias + residual, i.e. what the next layer consumes.
  // Returning o_hat itself would force the caller to add bias to the state tensor in place.
  auto C = torch::empty({M, (int)n_out},
                        torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  __half* cp = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      0.f, cp, M, N, Kb, (int)n_out, bp, rp, op,
      a_scale.contiguous().data_ptr<float>());
  C10_CUDA_CHECK(cudaGetLastError());
  return C;
}
