// =========================================================================
// Weight+activation int GEMM for the UNet Linear layers (W8A8 / W4A4), static
// scales, AWQ-tiling scheme (mma.m16n8k32.s8 / m16n8k64.s4 tensor cores;
// per-channel weight scale x static per-tensor activation scale).
//
//   C[M,N] fp16 = (A[M,K] int8 . B[N,K]^T int8)  * a_scale (scalar) * w_scale[n]
//
// This file provides the production Linear GEMM backend: the AWQ-tiling ports
// `gemm_w8a8_awq` / `gemm_w4a4_awq` (CTA_M=CTA_N=128, CTA_K=64, WARP_N=32,
// 4 warps, GWQ_STAGES=3; real ldmatrix.m8n8.x4 + XOR bank-swizzle
// `col ^ ((row/2)&3)` shared->register loads), plus the fused fp16->int8 /
// packed-int4 activation-quantize helpers (`quantize_act_int8` /
// `quantize_act_int4_pack`). Both GEMMs require N%128==0 and K%64 (int8) /
// K%128 (int4); callers pad the weight/scale offline and the activation at call
// time (see integration/kernels/wxax_linear.py).
//
// History: an earlier hand-written family (gemm_w8a8/gemm_w4a4 + int8-output
// variants, a templated MT/WideK/GW_STAGES kernel) lived here and was the
// backend before the ports beat it at every shape. It was retired 2026-07-18;
// a non-compiled copy is kept at csrc/kernels/backup/ for reference. See
// docs/quant_speedup_vs_fp16_2026-07-16/ (SESSION_REPORT / NEXT_STEPS) for the
// measurements that justified consolidating on the ports.
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "common.cuh"
#include "mma_int8.cuh"

// ---- fused fp16 -> int8 / packed-int4 activation quantize (static per-tensor scale) ----
__global__ void quant_act_int8_kernel(const __half* __restrict__ x, int8_t* __restrict__ out,
                                      float inv_scale, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int q = __float2int_rn(__half2float(x[i]) * inv_scale);
    out[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
}

__global__ void quant_act_int4_pack_kernel(const __half* __restrict__ x, int8_t* __restrict__ out,
                                           float inv_scale, int K, long nout) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;   // one output byte = 2 int4
  if (i < nout) {
    long base = (i / (K / 2)) * K + (i % (K / 2)) * 2;
    int q0 = __float2int_rn(__half2float(x[base]) * inv_scale);     q0 = q0 > 7 ? 7 : (q0 < -7 ? -7 : q0);
    int q1 = __float2int_rn(__half2float(x[base + 1]) * inv_scale); q1 = q1 > 7 ? 7 : (q1 < -7 ? -7 : q1);
    out[i] = (int8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
  }
}

//   Op:       fp16 -> int8 activation quantize (static per-tensor scale)
//   Inputs:   x fp16 [any shape; flattened to N=numel]; a_scale f64 (per-tensor scalar)
//   Outputs:  int8 [same shape as x]
//   Computes: out[i] = clamp(round(x[i] / a_scale), -127, 127)   (symmetric, no zero-point)
//   Fuses:    none (standalone activation-quantize helper feeding the W8A8 GEMM path)
//   Constraints: x reinterpreted as __half (must be fp16); no shape/size constraint
//   vs fp16:  n/a (quantization helper)
torch::Tensor quantize_act_int8(torch::Tensor x, double a_scale) {
  x = x.contiguous(); long n = x.numel();
  auto out = torch::empty_like(x, torch::TensorOptions().dtype(torch::kChar).device(x.device()));
  int T = 256; long blocks = (n + T - 1) / T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  quant_act_int8_kernel<<<blocks, T, 0, s>>>(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                                             out.data_ptr<int8_t>(), 1.f / (float)a_scale, n);
  return out;
}

//   Op:       fp16 -> packed-int4 activation quantize (static per-tensor scale)
//   Inputs:   x fp16 [M, K]; a_scale f64 (per-tensor scalar)
//   Outputs:  int8 [M, K/2] packed int4 (low nibble = even col, high nibble = odd col)
//   Computes: q = clamp(round(x / a_scale), -7, 7); out[m,k] = (q_even & 0xF) | ((q_odd & 0xF) << 4)
//   Fuses:    int4 pack (adjacent K-pairs packed into one byte; layout matches gemm_w4a4_awq's A operand)
//   Constraints: x is 2D [M, K] fp16 with K even; reinterpreted as __half
//   vs fp16:  n/a (quantization helper)
torch::Tensor quantize_act_int4_pack(torch::Tensor x, double a_scale) {
  x = x.contiguous(); int M = x.size(0), K = x.size(1); long nout = (long)M * (K / 2);
  auto out = torch::empty({M, K / 2}, torch::TensorOptions().dtype(torch::kChar).device(x.device()));
  int T = 256; long blocks = (nout + T - 1) / T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  quant_act_int4_pack_kernel<<<blocks, T, 0, s>>>(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                                                  out.data_ptr<int8_t>(), 1.f / (float)a_scale, K, nout);
  return out;
}

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

// `n_out` is the output row stride / valid column count. When n_out==N the store is dense (original
// behavior); when n_out<N (e.g. N padded up to CTA_N=128 but the real out_features is smaller) the
// kernel writes the unpadded [M, n_out] result directly, skipping the padded columns -- this removes
// the downstream slice+`.contiguous()` copy on padded qkv/proj GEMM outputs.
// Fused epilogue store: writes two adjacent fp16 columns (col,col+1) at C[idx], optionally adding a
// per-column bias[col] and an elementwise residual[idx] in fp32 before the half cast. bias/residual
// are nullptr for the plain dequant path (behavior identical to the original store).
__device__ __forceinline__ void gwq_store2(__half* C, const __half* bias, const __half* residual,
                                           size_t idx, int col, float v0, float v1) {
  if (bias) { v0 += __half2float(bias[col]); v1 += __half2float(bias[col + 1]); }
  if (residual) { float2 r = __half22float2(*(const __half2*)&residual[idx]); v0 += r.x; v1 += r.y; }
  *(__half2*)&C[idx] = __halves2half2(__float2half(v0), __float2half(v1));
}

__global__ void gemm_w8a8_kernel_awq(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     const float* __restrict__ w_scale, float a_scale,
                                     __half* __restrict__ C, int M, int N, int K, int n_out,
                                     const __half* __restrict__ bias, const __half* __restrict__ residual) {
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
        if (c0) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col0, col0, accv[0] * s00, accv[1] * s01);
        if (c1) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col1, col1, accv[4] * s10, accv[5] * s11);
      }
      if (row1 < M) {
        if (c0) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col0, col0, accv[2] * s00, accv[3] * s01);
        if (c1) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col1, col1, accv[6] * s10, accv[7] * s11);
      }
    }
  }
}

// ---- int8-OUTPUT variant (output-fusion + bandwidth fix) ----
// Identical (validated) mainloop to gemm_w8a8_kernel_awq, but the epilogue requantizes the fp32
// accumulator to INT8 instead of fp16 -- halving the dominant output write (M*N*2 -> M*N*1 bytes),
// which the roofline profile showed is ~86% of this GEMM's memory traffic and the reason it sits at
// 21-68% of the fp16-output roofline. `inv_out_scale[N]` = 127/per-column-absmax (calibrated); the
// downstream op dequants with out_scale = 1/inv_out_scale (or consumes int8 directly). Stores are
// packed to 16-bit (two adjacent int8 columns per store) so the write stays coalesced.
__global__ void gemm_w8a8_kernel_awq_out_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                            const float* __restrict__ w_scale, float a_scale,
                                            const float* __restrict__ inv_out_scale,
                                            int8_t* __restrict__ C, int M, int N, int K) {
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
      int* accv = acc[i][j];
      Cs[r0 * GWQ_CTA_N + c0] = q8(accv[0] * s00); Cs[r0 * GWQ_CTA_N + c0 + 1] = q8(accv[1] * s01);
      Cs[r0 * GWQ_CTA_N + c1] = q8(accv[4] * s10); Cs[r0 * GWQ_CTA_N + c1 + 1] = q8(accv[5] * s11);
      Cs[r1 * GWQ_CTA_N + c0] = q8(accv[2] * s00); Cs[r1 * GWQ_CTA_N + c0 + 1] = q8(accv[3] * s01);
      Cs[r1 * GWQ_CTA_N + c1] = q8(accv[6] * s10); Cs[r1 * GWQ_CTA_N + c1 + 1] = q8(accv[7] * s11);
    }
  }
  __syncthreads();
  for (int idx = t; idx < GWQ_CTA_M * GWQ_CTA_N / 16; idx += blockDim.x) {
    int rc = idx * 16, row = rc / GWQ_CTA_N, col = rc % GWQ_CTA_N;
    if (m0 + row < M)
      *(uint4*)&C[(size_t)(m0 + row) * N + n0 + col] = *(const uint4*)&Cs[row * GWQ_CTA_N + col];
  }
}

//   Op:       Linear W8A8 GEMM (int8 output, output-fusion variant)
//   Inputs:   A int8 [M, K] (quantized activation); B int8 [N, K] (quantized weight, one row per
//             output channel); w_scale f32 [N] (per-channel); a_scale f64 (per-tensor scalar);
//             inv_out_scale f32 [N] = 127/per-column-absmax (calibrated)
//   Outputs:  C int8 [M, N]
//   Computes: C[m,n] = clamp(round( (A[m,:].B[n,:]) * a_scale * w_scale[n] * inv_out_scale[n] ), -127, 127)
//             i.e. int32 accumulator dequantized then requantized to int8; downstream dequants with
//             out_scale = 1/inv_out_scale (or consumes int8 directly).
//   Fuses:    int8-output requant epilogue -- halves the output write (M*N*2 -> M*N*1 bytes) via a
//             smem-staged, 128-bit-coalesced store; a_scale/w_scale/inv_out_scale folded into one factor.
//   Constraints: A/B int8 CUDA; N%128==0, K%64==0 (pad B/w_scale at the call site)
//   vs fp16:  W8A8 GEMM ~1.46x / W4A4 ~1.83x vs fp16 F.linear on churches qkv/proj shapes (b128),
//             GEMM-only i.e. activation-quantize fused upstream. With a standalone activation quantize
//             the win is erased (int8 ~0.99x, int4 ~0.78x). Wins biggest at K>=384 (int4 up to 2.66x);
//             weakest at K=192.
torch::Tensor gemm_w8a8_awq_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                   double a_scale, torch::Tensor inv_out_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K && N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0, "shape/pad");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kChar).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq_out_i8<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, inv_out_scale.contiguous().data_ptr<float>(), C.data_ptr<int8_t>(), M, N, K);
  return C;
}

//   Op:       Linear W8A8 GEMM (production int8 Linear entry point)
//   Inputs:   A int8 [M, K] (quantized activation); B int8 [N, K] (quantized weight, one row per
//             output channel); w_scale f32 [N] (per-channel); a_scale f64 (per-tensor scalar)
//   Outputs:  C fp16 [M, N]
//   Computes: C[m,n] = (A[m,:].B[n,:] int8) * a_scale * w_scale[n]
//   Fuses:    fp16 dequant epilogue (int32 accumulator * a_scale * w_scale[n] -> fp16)
//   Constraints: A/B int8 CUDA; N%128==0, K%64==0 (pad B/w_scale at the call site)
//   vs fp16:  W8A8 GEMM ~1.46x / W4A4 ~1.83x vs fp16 F.linear on churches qkv/proj shapes (b128),
//             GEMM-only i.e. activation-quantize fused upstream. With a standalone activation quantize
//             the win is erased (int8 ~0.99x, int4 ~0.78x). Wins biggest at K>=384 (int4 up to 2.66x);
//             weakest at K=192.
torch::Tensor gemm_w8a8_awq(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0, "gemm_w8a8_awq needs N%128==0 (pad B/w_scale at the call site)");
  TORCH_CHECK(K % GWQ_CTA_K == 0, "gemm_w8a8_awq needs K%64==0");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K, N, nullptr, nullptr);
  return C;
}

//   Op:       Linear W8A8 GEMM (unpadded-output variant)
//   Inputs:   A int8 [M, K]; B int8 [N, K] (N padded to %128==0); w_scale f32 [N] (per-channel);
//             a_scale f64 (per-tensor scalar); n_out i64 = real out_features (even, in (0, N])
//   Outputs:  C fp16 [M, n_out]  (written with n_out row-stride; padded columns skipped)
//   Computes: C[m,n] = (A[m,:].B[n,:] int8) * a_scale * w_scale[n]  for n < n_out
//   Fuses:    fp16 dequant epilogue + unpadded store (removes the downstream out[:, :n_out].contiguous()
//             slice-copy on padded qkv/proj GEMMs)
//   Constraints: A/B int8 CUDA; N%128==0, K%64==0; n_out even and in (0, N]
//   vs fp16:  W8A8 GEMM ~1.46x / W4A4 ~1.83x vs fp16 F.linear on churches qkv/proj shapes (b128),
//             GEMM-only i.e. activation-quantize fused upstream. With a standalone activation quantize
//             the win is erased (int8 ~0.99x, int4 ~0.78x). Wins biggest at K>=384 (int4 up to 2.66x);
//             weakest at K=192.
torch::Tensor gemm_w8a8_awq_nout(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t n_out) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0, "gemm_w8a8_awq_nout needs padded N%128==0");
  TORCH_CHECK(K % GWQ_CTA_K == 0, "gemm_w8a8_awq_nout needs K%64==0");
  TORCH_CHECK(n_out > 0 && n_out <= N && n_out % 2 == 0, "n_out must be even and in (0, N]");
  auto C = torch::empty({M, (int)n_out}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K, (int)n_out, nullptr, nullptr);
  return C;
}

//   Op:       Linear W8A8 GEMM + fused bias + optional residual (attention proj / qkv path)
//   Inputs:   A int8 [M,K]; B int8 [N,K] (N%128); w_scale f32 [N]; a_scale f64; n_out i64 (even, <=N);
//             bias fp16 [n_out] or empty; residual fp16 [M,n_out] or empty
//   Outputs:  C fp16 [M,n_out] = dequant(A.B)*a_scale*w_scale[n] + bias[n] + residual[m,n]
//   Fuses:    dequant + bias add + residual add in the GEMM epilogue -> removes the separate
//             `out + bias` and `x + proj(out)` elementwise-add kernels (the residual-add glue).
torch::Tensor gemm_w8a8_awq_bias_res(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
                                     int64_t n_out, torch::Tensor bias, torch::Tensor residual) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0, "N%128==0, K%64==0");
  TORCH_CHECK(n_out > 0 && n_out <= N && n_out % 2 == 0, "n_out even in (0,N]");
  const __half* bp = nullptr; const __half* rp = nullptr;
  if (bias.numel()) { TORCH_CHECK(bias.numel() == n_out && bias.dtype() == torch::kHalf, "bias fp16 [n_out]"); bp = reinterpret_cast<const __half*>(bias.contiguous().data_ptr<at::Half>()); }
  if (residual.numel()) { TORCH_CHECK(residual.numel() == (int64_t)M * n_out && residual.dtype() == torch::kHalf, "residual fp16 [M,n_out]"); rp = reinterpret_cast<const __half*>(residual.contiguous().data_ptr<at::Half>()); }
  auto C = torch::empty({M, (int)n_out}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K, (int)n_out, bp, rp);
  return C;
}

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
__global__ void gemm_w4a4_kernel_awq(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     const float* __restrict__ w_scale, float a_scale,
                                     __half* __restrict__ C, int M, int N, int Kb, int n_out,
                                     const __half* __restrict__ bias, const __half* __restrict__ residual) {
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
        if (c0) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col0, col0, accv[0] * s00, accv[1] * s01);
        if (c1) gwq_store2(C, bias, residual, (size_t)row0 * n_out + col1, col1, accv[4] * s10, accv[5] * s11);
      }
      if (row1 < M) {
        if (c0) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col0, col0, accv[2] * s00, accv[3] * s01);
        if (c1) gwq_store2(C, bias, residual, (size_t)row1 * n_out + col1, col1, accv[6] * s10, accv[7] * s11);
      }
    }
  }
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
torch::Tensor gemm_w4a4_awq(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B packed int4 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), N = B.size(0);
  TORCH_CHECK(A.size(1) == K / 2 && B.size(1) == K / 2, "packed K/2 mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0, "gemm_w4a4_awq needs N%128==0 (pad B/w_scale at the call site)");
  TORCH_CHECK(K % (GWQ_CTA_K * 2) == 0, "gemm_w4a4_awq needs K%128==0 (GWQ_CTA_K=64 packed bytes = 128 logical int4 elements)");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int Kb = (int)K / 2;
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, Kb, N, nullptr, nullptr);
  return C;
}

//   Op:       Linear W4A4 GEMM (unpadded-output variant)
//   Inputs:   A int8 [M, K/2] packed int4; B int8 [N, K/2] packed int4 (N padded to %128==0);
//             w_scale f32 [N] (per-channel); a_scale f64 (per-tensor scalar); K i64 = logical K;
//             n_out i64 = real out_features (even, in (0, N])
//   Outputs:  C fp16 [M, n_out]  (written with n_out row-stride; padded columns skipped)
//   Computes: C[m,n] = (A[m,:].B[n,:] over K int4) * a_scale * w_scale[n]  for n < n_out
//   Fuses:    fp16 dequant epilogue + unpadded store (removes the downstream slice+.contiguous() copy
//             on padded qkv/proj int4 GEMMs)
//   Constraints: A/B packed-int4 CUDA (kChar); N%128==0, K%128==0; n_out even and in (0, N]
//   vs fp16:  W8A8 GEMM ~1.46x / W4A4 ~1.83x vs fp16 F.linear on churches qkv/proj shapes (b128),
//             GEMM-only i.e. activation-quantize fused upstream. With a standalone activation quantize
//             the win is erased (int8 ~0.99x, int4 ~0.78x). Wins biggest at K>=384 (int4 up to 2.66x);
//             weakest at K=192.
torch::Tensor gemm_w4a4_awq_nout(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K, int64_t n_out) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B packed int4 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), N = B.size(0);
  TORCH_CHECK(A.size(1) == K / 2 && B.size(1) == K / 2, "packed K/2 mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0, "gemm_w4a4_awq_nout needs padded N%128==0");
  TORCH_CHECK(K % (GWQ_CTA_K * 2) == 0, "gemm_w4a4_awq_nout needs K%128==0");
  TORCH_CHECK(n_out > 0 && n_out <= N && n_out % 2 == 0, "n_out must be even and in (0, N]");
  auto C = torch::empty({M, (int)n_out}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int Kb = (int)K / 2;
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, Kb, (int)n_out, nullptr, nullptr);
  return C;
}

//   Op:       Linear W4A4 GEMM + fused bias + optional residual (int4 attention proj / qkv path)
//   Inputs:   A int8 [M,K/2] packed int4; B int8 [N,K/2] packed int4 (N%128); w_scale f32 [N]; a_scale
//             f64; K i64 (logical); n_out i64 (even, <=N); bias fp16 [n_out] or empty; residual fp16
//             [M,n_out] or empty
//   Outputs:  C fp16 [M,n_out] = dequant(A.B)*a_scale*w_scale[n] + bias[n] + residual[m,n]
//   Fuses:    dequant + bias + residual in the epilogue (same as gemm_w8a8_awq_bias_res, int4 weights)
torch::Tensor gemm_w4a4_awq_bias_res(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
                                     int64_t K, int64_t n_out, torch::Tensor bias, torch::Tensor residual) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B packed int4 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), N = B.size(0);
  TORCH_CHECK(A.size(1) == K / 2 && B.size(1) == K / 2, "packed K/2 mismatch");
  TORCH_CHECK(N % GWQ_CTA_N == 0 && K % (GWQ_CTA_K * 2) == 0, "N%128==0, K%128==0");
  TORCH_CHECK(n_out > 0 && n_out <= N && n_out % 2 == 0, "n_out even in (0,N]");
  const __half* bp = nullptr; const __half* rp = nullptr;
  if (bias.numel()) { TORCH_CHECK(bias.numel() == n_out && bias.dtype() == torch::kHalf, "bias fp16 [n_out]"); bp = reinterpret_cast<const __half*>(bias.contiguous().data_ptr<at::Half>()); }
  if (residual.numel()) { TORCH_CHECK(residual.numel() == (int64_t)M * n_out && residual.dtype() == torch::kHalf, "residual fp16 [M,n_out]"); rp = reinterpret_cast<const __half*>(residual.contiguous().data_ptr<at::Half>()); }
  auto C = torch::empty({M, (int)n_out}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int Kb = (int)K / 2;
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, Kb, (int)n_out, bp, rp);
  return C;
}

// ---- int4 GEMM, INT8-OUTPUT variant (same output-fusion fix as gemm_w8a8_awq_out_i8) ----
__global__ void gemm_w4a4_kernel_awq_out_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                            const float* __restrict__ w_scale, float a_scale,
                                            const float* __restrict__ inv_out_scale,
                                            int8_t* __restrict__ C, int M, int N, int Kb) {
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
  const int nkt = Kb / GWQ_CTA_K;
#define GWQ4O_LOAD(kt, buf)                                                                         \
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
  for (int s = 0; s < GWQ_STAGES - 1; ++s) { if (s < nkt) { GWQ4O_LOAD(s * GWQ_CTA_K, s); } __pipeline_commit(); }
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
    if (li < nkt) { GWQ4O_LOAD(li * GWQ_CTA_K, li % GWQ_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(GWQ_STAGES - 2);
    __syncthreads();
  }
#undef GWQ4O_LOAD
  auto q8 = [](float v) -> int { int x = __float2int_rn(v); return x > 127 ? 127 : (x < -127 ? -127 : x); };
  __syncthreads();
  int8_t* Cs = &As[0][0];
#pragma unroll
  for (int i = 0; i < MI; ++i) {
    int r0 = i * GWQ_INTRIN_M + gid, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int c0 = warp_offset_n + j * GWQ_INTRIN_N + tig * 2, c1 = c0 + 8;
      int gc0 = n0 + c0, gc1 = n0 + c1;
      float s00 = a_scale * w_scale[gc0] * inv_out_scale[gc0], s01 = a_scale * w_scale[gc0 + 1] * inv_out_scale[gc0 + 1];
      float s10 = a_scale * w_scale[gc1] * inv_out_scale[gc1], s11 = a_scale * w_scale[gc1 + 1] * inv_out_scale[gc1 + 1];
      int* accv = acc[i][j];
      Cs[r0 * GWQ_CTA_N + c0] = q8(accv[0] * s00); Cs[r0 * GWQ_CTA_N + c0 + 1] = q8(accv[1] * s01);
      Cs[r0 * GWQ_CTA_N + c1] = q8(accv[4] * s10); Cs[r0 * GWQ_CTA_N + c1 + 1] = q8(accv[5] * s11);
      Cs[r1 * GWQ_CTA_N + c0] = q8(accv[2] * s00); Cs[r1 * GWQ_CTA_N + c0 + 1] = q8(accv[3] * s01);
      Cs[r1 * GWQ_CTA_N + c1] = q8(accv[6] * s10); Cs[r1 * GWQ_CTA_N + c1 + 1] = q8(accv[7] * s11);
    }
  }
  __syncthreads();
  for (int idx = t; idx < GWQ_CTA_M * GWQ_CTA_N / 16; idx += blockDim.x) {
    int rc = idx * 16, row = rc / GWQ_CTA_N, col = rc % GWQ_CTA_N;
    if (m0 + row < M)
      *(uint4*)&C[(size_t)(m0 + row) * N + n0 + col] = *(const uint4*)&Cs[row * GWQ_CTA_N + col];
  }
}

//   Op:       Linear W4A4 GEMM (int8 output, output-fusion variant)
//   Inputs:   A int8 [M, K/2] packed int4; B int8 [N, K/2] packed int4 (weight); w_scale f32 [N]
//             (per-channel); a_scale f64 (per-tensor scalar); K i64 = logical K;
//             inv_out_scale f32 [N] = 127/per-column-absmax (calibrated)
//   Outputs:  C int8 [M, N]
//   Computes: C[m,n] = clamp(round( (A[m,:].B[n,:] over K int4) * a_scale * w_scale[n] * inv_out_scale[n] ), -127, 127)
//             i.e. int32 accumulator dequantized then requantized to int8 (dequant downstream with
//             out_scale = 1/inv_out_scale).
//   Fuses:    int8-output requant epilogue -- halves the output write via a smem-staged, 128-bit-
//             coalesced store (same output-fusion fix as gemm_w8a8_awq_out_i8).
//   Constraints: A/B packed-int4 CUDA (kChar); N%128==0, K%128==0
//   vs fp16:  W8A8 GEMM ~1.46x / W4A4 ~1.83x vs fp16 F.linear on churches qkv/proj shapes (b128),
//             GEMM-only i.e. activation-quantize fused upstream. With a standalone activation quantize
//             the win is erased (int8 ~0.99x, int4 ~0.78x). Wins biggest at K>=384 (int4 up to 2.66x);
//             weakest at K=192.
torch::Tensor gemm_w4a4_awq_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                   double a_scale, int64_t K, torch::Tensor inv_out_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B packed int4 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), N = B.size(0);
  TORCH_CHECK(A.size(1) == K / 2 && B.size(1) == K / 2 && N % GWQ_CTA_N == 0 && K % (GWQ_CTA_K * 2) == 0, "shape/pad");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kChar).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int Kb = (int)K / 2;
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq_out_i8<<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, inv_out_scale.contiguous().data_ptr<float>(), C.data_ptr<int8_t>(), M, N, Kb);
  return C;
}
