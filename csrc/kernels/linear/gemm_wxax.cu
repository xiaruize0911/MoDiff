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
#include <c10/cuda/CUDAException.h>
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
// QKV_LAYOUT: 0=ordinary contiguous output, 1=per-head padded direct layouts,
// 2=compact QKV columns with padded K/VT only in the destination.
template <int QKV_LAYOUT = 0>
__global__ void gemm_w8a8_kernel_awq_out_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                            const float* __restrict__ w_scale, float a_scale,
                                            const float* __restrict__ inv_out_scale,
                                            const __half* __restrict__ bias,
                                            int8_t* __restrict__ C,
                                            int M, int N, int K, int n_out,
                                            int8_t* __restrict__ qout = nullptr,
                                            int8_t* __restrict__ kout = nullptr,
                                            int8_t* __restrict__ vtout = nullptr,
                                            int nh = 0, int T = 0, int hd = 0,
                                            int hp = 0) {
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
      float b00 = bias && gc0 < n_out ? __half2float(bias[gc0]) * inv_out_scale[gc0] : 0.f;
      float b01 = bias && gc0 + 1 < n_out ? __half2float(bias[gc0 + 1]) * inv_out_scale[gc0 + 1] : 0.f;
      float b10 = bias && gc1 < n_out ? __half2float(bias[gc1]) * inv_out_scale[gc1] : 0.f;
      float b11 = bias && gc1 + 1 < n_out ? __half2float(bias[gc1 + 1]) * inv_out_scale[gc1 + 1] : 0.f;
      int* accv = acc[i][j];
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
  gemm_w8a8_kernel_awq_out_i8<0><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, inv_out_scale.contiguous().data_ptr<float>(), (const __half*)nullptr,
      C.data_ptr<int8_t>(), M, N, K, N);
  return C;
}

torch::Tensor gemm_w8a8_awq_out_i8_bias_nout(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
    torch::Tensor inv_out_scale, torch::Tensor bias, int64_t n_out_) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar
              && B.dtype() == torch::kChar && bias.dtype() == torch::kHalf,
              "A/B int8 CUDA and bias fp16 required");
  A = A.contiguous(); B = B.contiguous();
  w_scale = w_scale.contiguous(); inv_out_scale = inv_out_scale.contiguous();
  bias = bias.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(0), n_out = (int)n_out_;
  TORCH_CHECK(B.size(1) == K && N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0
              && n_out > 0 && n_out <= N && n_out % 16 == 0
              && inv_out_scale.numel() == N && bias.numel() >= n_out,
              "invalid W8A8 int8-output bias/n_out shape");
  auto C = torch::empty({M, n_out},
      torch::TensorOptions().dtype(torch::kChar).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq_out_i8<0><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
      (float)a_scale, inv_out_scale.data_ptr<float>(),
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
      C.data_ptr<int8_t>(), M, N, K, n_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

std::vector<torch::Tensor> gemm_w8a8_awq_qkv_i8_layouts(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
    torch::Tensor inv_out_scale, torch::Tensor bias, int64_t nh_,
    int64_t T_, int64_t hd_, int64_t hp_) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar
              && B.is_cuda() && B.dtype() == torch::kChar
              && w_scale.is_cuda() && w_scale.dtype() == torch::kFloat32
              && inv_out_scale.is_cuda() && inv_out_scale.dtype() == torch::kFloat32
              && bias.is_cuda() && bias.dtype() == torch::kHalf,
              "direct-layout QKV requires CUDA int8 A/B, fp32 scales and fp16 bias");
  A = A.contiguous(); B = B.contiguous(); w_scale = w_scale.contiguous();
  inv_out_scale = inv_out_scale.contiguous(); bias = bias.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(0);
  const int nh = (int)nh_, T = (int)T_, hd = (int)hd_, hp = (int)hp_;
  const int n_out = 3 * nh * hp;
  TORCH_CHECK(B.size(1) == K && N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0
              && M % T == 0 && nh > 0 && T >= 64 && hd > 0 && hd % 8 == 0
              && hp >= hd && (hp == 32 || hp == 64)
              && n_out == N && n_out % 128 == 0
              && inv_out_scale.numel() == N && w_scale.numel() == N
              && bias.numel() == N,
              "invalid direct-layout W8A8 QKV shape");
  const int batch = M / T, BH = batch * nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(A.device());
  auto q = torch::empty({batch, T, nh, hp}, oi);
  auto k = torch::empty({BH, T, hp}, oi);
  auto vt = torch::empty({BH, hp, T}, oi);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq_out_i8<1><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
      (float)a_scale, inv_out_scale.data_ptr<float>(),
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
      (int8_t*)nullptr, M, N, K, n_out, q.data_ptr<int8_t>(),
      k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(), nh, T, hd, hp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, vt};
}

std::vector<torch::Tensor> gemm_w8a8_awq_qkv_i8_layouts_compact(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
    torch::Tensor inv_out_scale, torch::Tensor bias, int64_t nh_,
    int64_t T_, int64_t hd_, int64_t hp_) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar
              && B.is_cuda() && B.dtype() == torch::kChar
              && w_scale.is_cuda() && w_scale.dtype() == torch::kFloat32
              && inv_out_scale.is_cuda() && inv_out_scale.dtype() == torch::kFloat32
              && bias.is_cuda() && bias.dtype() == torch::kHalf,
              "compact-layout QKV requires CUDA int8 A/B, fp32 scales and fp16 bias");
  A = A.contiguous(); B = B.contiguous(); w_scale = w_scale.contiguous();
  inv_out_scale = inv_out_scale.contiguous(); bias = bias.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(0);
  const int nh = (int)nh_, T = (int)T_, hd = (int)hd_, hp = (int)hp_;
  const int n_out = 3 * nh * hd;
  TORCH_CHECK(B.size(1) == K && N % GWQ_CTA_N == 0 && K % GWQ_CTA_K == 0
              && M % T == 0 && nh > 0 && T >= 64
              && (hd == 24 || hd == 48) && hp == (hd == 24 ? 32 : 64)
              && n_out <= N && inv_out_scale.numel() == N
              && w_scale.numel() == N && bias.numel() >= n_out,
              "invalid compact-layout W8A8 QKV shape");
  const int batch = M / T, BH = batch * nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(A.device());
  auto q = torch::empty({batch, T, nh, hd}, oi);
  auto k = torch::empty({BH, T, hp}, oi);
  auto vt = torch::empty({BH, hp, T}, oi);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w8a8_kernel_awq_out_i8<2><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
      (float)a_scale, inv_out_scale.data_ptr<float>(),
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
      (int8_t*)nullptr, M, N, K, n_out, q.data_ptr<int8_t>(),
      k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(), nh, T, hd, hp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, vt};
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
// QKV_LAYOUT mirrors the w8a8 kernel's modes:
//   0 = ordinary contiguous int8 output (plain per-column inv_out_scale).
//   1 = per-head padded direct layouts -- emits Flash-ready Q/K/Vt from this one launch, the
//       way gemm_w8a8_kernel_awq_out_i8<1> does. Requires the offline hp-padded weight layout.
//   2 = compact token-major QKV codes, decoded per element, feeding
//       qkv_i4codes_i8v_rearrange_kernel. This was the original `QKV_EPILOGUE = true`.
//
// Mode 1 exists because mode 2 was measured at 2164.8 us for the T1024/hd24 QKV stage
// (GEMM 1561.6 + rearrange 603.2) against the w8a8 fused equivalent's ~637 us. Two separate
// causes, and mode 1 removes both: the second pass disappears, AND the per-element role decode
// (`gc % (3*hd)` then `/hd` and `%hd`, plus a `1.f/sv[d]` reciprocal, eight times per
// accumulator pair) collapses into the offline per-column vectors. As in the w8a8 kernel, the
// combined scale is hoisted once per column pair rather than recomputed per store, and every
// Q/K/V role decision has already been folded into inv_out_scale[] and lim[] host-side.
template <int QKV_LAYOUT = 0>
__global__ void gemm_w4a4_kernel_awq_out_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                            const float* __restrict__ w_scale, float a_scale,
                                            const float* __restrict__ inv_out_scale,
                                            int8_t* __restrict__ C, int M, int N, int Kb,
                                            const __half* __restrict__ bias, int n_out,
                                            int nh = 0, int hd = 0, float q_inv = 0.f,
                                            float k_inv = 0.f, const float* __restrict__ sv = nullptr,
                                            const float* __restrict__ lim = nullptr,
                                            int8_t* __restrict__ qout = nullptr,
                                            int8_t* __restrict__ kout = nullptr,
                                            int8_t* __restrict__ vtout = nullptr,
                                            int T = 0, int hp = 0) {
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
  __syncthreads();
  int8_t* Cs = &As[0][0];
#pragma unroll
  for (int i = 0; i < MI; ++i) {
    int r0 = i * GWQ_INTRIN_M + gid, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int c0 = warp_offset_n + j * GWQ_INTRIN_N + tig * 2, c1 = c0 + 8;
      int gc0 = n0 + c0, gc1 = n0 + c1;
      int* accv = acc[i][j];
      if constexpr (QKV_LAYOUT == 1) {
        // Padded direct layouts. Every Q/K/V role decision has already been folded offline into
        // inv_out_scale[] (1/sq on Q columns, 1/sk on K, 1/sv[d] on V) and lim[] (7 on Q/K, 127
        // on V), so there is no per-element decode at all -- the combined scale and the bias
        // term are hoisted once per column, exactly as gemm_w8a8_kernel_awq_out_i8 does. The
        // hp-padded lanes carry zero weight/scale/bias, so they compute an exact zero and no
        // separate tail-clear is needed.
        // The arithmetic is kept in mode 2's EXACT statement order -- v = acc*(a_scale*w_scale);
        // v += bias; rn(v * inv) -- rather than folded into one FFMA the way the w8a8 kernel can
        // afford to. Only the loads and the products that do not depend on acc are hoisted. That
        // costs one extra FMUL per element and buys a real gate: mode 1 is then bit-identical to
        // mode 2, so the port can be validated by byte equality instead of a tolerance. (It can
        // NOT be validated against the fp16-QKV production route, which rounds Q/K/V through
        // __half before quantizing and so legitimately differs by +-1 code.)
        const float aw00 = a_scale * w_scale[gc0], aw01 = a_scale * w_scale[gc0 + 1];
        const float aw10 = a_scale * w_scale[gc1], aw11 = a_scale * w_scale[gc1 + 1];
        const float bh00 = bias ? __half2float(bias[gc0]) : 0.f;
        const float bh01 = bias ? __half2float(bias[gc0 + 1]) : 0.f;
        const float bh10 = bias ? __half2float(bias[gc1]) : 0.f;
        const float bh11 = bias ? __half2float(bias[gc1 + 1]) : 0.f;
        const float iv00 = inv_out_scale[gc0], iv01 = inv_out_scale[gc0 + 1];
        const float iv10 = inv_out_scale[gc1], iv11 = inv_out_scale[gc1 + 1];
        const float l00 = lim[gc0], l01 = lim[gc0 + 1], l10 = lim[gc1], l11 = lim[gc1 + 1];
        auto qc = [](int a, float aw, float bh, float iv, float l) -> int8_t {
          float v = a * aw;
          v += bh;
          int x = __float2int_rn(v * iv);
          const int li = (int)l;
          return (int8_t)(x > li ? li : (x < -li ? -li : x));
        };
        Cs[r0 * GWQ_CTA_N + c0] = qc(accv[0], aw00, bh00, iv00, l00);
        Cs[r0 * GWQ_CTA_N + c0 + 1] = qc(accv[1], aw01, bh01, iv01, l01);
        Cs[r0 * GWQ_CTA_N + c1] = qc(accv[4], aw10, bh10, iv10, l10);
        Cs[r0 * GWQ_CTA_N + c1 + 1] = qc(accv[5], aw11, bh11, iv11, l11);
        Cs[r1 * GWQ_CTA_N + c0] = qc(accv[2], aw00, bh00, iv00, l00);
        Cs[r1 * GWQ_CTA_N + c0 + 1] = qc(accv[3], aw01, bh01, iv01, l01);
        Cs[r1 * GWQ_CTA_N + c1] = qc(accv[6], aw10, bh10, iv10, l10);
        Cs[r1 * GWQ_CTA_N + c1 + 1] = qc(accv[7], aw11, bh11, iv11, l11);
      } else {
#define GWQ_QCODE(ACC, COL) ([&]() {                                                        \
        const int gc_ = (COL);                                                               \
        if (gc_ >= n_out) return 0;                                                          \
        float v_ = (ACC) * (a_scale * w_scale[gc_]);                                         \
        if constexpr (QKV_LAYOUT == 2) {                                                     \
          if (bias) v_ += __half2float(bias[gc_]);                                           \
          const int local_ = gc_ % (3 * hd), sel_ = local_ / hd, d_ = local_ % hd;           \
          const float inv_ = sel_ == 0 ? q_inv : (sel_ == 1 ? k_inv : 1.f / sv[d_]);         \
          int x_ = __float2int_rn(v_ * inv_);                                                \
          const int lim_ = sel_ < 2 ? 7 : 127;                                               \
          return x_ > lim_ ? lim_ : (x_ < -lim_ ? -lim_ : x_);                              \
        } else {                                                                             \
          int x_ = __float2int_rn(v_ * inv_out_scale[gc_]);                                  \
          return x_ > 127 ? 127 : (x_ < -127 ? -127 : x_);                                  \
        }                                                                                    \
      }())
        Cs[r0 * GWQ_CTA_N + c0] = GWQ_QCODE(accv[0], gc0);
        Cs[r0 * GWQ_CTA_N + c0 + 1] = GWQ_QCODE(accv[1], gc0 + 1);
        Cs[r0 * GWQ_CTA_N + c1] = GWQ_QCODE(accv[4], gc1);
        Cs[r0 * GWQ_CTA_N + c1 + 1] = GWQ_QCODE(accv[5], gc1 + 1);
        Cs[r1 * GWQ_CTA_N + c0] = GWQ_QCODE(accv[2], gc0);
        Cs[r1 * GWQ_CTA_N + c0 + 1] = GWQ_QCODE(accv[3], gc0 + 1);
        Cs[r1 * GWQ_CTA_N + c1] = GWQ_QCODE(accv[6], gc1);
        Cs[r1 * GWQ_CTA_N + c1 + 1] = GWQ_QCODE(accv[7], gc1 + 1);
#undef GWQ_QCODE
      }
    }
  }
  __syncthreads();
  if constexpr (QKV_LAYOUT == 1) {
    // Transplanted from gemm_w8a8_kernel_awq_out_i8<1>. Valid verbatim here because at this
    // shape the int4 codes are stored UNPACKED (storage mode _QK_I4_VALUES_I8_MMA: signed int4
    // VALUES in one int8 byte each, so the s8 MMA can run at k=32 instead of padding to the
    // s4 MMA's k=64) and V is plain int8. Both are therefore byte-per-value, exactly as in
    // w8a8. The hd=48 shapes use nibble-packed Q/K and still need the rearrange kernel.
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
          *reinterpret_cast<uint4*>(&qout[((size_t)gm * nh + h) * hp + d0]) = val;
        } else {
          const int b = gm / T, tok = gm - b * T;
          *reinterpret_cast<uint4*>(
              &kout[((size_t)(b * nh + h) * T + tok) * hp + d0]) = val;
        }
      }
    }
    // V: gather a 16-token shared-memory column into one coalesced uint4 global store. The smem
    // reads are strided, which is free; the global write is what has to stay 128-bit.
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
          for (int i2 = 0; i2 < 16; ++i2)
            if (gm + i2 < M && tok + i2 < T)
              vtout[((size_t)(b * nh + h) * hp + d) * T + tok + i2] =
                  Cs[(row0 + i2) * GWQ_CTA_N + col];
        }
      }
    }
  } else {
    for (int idx = t; idx < GWQ_CTA_M * GWQ_CTA_N / 16; idx += blockDim.x) {
      int rc = idx * 16, row = rc / GWQ_CTA_N, col = rc % GWQ_CTA_N;
      if (m0 + row < M && n0 + col + 15 < n_out)
        *(uint4*)&C[(size_t)(m0 + row) * n_out + n0 + col] =
            *(const uint4*)&Cs[row * GWQ_CTA_N + col];
    }
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
  gemm_w4a4_kernel_awq_out_i8<0><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, inv_out_scale.contiguous().data_ptr<float>(), C.data_ptr<int8_t>(), M, N, Kb,
      nullptr, N);
  return C;
}

// Shape-specialized rearrange for the experimental QKV epilogue. The GEMM writes one compact
// token-major INT8 code matrix (Q/K are already restricted to the signed INT4 grid); this pass
// packs Q/K as requested and transposes V in 64-token shared-memory tiles.
template <int TILE_T>
__global__ void qkv_i4codes_i8v_rearrange_kernel(
    const int8_t* __restrict__ qkv, int8_t* __restrict__ q, int8_t* __restrict__ k,
    int8_t* __restrict__ vt, int nh, int T, int hd, int hp_qk, int hp_v,
    bool packed_qk) {
  const int bh = blockIdx.x, h = bh % nh, b = bh / nh;
  const int t0 = blockIdx.y * TILE_T, tt = min(TILE_T, T - t0);
  extern __shared__ int8_t vs[];
  const int qwidth = packed_qk ? hp_qk / 2 : hp_qk;

  if (packed_qk) {
    const int plane = tt * (hp_qk / 2);
    for (int idx = threadIdx.x; idx < 2 * plane; idx += blockDim.x) {
      const int sel = idx / plane, rem = idx - sel * plane;
      const int tl = rem / (hp_qk / 2), dp = rem % (hp_qk / 2), d0 = dp * 2;
      int v0 = 0, v1 = 0;
      if (d0 < hd) {
        const size_t base = (size_t)(b * T + t0 + tl) * (nh * 3 * hd)
                            + h * 3 * hd + sel * hd + d0;
        v0 = qkv[base];
        if (d0 + 1 < hd) v1 = qkv[base + 1];
      }
      int8_t* dst = sel == 0 ? q : k;
      dst[((size_t)bh * T + t0 + tl) * qwidth + dp] =
          (int8_t)((v0 & 0xf) | ((v1 & 0xf) << 4));
    }
  } else {
    const int plane = tt * hp_qk;
    for (int idx = threadIdx.x; idx < 2 * plane; idx += blockDim.x) {
      const int sel = idx / plane, rem = idx - sel * plane;
      const int tl = rem / hp_qk, d = rem % hp_qk;
      int8_t v = 0;
      if (d < hd)
        v = qkv[(size_t)(b * T + t0 + tl) * (nh * 3 * hd)
                + h * 3 * hd + sel * hd + d];
      (sel == 0 ? q : k)[((size_t)bh * T + t0 + tl) * qwidth + d] = v;
    }
  }

  for (int idx = threadIdx.x; idx < tt * hd; idx += blockDim.x) {
    const int tl = idx / hd, d = idx % hd;
    vs[idx] = qkv[(size_t)(b * T + t0 + tl) * (nh * 3 * hd)
                  + h * 3 * hd + 2 * hd + d];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < hp_v * tt; idx += blockDim.x) {
    const int d = idx / tt, tl = idx % tt;
    vt[((size_t)bh * hp_v + d) * T + t0 + tl] = d < hd ? vs[tl * hd + d] : 0;
  }
}

// Direct-layout W4A4 QKV epilogue: ONE launch emits the three Flash-ready tensors, the way
// gemm_w8a8_awq_qkv_i8_layouts does. Only the unpacked storage mode (84, signed INT4 values in
// one byte each, for the INT8 MMA route) is served here -- that is the T1024/hd24 production
// shape, and it is the case where the codes are byte-per-value so the w8a8 store block applies
// unchanged. The nibble-packed hd=48 shapes keep gemm_w4a4_awq_qkv_i4qk_i8v below.
//
// inv_out[] and lim[] are built offline against the SAME hp-padded channel space as B: inv_out
// folds 1/sq on Q columns, 1/sk on K columns and 1/sv[d] on V columns, and lim is 7 on Q/K and
// 127 on V. That is what removes the per-element role decode the two-kernel path pays for.
std::vector<torch::Tensor> gemm_w4a4_awq_qkv_i4qk_i8v_layouts(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K,
    torch::Tensor inv_out, torch::Tensor lim, torch::Tensor bias,
    int64_t nh_, int64_t T_, int64_t hd_, int64_t hp_, torch::Tensor sv) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar
              && w_scale.is_cuda() && w_scale.dtype() == torch::kFloat32
              && inv_out.is_cuda() && inv_out.dtype() == torch::kFloat32
              && lim.is_cuda() && lim.dtype() == torch::kFloat32
              && bias.is_cuda() && bias.dtype() == torch::kHalf,
              "direct-layout W4A4 QKV requires packed int4 A/B, fp32 scales and fp16 bias");
  A = A.contiguous(); B = B.contiguous(); w_scale = w_scale.contiguous();
  inv_out = inv_out.contiguous(); lim = lim.contiguous(); bias = bias.contiguous();
  sv = sv.to(torch::kFloat32).contiguous();
  const int M = A.size(0), N = B.size(0), Kb = (int)K / 2;
  const int nh = (int)nh_, T = (int)T_, hd = (int)hd_, hp = (int)hp_;
  const int n_out = 3 * nh * hp;
  TORCH_CHECK(A.size(1) == Kb && B.size(1) == Kb && N % GWQ_CTA_N == 0
              && K % (GWQ_CTA_K * 2) == 0 && M % T == 0
              && nh > 0 && T >= 64 && hd > 0 && hd % 8 == 0
              && hp >= hd && (hp == 32 || hp == 64)
              && n_out == N && n_out % 128 == 0
              && inv_out.numel() == N && lim.numel() == N
              && w_scale.numel() == N && bias.numel() == N && sv.numel() >= hd,
              "invalid direct-layout W4A4 QKV shape");
  const int batch = M / T, BH = batch * nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(A.device());
  auto q = torch::empty({batch, T, nh, hp}, oi);   // token-major, read directly by Flash
  auto k = torch::empty({BH, T, hp}, oi);
  auto vt = torch::empty({BH, hp, T}, oi);         // already transposed
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq_out_i8<1><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
      (float)a_scale, inv_out.data_ptr<float>(), (int8_t*)nullptr, M, N, Kb,
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()), n_out,
      nh, hd, 0.f, 0.f, (const float*)nullptr, lim.data_ptr<float>(),
      q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(), T, hp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, vt, sv};
}

// Experimental W4A4 QKV GEMM epilogue. Bias/dequantization and static Q/K/V requantization happen
// in the GEMM epilogue, so no FP16 QKV tensor is materialized. storage_mode is named at the Python
// call site: 4 = packed native INT4 Q/K, 84 = unpacked signed-INT4 values for the INT8 MMA route.
std::vector<torch::Tensor> gemm_w4a4_awq_qkv_i4qk_i8v(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K,
    int64_t n_out, torch::Tensor bias, int64_t nh, int64_t T, int64_t hd,
    int64_t hp_qk, int64_t hp_v, int64_t storage_mode,
    double sq, double sk, torch::Tensor sv) {
  constexpr int QK_I4_PACKED = 4, QK_I4_VALUES_I8_MMA = 84;
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar,
              "experimental QKV epilogue: A/B must be packed INT4 CUDA");
  A = A.contiguous(); B = B.contiguous(); w_scale = w_scale.contiguous();
  bias = bias.contiguous(); sv = sv.to(torch::kFloat32).contiguous();
  const int M = A.size(0), N = B.size(0), Kb = (int)K / 2;
  TORCH_CHECK(A.size(1) == Kb && B.size(1) == Kb && N % GWQ_CTA_N == 0
              && K % (GWQ_CTA_K * 2) == 0 && M % T == 0
              && n_out == nh * 3 * hd && n_out <= N && n_out % 16 == 0
              && bias.numel() == n_out && bias.dtype() == torch::kHalf
              && sv.numel() >= hd && hp_qk >= hd && hp_v >= hd
              && (storage_mode == QK_I4_PACKED || storage_mode == QK_I4_VALUES_I8_MMA),
              "experimental QKV epilogue: invalid shape/storage mode");
  const int b = M / (int)T, BH = b * (int)nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(A.device());
  auto raw = torch::empty({M, (int)n_out}, oi);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / GWQ_CTA_N, (M + GWQ_CTA_M - 1) / GWQ_CTA_M);
  gemm_w4a4_kernel_awq_out_i8<2><<<grid, GWQ_NUM_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
      (float)a_scale, nullptr, raw.data_ptr<int8_t>(), M, N, Kb,
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()), (int)n_out,
      (int)nh, (int)hd, 1.f / (float)sq, 1.f / (float)sk, sv.data_ptr<float>());
  const bool packed = storage_mode == QK_I4_PACKED;
  const int qwidth = packed ? (int)hp_qk / 2 : (int)hp_qk;
  auto q = torch::empty({BH, (int)T, qwidth}, oi);
  auto k = torch::empty({BH, (int)T, qwidth}, oi);
  auto vt = torch::empty({BH, (int)hp_v, (int)T}, oi);
  constexpr int TILE_T = 64;
  dim3 rgrid(BH, ((int)T + TILE_T - 1) / TILE_T);
  qkv_i4codes_i8v_rearrange_kernel<TILE_T><<<rgrid, 256, TILE_T * (int)hd, stream>>>(
      raw.data_ptr<int8_t>(), q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      (int)nh, (int)T, (int)hd, (int)hp_qk, (int)hp_v, packed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, vt, sv};
}
