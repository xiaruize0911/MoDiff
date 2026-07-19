// =========================================================================
// Weight+activation int GEMM for the UNet Linear layers (W8A8 / W4A4), static
// scales, AWQ-referenced (mma.m16n8k32.s8 / m16n8k64.s4 tensor cores; per-channel
// weight scale x static per-tensor activation scale). Self-contained tiled kernel
// built on the validated fragment mapping in mma_int8.cuh.
//
//   C[M,N] fp16 = (A[M,K] int8 . B[N,K]^T int8)  * a_scale (scalar) * w_scale[n]
//
// Double-buffered cp.async pipeline (GW_STAGES) hides global load latency.
// Templated on MT (m16 tiles per warp = register blocking): MT=2 reuses each B
// fragment across 2 M-tiles (best for large-M / small-N compute-bound shapes);
// MT=1 (smaller tile, higher occupancy) is best for large-N / small-M. The host
// picks MT per shape (gw_pick_mt).
//
// int8 has TWO K-tile widths, selected per-shape by the host (gemm_w8a8/_out_int8):
// WideK=true uses a 64-wide K-tile (GW_LDS8, doubled from the original 32) with 2
// m16n8k32 mma calls/iteration, mirroring AWQ's CTA_K=64/SHARED_K_ITERS=2 structure
// -- the short-K qkv shape (K=192) was mainloop-bound at 6 iterations of K=32, and
// halving the iteration count wins there. But the extra live registers (a0/a1/b0/b1
// vs a/b) cost the same on every shape, and only pay off when K is short enough
// that iteration overhead dominated -- measured ~5-7% REGRESSION at K=384. So
// WideK is gated to K<384 at the host level; K>=384 uses the original 32-wide
// single-mma path (WideK=false), unchanged from before this optimization.
// Neither touches the smem layout otherwise (no swizzle/ldmatrix -- deferred).
// int4's native m16n8k64 mma already covers 64 logical K per iteration (K packed
// 2-per-byte, Kb/32 tile), so gemm_w4a4_kernel is unaffected and keeps using GW_LDS.
// See docs/quant_speedup_vs_fp16_2026-07-16/NEXT_STEPS.md for the measured results.
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "../common.cuh"
#include "mma_int8.cuh"

#define GW_WARPS 4
#define GW_BN 64
#define GW_STAGES 4             // Stage 2 experiment: was 3. For C192 (WideK, nkt=3) this means the
                                 // whole K dimension gets prefetched in the prologue before any
                                 // compute starts (prologue runs GW_STAGES-1=3 iters = all of nkt=3),
                                 // instead of overlapping load/compute per-iteration. Narrow-path and
                                 // int4 shapes (nkt>=6) still get genuine software-pipelined overlap
                                 // with one more stage in flight. See NEXT_STEPS.md for the measured
                                 // result -- revert to 3 if this doesn't help.
#define GW_LDS 32               // smem row stride (bytes) = data width (dense; padding hurts occupancy)
#define GW_LDS8 64              // int8-only: doubled K-tile width in bytes (see file header)

template <int MT, bool OUT_I8 = false, bool WideK = true>
__global__ void gemm_w8a8_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 const float* __restrict__ w_scale, float a_scale,
                                 __half* __restrict__ C, int M, int N, int K,
                                 int8_t* __restrict__ Ci8 = nullptr,
                                 const float* __restrict__ oscale = nullptr,
                                 const float* __restrict__ bias = nullptr) {
  constexpr int WM = MT * 16, BM = GW_WARPS * WM;
  // WideK=true: 64-wide K-tile (GW_LDS8), 2 mma/iter -- wins on short K (mainloop-bound, e.g. K=192).
  // WideK=false: original 32-wide K-tile (GW_LDS), 1 mma/iter -- the extra a0/a1/b0/b1 register
  // pressure WideK pays on every shape nets a loss once K is long enough that iteration overhead was
  // already small (measured ~5-7% regression at K=384) -- see NEXT_STEPS.md Stage-1 measurement.
  // gemm_w8a8/gemm_w8a8_out_int8 select WideK by K at the host level (K<384 -> true).
  constexpr int LDS = WideK ? GW_LDS8 : GW_LDS;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * GW_BN, t = threadIdx.x;
  __shared__ int8_t As[GW_STAGES][BM * LDS];
  __shared__ int8_t Bs[GW_STAGES][GW_BN * LDS];
  int acc[MT][GW_BN / 8][4];
#pragma unroll
  for (int mi = 0; mi < MT; ++mi)
    for (int nt = 0; nt < GW_BN / 8; ++nt) { acc[mi][nt][0] = acc[mi][nt][1] = acc[mi][nt][2] = acc[mi][nt][3] = 0; }
  const int nkt = K / LDS;   // LDS-wide K-tile per iteration (32 or 64 -- see comment above)
#define GW8_LOAD(kt, buf)                                                                        \
  for (int c = t; c < BM * (LDS / 16); c += blockDim.x) {                                        \
    int r = c / (LDS / 16), off = (c % (LDS / 16)) * 16;                                         \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * LDS + off]),                                  \
                       (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off), (m0 + r) < M);     \
  }                                                                                              \
  for (int c = t; c < GW_BN * (LDS / 16); c += blockDim.x) {                                     \
    int r = c / (LDS / 16), off = (c % (LDS / 16)) * 16;                                         \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * LDS + off]),                                  \
                       (const uint4*)(B + (size_t)(n0 + r) * K + (kt) + off), (n0 + r) < N);     \
  }
#pragma unroll
  for (int s = 0; s < GW_STAGES - 1; ++s) { if (s < nkt) { GW8_LOAD(s * LDS, s); } __pipeline_commit(); }
  __pipeline_wait_prior(GW_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % GW_STAGES;
    if constexpr (WideK) {
      // a0/b0 cover the first 32-wide half of this 64-wide K-tile, a1/b1 the second --
      // two m16n8k32 mma calls per (mi,nt) below, one per half.
      unsigned a0[MT][4], a1[MT][4];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) {
        int rb = w * WM + mi * 16;
        a0[mi][0] = *(const int*)&As[buf][(rb + gid) * LDS + tig * 4];
        a0[mi][1] = *(const int*)&As[buf][(rb + gid + 8) * LDS + tig * 4];
        a0[mi][2] = *(const int*)&As[buf][(rb + gid) * LDS + tig * 4 + 16];
        a0[mi][3] = *(const int*)&As[buf][(rb + gid + 8) * LDS + tig * 4 + 16];
        a1[mi][0] = *(const int*)&As[buf][(rb + gid) * LDS + tig * 4 + 32];
        a1[mi][1] = *(const int*)&As[buf][(rb + gid + 8) * LDS + tig * 4 + 32];
        a1[mi][2] = *(const int*)&As[buf][(rb + gid) * LDS + tig * 4 + 48];
        a1[mi][3] = *(const int*)&As[buf][(rb + gid + 8) * LDS + tig * 4 + 48];
      }
#pragma unroll
      for (int nt = 0; nt < GW_BN / 8; ++nt) {
        unsigned b0[2], b1[2];
        b0[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * LDS + tig * 4];
        b0[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * LDS + tig * 4 + 16];
        b1[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * LDS + tig * 4 + 32];
        b1[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * LDS + tig * 4 + 48];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          modiff_mma_m16n8k32(acc[mi][nt], a0[mi], b0);
          modiff_mma_m16n8k32(acc[mi][nt], a1[mi], b1);
        }
      }
    } else {
      unsigned a[MT][4];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) {
        int rb = w * WM + mi * 16;
        a[mi][0] = *(const int*)&As[buf][(rb + gid) * LDS + tig * 4];
        a[mi][1] = *(const int*)&As[buf][(rb + gid + 8) * LDS + tig * 4];
        a[mi][2] = *(const int*)&As[buf][(rb + gid) * LDS + tig * 4 + 16];
        a[mi][3] = *(const int*)&As[buf][(rb + gid + 8) * LDS + tig * 4 + 16];
      }
#pragma unroll
      for (int nt = 0; nt < GW_BN / 8; ++nt) {
        unsigned b[2];
        b[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * LDS + tig * 4];
        b[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * LDS + tig * 4 + 16];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) modiff_mma_m16n8k32(acc[mi][nt], a[mi], b);
      }
    }
    int li = i + GW_STAGES - 1;
    if (li < nkt) { GW8_LOAD(li * LDS, li % GW_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(GW_STAGES - 2);
    __syncthreads();
  }
#undef GW8_LOAD
  // Vectorized epilogue: c0 is always even and c0+1 < N (N%64==0, c0 in [n0, n0+62] <= N-2),
  // so each (c0,c1) pair is a 4B-aligned half2 store -- halves the store count vs scalar.
  // OUT_I8: requantize to int8 = round(acc * a_scale * w_scale[c] * oscale[c]) instead (for the
  // fused qkv->flash path; oscale[c] = 127/absmax_calibrated per output column).
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
    int mwb = m0 + w * WM + mi * 16;
#pragma unroll
    for (int nt = 0; nt < GW_BN / 8; ++nt) {
      int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
      float s0 = a_scale * w_scale[c0], s1 = a_scale * w_scale[c0 + 1];
      if constexpr (OUT_I8) {
        float b0 = bias ? bias[c0] : 0.f, b1 = bias ? bias[c0 + 1] : 0.f;
        float oo0 = oscale[c0], oo1 = oscale[c0 + 1];
        if (r0 < M) {
          Ci8[(size_t)r0 * N + c0]     = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][0] * s0 + b0) * oo0)));
          Ci8[(size_t)r0 * N + c0 + 1] = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][1] * s1 + b1) * oo1)));
        }
        if (r1 < M) {
          Ci8[(size_t)r1 * N + c0]     = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][2] * s0 + b0) * oo0)));
          Ci8[(size_t)r1 * N + c0 + 1] = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][3] * s1 + b1) * oo1)));
        }
      } else {
        if (r0 < M) *(__half2*)&C[(size_t)r0 * N + c0] =
            __halves2half2(__float2half(acc[mi][nt][0] * s0), __float2half(acc[mi][nt][1] * s1));
        if (r1 < M) *(__half2*)&C[(size_t)r1 * N + c0] =
            __halves2half2(__float2half(acc[mi][nt][2] * s0), __float2half(acc[mi][nt][3] * s1));
      }
    }
  }
}

template <int MT, bool OUT_I8 = false>
__global__ void gemm_w4a4_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 const float* __restrict__ w_scale, float a_scale,
                                 __half* __restrict__ C, int M, int N, int K,
                                 int8_t* __restrict__ Ci8 = nullptr,
                                 const float* __restrict__ oscale = nullptr,
                                 const float* __restrict__ bias = nullptr) {
  constexpr int WM = MT * 16, BM = GW_WARPS * WM;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * GW_BN, t = threadIdx.x;
  const int Kb = K >> 1;
  __shared__ int8_t As[GW_STAGES][BM * GW_LDS];
  __shared__ int8_t Bs[GW_STAGES][GW_BN * GW_LDS];
  int acc[MT][GW_BN / 8][4];
#pragma unroll
  for (int mi = 0; mi < MT; ++mi)
    for (int nt = 0; nt < GW_BN / 8; ++nt) { acc[mi][nt][0] = acc[mi][nt][1] = acc[mi][nt][2] = acc[mi][nt][3] = 0; }
  const int nkt = Kb / 32;
#define GW4_LOAD(ktb, buf)                                                                       \
  for (int c = t; c < BM * 2; c += blockDim.x) {                                                 \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * GW_LDS + off]),                              \
                       (const uint4*)(A + (size_t)(m0 + r) * Kb + (ktb) + off), (m0 + r) < M);   \
  }                                                                                              \
  for (int c = t; c < GW_BN * 2; c += blockDim.x) {                                              \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * GW_LDS + off]),                              \
                       (const uint4*)(B + (size_t)(n0 + r) * Kb + (ktb) + off), (n0 + r) < N);   \
  }
#pragma unroll
  for (int s = 0; s < GW_STAGES - 1; ++s) { if (s < nkt) { GW4_LOAD(s * 32, s); } __pipeline_commit(); }
  __pipeline_wait_prior(GW_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % GW_STAGES;
    unsigned a[MT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
      int rb = w * WM + mi * 16;
      a[mi][0] = *(const int*)&As[buf][(rb + gid) * GW_LDS + tig * 4];
      a[mi][1] = *(const int*)&As[buf][(rb + gid + 8) * GW_LDS + tig * 4];
      a[mi][2] = *(const int*)&As[buf][(rb + gid) * GW_LDS + tig * 4 + 16];
      a[mi][3] = *(const int*)&As[buf][(rb + gid + 8) * GW_LDS + tig * 4 + 16];
    }
#pragma unroll
    for (int nt = 0; nt < GW_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * GW_LDS + tig * 4];
      b[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * GW_LDS + tig * 4 + 16];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) modiff_mma_m16n8k64_s4(acc[mi][nt], a[mi], b);
    }
    int li = i + GW_STAGES - 1;
    if (li < nkt) { GW4_LOAD(li * 32, li % GW_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(GW_STAGES - 2);
    __syncthreads();
  }
#undef GW4_LOAD
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
    int mwb = m0 + w * WM + mi * 16;
#pragma unroll
    for (int nt = 0; nt < GW_BN / 8; ++nt) {
      int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
      float s0 = a_scale * w_scale[c0], s1 = a_scale * w_scale[c0 + 1];
      if constexpr (OUT_I8) {
        float b0 = bias ? bias[c0] : 0.f, b1 = bias ? bias[c0 + 1] : 0.f;
        float oo0 = oscale[c0], oo1 = oscale[c0 + 1];
        if (r0 < M) {
          Ci8[(size_t)r0 * N + c0]     = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][0] * s0 + b0) * oo0)));
          Ci8[(size_t)r0 * N + c0 + 1] = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][1] * s1 + b1) * oo1)));
        }
        if (r1 < M) {
          Ci8[(size_t)r1 * N + c0]     = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][2] * s0 + b0) * oo0)));
          Ci8[(size_t)r1 * N + c0 + 1] = (int8_t)fminf(127.f, fmaxf(-127.f, roundf((acc[mi][nt][3] * s1 + b1) * oo1)));
        }
      } else {
        if (r0 < M) *(__half2*)&C[(size_t)r0 * N + c0] =
            __halves2half2(__float2half(acc[mi][nt][0] * s0), __float2half(acc[mi][nt][1] * s1));
        if (r1 < M) *(__half2*)&C[(size_t)r1 * N + c0] =
            __halves2half2(__float2half(acc[mi][nt][2] * s0), __float2half(acc[mi][nt][3] * s1));
      }
    }
  }
}

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

torch::Tensor quantize_act_int8(torch::Tensor x, double a_scale) {
  x = x.contiguous(); long n = x.numel();
  auto out = torch::empty_like(x, torch::TensorOptions().dtype(torch::kChar).device(x.device()));
  int T = 256; long blocks = (n + T - 1) / T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  quant_act_int8_kernel<<<blocks, T, 0, s>>>(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                                             out.data_ptr<int8_t>(), 1.f / (float)a_scale, n);
  return out;
}

torch::Tensor quantize_act_int4_pack(torch::Tensor x, double a_scale) {
  x = x.contiguous(); int M = x.size(0), K = x.size(1); long nout = (long)M * (K / 2);
  auto out = torch::empty({M, K / 2}, torch::TensorOptions().dtype(torch::kChar).device(x.device()));
  int T = 256; long blocks = (nout + T - 1) / T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  quant_act_int4_pack_kernel<<<blocks, T, 0, s>>>(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                                                  out.data_ptr<int8_t>(), 1.f / (float)a_scale, K, nout);
  return out;
}

// Shape-adaptive tile: MT=2 (register-blocked) for large-M / small-N compute-bound
// shapes; MT=1 (higher occupancy) otherwise. Matches the per-shape benchmark.
static inline int gw_pick_mt(int M, int N, int K) {
  const char* e = getenv("MODIFF_GW_MT");
  if (e) return atoi(e);
  // MT=2 (register-blocked) only pays when the K loop is long enough to be
  // compute-bound; for short-K (e.g. qkv K=192) it just cuts occupancy, so MT=1.
  return (M >= 2048 && N <= 768 && K >= 256) ? 2 : 1;
}

// int8-only: WideK=true (64-wide K-tile) wins on short-K (K=192, mainloop-bound);
// WideK=false (original 32-wide) wins once K is long enough that the doubled
// register pressure outweighs the shrinking iteration-count benefit (measured
// regression at K=384). Threshold picked from the measured crossover -- see
// docs/quant_speedup_vs_fp16_2026-07-16/NEXT_STEPS.md.
static inline bool gw_pick_widek(int K) {
  const char* e = getenv("MODIFF_GW_WIDEK");
  if (e) return atoi(e) != 0;
  return K < 384;
}

// A [M,K] int8, B [N,K] int8, w_scale [N] f32, a_scale scalar -> C [M,N] fp16
torch::Tensor gemm_w8a8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(N % GW_BN == 0, "need N%64==0");
  bool wideK = gw_pick_widek(K);
  TORCH_CHECK(wideK ? (K % 64 == 0) : (K % 32 == 0),
              "need K%64==0 (K<384, wide K-tile) or K%32==0 (K>=384) -- see gw_pick_widek");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* Ap = A.data_ptr<int8_t>(); auto* Bp = B.data_ptr<int8_t>();
  auto* Wp = w_scale.contiguous().data_ptr<float>();
  auto* Cp = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  if (gw_pick_mt(M, N, K) == 2) {
    const int BM = GW_WARPS * 32;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    if (wideK) gemm_w8a8_kernel<2, false, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, N, K);
    else       gemm_w8a8_kernel<2, false, false><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, N, K);
  } else {
    const int BM = GW_WARPS * 16;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    if (wideK) gemm_w8a8_kernel<1, false, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, N, K);
    else       gemm_w8a8_kernel<1, false, false><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, N, K);
  }
  return C;
}

// A [M,K/2] packed int4, B [N,K/2] packed int4, w_scale [N], a_scale scalar, K logical -> C [M,N] fp16
torch::Tensor gemm_w4a4(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B packed int4 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), N = B.size(0);
  TORCH_CHECK(A.size(1) == K / 2 && B.size(1) == K / 2, "packed K/2 mismatch");
  TORCH_CHECK(K % 64 == 0 && N % GW_BN == 0, "need K%64==0, N%64==0");
  auto C = torch::empty({M, (int)N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* Ap = A.data_ptr<int8_t>(); auto* Bp = B.data_ptr<int8_t>();
  auto* Wp = w_scale.contiguous().data_ptr<float>();
  auto* Cp = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  if (gw_pick_mt(M, (int)N, (int)K) == 2) {
    const int BM = GW_WARPS * 32;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w4a4_kernel<2><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, (int)N, (int)K);
  } else {
    const int BM = GW_WARPS * 16;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w4a4_kernel<1><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, (int)N, (int)K);
  }
  return C;
}

// int8-OUTPUT variants for the fused qkv->flash path: emit int8 [M,N] =
// round(acc * a_scale * w_scale[c] * oscale[c]) directly (no fp16 round-trip).
// oscale [N] f32 = 127/absmax_calibrated per output column (Q/K per-tensor, V per-channel).
static inline const float* gw_bias_ptr(torch::Tensor& bias, int N) {
  if (bias.numel() == 0) return nullptr;
  TORCH_CHECK(bias.numel() == N && bias.dtype() == torch::kFloat32, "bias must be [N] f32 or empty");
  return bias.contiguous().data_ptr<float>();
}

torch::Tensor gemm_w8a8_out_int8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                 double a_scale, torch::Tensor oscale, torch::Tensor bias) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(N % GW_BN == 0, "need N%64==0");
  bool wideK = gw_pick_widek(K);
  TORCH_CHECK(wideK ? (K % 64 == 0) : (K % 32 == 0),
              "need K%64==0 (K<384, wide K-tile) or K%32==0 (K>=384) -- see gw_pick_widek");
  TORCH_CHECK(oscale.numel() == N, "oscale must be [N]");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kChar).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* Ap = A.data_ptr<int8_t>(); auto* Bp = B.data_ptr<int8_t>();
  auto* Wp = w_scale.contiguous().data_ptr<float>();
  auto* Op = oscale.contiguous().data_ptr<float>();
  const float* Bp2 = gw_bias_ptr(bias, N);
  auto* Cp = C.data_ptr<int8_t>();
  if (gw_pick_mt(M, N, K) == 2) {
    const int BM = GW_WARPS * 32;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    if (wideK) gemm_w8a8_kernel<2, true, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, N, K, Cp, Op, Bp2);
    else       gemm_w8a8_kernel<2, true, false><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, N, K, Cp, Op, Bp2);
  } else {
    const int BM = GW_WARPS * 16;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    if (wideK) gemm_w8a8_kernel<1, true, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, N, K, Cp, Op, Bp2);
    else       gemm_w8a8_kernel<1, true, false><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, N, K, Cp, Op, Bp2);
  }
  return C;
}

// =========================================================================
// Stage 3 (plan-on-how-to-moonlit-pearl.md): AWQ-tiling-scheme validation kernel.
//
// Our own gemm_w8a8_kernel above partitions M across warps and shares B redundantly
// (each warp owns an M-slice, sees the full N-tile) -- a mirror image of AWQ's large-M
// config (w8a8_gemm_cuda.cu, num_out_feats>128 branch: CTA_M=128,CTA_N=128,CTA_K=64,
// WARP_M=128,WARP_N=32,STAGES=3), which partitions N across warps and shares A
// redundantly (WARP_M==CTA_M -> every warp reads the FULL M-range of A; WARP_N=32,
// CTA_N/WARP_N=4 warps split N into 4x32 slices). That mismatch means AWQ's exact
// ldmatrix/swizzle address formulas can't be dropped into our existing kernel --
// this is a standalone, from-scratch port of AWQ's scheme instead (same tile shape,
// same swizzle formula `col ^ ((row/2)&3)`, same ldmatrix.m8n8.x4 reads), built to
// validate that ldmatrix+swizzle+128-wide-N actually pays off before committing to
// a much riskier int4 port (no true AWQ W4A4 kernel exists to cross-check against).
//
// Deliberately NOT a full transcription of dense_kernel0: this version loads both
// INTRIN_K=32 sub-halves of a CTA_K=64 tile back-to-back before issuing the next
// global prefetch (AWQ's register-ping-pong overlaps that differently) -- simpler,
// still exercises the exact mechanism being validated. Not wired into gemm_w8a8's
// dispatch; requires N%128==0 (pad at the call site, exactly like AWQ's own kernel
// requires callers to do -- see awq_vs_ours.py's Np/Wqp/wsh padding).
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

__global__ void gemm_w8a8_kernel_awq(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     const float* __restrict__ w_scale, float a_scale,
                                     __half* __restrict__ C, int M, int N, int K) {
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
      if (row0 < M) {
        *(__half2*)&C[(size_t)row0 * N + col0] = __halves2half2(__float2half(accv[0] * s00), __float2half(accv[1] * s01));
        *(__half2*)&C[(size_t)row0 * N + col1] = __halves2half2(__float2half(accv[4] * s10), __float2half(accv[5] * s11));
      }
      if (row1 < M) {
        *(__half2*)&C[(size_t)row1 * N + col0] = __halves2half2(__float2half(accv[2] * s00), __float2half(accv[3] * s01));
        *(__half2*)&C[(size_t)row1 * N + col1] = __halves2half2(__float2half(accv[6] * s10), __float2half(accv[7] * s11));
      }
    }
  }
}

// A [M,K] int8, B [N,K] int8 (N%128==0 -- pad at call site), w_scale [N] f32, a_scale
// scalar -> C [M,N] fp16. Validation-only entry point for the Stage-3 kernel above.
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
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K);
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
__global__ void gemm_w4a4_kernel_awq(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     const float* __restrict__ w_scale, float a_scale,
                                     __half* __restrict__ C, int M, int N, int Kb) {
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
      if (row0 < M) {
        *(__half2*)&C[(size_t)row0 * N + col0] = __halves2half2(__float2half(accv[0] * s00), __float2half(accv[1] * s01));
        *(__half2*)&C[(size_t)row0 * N + col1] = __halves2half2(__float2half(accv[4] * s10), __float2half(accv[5] * s11));
      }
      if (row1 < M) {
        *(__half2*)&C[(size_t)row1 * N + col0] = __halves2half2(__float2half(accv[2] * s00), __float2half(accv[3] * s01));
        *(__half2*)&C[(size_t)row1 * N + col1] = __halves2half2(__float2half(accv[6] * s10), __float2half(accv[7] * s11));
      }
    }
  }
}

// A [M,K/2] packed int4, B [N,K/2] packed int4, w_scale [N] f32, a_scale scalar, K
// logical (N%128==0, K%128==0 -- pad at call site) -> C [M,N] fp16. Validation-only.
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
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, Kb);
  return C;
}

torch::Tensor gemm_w4a4_out_int8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                 double a_scale, int64_t K, torch::Tensor oscale, torch::Tensor bias) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B packed int4 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), N = B.size(0);
  TORCH_CHECK(A.size(1) == K / 2 && B.size(1) == K / 2, "packed K/2 mismatch");
  TORCH_CHECK(K % 64 == 0 && N % GW_BN == 0, "need K%64==0, N%64==0");
  TORCH_CHECK(oscale.numel() == N, "oscale must be [N]");
  auto C = torch::empty({M, (int)N}, torch::TensorOptions().dtype(torch::kChar).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* Ap = A.data_ptr<int8_t>(); auto* Bp = B.data_ptr<int8_t>();
  auto* Wp = w_scale.contiguous().data_ptr<float>();
  auto* Op = oscale.contiguous().data_ptr<float>();
  const float* Bp2 = gw_bias_ptr(bias, (int)N);
  auto* Cp = C.data_ptr<int8_t>();
  if (gw_pick_mt(M, (int)N, (int)K) == 2) {
    const int BM = GW_WARPS * 32;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w4a4_kernel<2, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, (int)N, (int)K, Cp, Op, Bp2);
  } else {
    const int BM = GW_WARPS * 16;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w4a4_kernel<1, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, (int)N, (int)K, Cp, Op, Bp2);
  }
  return C;
}
