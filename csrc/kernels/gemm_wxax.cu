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
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "../common.cuh"
#include "mma_int8.cuh"

#define GW_WARPS 4
#define GW_BN 64
#define GW_STAGES 3
#define GW_LDS 32               // smem row stride (bytes) = data width (dense; padding hurts occupancy)

template <int MT, bool OUT_I8 = false>
__global__ void gemm_w8a8_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 const float* __restrict__ w_scale, float a_scale,
                                 __half* __restrict__ C, int M, int N, int K,
                                 int8_t* __restrict__ Ci8 = nullptr,
                                 const float* __restrict__ oscale = nullptr,
                                 const float* __restrict__ bias = nullptr) {
  constexpr int WM = MT * 16, BM = GW_WARPS * WM;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * GW_BN, t = threadIdx.x;
  __shared__ int8_t As[GW_STAGES][BM * GW_LDS];
  __shared__ int8_t Bs[GW_STAGES][GW_BN * GW_LDS];
  int acc[MT][GW_BN / 8][4];
#pragma unroll
  for (int mi = 0; mi < MT; ++mi)
    for (int nt = 0; nt < GW_BN / 8; ++nt) { acc[mi][nt][0] = acc[mi][nt][1] = acc[mi][nt][2] = acc[mi][nt][3] = 0; }
  const int nkt = K / 32;
#define GW8_LOAD(kt, buf)                                                                        \
  for (int c = t; c < BM * 2; c += blockDim.x) {                                                 \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * GW_LDS + off]),                              \
                       (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off), (m0 + r) < M);     \
  }                                                                                              \
  for (int c = t; c < GW_BN * 2; c += blockDim.x) {                                              \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * GW_LDS + off]),                              \
                       (const uint4*)(B + (size_t)(n0 + r) * K + (kt) + off), (n0 + r) < N);     \
  }
#pragma unroll
  for (int s = 0; s < GW_STAGES - 1; ++s) { if (s < nkt) { GW8_LOAD(s * 32, s); } __pipeline_commit(); }
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
      for (int mi = 0; mi < MT; ++mi) modiff_mma_m16n8k32(acc[mi][nt], a[mi], b);
    }
    int li = i + GW_STAGES - 1;
    if (li < nkt) { GW8_LOAD(li * 32, li % GW_STAGES); }
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

// A [M,K] int8, B [N,K] int8, w_scale [N] f32, a_scale scalar -> C [M,N] fp16
torch::Tensor gemm_w8a8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(K % 32 == 0 && N % GW_BN == 0, "need K%32==0, N%64==0");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* Ap = A.data_ptr<int8_t>(); auto* Bp = B.data_ptr<int8_t>();
  auto* Wp = w_scale.contiguous().data_ptr<float>();
  auto* Cp = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  if (gw_pick_mt(M, N, K) == 2) {
    const int BM = GW_WARPS * 32;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w8a8_kernel<2><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, N, K);
  } else {
    const int BM = GW_WARPS * 16;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w8a8_kernel<1><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, Cp, M, N, K);
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
  TORCH_CHECK(K % 32 == 0 && N % GW_BN == 0, "need K%32==0, N%64==0");
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
    gemm_w8a8_kernel<2, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, N, K, Cp, Op, Bp2);
  } else {
    const int BM = GW_WARPS * 16;
    dim3 grid((M + BM - 1) / BM, N / GW_BN);
    gemm_w8a8_kernel<1, true><<<grid, GW_WARPS * 32, 0, stream>>>(Ap, Bp, Wp, (float)a_scale, nullptr, M, N, K, Cp, Op, Bp2);
  }
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
