// =========================================================================
// Weight+activation int GEMM for the UNet Linear layers (W8A8 / W4A4), static
// scales, AWQ-referenced (mma.m16n8k32.s8 / m16n8k64.s4 tensor cores; per-channel
// weight scale x static per-tensor activation scale). Referenced from AWQ's QServe
// W8A8 kernel (/workspace/llm-awq/awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu) but a
// self-contained tiled kernel built on the validated fragment mapping in mma_int8.cuh.
//
//   C[M,N] fp16 = (A[M,K] int8 . B[N,K]^T int8)  * a_scale (scalar) * w_scale[n]
//
// B is the weight [N=out, K=in] (row-major), A the activation [M=tokens, K=in].
// One warp per CTA computes a [BM=16, BN=64] output tile; loops K in steps of 32.
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "../common.cuh"
#include "mma_int8.cuh"

#define GW_WARPS 4
#define GW_BM (GW_WARPS * 16)   // 64 rows/CTA (one warp per 16-row sub-tile)
#define GW_BN 64

// Multi-warp CTA: GW_WARPS warps share the B smem tile (loaded once per CTA),
// each warp owns a 16-row A sub-tile. m16n8k32.s8 mma, int32 accumulate, dequant epilogue.
__global__ void gemm_w8a8_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 const float* __restrict__ w_scale, float a_scale,
                                 __half* __restrict__ C, int M, int N, int K) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * GW_BM;
  const int n0 = blockIdx.y * GW_BN;
  const int mw = m0 + w * 16;                       // this warp's row base
  __shared__ int8_t As[GW_BM * 32];
  __shared__ int8_t Bs[GW_BN * 32];
  int acc[GW_BN / 8][4];
#pragma unroll
  for (int nt = 0; nt < GW_BN / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0; }

  for (int kt = 0; kt < K; kt += 32) {
    for (int idx = threadIdx.x; idx < GW_BM * 32; idx += blockDim.x) {
      int r = idx >> 5, c = idx & 31;
      As[idx] = (m0 + r < M) ? A[(size_t)(m0 + r) * K + kt + c] : 0;
    }
    for (int idx = threadIdx.x; idx < GW_BN * 32; idx += blockDim.x) {
      int r = idx >> 5, c = idx & 31;
      Bs[idx] = (n0 + r < N) ? B[(size_t)(n0 + r) * K + kt + c] : 0;
    }
    __syncthreads();
    unsigned a[4];
    a[0] = *(const int*)&As[(w * 16 + gid) * 32 + tig * 4];
    a[1] = *(const int*)&As[(w * 16 + gid + 8) * 32 + tig * 4];
    a[2] = *(const int*)&As[(w * 16 + gid) * 32 + tig * 4 + 16];
    a[3] = *(const int*)&As[(w * 16 + gid + 8) * 32 + tig * 4 + 16];
#pragma unroll
    for (int nt = 0; nt < GW_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[(nt * 8 + gid) * 32 + tig * 4];
      b[1] = *(const int*)&Bs[(nt * 8 + gid) * 32 + tig * 4 + 16];
      modiff_mma_m16n8k32(acc[nt], a, b);
    }
    __syncthreads();
  }

#pragma unroll
  for (int nt = 0; nt < GW_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, c1 = c0 + 1;
    int r0 = mw + gid, r1 = mw + gid + 8;
    if (r0 < M) {
      if (c0 < N) C[(size_t)r0 * N + c0] = __float2half(acc[nt][0] * a_scale * w_scale[c0]);
      if (c1 < N) C[(size_t)r0 * N + c1] = __float2half(acc[nt][1] * a_scale * w_scale[c1]);
    }
    if (r1 < M) {
      if (c0 < N) C[(size_t)r1 * N + c0] = __float2half(acc[nt][2] * a_scale * w_scale[c0]);
      if (c1 < N) C[(size_t)r1 * N + c1] = __float2half(acc[nt][3] * a_scale * w_scale[c1]);
    }
  }
}

// ---- W4A4: same tiling, m16n8k64.s4, K=64/step, inputs packed 2 int4/byte ----
__global__ void gemm_w4a4_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 const float* __restrict__ w_scale, float a_scale,
                                 __half* __restrict__ C, int M, int N, int K) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * GW_BM;
  const int n0 = blockIdx.y * GW_BN;
  const int mw = m0 + w * 16;
  const int Kb = K >> 1;                 // packed bytes per row
  __shared__ int8_t As[GW_BM * 32];      // one k-tile = 64 int4 = 32 bytes/row
  __shared__ int8_t Bs[GW_BN * 32];
  int acc[GW_BN / 8][4];
#pragma unroll
  for (int nt = 0; nt < GW_BN / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0; }

  for (int ktb = 0; ktb < Kb; ktb += 32) {          // 32 bytes = 64 int4 per step
    for (int idx = threadIdx.x; idx < GW_BM * 32; idx += blockDim.x) {
      int r = idx >> 5, c = idx & 31;
      As[idx] = (m0 + r < M) ? A[(size_t)(m0 + r) * Kb + ktb + c] : 0;
    }
    for (int idx = threadIdx.x; idx < GW_BN * 32; idx += blockDim.x) {
      int r = idx >> 5, c = idx & 31;
      Bs[idx] = (n0 + r < N) ? B[(size_t)(n0 + r) * Kb + ktb + c] : 0;
    }
    __syncthreads();
    unsigned a[4];
    a[0] = *(const int*)&As[(w * 16 + gid) * 32 + tig * 4];
    a[1] = *(const int*)&As[(w * 16 + gid + 8) * 32 + tig * 4];
    a[2] = *(const int*)&As[(w * 16 + gid) * 32 + tig * 4 + 16];
    a[3] = *(const int*)&As[(w * 16 + gid + 8) * 32 + tig * 4 + 16];
#pragma unroll
    for (int nt = 0; nt < GW_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[(nt * 8 + gid) * 32 + tig * 4];
      b[1] = *(const int*)&Bs[(nt * 8 + gid) * 32 + tig * 4 + 16];
      modiff_mma_m16n8k64_s4(acc[nt], a, b);
    }
    __syncthreads();
  }

#pragma unroll
  for (int nt = 0; nt < GW_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, c1 = c0 + 1;
    int r0 = mw + gid, r1 = mw + gid + 8;
    if (r0 < M) {
      if (c0 < N) C[(size_t)r0 * N + c0] = __float2half(acc[nt][0] * a_scale * w_scale[c0]);
      if (c1 < N) C[(size_t)r0 * N + c1] = __float2half(acc[nt][1] * a_scale * w_scale[c1]);
    }
    if (r1 < M) {
      if (c0 < N) C[(size_t)r1 * N + c0] = __float2half(acc[nt][2] * a_scale * w_scale[c0]);
      if (c1 < N) C[(size_t)r1 * N + c1] = __float2half(acc[nt][3] * a_scale * w_scale[c1]);
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

// A [M,K] int8, B [N,K] int8, w_scale [N] f32, a_scale scalar -> C [M,N] fp16
torch::Tensor gemm_w8a8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar, "A/B int8 CUDA");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "K mismatch");
  TORCH_CHECK(K % 32 == 0 && N % GW_BN == 0, "need K%32==0, N%64==0");
  auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
  dim3 grid((M + GW_BM - 1) / GW_BM, N / GW_BN);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemm_w8a8_kernel<<<grid, GW_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K);
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
  dim3 grid((M + GW_BM - 1) / GW_BM, N / GW_BN);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemm_w4a4_kernel<<<grid, GW_WARPS * 32, 0, stream>>>(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), w_scale.contiguous().data_ptr<float>(),
      (float)a_scale, reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, (int)N, (int)K);
  return C;
}
