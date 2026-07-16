// =========================================================================
// Batched int8/int4 GEMMs for MoDiff's STANDARD (materialized) quantized attention:
//   QKᵀ:  S[BH,T,T]   (fp32 raw scores) = Q[BH,T,hd_pad] · K[BH,T,hd_pad]ᵀ
//   AV:   O[BH,T,hd]  (fp16)            = P[BH,T,T]       · Vᵀ[BH,hd_pad,T]
// plus a softmax/dequant/requant kernel (attn_softmax_requant.cu-style, here inline).
//
// Reuses the validated fragment mapping (mma_int8.cuh) and the gemm_wxax cp.async
// mainloop, adapted to (a) a batch dim (grid.z over BH), (b) an fp32 raw-score output
// for QKᵀ (int32 QKᵀ overflows fp16), and (c) a per-row × per-col dequant epilogue for AV.
// Scales are applied where they are natural: QKᵀ emits raw int32→fp32 and the softmax
// kernel applies per-token sq[i]·sk[j]; AV applies per-row sp[i]·per-channel sv[d].
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "../common.cuh"
#include "mma_int8.cuh"

#define AQ_WARPS 4
#define AQ_BN 64
#define AQ_STAGES 3
#define AQ_LDS 32          // smem row stride (bytes); dense int8

// ---- batched int8 QKᵀ: C[M,N] fp32 (raw acc) per batch; A[M,K],B[N,K] int8, K=hd_pad ----
__global__ void bmm_qk_s8_kernel(const int8_t* __restrict__ Aall, const int8_t* __restrict__ Ball,
                                 float* __restrict__ Call, int M, int N, int K) {
  const int bh = blockIdx.z;
  const int8_t* A = Aall + (size_t)bh * M * K;
  const int8_t* B = Ball + (size_t)bh * N * K;
  float* C = Call + (size_t)bh * M * N;
  constexpr int WM = 16, BM = AQ_WARPS * WM;          // MT=1
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * AQ_BN, t = threadIdx.x;
  __shared__ int8_t As[AQ_STAGES][BM * AQ_LDS];
  __shared__ int8_t Bs[AQ_STAGES][AQ_BN * AQ_LDS];
  int acc[AQ_BN / 8][4];
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0; }
  const int nkt = K / 32;
#define AQ_LOAD(kt, buf)                                                                         \
  for (int c = t; c < BM * 2; c += blockDim.x) {                                                 \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off), (m0 + r) < M);     \
  }                                                                                              \
  for (int c = t; c < AQ_BN * 2; c += blockDim.x) {                                              \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(B + (size_t)(n0 + r) * K + (kt) + off), (n0 + r) < N);     \
  }
#pragma unroll
  for (int s = 0; s < AQ_STAGES - 1; ++s) { if (s < nkt) { AQ_LOAD(s * 32, s); } __pipeline_commit(); }
  __pipeline_wait_prior(AQ_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % AQ_STAGES;
    unsigned a[4];
    int rb = w * WM;
    a[0] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4];
    a[1] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4];
    a[2] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4 + 16];
    a[3] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4 + 16];
#pragma unroll
    for (int nt = 0; nt < AQ_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4];
      b[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4 + 16];
      modiff_mma_m16n8k32(acc[nt], a, b);
    }
    int li = i + AQ_STAGES - 1;
    if (li < nkt) { AQ_LOAD(li * 32, li % AQ_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(AQ_STAGES - 2);
    __syncthreads();
  }
#undef AQ_LOAD
  // raw fp32 scores (no scale; softmax applies per-token sq·sk). c0 even, c0+1<N (N%64==0).
  int mwb = m0 + w * WM;
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
    if (r0 < M) { C[(size_t)r0 * N + c0] = (float)acc[nt][0]; C[(size_t)r0 * N + c0 + 1] = (float)acc[nt][1]; }
    if (r1 < M) { C[(size_t)r1 * N + c0] = (float)acc[nt][2]; C[(size_t)r1 * N + c0 + 1] = (float)acc[nt][3]; }
  }
}

// Q,K: int8 [BH,T,hd_pad] contiguous. Returns S [BH,T,T] fp32 (raw QKᵀ accumulator).
torch::Tensor attn_qk_int8(torch::Tensor Q, torch::Tensor K) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kChar && K.dtype() == torch::kChar, "Q/K int8 CUDA");
  Q = Q.contiguous(); K = K.contiguous();
  int BH = Q.size(0), T = Q.size(1), hd_pad = Q.size(2);
  TORCH_CHECK(K.size(0) == BH && K.size(1) == T && K.size(2) == hd_pad, "Q/K shape mismatch");
  TORCH_CHECK(hd_pad % 32 == 0 && T % AQ_BN == 0, "need hd_pad%32==0, T%64==0");
  auto S = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
  const int BM = AQ_WARPS * 16;
  dim3 grid((T + BM - 1) / BM, T / AQ_BN, BH);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bmm_qk_s8_kernel<<<grid, AQ_WARPS * 32, 0, stream>>>(
      Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(), S.data_ptr<float>(), T, T, hd_pad);
  return S;
}

// ---- softmax(dequant(S)) + requantize P to int8 ----
// One block per output row (bh,i). Dequant logit_j = S[i,j]·sq[i]·sk[j]·scale; online row-max/
// exp/sum in fp32; emit P_i8[j] = round(exp(logit_j - m)·127) ∈ [0,127] and per-row sp[i] =
// 1/(127·Σexp) so that p[i,j] = P_i8[i,j]·sp[i] (dequantized softmax prob). AV then applies sp[i].
#define AQ_SM_THREADS 256
__global__ void attn_softmax_requant_kernel(const float* __restrict__ S, const float* __restrict__ sq,
                                            const float* __restrict__ sk, int8_t* __restrict__ P,
                                            float* __restrict__ sp, int T, float scale) {
  const int row = blockIdx.x;                 // global row = bh*T + i
  const int bh = row / T;
  const float* Srow = S + (size_t)row * T;
  const float* skb = sk + (size_t)bh * T;
  int8_t* Prow = P + (size_t)row * T;
  const float sqi = sq[row] * scale;
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ float red[AQ_SM_THREADS];
  float m = -1e30f;
  for (int j = tid; j < T; j += nt) m = fmaxf(m, Srow[j] * sqi * skb[j]);
  red[tid] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float sum = 0.f;
  for (int j = tid; j < T; j += nt) sum += __expf(Srow[j] * sqi * skb[j] - m);
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  const float l = red[0];
  if (tid == 0) sp[row] = 1.f / (127.f * fmaxf(l, 1e-20f));
  for (int j = tid; j < T; j += nt) {
    int q = __float2int_rn(__expf(Srow[j] * sqi * skb[j] - m) * 127.f);
    Prow[j] = (int8_t)(q < 0 ? 0 : (q > 127 ? 127 : q));
  }
}

// ---- batched int8 AV: O[T,hd_pad] fp16 = P[T,T] · Vt[hd_pad,T]ᵀ, dequant sp[row]·sv[col] ----
__global__ void bmm_av_s8_kernel(const int8_t* __restrict__ Pall, const int8_t* __restrict__ Vall,
                                 const float* __restrict__ spall, const float* __restrict__ svall,
                                 __half* __restrict__ Oall, int M, int N, int K) {
  const int bh = blockIdx.z;
  const int8_t* A = Pall + (size_t)bh * M * K;        // P [T,T]
  const int8_t* B = Vall + (size_t)bh * N * K;        // Vt [hd_pad,T]
  const float* sp = spall + (size_t)bh * M;           // [T]
  const float* sv = svall + (size_t)bh * N;           // [hd_pad]
  __half* C = Oall + (size_t)bh * M * N;              // O [T,hd_pad]
  constexpr int WM = 16, BM = AQ_WARPS * WM;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * AQ_BN, t = threadIdx.x;
  __shared__ int8_t As[AQ_STAGES][BM * AQ_LDS];
  __shared__ int8_t Bs[AQ_STAGES][AQ_BN * AQ_LDS];
  int acc[AQ_BN / 8][4];
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0; }
  const int nkt = K / 32;
#define AV_LOAD(kt, buf)                                                                         \
  for (int c = t; c < BM * 2; c += blockDim.x) {                                                 \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(A + (size_t)(m0 + r) * K + (kt) + off), (m0 + r) < M);     \
  }                                                                                              \
  for (int c = t; c < AQ_BN * 2; c += blockDim.x) {                                              \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(B + (size_t)(n0 + r) * K + (kt) + off), (n0 + r) < N);     \
  }
#pragma unroll
  for (int s = 0; s < AQ_STAGES - 1; ++s) { if (s < nkt) { AV_LOAD(s * 32, s); } __pipeline_commit(); }
  __pipeline_wait_prior(AQ_STAGES - 2);
  __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % AQ_STAGES;
    unsigned a[4];
    int rb = w * WM;
    a[0] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4];
    a[1] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4];
    a[2] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4 + 16];
    a[3] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4 + 16];
#pragma unroll
    for (int nt = 0; nt < AQ_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4];
      b[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4 + 16];
      modiff_mma_m16n8k32(acc[nt], a, b);
    }
    int li = i + AQ_STAGES - 1;
    if (li < nkt) { AV_LOAD(li * 32, li % AQ_STAGES); }
    __pipeline_commit();
    __pipeline_wait_prior(AQ_STAGES - 2);
    __syncthreads();
  }
#undef AV_LOAD
  int mwb = m0 + w * WM;
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
    float v0 = sv[c0], v1 = sv[c0 + 1];
    if (r0 < M) { float s = sp[r0];
      C[(size_t)r0 * N + c0]     = __float2half(acc[nt][0] * s * v0);
      C[(size_t)r0 * N + c0 + 1] = __float2half(acc[nt][1] * s * v1); }
    if (r1 < M) { float s = sp[r1];
      C[(size_t)r1 * N + c0]     = __float2half(acc[nt][2] * s * v0);
      C[(size_t)r1 * N + c0 + 1] = __float2half(acc[nt][3] * s * v1); }
  }
}

// P int8 [BH,T,T], Vt int8 [BH,hd_pad,T], sp [BH,T], sv [BH,hd_pad] -> O fp16 [BH,T,hd_pad]
torch::Tensor attn_av_int8(torch::Tensor P, torch::Tensor Vt, torch::Tensor sp, torch::Tensor sv) {
  TORCH_CHECK(P.is_cuda() && P.dtype() == torch::kChar && Vt.dtype() == torch::kChar, "P/Vt int8 CUDA");
  P = P.contiguous(); Vt = Vt.contiguous();
  int BH = P.size(0), T = P.size(1), hd_pad = Vt.size(1);
  TORCH_CHECK(P.size(2) == T && Vt.size(2) == T, "P[BH,T,T]/Vt[BH,hd_pad,T] mismatch");
  TORCH_CHECK(T % 32 == 0 && hd_pad % AQ_BN == 0, "need T%32==0, hd_pad%64==0");
  auto O = torch::empty({BH, T, hd_pad}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
  const int BM = AQ_WARPS * 16;
  dim3 grid((T + BM - 1) / BM, hd_pad / AQ_BN, BH);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bmm_av_s8_kernel<<<grid, AQ_WARPS * 32, 0, stream>>>(
      P.data_ptr<int8_t>(), Vt.data_ptr<int8_t>(), sp.contiguous().data_ptr<float>(),
      sv.contiguous().data_ptr<float>(), reinterpret_cast<__half*>(O.data_ptr<at::Half>()), T, hd_pad, T);
  return O;
}

// S [BH,T,T] fp32 raw, sq/sk [BH,T] -> {P int8 [BH,T,T], sp [BH,T]}
std::vector<torch::Tensor> attn_softmax_requant(torch::Tensor S, torch::Tensor sq, torch::Tensor sk,
                                                double softmax_scale) {
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kFloat32 && S.dim() == 3, "S fp32 [BH,T,T]");
  S = S.contiguous();
  int BH = S.size(0), T = S.size(2);
  TORCH_CHECK(S.size(1) == T, "S must be [BH,T,T]");
  auto P = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kChar).device(S.device()));
  auto sp = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  attn_softmax_requant_kernel<<<BH * T, AQ_SM_THREADS, 0, stream>>>(
      S.data_ptr<float>(), sq.contiguous().data_ptr<float>(), sk.contiguous().data_ptr<float>(),
      P.data_ptr<int8_t>(), sp.data_ptr<float>(), T, (float)softmax_scale);
  return {P, sp};
}

// ============================ int4 (W4A4) score path ============================
// Packed int4 (2/byte). QKᵀ: Q,K [BH,T,hd_pad/2]; AV: P [BH,T,T/2], Vt [BH,hd_pad,T/2].
// Mirrors the int8 kernels with modiff_mma_m16n8k64_s4 and Kb=Klogical/2. int4 scores are
// aggressive (P has 8 levels) -> quality reported, not gated.
__global__ void bmm_qk_s4_kernel(const int8_t* __restrict__ Aall, const int8_t* __restrict__ Ball,
                                 float* __restrict__ Call, int M, int N, int K) {
  const int bh = blockIdx.z; const int Kb = K >> 1;
  const int8_t* A = Aall + (size_t)bh * M * Kb;
  const int8_t* B = Ball + (size_t)bh * N * Kb;
  float* C = Call + (size_t)bh * M * N;
  constexpr int WM = 16, BM = AQ_WARPS * WM;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * AQ_BN, t = threadIdx.x;
  __shared__ int8_t As[AQ_STAGES][BM * AQ_LDS];
  __shared__ int8_t Bs[AQ_STAGES][AQ_BN * AQ_LDS];
  int acc[AQ_BN / 8][4];
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0; }
  const int nkt = Kb / 32;
#define QK4_LOAD(ktb, buf)                                                                       \
  for (int c = t; c < BM * 2; c += blockDim.x) {                                                 \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(A + (size_t)(m0 + r) * Kb + (ktb) + off), (m0 + r) < M);   \
  }                                                                                              \
  for (int c = t; c < AQ_BN * 2; c += blockDim.x) {                                              \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(B + (size_t)(n0 + r) * Kb + (ktb) + off), (n0 + r) < N);   \
  }
#pragma unroll
  for (int s = 0; s < AQ_STAGES - 1; ++s) { if (s < nkt) { QK4_LOAD(s * 32, s); } __pipeline_commit(); }
  __pipeline_wait_prior(AQ_STAGES - 2); __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % AQ_STAGES;
    unsigned a[4]; int rb = w * WM;
    a[0] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4];
    a[1] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4];
    a[2] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4 + 16];
    a[3] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4 + 16];
#pragma unroll
    for (int nt = 0; nt < AQ_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4];
      b[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4 + 16];
      modiff_mma_m16n8k64_s4(acc[nt], a, b);
    }
    int li = i + AQ_STAGES - 1;
    if (li < nkt) { QK4_LOAD(li * 32, li % AQ_STAGES); }
    __pipeline_commit(); __pipeline_wait_prior(AQ_STAGES - 2); __syncthreads();
  }
#undef QK4_LOAD
  int mwb = m0 + w * WM;
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
    if (r0 < M) { C[(size_t)r0 * N + c0] = (float)acc[nt][0]; C[(size_t)r0 * N + c0 + 1] = (float)acc[nt][1]; }
    if (r1 < M) { C[(size_t)r1 * N + c0] = (float)acc[nt][2]; C[(size_t)r1 * N + c0 + 1] = (float)acc[nt][3]; }
  }
}

torch::Tensor attn_qk_int4(torch::Tensor Q, torch::Tensor K, int64_t hd_pad) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kChar && K.dtype() == torch::kChar, "Q/K packed int4 CUDA");
  Q = Q.contiguous(); K = K.contiguous();
  int BH = Q.size(0), T = Q.size(1);
  TORCH_CHECK(Q.size(2) == hd_pad / 2 && K.size(2) == hd_pad / 2, "packed hd_pad/2 mismatch");
  TORCH_CHECK(hd_pad % 64 == 0 && T % AQ_BN == 0, "need hd_pad%64==0, T%64==0");
  auto S = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
  const int BM = AQ_WARPS * 16;
  dim3 grid((T + BM - 1) / BM, T / AQ_BN, BH);
  bmm_qk_s4_kernel<<<grid, AQ_WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(), S.data_ptr<float>(), T, T, (int)hd_pad);
  return S;
}

// softmax + requant to PACKED int4 P (round(exp·7) in [0,7], pack pairs) + per-row sp=1/(7·Σexp)
__global__ void attn_softmax_requant4_kernel(const float* __restrict__ S, const float* __restrict__ sq,
                                             const float* __restrict__ sk, int8_t* __restrict__ P,
                                             float* __restrict__ sp, int T, float scale) {
  const int row = blockIdx.x; const int bh = row / T;
  const float* Srow = S + (size_t)row * T;
  const float* skb = sk + (size_t)bh * T;
  int8_t* Prow = P + (size_t)row * (T / 2);            // packed
  const float sqi = sq[row] * scale;
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ float red[AQ_SM_THREADS];
  float m = -1e30f;
  for (int j = tid; j < T; j += nt) m = fmaxf(m, Srow[j] * sqi * skb[j]);
  red[tid] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float sum = 0.f;
  for (int j = tid; j < T; j += nt) sum += __expf(Srow[j] * sqi * skb[j] - m);
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  const float l = red[0];
  if (tid == 0) sp[row] = 1.f / (7.f * fmaxf(l, 1e-20f));
  for (int jp = tid; jp < T / 2; jp += nt) {           // one packed byte = 2 int4
    int j0 = jp * 2;
    int q0 = __float2int_rn(__expf(Srow[j0]     * sqi * skb[j0]     - m) * 7.f); q0 = q0 < 0 ? 0 : (q0 > 7 ? 7 : q0);
    int q1 = __float2int_rn(__expf(Srow[j0 + 1] * sqi * skb[j0 + 1] - m) * 7.f); q1 = q1 < 0 ? 0 : (q1 > 7 ? 7 : q1);
    Prow[jp] = (int8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
  }
}

std::vector<torch::Tensor> attn_softmax_requant4(torch::Tensor S, torch::Tensor sq, torch::Tensor sk,
                                                 double softmax_scale) {
  S = S.contiguous(); int BH = S.size(0), T = S.size(2);
  auto P = torch::empty({BH, T, T / 2}, torch::TensorOptions().dtype(torch::kChar).device(S.device()));
  auto sp = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  attn_softmax_requant4_kernel<<<BH * T, AQ_SM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.data_ptr<float>(), sq.contiguous().data_ptr<float>(), sk.contiguous().data_ptr<float>(),
      P.data_ptr<int8_t>(), sp.data_ptr<float>(), T, (float)softmax_scale);
  return {P, sp};
}

__global__ void bmm_av_s4_kernel(const int8_t* __restrict__ Pall, const int8_t* __restrict__ Vall,
                                 const float* __restrict__ spall, const float* __restrict__ svall,
                                 __half* __restrict__ Oall, int M, int N, int K) {
  const int bh = blockIdx.z; const int Kb = K >> 1;
  const int8_t* A = Pall + (size_t)bh * M * Kb;
  const int8_t* B = Vall + (size_t)bh * N * Kb;
  const float* sp = spall + (size_t)bh * M;
  const float* sv = svall + (size_t)bh * N;
  __half* C = Oall + (size_t)bh * M * N;
  constexpr int WM = 16, BM = AQ_WARPS * WM;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * AQ_BN, t = threadIdx.x;
  __shared__ int8_t As[AQ_STAGES][BM * AQ_LDS];
  __shared__ int8_t Bs[AQ_STAGES][AQ_BN * AQ_LDS];
  int acc[AQ_BN / 8][4];
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0; }
  const int nkt = Kb / 32;
#define AV4_LOAD(ktb, buf)                                                                       \
  for (int c = t; c < BM * 2; c += blockDim.x) {                                                 \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&As[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(A + (size_t)(m0 + r) * Kb + (ktb) + off), (m0 + r) < M);   \
  }                                                                                              \
  for (int c = t; c < AQ_BN * 2; c += blockDim.x) {                                              \
    int r = c >> 1, off = (c & 1) * 16;                                                          \
    modiff_cp_async_cg(modiff_smem_ptr(&Bs[buf][r * AQ_LDS + off]),                              \
                       (const uint4*)(B + (size_t)(n0 + r) * Kb + (ktb) + off), (n0 + r) < N);   \
  }
#pragma unroll
  for (int s = 0; s < AQ_STAGES - 1; ++s) { if (s < nkt) { AV4_LOAD(s * 32, s); } __pipeline_commit(); }
  __pipeline_wait_prior(AQ_STAGES - 2); __syncthreads();
  for (int i = 0; i < nkt; ++i) {
    const int buf = i % AQ_STAGES;
    unsigned a[4]; int rb = w * WM;
    a[0] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4];
    a[1] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4];
    a[2] = *(const int*)&As[buf][(rb + gid) * AQ_LDS + tig * 4 + 16];
    a[3] = *(const int*)&As[buf][(rb + gid + 8) * AQ_LDS + tig * 4 + 16];
#pragma unroll
    for (int nt = 0; nt < AQ_BN / 8; ++nt) {
      unsigned b[2];
      b[0] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4];
      b[1] = *(const int*)&Bs[buf][(nt * 8 + gid) * AQ_LDS + tig * 4 + 16];
      modiff_mma_m16n8k64_s4(acc[nt], a, b);
    }
    int li = i + AQ_STAGES - 1;
    if (li < nkt) { AV4_LOAD(li * 32, li % AQ_STAGES); }
    __pipeline_commit(); __pipeline_wait_prior(AQ_STAGES - 2); __syncthreads();
  }
#undef AV4_LOAD
  int mwb = m0 + w * WM;
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
    float v0 = sv[c0], v1 = sv[c0 + 1];
    if (r0 < M) { float s = sp[r0];
      C[(size_t)r0 * N + c0]     = __float2half(acc[nt][0] * s * v0);
      C[(size_t)r0 * N + c0 + 1] = __float2half(acc[nt][1] * s * v1); }
    if (r1 < M) { float s = sp[r1];
      C[(size_t)r1 * N + c0]     = __float2half(acc[nt][2] * s * v0);
      C[(size_t)r1 * N + c0 + 1] = __float2half(acc[nt][3] * s * v1); }
  }
}

// P packed int4 [BH,T,T/2], Vt packed int4 [BH,hd_pad,T/2], sp [BH,T], sv [BH,hd_pad] -> O fp16 [BH,T,hd_pad]
torch::Tensor attn_av_int4(torch::Tensor P, torch::Tensor Vt, torch::Tensor sp, torch::Tensor sv, int64_t T) {
  TORCH_CHECK(P.is_cuda() && P.dtype() == torch::kChar && Vt.dtype() == torch::kChar, "P/Vt packed int4 CUDA");
  P = P.contiguous(); Vt = Vt.contiguous();
  int BH = P.size(0), hd_pad = Vt.size(1);
  TORCH_CHECK(P.size(1) == T && P.size(2) == T / 2 && Vt.size(2) == T / 2, "packed shape mismatch");
  TORCH_CHECK(T % 64 == 0 && hd_pad % AQ_BN == 0, "need T%64==0, hd_pad%64==0");
  auto O = torch::empty({BH, (int)T, hd_pad}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
  const int BM = AQ_WARPS * 16;
  dim3 grid(((int)T + BM - 1) / BM, hd_pad / AQ_BN, BH);
  bmm_av_s4_kernel<<<grid, AQ_WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      P.data_ptr<int8_t>(), Vt.data_ptr<int8_t>(), sp.contiguous().data_ptr<float>(),
      sv.contiguous().data_ptr<float>(), reinterpret_cast<__half*>(O.data_ptr<at::Half>()), (int)T, hd_pad, (int)T);
  return O;
}
