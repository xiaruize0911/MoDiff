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

// ---- batched int8 QKᵀ -> fp16 SCALED scores: C[M,N] fp16 = acc·sq[r]·sk[c]·scale per batch.
// Applying the per-token scales in the epilogue keeps the dequantized logits in fp16 range
// (int32 raw would overflow fp16) and halves the T×T score IO vs an fp32 raw dump. ----
__global__ void bmm_qk_s8_kernel(const int8_t* __restrict__ Aall, const int8_t* __restrict__ Ball,
                                 __half* __restrict__ Call, int M, int N, int K,
                                 const float* __restrict__ sqall, const float* __restrict__ skall,
                                 float scale) {
  const int bh = blockIdx.z;
  const int8_t* A = Aall + (size_t)bh * M * K;
  const int8_t* B = Ball + (size_t)bh * N * K;
  __half* C = Call + (size_t)bh * M * N;
  const float* sqb = sqall + (size_t)bh * M;
  const float* skb = skall + (size_t)bh * N;
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
  // fp16 scaled logits S[r,c] = acc·sq[r]·sk[c]·scale. c0 even, c0+1<N (N%64==0) -> half2 store.
  int mwb = m0 + w * WM;
#pragma unroll
  for (int nt = 0; nt < AQ_BN / 8; ++nt) {
    int c0 = n0 + nt * 8 + tig * 2, r0 = mwb + gid, r1 = mwb + gid + 8;
    float k0 = skb[c0], k1 = skb[c0 + 1];
    if (r0 < M) { float q = sqb[r0] * scale;
      *(__half2*)&C[(size_t)r0 * N + c0] = __halves2half2(__float2half(acc[nt][0] * q * k0), __float2half(acc[nt][1] * q * k1)); }
    if (r1 < M) { float q = sqb[r1] * scale;
      *(__half2*)&C[(size_t)r1 * N + c0] = __halves2half2(__float2half(acc[nt][2] * q * k0), __float2half(acc[nt][3] * q * k1)); }
  }
}

// Q,K int8 [BH,T,hd_pad]; sq,sk [BH,T] per-token; scale=1/sqrt(hd). Returns S [BH,T,T] fp16
// (dequantized+scaled logits, ready for softmax).
torch::Tensor attn_qk_int8(torch::Tensor Q, torch::Tensor K, torch::Tensor sq, torch::Tensor sk, double scale) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kChar && K.dtype() == torch::kChar, "Q/K int8 CUDA");
  Q = Q.contiguous(); K = K.contiguous();
  int BH = Q.size(0), T = Q.size(1), hd_pad = Q.size(2);
  TORCH_CHECK(K.size(0) == BH && K.size(1) == T && K.size(2) == hd_pad, "Q/K shape mismatch");
  TORCH_CHECK(hd_pad % 32 == 0 && T % AQ_BN == 0, "need hd_pad%32==0, T%64==0");
  auto S = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
  const int BM = AQ_WARPS * 16;
  dim3 grid((T + BM - 1) / BM, T / AQ_BN, BH);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bmm_qk_s8_kernel<<<grid, AQ_WARPS * 32, 0, stream>>>(
      Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(),
      reinterpret_cast<__half*>(S.data_ptr<at::Half>()), T, T, hd_pad,
      sq.contiguous().data_ptr<float>(), sk.contiguous().data_ptr<float>(), (float)scale);
  return S;
}

// ---- softmax(dequant(S)) + requantize P to int8 ----
// One block per output row (bh,i). Dequant logit_j = S[i,j]·sq[i]·sk[j]·scale; online row-max/
// exp/sum in fp32; emit P_i8[j] = round(exp(logit_j - m)·127) ∈ [0,127] and per-row sp[i] =
// 1/(127·Σexp) so that p[i,j] = P_i8[i,j]·sp[i] (dequantized softmax prob). AV then applies sp[i].
#define AQ_SM_THREADS 256
// Load 8 fp16 logits (one 128-bit uint4) into float f[8].
__device__ __forceinline__ void aq_ld8(const uint4* p, float f[8]) {
  uint4 v = *p;
  float2 a = __half22float2(*(const __half2*)&v.x), b = __half22float2(*(const __half2*)&v.y);
  float2 c = __half22float2(*(const __half2*)&v.z), d = __half22float2(*(const __half2*)&v.w);
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y; f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}
// Input S is fp16 pre-scaled logits (QKᵀ epilogue already applied sq·sk·scale). Two passes over
// the T×T row (down from three): (1) 128-bit-vectorized row max; (2) fused exp -> accumulate Σexp
// AND write int8 P[0,127] in the same pass (P needs only the max, not the sum — the 1/(127·Σexp)
// denominator is folded into per-row sp and applied later in AV). Requires T%8==0.
__global__ void attn_softmax_requant_kernel(const __half* __restrict__ S, int8_t* __restrict__ P,
                                            float* __restrict__ sp, int T) {
  const int row = blockIdx.x;
  const uint4* Srow = reinterpret_cast<const uint4*>(S + (size_t)row * T);
  int2* Prow = reinterpret_cast<int2*>(P + (size_t)row * T);      // 8 int8 per store
  const int tid = threadIdx.x, nt = blockDim.x, T8 = T >> 3;
  __shared__ float red[AQ_SM_THREADS];
  float f[8], m = -1e30f;
  for (int c = tid; c < T8; c += nt) { aq_ld8(Srow + c, f);
#pragma unroll
    for (int k = 0; k < 8; ++k) m = fmaxf(m, f[k]); }
  red[tid] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float sum = 0.f;
  for (int c = tid; c < T8; c += nt) { aq_ld8(Srow + c, f);
    int b0 = 0, b1 = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float e = __expf(f[k] - m); sum += e;
      int q = __float2int_rn(e * 127.f); q = q < 0 ? 0 : (q > 127 ? 127 : q);
      if (k < 4) b0 |= q << (k * 8); else b1 |= q << ((k - 4) * 8);
    }
    Prow[c] = make_int2(b0, b1);
  }
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) sp[row] = 1.f / (127.f * fmaxf(red[0], 1e-20f));
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

// S [BH,T,T] fp16 pre-scaled logits -> {P int8 [BH,T,T], sp [BH,T]}
std::vector<torch::Tensor> attn_softmax_requant(torch::Tensor S) {
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kHalf && S.dim() == 3, "S fp16 [BH,T,T]");
  S = S.contiguous();
  int BH = S.size(0), T = S.size(2);
  TORCH_CHECK(S.size(1) == T, "S must be [BH,T,T]");
  TORCH_CHECK(T % 8 == 0, "softmax needs T%8==0 (vectorized 128-bit rows)");
  auto P = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kChar).device(S.device()));
  auto sp = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  attn_softmax_requant_kernel<<<BH * T, AQ_SM_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()),
      P.data_ptr<int8_t>(), sp.data_ptr<float>(), T);
  return {P, sp};
}

// ============================ int4 (W4A4) score path ============================
// Packed int4 (2/byte). QKᵀ: Q,K [BH,T,hd_pad/2]; AV: P [BH,T,T/2], Vt [BH,hd_pad,T/2].
// Mirrors the int8 kernels with modiff_mma_m16n8k64_s4 and Kb=Klogical/2. int4 scores are
// aggressive (P has 8 levels) -> quality reported, not gated.
__global__ void bmm_qk_s4_kernel(const int8_t* __restrict__ Aall, const int8_t* __restrict__ Ball,
                                 __half* __restrict__ Call, int M, int N, int K,
                                 const float* __restrict__ sqall, const float* __restrict__ skall,
                                 float scale) {
  const int bh = blockIdx.z; const int Kb = K >> 1;
  const int8_t* A = Aall + (size_t)bh * M * Kb;
  const int8_t* B = Ball + (size_t)bh * N * Kb;
  __half* C = Call + (size_t)bh * M * N;
  const float* sqb = sqall + (size_t)bh * M;
  const float* skb = skall + (size_t)bh * N;
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
    float k0 = skb[c0], k1 = skb[c0 + 1];
    if (r0 < M) { float q = sqb[r0] * scale;
      *(__half2*)&C[(size_t)r0 * N + c0] = __halves2half2(__float2half(acc[nt][0] * q * k0), __float2half(acc[nt][1] * q * k1)); }
    if (r1 < M) { float q = sqb[r1] * scale;
      *(__half2*)&C[(size_t)r1 * N + c0] = __halves2half2(__float2half(acc[nt][2] * q * k0), __float2half(acc[nt][3] * q * k1)); }
  }
}

torch::Tensor attn_qk_int4(torch::Tensor Q, torch::Tensor K, int64_t hd_pad,
                           torch::Tensor sq, torch::Tensor sk, double scale) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kChar && K.dtype() == torch::kChar, "Q/K packed int4 CUDA");
  Q = Q.contiguous(); K = K.contiguous();
  int BH = Q.size(0), T = Q.size(1);
  TORCH_CHECK(Q.size(2) == hd_pad / 2 && K.size(2) == hd_pad / 2, "packed hd_pad/2 mismatch");
  TORCH_CHECK(hd_pad % 64 == 0 && T % AQ_BN == 0, "need hd_pad%64==0, T%64==0");
  auto S = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
  const int BM = AQ_WARPS * 16;
  dim3 grid((T + BM - 1) / BM, T / AQ_BN, BH);
  bmm_qk_s4_kernel<<<grid, AQ_WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(),
      reinterpret_cast<__half*>(S.data_ptr<at::Half>()), T, T, (int)hd_pad,
      sq.contiguous().data_ptr<float>(), sk.contiguous().data_ptr<float>(), (float)scale);
  return S;
}

// softmax + requant to PACKED int4 P (round(exp·7) in [0,7], pack pairs) + per-row sp=1/(7·Σexp)
// int4 variant: two vectorized passes; pass 2 fuses exp/Σexp and packs 8 int4 P[0,7] into one
// 32-bit store (byte b = q_2b | q_{2b+1}<<4, matching the AV unpack). Requires T%8==0.
__global__ void attn_softmax_requant4_kernel(const __half* __restrict__ S, int8_t* __restrict__ P,
                                             float* __restrict__ sp, int T) {
  const int row = blockIdx.x;
  const uint4* Srow = reinterpret_cast<const uint4*>(S + (size_t)row * T);
  int* Prow = reinterpret_cast<int*>(P + (size_t)row * (T / 2));  // 8 int4 (4 bytes) per store
  const int tid = threadIdx.x, nt = blockDim.x, T8 = T >> 3;
  __shared__ float red[AQ_SM_THREADS];
  float f[8], m = -1e30f;
  for (int c = tid; c < T8; c += nt) { aq_ld8(Srow + c, f);
#pragma unroll
    for (int k = 0; k < 8; ++k) m = fmaxf(m, f[k]); }
  red[tid] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float sum = 0.f;
  for (int c = tid; c < T8; c += nt) { aq_ld8(Srow + c, f);
    int packed = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float e = __expf(f[k] - m); sum += e;
      int q = __float2int_rn(e * 7.f); q = q < 0 ? 0 : (q > 7 ? 7 : q);
      packed |= (q & 0xF) << (k * 4);
    }
    Prow[c] = packed;
  }
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) sp[row] = 1.f / (7.f * fmaxf(red[0], 1e-20f));
}

std::vector<torch::Tensor> attn_softmax_requant4(torch::Tensor S) {
  TORCH_CHECK(S.dtype() == torch::kHalf, "S fp16"); S = S.contiguous();
  int BH = S.size(0), T = S.size(2);
  TORCH_CHECK(T % 8 == 0, "softmax4 needs T%8==0 (vectorized 128-bit rows)");
  auto P = torch::empty({BH, T, T / 2}, torch::TensorOptions().dtype(torch::kChar).device(S.device()));
  auto sp = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  attn_softmax_requant4_kernel<<<BH * T, AQ_SM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), P.data_ptr<int8_t>(), sp.data_ptr<float>(), T);
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

// ============================ fused Q/K/V quantize ============================
// Replaces the PyTorch per-token/per-channel quantize (~+5ms elementwise). Q,K per-token
// int8/int4 -> [BH,T,hp_qk]; V per-channel-over-T, transposed to channel-major [BH,hp_av,T]
// (the AV GEMM's B operand), int4 packed. Emits sq,sk [BH,T], sv [BH,hp_av].
template <int BITS>
__global__ void aq_qtok_kernel(const __half* __restrict__ X, int8_t* __restrict__ out,
                               float* __restrict__ sc, int hd, int hp) {
  const int r = blockIdx.x, lane = threadIdx.x;         // one warp per (bh,t) row
  const __half* xr = X + (size_t)r * hd;
  float amax = 0.f;
  for (int d = lane; d < hd; d += 32) amax = fmaxf(amax, fabsf(__half2float(xr[d])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
  amax = __shfl_sync(0xffffffff, amax, 0);
  const float Qm = (BITS == 8) ? 127.f : 7.f;
  const float scale = fmaxf(amax, 1e-8f) / Qm, inv = 1.f / scale;
  if (lane == 0) sc[r] = scale;
  if (BITS == 8) {
    int8_t* o8 = out + (size_t)r * hp;
    for (int d = lane; d < hp; d += 32) {
      int q = __float2int_rn((d < hd ? __half2float(xr[d]) * inv : 0.f));
      o8[d] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
  } else {
    int8_t* o4 = out + (size_t)r * (hp / 2);
    for (int dp = lane; dp < hp / 2; dp += 32) {
      int d0 = dp * 2;
      int q0 = __float2int_rn((d0 < hd ? __half2float(xr[d0]) * inv : 0.f)); q0 = q0 > 7 ? 7 : (q0 < -7 ? -7 : q0);
      int q1 = __float2int_rn((d0 + 1 < hd ? __half2float(xr[d0 + 1]) * inv : 0.f)); q1 = q1 > 7 ? 7 : (q1 < -7 ? -7 : q1);
      o4[dp] = (int8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  }
}

__global__ void aq_vscale_kernel(const __half* __restrict__ V, float* __restrict__ sv,
                                 int T, int hd, int hp_av, float Qm) {
  const int bd = blockIdx.x, bh = bd / hd, d = bd % hd;   // grid = BH*hd
  const __half* base = V + ((size_t)bh * T) * hd + d;
  const int tid = threadIdx.x, nt = blockDim.x;
  float amax = 0.f;
  for (int t = tid; t < T; t += nt) amax = fmaxf(amax, fabsf(__half2float(base[(size_t)t * hd])));
  __shared__ float red[256]; red[tid] = amax; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  if (tid == 0) sv[(size_t)bh * hp_av + d] = fmaxf(red[0], 1e-8f) / Qm;
}

template <int BITS>
__global__ void aq_vquant_trans_kernel(const __half* __restrict__ V, const float* __restrict__ sv,
                                       int8_t* __restrict__ vt, int T, int hd, int hp_av) {
  const int rd = blockIdx.x, bh = rd / hp_av, d = rd % hp_av;   // grid = BH*hp_av (channel-major rows)
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv = (d < hd) ? 1.f / sv[(size_t)bh * hp_av + d] : 0.f;
  const __half* vc = V + ((size_t)bh * T) * hd + d;
  if (BITS == 8) {
    int8_t* o = vt + (size_t)rd * T;
    for (int t = tid; t < T; t += nt) {
      int q = __float2int_rn((d < hd ? __half2float(vc[(size_t)t * hd]) * inv : 0.f));
      o[t] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
  } else {
    int8_t* o = vt + (size_t)rd * (T / 2);
    for (int tp = tid; tp < T / 2; tp += nt) {
      int t0 = tp * 2;
      int q0 = __float2int_rn((d < hd ? __half2float(vc[(size_t)t0 * hd]) * inv : 0.f)); q0 = q0 > 7 ? 7 : (q0 < -7 ? -7 : q0);
      int q1 = __float2int_rn((d < hd ? __half2float(vc[(size_t)(t0 + 1) * hd]) * inv : 0.f)); q1 = q1 > 7 ? 7 : (q1 < -7 ? -7 : q1);
      o[tp] = (int8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  }
}

// Q,K,V fp16 [BH,T,hd] -> {qi,ki [BH,T,hp_qk(/2)], vt [BH,hp_av,T(/2)], sq,sk [BH,T], sv [BH,hp_av]}
std::vector<torch::Tensor> quantize_attn_qkv(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                             int64_t hp_qk, int64_t hp_av, int64_t bits) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf, "Q/K/V fp16 CUDA");
  Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
  int BH = Q.size(0), T = Q.size(1), hd = Q.size(2);
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(Q.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  int qkw = (bits == 8) ? (int)hp_qk : (int)hp_qk / 2;
  int vtw = (bits == 8) ? T : T / 2;
  auto qi = torch::empty({BH, T, qkw}, oi), ki = torch::empty({BH, T, qkw}, oi);
  auto vt = torch::empty({BH, (int)hp_av, vtw}, oi);
  auto sq = torch::empty({BH, T}, of), sk = torch::empty({BH, T}, of);
  auto sv = torch::zeros({BH, (int)hp_av}, of);
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half* Qp = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
  const __half* Kp = reinterpret_cast<const __half*>(K.data_ptr<at::Half>());
  const __half* Vp = reinterpret_cast<const __half*>(V.data_ptr<at::Half>());
  float Qm = (bits == 8) ? 127.f : 7.f;
  if (bits == 8) {
    aq_qtok_kernel<8><<<BH * T, 32, 0, s>>>(Qp, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), hd, (int)hp_qk);
    aq_qtok_kernel<8><<<BH * T, 32, 0, s>>>(Kp, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), hd, (int)hp_qk);
    aq_vscale_kernel<<<BH * hd, 256, 0, s>>>(Vp, sv.data_ptr<float>(), T, hd, (int)hp_av, Qm);
    aq_vquant_trans_kernel<8><<<BH * (int)hp_av, 256, 0, s>>>(Vp, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), T, hd, (int)hp_av);
  } else {
    aq_qtok_kernel<4><<<BH * T, 32, 0, s>>>(Qp, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), hd, (int)hp_qk);
    aq_qtok_kernel<4><<<BH * T, 32, 0, s>>>(Kp, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), hd, (int)hp_qk);
    aq_vscale_kernel<<<BH * hd, 256, 0, s>>>(Vp, sv.data_ptr<float>(), T, hd, (int)hp_av, Qm);
    aq_vquant_trans_kernel<4><<<BH * (int)hp_av, 256, 0, s>>>(Vp, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), T, hd, (int)hp_av);
  }
  return {qi, ki, vt, sq, sk, sv};
}

// ==================== STATIC score path (calibrated, no runtime reduction) ====================
// Static softmax: the per-row max is replaced by a calibrated per-layer constant c (a safe upper
// bound on the logits). This drops pass 1 entirely -> single read of the T×T score matrix (vs 2
// for the dynamic kernel). softmax is shift-invariant so exp(S−c) is exact for c ≥ row max; the
// only int8 cost is that a row whose max < c maps its peak to exp(max−c)·127 < 127 (coarser
// resolution). fp32 exp gives ~88 units of headroom, so a generous c never overflows. Σexp is
// still data-dependent and folded into sp (applied in AV) exactly as the dynamic kernel does.
__global__ void attn_softmax_requant_static_kernel(const __half* __restrict__ S, int8_t* __restrict__ P,
                                                   float* __restrict__ sp, int T, float c) {
  const int row = blockIdx.x;
  const uint4* Srow = reinterpret_cast<const uint4*>(S + (size_t)row * T);
  int2* Prow = reinterpret_cast<int2*>(P + (size_t)row * T);
  const int tid = threadIdx.x, nt = blockDim.x, T8 = T >> 3;
  __shared__ float red[AQ_SM_THREADS];
  float f[8], sum = 0.f;
  for (int cc = tid; cc < T8; cc += nt) { aq_ld8(Srow + cc, f);
    int b0 = 0, b1 = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float e = __expf(f[k] - c); sum += e;
      int q = __float2int_rn(e * 127.f); q = q < 0 ? 0 : (q > 127 ? 127 : q);
      if (k < 4) b0 |= q << (k * 8); else b1 |= q << ((k - 4) * 8);
    }
    Prow[cc] = make_int2(b0, b1);
  }
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) sp[row] = 1.f / (127.f * fmaxf(red[0], 1e-20f));
}

std::vector<torch::Tensor> attn_softmax_requant_static(torch::Tensor S, double c) {
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kHalf && S.dim() == 3, "S fp16 [BH,T,T]");
  S = S.contiguous();
  int BH = S.size(0), T = S.size(2);
  TORCH_CHECK(S.size(1) == T, "S must be [BH,T,T]");
  TORCH_CHECK(T % 8 == 0, "softmax needs T%8==0 (vectorized 128-bit rows)");
  auto P = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kChar).device(S.device()));
  auto sp = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  attn_softmax_requant_static_kernel<<<BH * T, AQ_SM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), P.data_ptr<int8_t>(), sp.data_ptr<float>(), T, (float)c);
  return {P, sp};
}

__global__ void attn_softmax_requant4_static_kernel(const __half* __restrict__ S, int8_t* __restrict__ P,
                                                    float* __restrict__ sp, int T, float c) {
  const int row = blockIdx.x;
  const uint4* Srow = reinterpret_cast<const uint4*>(S + (size_t)row * T);
  int* Prow = reinterpret_cast<int*>(P + (size_t)row * (T / 2));
  const int tid = threadIdx.x, nt = blockDim.x, T8 = T >> 3;
  __shared__ float red[AQ_SM_THREADS];
  float f[8], sum = 0.f;
  for (int cc = tid; cc < T8; cc += nt) { aq_ld8(Srow + cc, f);
    int packed = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float e = __expf(f[k] - c); sum += e;
      int q = __float2int_rn(e * 7.f); q = q < 0 ? 0 : (q > 7 ? 7 : q);
      packed |= (q & 0xF) << (k * 4);
    }
    Prow[cc] = packed;
  }
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) sp[row] = 1.f / (7.f * fmaxf(red[0], 1e-20f));
}

std::vector<torch::Tensor> attn_softmax_requant4_static(torch::Tensor S, double c) {
  TORCH_CHECK(S.dtype() == torch::kHalf, "S fp16"); S = S.contiguous();
  int BH = S.size(0), T = S.size(2);
  TORCH_CHECK(T % 8 == 0, "softmax4 needs T%8==0 (vectorized 128-bit rows)");
  auto P = torch::empty({BH, T, T / 2}, torch::TensorOptions().dtype(torch::kChar).device(S.device()));
  auto sp = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  attn_softmax_requant4_static_kernel<<<BH * T, AQ_SM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(S.data_ptr<at::Half>()), P.data_ptr<int8_t>(), sp.data_ptr<float>(), T, (float)c);
  return {P, sp};
}

// ---- fp16 materialized softmax (no quantization): emits UNNORMALIZED exp weights + per-row sum.
// Caller computes O = bmm(P, V) then O /= rowsum (rowsum is per-row of P = per-row of O). This lets
// the fp16 attention share ONE materialized path with a dynamic (2-pass, per-row max) vs static
// (1-pass, calibrated c) softmax, so the static-vs-dynamic delta is measured with no other change. ----
__global__ void attn_softmax_fp16_dynamic_kernel(const __half* __restrict__ S, __half* __restrict__ P,
                                                 float* __restrict__ rowsum, int T) {
  const int row = blockIdx.x;
  const uint4* Srow = reinterpret_cast<const uint4*>(S + (size_t)row * T);
  uint4* Prow = reinterpret_cast<uint4*>(P + (size_t)row * T);
  const int tid = threadIdx.x, nt = blockDim.x, T8 = T >> 3;
  __shared__ float red[AQ_SM_THREADS];
  float f[8], m = -1e30f;
  for (int cc = tid; cc < T8; cc += nt) { aq_ld8(Srow + cc, f);
#pragma unroll
    for (int k = 0; k < 8; ++k) m = fmaxf(m, f[k]); }
  red[tid] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  m = red[0]; __syncthreads();
  float sum = 0.f;
  for (int cc = tid; cc < T8; cc += nt) { aq_ld8(Srow + cc, f);
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float e0 = __expf(f[2 * k] - m), e1 = __expf(f[2 * k + 1] - m);
      sum += e0 + e1; h[k] = __floats2half2_rn(e0, e1);
    }
    Prow[cc] = *reinterpret_cast<uint4*>(h);
  }
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) rowsum[row] = fmaxf(red[0], 1e-20f);
}

__global__ void attn_softmax_fp16_static_kernel(const __half* __restrict__ S, __half* __restrict__ P,
                                                float* __restrict__ rowsum, int T, float c) {
  const int row = blockIdx.x;
  const uint4* Srow = reinterpret_cast<const uint4*>(S + (size_t)row * T);
  uint4* Prow = reinterpret_cast<uint4*>(P + (size_t)row * T);
  const int tid = threadIdx.x, nt = blockDim.x, T8 = T >> 3;
  __shared__ float red[AQ_SM_THREADS];
  float f[8], sum = 0.f;
  for (int cc = tid; cc < T8; cc += nt) { aq_ld8(Srow + cc, f);
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      // clamp exp to 1.0: with c >= row max (well-calibrated) exp(S-c) <= 1 already, so this is a
      // no-op and the softmax is exact. When a static c under-estimates the logits at some timestep
      // (diffusion logit scale drifts ~2-3x over the trajectory), unnormalized P would exceed 1 and
      // the fp16 P@V bmm (summing T terms) overflows to inf -> NaN. Clamping to 1 keeps P in [0,1]
      // (sum <= T, bmm-safe); over-max elements tie at 1 (a bounded convex-combination fallback).
      float e0 = fminf(__expf(f[2 * k] - c), 1.0f), e1 = fminf(__expf(f[2 * k + 1] - c), 1.0f);
      sum += e0 + e1; h[k] = __floats2half2_rn(e0, e1);
    }
    Prow[cc] = *reinterpret_cast<uint4*>(h);
  }
  red[tid] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  if (tid == 0) rowsum[row] = fmaxf(red[0], 1e-20f);
}

std::vector<torch::Tensor> attn_softmax_fp16(torch::Tensor S, bool static_c, double c) {
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kHalf && S.dim() == 3, "S fp16 [BH,T,T]");
  S = S.contiguous();
  int BH = S.size(0), T = S.size(2);
  TORCH_CHECK(S.size(1) == T && T % 8 == 0, "S [BH,T,T], T%8==0");
  auto P = torch::empty({BH, T, T}, torch::TensorOptions().dtype(torch::kHalf).device(S.device()));
  auto rs = torch::empty({BH, T}, torch::TensorOptions().dtype(torch::kFloat32).device(S.device()));
  const __half* Sp = reinterpret_cast<const __half*>(S.data_ptr<at::Half>());
  __half* Pp = reinterpret_cast<__half*>(P.data_ptr<at::Half>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (static_c)
    attn_softmax_fp16_static_kernel<<<BH * T, AQ_SM_THREADS, 0, st>>>(Sp, Pp, rs.data_ptr<float>(), T, (float)c);
  else
    attn_softmax_fp16_dynamic_kernel<<<BH * T, AQ_SM_THREADS, 0, st>>>(Sp, Pp, rs.data_ptr<float>(), T);
  return {P, rs};
}

// ---- static Q/K/V quantize: calibrated per-tensor sq_c/sk_c + per-channel sv_vec, no reduction ----
template <int BITS>
__global__ void aq_qtok_static_kernel(const __half* __restrict__ X, int8_t* __restrict__ out,
                                      float* __restrict__ sc, int hd, int hp, float scale) {
  const int r = blockIdx.x, lane = threadIdx.x;
  const __half* xr = X + (size_t)r * hd;
  const float inv = 1.f / scale;
  if (lane == 0) sc[r] = scale;
  if (BITS == 8) {
    int8_t* o8 = out + (size_t)r * hp;
    for (int d = lane; d < hp; d += 32) {
      int q = __float2int_rn((d < hd ? __half2float(xr[d]) * inv : 0.f));
      o8[d] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
  } else {
    int8_t* o4 = out + (size_t)r * (hp / 2);
    for (int dp = lane; dp < hp / 2; dp += 32) {
      int d0 = dp * 2;
      int q0 = __float2int_rn((d0 < hd ? __half2float(xr[d0]) * inv : 0.f)); q0 = q0 > 7 ? 7 : (q0 < -7 ? -7 : q0);
      int q1 = __float2int_rn((d0 + 1 < hd ? __half2float(xr[d0 + 1]) * inv : 0.f)); q1 = q1 > 7 ? 7 : (q1 < -7 ? -7 : q1);
      o4[dp] = (int8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  }
}

// sv_vec: [hp_av] fp32 calibrated per-channel V scale (shared across BH). sq_c/sk_c: per-tensor scalars.
std::vector<torch::Tensor> quantize_attn_qkv_static(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                    int64_t hp_qk, int64_t hp_av, int64_t bits,
                                                    double sq_c, double sk_c, torch::Tensor sv_vec) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf, "Q/K/V fp16 CUDA");
  Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
  int BH = Q.size(0), T = Q.size(1), hd = Q.size(2);
  TORCH_CHECK(sv_vec.numel() == hp_av, "sv_vec must be [hp_av]");
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(Q.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  int qkw = (bits == 8) ? (int)hp_qk : (int)hp_qk / 2;
  int vtw = (bits == 8) ? T : T / 2;
  auto qi = torch::empty({BH, T, qkw}, oi), ki = torch::empty({BH, T, qkw}, oi);
  auto vt = torch::empty({BH, (int)hp_av, vtw}, oi);
  auto sq = torch::empty({BH, T}, of), sk = torch::empty({BH, T}, of);
  auto sv = sv_vec.to(of).view({1, (int)hp_av}).expand({BH, (int)hp_av}).contiguous();
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half* Qp = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
  const __half* Kp = reinterpret_cast<const __half*>(K.data_ptr<at::Half>());
  const __half* Vp = reinterpret_cast<const __half*>(V.data_ptr<at::Half>());
  if (bits == 8) {
    aq_qtok_static_kernel<8><<<BH * T, 32, 0, s>>>(Qp, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), hd, (int)hp_qk, (float)sq_c);
    aq_qtok_static_kernel<8><<<BH * T, 32, 0, s>>>(Kp, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), hd, (int)hp_qk, (float)sk_c);
    aq_vquant_trans_kernel<8><<<BH * (int)hp_av, 256, 0, s>>>(Vp, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), T, hd, (int)hp_av);
  } else {
    aq_qtok_static_kernel<4><<<BH * T, 32, 0, s>>>(Qp, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), hd, (int)hp_qk, (float)sq_c);
    aq_qtok_static_kernel<4><<<BH * T, 32, 0, s>>>(Kp, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), hd, (int)hp_qk, (float)sk_c);
    aq_vquant_trans_kernel<4><<<BH * (int)hp_av, 256, 0, s>>>(Vp, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), T, hd, (int)hp_av);
  }
  return {qi, ki, vt, sq, sk, sv};
}

// ==================== int8-OUTPUT qkv -> attention quantize fusion (prototype) ====================
// Consume the int8 qkv-linear output directly (gemm_w8a8_out_int8), skipping the fp16 round-trip:
// the qkv-linear emits int8 [M=b*T, 3C] with per-column oscale (true value = i8 / oscale[n]); this
// kernel dequants on the fly and re-quantizes to attention granularity (per-token Q/K, per-channel V),
// producing the same {qi,ki,vt,sq,sk,sv} as quantize_attn_qkv. 3C is ordered (nh,3,hd) so output
// column n(h,which,d) = (h*3+which)*hd+d. int8 (W8A8) prototype only.
__global__ void aq_qtok_from_i8_kernel(const int8_t* __restrict__ QKV, const float* __restrict__ iosc,
                                       int8_t* __restrict__ out, float* __restrict__ sc,
                                       int nh, int hd, int hp, int T, int C3, int which) {
  const int r = blockIdx.x, lane = threadIdx.x;      // r = bh*T + t over BH*T
  const int bh = r / T, t = r % T, b = bh / nh, h = bh % nh;
  const size_t in_base = (size_t)(b * T + t) * C3 + (size_t)(h * 3 + which) * hd;
  const int os_base = (h * 3 + which) * hd;
  float amax = 0.f;
  for (int d = lane; d < hd; d += 32)
    amax = fmaxf(amax, fabsf((float)QKV[in_base + d] * iosc[os_base + d]));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
  amax = __shfl_sync(0xffffffff, amax, 0);
  const float scale = fmaxf(amax, 1e-8f) / 127.f, inv = 1.f / scale;
  if (lane == 0) sc[r] = scale;
  int8_t* o8 = out + (size_t)r * hp;
  for (int d = lane; d < hp; d += 32) {
    float v = (d < hd) ? (float)QKV[in_base + d] * iosc[os_base + d] * inv : 0.f;
    int q = __float2int_rn(v);
    o8[d] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
}

__global__ void aq_vscale_from_i8_kernel(const int8_t* __restrict__ QKV, const float* __restrict__ iosc,
                                         float* __restrict__ sv, int nh, int hd, int hp_av, int T, int C3) {
  const int bd = blockIdx.x, bh = bd / hd, d = bd % hd, b = bh / nh, h = bh % nh;   // grid = BH*hd
  const int os = (h * 3 + 2) * hd + d;
  const int tid = threadIdx.x, nt = blockDim.x;
  float amax = 0.f;
  for (int t = tid; t < T; t += nt)
    amax = fmaxf(amax, fabsf((float)QKV[(size_t)(b * T + t) * C3 + (h * 3 + 2) * hd + d] * iosc[os]));
  __shared__ float red[256]; red[tid] = amax; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  if (tid == 0) sv[(size_t)bh * hp_av + d] = fmaxf(red[0], 1e-8f) / 127.f;
}

__global__ void aq_vquant_trans_from_i8_kernel(const int8_t* __restrict__ QKV, const float* __restrict__ iosc,
                                               const float* __restrict__ sv, int8_t* __restrict__ vt,
                                               int nh, int hd, int hp_av, int T, int C3) {
  const int rd = blockIdx.x, bh = rd / hp_av, d = rd % hp_av, b = bh / nh, h = bh % nh;  // grid = BH*hp_av
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv = (d < hd) ? 1.f / sv[(size_t)bh * hp_av + d] : 0.f;
  const int os = (h * 3 + 2) * hd + d;
  int8_t* o = vt + (size_t)rd * T;
  for (int t = tid; t < T; t += nt) {
    float v = (d < hd) ? (float)QKV[(size_t)(b * T + t) * C3 + (h * 3 + 2) * hd + d] * iosc[os] * inv : 0.f;
    int q = __float2int_rn(v);
    o[t] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
}

// qkv_i8 [b*T, 3C] int8 (from gemm_w8a8_out_int8), oscale [3C] -> {qi,ki,vt,sq,sk,sv} (int8 attention)
std::vector<torch::Tensor> quantize_attn_qkv_from_i8(torch::Tensor qkv_i8, torch::Tensor oscale,
                                                     int64_t nh, int64_t T, int64_t hp_qk, int64_t hp_av) {
  TORCH_CHECK(qkv_i8.is_cuda() && qkv_i8.dtype() == torch::kChar, "qkv_i8 int8 CUDA");
  qkv_i8 = qkv_i8.contiguous(); oscale = oscale.contiguous();
  int M = qkv_i8.size(0), C3 = qkv_i8.size(1);
  int b = M / (int)T, hd = C3 / (3 * (int)nh), BH = b * (int)nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(qkv_i8.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(qkv_i8.device());
  auto qi = torch::empty({BH, (int)T, (int)hp_qk}, oi), ki = torch::empty({BH, (int)T, (int)hp_qk}, oi);
  auto vt = torch::empty({BH, (int)hp_av, (int)T}, oi);
  auto sq = torch::empty({BH, (int)T}, of), sk = torch::empty({BH, (int)T}, of);
  auto sv = torch::zeros({BH, (int)hp_av}, of);
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const int8_t* Qp = qkv_i8.data_ptr<int8_t>(); auto ioscale = oscale.reciprocal().contiguous();
  const float* Ip = ioscale.data_ptr<float>();
  aq_qtok_from_i8_kernel<<<BH * (int)T, 32, 0, s>>>(Qp, Ip, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), (int)nh, hd, (int)hp_qk, (int)T, C3, 0);
  aq_qtok_from_i8_kernel<<<BH * (int)T, 32, 0, s>>>(Qp, Ip, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), (int)nh, hd, (int)hp_qk, (int)T, C3, 1);
  aq_vscale_from_i8_kernel<<<BH * hd, 256, 0, s>>>(Qp, Ip, sv.data_ptr<float>(), (int)nh, hd, (int)hp_av, (int)T, C3);
  aq_vquant_trans_from_i8_kernel<<<BH * (int)hp_av, 256, 0, s>>>(Qp, Ip, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), (int)nh, hd, (int)hp_av, (int)T, C3);
  return {qi, ki, vt, sq, sk, sv};
}
