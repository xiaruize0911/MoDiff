// =========================================================================
// Quantize prologue for the FUSED (flash) quantized attention path.
//   quantize_attn_qkv[_static] / _packed[_static] / _i4qk_i8v[_static]:
//     fp16 Q/K/V -> per-token int8/int4 Q/K + per-channel int8 (transposed) V + scales,
//     feeding flash_attn_int8_vt / flash_attn_int4_vt (csrc/kernels/attention/flash_attn_int8.cu).
//   The "packed" variants read the interleaved qkv [b,T,nh,3,hd] directly (no fp16 transpose copy);
//   the "static" variants take frozen calibrated scales (no runtime amax reduction).
// (The materialized int8/int4 attention score path — QKᵀ/softmax/AV int GEMMs — was removed;
//  flash is the sole quantized-attention path.)
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "common.cuh"

#define AQ_WARPS 4
#define AQ_BN 64
#define AQ_STAGES 3
#define AQ_LDS 32          // smem row stride (bytes); dense int8


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

// ==================== PACKED-QKV quantize (reads interleaved qkv, no transpose copy) ====================
// These read Q/K/V straight from the fused qkv-linear output [b,T,nh,3,hd] (contiguous, sel=0/1/2)
// instead of pre-transposed contiguous [BH,T,hd]. That removes the fp16 q/k/v.transpose().contiguous()
// copy (~1.2 GB/step at b128) AND the python unbind/transpose: the row->(b,h,t) decode + interleaved
// read is folded into the quantize sweep the kernel already does. Output layout is IDENTICAL to the
// non-packed kernels (qi/ki head-major [BH,T,hp], vt channel-major [BH,hp_av,T]).
//   row r = blockIdx.x = bh*T + t, bh = b*nh + h ; qkv flat idx = (((b*T+t)*nh+h)*3+sel)*hd + d
__device__ __forceinline__ size_t pk_row_off(int r, int nh, int T, int hd, int sel) {
  int t = r % T, bh = r / T, h = bh % nh, b = bh / nh;
  return ((((size_t)b * T + t) * nh + h) * 3 + sel) * (size_t)hd;
}
// Q/K per-token quantize from packed qkv (dynamic absmax). Same body as aq_qtok_kernel, packed read.
template <int BITS>
__global__ void aq_qtok_packed_kernel(const __half* __restrict__ QKV, int8_t* __restrict__ out,
                                      float* __restrict__ sc, int nh, int T, int hd, int hp, int sel) {
  const int r = blockIdx.x, lane = threadIdx.x;
  const __half* xr = QKV + pk_row_off(r, nh, T, hd, sel);
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
// Q/K per-token quantize from packed qkv (static/calibrated scale). Mirrors aq_qtok_static_kernel.
template <int BITS>
__global__ void aq_qtok_packed_static_kernel(const __half* __restrict__ QKV, int8_t* __restrict__ out,
                                             float* __restrict__ sc, int nh, int T, int hd, int hp,
                                             float scale, int sel) {
  const int r = blockIdx.x, lane = threadIdx.x;
  const __half* xr = QKV + pk_row_off(r, nh, T, hd, sel);
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
// V per-channel (over T) absmax from packed qkv (sel=2). Mirrors aq_vscale_kernel.
__global__ void aq_vscale_packed_kernel(const __half* __restrict__ QKV, float* __restrict__ sv,
                                        int nh, int T, int hd, int hp_av, float Qm) {
  const int bd = blockIdx.x, bh = bd / hd, d = bd % hd;   // grid = BH*hd
  const int h = bh % nh, b = bh / nh;
  const int tid = threadIdx.x, nt = blockDim.x;
  float amax = 0.f;
  for (int t = tid; t < T; t += nt) {
    size_t off = ((((size_t)b * T + t) * nh + h) * 3 + 2) * (size_t)hd + d;
    amax = fmaxf(amax, fabsf(__half2float(QKV[off])));
  }
  __shared__ float red[256]; red[tid] = amax; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  if (tid == 0) sv[(size_t)bh * hp_av + d] = fmaxf(red[0], 1e-8f) / Qm;
}
// V quantize + transpose to channel-major from packed qkv (sel=2), int8 (flash PV is int8 for both
// int8 and int4-QK paths). Mirrors aq_vquant_trans_kernel<8>.
__global__ void aq_vquant_trans_packed_kernel(const __half* __restrict__ QKV, const float* __restrict__ sv,
                                              int8_t* __restrict__ vt, int nh, int T, int hd, int hp_av) {
  const int rd = blockIdx.x, bh = rd / hp_av, d = rd % hp_av;   // grid = BH*hp_av
  const int h = bh % nh, b = bh / nh;
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv = (d < hd) ? 1.f / sv[(size_t)bh * hp_av + d] : 0.f;
  int8_t* o = vt + (size_t)rd * T;
  for (int t = tid; t < T; t += nt) {
    float val = 0.f;
    if (d < hd) { size_t off = ((((size_t)b * T + t) * nh + h) * 3 + 2) * (size_t)hd + d; val = __half2float(QKV[off]) * inv; }
    int q = __float2int_rn(val);
    o[t] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
}

// ---- ENTRYPOINT (packed dynamic). QK int8 or int4 (qk_bits), V always int8 (flash PV). Reads the
//      interleaved qkv [b,T,nh,3,hd] directly -> no transpose copy. Serves both int8 & int4 flash. ----
//   Inputs:   qkv fp16 [b,T,nh,3,hd] (contiguous, channel order (nh,3,hd)); nh,T,hd; hp_qk,hp_av;
//             qk_bits (8 or 4)
//   Outputs:  qi,ki int8/packed-int4 [BH,T,hp_qk(/2)] (head-major), vt int8 [BH,hp_av,T]
//             (channel-major), sq,sk f32 [BH,T], sv f32 [BH,hp_av]  (== quantize_attn_qkv layout)
std::vector<torch::Tensor> quantize_attn_qkv_packed(torch::Tensor qkv, int64_t nh, int64_t T, int64_t hd,
                                                    int64_t hp_qk, int64_t hp_av, int64_t qk_bits) {
  TORCH_CHECK(qkv.is_cuda() && qkv.dtype() == torch::kHalf, "qkv fp16 CUDA");
  qkv = qkv.contiguous();
  int b = qkv.numel() / ((int)nh * 3 * (int)hd * (int)T);
  int BH = b * (int)nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(qkv.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(qkv.device());
  int qkw = (qk_bits == 8) ? (int)hp_qk : (int)hp_qk / 2;
  auto qi = torch::empty({BH, (int)T, qkw}, oi), ki = torch::empty({BH, (int)T, qkw}, oi);
  auto vt = torch::empty({BH, (int)hp_av, (int)T}, oi);
  auto sq = torch::empty({BH, (int)T}, of), sk = torch::empty({BH, (int)T}, of);
  auto sv = torch::zeros({BH, (int)hp_av}, of);
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half* P = reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>());
  if (qk_bits == 8) {
    aq_qtok_packed_kernel<8><<<BH * (int)T, 32, 0, s>>>(P, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, 0);
    aq_qtok_packed_kernel<8><<<BH * (int)T, 32, 0, s>>>(P, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, 1);
  } else {
    aq_qtok_packed_kernel<4><<<BH * (int)T, 32, 0, s>>>(P, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, 0);
    aq_qtok_packed_kernel<4><<<BH * (int)T, 32, 0, s>>>(P, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, 1);
  }
  aq_vscale_packed_kernel<<<BH * (int)hd, 256, 0, s>>>(P, sv.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_av, 127.f);
  aq_vquant_trans_packed_kernel<<<BH * (int)hp_av, 256, 0, s>>>(P, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), (int)nh, (int)T, (int)hd, (int)hp_av);
  return {qi, ki, vt, sq, sk, sv};
}

// ---- ENTRYPOINT (packed static). Same as above but calibrated per-tensor sq_c/sk_c + per-channel
//      sv_vec [hp_av], no runtime amax. Serves int8 & int4 flash static paths. ----
std::vector<torch::Tensor> quantize_attn_qkv_packed_static(torch::Tensor qkv, int64_t nh, int64_t T, int64_t hd,
                                                           int64_t hp_qk, int64_t hp_av, int64_t qk_bits,
                                                           double sq_c, double sk_c, torch::Tensor sv_vec) {
  TORCH_CHECK(qkv.is_cuda() && qkv.dtype() == torch::kHalf, "qkv fp16 CUDA");
  TORCH_CHECK(sv_vec.numel() == hp_av, "sv_vec must be [hp_av]");
  qkv = qkv.contiguous();
  int b = qkv.numel() / ((int)nh * 3 * (int)hd * (int)T);
  int BH = b * (int)nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(qkv.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(qkv.device());
  int qkw = (qk_bits == 8) ? (int)hp_qk : (int)hp_qk / 2;
  auto qi = torch::empty({BH, (int)T, qkw}, oi), ki = torch::empty({BH, (int)T, qkw}, oi);
  auto vt = torch::empty({BH, (int)hp_av, (int)T}, oi);
  auto sq = torch::empty({BH, (int)T}, of), sk = torch::empty({BH, (int)T}, of);
  auto sv = sv_vec.to(of).view({1, (int)hp_av}).expand({BH, (int)hp_av}).contiguous();
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half* P = reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>());
  if (qk_bits == 8) {
    aq_qtok_packed_static_kernel<8><<<BH * (int)T, 32, 0, s>>>(P, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, (float)sq_c, 0);
    aq_qtok_packed_static_kernel<8><<<BH * (int)T, 32, 0, s>>>(P, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, (float)sk_c, 1);
  } else {
    aq_qtok_packed_static_kernel<4><<<BH * (int)T, 32, 0, s>>>(P, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, (float)sq_c, 0);
    aq_qtok_packed_static_kernel<4><<<BH * (int)T, 32, 0, s>>>(P, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), (int)nh, (int)T, (int)hd, (int)hp_qk, (float)sk_c, 1);
  }
  aq_vquant_trans_packed_kernel<<<BH * (int)hp_av, 256, 0, s>>>(P, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), (int)nh, (int)T, (int)hd, (int)hp_av);
  return {qi, ki, vt, sq, sk, sv};
}

// ---- int8 reshuffle consumer for the fused int8 GN->qkv (fused_gn_qkv_i8evt). The int8 qkv is
// ALREADY quantized with the flash static scales folded in (channel order (nh,{q,k,v},hd) ==
// [b,T,nh,3,hd]), so this only gathers Q/K (hd->hp pad) and transposes V to channel-major -- NO
// requant (int8->int8 copy). Output layout == quantize_attn_qkv_packed_static's qi/ki/vt; the scales
// sq/sk/sv are the calibrated constants, supplied by the caller. int8 Q/K only. ----
__global__ void from_i8_qtok_kernel(const int8_t* __restrict__ QKV, int8_t* __restrict__ out,
                                    int nh, int T, int hd, int hp, int sel) {
  const int r = blockIdx.x, lane = threadIdx.x;
  const int8_t* xr = QKV + pk_row_off(r, nh, T, hd, sel);
  int8_t* o8 = out + (size_t)r * hp;
  for (int d = lane; d < hp; d += 32) o8[d] = (d < hd) ? xr[d] : (int8_t)0;
}
__global__ void from_i8_vtrans_kernel(const int8_t* __restrict__ QKV, int8_t* __restrict__ vt,
                                      int nh, int T, int hd, int hp_av) {
  const int rd = blockIdx.x, bh = rd / hp_av, d = rd % hp_av;
  const int h = bh % nh, b = bh / nh;
  const int tid = threadIdx.x, nt = blockDim.x;
  int8_t* o = vt + (size_t)rd * T;
  for (int t = tid; t < T; t += nt) {
    int8_t val = 0;
    if (d < hd) { size_t off = ((((size_t)b * T + t) * nh + h) * 3 + 2) * (size_t)hd + d; val = QKV[off]; }
    o[t] = val;
  }
}
std::vector<torch::Tensor> quantize_attn_qkv_from_i8(torch::Tensor qkv_i8, int64_t nh, int64_t T,
                                                     int64_t hd, int64_t hp_qk, int64_t hp_av) {
  TORCH_CHECK(qkv_i8.is_cuda() && qkv_i8.dtype() == torch::kChar, "qkv_i8 int8 CUDA");
  qkv_i8 = qkv_i8.contiguous();
  int b = qkv_i8.numel() / ((int)nh * 3 * (int)hd * (int)T);
  int BH = b * (int)nh;
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(qkv_i8.device());
  auto qi = torch::empty({BH, (int)T, (int)hp_qk}, oi), ki = torch::empty({BH, (int)T, (int)hp_qk}, oi);
  auto vt = torch::empty({BH, (int)hp_av, (int)T}, oi);
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const int8_t* P = qkv_i8.data_ptr<int8_t>();
  from_i8_qtok_kernel<<<BH * (int)T, 32, 0, s>>>(P, qi.data_ptr<int8_t>(), (int)nh, (int)T, (int)hd, (int)hp_qk, 0);
  from_i8_qtok_kernel<<<BH * (int)T, 32, 0, s>>>(P, ki.data_ptr<int8_t>(), (int)nh, (int)T, (int)hd, (int)hp_qk, 1);
  from_i8_vtrans_kernel<<<BH * (int)hp_av, 256, 0, s>>>(P, vt.data_ptr<int8_t>(), (int)nh, (int)T, (int)hd, (int)hp_av);
  return {qi, ki, vt};
}

// ---- ENTRYPOINT (dynamic quantize front-end). Kernels: aq_qtok_kernel<BITS> (Q/K per-token),
//      aq_vscale_kernel + aq_vquant_trans_kernel<BITS> (V per-channel + transpose). ----
//   Op:       Attention W8A8/W4A4 — Q/K/V quantize
//   Inputs:   Q,K,V fp16 [BH,T,hd], hp_qk int64 (padded QK head dim), hp_av int64 (padded AV head
//             dim), bits int64 (8 or 4)
//   Outputs:  qi,ki int8/packed-int4 [BH,T,hp_qk(/2)], vt int8/packed-int4 [BH,hp_av,T(/2)]
//             (channel-major = AV B operand), sq,sk f32 [BH,T] (per-token), sv f32 [BH,hp_av]
//             (per-channel over T)
//   Computes: dynamic absmax quantize — Q/K per-token (per-row absmax/Qm), V per-channel over T
//             then transposed to channel-major; Qm=127 (int8) or 7 (int4); pad lanes zeroed
//   Fuses:    replaces the PyTorch per-token/per-channel quantize (~+5ms elementwise); V quantize +
//             transpose to channel-major done in one kernel
//   Constraints: bits ∈ {8,4}; int4 requires even hp (pairs packed 2/byte)
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

// MIXED int4-Q/K + int8-V quantize for the FUSED int4 FLASH path: Q/K packed int4 (matches
// flash_attn_int4), V int8-transposed [BH,hp_av,T] (flash int4 uses int8 PV). One pass over Q/K/V
// -> replaces "quantize_attn_qkv(...,4) [wastes int4 V] + eager int8 V". Reuses templated
// aq_qtok<4> (int4 Q/K) + aq_vquant_trans<8> (int8 V). Returns {q4,k4,vt(int8),sq,sk,sv}.
std::vector<torch::Tensor> quantize_attn_qkv_i4qk_i8v(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                     int64_t hp_qk, int64_t hp_av) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf, "Q/K/V fp16 CUDA");
  Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
  int BH = Q.size(0), T = Q.size(1), hd = Q.size(2);
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(Q.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto qi = torch::empty({BH, T, (int)hp_qk / 2}, oi), ki = torch::empty({BH, T, (int)hp_qk / 2}, oi);  // int4 packed
  auto vt = torch::empty({BH, (int)hp_av, T}, oi);                                                       // int8 transposed
  auto sq = torch::empty({BH, T}, of), sk = torch::empty({BH, T}, of);
  auto sv = torch::zeros({BH, (int)hp_av}, of);
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half* Qp = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
  const __half* Kp = reinterpret_cast<const __half*>(K.data_ptr<at::Half>());
  const __half* Vp = reinterpret_cast<const __half*>(V.data_ptr<at::Half>());
  aq_qtok_kernel<4><<<BH * T, 32, 0, s>>>(Qp, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), hd, (int)hp_qk);
  aq_qtok_kernel<4><<<BH * T, 32, 0, s>>>(Kp, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), hd, (int)hp_qk);
  aq_vscale_kernel<<<BH * hd, 256, 0, s>>>(Vp, sv.data_ptr<float>(), T, hd, (int)hp_av, 127.f);
  aq_vquant_trans_kernel<8><<<BH * (int)hp_av, 256, 0, s>>>(Vp, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), T, hd, (int)hp_av);
  return {qi, ki, vt, sq, sk, sv};
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

// ---- ENTRYPOINT (static/calibrated quantize front-end, no runtime reduction). Kernels:
//      aq_qtok_static_kernel<BITS> (Q/K), aq_vquant_trans_kernel<BITS> (V transpose). ----
//   Op:       Attention W8A8/W4A4 — Q/K/V quantize (static)
//   Inputs:   Q,K,V fp16 [BH,T,hd], hp_qk int64, hp_av int64, bits int64 (8 or 4), sq_c double,
//             sk_c double (per-tensor calibrated Q/K scales), sv_vec f32 [hp_av] (per-channel V
//             scale, shared across BH)
//   Outputs:  qi,ki int8/packed-int4 [BH,T,hp_qk(/2)], vt int8/packed-int4 [BH,hp_av,T(/2)]
//             (channel-major), sq,sk f32 [BH,T] (broadcast of sq_c/sk_c), sv f32 [BH,hp_av]
//             (broadcast of sv_vec)
//   Computes: fixed-scale quantize — Q/K use calibrated sq_c/sk_c (no absmax); V transposed to
//             channel-major; Qm=127 (int8) or 7 (int4)
//   Fuses:    no runtime reduction (calibrated scales); V quantize + transpose in one kernel
//   Constraints: sv_vec.numel()==hp_av; bits ∈ {8,4}
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

// STATIC mixed int4-Q/K + int8-V quantize (calibrated scales, no runtime amax -> single pass).
// Same outputs as quantize_attn_qkv_i4qk_i8v; sq_c/sk_c per-tensor Q/K int4 scales (=amax/7),
// sv_vec [hp_av] per-channel int8 V scale (=amax/127). Placed after aq_qtok_static_kernel's def.
std::vector<torch::Tensor> quantize_attn_qkv_i4qk_i8v_static(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                            int64_t hp_qk, int64_t hp_av,
                                                            double sq_c, double sk_c, torch::Tensor sv_vec) {
  TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf, "Q/K/V fp16 CUDA");
  Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
  int BH = Q.size(0), T = Q.size(1), hd = Q.size(2);
  TORCH_CHECK(sv_vec.numel() == hp_av, "sv_vec must be [hp_av]");
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(Q.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto qi = torch::empty({BH, T, (int)hp_qk / 2}, oi), ki = torch::empty({BH, T, (int)hp_qk / 2}, oi);
  auto vt = torch::empty({BH, (int)hp_av, T}, oi);
  auto sq = torch::empty({BH, T}, of), sk = torch::empty({BH, T}, of);
  auto sv = sv_vec.to(of).view({1, (int)hp_av}).expand({BH, (int)hp_av}).contiguous();
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half* Qp = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
  const __half* Kp = reinterpret_cast<const __half*>(K.data_ptr<at::Half>());
  const __half* Vp = reinterpret_cast<const __half*>(V.data_ptr<at::Half>());
  aq_qtok_static_kernel<4><<<BH * T, 32, 0, s>>>(Qp, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), hd, (int)hp_qk, (float)sq_c);
  aq_qtok_static_kernel<4><<<BH * T, 32, 0, s>>>(Kp, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), hd, (int)hp_qk, (float)sk_c);
  aq_vquant_trans_kernel<8><<<BH * (int)hp_av, 256, 0, s>>>(Vp, sv.data_ptr<float>(), vt.data_ptr<int8_t>(), T, hd, (int)hp_av);
  return {qi, ki, vt, sq, sk, sv};
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

// ---- ENTRYPOINT (fp16 reference softmax, no quantization). Kernels:
//      attn_softmax_fp16_dynamic_kernel (static_c=false) / attn_softmax_fp16_static_kernel (true). ----
//   Op:       Attention fp16 (reference) — softmax (materialized)
//   Inputs:   S fp16 [BH,T,T] (pre-scaled logits), static_c bool (static vs dynamic), c double
//             (calibrated max, used only when static_c=true)
//   Outputs:  P fp16 [BH,T,T] (UNNORMALIZED exp weights), rowsum f32 [BH,T]
//   Computes: dynamic: P[i,j]=exp(S[i,j]−max_j); static: P[i,j]=min(exp(S[i,j]−c),1) (clamp keeps
//             the fp16 P@V bmm from overflowing when c under-estimates the logits). Caller does
//             O=bmm(P,V) then O/=rowsum.
//   Fuses:    none (no quantize). One shared materialized fp16 path so the static-vs-dynamic softmax
//             delta is measured with nothing else changed.
//   Constraints: T%8==0
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
