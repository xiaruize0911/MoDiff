// =========================================================================
// Fused quantize for the flash-attention score path. Replaces the per-block
// PyTorch quantize (absmax/round/clamp/pad on q,k,v every attention block every
// DDIM step) with three light CUDA kernels:
//   - per-token int8 for Q and K  ([B,H,T,hd] fp16 -> [B,H,T,hd_pad] int8 + [B,H,T] scale)
//   - per-channel int8 for V       (scale over T per head_dim, then quantize+pad)
// Scale convention matches quantized_attention.py: dequant scale = absmax/127.
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

#include "../common.cuh"

__global__ void fa_quant_pertoken_kernel(const __half* __restrict__ x, int8_t* __restrict__ xi,
                                         float* __restrict__ sc, int hd, int hd_pad) {
  int r = blockIdx.x, lane = threadIdx.x;            // one warp per row (b,h,t)
  const __half* xr = x + (size_t)r * hd;
  float amax = 0.f;
  for (int d = lane; d < hd; d += 32) amax = fmaxf(amax, fabsf(__half2float(xr[d])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
  amax = __shfl_sync(0xffffffff, amax, 0);
  float scale = fmaxf(amax, 1e-8f) / 127.f, inv = 1.f / scale;
  int8_t* xir = xi + (size_t)r * hd_pad;
  for (int d = lane; d < hd_pad; d += 32) {
    float val = (d < hd) ? __half2float(xr[d]) * inv : 0.f;
    int q = __float2int_rn(val);
    xir[d] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
  if (lane == 0) sc[r] = scale;
}

__global__ void fa_vscale_kernel(const __half* __restrict__ v, float* __restrict__ sv,
                                 int T, int hd) {
  int bh = blockIdx.x / hd, d = blockIdx.x % hd, tid = threadIdx.x, nt = blockDim.x;
  const __half* base = v + (size_t)bh * T * hd + d;
  float amax = 0.f;
  for (int t = tid; t < T; t += nt) amax = fmaxf(amax, fabsf(__half2float(base[(size_t)t * hd])));
  __shared__ float red[256];
  red[tid] = amax; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  if (tid == 0) sv[bh * hd + d] = fmaxf(red[0], 1e-8f) / 127.f;
}

__global__ void fa_vquant_kernel(const __half* __restrict__ v, const float* __restrict__ sv,
                                 int8_t* __restrict__ vi, int T, int hd, int hd_pad) {
  int r = blockIdx.x, d = threadIdx.x;               // r = (b,h,t) row; block = hd_pad threads
  if (d >= hd_pad) return;
  int bh = r / T;
  int8_t* vir = vi + (size_t)r * hd_pad;
  if (d < hd) {
    float val = __half2float(v[(size_t)r * hd + d]) / sv[bh * hd + d];
    int q = __float2int_rn(val);
    vir[d] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  } else {
    vir[d] = 0;
  }
}

// q,k,v: [B,H,T,hd] fp16 contiguous. Returns {qi,ki,vi [B,H,T,hd_pad] int8,
// sq,sk [B,H,T] f32, sv [B,H,hd] f32}.
std::vector<torch::Tensor> quantize_qkv_int8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                             int64_t hd_pad) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.dtype() == torch::kHalf, "q [B,H,T,hd] fp16");
  q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
  int B = q.size(0), Hh = q.size(1), T = q.size(2), hd = q.size(3);
  int rows = B * Hh * T, BH = B * Hh;
  TORCH_CHECK(hd_pad >= hd && hd_pad % 4 == 0 && hd_pad <= 128, "bad hd_pad");
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(q.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  auto qi = torch::empty({B, Hh, T, (int)hd_pad}, oi);
  auto ki = torch::empty({B, Hh, T, (int)hd_pad}, oi);
  auto vi = torch::empty({B, Hh, T, (int)hd_pad}, oi);
  auto sq = torch::empty({B, Hh, T}, of);
  auto sk = torch::empty({B, Hh, T}, of);
  auto sv = torch::empty({B, Hh, hd}, of);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fa_quant_pertoken_kernel<<<rows, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), qi.data_ptr<int8_t>(),
      sq.data_ptr<float>(), hd, hd_pad);
  fa_quant_pertoken_kernel<<<rows, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(k.data_ptr<at::Half>()), ki.data_ptr<int8_t>(),
      sk.data_ptr<float>(), hd, hd_pad);
  fa_vscale_kernel<<<BH * hd, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(v.data_ptr<at::Half>()), sv.data_ptr<float>(), T, hd);
  fa_vquant_kernel<<<rows, (int)hd_pad, 0, stream>>>(
      reinterpret_cast<const __half*>(v.data_ptr<at::Half>()), sv.data_ptr<float>(),
      vi.data_ptr<int8_t>(), T, hd, hd_pad);
  return {qi, ki, vi, sq, sk, sv};
}
