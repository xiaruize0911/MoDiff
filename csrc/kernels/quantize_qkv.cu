// =========================================================================
// Fused quantize for the flash-attention score path. Reads the packed qkv
// tensor [B,T,nh,3,hd] (fp16, as produced by fused_gn_qkv / the qkv Linear)
// DIRECTLY and emits the flash-layout int8 operands + scales -- fusing the
// channel-major->head-major transpose and avoiding the 3 .contiguous() copies
// the PyTorch path needed. Three light kernels:
//   - per-token int8 for Q and K  -> [B,nh,T,hd_pad] int8 + [B,nh,T] scale
//   - per-channel int8 for V       (scale over T per head_dim) -> + [B,nh,hd] scale
// Scale convention matches quantized_attention.py: dequant scale = absmax/127.
//
// Packed input offset for (b,h,t,d), component c in {0=q,1=k,2=v}:
//   qkv[ (((b*T + t)*nh + h)*3 + c)*hd + d ]   (d contiguous -> coalesced row)
// Output row index (head-major): r = (b*nh + h)*T + t.
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

#include "../common.cuh"

__device__ __forceinline__ size_t qkv_row_base(int r, int nh, int T, int hd, int c) {
  int t = r % T, h = (r / T) % nh, b = r / (T * nh);
  return ((size_t)(((size_t)(b * T + t) * nh + h) * 3 + c)) * hd;
}

__global__ void faq_pertoken_kernel(const __half* __restrict__ qkv, int8_t* __restrict__ xi,
                                    float* __restrict__ sc, int nh, int T, int hd, int hd_pad, int c) {
  int r = blockIdx.x, lane = threadIdx.x;            // one warp per output row (b,h,t)
  const __half* xr = qkv + qkv_row_base(r, nh, T, hd, c);
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

__global__ void faq_vscale_kernel(const __half* __restrict__ qkv, float* __restrict__ sv,
                                  int nh, int T, int hd) {
  int bh = blockIdx.x / hd, d = blockIdx.x % hd, tid = threadIdx.x, nt = blockDim.x;
  int b = bh / nh, h = bh % nh;
  const __half* base = qkv + ((size_t)((size_t)(b * T) * nh + h) * 3 + 2) * hd + d;
  size_t stride = (size_t)nh * 3 * hd;
  float amax = 0.f;
  for (int t = tid; t < T; t += nt) amax = fmaxf(amax, fabsf(__half2float(base[(size_t)t * stride])));
  __shared__ float red[256];
  red[tid] = amax; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
  if (tid == 0) sv[bh * hd + d] = fmaxf(red[0], 1e-8f) / 127.f;
}

__global__ void faq_vquant_kernel(const __half* __restrict__ qkv, const float* __restrict__ sv,
                                  int8_t* __restrict__ vi, int nh, int T, int hd, int hd_pad) {
  int r = blockIdx.x, d = threadIdx.x;               // r = (b,h,t); block = hd_pad threads
  if (d >= hd_pad) return;
  int bh = r / T;
  int8_t* vir = vi + (size_t)r * hd_pad;
  if (d < hd) {
    const __half* xr = qkv + qkv_row_base(r, nh, T, hd, 2);
    float val = __half2float(xr[d]) / sv[bh * hd + d];
    int q = __float2int_rn(val);
    vir[d] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  } else {
    vir[d] = 0;
  }
}

// qkv: [B,T,nh,3,hd] fp16 contiguous. Returns {qi,ki,vi [B,nh,T,hd_pad] int8,
// sq,sk [B,nh,T] f32, sv [B,nh,hd] f32}.
std::vector<torch::Tensor> quantize_qkv_int8(torch::Tensor qkv, int64_t nh, int64_t hd_pad) {
  TORCH_CHECK(qkv.is_cuda() && qkv.dtype() == torch::kHalf, "qkv fp16 CUDA");
  qkv = qkv.contiguous();
  int B = qkv.size(0), T = qkv.size(1), hd = qkv.size(-1);
  int rows = B * (int)nh * T, BH = B * (int)nh;
  TORCH_CHECK(hd_pad >= hd && hd_pad % 4 == 0 && hd_pad <= 128, "bad hd_pad");
  auto oi = torch::TensorOptions().dtype(torch::kChar).device(qkv.device());
  auto of = torch::TensorOptions().dtype(torch::kFloat32).device(qkv.device());
  auto qi = torch::empty({B, (int)nh, T, (int)hd_pad}, oi);
  auto ki = torch::empty({B, (int)nh, T, (int)hd_pad}, oi);
  auto vi = torch::empty({B, (int)nh, T, (int)hd_pad}, oi);
  auto sq = torch::empty({B, (int)nh, T}, of);
  auto sk = torch::empty({B, (int)nh, T}, of);
  auto sv = torch::empty({B, (int)nh, hd}, of);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const __half* p = reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>());

  faq_pertoken_kernel<<<rows, 32, 0, stream>>>(p, qi.data_ptr<int8_t>(), sq.data_ptr<float>(), nh, T, hd, hd_pad, 0);
  faq_pertoken_kernel<<<rows, 32, 0, stream>>>(p, ki.data_ptr<int8_t>(), sk.data_ptr<float>(), nh, T, hd, hd_pad, 1);
  faq_vscale_kernel<<<BH * hd, 256, 0, stream>>>(p, sv.data_ptr<float>(), nh, T, hd);
  faq_vquant_kernel<<<rows, (int)hd_pad, 0, stream>>>(p, sv.data_ptr<float>(), vi.data_ptr<int8_t>(), nh, T, hd, hd_pad);
  return {qi, ki, vi, sq, sk, sv};
}
