// =========================================================================
// Fused GroupNorm -> qkv Linear for the self-attention blocks.
//
// The pre-attention GroupNorm is memory-bound and, together with the qkv GEMM,
// is a large slice of the diffusion-UNet attention cost. This fuses the two:
//   1. gn_stats_kernel  -- per-(sample,group) mean/rstd from the NHWC activation,
//      emitted as per-(sample,channel) scale[N,C]=rstd and bias[N,C]=-mean*rstd+SHIFT.
//   2. a CUTLASS 1x1-conv fprop with the mainloop scale+bias fusion (example 25)
//      that applies (scale*x+bias) to the activations *inside* the mainloop, so the
//      normalized tensor is never written. The qkv weight (with GroupNorm's affine
//      gamma folded in) is the conv filter; GroupNorm's beta + the qkv bias are the
//      per-output-channel epilogue bias.
//
// Two wrinkles vs the stock CUTLASS fusion:
//  * GroupNorm's normalize is PER-SAMPLE (scale/bias depend on the activation's
//    sample n). The stock fusion shares one [1,C] scale vector across the batch;
//    ImplicitGemmConvolutionFusionPerSample offsets the scale/bias pointer by
//    sample*C per threadblock. Valid only when the M-tile stays within one sample,
//    i.e. tokens T = H*W is a multiple of the tile's kM (the Python caller gates
//    on this and falls back to cuBLAS+GroupNorm otherwise).
//  * The stock fusion applies scale+bias+RELU. GroupNorm->qkv must NOT relu the
//    (sign-bearing) normalized activations. We absorb the relu with a constant
//    SHIFT: bias carries +SHIFT so the pre-relu value (x-mean)*rstd+SHIFT is always
//    >=0 (normalized activations are ~unit variance; SHIFT is far beyond any real
//    value -> relu is identity). The induced per-output-channel constant
//    SHIFT * sum_c Wf[c,j] is subtracted back in the epilogue bias (done in Python,
//    static, no per-sample dependence).
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_fusion.h"
#include "cutlass/conv/device/implicit_gemm_convolution_fusion.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "common.cuh"
#include "implicit_gemm_fusion_persample.h"
#include "implicit_gemm_fusion_persample_evt.h"

using Arch = cutlass::arch::Sm80;

// -------- per-(sample,group) stats -> per-(sample,channel) scale/bias --------
// Two passes for coalesced reads AND high occupancy:
//  1. gn_accum: grid (N, token_tiles), blockDim C. Thread c sums its channel over a
//     token tile (consecutive threads -> consecutive channels -> COALESCED loads),
//     then atomicAdds the tile's per-channel partials to global sumC/sumsqC [N,C].
//     Splitting tokens into tiles gives many blocks (good SM occupancy on big T).
//  2. gn_finalize: grid (N,), blockDim C. Combine per-channel sums within each group,
//     emit scale=rstd and bias=-mean*rstd+SHIFT.
__global__ void gn_accum_kernel(
    const __half* __restrict__ x, float* __restrict__ sumC, float* __restrict__ sumsqC,
    int N, int C, int T, int TILE) {
  int n = blockIdx.x;
  int t0 = blockIdx.y * TILE;
  int t1 = min(t0 + TILE, T);
  int c = threadIdx.x;
  float s = 0.f, s2 = 0.f;
  const __half* base = x + (long)n * T * C + c;
  for (int t = t0; t < t1; t++) {
    float v = __half2float(base[(long)t * C]);
    s += v; s2 += v * v;
  }
  atomicAdd(&sumC[(long)n * C + c], s);
  atomicAdd(&sumsqC[(long)n * C + c], s2);
}

__global__ void gn_finalize_kernel(
    const float* __restrict__ sumC, const float* __restrict__ sumsqC,
    __half* __restrict__ scale, __half* __restrict__ bias,
    int N, int C, int T, int Cg, float eps, float shift) {
  extern __shared__ float smem[];   // [C] sums, [C] sumsq
  float* ss = smem; float* ss2 = smem + C;
  int n = blockIdx.x, c = threadIdx.x;
  ss[c] = sumC[(long)n * C + c]; ss2[c] = sumsqC[(long)n * C + c];
  __syncthreads();
  int g = c / Cg;
  float gs = 0.f, gs2 = 0.f;
  for (int i = 0; i < Cg; i++) { gs += ss[g * Cg + i]; gs2 += ss2[g * Cg + i]; }
  float cnt = (float)T * Cg;
  float mean = gs / cnt;
  float var = gs2 / cnt - mean * mean;
  float rstd = rsqrtf(var + eps);
  scale[(long)n * C + c] = __float2half(rstd);
  bias[(long)n * C + c]  = __float2half(-mean * rstd + shift);
}

// ---- CUTLASS per-sample fused conv (fp16, 1x1, NHWC), tile 128x256x32 ----
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>;

using DefaultFusionKernel = typename cutlass::conv::kernel::DefaultConv2dFpropFusion<
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

using PerSampleKernel = cutlass::conv::kernel::ImplicitGemmConvolutionFusionPerSample<
    typename DefaultFusionKernel::Mma, typename DefaultFusionKernel::Epilogue,
    typename DefaultFusionKernel::ThreadblockSwizzle,
    cutlass::conv::Operator::kFprop, cutlass::conv::Conv2dProblemSize>;

using FusedConvOp = cutlass::conv::device::ImplicitGemmConvolutionFusion<PerSampleKernel>;

// ---- int8-OUTPUT variant: same fp16 GN+qkv mainloop, but the epilogue clamps to int8.
// The per-output-column requant oscale[n]=127/absmax is folded into the conv weight and epilogue
// bias offline (both scale by oscale[n]), so the GEMM emits (real qkv * oscale) and the epilogue
// just rounds/clamps to int8 -- no per-column epilogue needed. Consumed by quantize_attn_qkv_from_i8. ----
using EpilogueOpI8 = cutlass::epilogue::thread::LinearCombinationClamp<
    int8_t, 128 / cutlass::sizeof_bits<int8_t>::value, float, float>;
using DefaultFusionKernelI8 = typename cutlass::conv::kernel::DefaultConv2dFpropFusion<
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOpI8,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
using PerSampleKernelI8 = cutlass::conv::kernel::ImplicitGemmConvolutionFusionPerSample<
    typename DefaultFusionKernelI8::Mma, typename DefaultFusionKernelI8::Epilogue,
    typename DefaultFusionKernelI8::ThreadblockSwizzle,
    cutlass::conv::Operator::kFprop, cutlass::conv::Conv2dProblemSize>;
using FusedConvOpI8 = cutlass::conv::device::ImplicitGemmConvolutionFusion<PerSampleKernelI8>;

// -------- host entry --------
//   Op:       Fused GroupNorm -> qkv projection (fp16, 1x1 conv, NHWC)
//   Inputs:   x        fp16 [N,C,H,W] channels_last
//             weight   fp16 [K=3C,1,1,C] conv filter (qkv.weight * gn.weight/gamma),
//                      K/R/S/C order
//             epi_bias fp16 [K] per-output-channel epilogue bias
//                      (qkv.bias + qkv.w@gn.beta - SHIFT*colsum(Wf))
//             groups int; eps double; shift double (the SHIFT relu-avoidance constant)
//   Outputs:  fp16 [N,K,H,W] channels_last (token-major [N,T,K] via a free view)
//   Computes: per-(sample,group) GroupNorm stats -> scale[N,C]=rstd,
//             bias[N,C]=-mean*rstd+SHIFT (gn_accum + gn_finalize kernels), then a CUTLASS
//             1x1-conv fprop that applies (scale*x+bias) inside the mainloop and does the
//             qkv GEMM; result = qkv(GN(x)) = qkv projection of the normalized activation
//   Fuses:    GroupNorm folded into the qkv conv -- the normalized tensor is never
//             materialized (gamma folded into weight, beta+qkv.bias into epi_bias,
//             per-sample scale/bias applied in the CUTLASS mainloop); relu absorbed via SHIFT
//   Constraints: fp16; x channels_last; C%groups==0; Sm80. Per-sample fusion is valid only
//             when tokens T=H*W is a multiple of the tile's kM (Python caller gates on this
//             and falls back to cuBLAS+GroupNorm otherwise)
//   vs fp16:  fuses GroupNorm + the qkv projection into one CUTLASS kernel, removing the
//             fp16 qkv materialization + reshape copy.
torch::Tensor fused_gn_qkv(
    torch::Tensor x, torch::Tensor weight, torch::Tensor epi_bias,
    int groups, double eps, double shift) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  int K = weight.size(0);
  int T = H * W, Cg = C / groups;

  auto opt_h = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
  auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  auto scale = torch::empty({N, C}, opt_h);
  auto bias  = torch::empty({N, C}, opt_h);
  auto sumC   = torch::zeros({N, C}, opt_f);
  auto sumsqC = torch::zeros({N, C}, opt_f);

  const int TILE = 128;
  int ttiles = (T + TILE - 1) / TILE;
  dim3 accum_grid(N, ttiles);
  gn_accum_kernel<<<accum_grid, C, 0, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      sumC.data_ptr<float>(), sumsqC.data_ptr<float>(), N, C, T, TILE);
  gn_finalize_kernel<<<N, C, 2 * C * (int)sizeof(float), stream>>>(
      sumC.data_ptr<float>(), sumsqC.data_ptr<float>(),
      reinterpret_cast<__half*>(scale.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(bias.data_ptr<at::Half>()),
      N, C, T, Cg, (float)eps, (float)shift);

  auto out_opt = torch::TensorOptions().dtype(torch::kFloat16).device(x.device())
                     .memory_format(torch::MemoryFormat::ChannelsLast);
  auto output = torch::empty({N, K, H, W}, out_opt);

  cutlass::conv::Conv2dProblemSize problem(
      {N, H, W, C}, {K, 1, 1, C}, {0, 0, 0, 0}, {1, 1}, {1, 1},
      {N, H, W, K}, cutlass::conv::Mode::kCrossCorrelation, 1);

  auto* xp  = reinterpret_cast<cutlass::half_t*>(x.data_ptr<at::Half>());
  auto* wp  = reinterpret_cast<cutlass::half_t*>(weight.data_ptr<at::Half>());
  auto* scp = reinterpret_cast<cutlass::half_t*>(scale.data_ptr<at::Half>());
  auto* bip = reinterpret_cast<cutlass::half_t*>(bias.data_ptr<at::Half>());
  auto* ebp = reinterpret_cast<cutlass::half_t*>(epi_bias.data_ptr<at::Half>());
  auto* op_ = reinterpret_cast<cutlass::half_t*>(output.data_ptr<at::Half>());

  using LSB = cutlass::layout::RowMajor;
  using LN  = cutlass::layout::TensorNHWC;
  typename FusedConvOp::Arguments args{
      problem,
      {xp, LN({C, W * C, H * W * C})},
      {wp, LN({C, 1 * C, 1 * 1 * C})},
      {scp, LSB(C)}, {bip, LSB(C)},
      {ebp, LN::Stride(0)},                     // per-channel bias broadcast (beta=1)
      {op_, LN({K, W * K, H * W * K})},
      {1.0f, 1.0f}};

  FusedConvOp op;
  cutlass::Status st = op.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "fused_gn_qkv: can_implement failed (N=", N, " C=", C, " H=", H, " W=", W,
              " K=", K, " T=", T, "): ", cutlass::cutlassGetStatusString(st));
  size_t ws = op.get_workspace_size(args);
  auto workspace = torch::empty({(long)ws},
      torch::TensorOptions().dtype(torch::kByte).device(x.device()));
  st = op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv: initialize failed: ",
              cutlass::cutlassGetStatusString(st));
  st = op(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv: run failed: ",
              cutlass::cutlassGetStatusString(st));
  return output;
}

// -------- host entry --------
//   Op:       Fused GroupNorm -> qkv projection, INT8 output (fp16 mainloop, 1x1 conv, NHWC)
//   Inputs:   x        fp16 [N,C,H,W] channels_last
//             weight   fp16 [K=3C,1,1,C] conv filter (qkv.weight * gn.weight * oscale[n])
//             epi_bias int8 [K] per-output-channel bias source (oscale-folded, rounded)
//             groups int; eps double; shift double (SHIFT relu-avoidance constant)
//   Outputs:  int8 [N,K,H,W] channels_last (token-major [N,T,K] via a free view)
//   Computes: identical fp16 GN + qkv mainloop as fused_gn_qkv, but the epilogue clamps to
//             int8. The per-output-column requant oscale[n]=127/absmax_col is folded into
//             weight and epi_bias offline (both scaled by oscale), so the GEMM emits
//             (real qkv * oscale) and the epilogue just rounds/clamps -> int8
//   Fuses:    GroupNorm + the qkv projection + the int8 output requant into one CUTLASS
//             kernel, removing the fp16 qkv materialization + reshape copy. Consumed by
//             quantize_attn_qkv_from_i8
//   Constraints: fp16 input; x channels_last; C%groups==0; Sm80; weight/epi_bias must already
//             be oscale-folded. Same per-sample T=H*W multiple-of-kM gate as fused_gn_qkv
//   vs fp16:  fuses GroupNorm + the qkv projection (and the int8 output requant) into one
//             CUTLASS kernel, removing the fp16 qkv materialization + reshape copy.
//
// int8-output variant. weight/epi_bias must already have oscale[n] folded in (both scaled by
// oscale[n]=127/absmax_col). Returns int8 [N, K, H, W] channels_last (token-major [N,T,K] view).
torch::Tensor fused_gn_qkv_int8(
    torch::Tensor x, torch::Tensor weight, torch::Tensor epi_bias,
    int groups, double eps, double shift) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  int K = weight.size(0);
  int T = H * W, Cg = C / groups;

  auto opt_h = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
  auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  auto scale = torch::empty({N, C}, opt_h);
  auto bias  = torch::empty({N, C}, opt_h);
  auto sumC   = torch::zeros({N, C}, opt_f);
  auto sumsqC = torch::zeros({N, C}, opt_f);

  const int TILE = 128;
  int ttiles = (T + TILE - 1) / TILE;
  dim3 accum_grid(N, ttiles);
  gn_accum_kernel<<<accum_grid, C, 0, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      sumC.data_ptr<float>(), sumsqC.data_ptr<float>(), N, C, T, TILE);
  gn_finalize_kernel<<<N, C, 2 * C * (int)sizeof(float), stream>>>(
      sumC.data_ptr<float>(), sumsqC.data_ptr<float>(),
      reinterpret_cast<__half*>(scale.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(bias.data_ptr<at::Half>()),
      N, C, T, Cg, (float)eps, (float)shift);

  auto out_opt = torch::TensorOptions().dtype(torch::kChar).device(x.device())
                     .memory_format(torch::MemoryFormat::ChannelsLast);
  auto output = torch::empty({N, K, H, W}, out_opt);

  cutlass::conv::Conv2dProblemSize problem(
      {N, H, W, C}, {K, 1, 1, C}, {0, 0, 0, 0}, {1, 1}, {1, 1},
      {N, H, W, K}, cutlass::conv::Mode::kCrossCorrelation, 1);

  auto* xp  = reinterpret_cast<cutlass::half_t*>(x.data_ptr<at::Half>());
  auto* wp  = reinterpret_cast<cutlass::half_t*>(weight.data_ptr<at::Half>());
  auto* scp = reinterpret_cast<cutlass::half_t*>(scale.data_ptr<at::Half>());
  auto* bip = reinterpret_cast<cutlass::half_t*>(bias.data_ptr<at::Half>());
  auto* ebp = reinterpret_cast<int8_t*>(epi_bias.data_ptr<int8_t>());   // int8 bias source (oscale-folded, rounded)
  auto* op_ = reinterpret_cast<int8_t*>(output.data_ptr<int8_t>());

  using LSB = cutlass::layout::RowMajor;
  using LN  = cutlass::layout::TensorNHWC;
  typename FusedConvOpI8::Arguments args{
      problem,
      {xp, LN({C, W * C, H * W * C})},
      {wp, LN({C, 1 * C, 1 * 1 * C})},
      {scp, LSB(C)}, {bip, LSB(C)},
      {ebp, LN::Stride(0)},
      {op_, LN({K, W * K, H * W * K})},
      {1.0f, 1.0f}};

  FusedConvOpI8 op;
  cutlass::Status st = op.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "fused_gn_qkv_int8: can_implement failed: ", cutlass::cutlassGetStatusString(st));
  size_t ws = op.get_workspace_size(args);
  auto workspace = torch::empty({(long)ws},
      torch::TensorOptions().dtype(torch::kByte).device(x.device()));
  st = op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv_int8: initialize failed: ",
              cutlass::cutlassGetStatusString(st));
  st = op(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv_int8: run failed: ",
              cutlass::cutlassGetStatusString(st));
  return output;
}

// ==== int8-output variant via a custom EVT epilogue (fixes fused_gn_qkv_int8's signed-qkv bug) ====
// fused_gn_qkv_int8's epilogue bias is int8 and cannot hold the -oscale*SHIFT*colsum correction
// (|.|~1600 >> 127). Here the epilogue is an Sm80 EVT tree: acc + RowBroadcast(FP32 bias) ->
// AuxStore int8 -- the fp32 bias node holds the correction, only the small result is clamped to int8.
// Driven by the merged per-sample-fusion + EVT kernel (implicit_gemm_fusion_persample_evt.h).
namespace ct = cutlass::epilogue::threadblock;
using EvtSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EvtTB = cutlass::gemm::GemmShape<128, 256, 32>;
using EvtW  = cutlass::gemm::GemmShape<64, 64, 32>;
using EvtI  = cutlass::gemm::GemmShape<16, 8, 16>;
static int const kEvtAlignI8 = 128 / cutlass::sizeof_bits<int8_t>::value;   // 16
using EvtEpiOpI8 = cutlass::epilogue::thread::LinearCombinationClamp<int8_t, kEvtAlignI8, float, float>;
using EvtDefaultFusion = typename cutlass::conv::kernel::DefaultConv2dFpropFusion<
    cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::TensorNHWC, float,
    cutlass::arch::OpClassTensorOp, Arch, EvtTB, EvtW, EvtI,
    EvtEpiOpI8, EvtSwizzle, 3, cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
using EvtMma = typename EvtDefaultFusion::Mma;
using EvtDefaultEpi = typename EvtDefaultFusion::Epilogue;
using EvtTileMap = ct::OutputTileThreadLayout<EvtTB, EvtW, int8_t, kEvtAlignI8, 1>;
using EvtAccum  = ct::VisitorAccFetch;
using EvtBiasRow = ct::VisitorRowBroadcast<EvtTileMap, float, cute::Stride<cute::_0, cute::_1, int32_t>>;
using EvtAdd = ct::VisitorCompute<cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using EvtAuxSt = ct::VisitorAuxStore<EvtTileMap, int8_t, cutlass::FloatRoundStyle::round_to_nearest,
                                     cute::Stride<int64_t, cute::_1, int64_t>>;
using EvtAddBias = ct::Sm80EVT<EvtAdd, EvtAccum, EvtBiasRow>;
using EvtTree = ct::Sm80EVT<EvtAuxSt, EvtAddBias>;
using EvtEpi = ct::EpilogueWithVisitorCallbacks<EvtDefaultEpi, EvtTree, 1>;
using EvtConvKernel = modiff::ImplicitGemmConvolutionFusionPerSampleEVT<
    EvtMma, EvtEpi, EvtSwizzle, cutlass::conv::Operator::kFprop>;
using EvtConvOp = cutlass::conv::device::ImplicitGemmConvolution<EvtConvKernel>;

// -------- host entry: int8 fused GN->qkv with fp32 EVT bias (signed-qkv-correct) --------
//   Inputs:  x fp16 [N,C,H,W] channels_last; weight fp16 [3C,1,1,C] (oscale*gamma folded);
//            bias_f32 fp32 [3C] EVT bias = oscale*(qkv.bias + qkv.w@gn.beta - SHIFT*colsum(Wf));
//            groups int; eps double; shift double (SHIFT relu-avoidance constant).
//   Output:  int8 [N,3C,H,W] channels_last (token-major [N,T,3C], channel order (nh,{q,k,v},hd)).
//   The GN stats (per-sample scale=rstd, bias=-mean*rstd+SHIFT) are computed here (gn_accum/finalize);
//   the mainloop applies them per-sample; the EVT epilogue adds bias_f32 and clamps to int8.
torch::Tensor fused_gn_qkv_i8evt(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias_f32,
    int groups, double eps, double shift) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  int K = weight.size(0);
  int T = H * W, Cg = C / groups;
  int64_t M = (int64_t)N * T, ldK = K;

  auto opt_h = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
  auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  auto scale = torch::empty({N, C}, opt_h);
  auto bias  = torch::empty({N, C}, opt_h);
  auto sumC   = torch::zeros({N, C}, opt_f);
  auto sumsqC = torch::zeros({N, C}, opt_f);

  const int TILE = 128;
  int ttiles = (T + TILE - 1) / TILE;
  dim3 accum_grid(N, ttiles);
  gn_accum_kernel<<<accum_grid, C, 0, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      sumC.data_ptr<float>(), sumsqC.data_ptr<float>(), N, C, T, TILE);
  gn_finalize_kernel<<<N, C, 2 * C * (int)sizeof(float), stream>>>(
      sumC.data_ptr<float>(), sumsqC.data_ptr<float>(),
      reinterpret_cast<__half*>(scale.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(bias.data_ptr<at::Half>()),
      N, C, T, Cg, (float)eps, (float)shift);

  auto out = torch::empty({N, K, H, W}, torch::TensorOptions().dtype(torch::kChar).device(x.device())
                          .memory_format(torch::MemoryFormat::ChannelsLast));
  cutlass::conv::Conv2dProblemSize problem(
      {N, H, W, C}, {K, 1, 1, C}, {0, 0, 0, 0}, {1, 1}, {1, 1},
      {N, H, W, K}, cutlass::conv::Mode::kCrossCorrelation, 1);

  auto* xp = reinterpret_cast<cutlass::half_t*>(x.data_ptr<at::Half>());
  auto* wp = reinterpret_cast<cutlass::half_t*>(weight.data_ptr<at::Half>());
  auto* scp = reinterpret_cast<cutlass::half_t*>(scale.data_ptr<at::Half>());
  auto* bip = reinterpret_cast<cutlass::half_t*>(bias.data_ptr<at::Half>());
  auto* bf  = bias_f32.data_ptr<float>();
  auto* op_ = reinterpret_cast<int8_t*>(out.data_ptr<int8_t>());

  using LN = cutlass::layout::TensorNHWC; using LSB = cutlass::layout::RowMajor;
  typename EvtConvKernel::TensorRefA refA{xp, LN({C, W * C, H * W * C})};
  typename EvtConvKernel::TensorRefB refB{wp, LN({C, 1 * C, 1 * 1 * C})};
  typename EvtConvKernel::TensorRefScaleBias refScale{scp, LSB(C)};
  typename EvtConvKernel::TensorRefScaleBias refBias{bip, LSB(C)};
  typename EvtTree::Arguments ep{
    { {},
      {bf, 0.f, {cute::_0{}, cute::_1{}, (int32_t)K}},
      {} },
    {op_, {ldK, cute::_1{}, M * K}} };
  typename EvtConvOp::Arguments args{problem, refA, refB, refScale, refBias, ep};
  EvtConvOp op;
  cutlass::Status st = op.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv_i8evt: can_implement: ",
              cutlass::cutlassGetStatusString(st));
  auto ws = torch::empty({(long)op.get_workspace_size(args)},
                         torch::TensorOptions().dtype(torch::kByte).device(x.device()));
  st = op.initialize(args, ws.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv_i8evt: initialize: ",
              cutlass::cutlassGetStatusString(st));
  st = op(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused_gn_qkv_i8evt: run: ",
              cutlass::cutlassGetStatusString(st));
  return out;
}
