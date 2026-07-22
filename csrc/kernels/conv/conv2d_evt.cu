// EVT-fused conv2d fprop epilogues (SM80), driven by the custom EVT-capable conv kernel driver
// (implicit_gemm_conv_evt.h). CUTLASS 4.6.1 has no EVT-on-conv path, so these hand-assemble an
// Sm80 Epilogue Visitor Tree onto the int8/int4 conv Mma and drive it with modiff::
// ImplicitGemmConvolutionEVT. Two fusions, each avoiding the post-conv scratch tensor entirely:
//   D1 (baseline):  out = acc*alpha*weight_scale[k] + bias[k] + residual[elem]  -> fp16
//                   (replaces conv2d_intX_fprop_deepfuse_bias_residual_fp16's fp16 scratch + store)
//   D2 (MoDiff):    o_hat[elem] += acc*alpha*weight_scale[k]  (in place, fp16);
//                   out = o_hat_new + residual[elem]  -> fp16   (dual store)
//                   (replaces conv2d_intX_fprop_o_hat_residual's fp32 conv_out round-trip)
// alpha is read on-device from a 1-elem fp32 tensor (no host sync). weight_scale/bias are FP32 [K]
// (matching the fp32 weight_scale + fp32 accumulate of scale_accumulate_residual_half_cache_kernel /
// the fp32 conv accumulate of conv2d_intX_fprop, so D2's o_hat stays BIT-EXACT).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/gemm/gemm.h"

#include "implicit_gemm_conv_evt.h"
#include "common.cuh"

namespace {
using cutlass::layout::TensorNHWC;
using Arch = cutlass::arch::Sm80;
using EC = cutlass::half_t;          // output / aux (o_hat, residual) element
using ES = float;                    // per-channel weight_scale/bias element (fp32 -> bit-exact vs fp32 refs)
using ECompute = float;
namespace ct = cutlass::epilogue::threadblock;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
static int const kAlignC = 128 / cutlass::sizeof_bits<EC>::value;   // 8

// EVT node set + the two trees (D1, D2), parameterized by the conv's DefaultEpilogue + Mma + tile.
template <class DefaultConvKernel, class TBShape, class WShape>
struct Evt {
  using Mma = typename DefaultConvKernel::Mma;
  using DefaultEpi = typename DefaultConvKernel::Epilogue;
  using TileMap = ct::OutputTileThreadLayout<TBShape, WShape, EC, kAlignC, 1>;

  using Accum  = ct::VisitorAccFetch;
  using Alpha  = ct::VisitorScalarBroadcast<ECompute, cute::Stride<cute::_0, cute::_0, cute::_0>>;
  using RowVec = ct::VisitorRowBroadcast<TileMap, ES, cute::Stride<cute::_0, cute::_1, int32_t>>;  // per-channel
  using AuxLd  = ct::VisitorAuxLoad<TileMap, EC, cute::Stride<int64_t, cute::_1, int64_t>>;        // per-element
  using AuxSt  = ct::VisitorAuxStore<TileMap, EC, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
  using Mul = ct::VisitorCompute<cutlass::multiplies, ECompute, ECompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using Add = ct::VisitorCompute<cutlass::plus,       ECompute, ECompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using E_MulA  = ct::Sm80EVT<Mul, Accum, Alpha>;      // acc*alpha
  using E_MulWS = ct::Sm80EVT<Mul, E_MulA, RowVec>;    // *weight_scale
  // D1: +bias (+residual) -> store. (AuxLoad has no null guard, so residual needs its own tree.)
  using E_AddB  = ct::Sm80EVT<Add, E_MulWS, RowVec>;
  using E_AddR  = ct::Sm80EVT<Add, E_AddB, AuxLd>;
  using EVTD1   = ct::Sm80EVT<AuxSt, E_AddR>;      // with residual
  using EVTD1nr = ct::Sm80EVT<AuxSt, E_AddB>;      // no residual
  // D2: o_hat RMW (store) then +residual (store)
  using E_OhatNew = ct::Sm80EVT<Add, AuxLd, E_MulWS>;
  using E_StOhat  = ct::Sm80EVT<AuxSt, E_OhatNew>;
  using E_AddRes  = ct::Sm80EVT<Add, E_StOhat, AuxLd>;
  using EVTD2   = ct::Sm80EVT<AuxSt, E_AddRes>;

  using EpiD1   = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD1, 1>;
  using EpiD1nr = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD1nr, 1>;
  using EpiD2 = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD2, 1>;
  using KernelD1   = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD1,   Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD1nr = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD1nr, Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD2   = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD2,   Swizzle, cutlass::conv::Operator::kFprop>;
  using OpD1   = cutlass::conv::device::ImplicitGemmConvolution<KernelD1>;
  using OpD1nr = cutlass::conv::device::ImplicitGemmConvolution<KernelD1nr>;
  using OpD2   = cutlass::conv::device::ImplicitGemmConvolution<KernelD2>;
};

// int8 conv config (matches Conv2dInt8DequantFp16Kernel in conv2d_int8.cu)
using I8Def = typename cutlass::conv::kernel::DefaultConv2dFprop<
    int8_t, TensorNHWC, int8_t, TensorNHWC, EC, TensorNHWC, int32_t,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128,128,128>, cutlass::gemm::GemmShape<64,64,64>, cutlass::gemm::GemmShape<16,8,32>,
    cutlass::epilogue::thread::LinearCombination<EC, kAlignC, int32_t, float>,
    Swizzle, 3, cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized, cutlass::conv::StrideSupport::kStrided>::Kernel;
using I8 = Evt<I8Def, cutlass::gemm::GemmShape<128,128,128>, cutlass::gemm::GemmShape<64,64,64>>;

// int4 conv config (matches Conv2dInt4Kernel in conv2d_int4.cu)
using I4Def = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::int4b_t, TensorNHWC, cutlass::int4b_t, TensorNHWC, EC, TensorNHWC, int32_t,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128,128,128>, cutlass::gemm::GemmShape<64,64,128>, cutlass::gemm::GemmShape<16,8,64>,
    cutlass::epilogue::thread::LinearCombination<EC, kAlignC, int32_t, float>,
    Swizzle, 3, cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized, cutlass::conv::StrideSupport::kStrided>::Kernel;
using I4 = Evt<I4Def, cutlass::gemm::GemmShape<128,128,128>, cutlass::gemm::GemmShape<64,64,128>>;

template <class EvtT, class ElemAB>
void run_d1(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp, ES* biasp, EC* resp, EC* outp,
            cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
            int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); EC ze(0);
  // bias broadcast is null-guarded (RowBroadcast fills null_default=0 when ptr==null); residual
  // AuxLoad is NOT null-guarded, so a residual-free conv uses the EVTD1nr tree (no AuxLoad node).
  using LN = TensorNHWC;
  typename EvtT::KernelD1::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD1::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  if (resp != nullptr) {
    typename EvtT::EVTD1::Arguments ep{
      { { { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },
            {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
          {biasp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },          // +bias (null->0)
        {resp, ze, {ldK, cute::_1{}, M*K}}, {} },                          // +residual
      {outp, {ldK, cute::_1{}, M*K}} };
    typename EvtT::OpD1::Arguments args{problem, refA, refB, ep};
    typename EvtT::OpD1 op;
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d1 can_implement");
    auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d1 init");
    TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d1 run");
  } else {
    typename EvtT::EVTD1nr::Arguments ep{
      { { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },
          {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
        {biasp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },            // +bias (null->0)
      {outp, {ldK, cute::_1{}, M*K}} };
    typename EvtT::OpD1nr::Arguments args{problem, refA, refB, ep};
    typename EvtT::OpD1nr op;
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d1nr can_implement");
    auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d1nr init");
    TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d1nr run");
  }
}

template <class EvtT, class ElemAB>
void run_d2(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp, EC* ohatp, EC* resp, EC* outp,
            cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
            int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); EC ze(0); int64_t MK = M * K;
  typename EvtT::EVTD2::Arguments ep{
    {                                                            // E_AddRes
      {                                                          //   E_StOhat
        {                                                        //     E_OhatNew
          {ohatp, ze, {ldK, cute::_1{}, MK}},                    //       AuxLd o_hat_old
          { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },        //       E_MulA
            {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },//       *weight_scale
          {} },                                                 //     Add
        {ohatp, {ldK, cute::_1{}, MK}}                           //   AuxSt -> o_hat (in place)
      },
      {resp, ze, {ldK, cute::_1{}, MK}}, {}                      //   +residual
    },
    {outp, {ldK, cute::_1{}, MK}}                                // AuxSt -> out
  };
  using LN = TensorNHWC;
  typename EvtT::KernelD2::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD2::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  typename EvtT::OpD2::Arguments args{problem, refA, refB, ep};
  typename EvtT::OpD2 op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d2 can_implement");
  auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d2 init");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d2 run");
}

cutlass::conv::Conv2dProblemSize make_problem(int N, int H, int W, int C, int K, int R, int S,
                                              int sh, int sw, int ph, int pw, int dh, int dw,
                                              int& P, int& Q) {
  P = (H + 2*ph - dh*(R-1) - 1) / sh + 1;
  Q = (W + 2*pw - dw*(S-1) - 1) / sw + 1;
  return cutlass::conv::Conv2dProblemSize({N,H,W,C},{K,R,S,C},{ph,pw,ph,pw},{sh,sw},{dh,dw},
                                          {N,P,Q,K}, cutlass::conv::Mode::kCrossCorrelation, 1);
}
} // namespace

// ---- host entry points ----
torch::Tensor conv2d_int8_evt_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(output); CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output fp16");
  // int8 input is logical [N,C,H,W] channels_last (matches conv2d_int8_fprop's dim extraction).
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d1<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      bias.numel()? bias.data_ptr<float>() : nullptr,
      residual.numel()? reinterpret_cast<EC*>(residual.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor conv2d_int4_evt_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(output); CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output fp16");
  // int4 input is logical [N,H,W,C_packed] contiguous; C_logical = 2*C_packed (matches conv2d_int4_fprop).
  int N=input.size(0),H=input.size(1),W=input.size(2),C=2*weight_packed.size(3),K=weight_packed.size(0),R=weight_packed.size(1),S=weight_packed.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d1<I4,cutlass::int4b_t>(
      (cutlass::int4b_t*)input.data_ptr(), (cutlass::int4b_t*)weight_packed.data_ptr(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      bias.numel()? bias.data_ptr<float>() : nullptr,
      residual.numel()? reinterpret_cast<EC*>(residual.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor conv2d_int8_evt_o_hat_residual(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "o_hat/out fp16");
  // int8 input is logical [N,C,H,W] channels_last (matches conv2d_int8_fprop's dim extraction).
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor conv2d_int4_evt_o_hat_residual(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "o_hat/out fp16");
  // int4 input is logical [N,H,W,C_packed] contiguous; C_logical = 2*C_packed (matches conv2d_int4_fprop).
  int N=input.size(0),H=input.size(1),W=input.size(2),C=2*weight_packed.size(3),K=weight_packed.size(0),R=weight_packed.size(1),S=weight_packed.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2<I4,cutlass::int4b_t>(
      (cutlass::int4b_t*)input.data_ptr(), (cutlass::int4b_t*)weight_packed.data_ptr(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}
