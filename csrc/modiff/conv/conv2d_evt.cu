// ============================================================================================
// MoDiff EVT-fused conv2d epilogue: the D2 fusion. Baseline twin (D1): csrc/baseline/conv/conv2d_evt.cu
//
//   D2 (MoDiff): o_hat[elem] += acc*alpha*weight_scale[k]  (in place, fp16);
//                out = o_hat_new + residual[elem] -> fp16  (dual store)
//                replaces conv2d_intX_fprop_o_hat_residual's fp32 conv_out round-trip.
//
// weight_scale/bias are FP32 [K], matching the fp32 accumulate of the scale_accumulate kernels, so
// D2's o_hat stays BIT-EXACT against the unfused path. alpha is read on-device from a 1-elem fp32
// tensor (no host sync).
//
// Family 4 of the csrc/ datapath split (2026-08-12). run_d1 and the two *_evt_bias_residual_fp16
// exports stay baseline; run_d2 / run_d2nr and the four *_evt_o_hat* exports are here.
//
// The EVT type machinery (the anonymous-namespace preamble: Evt<> struct, the I8/I4 aliases, the
// Swizzle/alignment constants) and make_problem are COPIED, because both trees' drivers need them.
// They sit in an anonymous namespace in both files, so the duplication cannot collide at link time.
// KEEP THEM IDENTICAL to the baseline twin -- D2's o_hat is bit-exact against the unfused path only
// as long as the type parameters match. See csrc/README.md.
// ============================================================================================
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
#include "../common/common.cuh"

// ---- COPY of the anonymous-namespace EVT preamble ----
namespace {
using cutlass::layout::TensorNHWC;
using Arch = cutlass::arch::Sm80;
using EC = cutlass::half_t;          // output / aux (o_hat, residual) element
using ES = float;                    // per-channel weight_scale/bias element (fp32 -> bit-exact vs fp32 refs)
using ECompute = float;
namespace ct = cutlass::epilogue::threadblock;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
static int const kAlignC = 128 / cutlass::sizeof_bits<EC>::value;   // 8

// RegReduceFn for VisitorRowReduction must be `template <class> class`, i.e. exactly one
// parameter; cutlass::maximum_absolute_value_reduction takes two (T, PropagateNaN), so wrap it.
// Accumulates max(acc, |x|), which with reduction_identity 0 gives a per-channel abs-max.
template <class T>
struct AbsMaxReduce {
  CUTLASS_HOST_DEVICE T operator()(T const& acc, T const& x) const {
    T a = x < T(0) ? -x : x;
    return acc > a ? acc : a;
  }
};

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
  // Q8: o_hat stored as int8 with a per-output-channel scale. AuxLoad/AuxStore are element-generic
  // (vec_bits = kElementsPerAccess * sizeof_bits<Element>, so int8 gives a 64-bit vector access and
  // the same 8 elements/thread as the fp16 twin), and float->int8 goes through
  // `cvt.rni.sat.s8.f32`, which SATURATES -- so no clamp nodes are needed.
  using AuxLdQ = ct::VisitorAuxLoad<TileMap, int8_t, cute::Stride<int64_t, cute::_1, int64_t>>;
  using AuxStQ = ct::VisitorAuxStore<TileMap, int8_t, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
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
  using E_StOhat  = ct::Sm80EVT<AuxSt, E_OhatNew>;   // = EVTD2nr: o_hat RMW single store (no residual)
  using E_AddRes  = ct::Sm80EVT<Add, E_StOhat, AuxLd>;
  using EVTD2   = ct::Sm80EVT<AuxSt, E_AddRes>;
  using EVTD2nr = E_StOhat;
  // D2q: int8 o_hat RMW. dequant(codes, s_read) + conv, requantize on s_write_inv, store int8.
  // s_read is the scale the codes were WRITTEN with last step and s_write is this step's; both come
  // from a calibrated per-(layer, step, channel) table, which is the only granularity the EVT can
  // express -- a dynamic amax over all pixels would need every CTA to finish before the store, and
  // a scale frozen at t=T measures 4.50x of the run-to-run floor because some layers shrink 0.63x
  // over the trajectory and waste the range. Per-step per-channel measures 2.44x (+SR) / 2.78x
  // (none) -- indistinguishable. See docs/ohat_compress_2026-09-03/FINDINGS.md.
  using E_DeqQ  = ct::Sm80EVT<Mul, AuxLdQ, RowVec>;      // codes * s_read
  using E_NewQ  = ct::Sm80EVT<Add, E_DeqQ, E_MulWS>;     // + conv
  using E_CodeQ = ct::Sm80EVT<Mul, E_NewQ, RowVec>;      // * (1/s_write)
  using EVTD2q  = ct::Sm80EVT<AuxStQ, E_CodeQ>;          // -> int8, saturating
  // D2qr: same, plus a per-output-channel abs-max of the NEW o_hat written out as a side effect of
  // the same pass. That makes the scale self-contained -- step t writes the amax step t+1 needs --
  // so no calibrated table is required. VisitorRowReduction reduces along M into a [K] vector with
  // atomics (workspace 0, identity 0, so the buffer must be zeroed first) and passes the value
  // through, exactly like AuxStore does in the D2 tree.
  using RowRed = ct::VisitorRowReduction<AbsMaxReduce, cutlass::atomic_maximum, TileMap,
      ES, ECompute, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<cute::_0, cute::_1, int32_t>>;
  using E_RedQ   = ct::Sm80EVT<RowRed, E_NewQ>;
  using E_CodeQR = ct::Sm80EVT<Mul, E_RedQ, RowVec>;
  using EVTD2qr  = ct::Sm80EVT<AuxStQ, E_CodeQR>;
  // Skip-K: read o_hat_old, write only `out` (no o_hat store). Residual:
  //   out = o_hat_old + conv + residual. No-residual reuses EVTD2nr with AuxSt -> out.
  using E_AddResSkip = ct::Sm80EVT<Add, E_OhatNew, AuxLd>;
  using EVTD2skip = ct::Sm80EVT<AuxSt, E_AddResSkip>;

  using EpiD1   = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD1, 1>;
  using EpiD1nr = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD1nr, 1>;
  using EpiD2 = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD2, 1>;
  using EpiD2nr = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD2nr, 1>;
  using EpiD2skip = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD2skip, 1>;
  using EpiD2q = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD2q, 1>;
  using EpiD2qr = ct::EpilogueWithVisitorCallbacks<DefaultEpi, EVTD2qr, 1>;
  using KernelD1   = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD1,   Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD1nr = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD1nr, Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD2   = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD2,   Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD2nr = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD2nr, Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD2skip = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD2skip, Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD2q = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD2q, Swizzle, cutlass::conv::Operator::kFprop>;
  using KernelD2qr = modiff::ImplicitGemmConvolutionEVT<Mma, EpiD2qr, Swizzle, cutlass::conv::Operator::kFprop>;
  using OpD1   = cutlass::conv::device::ImplicitGemmConvolution<KernelD1>;
  using OpD1nr = cutlass::conv::device::ImplicitGemmConvolution<KernelD1nr>;
  using OpD2   = cutlass::conv::device::ImplicitGemmConvolution<KernelD2>;
  using OpD2nr = cutlass::conv::device::ImplicitGemmConvolution<KernelD2nr>;
  using OpD2skip = cutlass::conv::device::ImplicitGemmConvolution<KernelD2skip>;
  using OpD2q = cutlass::conv::device::ImplicitGemmConvolution<KernelD2q>;
  using OpD2qr = cutlass::conv::device::ImplicitGemmConvolution<KernelD2qr>;
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

// D2q: int8 o_hat RMW in place. o_hat_i8 = sat_i8( (codes*s_read + acc*alpha*weight_scale) * s_write_inv )
template <class EvtT, class ElemAB>
void run_d2q(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp,
             int8_t* qp, const ES* s_read, const ES* s_winv,
             cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
             int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); int8_t zq(0); int64_t MK = M * K;
  typename EvtT::EVTD2q::Arguments ep{
    {                                                              // E_CodeQ
      {                                                            //   E_NewQ
        {                                                          //     E_DeqQ
          {qp, zq, {ldK, cute::_1{}, MK}},                         //       AuxLdQ codes
          {const_cast<ES*>(s_read), z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
        { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },             //     E_MulA
          {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },     //     *weight_scale
        {} },                                                      //     Add
      {const_cast<ES*>(s_winv), z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
    {qp, {ldK, cute::_1{}, MK}}                                    // AuxStQ -> codes (in place)
  };
  using LN = TensorNHWC;
  typename EvtT::KernelD2q::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD2q::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  typename EvtT::OpD2q::Arguments args{problem, refA, refB, ep};
  typename EvtT::OpD2q op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d2q can_implement");
  auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d2q init");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d2q run");
}

// D2qr: as run_d2q, plus per-channel amax of the new o_hat into `amax_out` (zero it first).
template <class EvtT, class ElemAB>
void run_d2qr(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp,
              int8_t* qp, const ES* s_read, const ES* s_winv, ES* amax_out,
              cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
              int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); int8_t zq(0); int64_t MK = M * K;
  typename EvtT::EVTD2qr::Arguments ep{
    {                                                              // E_CodeQR
      {                                                            //   E_RedQ
        {                                                          //     E_NewQ
          {                                                        //       E_DeqQ
            {qp, zq, {ldK, cute::_1{}, MK}},
            {const_cast<ES*>(s_read), z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
          { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },
            {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
          {} },
        {amax_out, ECompute(0), {cute::_0{}, cute::_1{}, (int32_t)K}} },   //     RowRed
      {const_cast<ES*>(s_winv), z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },
    {qp, {ldK, cute::_1{}, MK}}
  };
  using LN = TensorNHWC;
  typename EvtT::KernelD2qr::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD2qr::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  typename EvtT::OpD2qr::Arguments args{problem, refA, refB, ep};
  typename EvtT::OpD2qr op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d2qr can_implement");
  auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d2qr init");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d2qr run");
}

// D2 no-residual: o_hat[elem] += acc*alpha*weight_scale[k] (in place, single store). No `out`.
template <class EvtT, class ElemAB>
void run_d2nr(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp, EC* ohatp,
              cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
              int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); EC ze(0); int64_t MK = M * K;
  typename EvtT::EVTD2nr::Arguments ep{
    {                                                          // E_OhatNew
      {ohatp, ze, {ldK, cute::_1{}, MK}},                      //   AuxLd o_hat_old
      { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },          //   E_MulA
        {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },  //   *weight_scale
      {} },                                                   // Add
    {ohatp, {ldK, cute::_1{}, MK}}                             // AuxSt -> o_hat (in place)
  };
  using LN = TensorNHWC;
  typename EvtT::KernelD2nr::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD2nr::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  typename EvtT::OpD2nr::Arguments args{problem, refA, refB, ep};
  typename EvtT::OpD2nr op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d2nr can_implement");
  auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d2nr init");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d2nr run");
}

// Skip-K residual: out = o_hat_old + conv + residual. o_hat is AuxLd only.
template <class EvtT, class ElemAB>
void run_d2_skip(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp, EC* ohatp, EC* resp, EC* outp,
            cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
            int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); EC ze(0); int64_t MK = M * K;
  typename EvtT::EVTD2skip::Arguments ep{
    {                                                            // E_AddResSkip
      {                                                          //   E_OhatNew
        {ohatp, ze, {ldK, cute::_1{}, MK}},                      //     AuxLd o_hat_old
        { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },          //     E_MulA
          {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },  //     *weight_scale
        {} },                                                   //   Add
      {resp, ze, {ldK, cute::_1{}, MK}}, {}                      //   +residual
    },
    {outp, {ldK, cute::_1{}, MK}}                                // AuxSt -> out only
  };
  using LN = TensorNHWC;
  typename EvtT::KernelD2skip::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD2skip::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  typename EvtT::OpD2skip::Arguments args{problem, refA, refB, ep};
  typename EvtT::OpD2skip op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d2skip can_implement");
  auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d2skip init");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d2skip run");
}

// Skip-K no-residual: out = o_hat_old + conv. Same EVTD2nr tree, AuxSt to `out` not o_hat.
template <class EvtT, class ElemAB>
void run_d2nr_skip(ElemAB* xp, ElemAB* wp, const float* alpha_ptr, ES* wsp, EC* ohatp, EC* outp,
              cutlass::conv::Conv2dProblemSize const& problem, int C, int R, int S, int K,
              int64_t M, int64_t ldK, cudaStream_t stream) {
  ES z(0); EC ze(0); int64_t MK = M * K;
  typename EvtT::EVTD2nr::Arguments ep{
    {                                                          // E_OhatNew
      {ohatp, ze, {ldK, cute::_1{}, MK}},                      //   AuxLd o_hat_old
      { { {}, {{ECompute(0)}, {alpha_ptr}, {}}, {} },          //   E_MulA
        {wsp, z, {cute::_0{}, cute::_1{}, (int32_t)K}}, {} },  //   *weight_scale
      {} },                                                   // Add
    {outp, {ldK, cute::_1{}, MK}}                             // AuxSt -> out (not o_hat)
  };
  using LN = TensorNHWC;
  typename EvtT::KernelD2nr::TensorRefA refA{xp, LN({C, problem.W*C, problem.H*problem.W*C})};
  typename EvtT::KernelD2nr::TensorRefB refB{wp, LN({C, S*C, R*S*C})};
  typename EvtT::OpD2nr::Arguments args{problem, refA, refB, ep};
  typename EvtT::OpD2nr op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "evt d2nr skip can_implement");
  auto ws = torch::empty({(long)op.get_workspace_size(args)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
  TORCH_CHECK(op.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "evt d2nr skip init");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "evt d2nr skip run");
}

// ---- COPY of make_problem ----
cutlass::conv::Conv2dProblemSize make_problem(int N, int H, int W, int C, int K, int R, int S,
                                              int sh, int sw, int ph, int pw, int dh, int dw,
                                              int& P, int& Q) {
  P = (H + 2*ph - dh*(R-1) - 1) / sh + 1;
  Q = (W + 2*pw - dw*(S-1) - 1) / sw + 1;
  return cutlass::conv::Conv2dProblemSize({N,H,W,C},{K,R,S,C},{ph,pw,ph,pw},{sh,sw},{dh,dw},
                                          {N,P,Q,K}, cutlass::conv::Mode::kCrossCorrelation, 1);
}

} // namespace

// ---- host entry points (moved) ----

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

// ---- D2 no-residual: o_hat RMW in place (replaces conv2d_intX_fprop_o_hat's fp32 round-trip) ----
torch::Tensor conv2d_int8_evt_o_hat(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16, "o_hat fp16");
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2nr<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return o_hat;
}

torch::Tensor conv2d_int4_evt_o_hat(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16, "o_hat fp16");
  int N=input.size(0),H=input.size(1),W=input.size(2),C=2*weight_packed.size(3),K=weight_packed.size(0),R=weight_packed.size(1),S=weight_packed.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2nr<I4,cutlass::int4b_t>(
      (cutlass::int4b_t*)input.data_ptr(), (cutlass::int4b_t*)weight_packed.data_ptr(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return o_hat;
}

// ---- D2q: int8 o_hat, per-output-channel scale ----
torch::Tensor conv2d_int8_evt_o_hat_q8(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat_q, torch::Tensor s_read, torch::Tensor s_write_inv,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat_q);
  TORCH_CHECK(o_hat_q.scalar_type()==torch::kChar, "o_hat_q must be int8");
  TORCH_CHECK(s_read.scalar_type()==torch::kFloat32 && s_write_inv.scalar_type()==torch::kFloat32,
              "o_hat scales must be fp32");
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  TORCH_CHECK(s_read.numel()==K && s_write_inv.numel()==K, "o_hat scales must be [K]");
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2q<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(), o_hat_q.data_ptr<int8_t>(),
      s_read.data_ptr<float>(), s_write_inv.data_ptr<float>(), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return o_hat_q;
}

torch::Tensor conv2d_int4_evt_o_hat_q8(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat_q, torch::Tensor s_read, torch::Tensor s_write_inv,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat_q);
  TORCH_CHECK(o_hat_q.scalar_type()==torch::kChar, "o_hat_q must be int8");
  int N=input.size(0),H=input.size(1),W=input.size(2),C=2*weight_packed.size(3),K=weight_packed.size(0),R=weight_packed.size(1),S=weight_packed.size(2);
  TORCH_CHECK(s_read.numel()==K && s_write_inv.numel()==K, "o_hat scales must be [K]");
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2q<I4,cutlass::int4b_t>(
      (cutlass::int4b_t*)input.data_ptr(), (cutlass::int4b_t*)weight_packed.data_ptr(),
      inv_scale.data_ptr<float>(), weight_scales.data_ptr<float>(), o_hat_q.data_ptr<int8_t>(),
      s_read.data_ptr<float>(), s_write_inv.data_ptr<float>(), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return o_hat_q;
}

torch::Tensor conv2d_int8_evt_o_hat_q8r(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat_q, torch::Tensor s_read, torch::Tensor s_write_inv, torch::Tensor amax_out,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat_q);
  TORCH_CHECK(o_hat_q.scalar_type()==torch::kChar, "o_hat_q must be int8");
  TORCH_CHECK(amax_out.scalar_type()==torch::kFloat32, "amax_out must be fp32");
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  TORCH_CHECK(s_read.numel()==K && s_write_inv.numel()==K && amax_out.numel()==K, "scales must be [K]");
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2qr<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(), o_hat_q.data_ptr<int8_t>(),
      s_read.data_ptr<float>(), s_write_inv.data_ptr<float>(), amax_out.data_ptr<float>(),
      problem, C,R,S,K,M,ldK, at::cuda::getCurrentCUDAStream());
  return o_hat_q;
}

// Skip-K: out = o_hat_old + conv, no o_hat store.
torch::Tensor conv2d_int8_evt_o_hat_skip(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "o_hat/out fp16");
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2nr_skip<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor conv2d_int4_evt_o_hat_skip(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "o_hat/out fp16");
  int N=input.size(0),H=input.size(1),W=input.size(2),C=2*weight_packed.size(3),K=weight_packed.size(0),R=weight_packed.size(1),S=weight_packed.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2nr_skip<I4,cutlass::int4b_t>(
      (cutlass::int4b_t*)input.data_ptr(), (cutlass::int4b_t*)weight_packed.data_ptr(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor conv2d_int8_evt_o_hat_residual_skip(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "o_hat/out fp16");
  int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3),K=weight.size(0),R=weight.size(1),S=weight.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2_skip<I8,int8_t>(
      input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}

torch::Tensor conv2d_int4_evt_o_hat_residual_skip(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw) {
  CHECK_CUDA(o_hat); TORCH_CHECK(o_hat.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "o_hat/out fp16");
  int N=input.size(0),H=input.size(1),W=input.size(2),C=2*weight_packed.size(3),K=weight_packed.size(0),R=weight_packed.size(1),S=weight_packed.size(2);
  int P,Q; auto problem = make_problem(N,H,W,C,K,R,S,sh,sw,ph,pw,dh,dw,P,Q);
  int64_t M=(int64_t)N*P*Q, ldK=K;
  run_d2_skip<I4,cutlass::int4b_t>(
      (cutlass::int4b_t*)input.data_ptr(), (cutlass::int4b_t*)weight_packed.data_ptr(), inv_scale.data_ptr<float>(),
      weight_scales.data_ptr<float>(),
      reinterpret_cast<EC*>(o_hat.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<EC*>(output.data_ptr<at::Half>()), problem, C,R,S,K,M,ldK,
      at::cuda::getCurrentCUDAStream());
  return output;
}
