// Custom conv2d-fprop kernel driver that accepts an Sm80 Epilogue Visitor Tree (EVT) epilogue.
// CUTLASS 4.6.1's stock cutlass::conv::kernel::ImplicitGemmConvolution hard-codes the classic
// 4-arg thread-epilogue convention and pulls ElementOutput/ElementCompute off the output op, so it
// cannot drive an EpilogueWithVisitorCallbacks epilogue (EVT is wired only into
// gemm::kernel::GemmUniversalwithVisitor). This is a minimal clone of
// cutlass/conv/kernel/implicit_gemm_convolution.h with the epilogue section replaced by the
// GEMM-visitor convention (gemm_universal_with_visitor.h L303-311): construct the epilogue
// Params-first (5 args) and call it as (accumulators, tile_offset, problem_shape, thread_idx).
// No split-K (Sm80 EVT does not support it; these convs use a single K-slice).
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cute/tensor.hpp"

namespace modiff {

template <
  typename Mma_,
  typename Epilogue_,                       // an EpilogueWithVisitorCallbacks (EVT) epilogue
  typename ThreadblockSwizzle_,
  cutlass::conv::Operator ConvOperator,
  typename ConvProblemSize_ = cutlass::conv::Conv2dProblemSize
>
struct ImplicitGemmConvolutionEVT {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;        // EVT tag: has ElementAccumulator, Params
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static cutlass::conv::Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA  = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB  = typename Mma::IteratorB::Layout;
  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  // Typedef surface the device adapter (conv/device/implicit_gemm_convolution.h) reads off the
  // kernel. The EVT OutputOp is a tag (no ElementOutput/ElementCompute), so ElementC/ElementCompute
  // are fixed here (our EVT convs are fp16-out / fp32-compute).
  using ElementC = cutlass::half_t;
  using LayoutC  = LayoutA;
  using ElementCompute = float;
  using WarpMmaOperator = typename Mma::Policy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator    = typename ArchMmaOperator::Operator;
  using OperatorClass   = typename WarpMmaOperator::OperatorClass;
  using ArchTag         = typename WarpMmaOperator::ArchTag;
  using WarpShape       = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;
  static cutlass::conv::StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;
  static int const kConvDim = Mma::IteratorA::kConvDim;

  using ThreadblockShape = typename Mma::Shape;
  static int const kStages = Mma::kStages;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm;
  static cutlass::conv::GroupMode const kGroupMode = cutlass::conv::GroupMode::kNone;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;

  using ConvProblemSize = ConvProblemSize_;

  // ---- Arguments (host) ----
  struct Arguments {
    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    typename EpilogueOutputOp::Params output_op;   // = Epilogue::FusionCallbacks::Arguments (EVT arg tree)
    cutlass::conv::SplitKMode split_k_mode;        // read by the device adapter (single K-slice only)

    CUTLASS_HOST_DEVICE Arguments(): split_k_mode(cutlass::conv::SplitKMode::kSerial) {}
    CUTLASS_HOST_DEVICE Arguments(
      ConvProblemSize const &problem_size,
      TensorRefA const &ref_A,
      TensorRefB const &ref_B,
      typename EpilogueOutputOp::Params const &output_op,
      cutlass::conv::SplitKMode const &split_k_mode = cutlass::conv::SplitKMode::kSerial
    ): problem_size(problem_size), ref_A(ref_A), ref_B(ref_B), output_op(output_op),
       split_k_mode(split_k_mode) {}
  };

  // ---- Params (device) ----
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    cutlass::gemm::GemmCoord implicit_gemm_problem_size;
    int swizzle_log_tile;
    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;
    typename Epilogue::Params output_op;                       // EVT device params
    cute::Shape<int32_t,int32_t,int32_t> problem_shape;

    CUTLASS_HOST_DEVICE Params(): swizzle_log_tile(0), gemm_k_iterations(0) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args, int * /*semaphore*/ = nullptr):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      output_op(Epilogue::FusionCallbacks::to_underlying_arguments(
        cute::make_shape(
          cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size).m(),
          cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size).n(),
          int32_t(1)),
        args.output_op, nullptr)),
      problem_shape(cute::make_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size).m(),
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size).n(),
        int32_t(1)))
    {
      gemm_k_iterations = cutlass::conv::implicit_gemm_k_iterations(
        kConvolutionalOperator, ThreadblockShape::kK, args.problem_size,
        kIteratorAlgorithm, kGroupMode, ThreadblockShape::kN);
      gemm_k_iterations_per_channel = cutlass::conv::implicit_gemm_k_iterations_per_channel(
        kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);
      ThreadblockSwizzle threadblock_swizzle;
      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_HOST_DEVICE ImplicitGemmConvolutionEVT() {}

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
      return;
    }
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iterator_A(
      params.iterator_A, params.problem_size, params.ptr_A, thread_idx,
      cutlass::MatrixCoord(threadblock_tile_idx.m() * Mma::Shape::kM,
                           threadblock_tile_idx.k() * Mma::Shape::kK));
    typename Mma::IteratorB iterator_B(
      params.iterator_B, params.problem_size, params.ptr_B, thread_idx,
      cutlass::MatrixCoord(threadblock_tile_idx.k() * Mma::Shape::kK,
                           threadblock_tile_idx.n() * Mma::Shape::kN));

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators,
        params.gemm_k_iterations_per_channel);

    // ---- EVT epilogue (GEMM-visitor convention: Params-first ctor, tile-offset call) ----
    threadblock_tile_idx = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    Epilogue epilogue(params.output_op, shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(accumulators, threadblock_tile_idx, params.problem_shape, thread_idx);
  }
};

} // namespace modiff
