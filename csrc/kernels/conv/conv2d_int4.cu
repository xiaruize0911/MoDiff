// =========================================================================
// CUTLASS INT4 Conv2d integration.
//
//   Op: Conv2d W4A4 -- CUTLASS implicit-GEMM int4 convolution, NHWC (Ampere/Sm80).
//
// Same structure as conv2d_int8.cu, but both activations and weights are
// packed 2-per-byte INT4 values (see quantize.cu / modiff_delta_quantize.cu
// for the packing kernels). Inputs are physically (N, H, W, C/2) and weights
// (K, R, S, C/2); CUTLASS is told the *logical* (unpacked) channel count via
// the problem size so its INT4 tensor-core path can address individual nibbles.
//
// Naming convention mirrors conv2d_int8.cu (suffixes compose; verified below):
//   *_fprop              = base implicit-GEMM conv; int32 accumulate, output scaled only by the
//                          scalar activation alpha (`scales`/`inv_scale`) -- no per-channel weight scale.
//   *_dequant_fp16_tuned = "deep-fuse" epilogue: per-channel weight scale folded into the CUTLASS
//                          epilogue, writes FP16 directly (no FP32 temp); tile picked by config_id.
//   *_tuned / *_num_tuned_configs = threadblock/warp tile is a template param; the Python autotuner
//                          picks the fastest `config_id` per shape.
//   *_o_hat              = MoDiff delta cache: o_hat_cache += conv * weight_scale[ch] (separate kernel).
//   *_no_ohat[_prealloc] = plain per-channel dequant store, no cache; _prealloc writes a caller buffer.
//   *_bias / *_bias_residual = + per-channel bias / + bias + per-element skip residual.
//   *_relu_requant_int4  = dequant(+bias)+optional ReLU, requantized AND re-packed to INT4 output
//                          (keeps the chain int4->int4). NOTE: this + *_bias_residual_dual run the
//                          deep-fuse GEMM internally, so their `weight_scales` arg is FP16, not fp32.
//   *_dual               = DUAL output: FP16 block output (x_{N+1}) + requantized packed-INT4 (next
//                          block conv1 input) in one store, fusing the block-entry quantize.
//
//   vs fp16: Standalone int8 conv (dynamic quantize + int8 GEMM + fp16 dequant output) measured
//     0.48-1.16x vs fp16 cuDNN on churches shapes (b128): loses on large-spatial/low-channel,
//     wins only high-channel/low-spatial (768ch 4x4 = 1.16x) -- cuDNN fp16 conv is highly optimized
//     and the per-call quantize+dequant is overhead. The e2e ~2x int8 win comes from the FUSED
//     int8->int8 chain (deepfuse relu+requant output, quantize done once at entry, no fp16
//     round-trip), which the standalone ratio understates (same effect as linear's fused-quant).
// =========================================================================

#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

#include "common.cuh"
#include "conv_epilogue.cuh"

using Arch = cutlass::arch::Sm80;

using Conv2dInt4Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  cutlass::int4b_t, cutlass::layout::TensorNHWC,
  cutlass::int4b_t, cutlass::layout::TensorNHWC,
  float, cutlass::layout::TensorNHWC,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  Arch,
  cutlass::gemm::GemmShape<128, 128, 128>,
  cutlass::gemm::GemmShape<64, 64, 128>,
  cutlass::gemm::GemmShape<16, 8, 64>, // INT4 often uses K=64 for instruction shape on Ampere
  cutlass::epilogue::thread::LinearCombination<
    float, 1, int32_t, float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3, // Stages (was 2, 3 improves memory latency pipelining)
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

using Conv2dInt4Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt4Kernel>;

// Deep-fuse epilogue for int4 (identical logic to conv2d_int8.cu's
// Int8DequantScaleSource; kept separate per TU to avoid an ODR clash). Dequantizes
// the int32 accumulator straight to FP16 as accumulator * alpha * source, where
// alpha = the activation inverse quant scale (scalar, device ptr) and source = the
// per-output-channel weight scale (fp16, broadcast as the epilogue "C" operand). This
// folds the per-channel weight_scale into the GEMM epilogue so the int4 conv writes
// FP16 directly -- eliminating the FP32 conv_out temporary the plain path materialized.
template <int Count>
class Int4DequantScaleSource {
public:
  using ElementOutput = cutlass::half_t;
  using ElementSource = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementScalar = ElementCompute;
  using ElementC = ElementSource;
  using ElementD = ElementOutput;
  static int const kCount = Count;
  static cutlass::epilogue::thread::ScaleType::Kind const kScale =
      cutlass::epilogue::thread::ScaleType::Default;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentSource = cutlass::Array<ElementSource, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  struct Params {
    ElementCompute alpha;
    ElementCompute const *alpha_ptr;
    CUTLASS_HOST_DEVICE Params(): alpha(ElementCompute(1)), alpha_ptr(nullptr) {}
    CUTLASS_HOST_DEVICE Params(ElementCompute alpha): alpha(alpha), alpha_ptr(nullptr) {}
    CUTLASS_HOST_DEVICE Params(ElementCompute const *alpha_ptr): alpha(ElementCompute(1)), alpha_ptr(alpha_ptr) {}
  };
private:
  ElementCompute alpha_;
public:
  CUTLASS_HOST_DEVICE explicit Int4DequantScaleSource(Params const &params) {
    alpha_ = params.alpha_ptr ? *params.alpha_ptr : params.alpha;
  }
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
  CUTLASS_HOST_DEVICE FragmentOutput operator()(
      FragmentAccumulator const &accumulator, FragmentSource const &source) const {
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i)
      output[i] = ElementOutput(float(accumulator[i]) * alpha_ * float(source[i]));
    return output;
  }
  CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i)
      output[i] = ElementOutput(float(accumulator[i]) * alpha_);
    return output;
  }
};

// Base CUTLASS INT4 conv. `scales` is the *scalar* activation dequant factor,
// same convention as conv2d_int8_fprop — per-channel weight dequant always
// happens separately (scale_accumulate/scale_store, or a Python-side multiply).
//   Inputs:   input packed-int4 (int8 storage) [N,H,W,C/2] contiguous (NHWC physical, 2 nibbles/byte);
//             weight_packed packed-int4 [K,R,S,C/2]; scales fp32 (1-elem = CUTLASS alpha = activation
//             dequant); bias fp32 [K] or empty; stride/pad/dilation.
//   Outputs:  fp32 [N,K,H_out,W_out] channels_last = int32_accum * alpha (+ bias broadcast).
//   Fuses:    scalar-alpha dequant + optional bias (CUTLASS epilogue). NOT per-channel weight scale.
//   Constraints: logical C = 2*C_packed; input.size(3)==weight_packed.size(3); K=128 threadblock tile
//             (K-tile minimum for int4); device-alpha only when scales is 1-elem CUDA and bias empty.
torch::Tensor conv2d_int4_fprop(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(input);

    // Input is packed (N, H, W, C/2) contiguous; C/2 is the packed channel dim.
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);

    // Weight is packed (K, R, S, C/2) contiguous.
    int K_packed = weight_packed.size(0);
    int R = weight_packed.size(1);
    int S = weight_packed.size(2);
    int C_packed = weight_packed.size(3);

    int C_logical = C_packed * 2;

    TORCH_CHECK(input.size(3) == C_packed, "Input/Weight channel mismatch");

    void* input_ptr_raw = input.data_ptr();

    cutlass::int4b_t* input_ptr = (cutlass::int4b_t*)input_ptr_raw;
    cutlass::int4b_t* weight_ptr = (cutlass::int4b_t*)weight_packed.data_ptr();

    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    int K_logical = K_packed;

    auto out_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto output = torch::empty({N, K_logical, H_out, W_out}, out_options);

    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C_logical},
        {K_packed, R, S, C_logical},
        {padding_h, padding_h, padding_w, padding_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {N, H_out, W_out, K_packed},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );

    // Build epilogue params: use device pointer when scales is on GPU to avoid D2H sync
    using Int4EpilogueParams = typename Conv2dInt4Op::EpilogueOutputOp::Params;
    Int4EpilogueParams ep;
    bool use_device_alpha = (scales.numel() == 1 && scales.is_cuda() && bias.numel() == 0);
    if (use_device_alpha) {
        ep = Int4EpilogueParams(scales.data_ptr<float>());
    } else {
        float alpha = 1.0f;
        if (scales.numel() == 1) alpha = scales.item<float>();
        float beta = (bias.numel() > 0) ? 1.0f : 0.0f;
        ep = Int4EpilogueParams(alpha, beta);
    }

    Conv2dInt4Op op;
    Conv2dInt4Op::Arguments args(
        problem_size,
        {input_ptr, {C_logical, W * C_logical, H * W * C_logical}},
        {weight_ptr, {C_logical, S * C_logical, R * S * C_logical}},
        {(float*)((bias.numel() > 0) ? bias.data_ptr() : nullptr), {0,0,0}},
        {output.data_ptr<float>(), {K_packed, W_out * K_packed, H_out * W_out * K_packed}},
        ep
    );

    // See conv2d_int8_fprop for why this check exists: fail synchronously and
    // clearly for problem sizes this tile configuration can't handle, instead
    // of letting a bad launch corrupt the stream asynchronously.
    cutlass::Status can_status = Conv2dInt4Op::can_implement(args);
    if (can_status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT4 Kernel cannot implement this problem size: "
                  << cutlass::cutlassGetStatusString(can_status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT4 Kernel cannot implement this problem size (N=", N,
                    " C=", C_logical, " H=", H, " W=", W, " K=", K_packed, " R=", R, " S=", S, ")");
    }

    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));

    cutlass::Status status = op(args, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT4 Kernel Failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT4 Kernel failed");
    }

    return output;
}

// ---- Multi-tile (autotunable) INT4 conv, FP32 out ----
// Tile-parametric mirror of Conv2dInt4Kernel: threadblock/warp shape + pipeline
// stages are template params so several configs can be instantiated and the fastest
// picked per-shape at calibration time (the int4 analogue of the int8 tuned tile set;
// closes the "int4 is single-tile" gap). Epilogue stays fp32 LinearCombination (the
// per-channel weight_scale is applied in the store kernels, orthogonal to the tile).
template <typename ThreadblockShape, typename WarpShape, int Stages>
struct Int4ConvConfig {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::int4b_t, cutlass::layout::TensorNHWC,
    cutlass::int4b_t, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    Arch,
    ThreadblockShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<float, 1, int32_t, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided
  >::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

template <typename Op>
static bool run_int4_fprop_tuned(
    cutlass::conv::Conv2dProblemSize const& ps,
    cutlass::int4b_t* ip, cutlass::int4b_t* wp, float* out_ptr, float* alpha_ptr,
    int C, int W, int H, int K, int R, int S, int H_out, int W_out,
    torch::TensorOptions const& opts, cudaStream_t stream) {
  using EpParams = typename Op::EpilogueOutputOp::Params;
  EpParams ep(alpha_ptr);   // device-side scalar alpha (= inv_scale), beta = 0
  typename Op::Arguments args(
      ps,
      {ip, {C, W * C, H * W * C}},
      {wp, {C, S * C, R * S * C}},
      {(float*)nullptr, {0, 0, 0}},
      {out_ptr, {K, W_out * K, H_out * W_out * K}},
      ep);
  if (Op::can_implement(args) != cutlass::Status::kSuccess) return false;
  Op op;
  size_t ws = op.get_workspace_size(args);
  auto workspace = torch::empty({(long)ws}, opts.dtype(torch::kByte));
  return op(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess;
}

// int4 tile set (K=128 threadblock, instruction K=64). Wide-N tiles (2,5) target the
// memory-bound 1x1 expand convs; square tiles the big-K 3x3s. 0 = default.
using I4Cfg0 = Int4ConvConfig<cutlass::gemm::GemmShape<128,128,128>, cutlass::gemm::GemmShape<64,64,128>, 3>;
using I4Cfg1 = Int4ConvConfig<cutlass::gemm::GemmShape<256,128,128>, cutlass::gemm::GemmShape<64,64,128>, 3>;
using I4Cfg2 = Int4ConvConfig<cutlass::gemm::GemmShape<128,256,128>, cutlass::gemm::GemmShape<64,64,128>, 3>;
using I4Cfg3 = Int4ConvConfig<cutlass::gemm::GemmShape<128, 64,128>, cutlass::gemm::GemmShape<64,32,128>, 4>;
using I4Cfg4 = Int4ConvConfig<cutlass::gemm::GemmShape< 64,128,128>, cutlass::gemm::GemmShape<32,64,128>, 4>;
using I4Cfg5 = Int4ConvConfig<cutlass::gemm::GemmShape< 64,256,128>, cutlass::gemm::GemmShape<32,64,128>, 3>;
// NOTE: warp-K must be >=128 for int4 (CUTLASS mma_base requires kWarpGemmIterations =
// warpK/instK = warpK/64 >= 2 and even -> warpK>=128). So the K=128 threadblock tile is
// the MINIMUM; a K=64 tile does not compile. The 6 configs above span the full tunable
// M/N space and cluster within ~1% on small-N/small-spatial 3x3 (tuning is exhausted there).

// Number of int4 tile configs the autotuner may pick from.
//   Inputs:   none.   Outputs: int64 count (currently 6; valid config_id range [0,6)).
//   Fuses:    none (metadata for conv2d_int4_fprop_tuned / conv2d_int4_dequant_fp16_tuned).
int64_t conv2d_int4_num_tuned_configs() { return 6; }

// Run tile `config_id` of the int4 conv -> fresh FP32 [N,K,H_out,W_out] channels_last
// output (= raw_int32 * inv_scale). config_id < 0 uses the default fixed-tile
// conv2d_int4_fprop. The autotuner in Python tries each and skips ones that can't
// implement the shape.
//   Inputs:   input packed-int4 [N,H,W,C/2]; weight_packed [K,R,S,C/2]; inv_scale fp32 (1-elem
//             device alpha); config_id int64 (<0 = default fixed-tile conv2d_int4_fprop); stride/pad/dilation.
//   Outputs:  fresh fp32 [N,K,H_out,W_out] channels_last = int32_accum * inv_scale.
//   Fuses:    scalar-alpha dequant only (fp32 LinearCombination epilogue, beta=0); tile autotuned.
//   Constraints: config_id in [-1,6); raises if the tile can't implement the shape.
torch::Tensor conv2d_int4_fprop_tuned(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) {
  if (config_id < 0) {
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    return conv2d_int4_fprop(input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(input);
  int N = input.size(0), H = input.size(1), W = input.size(2);
  int K = weight_packed.size(0), R = weight_packed.size(1), S = weight_packed.size(2);
  int C = weight_packed.size(3) * 2;
  TORCH_CHECK(input.size(3) == weight_packed.size(3), "Input/Weight channel mismatch");
  int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
  int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
  auto out_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device())
      .memory_format(torch::MemoryFormat::ChannelsLast);
  auto output = torch::empty({N, K, H_out, W_out}, out_options);
  cutlass::conv::Conv2dProblemSize ps(
      {N, H, W, C}, {K, R, S, C}, {padding_h, padding_h, padding_w, padding_w},
      {stride_h, stride_w}, {dilation_h, dilation_w}, {N, H_out, W_out, K},
      cutlass::conv::Mode::kCrossCorrelation, 1);
  auto* ip = (cutlass::int4b_t*)input.data_ptr();
  auto* wp = (cutlass::int4b_t*)weight_packed.data_ptr();
  float* alpha = inv_scale.data_ptr<float>();
  auto opts = input.options();
  bool ok = false;
#define RUN_I4CFG(CFG) run_int4_fprop_tuned<CFG::Op>(ps, ip, wp, output.data_ptr<float>(), alpha, C, W, H, K, R, S, H_out, W_out, opts, stream)
  switch (config_id) {
    case 0: ok = RUN_I4CFG(I4Cfg0); break;
    case 1: ok = RUN_I4CFG(I4Cfg1); break;
    case 2: ok = RUN_I4CFG(I4Cfg2); break;
    case 3: ok = RUN_I4CFG(I4Cfg3); break;
    case 4: ok = RUN_I4CFG(I4Cfg4); break;
    case 5: ok = RUN_I4CFG(I4Cfg5); break;
    default: TORCH_CHECK(false, "invalid int4 config_id ", config_id);
  }
#undef RUN_I4CFG
  TORCH_CHECK(ok, "int4 tuned config ", config_id, " cannot implement (N=", N,
              " C=", C, " H=", H, " W=", W, " K=", K, " R=", R, ")");
  return output;
}

// ---- Deep-fuse (FP16-out, no fp32 temp) INT4 conv, tile-autotunable ----
// Folds per-channel weight_scale into the CUTLASS epilogue (Int4DequantScaleSource)
// so the conv writes fully-scaled fp16 directly -- the int4 analogue of
// conv2d_int8_dequant_fp16_tuned. Removes the fp32 conv_out round-trip.
template <typename ThreadblockShape, typename WarpShape, int Stages>
struct Int4DequantFp16Config {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::int4b_t, cutlass::layout::TensorNHWC,
    cutlass::int4b_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    int32_t,
    cutlass::arch::OpClassTensorOp, Arch,
    ThreadblockShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 64>,
    Int4DequantScaleSource<8>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages, cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized, cutlass::conv::StrideSupport::kStrided
  >::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

template <typename Op>
static bool run_int4_dequant_fp16(
    cutlass::conv::Conv2dProblemSize const& ps,
    cutlass::int4b_t* ip, cutlass::int4b_t* wp, cutlass::half_t* scale_ptr,
    cutlass::half_t* out_ptr, float* alpha_ptr,
    int C, int W, int H, int K, int R, int S, int H_out, int W_out,
    torch::TensorOptions const& opts, cudaStream_t stream) {
  using EpParams = typename Op::EpilogueOutputOp::Params;
  EpParams ep(alpha_ptr);
  typename Op::Arguments args(
      ps,
      {ip, {C, W * C, H * W * C}},
      {wp, {C, S * C, R * S * C}},
      {scale_ptr, {0, 0, 0}},
      {out_ptr, {K, W_out * K, H_out * W_out * K}},
      ep);
  if (Op::can_implement(args) != cutlass::Status::kSuccess) return false;
  Op op;
  size_t ws = op.get_workspace_size(args);
  auto workspace = torch::empty({(long)ws}, opts.dtype(torch::kByte));
  return op(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess;
}

using I4DF0 = Int4DequantFp16Config<cutlass::gemm::GemmShape<128,128,128>, cutlass::gemm::GemmShape<64,64,128>, 3>;
using I4DF1 = Int4DequantFp16Config<cutlass::gemm::GemmShape<256,128,128>, cutlass::gemm::GemmShape<64,64,128>, 3>;
using I4DF2 = Int4DequantFp16Config<cutlass::gemm::GemmShape<128,256,128>, cutlass::gemm::GemmShape<64,64,128>, 3>;
using I4DF3 = Int4DequantFp16Config<cutlass::gemm::GemmShape<128, 64,128>, cutlass::gemm::GemmShape<64,32,128>, 4>;
using I4DF4 = Int4DequantFp16Config<cutlass::gemm::GemmShape< 64,128,128>, cutlass::gemm::GemmShape<32,64,128>, 4>;
using I4DF5 = Int4DequantFp16Config<cutlass::gemm::GemmShape< 64,256,128>, cutlass::gemm::GemmShape<32,64,128>, 3>;

// Deep-fuse int4 conv writing fully-scaled FP16 into a preallocated `output`
// (channels_last [N,K,H_out,W_out]). config_id<0 -> default tile (0).
//   Inputs:   input packed-int4 [N,H,W,C/2]; weight_packed [K,R,S,C/2]; inv_scale fp32 (1-elem alpha);
//             weight_scales_half fp16 [K] (folded into the epilogue); output fp16 [N,K,H_out,W_out]
//             cl PREallocated; config_id int64 (<0 -> tile 0); stride/pad/dilation.
//   Outputs:  output (fp16, in place) = int32_accum * inv_scale * weight_scale[ch].
//   Fuses:    deep-fuse dequant (scalar alpha + per-channel weight scale in CUTLASS epilogue) -> fp16,
//             NO fp32 temp; tile autotuned.
//   Constraints: output & weight_scales_half fp16; config_id in [0,6) (or <0 -> 0); K % 8 == 0.
torch::Tensor conv2d_int4_dequant_fp16_tuned(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(input); CHECK_CUDA(output); CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be float16");
  TORCH_CHECK(weight_scales_half.scalar_type() == torch::kFloat16, "weight scales must be float16");
  int N = input.size(0), H = input.size(1), W = input.size(2);
  int K = weight_packed.size(0), R = weight_packed.size(1), S = weight_packed.size(2);
  int C = weight_packed.size(3) * 2;
  TORCH_CHECK(input.size(3) == weight_packed.size(3), "Input/Weight channel mismatch");
  int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
  int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
  cutlass::conv::Conv2dProblemSize ps(
      {N, H, W, C}, {K, R, S, C}, {padding_h, padding_h, padding_w, padding_w},
      {stride_h, stride_w}, {dilation_h, dilation_w}, {N, H_out, W_out, K},
      cutlass::conv::Mode::kCrossCorrelation, 1);
  auto* ip = (cutlass::int4b_t*)input.data_ptr();
  auto* wp = (cutlass::int4b_t*)weight_packed.data_ptr();
  auto* sp = reinterpret_cast<cutlass::half_t*>(weight_scales_half.data_ptr<at::Half>());
  auto* op = reinterpret_cast<cutlass::half_t*>(output.data_ptr<at::Half>());
  float* alpha = inv_scale.data_ptr<float>();
  auto opts = input.options();
  if (config_id < 0) config_id = 0;
  bool ok = false;
#define RUN_I4DF(CFG) run_int4_dequant_fp16<CFG::Op>(ps, ip, wp, sp, op, alpha, C, W, H, K, R, S, H_out, W_out, opts, stream)
  switch (config_id) {
    case 0: ok = RUN_I4DF(I4DF0); break;
    case 1: ok = RUN_I4DF(I4DF1); break;
    case 2: ok = RUN_I4DF(I4DF2); break;
    case 3: ok = RUN_I4DF(I4DF3); break;
    case 4: ok = RUN_I4DF(I4DF4); break;
    case 5: ok = RUN_I4DF(I4DF5); break;
    default: TORCH_CHECK(false, "invalid int4 deep-fuse config_id ", config_id);
  }
#undef RUN_I4DF
  TORCH_CHECK(ok, "int4 deep-fuse config ", config_id, " cannot implement (N=", N,
              " C=", C, " H=", H, " W=", W, " K=", K, " R=", R, ")");
  return output;
}

// Deep-fuse int4 conv (weight_scale folded into the CUTLASS epilogue -> fp16, no
// fp32 temp) + per-channel bias + optional per-element skip residual, in a single
// fp16 `output`. The int4 analogue of conv2d_int8_fprop_deepfuse_bias_residual_fp16:
// replaces conv2d_int4_fprop_no_ohat_prealloc_bias[_residual]'s fp32 conv_out +
// scale_bias[_residual]_store_half<float> (reads fp32) with a deep-fuse conv into an
// fp16 scratch + a cheap from_half store (reads fp16, half the store bandwidth).
//   Inputs:   input packed-int4 [N,H,W,C/2]; weight_packed [K,R,S,C/2]; inv_scale fp32 (1-elem alpha);
//             weight_scales_half fp16 [K]; bias fp32/fp16 [K] or empty; residual fp16
//             [N,K,H_out,W_out] cl or empty; output fp16 (same shape) PREallocated; config_id int64
//             (<0 -> default tile); stride/pad/dilation.
//   Outputs:  output (fp16, in place) = fp16(int32_accum * alpha * weight_scale[ch]) + bias[ch] + residual.
//   Fuses:    deep-fuse dequant (per-channel weight scale in epilogue, NO fp32 temp) + bias + skip residual.
//   Constraints: output & weight_scales_half (& residual if given) fp16; K % 8 == 0; bias numel()==K or 0.
torch::Tensor conv2d_int4_fprop_deepfuse_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor bias, torch::Tensor residual,
    torch::Tensor output, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(output); CHECK_CONTIGUOUS(output);
    TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be float16");

    // Deep-fuse conv writes fully weight-scaled fp16 into scratch (no fp32 temp).
    auto scratch = torch::empty_like(output);
    conv2d_int4_dequant_fp16_tuned(
        input, weight_packed, inv_scale, weight_scales_half, scratch, config_id,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);

    const bool has_bias = bias.numel() > 0;
    const bool has_res = residual.numel() > 0;
    if (has_bias) TORCH_CHECK(bias.numel() == weight_scales_half.numel(), "bias size mismatch");
    if (has_res) TORCH_CHECK(residual.scalar_type() == torch::kFloat16 && residual.numel() == output.numel(),
                             "residual must be fp16 and match output shape");
    int num_elements = scratch.numel();
    int num_channels = weight_scales_half.numel();
    int block_size = 256, grid_size = (num_elements + block_size - 1) / block_size;
    const __half* deq = reinterpret_cast<const __half*>(scratch.data_ptr<at::Half>());
    __half* out_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    const bool bias_half = has_bias && bias.scalar_type() == torch::kFloat16;

    if (has_res) {
        const __half* res = reinterpret_cast<const __half*>(residual.data_ptr<at::Half>());
        if (bias_half)
            bias_residual_store_half_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                deq, reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()), res, out_ptr, num_elements, num_channels);
        else if (has_bias)
            bias_residual_store_half_from_half_kernel<float><<<grid_size, block_size, 0, stream>>>(
                deq, bias.data_ptr<float>(), res, out_ptr, num_elements, num_channels);
        else
            bias_residual_store_half_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                deq, (const __half*)nullptr, res, out_ptr, num_elements, num_channels);
    } else {
        if (bias_half)
            bias_store_half_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                deq, reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()), out_ptr, num_elements, num_channels);
        else if (has_bias)
            bias_store_half_from_half_kernel<float><<<grid_size, block_size, 0, stream>>>(
                deq, bias.data_ptr<float>(), out_ptr, num_elements, num_channels);
        else
            bias_store_half_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                deq, (const __half*)nullptr, out_ptr, num_elements, num_channels);
    }
    return output;
}

// conv2d_int4_fprop, then output = raw_output * weight_scales[channel], written
// into a caller-provided buffer (see conv2d_int8_fprop_no_ohat_prealloc for the
// same "prealloc only covers the final buffer" caveat).
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales fp32 [K];
//             output fp32 or fp16 [N,K,H_out,W_out] cl PREallocated; stride/pad/dilation.
//   Outputs:  output (fp32/fp16, in place) = int32_accum * inv_scale * weight_scale[ch].
//   Fuses:    per-channel weight-scale dequant (separate scale_store[_half] kernel). No o_hat.
//   Constraints: output fp32 or fp16, shape must match the conv output.
torch::Tensor conv2d_int4_fprop_no_ohat_prealloc(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    auto conv_out = conv2d_int4_fprop(
        input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(output);
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat32 || output.scalar_type() == torch::kFloat16,
        "output must be float32 or float16"
    );
    TORCH_CHECK(output.sizes() == conv_out.sizes(), "output shape mismatch for conv2d_int4_fprop_no_ohat_prealloc");

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    if (output.scalar_type() == torch::kFloat16) {
        if (num_channels % 2 == 0) {
            int grid_size_vec2 = (((num_elements + 1) / 2) + block_size - 1) / block_size;
            scale_store_half_vec2_kernel<<<grid_size_vec2, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                num_elements,
                num_channels
            );
        } else {
            scale_store_half_kernel<<<grid_size, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                num_elements,
                num_channels
            );
        }
    } else {
        scale_store_kernel<<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            output.data_ptr<float>(),
            num_elements,
            num_channels
        );
    }

    return output;
}

// Same as conv2d_int4_fprop_no_ohat_prealloc, but also adds a per-channel bias.
// Mirrors conv2d_int8.cu's conv2d_int8_fprop_no_ohat_prealloc_bias: the
// scale_bias_store_kernel / scale_bias_store_half_kernel epilogue templates in
// conv_epilogue.cuh are shared between the INT8 and INT4 paths, so no new
// kernel body is needed here, only this wrapper.
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales fp32 [K];
//             bias fp32 or fp16 [K]; output fp32 or fp16 [N,K,H_out,W_out] cl PREallocated.
//   Outputs:  output (in place) = int32_accum * inv_scale * weight_scale[ch] + bias[ch].
//   Fuses:    per-channel weight-scale dequant + per-channel bias (separate scale_bias_store kernel).
//   Constraints: output & bias fp32 or fp16; bias.numel()==K.
torch::Tensor conv2d_int4_fprop_no_ohat_prealloc_bias(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    auto conv_out = conv2d_int4_fprop(
        input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(output);
    CHECK_CUDA(bias);
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(
        bias.scalar_type() == torch::kFloat32 || bias.scalar_type() == torch::kFloat16,
        "bias must be float32 or float16"
    );
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat32 || output.scalar_type() == torch::kFloat16,
        "output must be float32 or float16"
    );
    TORCH_CHECK(output.sizes() == conv_out.sizes(), "output shape mismatch for conv2d_int4_fprop_no_ohat_prealloc_bias");
    TORCH_CHECK(bias.numel() == weight_scales.numel(), "bias size must match output channels");

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    if (output.scalar_type() == torch::kFloat16) {
        if (bias.scalar_type() == torch::kFloat16) {
            scale_bias_store_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                num_elements,
                num_channels
            );
        } else {
            scale_bias_store_half_kernel<float><<<grid_size, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                num_elements,
                num_channels
            );
        }
    } else {
        if (bias.scalar_type() == torch::kFloat16) {
            scale_bias_store_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                output.data_ptr<float>(),
                num_elements,
                num_channels
            );
        } else {
            scale_bias_store_kernel<float><<<grid_size, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                output.data_ptr<float>(),
                num_elements,
                num_channels
            );
        }
    }

    return output;
}

// Same as conv2d_int4_fprop_no_ohat_prealloc_bias, but also adds a per-element
// residual (channels_last FP16, same shape as output) in the store epilogue --
// fuses a ResBlock's skip-connection add into the conv. `bias` may be empty.
// FP16 output only. Reuses the shared scale_bias_residual_store_half_kernel.
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales fp32 [K];
//             bias fp32/fp16 [K] or empty; residual fp16 [N,K,H_out,W_out] cl; output fp16 (same
//             shape) PREallocated.
//   Outputs:  output (fp16, in place) = conv*inv_scale*weight_scale[ch] + bias[ch] + residual[i].
//   Fuses:    per-channel weight-scale dequant + optional bias + per-element skip residual.
//   Constraints: output & residual fp16; residual same shape as output; bias (if given) numel()==K.
torch::Tensor conv2d_int4_fprop_no_ohat_prealloc_bias_residual(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    auto conv_out = conv2d_int4_fprop(
        input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(output);
    CHECK_CUDA(residual);
    TORCH_CHECK(residual.is_contiguous(residual.suggest_memory_format()),
                "residual must be contiguous");
    TORCH_CHECK(output.scalar_type() == torch::kFloat16,
                "conv2d_int4_fprop_no_ohat_prealloc_bias_residual: output must be float16");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(output.sizes() == conv_out.sizes(), "output shape mismatch");
    TORCH_CHECK(residual.numel() == conv_out.numel(), "residual size must match output");
    const bool has_bias = bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == weight_scales.numel(), "bias size must match output channels");
    }

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    const __half* res_ptr = reinterpret_cast<const __half*>(residual.data_ptr<at::Half>());
    __half* out_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    if (has_bias && bias.scalar_type() == torch::kFloat16) {
        scale_bias_residual_store_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            res_ptr, out_ptr, num_elements, num_channels);
    } else if (has_bias) {
        scale_bias_residual_store_half_kernel<float><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            bias.data_ptr<float>(), res_ptr, out_ptr, num_elements, num_channels);
    } else {
        scale_bias_residual_store_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            (const __half*)nullptr, res_ptr, out_ptr, num_elements, num_channels);
    }
    return output;
}

// INT4->INT4 chaining conv: dequant + bias + optional ReLU, requantized+packed to the
// NEXT conv's int4 input scale. Lets activations stay int4 across conv1->conv2->conv3
// (the int4 analogue of conv2d_int8_fprop_deepfuse_relu_requant_int8). `output` is the
// packed int4 buffer [N, H_out, W_out, K/2] (int8 dtype, 2 nibbles/byte).
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales FP16 [K]
//             (folded into the deep-fuse epilogue -- note: fp16 here, unlike the no_ohat variants);
//             bias fp32/fp16 [K] or empty; requant_scale fp32 (1-elem = NEXT conv's int4 input scale);
//             output packed-int4 (int8 storage) [N,H_out,W_out,K/2]; apply_relu; config_id; stride/pad/dilation.
//   Outputs:  output (packed int4, in place) = round(clamp((deq+bias[ch])[,relu] * requant_scale, +-7)) nibbles.
//   Fuses:    deep-fuse dequant (fp16 scratch, NO fp32 temp) + optional bias + optional ReLU
//             + requantize + pack to INT4 (int4->int4 chaining).
//   Constraints: output int8-packed contiguous, numel()==scratch/2; weight_scales fp16; requant_scale scalar.
torch::Tensor conv2d_int4_fprop_relu_requant_int4(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor requant_scale,
    torch::Tensor output,
    bool apply_relu,
    int64_t config_id,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Deep-fuse: conv writes fully-scaled fp16 scratch (weight_scale in the CUTLASS
    // epilogue, no fp32 temp); then bias + ReLU + requant + pack read the fp16.
    // `weight_scales` here is the FP16 per-channel scale (the epilogue source).
    int N = input.size(0), H = input.size(1), W = input.size(2);
    int K = weight_packed.size(0), R = weight_packed.size(1), S = weight_packed.size(2);
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    auto scratch = torch::empty({N, K, H_out, W_out},
        input.options().dtype(torch::kFloat16).memory_format(torch::MemoryFormat::ChannelsLast));
    conv2d_int4_dequant_fp16_tuned(input, weight_packed, inv_scale, weight_scales, scratch, config_id,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    CHECK_CUDA(output);
    TORCH_CHECK(output.is_contiguous(), "packed int4 output must be contiguous");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8 (packed int4)");
    TORCH_CHECK(requant_scale.numel() == 1, "requant_scale must be a scalar tensor");
    const bool has_bias = bias.numel() > 0;
    if (has_bias) TORCH_CHECK(bias.numel() == weight_scales.numel(), "bias size mismatch");
    int num_channels = weight_scales.numel();
    int num_packed = scratch.numel() / 2;
    TORCH_CHECK(output.numel() == num_packed, "packed output size mismatch");
    int block_size = 256, grid_size = (num_packed + block_size - 1) / block_size;
    const __half* deq = reinterpret_cast<const __half*>(scratch.data_ptr<at::Half>());
    const float* rq = requant_scale.data_ptr<float>();
    int8_t* out_ptr = reinterpret_cast<int8_t*>(output.data_ptr<int8_t>());
    if (has_bias && bias.scalar_type() == torch::kFloat16) {
        bias_relu_requant_pack_int4_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            deq, reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            rq, out_ptr, num_packed, num_channels, apply_relu);
    } else if (has_bias) {
        bias_relu_requant_pack_int4_from_half_kernel<float><<<grid_size, block_size, 0, stream>>>(
            deq, bias.data_ptr<float>(), rq, out_ptr, num_packed, num_channels, apply_relu);
    } else {
        bias_relu_requant_pack_int4_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            deq, (const __half*)nullptr, rq, out_ptr, num_packed, num_channels, apply_relu);
    }
    return output;
}

// INT4 DUAL-OUTPUT conv3 (block-entry-quantize fusion): dequant + bias + fp16 skip
// residual + ReLU, emitting BOTH the fp16 block output (out_half) AND its requantized
// packed-int4 (out_packed, the next block conv1's input) in one store.
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales FP16 [K]
//             (deep-fuse epilogue source); bias fp32/fp16 [K] or empty; residual fp16 [N,K,H_out,W_out]
//             cl; requant_scale fp32 (1-elem = next block conv1's int4 input scale); out_half fp16 (same
//             shape); out_packed packed-int4 (int8 storage) [N,H_out,W_out,K/2]; apply_relu; config_id;
//             stride/pad/dilation.
//   Outputs:  DUAL (both in place): out_half (fp16, returned) = (deq+bias[ch]+residual)[,relu];
//             out_packed = round(clamp(out_half * requant_scale, +-7)) packed int4 nibbles.
//   Fuses:    deep-fuse dequant (NO fp32 temp) + optional bias + skip residual + optional ReLU
//             + block-entry requantize + pack to INT4, in one store.
//   Constraints: out_half & residual fp16; out_packed int8-packed contiguous; requant_scale scalar;
//             weight_scales fp16.
torch::Tensor conv2d_int4_fprop_bias_residual_dual(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor requant_scale,
    torch::Tensor out_half,
    torch::Tensor out_packed,
    bool apply_relu,
    int64_t config_id,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Deep-fuse conv -> fully-scaled fp16 scratch (no fp32 temp); then bias + residual
    // + ReLU + dual (fp16 + packed int4) store read the fp16. `weight_scales` is the
    // FP16 per-channel scale (the CUTLASS epilogue source).
    int N = input.size(0), H = input.size(1), W = input.size(2);
    int K = weight_packed.size(0), R = weight_packed.size(1), S = weight_packed.size(2);
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    auto scratch = torch::empty({N, K, H_out, W_out},
        input.options().dtype(torch::kFloat16).memory_format(torch::MemoryFormat::ChannelsLast));
    conv2d_int4_dequant_fp16_tuned(input, weight_packed, inv_scale, weight_scales, scratch, config_id,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    CHECK_CUDA(out_half); CHECK_CONTIGUOUS(out_half);
    CHECK_CUDA(out_packed);
    TORCH_CHECK(out_packed.is_contiguous(), "packed int4 output must be contiguous");
    CHECK_CUDA(residual);
    TORCH_CHECK(out_half.scalar_type() == torch::kFloat16, "out_half must be float16");
    TORCH_CHECK(out_packed.scalar_type() == torch::kInt8, "out_packed must be int8 (packed int4)");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(out_half.sizes() == scratch.sizes(), "out_half shape mismatch");
    TORCH_CHECK(requant_scale.numel() == 1, "requant_scale must be a scalar tensor");
    const bool has_bias = bias.numel() > 0;
    if (has_bias) TORCH_CHECK(bias.numel() == weight_scales.numel(), "bias size mismatch");
    int num_channels = weight_scales.numel();
    int num_packed = scratch.numel() / 2;
    TORCH_CHECK(out_packed.numel() == num_packed, "packed output size mismatch");
    int block_size = 256, grid_size = (num_packed + block_size - 1) / block_size;
    const __half* deq = reinterpret_cast<const __half*>(scratch.data_ptr<at::Half>());
    const __half* res_ptr = reinterpret_cast<const __half*>(residual.data_ptr<at::Half>());
    const float* rq = requant_scale.data_ptr<float>();
    __half* outh = reinterpret_cast<__half*>(out_half.data_ptr<at::Half>());
    int8_t* outp = reinterpret_cast<int8_t*>(out_packed.data_ptr<int8_t>());
    if (has_bias && bias.scalar_type() == torch::kFloat16) {
        bias_residual_relu_dual_store_pack_int4_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            deq, reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            res_ptr, rq, outh, outp, num_packed, num_channels, apply_relu);
    } else if (has_bias) {
        bias_residual_relu_dual_store_pack_int4_from_half_kernel<float><<<grid_size, block_size, 0, stream>>>(
            deq, bias.data_ptr<float>(), res_ptr, rq, outh, outp, num_packed, num_channels, apply_relu);
    } else {
        bias_residual_relu_dual_store_pack_int4_from_half_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            deq, (const __half*)nullptr, res_ptr, rq, outh, outp, num_packed, num_channels, apply_relu);
    }
    return out_half;
}

// Same as conv2d_int4_fprop_no_ohat_prealloc, but allocates its own output buffer.
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales fp32 [K];
//             stride/pad/dilation.
//   Outputs:  fresh fp32 [N,K,H_out,W_out] channels_last = int32_accum * inv_scale * weight_scale[ch].
//   Fuses:    per-channel weight-scale dequant (allocates output, then delegates to *_no_ohat_prealloc).
//   Constraints: none beyond the base conv's tile requirements.
torch::Tensor conv2d_int4_fprop_no_ohat(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int K = weight_packed.size(0);
    int R = weight_packed.size(1);
    int S = weight_packed.size(2);
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto output = torch::empty(
        {N, K, H_out, W_out},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast)
    );

    return conv2d_int4_fprop_no_ohat_prealloc(
        input,
        weight_packed,
        inv_scale,
        weight_scales,
        output,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w
    );
}

// conv2d_int4_fprop, then o_hat_cache += raw_output * weight_scales[channel].
//   Inputs:   input packed-int4; weight_packed; inv_scale fp32 (1-elem alpha); weight_scales fp32 [K]
//             (per-channel); o_hat_cache fp16 or fp32 [N,K,H_out,W_out] cl (in-place accumulate target);
//             stride/pad/dilation.
//   Outputs:  o_hat_cache (returned, in place) += int32_accum * inv_scale * weight_scale[ch].
//   Fuses:    per-channel weight-scale dequant + MoDiff o_hat cache accumulate (separate
//             scale_accumulate[_half] kernel after the raw fprop).
//   Constraints: o_hat_cache fp16 or fp32; weight_scales.numel()==K.
torch::Tensor conv2d_int4_fprop_o_hat(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,      // scalar tensor for CUTLASS alpha
    torch::Tensor weight_scales,  // per-channel vector
    torch::Tensor o_hat_cache,    // in-place output accumulate target
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));

    auto conv_out = conv2d_int4_fprop(
        input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    if (o_hat_cache.scalar_type() == torch::kFloat16) {
        if (num_channels % 2 == 0) {
            int grid_size_vec2 = (((num_elements + 1) / 2) + block_size - 1) / block_size;
            scale_accumulate_half_cache_vec2_kernel<<<grid_size_vec2, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                reinterpret_cast<__half*>(o_hat_cache.data_ptr<at::Half>()),
                num_elements,
                num_channels
            );
        } else {
            scale_accumulate_half_cache_kernel<<<grid_size, block_size, 0, stream>>>(
                conv_out.data_ptr<float>(),
                weight_scales.data_ptr<float>(),
                reinterpret_cast<__half*>(o_hat_cache.data_ptr<at::Half>()),
                num_elements,
                num_channels
            );
        }
    } else {
        scale_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            o_hat_cache.data_ptr<float>(),
            num_elements,
            num_channels
        );
    }

    return o_hat_cache;
}

// int4 counterpart of conv2d_int8_fprop_o_hat_residual: o_hat_cache += conv*ws[ch]
// (byte-identical cache write) AND output = (updated o_hat) + residual, fusing the
// ResBlock skip-add. o_hat_cache/residual/output fp16 channels_last. Returns output.
torch::Tensor conv2d_int4_fprop_o_hat_residual(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor o_hat_cache,
    torch::Tensor residual,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    auto conv_out = conv2d_int4_fprop(
        input, weight_packed, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    TORCH_CHECK(o_hat_cache.scalar_type() == torch::kFloat16,
                "conv2d_int4_fprop_o_hat_residual: o_hat_cache must be float16");
    CHECK_CUDA(residual); CHECK_CUDA(output);
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16 && output.scalar_type() == torch::kFloat16,
                "conv2d_int4_fprop_o_hat_residual: residual/output must be float16");
    TORCH_CHECK(residual.numel() == conv_out.numel() && output.numel() == conv_out.numel(),
                "conv2d_int4_fprop_o_hat_residual: residual/output size must match conv output");

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    if (num_channels % 2 == 0) {
        int grid_size_vec2 = (((num_elements + 1) / 2) + block_size - 1) / block_size;
        scale_accumulate_residual_half_cache_vec2_kernel<<<grid_size_vec2, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            reinterpret_cast<__half*>(o_hat_cache.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_elements,
            num_channels
        );
    } else {
        scale_accumulate_residual_half_cache_kernel<<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            reinterpret_cast<__half*>(o_hat_cache.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_elements,
            num_channels
        );
    }
    return output;
}
