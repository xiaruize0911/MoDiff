// =========================================================================
// CUTLASS INT8 Conv2d integration.
//
// All convolutions here run on NHWC-physical (channels_last) INT8 activations
// and INT8 weights stored in KRSC order, accumulating in int32 internally.
// There are three output "shapes" of the same underlying matmul, layered by
// how much of the dequant/postprocess work is fused into the CUTLASS call:
//
//   conv2d_int8_fprop                       -> raw CUTLASS call, output scaled
//                                               by a single scalar `scales`
//                                               (see note on that param below)
//   conv2d_int8_fprop_dequant_fp16_prealloc -> deepest fusion: CUTLASS epilogue
//                                               itself applies the per-channel
//                                               weight scale and writes FP16
//                                               directly (no FP32 temporary)
//   conv2d_int8_fprop_*_no_ohat/_o_hat      -> conv2d_int8_fprop followed by a
//                                               separate scale_store/scale_accumulate
//                                               kernel (see conv_epilogue.cu) that
//                                               applies the per-channel weight scale
// =========================================================================

#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

#include "../common.cuh"
#include "conv_epilogue.cuh"

// Architecture: Ampere (Sm80)
using Arch = cutlass::arch::Sm80;

// ---- CUTLASS kernel/op type definitions ----

using Conv2dInt8Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  int8_t, cutlass::layout::TensorNHWC,
  int8_t, cutlass::layout::TensorNHWC,
  float, cutlass::layout::TensorNHWC,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  Arch,
  cutlass::gemm::GemmShape<128, 128, 128>,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<16, 8, 32>,
  cutlass::epilogue::thread::LinearCombination<
    float,
    1, // ElementCount for vector load from Accum/C
    int32_t,
    float,
    cutlass::epilogue::thread::ScaleType::Default
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3, // Stages (was 2, 3 improves memory latency pipelining)
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

using Conv2dInt8Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt8Kernel>;

// Custom epilogue for the "deep fuse" path: dequantizes the int32 accumulator
// directly to FP16 as `accumulator * alpha * source`, where `alpha` is the
// activation's inverse quant scale and `source` is the per-output-channel
// weight scale (broadcast as the epilogue's "C" operand). This replaces a
// separate scale_store_half_kernel launch (see conv2d_int8_fprop_no_ohat_prealloc)
// with a single fused CUTLASS call that never materializes an FP32 temporary.
template <int Count>
class Int8DequantScaleSource {
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

    CUTLASS_HOST_DEVICE
    Params(): alpha(ElementCompute(1)), alpha_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha): alpha(alpha), alpha_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr): alpha(ElementCompute(1)), alpha_ptr(alpha_ptr) {}
  };

private:
  ElementCompute alpha_;

public:
  CUTLASS_HOST_DEVICE
  explicit Int8DequantScaleSource(Params const &params) {
    alpha_ = params.alpha_ptr ? *params.alpha_ptr : params.alpha;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source) const {
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      float value = float(accumulator[i]) * alpha_ * float(source[i]);
      output[i] = ElementOutput(value);
    }
    return output;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      output[i] = ElementOutput(float(accumulator[i]) * alpha_);
    }
    return output;
  }
};

using Conv2dInt8DequantFp16Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  int8_t, cutlass::layout::TensorNHWC,
  int8_t, cutlass::layout::TensorNHWC,
  cutlass::half_t, cutlass::layout::TensorNHWC,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  Arch,
  cutlass::gemm::GemmShape<128, 128, 128>,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<16, 8, 32>,
  Int8DequantScaleSource<8>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

using Conv2dInt8DequantFp16Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt8DequantFp16Kernel>;

// ---- Host functions ----

// Base CUTLASS INT8 conv. `scales` is the *scalar* activation dequant factor
// (CUTLASS epilogue alpha) — every call site in this project passes a
// 1-element tensor here; per-channel weight dequant always happens in a
// separate step (scale_accumulate/scale_store, or a Python-side multiply by
// weight_scale_channel). `bias`, if non-empty, is added by the CUTLASS
// epilogue itself (broadcast per output channel).
torch::Tensor conv2d_int8_fprop(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    // Weight is permuted to (K, R, S, C) physical/logical layout.
    int R = weight.size(1);
    int S = weight.size(2);

    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto output = torch::empty({N, K, H_out, W_out}, options);

    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C},
        {K, R, S, C},
        {padding_h, padding_h, padding_w, padding_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {N, H_out, W_out, K},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );

    int8_t* input_ptr = reinterpret_cast<int8_t*>(input.data_ptr());
    int8_t* weight_ptr = reinterpret_cast<int8_t*>(weight.data_ptr());
    float* output_ptr = output.data_ptr<float>();
    float* bias_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    // Build epilogue params: use device pointer when scales is on GPU to avoid D2H sync
    using Int8EpilogueParams = typename Conv2dInt8Op::EpilogueOutputOp::Params;
    Int8EpilogueParams ep;
    bool use_device_alpha = (scales.numel() == 1 && scales.is_cuda() && bias.numel() == 0);
    if (use_device_alpha) {
        // Device pointer path: CUTLASS reads alpha from GPU memory, beta=0 (no bias)
        ep = Int8EpilogueParams(scales.data_ptr<float>());
    } else {
        float alpha = 1.0f;
        if (scales.numel() == 1) alpha = scales.item<float>();
        float beta = (bias.numel() > 0) ? 1.0f : 0.0f;
        ep = Int8EpilogueParams(alpha, beta);
    }

    Conv2dInt8Op op;
    Conv2dInt8Op::Arguments args(
        problem_size,
        {input_ptr, {C, W * C, H * W * C}},
        {weight_ptr, {C, S * C, R * S * C}},
        {bias_ptr, {0,0,0}}, // Broadcast bias
        {output_ptr, {K, W_out * K, H_out * W_out * K}},
        ep
    );

    // Fail with a clear, synchronous error for problem sizes this fixed tile
    // configuration can't handle (e.g. C or K far smaller than the 128x128x128
    // threadblock tile), instead of letting a bad launch corrupt the CUDA
    // stream asynchronously and surface as a confusing error on some later,
    // unrelated call.
    cutlass::Status can_status = Conv2dInt8Op::can_implement(args);
    if (can_status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT8 Kernel cannot implement this problem size: "
                  << cutlass::cutlassGetStatusString(can_status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT8 Kernel cannot implement this problem size (N=", N,
                    " C=", C, " H=", H, " W=", W, " K=", K, " R=", R, " S=", S, ")");
    }

    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));

    cutlass::Status status = op(args, workspace.data_ptr(), stream);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT8 Kernel Failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS Kernel failed");
    }

    return output;
}

// "Deep fuse" path: CUTLASS epilogue itself computes
// output = int32_accum * inv_scale * weight_scales_half[channel], writing FP16
// directly. Avoids the FP32 temporary + separate scale_store_half_kernel that
// conv2d_int8_fprop_no_ohat_prealloc needs.
torch::Tensor conv2d_int8_fprop_dequant_fp16_prealloc(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales_half,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(output);
    CHECK_CUDA(weight_scales_half);
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be int8");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be float16");
    TORCH_CHECK(weight_scales_half.scalar_type() == torch::kFloat16, "weight scales must be float16");
    TORCH_CHECK(weight_scales_half.is_contiguous(), "weight scales must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = weight.size(1);
    int S = weight.size(2);

    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    TORCH_CHECK(output.size(0) == N && output.size(1) == K && output.size(2) == H_out && output.size(3) == W_out,
                "output shape mismatch for conv2d_int8_fprop_dequant_fp16_prealloc");
    TORCH_CHECK(weight_scales_half.numel() == K, "weight scales size must match output channels");

    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C},
        {K, R, S, C},
        {padding_h, padding_h, padding_w, padding_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {N, H_out, W_out, K},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );

    int8_t* input_ptr = reinterpret_cast<int8_t*>(input.data_ptr());
    int8_t* weight_ptr = reinterpret_cast<int8_t*>(weight.data_ptr());
    auto* scale_ptr = reinterpret_cast<cutlass::half_t*>(weight_scales_half.data_ptr<at::Half>());
    auto* output_ptr = reinterpret_cast<cutlass::half_t*>(output.data_ptr<at::Half>());

    using DequantEpilogueParams = typename Conv2dInt8DequantFp16Op::EpilogueOutputOp::Params;
    DequantEpilogueParams ep(inv_scale.data_ptr<float>());

    Conv2dInt8DequantFp16Op op;
    Conv2dInt8DequantFp16Op::Arguments args(
        problem_size,
        {input_ptr, {C, W * C, H * W * C}},
        {weight_ptr, {C, S * C, R * S * C}},
        {scale_ptr, {0, 0, 0}},
        {output_ptr, {K, W_out * K, H_out * W_out * K}},
        ep
    );

    cutlass::Status can_status = Conv2dInt8DequantFp16Op::can_implement(args);
    if (can_status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT8 dequant FP16 Kernel cannot implement: "
                  << cutlass::cutlassGetStatusString(can_status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT8 dequant FP16 Kernel cannot implement");
    }

    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));

    cutlass::Status status = op(args, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT8 dequant FP16 Kernel Failed: "
                  << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT8 dequant FP16 Kernel failed");
    }

    return output;
}

// conv2d_int8_fprop, then o_hat_cache += raw_output * weight_scales[channel].
torch::Tensor conv2d_int8_fprop_o_hat(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,      // scalar tensor for CUTLASS alpha
    torch::Tensor weight_scales,  // per-channel vector
    torch::Tensor o_hat_cache,    // in-place output accumulate target
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));

    auto conv_out = conv2d_int8_fprop(
        input, weight, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    // FP16 cache support cuts resident MoDiff cache memory/bandwidth while
    // preserving the existing FP32 path for dynamic/un-calibrated runs.
    if (o_hat_cache.scalar_type() == torch::kFloat16) {
        scale_accumulate_half_cache_kernel<<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            reinterpret_cast<__half*>(o_hat_cache.data_ptr<at::Half>()),
            num_elements,
            num_channels
        );
    } else {
        scale_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            o_hat_cache.data_ptr<float>(),
            num_elements,
            num_channels
        );
    }

    // Return o_hat_cache itself for identical graph tracking
    return o_hat_cache;
}

// conv2d_int8_fprop, then output = raw_output * weight_scales[channel], written
// into a caller-provided buffer (still allocates its own intermediate raw
// conv output + CUTLASS workspace internally — "prealloc" only avoids
// allocating the *final* dequantized output tensor).
torch::Tensor conv2d_int8_fprop_no_ohat_prealloc(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto empty_bias = torch::empty({0}, torch::TensorOptions().device(input.device()));
    auto conv_out = conv2d_int8_fprop(
        input, weight, inv_scale, empty_bias,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    );

    CHECK_CUDA(output);
    CHECK_CONTIGUOUS(output);
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat32 || output.scalar_type() == torch::kFloat16,
        "output must be float32 or float16"
    );
    TORCH_CHECK(output.sizes() == conv_out.sizes(), "output shape mismatch for conv2d_int8_fprop_no_ohat_prealloc");

    int num_elements = conv_out.numel();
    int num_channels = weight_scales.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    if (output.scalar_type() == torch::kFloat16) {
        scale_store_half_kernel<<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_elements,
            num_channels
        );
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

// Same as conv2d_int8_fprop_no_ohat_prealloc, but also adds a per-channel bias.
torch::Tensor conv2d_int8_fprop_no_ohat_prealloc_bias(
    torch::Tensor input,
    torch::Tensor weight,
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
    auto conv_out = conv2d_int8_fprop(
        input, weight, inv_scale, empty_bias,
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
    TORCH_CHECK(output.sizes() == conv_out.sizes(), "output shape mismatch for conv2d_int8_fprop_no_ohat_prealloc_bias");
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

// Same as conv2d_int8_fprop_no_ohat_prealloc, but allocates its own output buffer.
torch::Tensor conv2d_int8_fprop_no_ohat(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = weight.size(1);
    int S = weight.size(2);
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto output = torch::empty(
        {N, K, H_out, W_out},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast)
    );

    return conv2d_int8_fprop_no_ohat_prealloc(
        input,
        weight,
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
