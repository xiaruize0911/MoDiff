// =========================================================================
// CUTLASS INT4 Conv2d integration.
//
// Same structure as conv2d_int8.cu, but both activations and weights are
// packed 2-per-byte INT4 values (see quantize.cu / modiff_delta_quantize.cu
// for the packing kernels). Inputs are physically (N, H, W, C/2) and weights
// (K, R, S, C/2); CUTLASS is told the *logical* (unpacked) channel count via
// the problem size so its INT4 tensor-core path can address individual nibbles.
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

// Base CUTLASS INT4 conv. `scales` is the *scalar* activation dequant factor,
// same convention as conv2d_int8_fprop — per-channel weight dequant always
// happens separately (scale_accumulate/scale_store, or a Python-side multiply).
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

int64_t conv2d_int4_num_tuned_configs() { return 6; }

// Run tile `config_id` of the int4 conv -> fresh FP32 [N,K,H_out,W_out] channels_last
// output (= raw_int32 * inv_scale). config_id < 0 uses the default fixed-tile
// conv2d_int4_fprop. The autotuner in Python tries each and skips ones that can't
// implement the shape.
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

// conv2d_int4_fprop, then output = raw_output * weight_scales[channel], written
// into a caller-provided buffer (see conv2d_int8_fprop_no_ohat_prealloc for the
// same "prealloc only covers the final buffer" caveat).
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

// Same as conv2d_int4_fprop_no_ohat_prealloc, but also adds a per-channel bias.
// Mirrors conv2d_int8.cu's conv2d_int8_fprop_no_ohat_prealloc_bias: the
// scale_bias_store_kernel / scale_bias_store_half_kernel epilogue templates in
// conv_epilogue.cuh are shared between the INT8 and INT4 paths, so no new
// kernel body is needed here, only this wrapper.
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
    auto conv_out = conv2d_int4_fprop_tuned(
        input, weight_packed, inv_scale, config_id,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    CHECK_CUDA(output);
    TORCH_CHECK(output.is_contiguous(), "packed int4 output must be contiguous");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8 (packed int4)");
    TORCH_CHECK(requant_scale.numel() == 1, "requant_scale must be a scalar tensor");
    const bool has_bias = bias.numel() > 0;
    if (has_bias) TORCH_CHECK(bias.numel() == weight_scales.numel(), "bias size mismatch");
    int num_channels = weight_scales.numel();
    int num_packed = conv_out.numel() / 2;
    TORCH_CHECK(output.numel() == num_packed, "packed output size mismatch");
    int block_size = 256, grid_size = (num_packed + block_size - 1) / block_size;
    const float* rq = requant_scale.data_ptr<float>();
    int8_t* out_ptr = reinterpret_cast<int8_t*>(output.data_ptr<int8_t>());
    if (has_bias && bias.scalar_type() == torch::kFloat16) {
        scale_bias_relu_requant_pack_int4_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            rq, out_ptr, num_packed, num_channels, apply_relu);
    } else if (has_bias) {
        scale_bias_relu_requant_pack_int4_kernel<float><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            bias.data_ptr<float>(), rq, out_ptr, num_packed, num_channels, apply_relu);
    } else {
        scale_bias_relu_requant_pack_int4_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            (const __half*)nullptr, rq, out_ptr, num_packed, num_channels, apply_relu);
    }
    return output;
}

// INT4 DUAL-OUTPUT conv3 (block-entry-quantize fusion): dequant + bias + fp16 skip
// residual + ReLU, emitting BOTH the fp16 block output (out_half) AND its requantized
// packed-int4 (out_packed, the next block conv1's input) in one store.
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
    auto conv_out = conv2d_int4_fprop_tuned(
        input, weight_packed, inv_scale, config_id,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    CHECK_CUDA(out_half); CHECK_CONTIGUOUS(out_half);
    CHECK_CUDA(out_packed);
    TORCH_CHECK(out_packed.is_contiguous(), "packed int4 output must be contiguous");
    CHECK_CUDA(residual);
    TORCH_CHECK(out_half.scalar_type() == torch::kFloat16, "out_half must be float16");
    TORCH_CHECK(out_packed.scalar_type() == torch::kInt8, "out_packed must be int8 (packed int4)");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(out_half.sizes() == conv_out.sizes(), "out_half shape mismatch");
    TORCH_CHECK(requant_scale.numel() == 1, "requant_scale must be a scalar tensor");
    const bool has_bias = bias.numel() > 0;
    if (has_bias) TORCH_CHECK(bias.numel() == weight_scales.numel(), "bias size mismatch");
    int num_channels = weight_scales.numel();
    int num_packed = conv_out.numel() / 2;
    TORCH_CHECK(out_packed.numel() == num_packed, "packed output size mismatch");
    int block_size = 256, grid_size = (num_packed + block_size - 1) / block_size;
    const __half* res_ptr = reinterpret_cast<const __half*>(residual.data_ptr<at::Half>());
    const float* rq = requant_scale.data_ptr<float>();
    __half* outh = reinterpret_cast<__half*>(out_half.data_ptr<at::Half>());
    int8_t* outp = reinterpret_cast<int8_t*>(out_packed.data_ptr<int8_t>());
    if (has_bias && bias.scalar_type() == torch::kFloat16) {
        bias_residual_relu_dual_store_pack_int4_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            res_ptr, rq, outh, outp, num_packed, num_channels, apply_relu);
    } else if (has_bias) {
        bias_residual_relu_dual_store_pack_int4_kernel<float><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            bias.data_ptr<float>(), res_ptr, rq, outh, outp, num_packed, num_channels, apply_relu);
    } else {
        bias_residual_relu_dual_store_pack_int4_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            conv_out.data_ptr<float>(), weight_scales.data_ptr<float>(),
            (const __half*)nullptr, res_ptr, rq, outh, outp, num_packed, num_channels, apply_relu);
    }
    return out_half;
}

// Same as conv2d_int4_fprop_no_ohat_prealloc, but allocates its own output buffer.
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

    return o_hat_cache;
}
