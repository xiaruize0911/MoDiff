// ============================================================================================
// MoDiff int8 conv: the o_hat delta-cache entry points. Baseline twin: csrc/baseline/conv/conv2d_int8.cu
//
//   o_hat_cache += conv(input, weight) * weight_scale[ch]      (in place, Eq 9)
//
// Family 4 of the csrc/ datapath split (2026-08-12). The CUTLASS int8 conv Op instantiation and
// conv2d_int8_fprop are COPIED, not shared: conv2d_int8_fprop is also called by four baseline
// entry points (*_no_ohat_prealloc*, *_relu_requant_int8), so it stays there too, and the copy
// here is `static`. Duplicating a CUTLASS Op instantiation is the compile-time cost this family
// was expected to carry -- measured in csrc/README.md.

// KEEP THE COPIES IDENTICAL to their baseline twins -- every A/B in docs/ compares the two
// datapaths, so a numerical edit here and not there invalidates them. See csrc/README.md.
// ============================================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

// Explicit relative paths, NOT bare: a bare include resolves through the global -I csrc and would
// pick a different tree's copy. See csrc/README.md.
#include "../common/common.cuh"
#include "conv_epilogue.cuh"

// ---- COPY of CUTLASS int8 conv type block ----
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

// ---- COPY of conv2d_int8_fprop ----
// Base CUTLASS INT8 conv. `scales` is the *scalar* activation dequant factor
// (CUTLASS epilogue alpha) — every call site in this project passes a
// 1-element tensor here; per-channel weight dequant always happens in a
// separate step (scale_accumulate/scale_store, or a Python-side multiply by
// weight_scale_channel). `bias`, if non-empty, is added by the CUTLASS
// epilogue itself (broadcast per output channel).
//   Inputs:   input int8 [N,C,H,W] channels_last (NHWC physical); weight int8 [K,R,S,C]
//             (KRSC); scales fp32 (1-elem = CUTLASS alpha = activation dequant); bias fp32
//             [K] or empty; stride/pad/dilation ints.
//   Outputs:  fp32 [N,K,H_out,W_out] channels_last = int32_accum * alpha (+ bias broadcast).
//   Fuses:    scalar-alpha dequant + optional bias (CUTLASS epilogue). NOT per-channel weight scale.
//   Constraints: Sm80; fixed 128x128x128 tile (C/K must be large enough -- can_implement guards);
//             device-alpha (no D2H sync) only when scales is 1-elem CUDA and bias is empty.
static torch::Tensor conv2d_int8_fprop(
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

// ==== MoDiff o_hat entry points (moved) ====

// conv2d_int8_fprop, then o_hat_cache += raw_output * weight_scales[channel].
//   Inputs:   input int8 [N,C,H,W] cl; weight int8 [K,R,S,C]; inv_scale fp32 (1-elem alpha);
//             weight_scales fp32 [K] (per-channel); o_hat_cache fp16 or fp32 [N,K,H_out,W_out]
//             cl (in-place accumulate target); stride/pad/dilation.
//   Outputs:  o_hat_cache (returned, in place) += int32_accum * inv_scale * weight_scale[ch].
//   Fuses:    per-channel weight-scale dequant + MoDiff o_hat cache accumulate (separate
//             scale_accumulate[_half] kernel after the raw fprop).
//   Constraints: o_hat_cache fp16 or fp32; weight_scales.numel()==K.
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
        // Vec2 fast path requires num_channels % 2 == 0 (see conv_epilogue.cu's header
        // comment on the channel-boundary hazard -- same reasoning as the float4 gate,
        // one step down in width). Right-size the grid for the 2-wide step.
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

    // Return o_hat_cache itself for identical graph tracking
    return o_hat_cache;
}

// Same as conv2d_int8_fprop_o_hat (o_hat_cache += conv*weight_scale[ch]), but ALSO
// writes `output` = (updated o_hat_cache) + residual, fusing the ResBlock skip-add
// into the accumulate pass (removes the trailing aten::add on the modiff path).
// o_hat_cache must be fp16; residual/output fp16 channels_last matching the conv
// output. The cache write is byte-identical to conv2d_int8_fprop_o_hat, so temporal
// evolution is unchanged; only `output` carries the residual. Returns `output`.
//   Constraints: o_hat_cache/residual/output fp16; residual & output shape == conv out.
torch::Tensor conv2d_int8_fprop_o_hat_residual(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor o_hat_cache,    // in-place accumulate target (fp16)
    torch::Tensor residual,       // fp16 channels_last, per-element skip
    torch::Tensor output,         // fp16 channels_last, preallocated
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

    TORCH_CHECK(o_hat_cache.scalar_type() == torch::kFloat16,
                "conv2d_int8_fprop_o_hat_residual: o_hat_cache must be float16");
    CHECK_CUDA(residual); CHECK_CUDA(output);
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16 && output.scalar_type() == torch::kFloat16,
                "conv2d_int8_fprop_o_hat_residual: residual/output must be float16");
    TORCH_CHECK(residual.numel() == conv_out.numel() && output.numel() == conv_out.numel(),
                "conv2d_int8_fprop_o_hat_residual: residual/output size must match conv output");

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
