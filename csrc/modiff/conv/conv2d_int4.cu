// ============================================================================================
// MoDiff int4 conv: the o_hat delta-cache entry points. Baseline twin: csrc/baseline/conv/conv2d_int4.cu
//
//   o_hat_cache += conv(input, weight) * weight_scale[ch]      (in place, Eq 9)
//
// Family 4 of the csrc/ datapath split (2026-08-12). Same shape as the int8 twin: the CUTLASS int4
// conv Op instantiation and conv2d_int4_fprop are COPIED (conv2d_int4_fprop is also called by the
// baseline *_no_ohat* / *_relu_requant_int4 entry points), the copy here is `static`.

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

#include "../common/common.cuh"
#include "conv_epilogue.cuh"

// ---- COPY of CUTLASS int4 conv type block ----
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

// ---- COPY of conv2d_int4_fprop ----
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
static torch::Tensor conv2d_int4_fprop(
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

// ==== MoDiff o_hat entry points (moved) ====

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
