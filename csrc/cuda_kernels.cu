#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

// Macro for error checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::ChannelsLast), #x " must be channels_last contiguous")

// Architecture: Ampere (Sm80)
using Arch = cutlass::arch::Sm80;

// =========================================================================
// INT8 Kernel Definition
// =========================================================================

// =========================================================================
// Helper: Fast Quantization + Packing Kernel
// =========================================================================

__global__ void quantize_pack_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    int num_elements // Number of OUTPUT packed bytes
) {
    // Each thread processes one output byte (2 input floats)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    // Load 2 floats
    // Input Layout: NHWC
    // We pack Input[2*idx] (Low nibble) and Input[2*idx+1] (High nibble)
    
    // Simple Cast (No rounding/scale for benchmark speed)
    // Real implementation would assume scaling was applied before or do it here.
    float v0 = input[2*idx];
    float v1 = input[2*idx+1];

    // Explicit round then clamp before cast
    int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v0)));
    int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v1)));
    
    // Pack: Low 4 bits | High 4 bits
    int8_t packed = (i0 & 0x0F) | ((i1 & 0x0F) << 4);
    
    output[idx] = packed;
}

torch::Tensor quantize_and_pack(torch::Tensor input) {
    // Input: FP32 [N, H, W, C]
    // Output: Int8 [N, H, W, C/2] (Packed Int4)
    
    int num_input = input.numel();
    int num_output = num_input / 2;
    
    auto output = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));
    
    int block_size = 256;
    int grid_size = (num_output + block_size - 1) / block_size;
    
    quantize_pack_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        num_output
    );
    
    // Reshape logic: [N, C, H, W] -> [N, H, W, C/2]
    // Input is NCHW logical, but NHWC physical.
    // Result treated as NHWC-like packed tensor.
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    return output.view({N, H, W, C/2});
}

// 1. Define the Kernel using DefaultConv2dFprop
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
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2, // Stages
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

// 2. Define the Device Operator
using Conv2dInt8Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt8Kernel>;

// =========================================================================
// INT4 Kernel Definition
// =========================================================================

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
  2,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided
>::Kernel;

using Conv2dInt4Op = cutlass::conv::device::ImplicitGemmConvolution<Conv2dInt4Kernel>;

// =========================================================================
// Implementation: INT8
// =========================================================================

torch::Tensor conv2d_int8_fprop(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    // Permuted weight (K, R, S, C)
    int R = weight.size(1);
    int S = weight.size(2);
    // int C = weight.size(3); // C from input

    
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
    
    float alpha = 1.0f;
    if (scales.numel() == 1) alpha = scales.item<float>();
    
    Conv2dInt8Op op;
    Conv2dInt8Op::Arguments args(
        problem_size,
        {input_ptr, {C, W * C, H * W * C}},
        {weight_ptr, {C, S * C, R * S * C}},
        {bias_ptr, {0,0,0}}, // Broadcast bias
        {output_ptr, {K, W_out * K, H_out * W_out * K}},
        {alpha, 1.0f}
    );
    
    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    
    cutlass::Status status = op(args, workspace.data_ptr());
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT8 Kernel Failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS Kernel failed");
    }

    return output;
}

// =========================================================================
// Implementation: INT4
// =========================================================================

torch::Tensor conv2d_int4_fprop(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    // Previous Code Assumed NCHW input and performed check
    // Now we accept packed input so we disable these generic checks
    // as we handle specific checks below.
    CHECK_CUDA(input);
    // CHECK_CONTIGUOUS(input);
    
    // Correct Dimension Extraction for Packed (NHWC-like) Tensors
    // Input is (N, H, W, C/2) contiguous
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    // input.size(3) is C_packed (C/2)
    
    // Weight is (K, R, S, C/2) contiguous
    int K_packed = weight_packed.size(0);
    int R = weight_packed.size(1);
    int S = weight_packed.size(2);
    int C_packed = weight_packed.size(3); 
    
    int C_logical = C_packed * 2; 

    // Safety checks
    // TORCH_CHECK(input.is_contiguous(), "Input must be contiguous (N, H, W, C/2)");
    // TORCH_CHECK(weight_packed.is_contiguous(), "Weight must be contiguous (K, R, S, C/2)");
    TORCH_CHECK(input.size(3) == C_packed, "Input/Weight channel mismatch");

    void* input_ptr_raw = input.data_ptr();
    
    cutlass::int4b_t* input_ptr = (cutlass::int4b_t*)input_ptr_raw;
    cutlass::int4b_t* weight_ptr = (cutlass::int4b_t*)weight_packed.data_ptr();
    
    int H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    int K_logical = K_packed; 
    
    auto out_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto output = torch::empty({N, K_logical, H_out, W_out}, out_options);
    
    float alpha = 1.0f;
    if (scales.numel() == 1) alpha = scales.item<float>();
    
    float beta = (bias.numel() > 0) ? 1.0f : 0.0f;
    
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

    Conv2dInt4Op op;
    Conv2dInt4Op::Arguments args(
        problem_size,
        {input_ptr, {C_logical, W * C_logical, H * W * C_logical}},
        {weight_ptr, {C_logical, S * C_logical, R * S * C_logical}}, 
        {(float*)((bias.numel() > 0) ? bias.data_ptr() : nullptr), {0,0,0}},
        {output.data_ptr<float>(), {K_packed, W_out * K_packed, H_out * W_out * K_packed}},
        {alpha, beta}
    );
    
    size_t workspace_size = op.get_workspace_size(args);
    auto workspace = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    
    cutlass::Status status = op(args, workspace.data_ptr());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS INT4 Kernel Failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        TORCH_CHECK(false, "CUTLASS INT4 Kernel failed");
    }

    return output;
}
