/*
 * INT8 Convolution using CUTLASS Implicit GEMM
 * Uses Tensor Core acceleration for INT8 (dp4a/mma.sync)
 * 
 * CUTLASS provides highly optimized implicit GEMM kernels that fuse
 * im2col with GEMM for better memory efficiency.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// For sm_89 (Ada Lovelace / L4), use the appropriate architecture tag
using SmArch = cutlass::arch::Sm89;

// ============================================================================
// CUTLASS INT8 Conv2d Configuration
// ============================================================================

// Define the INT8 Tensor Core convolution operation
// Input: int8_t, Weight: int8_t, Output: int32_t (then dequantized to float)
using ElementA = int8_t;      // Input
using ElementB = int8_t;      // Weight
using ElementC = int32_t;     // Accumulator
using ElementD = int32_t;     // Output (before dequantization)

using LayoutA = cutlass::layout::TensorNHWC;  // NHWC for input
using LayoutB = cutlass::layout::TensorNHWC;  // NHWC for weight (KRSC)
using LayoutC = cutlass::layout::TensorNHWC;  // NHWC for output

// Thread block tile shape: M=128, N=128, K=64
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
// Warp tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
// Instruction shape for INT8 Tensor Cores
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

// Number of pipeline stages
constexpr int NumStages = 3;

// Operation class for INT8 Tensor Cores
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Define the convolution kernel type
using Conv2dFprop = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementC,  // Accumulator type
    OperatorClass,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementC,
        ElementC
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,  // INT8 multiply-add with saturation
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

// Device-level convolution operator
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFprop>;

// ============================================================================
// Helper kernels for quantization/dequantization
// ============================================================================

// Find maximum absolute value (for per-tensor quantization)
__global__ void find_max_abs_kernel(
    const float* __restrict__ input,
    float* __restrict__ max_val,
    int total
) {
    __shared__ float smax[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = 0.0f;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }
    smax[tid] = local_max;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(max_val), __float_as_int(smax[0]));
    }
}

// Quantize FP32 to INT8 (NCHW -> NHWC conversion included)
__global__ void quantize_nchw_to_nhwc_kernel(
    const float* __restrict__ input,  // NCHW
    int8_t* __restrict__ output,      // NHWC
    float inv_scale,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;
    
    // Output index in NHWC
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n = idx / (C * W * H);
    
    // Input index in NCHW
    int input_idx = ((n * C + c) * H + h) * W + w;
    
    float val = input[input_idx] * inv_scale;
    val = fmaxf(-127.0f, fminf(127.0f, rintf(val)));
    output[idx] = static_cast<int8_t>(val);
}

// Quantize weight FP32 to INT8 (KCRS -> KRSC for CUTLASS)
// Per-channel quantization
__global__ void quantize_weight_kernel(
    const float* __restrict__ weight,  // KCRS (K, C, R, S)
    int8_t* __restrict__ output,       // KRSC for CUTLASS
    const float* __restrict__ scales,  // [K] per-channel scales
    int K, int C, int R, int S
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * R * S * C;
    if (idx >= total) return;
    
    // Output index in KRSC
    int c = idx % C;
    int s = (idx / C) % S;
    int r = (idx / (C * S)) % R;
    int k = idx / (C * S * R);
    
    // Input index in KCRS
    int input_idx = ((k * C + c) * R + r) * S + s;
    
    float inv_scale = 127.0f / fmaxf(scales[k], 1e-8f);
    float val = weight[input_idx] * inv_scale;
    val = fmaxf(-127.0f, fminf(127.0f, rintf(val)));
    output[idx] = static_cast<int8_t>(val);
}

// Dequantize INT32 to FP32 (NHWC -> NCHW conversion included)
__global__ void dequantize_nhwc_to_nchw_kernel(
    const int32_t* __restrict__ input,  // NHWC
    float* __restrict__ output,          // NCHW
    const float* __restrict__ input_scale,   // per-tensor [1]
    const float* __restrict__ weight_scales, // per-channel [K]
    const float* __restrict__ bias,          // [K] or nullptr
    int N, int K, int H, int W,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * H * W;
    if (idx >= total) return;
    
    // Output index in NCHW
    int w = idx % W;
    int h = (idx / W) % H;
    int k = (idx / (W * H)) % K;
    int n = idx / (W * H * K);
    
    // Input index in NHWC
    int input_idx = ((n * H + h) * W + w) * K + k;
    
    // Scale = input_scale * weight_scale / (127 * 127)
    // But our quantization uses: q = round(x * 127 / max)
    // So dequant: x = q * max / 127
    // Combined: output = int32 * (input_max/127) * (weight_max/127)
    float scale = input_scale[0] * weight_scales[k];
    float val = static_cast<float>(input[input_idx]) * scale;
    
    if (has_bias && bias != nullptr) {
        val += bias[k];
    }
    
    output[idx] = val;
}

// ============================================================================
// C Interface Functions
// ============================================================================

extern "C" {

// Main convolution function using CUTLASS
cudaError_t conv2d_int8_cutlass(
    const int8_t* input,      // NHWC [N, H, W, C]
    const int8_t* weight,     // KRSC [K, R, S, C]
    int32_t* output,          // NHWC [N, H_out, W_out, K]
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
) {
    // Calculate output dimensions
    int H_out = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    // Define problem size
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C},           // Input NHWC
        {K, R, S, C},           // Filter KRSC
        {pad_h, pad_h, pad_w, pad_w},  // Padding (top, bottom, left, right)
        {stride_h, stride_w},   // Stride
        {dilation_h, dilation_w}, // Dilation
        {N, H_out, W_out, K},   // Output NHWC
        cutlass::conv::Mode::kCrossCorrelation,
        1  // Split-K slices
    );
    
    // Create CUTLASS convolution arguments
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        {const_cast<int8_t*>(input), {C, W * C, H * W * C}},    // TensorRef A (input)
        {const_cast<int8_t*>(weight), {C, S * C, R * S * C}},   // TensorRef B (weight)
        {output, {K, W_out * K, H_out * W_out * K}},            // TensorRef C (output, for beta)
        {output, {K, W_out * K, H_out * W_out * K}},            // TensorRef D (output)
        {1, 0}  // alpha=1, beta=0
    };
    
    // Instantiate CUTLASS kernel
    ImplicitGemm conv_op;
    
    // Check if problem is supported
    cutlass::Status status = conv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    
    // Get workspace size
    size_t workspace_size = conv_op.get_workspace_size(arguments);
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Initialize the kernel
    status = conv_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return cudaErrorLaunchFailure;
    }
    
    // Run the convolution
    status = conv_op(stream);
    
    // Free workspace
    if (workspace) cudaFree(workspace);
    
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }
    
    return cudaSuccess;
}

// Quantize input (NCHW FP32 -> NHWC INT8)
void quantize_input_cutlass(
    const float* input,    // NCHW
    int8_t* output,        // NHWC
    float* scale,          // output: max_val / 127
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    // Find max absolute value
    float* d_max;
    cudaMalloc(&d_max, sizeof(float));
    cudaMemsetAsync(d_max, 0, sizeof(float), stream);
    
    find_max_abs_kernel<<<min(blocks, 1024), threads, 0, stream>>>(
        input, d_max, total
    );
    
    // Copy max value back and compute scale
    float h_max;
    cudaMemcpyAsync(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    float h_scale = h_max / 127.0f;
    if (h_scale < 1e-8f) h_scale = 1e-8f;
    cudaMemcpyAsync(scale, &h_scale, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    float inv_scale = 127.0f / h_max;
    
    // Quantize with layout conversion
    quantize_nchw_to_nhwc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, inv_scale, N, C, H, W
    );
    
    cudaFree(d_max);
}

// Quantize weight (KCRS FP32 -> KRSC INT8)
void quantize_weight_cutlass(
    const float* weight,   // KCRS
    int8_t* output,        // KRSC
    const float* scales,   // [K] per-channel max values
    int K, int C, int R, int S,
    cudaStream_t stream
) {
    int total = K * C * R * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    quantize_weight_kernel<<<blocks, threads, 0, stream>>>(
        weight, output, scales, K, C, R, S
    );
}

// Dequantize output (NHWC INT32 -> NCHW FP32)
void dequantize_output_cutlass(
    const int32_t* input,          // NHWC
    float* output,                  // NCHW
    const float* input_scale,       // per-tensor [1]
    const float* weight_scales,     // per-channel [K]
    const float* bias,              // [K] or nullptr
    int N, int K, int H, int W,
    bool has_bias,
    cudaStream_t stream
) {
    int total = N * K * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    dequantize_nhwc_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
        input, output, input_scale, weight_scales, bias,
        N, K, H, W, has_bias
    );
}

} // extern "C"
