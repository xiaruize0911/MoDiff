/*
 * Python interface for INT4 CUTLASS convolution kernels
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// External C functions from conv_int4_cutlass.cu
extern "C" {
    void quantize_weight_int4(
        const float* weight,
        uint8_t* weight_int4,
        float* scales,
        int K, int R, int S, int C,
        cudaStream_t stream
    );
    
    void conv2d_int4_forward(
        const float* input,
        const uint8_t* weight_int4,
        const float* weight_scales,
        const float* bias,
        float* output,
        int N, int C, int H, int W,
        int K, int R, int S,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        cudaStream_t stream
    );
    
    void conv2d_int4_static_forward(
        const float* input,
        const uint8_t* weight_int4,
        const float* weight_scales,
        const float* bias,
        float* output,
        float input_scale,
        int N, int C, int H, int W,
        int K, int R, int S,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        cudaStream_t stream
    );
    
    // INT4 arithmetic functions (from int4_arithmetic.cu)
    void quantize_to_int4_packed(
        const float* input,
        uint8_t* output_packed,
        float* out_scale,
        int numel,
        cudaStream_t stream
    );
    
    void dequantize_from_int4_packed(
        const uint8_t* input_packed,
        float* output,
        float scale,
        int numel,
        cudaStream_t stream
    );
    
    void subtract_int4_packed(
        const uint8_t* a_packed,
        const uint8_t* b_packed,
        uint8_t* output_packed,
        int numel,
        cudaStream_t stream
    );
    
    void add_int4_packed(
        const uint8_t* a_packed,
        const uint8_t* b_packed,
        uint8_t* output_packed,
        int numel,
        cudaStream_t stream
    );
    
    void add_int4_packed_inplace(
        uint8_t* a_packed,
        const uint8_t* b_packed,
        int numel,
        cudaStream_t stream
    );
    
    // Native INT4 convolution (from conv_int4_cutlass_native.cu)
    cudaError_t conv2d_int4_native(
        const uint8_t* input_packed,
        const uint8_t* weight_packed,
        const float* weight_scales,
        const float* bias,
        float* output,
        int N, int H, int W, int C,
        int K, int R, int S,
        int pad_h, int pad_w,
        int stride_h, int stride_w,
        int dilation_h, int dilation_w,
        float input_scale,
        cudaStream_t stream
    );
}

// PyTorch wrapper: Quantize weight to INT4
std::tuple<torch::Tensor, torch::Tensor> quantize_weight(torch::Tensor weight) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (K, C, R, S)");
    
    auto weight_contig = weight.contiguous();
    int K = weight.size(0);
    int C = weight.size(1);
    int R = weight.size(2);
    int S = weight.size(3);
    
    // Allocate INT4 weight (packed, 2 values per byte)
    int total_elements = K * C * R * S;
    int packed_size = (total_elements + 1) / 2;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(weight.device());
    auto weight_int4 = torch::zeros({packed_size}, options);
    
    auto scale_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(weight.device());
    auto scales = torch::zeros({K}, scale_options);
    
    // Convert NCHW to KRSC (K, R, S, C) for per-channel quantization
    auto weight_krsc = weight.permute({0, 2, 3, 1}).contiguous();
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    quantize_weight_int4(
        weight_krsc.data_ptr<float>(),
        weight_int4.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        K, R, S, C,
        stream
    );
    
    return std::make_tuple(weight_int4, scales);
}

// PyTorch wrapper: INT4 Conv2d (dynamic quantization)
torch::Tensor conv2d_int4(
    torch::Tensor input,
    torch::Tensor weight_int4,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    
    auto input_contig = input.contiguous();
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int K = weight_scales.size(0);
    // Infer R, S from packed weight size and K, C
    int total_weight_elements = weight_int4.size(0) * 2;
    int RS = total_weight_elements / (K * C);
    int R = static_cast<int>(std::sqrt(RS));
    int S = R;
    
    // Calculate output size
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({N, K, P, Q}, input.options());
    
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_int4_forward(
        input_contig.data_ptr<float>(),
        weight_int4.data_ptr<uint8_t>(),
        weight_scales.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        K, R, S,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        stream
    );
    
    return output;
}

// PyTorch wrapper: INT4 Conv2d (static scale)
torch::Tensor conv2d_int4_static(
    torch::Tensor input,
    torch::Tensor weight_int4,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    float input_scale,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    
    auto input_contig = input.contiguous();
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int K = weight_scales.size(0);
    int total_weight_elements = weight_int4.size(0) * 2;
    int RS = total_weight_elements / (K * C);
    int R = static_cast<int>(std::sqrt(RS));
    int S = R;
    
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({N, K, P, Q}, input.options());
    
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_int4_static_forward(
        input_contig.data_ptr<float>(),
        weight_int4.data_ptr<uint8_t>(),
        weight_scales.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        input_scale,
        N, C, H, W,
        K, R, S,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        stream
    );
    
    return output;
}

// ============================================================================
// INT4 Arithmetic Operations
// ============================================================================

std::tuple<torch::Tensor, float> quantize_to_int4_packed_py(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    int numel = input.numel();
    int packed_size = (numel + 1) / 2;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(input.device());
    auto output_packed = torch::empty({packed_size}, options);
    
    float scale = 0.0f;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    quantize_to_int4_packed(
        input.data_ptr<float>(),
        output_packed.data_ptr<uint8_t>(),
        &scale,
        numel,
        stream
    );
    
    return std::make_tuple(output_packed, scale);
}

torch::Tensor dequantize_from_int4_packed_py(torch::Tensor input_packed, float scale, std::vector<int64_t> shape) {
    TORCH_CHECK(input_packed.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input_packed.dtype() == torch::kUInt8, "Input must be uint8");
    
    int numel = 1;
    for (auto s : shape) numel *= s;
    
    auto output = torch::empty(shape, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_packed.device()));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    dequantize_from_int4_packed(
        input_packed.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        scale,
        numel,
        stream
    );
    
    return output;
}

torch::Tensor subtract_int4_packed_py(torch::Tensor a_packed, torch::Tensor b_packed, int numel) {
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8 && b_packed.dtype() == torch::kUInt8, "Inputs must be uint8");
    
    auto output_packed = torch::empty_like(a_packed);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    subtract_int4_packed(
        a_packed.data_ptr<uint8_t>(),
        b_packed.data_ptr<uint8_t>(),
        output_packed.data_ptr<uint8_t>(),
        numel,
        stream
    );
    
    return output_packed;
}

torch::Tensor add_int4_packed_py(torch::Tensor a_packed, torch::Tensor b_packed, int numel) {
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8 && b_packed.dtype() == torch::kUInt8, "Inputs must be uint8");
    
    auto output_packed = torch::empty_like(a_packed);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_int4_packed(
        a_packed.data_ptr<uint8_t>(),
        b_packed.data_ptr<uint8_t>(),
        output_packed.data_ptr<uint8_t>(),
        numel,
        stream
    );
    
    return output_packed;
}

void add_int4_packed_inplace_py(torch::Tensor a_packed, torch::Tensor b_packed, int numel) {
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8 && b_packed.dtype() == torch::kUInt8, "Inputs must be uint8");
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_int4_packed_inplace(
        a_packed.data_ptr<uint8_t>(),
        b_packed.data_ptr<uint8_t>(),
        numel,
        stream
    );
}

torch::Tensor conv2d_int4_native_py(
    torch::Tensor input_packed,
    torch::Tensor weight_packed,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    float input_scale,
    std::vector<int64_t> input_shape,  // [N, H, W, C]
    std::vector<int64_t> weight_shape, // [K, R, S, C]
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    TORCH_CHECK(input_packed.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight_packed.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(input_shape.size() == 4, "Input shape must be [N, H, W, C]");
    TORCH_CHECK(weight_shape.size() == 4, "Weight shape must be [K, R, S, C]");
    
    int N = input_shape[0];
    int H = input_shape[1];
    int W = input_shape[2];
    int C = input_shape[3];
    
    int K = weight_shape[0];
    int R = weight_shape[1];
    int S = weight_shape[2];
    
    // Calculate output size
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({N, P, Q, K}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_packed.device()));
    
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = conv2d_int4_native(
        input_packed.data_ptr<uint8_t>(),
        weight_packed.data_ptr<uint8_t>(),
        weight_scales.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, H, W, C,
        K, R, S,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        input_scale,
        stream
    );
    
    TORCH_CHECK(err == cudaSuccess, "INT4 native convolution failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weight", &quantize_weight, "Quantize weight to INT4 (per-channel)");
    m.def("conv2d_int4", &conv2d_int4, "INT4 Conv2d (dynamic quantization)");
    m.def("conv2d_int4_static", &conv2d_int4_static, "INT4 Conv2d (static scale)");
    
    // INT4 arithmetic operations
    m.def("quantize_to_int4_packed", &quantize_to_int4_packed_py, "Quantize FP32 to INT4 packed");
    m.def("dequantize_from_int4_packed", &dequantize_from_int4_packed_py, "Dequantize INT4 packed to FP32");
    m.def("subtract_int4_packed", &subtract_int4_packed_py, "Subtract INT4 packed tensors");
    m.def("add_int4_packed", &add_int4_packed_py, "Add INT4 packed tensors");
    m.def("add_int4_packed_inplace", &add_int4_packed_inplace_py, "Add INT4 packed tensors in-place");
    
    // Native INT4 convolution
    m.def("conv2d_int4_native", &conv2d_int4_native_py, "Native INT4 Conv2d (no INT8 unpacking)");
}
