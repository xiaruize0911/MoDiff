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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weight", &quantize_weight, "Quantize weight to INT4 (per-channel)");
    m.def("conv2d_int4", &conv2d_int4, "INT4 Conv2d (dynamic quantization)");
    m.def("conv2d_int4_static", &conv2d_int4_static, "INT4 Conv2d (static scale)");
}
