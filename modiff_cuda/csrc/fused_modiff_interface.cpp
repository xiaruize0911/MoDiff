/*
 * PyTorch C++ Interface for Fused MoDiff INT8 Kernels
 * 
 * Provides bindings for:
 * - fused_residual_quantize: Compute residual + quantize in one kernel
 * - fused_dequantize_accumulate: Dequantize conv output + accumulate with cache
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// External declarations from fused_modiff_kernels.cu
extern "C" {

void fused_modiff_residual_quantize(
    const float* input,
    const float* cache,
    float* updated_cache,
    int8_t* output,
    float* scale,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void fused_modiff_residual_quantize_fp16(
    const __half* input,
    const __half* cache,
    __half* updated_cache,
    int8_t* output,
    float* scale,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void fused_modiff_dequantize_accumulate(
    const int32_t* conv_output,
    const float* output_cache,
    float* output,
    float* updated_cache,
    float input_scale,
    const float* weight_scales,
    const float* bias,
    int N, int K, int H, int W,
    bool has_bias,
    bool has_cache,
    cudaStream_t stream
);

void fused_modiff_dequantize_accumulate_fp16(
    const int32_t* conv_output,
    const __half* output_cache,
    __half* output,
    __half* updated_cache,
    float input_scale,
    const float* weight_scales,
    const float* bias,
    int N, int K, int H, int W,
    bool has_bias,
    bool has_cache,
    cudaStream_t stream
);

}  // extern "C"


/**
 * Fused residual computation and quantization for MoDiff
 * 
 * Computes: residual = input - cache, quantize(residual), cache = input
 * All in a single kernel for reduced launch overhead.
 * 
 * @param input: Current activation [N, C, H, W] NCHW FP32
 * @param cache: Previous cached activation [N, C, H, W] NCHW FP32 (will be updated in-place)
 * @return: Tuple of (quantized_residual [N, H, W, C] NHWC INT8, scale FP32)
 */
std::tuple<torch::Tensor, torch::Tensor> fused_residual_quantize(
    torch::Tensor input,
    torch::Tensor cache
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(cache.is_cuda(), "Cache must be CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");
    TORCH_CHECK(input.sizes() == cache.sizes(), "Input and cache must have same shape");
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    // Output is NHWC for CUTLASS
    auto options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    
    torch::Tensor output = torch::empty({N, H, W, C}, options_int8);
    torch::Tensor scale = torch::zeros({1}, options_fp32);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    if (input.dtype() == torch::kFloat16) {
        fused_modiff_residual_quantize_fp16(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(cache.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(cache.data_ptr<at::Half>()),  // Update in-place
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            N, C, H, W,
            stream
        );
    } else {
        // FP32 path
        auto input_fp32 = input.to(torch::kFloat32).contiguous();
        auto cache_fp32 = cache.to(torch::kFloat32).contiguous();
        
        fused_modiff_residual_quantize(
            input_fp32.data_ptr<float>(),
            cache_fp32.data_ptr<float>(),
            cache_fp32.data_ptr<float>(),  // Update in-place (this won't work, need explicit output)
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            N, C, H, W,
            stream
        );
        
        // Copy back to cache
        cache.copy_(cache_fp32);
    }
    
    // Convert scale to actual dequantization scale (max / 127)
    // Synchronize to get the scale value
    cudaStreamSynchronize(stream);
    float max_val = scale.item<float>();
    scale.fill_(max_val / 127.0f);
    
    return std::make_tuple(output, scale);
}


/**
 * Fused dequantization and cache accumulation for MoDiff
 * 
 * Computes: output = dequant(conv_output) + output_cache
 * With in-place cache update.
 * 
 * @param conv_output: INT32 convolution output [N, H, W, K] NHWC
 * @param output_cache: Previous output cache [N, K, H, W] NCHW (will be updated)
 * @param input_scale: Input quantization scale
 * @param weight_scales: Per-channel weight scales [K]
 * @param bias: Optional bias [K] (can be empty tensor)
 * @param has_cache: Whether to accumulate with cache (false for first step)
 * @return: Output tensor [N, K, H, W] NCHW
 */
torch::Tensor fused_dequantize_accumulate(
    torch::Tensor conv_output,
    torch::Tensor output_cache,
    float input_scale,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    bool has_cache
) {
    TORCH_CHECK(conv_output.is_cuda(), "conv_output must be CUDA tensor");
    TORCH_CHECK(conv_output.dtype() == torch::kInt32, "conv_output must be INT32");
    TORCH_CHECK(conv_output.dim() == 4, "conv_output must be 4D [N, H, W, K]");
    
    int N = conv_output.size(0);
    int H = conv_output.size(1);
    int W = conv_output.size(2);
    int K = conv_output.size(3);
    
    bool has_bias = bias.numel() > 0;
    
    // Output is NCHW
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(conv_output.device());
    torch::Tensor output = torch::empty({N, K, H, W}, options);
    
    // Ensure output_cache is correct shape if has_cache
    if (has_cache) {
        TORCH_CHECK(output_cache.sizes() == output.sizes(), 
                   "output_cache must match output shape");
    }
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    fused_modiff_dequantize_accumulate(
        conv_output.data_ptr<int32_t>(),
        has_cache ? output_cache.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        output_cache.data_ptr<float>(),  // Update cache
        input_scale,
        weight_scales.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        N, K, H, W,
        has_bias,
        has_cache,
        stream
    );
    
    return output;
}


/**
 * Quantize input tensor for CUTLASS (NCHW -> NHWC)
 * Standalone version for first timestep
 */
std::tuple<torch::Tensor, torch::Tensor> quantize_input_nchw_to_nhwc(
    torch::Tensor input
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    // Find max for scale
    auto input_fp32 = input.to(torch::kFloat32).contiguous();
    float max_val = input_fp32.abs().max().item<float>();
    float scale = max_val / 127.0f;
    float inv_scale = (max_val > 1e-8f) ? (127.0f / max_val) : 0.0f;
    
    // Quantize with layout conversion
    // NCHW [n, c, h, w] -> NHWC [n, h, w, c]
    auto permuted = input_fp32.permute({0, 2, 3, 1}).contiguous();
    auto quantized = (permuted * inv_scale).round().clamp(-127, 127).to(torch::kInt8);
    
    auto scale_tensor = torch::tensor({scale}, input.options().dtype(torch::kFloat32));
    
    return std::make_tuple(quantized, scale_tensor);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_residual_quantize", &fused_residual_quantize,
          "Fused residual computation + quantization for MoDiff",
          py::arg("input"), py::arg("cache"));
    
    m.def("fused_dequantize_accumulate", &fused_dequantize_accumulate,
          "Fused dequantization + cache accumulation for MoDiff",
          py::arg("conv_output"), py::arg("output_cache"), 
          py::arg("input_scale"), py::arg("weight_scales"),
          py::arg("bias"), py::arg("has_cache"));
    
    m.def("quantize_input_nchw_to_nhwc", &quantize_input_nchw_to_nhwc,
          "Quantize input tensor with NCHW to NHWC conversion",
          py::arg("input"));
}
