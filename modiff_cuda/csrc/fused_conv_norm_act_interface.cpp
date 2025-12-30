/*
 * PyTorch C++ Interface for Fused Conv + GroupNorm + SiLU Kernels
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations from CUDA kernels
void launch_fused_groupnorm_silu(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);

void launch_fused_groupnorm_silu_fp16(
    const half* input,
    half* output,
    const half* gamma,
    const half* beta,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);

void launch_fused_depthwise_conv3x3_groupnorm_silu(
    const float* input,
    float* output,
    const float* weight,
    const float* conv_bias,
    const float* gamma,
    const float* beta,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);

void launch_fused_conv_groupnorm_silu_two_pass(
    const float* conv_output,
    float* output,
    float* group_mean,
    float* group_var,
    const float* gamma,
    const float* beta,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);


/**
 * Fused GroupNorm + SiLU
 * 
 * Takes conv output and applies GroupNorm followed by SiLU activation
 * in a single kernel pass.
 * 
 * Args:
 *   input: [N, C, H, W] tensor (conv output)
 *   gamma: [C] GroupNorm scale parameter
 *   beta: [C] GroupNorm bias parameter
 *   num_groups: Number of groups for GroupNorm
 *   eps: Epsilon for numerical stability
 * 
 * Returns:
 *   output: [N, C, H, W] tensor after GroupNorm + SiLU
 */
torch::Tensor fused_groupnorm_silu(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    TORCH_CHECK(C % num_groups == 0, "Channels must be divisible by num_groups");
    TORCH_CHECK(gamma.size(0) == C, "Gamma must have C elements");
    TORCH_CHECK(beta.size(0) == C, "Beta must have C elements");
    
    auto output = torch::empty_like(input);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (input.scalar_type() == torch::kFloat32) {
        launch_fused_groupnorm_silu(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            N, C, H, W, num_groups, eps, stream
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        launch_fused_groupnorm_silu_fp16(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(gamma.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(beta.data_ptr<at::Half>()),
            N, C, H, W, num_groups, eps, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Use float32 or float16.");
    }
    
    return output;
}


/**
 * Fused Conv (external) + GroupNorm + SiLU using two-pass approach
 * 
 * This is for standard convolutions where conv is computed externally.
 * Uses two passes: one for stats, one for apply.
 * 
 * Args:
 *   conv_output: [N, C, H, W] output from Conv2d
 *   gamma: [C] GroupNorm scale
 *   beta: [C] GroupNorm bias
 *   num_groups: Number of groups
 *   eps: Epsilon
 *   group_mean: [N, num_groups] temporary buffer (optional, will allocate if None)
 *   group_var: [N, num_groups] temporary buffer (optional, will allocate if None)
 * 
 * Returns:
 *   output: [N, C, H, W] after GroupNorm + SiLU
 */
torch::Tensor fused_conv_groupnorm_silu_two_pass(
    torch::Tensor conv_output,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    c10::optional<torch::Tensor> group_mean_opt,
    c10::optional<torch::Tensor> group_var_opt
) {
    TORCH_CHECK(conv_output.is_cuda(), "conv_output must be a CUDA tensor");
    TORCH_CHECK(conv_output.scalar_type() == torch::kFloat32, "Only float32 supported for two-pass");
    
    int N = conv_output.size(0);
    int C = conv_output.size(1);
    int H = conv_output.size(2);
    int W = conv_output.size(3);
    
    // Allocate or use provided temp buffers
    torch::Tensor group_mean, group_var;
    if (group_mean_opt.has_value()) {
        group_mean = group_mean_opt.value();
    } else {
        group_mean = torch::empty({N, num_groups}, conv_output.options());
    }
    if (group_var_opt.has_value()) {
        group_var = group_var_opt.value();
    } else {
        group_var = torch::empty({N, num_groups}, conv_output.options());
    }
    
    auto output = torch::empty_like(conv_output);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_fused_conv_groupnorm_silu_two_pass(
        conv_output.data_ptr<float>(),
        output.data_ptr<float>(),
        group_mean.data_ptr<float>(),
        group_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, H, W, num_groups, eps, stream
    );
    
    return output;
}


/**
 * Fused Depthwise Conv3x3 + GroupNorm + SiLU
 * 
 * All three operations in a single kernel for depthwise convolutions.
 * 
 * Args:
 *   input: [N, C, H, W]
 *   weight: [C, 1, 3, 3] depthwise conv weights
 *   conv_bias: [C] conv bias (optional)
 *   gamma: [C] GroupNorm scale
 *   beta: [C] GroupNorm bias
 *   num_groups: Number of groups
 *   eps: Epsilon
 * 
 * Returns:
 *   output: [N, C, H, W]
 */
torch::Tensor fused_depthwise_conv3x3_groupnorm_silu(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> conv_bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "Only 3x3 kernels supported");
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_fused_depthwise_conv3x3_groupnorm_silu(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.has_value() ? conv_bias.value().data_ptr<float>() : nullptr,
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, H, W, num_groups, eps, stream
    );
    
    return output;
}


/**
 * Benchmark helper: Fused vs Sequential
 * 
 * Returns timing for fused GroupNorm+SiLU vs sequential operations
 */
std::vector<float> benchmark_fused_vs_sequential(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    int warmup_iters,
    int bench_iters
) {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        auto out = fused_groupnorm_silu(input, gamma, beta, num_groups, 1e-5);
    }
    cudaDeviceSynchronize();
    
    // Benchmark fused
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; i++) {
        auto out = fused_groupnorm_silu(input, gamma, beta, num_groups, 1e-5);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fused_ms;
    cudaEventElapsedTime(&fused_ms, start, stop);
    fused_ms /= bench_iters;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return {fused_ms};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_groupnorm_silu", &fused_groupnorm_silu,
          "Fused GroupNorm + SiLU activation",
          py::arg("input"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("num_groups"),
          py::arg("eps") = 1e-5f);
    
    m.def("fused_conv_groupnorm_silu_two_pass", &fused_conv_groupnorm_silu_two_pass,
          "Fused Conv output + GroupNorm + SiLU (two-pass for standard conv)",
          py::arg("conv_output"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("num_groups"),
          py::arg("eps") = 1e-5f,
          py::arg("group_mean") = py::none(),
          py::arg("group_var") = py::none());
    
    m.def("fused_depthwise_conv3x3_groupnorm_silu", &fused_depthwise_conv3x3_groupnorm_silu,
          "Fused Depthwise 3x3 Conv + GroupNorm + SiLU",
          py::arg("input"),
          py::arg("weight"),
          py::arg("conv_bias"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("num_groups"),
          py::arg("eps") = 1e-5f);
    
    m.def("benchmark_fused_vs_sequential", &benchmark_fused_vs_sequential,
          "Benchmark fused vs sequential operations");
}
