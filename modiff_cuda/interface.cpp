#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Forward declarations of CUDA functions
extern "C" {
void conv2d_fast_w8a8_cuda(
    const int8_t* input, const int8_t* weight, at::Half* output,
    const float* act_scale, const float* weight_scales,
    float* max_output,
    int N, int C_in, int H, int W, int C_out, int K_size,
    int stride, int padding, cudaStream_t stream
);

void conv2d_fast_w8a8_accum_cuda(
    const int8_t* input, const int8_t* weight, at::Half* output, const at::Half* prev_output,
    const float* act_scale, const float* weight_scales,
    float* max_output,
    int N, int C_in, int H, int W, int C_out, int K_size,
    int stride, int padding, cudaStream_t stream
);

void quantize_tensor_cuda(const float* input, int8_t* output, const float* scale_ptr, int size, cudaStream_t stream);
void quantize_tensor_half_cuda(const at::Half* input, int8_t* output, const float* scale_ptr, int size, cudaStream_t stream);

void quantize_permute_cuda(const float* input, int8_t* output, const float* scale_ptr, int N, int C, int H, int W, cudaStream_t stream);
void quantize_permute_half_cuda(const at::Half* input, int8_t* output, const float* scale_ptr, int N, int C, int H, int W, cudaStream_t stream);

void permute_half_nhwc_nchw_cuda(const at::Half* input, at::Half* output, float* max_val, int N, int C, int H, int W, cudaStream_t stream);

void find_max_abs_cuda(const float* input, float* max_val, int size, cudaStream_t stream);
void find_max_abs_half_cuda(const at::Half* input, float* max_val, int size, cudaStream_t stream);

void find_max_abs_diff_cuda(const float* x, const float* prev, float* max_val, int size, cudaStream_t stream);
void find_max_abs_diff_half_cuda(const at::Half* x, const at::Half* prev, float* max_val, int size, cudaStream_t stream);
void modiff_update_cuda(const float* x, float* prev, int8_t* out_q, const float* max_val_ptr, int size, cudaStream_t stream);
void modiff_update_half_cuda(const at::Half* x, at::Half* prev, int8_t* out_q, const float* max_val_ptr, int size, cudaStream_t stream);

void find_max_abs_diff_permute_cuda(const float* x, const float* prev, float* max_val, int N, int C, int H, int W, cudaStream_t stream);
void find_max_abs_diff_permute_half_cuda(const at::Half* x, const at::Half* prev, float* max_val, int N, int C, int H, int W, cudaStream_t stream);
void modiff_update_permute_cuda(const float* x, float* prev, int8_t* out_q, const float* max_val_ptr, int N, int C, int H, int W, cudaStream_t stream);
void modiff_update_permute_half_cuda(const at::Half* x, at::Half* prev, int8_t* out_q, const float* max_val_ptr, int N, int C, int H, int W, cudaStream_t stream);
}

at::Tensor find_max_abs(at::Tensor input) {
    auto max_val = at::zeros({1}, input.options().dtype(at::kFloat));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kHalf) {
        find_max_abs_half_cuda(input.data_ptr<at::Half>(), max_val.data_ptr<float>(), input.numel(), stream);
    } else {
        find_max_abs_cuda(input.data_ptr<float>(), max_val.data_ptr<float>(), input.numel(), stream);
    }
    return max_val;
}

void modiff_quantize_update(at::Tensor x, at::Tensor prev, at::Tensor out_q, at::Tensor max_val) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int size = x.numel();
    
    if (x.scalar_type() == at::kHalf) {
        find_max_abs_diff_half_cuda(x.data_ptr<at::Half>(), prev.data_ptr<at::Half>(), max_val.data_ptr<float>(), size, stream);
        modiff_update_half_cuda(x.data_ptr<at::Half>(), prev.data_ptr<at::Half>(), out_q.data_ptr<int8_t>(), max_val.data_ptr<float>(), size, stream);
    } else {
        find_max_abs_diff_cuda(x.data_ptr<float>(), prev.data_ptr<float>(), max_val.data_ptr<float>(), size, stream);
        modiff_update_cuda(x.data_ptr<float>(), prev.data_ptr<float>(), out_q.data_ptr<int8_t>(), max_val.data_ptr<float>(), size, stream);
    }
}

void modiff_quantize_update_permute(at::Tensor x, at::Tensor prev, at::Tensor out_q, at::Tensor max_val) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    if (x.scalar_type() == at::kHalf) {
        find_max_abs_diff_permute_half_cuda(x.data_ptr<at::Half>(), prev.data_ptr<at::Half>(), max_val.data_ptr<float>(), N, C, H, W, stream);
        modiff_update_permute_half_cuda(x.data_ptr<at::Half>(), prev.data_ptr<at::Half>(), out_q.data_ptr<int8_t>(), max_val.data_ptr<float>(), N, C, H, W, stream);
    } else {
        find_max_abs_diff_permute_cuda(x.data_ptr<float>(), prev.data_ptr<float>(), max_val.data_ptr<float>(), N, C, H, W, stream);
        modiff_update_permute_cuda(x.data_ptr<float>(), prev.data_ptr<float>(), out_q.data_ptr<int8_t>(), max_val.data_ptr<float>(), N, C, H, W, stream);
    }
}

// Returns (output, max_val)
std::vector<at::Tensor> permute_half_nhwc_nchw(at::Tensor input, bool compute_max) {
    // Input is NHWC
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);
    
    // Output is NCHW
    auto output = at::empty({N, C, H, W}, input.options().memory_format(at::MemoryFormat::Contiguous));
    at::Tensor max_val;
    
    if (compute_max) {
        max_val = at::zeros({1}, input.options().dtype(at::kFloat));
    } else {
        max_val = at::empty({0}, input.options().dtype(at::kFloat)); // Dummy
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    permute_half_nhwc_nchw_cuda(
        input.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        compute_max ? max_val.data_ptr<float>() : nullptr,
        N, C, H, W,
        stream
    );
    return {output, max_val};
}

at::Tensor quantize_permute(at::Tensor input, at::Tensor scale) {
    // Input is NCHW
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    // Output is NHWC
    auto output = at::empty({N, H, W, C}, input.options().dtype(at::kChar).memory_format(at::MemoryFormat::Contiguous));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kHalf) {
        quantize_permute_half_cuda(
            input.data_ptr<at::Half>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            N, C, H, W,
            stream
        );
    } else {
        quantize_permute_cuda(
            input.data_ptr<float>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            N, C, H, W,
            stream
        );
    }
    return output;
}

at::Tensor quantize_tensor(at::Tensor input, at::Tensor scale) {
    auto output = at::empty_like(input, input.options().dtype(at::kChar));
    int size = input.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kHalf) {
        quantize_tensor_half_cuda(
            input.data_ptr<at::Half>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            size,
            stream
        );
    } else {
        quantize_tensor_cuda(
            input.data_ptr<float>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            size,
            stream
        );
    }
    return output;
}

std::vector<at::Tensor> conv2d_fast_w8a8(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor act_scale,
    at::Tensor weight_scales,
    int kernel_size,
    int stride,
    int padding,
    bool compute_max) {
    
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C_in = input.size(3);
    
    int C_out = weight.size(0);
    
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto output = at::empty({N, H_out, W_out, C_out}, input.options().dtype(at::kHalf));
    
    at::Tensor max_val;
    if (compute_max) {
        max_val = at::zeros({1}, input.options().dtype(at::kFloat));
    } else {
        max_val = at::empty({0}, input.options().dtype(at::kFloat));
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_fast_w8a8_cuda(
        input.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        output.data_ptr<at::Half>(),
        act_scale.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        compute_max ? max_val.data_ptr<float>() : nullptr,
        N, C_in, H, W, C_out, kernel_size,
        stride, padding,
        stream
    );
    
    return {output, max_val};
}

std::vector<at::Tensor> conv2d_fast_w8a8_accum(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor prev_output,
    at::Tensor act_scale,
    at::Tensor weight_scales,
    int kernel_size,
    int stride,
    int padding,
    bool compute_max) {
    
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C_in = input.size(3);
    
    int C_out = weight.size(0);
    
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto output = at::empty({N, H_out, W_out, C_out}, input.options().dtype(at::kHalf));
    
    at::Tensor max_val;
    if (compute_max) {
        max_val = at::zeros({1}, input.options().dtype(at::kFloat));
    } else {
        max_val = at::empty({0}, input.options().dtype(at::kFloat));
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_fast_w8a8_accum_cuda(
        input.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        (at::Half*)output.data_ptr<at::Half>(),
        (const at::Half*)prev_output.data_ptr<at::Half>(),
        act_scale.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        compute_max ? max_val.data_ptr<float>() : nullptr,
        N, C_in, H, W, C_out, kernel_size,
        stride, padding,
        stream
    );
    
    return {output, max_val};
}

PYBIND11_MODULE(modiff_cuda_backend, m) {
  m.def("conv2d_fast_w8a8", &conv2d_fast_w8a8, "Fast W8A8 Conv2d (Implicit GEMM + dp4a + scaling)",
        py::arg("input"), py::arg("weight"), py::arg("act_scale"), py::arg("weight_scales"),
        py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 1, py::arg("compute_max") = false);

  m.def("conv2d_fast_w8a8_accum", &conv2d_fast_w8a8_accum, "Fast W8A8 Conv2d with Accumulation",
        py::arg("input"), py::arg("weight"), py::arg("prev_output"), py::arg("act_scale"), py::arg("weight_scales"),
        py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 1, py::arg("compute_max") = false);

  m.def("quantize_tensor", &quantize_tensor, "Fast Quantization Kernel",
        py::arg("input"), py::arg("scale"));

  m.def("quantize_permute", &quantize_permute, "Fused Permute + Quantize Kernel (NCHW -> NHWC)",
        py::arg("input"), py::arg("scale"));

  m.def("permute_half_nhwc_nchw", &permute_half_nhwc_nchw, "Permute Kernel (NHWC -> NCHW) for Half",
        py::arg("input"), py::arg("compute_max") = false);

  m.def("find_max_abs", &find_max_abs, "Find Max Abs Value",
        py::arg("input"));

  m.def("modiff_quantize_update", &modiff_quantize_update, "Fused MoDiff Quantize and Update",
        py::arg("x"), py::arg("prev"), py::arg("out_q"), py::arg("max_val"));

  m.def("modiff_quantize_update_permute", &modiff_quantize_update_permute, "Fused MoDiff Quantize and Update with Permute",
        py::arg("x"), py::arg("prev"), py::arg("out_q"), py::arg("max_val"));
}
