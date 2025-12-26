#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of the CUDA function
void conv2d_int8_cublas(
    const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int N, int C_in, int H, int W,
    int C_out, int kernel_size,
    int stride, int padding
);

// PyTorch wrapper
torch::Tensor conv2d_int8_cublas_torch(
    torch::Tensor input,   // [N, H, W, C_in] INT8
    torch::Tensor weight,  // [C_out, C_in, K, K] INT8, will reshape internally
    int stride = 1,
    int padding = 0
) {
    // Get dimensions
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C_in = input.size(3);
    
    int C_out = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate output tensor [N, H_out, W_out, C_out]
    auto output = torch::empty(
        {N, H_out, W_out, C_out},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device())
    );
    
    // Reshape weight from [C_out, C_in, K, K] to [C_out, K*K*C_in]
    auto weight_reshaped = weight.view({C_out, kernel_size * kernel_size * C_in});
    
    // Call CUDA kernel
    conv2d_int8_cublas(
        input.data_ptr<int8_t>(),
        weight_reshaped.data_ptr<int8_t>(),
        output.data_ptr<int32_t>(),
        N, C_in, H, W,
        C_out, kernel_size,
        stride, padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_int8_cublas", &conv2d_int8_cublas_torch, "INT8 Convolution using cuBLAS",
          py::arg("input"),
          py::arg("weight"),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}
