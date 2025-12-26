#include <torch/extension.h>

void conv2d_simple_cuda(
    const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding
);

torch::Tensor conv2d_simple(
    torch::Tensor input,   // [N, H, W, C_in]
    torch::Tensor weight,  // [C_out, K, K, C_in]
    int stride = 1,
    int padding = 0
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, H, W, C_in]");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D [C_out, K, K, C_in]");
    TORCH_CHECK(input.dtype() == torch::kInt8, "Input must be INT8");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be INT8");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C_in = input.size(3);
    
    int C_out = weight.size(0);
    int K = weight.size(1);
    
    TORCH_CHECK(weight.size(2) == K, "Weight must be square kernel");
    TORCH_CHECK(weight.size(3) == C_in, "Weight C_in must match input");
    
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    auto output = torch::empty(
        {N, H_out, W_out, C_out},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device())
    );
    
    conv2d_simple_cuda(
        input.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        output.data_ptr<int32_t>(),
        N, C_in, H, W,
        C_out, K,
        stride, padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_simple", &conv2d_simple, "Simple INT8 Conv2d",
          py::arg("input"),
          py::arg("weight"),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}
