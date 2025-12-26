#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void simple_test_kernel(half *C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        C[idx] = __float2half(42.0f);
    }
}

at::Tensor simple_test(int M, int N) {
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    
    simple_test_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simple_test", &simple_test, "Simple test kernel");
}
