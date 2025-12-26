#include <torch/extension.h>
#include <cuda_fp16.h>

void test_global_write_run(half *C, int M, int N);
void test_shared_memory_run(half *C, int M, int N, int smem_size);
void test_conv_smem_pattern_run(half *C, int M, int N, int smem_size);

at::Tensor test_global_write(int M, int N) {
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    test_global_write_run(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N
    );
    return output;
}

at::Tensor test_shared_memory(int M, int N) {
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    int smem_size = 34816;  // Same as Conv kernel
    test_shared_memory_run(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N, smem_size
    );
    return output;
}

at::Tensor test_conv_smem_pattern(int M, int N) {
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    int smem_size = 34816;  // Same as Conv kernel
    test_conv_smem_pattern_run(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N, smem_size
    );
    return output;
}
