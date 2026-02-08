#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA functions
torch::Tensor conv2d_int4_fprop(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor conv2d_int8_fprop(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor quantize_and_pack(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_int4_fprop", &conv2d_int4_fprop, "Conv2d INT4 Forward (CUTLASS)");
    m.def("conv2d_int8_fprop", &conv2d_int8_fprop, "Conv2d INT8 Forward (CUTLASS)");
    m.def("quantize_and_pack", &quantize_and_pack, "Fast Quantization and Packing for INT4");
}
