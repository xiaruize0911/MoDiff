#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Im2col kernel for convolution
// Transforms input tensor into column format suitable for GEMM
// Input: [N, H, W, C] or [N, C, H, W]
// Output: [N*H_out*W_out, K*K*C] where K is kernel size
template <typename T>
__global__ void im2col_kernel(
    const T* input,
    T* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out
) {
    // Each thread handles one element in the output column matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H_out * W_out * kernel_h * kernel_w * C;
    
    if (idx >= total_elements) return;
    
    // Decode indices
    int w_col = idx % C;
    idx /= C;
    int kw = idx % kernel_w;
    idx /= kernel_w;
    int kh = idx % kernel_h;
    idx /= kernel_h;
    int w_out = idx % W_out;
    idx /= W_out;
    int h_out = idx % H_out;
    int n = idx / H_out;
    
    // Compute input position
    int h_in = h_out * stride_h - pad_h + kh;
    int w_in = w_out * stride_w - pad_w + kw;
    
    // Check bounds and handle padding
    T value = 0;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        // Input layout: [N, H, W, C]
        int input_idx = ((n * H + h_in) * W + w_in) * C + w_col;
        value = input[input_idx];
    }
    
    // Output layout: [N*H_out*W_out, K*K*C]
    int out_row = n * H_out * W_out + h_out * W_out + w_out;
    int out_col = (kh * kernel_w + kw) * C + w_col;
    int out_idx = out_row * (kernel_h * kernel_w * C) + out_col;
    
    output[out_idx] = value;
}

// Host function to launch im2col
void im2col_cuda(
    const int8_t* input,
    int8_t* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out
) {
    int total_elements = N * H_out * W_out * kernel_h * kernel_w * C;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    im2col_kernel<int8_t><<<blocks, threads>>>(
        input, output,
        N, C, H, W,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        H_out, W_out
    );
}
