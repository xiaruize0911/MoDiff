#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Simple INT8 convolution kernel - correctness over performance
// No tensor cores, no async copy, no swizzling
// Just straightforward im2col + naive GEMM

__global__ void conv2d_simple_kernel(
    const int8_t* __restrict__ input,    // [N, H, W, C_in]
    const int8_t* __restrict__ weight,   // [C_out, K, K, C_in]
    int32_t* __restrict__ output,        // [N, H_out, W_out, C_out]
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * C_out;
    
    if (idx >= total) return;
    
    // Decode output position [n, h_out, w_out, c_out]
    int c_out = idx % C_out;
    int w_out = (idx / C_out) % W_out;
    int h_out = (idx / (C_out * W_out)) % H_out;
    int n = idx / (C_out * W_out * H_out);
    
    int32_t sum = 0;
    
    // Convolve over kernel
    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            for (int c_in = 0; c_in < C_in; c_in++) {
                // Input position
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                // Handle padding
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * H + h_in) * W + w_in) * C_in + c_in;
                    int weight_idx = ((c_out * K + kh) * K + kw) * C_in + c_in;
                    
                    int8_t in_val = input[input_idx];
                    int8_t w_val = weight[weight_idx];
                    sum += (int32_t)in_val * (int32_t)w_val;
                }
            }
        }
    }
    
    output[idx] = sum;
}

void conv2d_simple_cuda(
    const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    int total_outputs = N * H_out * W_out * C_out;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    conv2d_simple_kernel<<<blocks, threads>>>(
        input, weight, output,
        N, C_in, H, W,
        C_out, K,
        stride, padding,
        H_out, W_out
    );
}
