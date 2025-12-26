#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimized INT8 convolution kernel
// Incremental optimization: Tiling + Shared Memory + dp4a

#define BLOCK_SIZE 64

__global__ void conv2d_tiled_dp4a_kernel(
    const int8_t* __restrict__ input,    // [N, H, W, C_in]
    const int8_t* __restrict__ weight,   // [C_out, K, K, C_in]
    int32_t* __restrict__ output,        // [N, H_out, W_out, C_out]
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    // Each block handles a tile of (H_out * W_out) and C_out
    // For now, let's keep it simple: 
    // blockIdx.x -> (h_out, w_out) 
    // blockIdx.y -> c_out tile
    
    int hw_out = blockIdx.x;
    int c_out_base = blockIdx.y * BLOCK_SIZE;
    
    int w_out = hw_out % W_out;
    int h_out = hw_out / W_out;
    
    if (h_out >= H_out) return;

    int tid = threadIdx.x;
    int c_out = c_out_base + tid;
    
    if (c_out >= C_out) return;

    int32_t sum = 0;

    // Shared memory for input tile (C_in channels for one spatial position)
    // Assuming C_in is at most 1024 for now
    __shared__ int8_t s_input[1024];

    // Loop over kernel height and width
    for (int kh = 0; kh < K; kh++) {
        int h_in = h_out * stride - padding + kh;
        if (h_in < 0 || h_in >= H) continue;

        for (int kw = 0; kw < K; kw++) {
            int w_in = w_out * stride - padding + kw;
            if (w_in < 0 || w_in >= W) continue;

            // Load input into shared memory cooperatively
            const int8_t* in_ptr_global = &input[(((0 * H + h_in) * W + w_in) * C_in)];
            for (int i = tid; i < C_in; i += BLOCK_SIZE) {
                s_input[i] = in_ptr_global[i];
            }
            __syncthreads();

            // Now loop over C_in in chunks of 4 for dp4a
            const int8_t* w_ptr = &weight[((c_out * K + kh) * K + kw) * C_in];

            for (int c = 0; c < C_in; c += 4) {
                int32_t in_v = *(int32_t*)(&s_input[c]);
                int32_t w_v = *(int32_t*)(w_ptr + c);
                
                sum = __dp4a(in_v, w_v, sum);
            }
            __syncthreads();
        }
    }

    // Write output for n=0
    output[(hw_out * C_out) + c_out] = sum;
}

void conv2d_tiled_cuda(
    const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int stride, int padding
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    dim3 grid(H_out * W_out, (C_out + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    conv2d_tiled_dp4a_kernel<<<grid, block>>>(
        input, weight, output,
        N, C_in, H, W,
        C_out, K,
        stride, padding,
        H_out, W_out
    );
}
