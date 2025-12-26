#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

// Forward declaration
void im2col_cuda(
    const int8_t* input,
    int8_t* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out
);

// Convolution using im2col + cuBLAS GEMM
void conv2d_int8_cublas(
    const int8_t* input,      // [N, H, W, C_in]
    const int8_t* weight,     // [K, K, C_in, C_out] reshaped to [C_out, K*K*C_in]
    int32_t* output,          // [N, H_out, W_out, C_out]
    int N, int C_in, int H, int W,
    int C_out, int kernel_size,
    int stride, int padding
) {
    // Calculate output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate temporary buffer for im2col output
    int col_size = N * H_out * W_out * kernel_size * kernel_size * C_in;
    int8_t* d_col;
    cudaMalloc(&d_col, col_size * sizeof(int8_t));
    
    // Perform im2col transformation
    im2col_cuda(
        input, d_col,
        N, C_in, H, W,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        H_out, W_out
    );
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform GEMM: output = col @ weight^T
    // col is [M, K] where M = N*H_out*W_out, K = kernel_size^2 * C_in
    // weight is [C_out, K] (already transposed)
    // output is [M, C_out]
    
    int M = N * H_out * W_out;
    int K = kernel_size * kernel_size * C_in;
    int lda = K;
    int ldb = K;
    int ldc = C_out;
    
    int32_t alpha = 1;
    int32_t beta = 0;
    
    // Use cublasGemmEx for INT8 computation
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,      // weight is transposed
        CUBLAS_OP_N,      // col is not transposed
        C_out,            // m: rows of weight^T and output
        M,                // n: cols of col and output
        K,                // k: cols of weight^T and rows of col
        &alpha,
        weight, CUDA_R_8I, ldb,  // weight [C_out, K]
        d_col, CUDA_R_8I, lda,   // col [M, K]
        &beta,
        output, CUDA_R_32I, ldc, // output [M, C_out] but stored as [C_out, M]
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT
    );
    
    // Cleanup
    cudaFree(d_col);
    cublasDestroy(handle);
}
