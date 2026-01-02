/*
 * Simple INT4 implementation using FP16 fallback
 * For production, this should use CUTLASS INT4 Tensor Cores
 * Current approach: INT4 quantization for memory, FP16 compute
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Simple unpack INT4 to FP16 kernel
__global__ void unpack_int4_to_fp16_kernel(
    const uint8_t* __restrict__ packed,
    __half* __restrict__ unpacked,
    float scale,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    int packed_idx = idx / 2;
    uint8_t p = packed[packed_idx];
    
    int8_t val;
    if (idx % 2 == 0) {
        val = p & 0x0F;
    } else {
        val = (p >> 4) & 0x0F;
    }
    
    // Sign extend from 4-bit
    if (val & 0x08) {
        val |= 0xF0;
    }
    
    unpacked[idx] = __float2half(static_cast<float>(val) * scale);
}

extern "C" {

// Quantize weight to INT4
void quantize_weight_int4_simple(
    const float* weight,
    uint8_t* weight_int4,
    float* scales,
    int K, int R, int S, int C
) {
    // Per-channel quantization on CPU for simplicity
    for (int k = 0; k < K; k++) {
        float max_val = 0.0f;
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                for (int c = 0; c < C; c++) {
                    int idx = ((k * R + r) * S + s) * C + c;
                    max_val = fmaxf(max_val, fabsf(weight[idx]));
                }
            }
        }
        
        float scale = max_val / 7.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        scales[k] = scale;
        
        // Quantize and pack
        for (int i = 0; i < R * S * C; i += 2) {
            int idx0 = k * R * S * C + i;
            int idx1 = idx0 + 1;
            
            int quant0 = (int)roundf(weight[idx0] / scale);
            quant0 = fmaxf(-8.0f, fminf(7.0f, (float)quant0));
            
            int quant1 = 0;
            if (i + 1 < R * S * C) {
                quant1 = (int)roundf(weight[idx1] / scale);
                quant1 = fmaxf(-8.0f, fminf(7.0f, (float)quant1));
            }
            
            int packed_idx = idx0 / 2;
            weight_int4[packed_idx] = (quant0 & 0x0F) | ((quant1 & 0x0F) << 4);
        }
    }
}

} // extern "C"
