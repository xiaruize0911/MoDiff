/**
 * INT4 Packed Arithmetic Operations
 * 
 * Implements efficient INT4 operations on packed uint8 data (2 INT4 values per byte).
 * Supports MoDiff temporal caching in true INT4 precision without FP32 round-trips.
 * 
 * INT4 Format:
 * - Packed: 2 INT4 values per uint8 byte
 * - Lower 4 bits: first value (sign-extended)
 * - Upper 4 bits: second value (sign-extended)
 * - Range: -8 to 7 (signed 4-bit)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Quantization/Dequantization Kernels
// ============================================================================

/**
 * Quantize FP32 tensor to packed INT4 (uint8).
 * Each byte stores 2 INT4 values.
 */
__global__ void quantize_to_int4_packed_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output_packed,
    float scale,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= numel) return;
    
    // Quantize two values and pack into one byte
    int idx0 = idx * 2;
    int idx1 = idx0 + 1;
    
    // Quantize first value (lower 4 bits)
    float val0 = input[idx0];
    int8_t q0 = __float2int_rn(val0 / scale);
    q0 = max(-8, min(7, (int)q0));  // Clamp to [-8, 7]
    uint8_t packed0 = q0 & 0x0F;
    
    // Quantize second value (upper 4 bits)
    uint8_t packed1 = 0;
    if (idx1 < numel) {
        float val1 = input[idx1];
        int8_t q1 = __float2int_rn(val1 / scale);
        q1 = max(-8, min(7, (int)q1));
        packed1 = (q1 & 0x0F) << 4;
    }
    
    output_packed[idx] = packed0 | packed1;
}

/**
 * Dequantize packed INT4 (uint8) to FP32 tensor.
 */
__global__ void dequantize_from_int4_packed_kernel(
    const uint8_t* __restrict__ input_packed,
    float* __restrict__ output,
    float scale,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= numel) return;
    
    uint8_t packed = input_packed[idx];
    
    // Extract lower 4 bits (first value)
    int8_t q0 = packed & 0x0F;
    if (q0 & 0x08) q0 |= 0xF0;  // Sign extend
    int idx0 = idx * 2;
    output[idx0] = q0 * scale;
    
    // Extract upper 4 bits (second value)
    int idx1 = idx0 + 1;
    if (idx1 < numel) {
        int8_t q1 = (packed >> 4) & 0x0F;
        if (q1 & 0x08) q1 |= 0xF0;  // Sign extend
        output[idx1] = q1 * scale;
    }
}

/**
 * Compute dynamic scale for INT4 quantization (99.99 percentile).
 */
__global__ void compute_int4_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ max_val,
    int numel
) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load max absolute value
    float local_max = 0.0f;
    if (idx < numel) {
        local_max = fabsf(input[idx]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    
    // Reduce max within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write block max
    if (tid == 0) {
        atomicMax((int*)max_val, __float_as_int(sdata[0]));
    }
}

// ============================================================================
// INT4 Packed Arithmetic Kernels
// ============================================================================

/**
 * Subtract two INT4 packed tensors: output = a - b
 * Result is clamped to [-8, 7] range.
 */
__global__ void subtract_int4_packed_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ b_packed,
    uint8_t* __restrict__ output_packed,
    int packed_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= packed_numel) return;
    
    uint8_t a_byte = a_packed[idx];
    uint8_t b_byte = b_packed[idx];
    
    // Extract and subtract lower 4 bits
    int8_t a0 = a_byte & 0x0F;
    if (a0 & 0x08) a0 |= 0xF0;
    int8_t b0 = b_byte & 0x0F;
    if (b0 & 0x08) b0 |= 0xF0;
    int8_t diff0 = a0 - b0;
    diff0 = max(-8, min(7, (int)diff0));
    uint8_t packed0 = diff0 & 0x0F;
    
    // Extract and subtract upper 4 bits
    int8_t a1 = (a_byte >> 4) & 0x0F;
    if (a1 & 0x08) a1 |= 0xF0;
    int8_t b1 = (b_byte >> 4) & 0x0F;
    if (b1 & 0x08) b1 |= 0xF0;
    int8_t diff1 = a1 - b1;
    diff1 = max(-8, min(7, (int)diff1));
    uint8_t packed1 = (diff1 & 0x0F) << 4;
    
    output_packed[idx] = packed0 | packed1;
}

/**
 * Add two INT4 packed tensors: output = a + b
 * Result is clamped to [-8, 7] range.
 */
__global__ void add_int4_packed_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ b_packed,
    uint8_t* __restrict__ output_packed,
    int packed_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= packed_numel) return;
    
    uint8_t a_byte = a_packed[idx];
    uint8_t b_byte = b_packed[idx];
    
    // Extract and add lower 4 bits
    int8_t a0 = a_byte & 0x0F;
    if (a0 & 0x08) a0 |= 0xF0;
    int8_t b0 = b_byte & 0x0F;
    if (b0 & 0x08) b0 |= 0xF0;
    int8_t sum0 = a0 + b0;
    sum0 = max(-8, min(7, (int)sum0));
    uint8_t packed0 = sum0 & 0x0F;
    
    // Extract and add upper 4 bits
    int8_t a1 = (a_byte >> 4) & 0x0F;
    if (a1 & 0x08) a1 |= 0xF0;
    int8_t b1 = (b_byte >> 4) & 0x0F;
    if (b1 & 0x08) b1 |= 0xF0;
    int8_t sum1 = a1 + b1;
    sum1 = max(-8, min(7, (int)sum1));
    uint8_t packed1 = (sum1 & 0x0F) << 4;
    
    output_packed[idx] = packed0 | packed1;
}

/**
 * In-place add: a = a + b
 */
__global__ void add_int4_packed_inplace_kernel(
    uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ b_packed,
    int packed_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= packed_numel) return;
    
    uint8_t a_byte = a_packed[idx];
    uint8_t b_byte = b_packed[idx];
    
    // Extract and add lower 4 bits
    int8_t a0 = a_byte & 0x0F;
    if (a0 & 0x08) a0 |= 0xF0;
    int8_t b0 = b_byte & 0x0F;
    if (b0 & 0x08) b0 |= 0xF0;
    int8_t sum0 = a0 + b0;
    sum0 = max(-8, min(7, (int)sum0));
    uint8_t packed0 = sum0 & 0x0F;
    
    // Extract and add upper 4 bits
    int8_t a1 = (a_byte >> 4) & 0x0F;
    if (a1 & 0x08) a1 |= 0xF0;
    int8_t b1 = (b_byte >> 4) & 0x0F;
    if (b1 & 0x08) b1 |= 0xF0;
    int8_t sum1 = a1 + b1;
    sum1 = max(-8, min(7, (int)sum1));
    uint8_t packed1 = (sum1 & 0x0F) << 4;
    
    a_packed[idx] = packed0 | packed1;
}

// ============================================================================
// Host API Functions
// ============================================================================

extern "C" {

/**
 * Quantize FP32 to INT4 packed (uint8).
 * Returns: (packed_tensor, scale)
 */
void quantize_to_int4_packed(
    const float* input,
    uint8_t* output_packed,
    float* out_scale,
    int numel,
    cudaStream_t stream = 0
) {
    // Compute scale (max/7 for INT4 range -8 to 7)
    float* d_max;
    cudaMalloc(&d_max, sizeof(float));
    cudaMemset(d_max, 0, sizeof(float));
    
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    compute_int4_scale_kernel<<<blocks, threads, 0, stream>>>(
        input, d_max, numel
    );
    
    float h_max;
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_max);
    
    float scale = (h_max < 1e-8f) ? 1.0f : (h_max / 7.0f);
    *out_scale = scale;
    
    // Quantize and pack
    int packed_numel = (numel + 1) / 2;
    int pack_blocks = (packed_numel + threads - 1) / threads;
    quantize_to_int4_packed_kernel<<<pack_blocks, threads, 0, stream>>>(
        input, output_packed, scale, numel
    );
}

/**
 * Dequantize INT4 packed (uint8) to FP32.
 */
void dequantize_from_int4_packed(
    const uint8_t* input_packed,
    float* output,
    float scale,
    int numel,
    cudaStream_t stream = 0
) {
    int packed_numel = (numel + 1) / 2;
    int threads = 256;
    int blocks = (packed_numel + threads - 1) / threads;
    dequantize_from_int4_packed_kernel<<<blocks, threads, 0, stream>>>(
        input_packed, output, scale, numel
    );
}

/**
 * Subtract INT4 packed: output = a - b
 */
void subtract_int4_packed(
    const uint8_t* a_packed,
    const uint8_t* b_packed,
    uint8_t* output_packed,
    int numel,
    cudaStream_t stream = 0
) {
    int packed_numel = (numel + 1) / 2;
    int threads = 256;
    int blocks = (packed_numel + threads - 1) / threads;
    subtract_int4_packed_kernel<<<blocks, threads, 0, stream>>>(
        a_packed, b_packed, output_packed, packed_numel
    );
}

/**
 * Add INT4 packed: output = a + b
 */
void add_int4_packed(
    const uint8_t* a_packed,
    const uint8_t* b_packed,
    uint8_t* output_packed,
    int numel,
    cudaStream_t stream = 0
) {
    int packed_numel = (numel + 1) / 2;
    int threads = 256;
    int blocks = (packed_numel + threads - 1) / threads;
    add_int4_packed_kernel<<<blocks, threads, 0, stream>>>(
        a_packed, b_packed, output_packed, packed_numel
    );
}

/**
 * Add INT4 packed in-place: a = a + b
 */
void add_int4_packed_inplace(
    uint8_t* a_packed,
    const uint8_t* b_packed,
    int numel,
    cudaStream_t stream = 0
) {
    int packed_numel = (numel + 1) / 2;
    int threads = 256;
    int blocks = (packed_numel + threads - 1) / threads;
    add_int4_packed_inplace_kernel<<<blocks, threads, 0, stream>>>(
        a_packed, b_packed, packed_numel
    );
}

}  // extern "C"
