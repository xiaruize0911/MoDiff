// =========================================================================
// Standalone elementwise quantize / pack / dequant-accumulate kernels.
//
// These are the simplest building blocks in the extension: pure elementwise
// transforms with no MoDiff cache state and no CUTLASS involvement. Each one
// fuses what would otherwise be 2-5 separate PyTorch elementwise kernel
// launches (multiply, round, clamp, cast, pack) into a single launch.
//
// Layout contract: quantize_and_pack / scale_quantize_and_pack read raw flat
// memory and reinterpret adjacent element pairs as an NHWC channel pair, so
// the input tensor must be channels_last-contiguous (CHECK_CONTIGUOUS below).
// scale_quantize_int8 / dequant_accumulate_* are purely elementwise and work
// on any layout.
// =========================================================================

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "../common.cuh"

// Fused scale + quantize to INT8 kernel (vectorized float4 loads).
// Replaces: x_scaled = x * scale; x_int8 = x_scaled.round().clamp(-127, 127).to(int8)
// Scale is read from device memory (no CPU sync needed).
__global__ void scale_quantize_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 in_v = reinterpret_cast<const float4*>(input)[idx4];
        int8_t o0 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.x * scale)));
        int8_t o1 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.y * scale)));
        int8_t o2 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.z * scale)));
        int8_t o3 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(in_v.w * scale)));
        // 32-bit coalesced write for 4 int8 values
        reinterpret_cast<int32_t*>(output)[idx4] =
            ((unsigned char)o0) | ((unsigned char)o1 << 8) |
            ((unsigned char)o2 << 16) | ((unsigned char)o3 << 24);
    } else {
        // Scalar tail
        for (int i = base; i < num_elements; i++) {
            output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(input[i] * scale)));
        }
    }
}

// Fused dequant + accumulate for INT4 cache update (vectorized float4).
// Replaces: a_hat_cache += round(clamp(residual * scale, -7, 7)) / scale
// (round/clamp is the quantize step; dividing back by scale dequantizes it before
// accumulating, so the cache always holds an FP32 approximation, never raw int4 codes.)
__global__ void dequant_accumulate_int4_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];
        float4 c_v = reinterpret_cast<float4*>(a_hat_cache)[idx4];
        c_v.x += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.x * scale))) * inv_scale;
        c_v.y += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.y * scale))) * inv_scale;
        c_v.z += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.z * scale))) * inv_scale;
        c_v.w += fmaxf(-7.0f, fminf(7.0f, roundf(r_v.w * scale))) * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            a_hat_cache[i] += fmaxf(-7.0f, fminf(7.0f, roundf(r * scale))) * inv_scale;
        }
    }
}

// Fused dequant + accumulate for INT8 cache update (vectorized float4).
// Replaces: a_hat_cache += round(clamp(residual * scale, -127, 127)) / scale
__global__ void dequant_accumulate_int8_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;
    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];
        float4 c_v = reinterpret_cast<float4*>(a_hat_cache)[idx4];
        c_v.x += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.x * scale))) * inv_scale;
        c_v.y += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.y * scale))) * inv_scale;
        c_v.z += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.z * scale))) * inv_scale;
        c_v.w += fmaxf(-127.0f, fminf(127.0f, roundf(r_v.w * scale))) * inv_scale;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            a_hat_cache[i] += fmaxf(-127.0f, fminf(127.0f, roundf(r * scale))) * inv_scale;
        }
    }
}

// Packs 2 already-clamped-to-[-7,7] float values per output byte (low nibble = the
// first/even element, high nibble = the second/odd element). No scaling is applied
// here — the caller is expected to have pre-scaled the input into int4 range; see
// scale_quantize_pack_kernel below for the version that also scales.
__global__ void quantize_pack_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    int num_elements // Number of OUTPUT packed bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Use float2 vectorized load for 2 adjacent floats
    float2 vals = reinterpret_cast<const float2*>(input)[idx];

    int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(vals.x)));
    int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(vals.y)));

    output[idx] = (i0 & 0x0F) | ((i1 & 0x0F) << 4);
}

// Fused version: applies per-tensor scale before quantizing and packing.
// Eliminates a separate x*scale kernel launch + global memory round-trip.
// Scale is read from device memory (no CPU sync needed).
__global__ void scale_quantize_pack_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements // Number of OUTPUT packed bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float scale = *scale_ptr;
    float2 vals = reinterpret_cast<const float2*>(input)[idx];
    float v0 = vals.x * scale;
    float v1 = vals.y * scale;

    int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v0)));
    int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v1)));

    output[idx] = (i0 & 0x0F) | ((i1 & 0x0F) << 4);
}

// input: FP32, channels_last-contiguous NCHW logical / NHWC physical.
// output: INT8 [N, H, W, C/2] with 2 packed int4 values per byte.
torch::Tensor quantize_and_pack(torch::Tensor input) {
    CHECK_CONTIGUOUS(input);

    int num_input = input.numel();
    int num_output = num_input / 2;

    auto output = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block_size = 256;
    int grid_size = (num_output + block_size - 1) / block_size;

    quantize_pack_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        num_output
    );

    // input is channels_last (NCHW logical / NHWC physical), so the packed flat
    // buffer above is already in [N, H, W, C/2] element order.
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    return output.view({N, H, W, C / 2});
}

// Same as quantize_and_pack, but multiplies by `scale` (device scalar) before
// quantizing, fusing the activation-scale multiply into the same kernel.
torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale) {
    CHECK_CONTIGUOUS(input);

    int num_input = input.numel();
    int num_output = num_input / 2;

    auto output = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block_size = 256;
    int grid_size = (num_output + block_size - 1) / block_size;

    scale_quantize_pack_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        num_output
    );

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    return output.view({N, H, W, C / 2});
}

// Fused: input * scale -> round -> clamp(-127,127) -> int8 (vectorized float4).
// Layout-agnostic (purely elementwise), unlike the pack functions above.
torch::Tensor scale_quantize_int8(torch::Tensor input, torch::Tensor scale) {
    auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = input.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    scale_quantize_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        num_elements
    );

    return output;
}

// In-place: a_hat_cache += round(clamp(residual * scale, -7, 7)) / scale
void dequant_accumulate_int4(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = residual.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    dequant_accumulate_int4_kernel<<<grid_size, block_size, 0, stream>>>(
        residual.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        scale.data_ptr<float>(),
        num_elements
    );
}

// In-place: a_hat_cache += round(clamp(residual * scale, -127, 127)) / scale
void dequant_accumulate_int8(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = residual.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    dequant_accumulate_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        residual.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        scale.data_ptr<float>(),
        num_elements
    );
}

// =========================================================================
// Dynamic per-tensor scale discovery for the plain (non-MoDiff) baseline.
//
// This is the cache-free counterpart of sub_absmax_scale
// (modiff_delta_quantize.cu): it reduces absmax(x) over the whole tensor and
// writes scale = Q_level/max(absmax,eps) and inv_scale = 1/scale, entirely on
// GPU (no host sync), but never reads or writes an a_hat_cache and has no
// notion of a residual — "the residual" for a true baseline is just x itself.
//
// Before this existed, callers that wanted a fused dynamic-scale baseline had
// to fake it by passing a permanently-zero a_hat_cache into the MoDiff
// kernels (sub_absmax_scale / step1_quantize_no_ahat_fprop), which still paid
// for reading and subtracting that zero tensor on every call. This kernel
// does only the work a baseline actually needs.
// =========================================================================
__global__ void absmax_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ absmax_buf,          // Must be 0 on entry (self-resetting)
    float* __restrict__ scale_out,           // Q_level / max(absmax, eps)
    float* __restrict__ inv_scale_out,       // max(absmax, eps) / Q_level (CUTLASS alpha)
    unsigned int* __restrict__ retire_count, // Must be 0 on entry (self-resetting)
    float Q_level,                           // 7.0 for INT4, 127.0 for INT8
    int num_elements
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    float local_max = 0.0f;
    int num_elements4 = num_elements / 4;
    for (int i4 = blockIdx.x * blockDim.x + tid; i4 < num_elements4; i4 += num_threads) {
        float4 x_v = reinterpret_cast<const float4*>(x)[i4];
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(x_v.x), fabsf(x_v.y)),
                                            fmaxf(fabsf(x_v.z), fabsf(x_v.w))));
    }
    int tail_start = num_elements4 * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + tid; i < num_elements; i += num_threads) {
        local_max = fmaxf(local_max, fabsf(x[i]));
    }

    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        float val = sdata[0];
        unsigned int* addr = (unsigned int*)absmax_buf;
        unsigned int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
        } while (assumed != old);

        __threadfence();
        unsigned int ticket = atomicAdd(retire_count, 1u);
        if (ticket == gridDim.x - 1) {
            float absmax = *absmax_buf;
            float am = fmaxf(absmax, 1e-6f);
            *scale_out = Q_level / am;
            *inv_scale_out = am / Q_level;
            *absmax_buf = 0.0f;
            *retire_count = 0;
        }
    }
}

// C++ wrapper: absmax(x) -> scale_out (Q/max), inv_scale_out (max/Q = CUTLASS alpha).
// absmax_buf and retire_count must be persistent, initialized to 0 (kernel self-resets).
void compute_dynamic_scale(
    torch::Tensor x,
    torch::Tensor absmax_buf,      // 1-element float32, init 0
    torch::Tensor scale_out,       // 1-element float32 output
    torch::Tensor inv_scale_out,   // 1-element float32 output
    torch::Tensor retire_count,    // 1-element int32, init 0
    float Q_level
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = x.numel();
    int block_size = 256;
    int grid_size = min((num_elements + block_size - 1) / block_size, 1024);

    absmax_scale_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        absmax_buf.data_ptr<float>(),
        scale_out.data_ptr<float>(),
        inv_scale_out.data_ptr<float>(),
        (unsigned int*)retire_count.data_ptr<int>(),
        Q_level,
        num_elements
    );
}

// Fused compute_dynamic_scale + scale_quantize_int8: plain dynamic INT8
// quantization of x with no MoDiff cache involved anywhere.
torch::Tensor dynamic_quantize_int8_fprop(
    torch::Tensor x,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count
) {
    compute_dynamic_scale(x, absmax_buf, scale_buf, inv_scale_buf, retire_count, 127.0f);
    return scale_quantize_int8(x, scale_buf);
}

// Fused compute_dynamic_scale + scale_quantize_and_pack: plain dynamic INT4
// quantization of x (packed 2-per-byte) with no MoDiff cache involved anywhere.
torch::Tensor dynamic_quantize_pack_int4_fprop(
    torch::Tensor x,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count
) {
    compute_dynamic_scale(x, absmax_buf, scale_buf, inv_scale_buf, retire_count, 7.0f);
    return scale_quantize_and_pack(x, scale_buf);
}
