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
#include <cuda_fp16.h>

// Explicit relative path, NOT bare: a bare include resolves through the global
// -I csrc and would pick csrc/common.cuh -- the un-migrated copy -- making this
// tree's own copy decoration. See csrc/README.md.
#include "../common/common.cuh"

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

// Same as dequant_accumulate_int4_kernel, but also writes the dequantized
// quantize-then-dequantize value to r_dq_out. Used by OptimizedInt4Linear's
// _forward_modulated_fused, which used to compute this exact value itself via
// 4 separate PyTorch ops (mul/round/clamp/mul) purely to feed the FP16 GEMM,
// then call dequant_accumulate_int4 to recompute the identical value *again*
// just to update the cache. This fuses both into the one kernel launch that
// was already doing the work.
__global__ void dequant_accumulate_and_return_int4_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    float* __restrict__ r_dq_out,
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
        float4 dq_v;
        dq_v.x = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.x * scale))) * inv_scale;
        dq_v.y = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.y * scale))) * inv_scale;
        dq_v.z = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.z * scale))) * inv_scale;
        dq_v.w = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.w * scale))) * inv_scale;
        c_v.x += dq_v.x;
        c_v.y += dq_v.y;
        c_v.z += dq_v.z;
        c_v.w += dq_v.w;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
        reinterpret_cast<float4*>(r_dq_out)[idx4] = dq_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            float dq = fmaxf(-7.0f, fminf(7.0f, roundf(r * scale))) * inv_scale;
            a_hat_cache[i] += dq;
            r_dq_out[i] = dq;
        }
    }
}

// INT8 counterpart of dequant_accumulate_and_return_int4_kernel; see that
// kernel's comment for why r_dq_out exists.
__global__ void dequant_accumulate_and_return_int8_kernel(
    const float* __restrict__ residual,
    float* __restrict__ a_hat_cache,
    float* __restrict__ r_dq_out,
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
        float4 dq_v;
        dq_v.x = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.x * scale))) * inv_scale;
        dq_v.y = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.y * scale))) * inv_scale;
        dq_v.z = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.z * scale))) * inv_scale;
        dq_v.w = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.w * scale))) * inv_scale;
        c_v.x += dq_v.x;
        c_v.y += dq_v.y;
        c_v.z += dq_v.z;
        c_v.w += dq_v.w;
        reinterpret_cast<float4*>(a_hat_cache)[idx4] = c_v;
        reinterpret_cast<float4*>(r_dq_out)[idx4] = dq_v;
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            float dq = fmaxf(-127.0f, fminf(127.0f, roundf(r * scale))) * inv_scale;
            a_hat_cache[i] += dq;
            r_dq_out[i] = dq;
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
//   Op:       Quantize (activation int4) + pack
//   Inputs:   input FP32 [N,C,H,W] channels_last-contiguous (NHWC physical);
//             values assumed already pre-scaled into int4 range by the caller
//   Outputs:  INT8 [N,H,W,C/2] (2 int4 nibbles/byte: low=even channel, high=odd)
//   Computes: out = pack(clamp(round(input), -7, 7))  -- no scaling applied here
//   Fuses:    round + clamp + cast + nibble-pack into one kernel launch
//   Constraints: channels_last-contiguous (CHECK_CONTIGUOUS); numel even (2 elems/byte)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
//   Op:       Quantize (activation int4) + pack (static scale)
//   Inputs:   input FP32 [N,C,H,W] channels_last-contiguous (NHWC physical);
//             scale FP32 [1] device scalar (read on-device, no CPU sync)
//   Outputs:  INT8 [N,H,W,C/2] (2 int4 nibbles/byte: low=even channel, high=odd)
//   Computes: out = pack(clamp(round(input * scale), -7, 7))
//   Fuses:    scale-multiply + round + clamp + cast + nibble-pack (removes a
//             separate x*scale kernel launch and its global-memory round-trip)
//   Constraints: channels_last-contiguous (CHECK_CONTIGUOUS); numel even
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
//   Op:       Quantize (activation int8, static scale)
//   Inputs:   input FP32 [any shape]; scale FP32 [1] device scalar (no CPU sync)
//   Outputs:  INT8 [same shape as input]
//   Computes: out = clamp(round(input * scale), -127, 127)
//   Fuses:    scale-multiply + round + clamp + cast into one launch
//   Constraints: none (layout-agnostic, purely elementwise); float4-vectorized with scalar tail
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
//   Op:       MoDiff int4 cache update (dequant + accumulate)
//   Inputs:   residual FP32 [any shape]; a_hat_cache FP32 [same shape] (in-place);
//             scale FP32 [1] device scalar
//   Outputs:  none — a_hat_cache updated in place (FP32 approximation, never raw int4 codes)
//   Computes: a_hat_cache += clamp(round(residual * scale), -7, 7) / scale
//             (quantize the residual to int4 then dequantize before accumulating)
//   Fuses:    mul + round + clamp + div + accumulate into one launch (float4-vectorized)
//   Constraints: residual and a_hat_cache same numel/layout
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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
//   Op:       MoDiff int8 cache update (dequant + accumulate)
//   Inputs:   residual FP32 [any shape]; a_hat_cache FP32 [same shape] (in-place);
//             scale FP32 [1] device scalar
//   Outputs:  none — a_hat_cache updated in place (FP32 approximation, never raw int8 codes)
//   Computes: a_hat_cache += clamp(round(residual * scale), -127, 127) / scale
//   Fuses:    mul + round + clamp + div + accumulate into one launch (float4-vectorized)
//   Constraints: residual and a_hat_cache same numel/layout
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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

// In-place: a_hat_cache += dq; r_dq_out = dq, where dq = round(clamp(residual * scale, -7, 7)) / scale.
//   Op:       MoDiff int4 cache update (dequant + accumulate), also returning dq
//   Inputs:   residual FP32 [any shape]; a_hat_cache FP32 [same shape] (in-place);
//             scale FP32 [1] device scalar; r_dq_out FP32 [same shape] (output)
//   Outputs:  none returned — a_hat_cache += dq in place; r_dq_out = dq
//   Computes: dq = clamp(round(residual * scale), -7, 7) / scale; a_hat_cache += dq; r_dq_out = dq
//   Fuses:    computes the dequantized residual once and feeds it to BOTH the cache
//             update and the caller's FP16 GEMM, avoiding a redundant re-quantize pass
//             (see OptimizedInt4Linear._forward_modulated_fused)
//   Constraints: residual, a_hat_cache, r_dq_out same numel/layout
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
void dequant_accumulate_and_return_int4(torch::Tensor residual, torch::Tensor a_hat_cache,
                                         torch::Tensor scale, torch::Tensor r_dq_out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = residual.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    dequant_accumulate_and_return_int4_kernel<<<grid_size, block_size, 0, stream>>>(
        residual.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        r_dq_out.data_ptr<float>(),
        scale.data_ptr<float>(),
        num_elements
    );
}

// In-place: a_hat_cache += dq; r_dq_out = dq, where dq = round(clamp(residual * scale, -127, 127)) / scale.
//   Op:       MoDiff int8 cache update (dequant + accumulate), also returning dq
//   Inputs:   residual FP32 [any shape]; a_hat_cache FP32 [same shape] (in-place);
//             scale FP32 [1] device scalar; r_dq_out FP32 [same shape] (output)
//   Outputs:  none returned — a_hat_cache += dq in place; r_dq_out = dq
//   Computes: dq = clamp(round(residual * scale), -127, 127) / scale; a_hat_cache += dq; r_dq_out = dq
//   Fuses:    computes the dequantized residual once and feeds both the cache update
//             and the caller's FP16 GEMM (int8 counterpart of the int4 return variant)
//   Constraints: residual, a_hat_cache, r_dq_out same numel/layout
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
void dequant_accumulate_and_return_int8(torch::Tensor residual, torch::Tensor a_hat_cache,
                                         torch::Tensor scale, torch::Tensor r_dq_out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = residual.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    dequant_accumulate_and_return_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        residual.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        r_dq_out.data_ptr<float>(),
        scale.data_ptr<float>(),
        num_elements
    );
}

// compute_dynamic_scale / dynamic_quantize_int8_fprop / dynamic_quantize_pack_int4_fprop
// (the cache-free baseline dynamic-scale path) moved to modiff_delta_quantize.cu:
// they're now implemented via sub_absmax_scale_kernel with a_hat_cache=nullptr,
// residual=nullptr, instead of this file's own near-duplicate absmax_scale_kernel.

// =========================================================================
// Fused attention-output transpose + int8 quantize (proj-side quantize fusion).
//
// The token-major AttentionBlock produces the attention output head-major as
// a = [b, nh, T, hd], then MUST run `a.transpose(1,2).reshape(b,T,C)` — a
// physical strided copy — before the proj Linear, which then runs its OWN
// elementwise quantize pass over [b*T, C]. That is two full O(b*T*C) memory
// passes (layout copy + quantize). This kernel fuses them into ONE: it gathers
// the head-major input into token-major order AND quantizes to int8 in a single
// pass, emitting int8 [b*T, C] directly for gemm_w8a8_awq. Bit-identical to
// quantize_act_int8(a.transpose(1,2).reshape(b,T,C).contiguous(), a_scale).
//   out[b*T + t, h*hd + d] = clamp(round(a[b,h,t,d] / a_scale), -127, 127)
// Requires C = nh*hd to be a multiple of 64 (the W8A8 GEMM K-tile), which holds
// for every attention width here (192/384/768) so no K-pad is needed.
// =========================================================================
__global__ void quant_attn_out_int8_kernel(const __half* __restrict__ a, int8_t* __restrict__ out,
                                           float inv_scale, int nh, int T, int hd, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;   // one output element in [b, T, C] order
  if (i >= n) return;
  int C = nh * hd;
  int col = (int)(i % C);          // = h*hd + d  (token-major channel)
  long row = i / C;                // = b*T + t
  int t = (int)(row % T);
  int bb = (int)(row / T);
  int h = col / hd;
  int d = col - h * hd;
  long in_idx = (((long)bb * nh + h) * T + t) * hd + d;   // a[b, h, t, d], head-major
  int q = __float2int_rn(__half2float(a[in_idx]) * inv_scale);
  out[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
}

// Fused dequant + per-column bias for an int8-output GEMM (gemm_w8a8_awq_out_i8): one pass,
//   out[m,n] = in_i8[m,n] * out_scale[n] + bias[n]  (fp16 out).
// Lets the int8-output Linear fold its dequant into what was the standalone +bias add, so the epilogue
// stays one op (read int8 M*N + write fp16 M*N) instead of an extra fp16 round-trip.
__global__ void dequant_bias_i8_kernel(const int8_t* __restrict__ in, const float* __restrict__ out_scale,
                                       const __half* __restrict__ bias, __half* __restrict__ out,
                                       int N, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int col = (int)(i % N);
  out[i] = __float2half((float)in[i] * out_scale[col] + __half2float(bias[col]));
}

// Host wrapper for dequant_bias_i8_kernel (see above for the fusion rationale).
//   Op:       Dequant + per-column bias (int8-output GEMM epilogue)
//   Inputs:   in INT8 [M,N] (int8-GEMM output); out_scale FP32 [N] per-column
//             dequant scale; bias FP16 [N] per-column bias
//   Outputs:  FP16 [M,N]
//   Computes: out[m,n] = in[m,n] * out_scale[n] + bias[n]
//   Fuses:    dequant + bias-add in one pass (folds the int8-output Linear's dequant
//             into what was the standalone +bias, avoiding an extra fp16 round-trip)
//   Constraints: `in` and `out_scale` made contiguous internally
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor dequant_bias_i8(torch::Tensor in, torch::Tensor out_scale, torch::Tensor bias) {
  in = in.contiguous();
  int M = in.size(0), N = in.size(1); long n = (long)M * N;
  auto out = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(in.device()));
  int T = 256; long blocks = (n + T - 1) / T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  dequant_bias_i8_kernel<<<blocks, T, 0, s>>>(
      in.data_ptr<int8_t>(), out_scale.contiguous().data_ptr<float>(),
      reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), N, n);
  return out;
}

// a: [b, nh, T, hd] fp16 (attention output). Returns int8 [b*T, nh*hd] token-major.
//   Op:       Quantize (attention output int8) + head-major->token-major transpose
//   Inputs:   a FP16 [b, nh, T, hd] (attention output, head-major); a_scale double (per-tensor)
//   Outputs:  INT8 [b*T, nh*hd] token-major (C = nh*hd), ready for gemm_w8a8_awq
//   Computes: out[b*T+t, h*hd+d] = clamp(round(a[b,h,t,d] / a_scale), -127, 127)
//   Fuses:    the transpose-copy (a.transpose(1,2).reshape) + a separate int8 quantize
//             into ONE gather-and-quantize pass; bit-identical to
//             quantize_act_int8(a.transpose(1,2).reshape(b,T,C).contiguous(), a_scale)
//   Constraints: C = nh*hd must be a multiple of 64 (W8A8 GEMM K-tile); holds for
//                widths 192/384/768 so no K-pad is needed. `a` made contiguous internally
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor quantize_attn_out_int8(torch::Tensor a, double a_scale) {
  a = a.contiguous();
  int b = a.size(0), nh = a.size(1), T = a.size(2), hd = a.size(3);
  int C = nh * hd;
  long n = (long)b * T * C;
  auto out = torch::empty({(long)b * T, C}, torch::TensorOptions().dtype(torch::kChar).device(a.device()));
  int TH = 256; long blocks = (n + TH - 1) / TH;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  quant_attn_out_int8_kernel<<<blocks, TH, 0, s>>>(
      reinterpret_cast<const __half*>(a.data_ptr<at::Half>()), out.data_ptr<int8_t>(),
      1.f / (float)a_scale, nh, T, hd, n);
  return out;
}

// int4 variant of quantize_attn_out_int8: fused transpose + int4 quantize + pack (2 channels/byte,
// gemm_w4a4 layout) for the proj-side fusion. a[b,nh,T,hd] fp16 -> int8[b*T, C/2] packed int4
// (low nibble = even channel, high = odd), matching quantize_act_int4_pack. hd is even, so a
// (c, c+1) pair never crosses a head boundary. C = nh*hd must be even (always here).
__global__ void quant_attn_out_int4_pack_kernel(const __half* __restrict__ a, int8_t* __restrict__ out,
                                                float inv_scale, int nh, int T, int hd, int Kpad, long nout) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;   // one output byte in [b, T, Kpad/2] order
  if (i >= nout) return;
  int C = nh * hd, Kh = Kpad / 2;  // byte-cols/row = Kpad/2 (padded); real channels = C
  int bc = (int)(i % Kh);          // byte-col
  int c0 = 2 * bc, c1 = c0 + 1;
  if (c0 >= C) { out[i] = 0; return; }   // K-pad byte (both channels >= C): zero-fill. c0 even & C
                                         // even -> c0>=C implies c1>=C, so never a half-real byte.
  long row = i / Kh;               // = b*T + t
  int t = (int)(row % T), bb = (int)(row / T);
  int h0 = c0 / hd, d0 = c0 - h0 * hd, h1 = c1 / hd, d1 = c1 - h1 * hd;
  long i0 = (((long)bb * nh + h0) * T + t) * hd + d0;
  long i1 = (((long)bb * nh + h1) * T + t) * hd + d1;
  int q0 = __float2int_rn(__half2float(a[i0]) * inv_scale); q0 = q0 > 7 ? 7 : (q0 < -7 ? -7 : q0);
  int q1 = __float2int_rn(__half2float(a[i1]) * inv_scale); q1 = q1 > 7 ? 7 : (q1 < -7 ? -7 : q1);
  out[i] = (int8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
}

// Host wrapper for the int4 attention-output quantize+pack (see kernel comment above).
//   Op:       Quantize (attention output int4) + transpose + pack
//   Inputs:   a FP16 [b, nh, T, hd] (attention output, head-major); a_scale double (per-tensor)
//   Outputs:  INT8 [b*T, C/2] packed int4 token-major (low nibble=even channel, high=odd),
//             matching quantize_act_int4_pack / the gemm_w4a4 layout
//   Computes: q = clamp(round(a[b,h,t,d] / a_scale), -7, 7); pack channel pairs per byte
//   Fuses:    head-major->token-major transpose + int4 quantize + nibble-pack in one pass
//   k_pad:    padded output channel count K (>= C, even); output is [b*T, k_pad/2] with real channels
//             0..C-1 packed and C..k_pad-1 zero-filled — lets the fused int4 proj GEMM
//             (gemm_w4a4_awq_bias_res, which takes an explicit padded K) read this directly with no
//             fp16 F.pad copy (enables the dominant int4 C=192 -> K=256 attention block). k_pad<=0 -> C.
//   Constraints: hd even (a (c,c+1) pair never crosses a head boundary); C = nh*hd even
//                (always here); k_pad even and >= C. `a` made contiguous internally
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor quantize_attn_out_int4_pack(torch::Tensor a, double a_scale, int64_t k_pad) {
  a = a.contiguous();
  int b = a.size(0), nh = a.size(1), T = a.size(2), hd = a.size(3);
  int C = nh * hd;
  int Kpad = (k_pad > 0) ? (int)k_pad : C;
  TORCH_CHECK(Kpad % 2 == 0 && Kpad >= C, "quantize_attn_out_int4_pack: k_pad must be even and >= C=nh*hd");
  long nout = (long)b * T * (Kpad / 2);
  auto out = torch::empty({(long)b * T, Kpad / 2}, torch::TensorOptions().dtype(torch::kChar).device(a.device()));
  int TH = 256; long blocks = (nout + TH - 1) / TH;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  quant_attn_out_int4_pack_kernel<<<blocks, TH, 0, s>>>(
      reinterpret_cast<const __half*>(a.data_ptr<at::Half>()), out.data_ptr<int8_t>(),
      1.f / (float)a_scale, nh, T, hd, Kpad, nout);
  return out;
}
