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
    int num_elements, // Number of OUTPUT packed bytes
    // ACTIVATION ZERO POINT (plan fix #2), same contract as
    // group_norm_silu_quantize_pack_nhwc_vec2_kernel: a_q = clamp(round(a*s) + z, -7, 7), and the
    // dequantization's -z*sum(w_q) term is folded into the conv bias at calibration time
    // (OptimizedInt4Conv2d._refold_zp_bias), so no GEMM or epilogue sees z.
    //
    // THIS SITE IS WHY THE FIRST END-TO-END ZERO-POINT RUN DIVERGED (relL2 7.3057 on the MoDiff
    // arm). It is MoDiff's t=T entry point -- _forward_first_step -> _int4_conv -> here -- and it
    // is the ONE quantize per layer per sample whose conv actually adds the corrected bias. It was
    // not merely unimplemented: it was UNGUARDED, so the census in
    // docs/zero_point_2026-08-13/FINDINGS.md never listed it, and the 62 delta-path sites it did
    // list are z-free by construction. o_hat is then accumulated over every remaining step, so a
    // t=T offset of z*sum(w_q)*ws/s per output channel never washes out.
    // Measured and re-classified in docs/zp_coverage_2026-08-13/data/site_census.json.
    //
    // z == 0 reproduces the old kernel EXACTLY (+0.0f before the same round/clamp).
    float zp = 0.0f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float scale = *scale_ptr;
    float2 vals = reinterpret_cast<const float2*>(input)[idx];
    float v0 = vals.x * scale;
    float v1 = vals.y * scale;

    int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v0) + zp));
    int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(v1) + zp));

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
static torch::Tensor scale_quantize_and_pack_impl(torch::Tensor input, torch::Tensor scale,
                                                  double zero_point) {
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
        num_output,
        (float)zero_point
    );

    // NO 2D BRANCH, unlike step1_static_quantize_pack_int4_fprop: CHECK_CONTIGUOUS above requires
    // channels_last, which a 2D tensor can never satisfy, so a [M, K] reshape here would be dead
    // code. The linear path reaches the step1_* kernel instead.
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    return output.view({N, H, W, C / 2});
}

// TWO ARITIES, not a C++ default argument -- pybind11 does not inherit defaults, and this keeps
// every existing 2-argument caller (integration/ plus ~8 archived docs/*/scripts) untouched. Same
// pattern as group_norm_silu_quantize_pack_nhwc{,_zp}.
torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale) {
    return scale_quantize_and_pack_impl(input, scale, 0.0);
}

torch::Tensor scale_quantize_and_pack_zp(torch::Tensor input, torch::Tensor scale,
                                         double zero_point) {
    return scale_quantize_and_pack_impl(input, scale, zero_point);
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

// ================================================================================================
// COPIED from csrc/modiff/quantize/delta_quantize.cu (family 1b, 2026-08-12). These are also used
// by MoDiff-side code that stays in that file, so they are duplicated here rather than shared
// across trees -- see csrc/README.md. The host function and the __global__ kernel are marked
// `static` so they cannot collide with the exported originals at link time; the two are otherwise
// byte-identical to their twins and should stay that way.
// ================================================================================================

// COPY of load_as_float
__device__ __forceinline__ float load_as_float(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float load_as_float(const __half* p, int i) { return __half2float(p[i]); }

// COPY of load_as_float2
__device__ __forceinline__ float2 load_as_float2(const float* p, int i) {
    return reinterpret_cast<const float2*>(p)[i >> 1];
}
__device__ __forceinline__ float2 load_as_float2(const __half* p, int i) {
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}

// COPY of avgpool4_as_stored
// Bit-exact reproduction of ATen's avg_pool2d(kernel_size=2, stride=2, padding=0) forward for a
// single 2x2 non-overlapping window: sums the 4 window elements as float in (kh,kw) row-major
// order, scales by 0.25f, then -- ONLY when T_IN is half-precision storage -- rounds that sum
// through __half (round-to-nearest-even, matching what avg_pool2d actually writes to its fp16
// output tensor) before handing it back out as float. Verified against a real
// nn.AvgPool2d(2,2).cuda() fp16 forward: this reproduces it bit-for-bit (torch.equal), not just
// approximately, which is the same bar every other kernel in this file is held to.
template <typename T_IN> __device__ __forceinline__ float avgpool4_as_stored(float sum);
template <> __device__ __forceinline__ float avgpool4_as_stored<__half>(float sum) {
    return __half2float(__float2half_rn(sum));
}
template <> __device__ __forceinline__ float avgpool4_as_stored<float>(float sum) { return sum; }

// COPY of sub_absmax_scale_kernel
// Fused: residual = x - a_hat_cache (with optional SmoothQuant), absmax reduction
// over the whole tensor, then scale = Q_level/max(absmax,eps) and its inverse.
// Replaces 6 separate kernel launches (sub, abs, amax, clamp, div, div) with one
// cooperative kernel using atomic float-max + a block-retirement counter to elect
// a single "last" block to finalize the scale.
//
// a_hat_cache and residual are independently nullable, subsuming what used to be
// a separate near-duplicate kernel (quantize.cu's absmax_scale_kernel) for the
// cache-free baseline case:
//   a_hat_cache == nullptr -> no cache to subtract, "residual" is just x itself
//                             (still smoothed if smooth_inv is set)
//   residual == nullptr    -> pure reduction, nothing to materialize (this is
//                             exactly what the baseline path needs: it only wants
//                             scale/inv_scale, and quantizing straight from x
//                             costs nothing extra since x is already resident)
// Both null is the baseline case (compute_dynamic_scale); both non-null is the
// original MoDiff dynamic case (sub_absmax_scale).
static __global__ void sub_absmax_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ a_hat_cache,   // nullptr => baseline, no cache to subtract
    float* __restrict__ residual,            // nullptr => don't materialize, reduction only
    float* __restrict__ absmax_buf,     // Must be 0 on entry (self-resetting)
    float* __restrict__ scale_out,      // Q_level / max(absmax, eps)
    float* __restrict__ inv_scale_out,  // max(absmax, eps) / Q_level  (CUTLASS alpha)
    unsigned int* __restrict__ retire_count, // Must be 0 on entry (self-resetting)
    const float* __restrict__ smooth_inv,  // Per-channel SmoothQuant inverse (NULL=skip)
    int num_channels,                    // C for NHWC channel indexing
    float Q_level,                      // 7.0 for INT4, 127.0 for INT8
    int num_elements
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    // Step 1: Compute residual (with optional SmoothQuant) and find thread-local abs max
    float local_max = 0.0f;
    int num_elements4 = num_elements / 4;
    // Vectorized loop: process 4 elements at a time
    for (int i4 = blockIdx.x * blockDim.x + tid; i4 < num_elements4; i4 += num_threads) {
        float4 x_v = reinterpret_cast<const float4*>(x)[i4];
        // Fused SmoothQuant: x_smooth = x * smooth_inv[channel]
        if (smooth_inv != nullptr) {
            int ch = (i4 * 4) % num_channels;
            x_v.x *= smooth_inv[ch];
            x_v.y *= smooth_inv[ch + 1];
            x_v.z *= smooth_inv[ch + 2];
            x_v.w *= smooth_inv[ch + 3];
        }
        float4 r_v;
        if (a_hat_cache != nullptr) {
            float4 c_v = reinterpret_cast<const float4*>(a_hat_cache)[i4];
            r_v.x = x_v.x - c_v.x;
            r_v.y = x_v.y - c_v.y;
            r_v.z = x_v.z - c_v.z;
            r_v.w = x_v.w - c_v.w;
        } else {
            r_v = x_v;
        }
        if (residual != nullptr) {
            reinterpret_cast<float4*>(residual)[i4] = r_v;
        }
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(r_v.x), fabsf(r_v.y)),
                                            fmaxf(fabsf(r_v.z), fabsf(r_v.w))));
    }
    // Scalar tail for remaining elements
    int tail_start = num_elements4 * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + tid; i < num_elements; i += num_threads) {
        float xval = x[i];
        if (smooth_inv != nullptr) {
            xval *= smooth_inv[i % num_channels];
        }
        float r = (a_hat_cache != nullptr) ? (xval - a_hat_cache[i]) : xval;
        if (residual != nullptr) {
            residual[i] = r;
        }
        local_max = fmaxf(local_max, fabsf(r));
    }

    // Step 2: Block-level reduction in shared memory
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        // Step 3: Atomic float max across blocks (CAS-based)
        float val = sdata[0];
        unsigned int* addr = (unsigned int*)absmax_buf;
        unsigned int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
        } while (assumed != old);

        // Step 4: Count completed blocks; last block computes + writes scale
        __threadfence();  // Ensure atomicCAS is visible before counting
        unsigned int ticket = atomicAdd(retire_count, 1u);
        if (ticket == gridDim.x - 1) {
            float absmax = *absmax_buf;
            float am = fmaxf(absmax, 1e-6f);
            *scale_out = Q_level / am;
            *inv_scale_out = am / Q_level;
            // Self-reset for next invocation (safe: next kernel starts after this one)
            *absmax_buf = 0.0f;
            *retire_count = 0;
        }
    }
}

// COPY of sub_absmax_scale
// C++ wrapper: performs subtraction + absmax reduction + scale computation in one kernel.
// Outputs: residual, scale_out (Q/max), inv_scale_out (max/Q = CUTLASS alpha).
// absmax_buf and retire_count must be persistent, initialized to 0 (kernel self-resets).
//   Op:       MoDiff temporal-delta residual + absmax + dynamic-scale (shared reduction)
//   Inputs:   x FP32 [N,C,H,W]; a_hat_cache FP32 [same] MoDiff cache; residual FP32 [same]
//             out buffer; absmax_buf FP32 [1] (init 0, self-reset); scale_out FP32 [1] out;
//             inv_scale_out FP32 [1] out; retire_count int32 [1] (init 0, self-reset);
//             Q_level float (7.0 int4 / 127.0 int8); smooth_inv FP32 [C] per-channel
//             SmoothQuant inverse (empty = skip)
//   Outputs:  residual = (x*smooth_inv) - a_hat_cache; scale_out = Q_level/max(absmax,1e-6);
//             inv_scale_out = max(absmax,1e-6)/Q_level (CUTLASS alpha)
//   Computes: temporal delta of x against the cache, then the per-tensor dynamic scale
//             from the delta's absmax
//   Fuses:    6 elementwise launches (sub, abs, amax, clamp, div, div) into one cooperative
//             kernel (atomic float-max + block-retirement counter elects the last block to
//             finalize the scale); no host sync
//   Constraints: absmax_buf & retire_count must be 0 on entry (self-resetting); with
//                smooth_inv the float4 quads assume num_channels % 4 == 0
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
static void sub_absmax_scale(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual,
    torch::Tensor absmax_buf,      // 1-element float32, init 0
    torch::Tensor scale_out,       // 1-element float32 output
    torch::Tensor inv_scale_out,   // 1-element float32 output
    torch::Tensor retire_count,    // 1-element int32, init 0
    float Q_level,
    torch::Tensor smooth_inv       // Per-channel SmoothQuant inverse (empty = skip)
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_elements = x.numel();
    int block_size = 256;
    int grid_size = min((num_elements + block_size - 1) / block_size, 1024);

    // Determine smooth_inv pointer and num_channels
    const float* smooth_ptr = nullptr;
    int num_channels = 0;
    if (smooth_inv.numel() > 0) {
        smooth_ptr = smooth_inv.data_ptr<float>();
        num_channels = smooth_inv.numel();
    }

    sub_absmax_scale_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        a_hat_cache.data_ptr<float>(),
        residual.data_ptr<float>(),
        absmax_buf.data_ptr<float>(),
        scale_out.data_ptr<float>(),
        inv_scale_out.data_ptr<float>(),
        (unsigned int*)retire_count.data_ptr<int>(),
        smooth_ptr,
        num_channels,
        Q_level,
        num_elements
    );
}

// ================================================================================================
// Family 1b of the csrc/ datapath split (2026-08-12). Moved out of
// csrc/modiff/quantize/delta_quantize.cu -- every function below is a *_noahat / *_no_ahat variant
// that carries no PERSISTED MoDiff state (it never writes a_hat back), so it belongs to the
// baseline datapath. Two of them (step1_quantize_no_ahat_fprop,
// step1_quantize_pack_int4_no_ahat_fprop) still READ an a_hat_cache argument to form a residual --
// see the COPY block below for how that dependency is resolved.
// ================================================================================================

// Same quantization as above, but does NOT touch a_hat_cache. Used by the
// "_no_ahat" step1 wrappers where the caller manages the cache update itself
// (or intentionally skips it, e.g. for a one-off/no-caching baseline run).
__global__ void quantize_only_int8_kernel(
    const float* __restrict__ residual,
    int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];

        float q0 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.x * scale)));
        float q1 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.y * scale)));
        float q2 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.z * scale)));
        float q3 = fmaxf(-127.0f, fminf(127.0f, roundf(r_v.w * scale)));

        reinterpret_cast<int32_t*>(output_int8)[idx4] =
            ((unsigned char)(int8_t)q0) | ((unsigned char)(int8_t)q1 << 8) |
            ((unsigned char)(int8_t)q2 << 16) | ((unsigned char)(int8_t)q3 << 24);
    } else {
        for (int i = base; i < num_elements; i++) {
            float r = residual[i];
            float q = fmaxf(-127.0f, fminf(127.0f, roundf(r * scale)));
            output_int8[i] = (int8_t)q;
        }
    }
}

// INT4 variant of quantize_only_int8_kernel: quantize + pack, no cache update.
__global__ void quantize_pack_only_kernel_int4(
    const float* __restrict__ residual,
    int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr,
    int num_elements
) {
    float scale = *scale_ptr;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (base + 3 < num_elements) {
        float4 r_v = reinterpret_cast<const float4*>(residual)[idx4];

        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.x * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.y * scale)));
        float q2 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.z * scale)));
        float q3 = fmaxf(-7.0f, fminf(7.0f, roundf(r_v.w * scale)));

        int8_t b0 = ((int8_t)q0 & 0x0F) | (((int8_t)q1 & 0x0F) << 4);
        int8_t b1 = ((int8_t)q2 & 0x0F) | (((int8_t)q3 & 0x0F) << 4);
        reinterpret_cast<int16_t*>(output_packed)[idx4] = (uint16_t)(uint8_t)b0 | ((uint16_t)(uint8_t)b1 << 8);
    } else {
        for (int i = base; i < num_elements; i += 2) {
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(residual[i] * scale)));
            float q1 = 0.0f;
            if (i + 1 < num_elements) {
                q1 = fmaxf(-7.0f, fminf(7.0f, roundf(residual[i + 1] * scale)));
            }
            output_packed[i / 2] = (((int8_t)q0 & 0x0F) | ((((int8_t)q1 & 0x0F) << 4)));
        }
    }
}

// ---- TRUE cache-free static quantize (baseline, NO a_hat): same as the *_update_ahat kernels
// above but without the a_hat read/subtract/write. Baseline conv doesn't do temporal caching, so
// residual = x - a_hat collapses to x; dropping a_hat removes the a_hat read+write + the zero-fill.
template <typename T_IN>
__global__ void static_quantize_int8_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float xval = load_as_float(x, i);
        if (smooth_inv != nullptr) xval *= smooth_inv[i % num_channels];
        float q = fmaxf(-127.0f, fminf(127.0f, roundf(xval * scale)));
        output_int8[i] = static_cast<int8_t>(q);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_int8_noahat_kernel. Pair-major
// grid-stride loop with an inline single-element epilogue for odd num_elements, same technique
// as static_quantize_and_update_ahat_kernel_int8_half_cache_vec2 above. Requires
// num_channels % 2 == 0 for the smooth_inv fast path (not a new risk -- same argument as that
// kernel: base is always even here, so base % num_channels is even too when num_channels is
// even, so a pair never straddles a channel boundary); the caller only dispatches here when
// that holds. Packs the 2 output int8 codes into one int16_t store.
template <typename T_IN>
__global__ void static_quantize_int8_noahat_vec2_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    const int stride = blockDim.x * gridDim.x;

    for (int base = 2 * (blockIdx.x * blockDim.x + threadIdx.x); base < num_elements; base += 2 * stride) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            int8_t q0 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(xv.x * scale)));
            int8_t q1 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(xv.y * scale)));
            reinterpret_cast<int16_t*>(output_int8)[base >> 1] =
                (int16_t)(((unsigned char)q0) | (((unsigned char)q1) << 8));
        } else {
            float xval = load_as_float(x, base);
            if (smooth_inv != nullptr) xval *= smooth_inv[base % num_channels];
            float q = fmaxf(-127.0f, fminf(127.0f, roundf(xval * scale)));
            output_int8[base] = static_cast<int8_t>(q);
        }
    }
}

template <typename T_IN>
__global__ void static_quantize_pack_int4_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        float x0 = load_as_float(x, base);
        if (smooth_inv != nullptr) x0 *= smooth_inv[base % num_channels];
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(x0 * scale)));
        float q1 = 0.0f;
        if (base + 1 < num_elements) {
            float x1 = load_as_float(x, base + 1);
            if (smooth_inv != nullptr) x1 *= smooth_inv[(base + 1) % num_channels];
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf(x1 * scale)));
        }
        output_packed[base / 2] =
            (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Vectorized (half2/float2) counterpart of static_quantize_pack_int4_noahat_kernel. The scalar
// kernel above is already pair-major (base = idx*2); this only widens the per-pair load from
// two independent scalar accesses to one vectorized half2/float2 access -- same technique as
// static_quantize_pack_and_update_ahat_kernel_int4_half_cache_vec2 above. Same num_channels % 2
// == 0 gate rationale.
template <typename T_IN>
__global__ void static_quantize_pack_int4_noahat_vec2_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int num_channels, int num_elements) {
    float scale = *scale_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int base = idx * 2; base < num_elements; base += stride * 2) {
        if (base + 1 < num_elements) {
            float2 xv = load_as_float2(x, base);
            if (smooth_inv != nullptr) {
                int c0 = base % num_channels;
                float2 sm = *reinterpret_cast<const float2*>(&smooth_inv[c0]);
                xv.x *= sm.x; xv.y *= sm.y;
            }
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(xv.x * scale)));
            float q1 = fmaxf(-7.0f, fminf(7.0f, roundf(xv.y * scale)));
            output_packed[base / 2] =
                (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
        } else {
            float x0 = load_as_float(x, base);
            if (smooth_inv != nullptr) x0 *= smooth_inv[base % num_channels];
            float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(x0 * scale)));
            output_packed[base / 2] = static_cast<int8_t>(q0) & 0x0F;
        }
    }
}

// Dynamic INT8, but a_hat_cache is left untouched (caller manages the cache itself).
//   Op:       MoDiff temporal-delta quantize (dynamic int8), NO cache write — benchmark-only
//   Inputs:   same as step1_quantize_fprop (x, a_hat_cache, residual_buf, absmax_buf,
//             scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv)
//   Outputs:  INT8 [same shape as x]; a_hat_cache NOT modified
//   Computes: residual = (x*smooth_inv) - a_hat_cache; scale = 127/absmax(residual);
//             out = clamp(round(residual*scale), -127, 127)
//   Fuses:    sub_absmax_scale + quantize (skips the a_hat write to isolate its cost in
//             microbenchmarks). Still READS a_hat_cache to form the residual — NOT a
//             cache-free substitute; for that use dynamic_quantize_int8_fprop
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_quantize_no_ahat_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);

    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    quantize_only_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        residual_buf.data_ptr<float>(),
        x_int8.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_elements
    );

    return x_int8;
}

// Dynamic INT4 without a_hat_cache update (caller manages the cache itself).
//   Op:       MoDiff temporal-delta quantize (dynamic int4) + pack, NO cache write — benchmark-only
//   Inputs:   same as step1_quantize_pack_int4_fprop (x, a_hat_cache, residual_buf, absmax_buf,
//             scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv)
//   Outputs:  INT8 [N,H,W,C/2] packed int4; a_hat_cache NOT modified
//   Computes: residual = (x*smooth_inv) - a_hat_cache; scale = 7/absmax(residual);
//             out = pack(clamp(round(residual*scale), -7, 7))
//   Fuses:    sub_absmax_scale + quantize + nibble-pack (skips the a_hat write). Still READS
//             a_hat_cache — NOT a cache-free substitute; for that use dynamic_quantize_pack_int4_fprop
//   Constraints: absmax_buf & retire_count 0 on entry (self-reset)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor step1_quantize_pack_int4_no_ahat_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sub_absmax_scale(x, a_hat_cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count, Q_level, smooth_inv);

    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;

    quantize_pack_only_kernel_int4<<<grid_size, block_size, 0, stream>>>(
        residual_buf.data_ptr<float>(),
        x_packed.data_ptr<int8_t>(),
        scale_buf.data_ptr<float>(),
        num_input
    );

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    return x_packed.view({N, H, W, C / 2});
}

// Cache-free static int8 quantize (baseline conv): fp16/fp32 x + static scale (+optional smooth),
// NO a_hat. Bit-identical output to step1_static_quantize_fprop(x, a_hat=0, scale) but without the
// a_hat read/write + the caller's per-call a_hat zero-fill. Output int8, same shape/layout as x.
torch::Tensor step1_static_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto x_int8 = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    int num_elements = x.numel();
    int block_size = 256;
    int num_work_items = (num_elements + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : (int)x.size(1);
    // Vec2 fast path requires num_channels % 2 == 0 for the smooth_inv half2 load -- see
    // step1_static_quantize_fprop's identical gate. Right-size the grid for the 2-wide step.
    const bool use_vec2 = (num_channels % 2 == 0);
    const int num_work_items_vec2 = (num_elements + 1) / 2;
    const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
    if (x.scalar_type() == torch::kHalf) {
        if (use_vec2) {
            static_quantize_int8_noahat_vec2_kernel<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        } else {
            static_quantize_int8_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        }
    } else {
        if (use_vec2) {
            static_quantize_int8_noahat_vec2_kernel<float><<<grid_size_vec2, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        } else {
            static_quantize_int8_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_int8.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_elements);
        }
    }
    return x_int8;
}

// Cache-free static int4 quantize+pack (baseline conv): fp16/fp32 x + static scale (+optional
// smooth), NO a_hat. Output packed int8 [N,H,W,C/2] (same layout as step1_static_quantize_pack_int4_fprop).
torch::Tensor step1_static_quantize_pack_int4_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int num_input = x.numel();
    int num_output = num_input / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    int block_size = 256;
    int num_work_items = (num_input + 3) / 4;
    int grid_size = (num_work_items + block_size - 1) / block_size;
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : (int)x.size(1);
    // See step1_static_quantize_pack_int4_fprop for the use_vec2 gate rationale.
    const bool use_vec2 = (num_channels % 2 == 0);
    const int num_work_items_vec2 = (num_input + 1) / 2;
    const int grid_size_vec2 = (num_work_items_vec2 + block_size - 1) / block_size;
    if (x.scalar_type() == torch::kHalf) {
        if (use_vec2) {
            static_quantize_pack_int4_noahat_vec2_kernel<__half><<<grid_size_vec2, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        } else {
            static_quantize_pack_int4_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        }
    } else {
        if (use_vec2) {
            static_quantize_pack_int4_noahat_vec2_kernel<float><<<grid_size_vec2, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        } else {
            static_quantize_pack_int4_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
                x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
                scale_buf.data_ptr<float>(), smooth_ptr, num_channels, num_input);
        }
    }
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    return x_packed.view({N, H, W, C / 2});
}

// ---- Upsample(nearest, 2x) + static quantize fusion (baseline conv, NO a_hat): fold the
// Upsample.forward -> F.interpolate(scale_factor=2, mode="nearest") pass into the following
// conv's quantize prologue. Nearest 2x upsample repeats each source pixel into a 2x2 block, so
// this reads the SMALL pre-upsample [N,C,H,W] tensor once and writes the LARGE quantized
// [N,C,2H,2W] output directly with 2x2-repeat addressing -- the intermediate fp16
// [N,C,2H,2W] upsampled tensor (what F.interpolate would materialize, then the plain
// static_quantize_*_noahat_kernel would re-read) is never allocated. Per-element quantize math
// is copy-pasted from static_quantize_int8_noahat_kernel / static_quantize_pack_int4_noahat_kernel
// (same scale/smooth_inv/clamp), just reading through the repeat-by-2 address instead of `x[i]`
// directly -- so output is bit-identical to F.interpolate(x) -> static_quantize_*_noahat_kernel.
template <typename T_IN>
__global__ void upsample2x_quantize_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out,
    // MoDiff, opt-in: subtract a_hat before quantizing and advance it in place. nullptr keeps the
    // baseline behaviour BIT-IDENTICALLY (cache reads as 0 and nothing is stored), so one kernel
    // serves both modes -- no cloned MoDiff twin. The loop already grids over OUTPUT elements,
    // which is what makes this work: nearest-2x upsample gives each output position its own a_hat
    // entry, and a clone gridding over input positions would have had to quantize four times.
    __half* __restrict__ a_hat_cache) {
    float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const int Wo = W * 2, Ho = H * 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long i = idx; i < num_elements_out; i += (long)blockDim.x * gridDim.x) {
        int c = (int)(i % C);
        long rest = i / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho >> 1, wi = wo >> 1;
        long i_in = ((n * H + hi) * (long)W + wi) * C + c;
        float xval = load_as_float(x, (int)i_in);
        if (smooth_inv != nullptr) xval *= smooth_inv[c % num_channels];
        const float cache = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[i]) : 0.0f;
        float q = fmaxf(-127.0f, fminf(127.0f, roundf((xval - cache) * scale)));
        if (a_hat_cache != nullptr) a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        output_int8[i] = static_cast<int8_t>(q);
    }
}

template <typename T_IN>
__global__ void upsample2x_quantize_pack_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out,
    __half* __restrict__ a_hat_cache,     // nullptr => baseline, bit-identical (see int8 sibling)
    // ACTIVATION ZERO POINT (plan fix #2), AND THIS KERNEL HAS TWO ROLES:
    //   a_hat_cache == nullptr  -> quantizes the ACTIVATION on the activation grid, feeding a conv
    //                              that adds the zp-corrected bias.  z applies.
    //   a_hat_cache != nullptr  -> quantizes the temporal DELTA (x - a_hat) and updates a_hat with
    //                              q/s.  z does NOT apply: the delta is a difference of activations,
    //                              so z cancels, and adding it here would corrupt the a_hat update
    //                              (which would then need (q - z)/s) while the o_hat conv adds no
    //                              bias at all.
    // The host wrapper TORCH_CHECKs the second case, so this parameter is only ever non-zero in the
    // first. z == 0 reproduces the old kernel exactly in both.
    float zp = 0.0f) {
    float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const int Wo = W * 2, Ho = H * 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    for (long base = idx * 2; base < num_elements_out; base += stride * 2) {
        int c0 = (int)(base % C);
        long rest = base / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho >> 1, wi = wo >> 1;
        long pix_in = (n * H + hi) * (long)W + wi;
        float x0 = load_as_float(x, (int)(pix_in * C + c0));
        if (smooth_inv != nullptr) x0 *= smooth_inv[c0 % num_channels];
        const float c_0 = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[base]) : 0.0f;
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((x0 - c_0) * scale) + zp));
        if (a_hat_cache != nullptr) a_hat_cache[base] = __float2half_rn(c_0 + q0 * inv_scale);
        float q1 = 0.0f;
        if (base + 1 < num_elements_out) {
            float x1 = load_as_float(x, (int)(pix_in * C + c0 + 1));
            if (smooth_inv != nullptr) x1 *= smooth_inv[(c0 + 1) % num_channels];
            const float c_1 = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[base + 1]) : 0.0f;
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf((x1 - c_1) * scale) + zp));
            if (a_hat_cache != nullptr)
                a_hat_cache[base + 1] = __float2half_rn(c_1 + q1 * inv_scale);
        }
        output_packed[base / 2] = (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Op: fused Upsample(nearest,2x) + cache-free static int8 quantize (baseline conv, no a_hat).
// Inputs: x FP16/FP32 [N,C,H,W] channels_last (pre-upsample); scale_buf FP32 [1]; smooth_inv FP32 [C] (empty = skip).
// Output: INT8 [N,C,2H,2W] channels_last -- feeds the Upsample.conv's conv2d_int*_evt_* directly.
torch::Tensor upsample2x_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache) {
    // Name kept for pybind/API compatibility (csrc/modiff_kernels_api.h notes the _noahat spelling
    // is load-bearing); an EMPTY a_hat_cache is the no-a_hat baseline it was named for.
    TORCH_CHECK(x.dim() == 4, "upsample2x_quantize_noahat_fprop: x must be [N,C,H,W]");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto y = torch::empty({N, C, H * 2, W * 2}, options);
    long num_elements_out = (long)N * C * H * 2 * W * 2;
    int block_size = 256;
    int grid_size = (int)std::min<long>((num_elements_out + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    __half* cache_ptr = nullptr;
    if (a_hat_cache.numel() > 0) {
        TORCH_CHECK(a_hat_cache.scalar_type() == torch::kHalf,
                    "upsample2x_quantize_noahat_fprop: a_hat_cache must be fp16");
        TORCH_CHECK(a_hat_cache.numel() == num_elements_out,
                    "upsample2x_quantize_noahat_fprop: a_hat_cache must be POST-upsample sized "
                    "(N*C*2H*2W)");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
    if (x.scalar_type() == torch::kHalf) {
        upsample2x_quantize_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out,
            cache_ptr);
    } else {
        upsample2x_quantize_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out,
            cache_ptr);
    }
    return y;
}

// int4-packed counterpart of upsample2x_quantize_noahat_fprop. Output packed int8 [N,2H,2W,C/2]
// (same layout convention as step1_static_quantize_pack_int4_noahat_fprop). Requires C%2==0
// (channel-pair packing), matching every other int4 quantize kernel in this file.
static torch::Tensor upsample2x_quantize_pack_noahat_fprop_impl(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache, double zero_point) {
    TORCH_CHECK(x.dim() == 4, "upsample2x_quantize_pack_noahat_fprop: x must be [N,C,H,W]");
    // See the kernel's zp comment: with an a_hat cache this kernel quantizes a DELTA, where the
    // zero point is undefined and would also break the a_hat update. Refuse rather than pick.
    TORCH_CHECK(a_hat_cache.numel() == 0 || zero_point == 0.0,
                "upsample2x_quantize_pack_noahat_fprop: a non-zero activation zero point was passed "
                "together with an a_hat cache, i.e. to a DELTA quantize. The zero point belongs to "
                "the activation grid only -- pass 0.0 on the MoDiff path");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % 2 == 0, "upsample2x_quantize_pack_noahat_fprop: C must be even for int4 packing");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    long num_elements_out = (long)N * C * H * 2 * W * 2;
    long num_output = num_elements_out / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    int block_size = 256;
    long num_work_items = (num_elements_out + 3) / 4;
    int grid_size = (int)std::min<long>((num_work_items + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    __half* cache_ptr = nullptr;
    if (a_hat_cache.numel() > 0) {
        TORCH_CHECK(a_hat_cache.scalar_type() == torch::kHalf,
                    "upsample2x_quantize_pack_noahat_fprop: a_hat_cache must be fp16");
        TORCH_CHECK(a_hat_cache.numel() == num_elements_out,
                    "upsample2x_quantize_pack_noahat_fprop: a_hat_cache must be POST-upsample sized");
        cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    }
    if (x.scalar_type() == torch::kHalf) {
        upsample2x_quantize_pack_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out, cache_ptr,
            (float)zero_point);
    } else {
        upsample2x_quantize_pack_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out, cache_ptr,
            (float)zero_point);
    }
    return x_packed.view({N, H * 2, W * 2, C / 2});
}

// Two arities, same rationale as the other _zp pairs in this tree.
torch::Tensor upsample2x_quantize_pack_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache) {
    return upsample2x_quantize_pack_noahat_fprop_impl(x, scale_buf, smooth_inv, a_hat_cache, 0.0);
}

torch::Tensor upsample2x_quantize_pack_noahat_fprop_zp(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache, double zero_point) {
    return upsample2x_quantize_pack_noahat_fprop_impl(x, scale_buf, smooth_inv, a_hat_cache,
                                                      zero_point);
}

// Op: fused Downsample(avg_pool,2x2,stride2) + cache-free static int8 quantize (baseline conv, no
// a_hat). Mirrors upsample2x_quantize_noahat_kernel's structure and the fusion rationale in its
// header comment above, but for the *down* transition: reads the LARGE pre-pool [N,C,H,W] tensor
// once, computes each output pixel's 2x2 average directly (via avgpool4_as_stored, bit-exact to
// nn.AvgPool2d's fp16 output), and quantizes -- never materializing the fp16 pooled [N,C,H/2,W/2]
// intermediate that Downsample.forward would otherwise write and the plain
// static_quantize_*_noahat_kernel would re-read.
template <typename T_IN>
__global__ void avgpool2x_quantize_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_int8,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out) {
    float scale = *scale_ptr;
    const int Wo = W / 2, Ho = H / 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long i = idx; i < num_elements_out; i += (long)blockDim.x * gridDim.x) {
        int c = (int)(i % C);
        long rest = i / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho * 2, wi = wo * 2;
        long row0 = (n * H + hi) * (long)W + wi;
        long row1 = (n * H + hi + 1) * (long)W + wi;
        float sum = load_as_float(x, (int)(row0 * C + c)) + load_as_float(x, (int)((row0 + 1) * C + c))
                  + load_as_float(x, (int)(row1 * C + c)) + load_as_float(x, (int)((row1 + 1) * C + c));
        float xval = avgpool4_as_stored<T_IN>(sum * 0.25f);
        if (smooth_inv != nullptr) xval *= smooth_inv[c % num_channels];
        float q = fmaxf(-127.0f, fminf(127.0f, roundf(xval * scale)));
        output_int8[i] = static_cast<int8_t>(q);
    }
}

template <typename T_IN>
__global__ void avgpool2x_quantize_pack_noahat_kernel(
    const T_IN* __restrict__ x, int8_t* __restrict__ output_packed,
    const float* __restrict__ scale_ptr, const float* __restrict__ smooth_inv,
    int C, int H, int W, int num_channels, long num_elements_out) {
    float scale = *scale_ptr;
    const int Wo = W / 2, Ho = H / 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    for (long base = idx * 2; base < num_elements_out; base += stride * 2) {
        int c0 = (int)(base % C);
        long rest = base / C;
        int wo = (int)(rest % Wo);
        long rest2 = rest / Wo;
        int ho = (int)(rest2 % Ho);
        long n = rest2 / Ho;
        int hi = ho * 2, wi = wo * 2;
        long row0 = (n * H + hi) * (long)W + wi;
        long row1 = (n * H + hi + 1) * (long)W + wi;
        float sum0 = load_as_float(x, (int)(row0 * C + c0)) + load_as_float(x, (int)((row0 + 1) * C + c0))
                   + load_as_float(x, (int)(row1 * C + c0)) + load_as_float(x, (int)((row1 + 1) * C + c0));
        float x0 = avgpool4_as_stored<T_IN>(sum0 * 0.25f);
        if (smooth_inv != nullptr) x0 *= smooth_inv[c0 % num_channels];
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(x0 * scale)));
        float q1 = 0.0f;
        if (base + 1 < num_elements_out) {
            int c1 = c0 + 1;
            float sum1 = load_as_float(x, (int)(row0 * C + c1)) + load_as_float(x, (int)((row0 + 1) * C + c1))
                       + load_as_float(x, (int)(row1 * C + c1)) + load_as_float(x, (int)((row1 + 1) * C + c1));
            float x1 = avgpool4_as_stored<T_IN>(sum1 * 0.25f);
            if (smooth_inv != nullptr) x1 *= smooth_inv[c1 % num_channels];
            q1 = fmaxf(-7.0f, fminf(7.0f, roundf(x1 * scale)));
        }
        output_packed[base / 2] = (static_cast<int8_t>(q0) & 0x0F) | ((static_cast<int8_t>(q1) & 0x0F) << 4);
    }
}

// Op: fused Downsample(avg_pool,2x2) + cache-free static int8 quantize (baseline conv, no a_hat).
// Inputs: x FP16/FP32 [N,C,H,W] channels_last (pre-pool, LARGE); scale_buf FP32 [1]; smooth_inv FP32 [C] (empty = skip).
// Output: INT8 [N,C,H/2,W/2] channels_last -- feeds the downsample ResBlock's in_conv directly.
torch::Tensor avgpool2x_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    TORCH_CHECK(x.dim() == 4, "avgpool2x_quantize_noahat_fprop: x must be [N,C,H,W]");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "avgpool2x_quantize_noahat_fprop: H,W must be even");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device()).memory_format(torch::MemoryFormat::ChannelsLast);
    auto y = torch::empty({N, C, H / 2, W / 2}, options);
    long num_elements_out = (long)N * C * (H / 2) * (W / 2);
    int block_size = 256;
    int grid_size = (int)std::min<long>((num_elements_out + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    if (x.scalar_type() == torch::kHalf) {
        avgpool2x_quantize_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    } else {
        avgpool2x_quantize_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), y.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    }
    return y;
}

// int4-packed counterpart of avgpool2x_quantize_noahat_fprop. Output packed int8 [N,H/2,W/2,C/2]
// (same layout convention as upsample2x_quantize_pack_noahat_fprop). Requires C%2==0.
torch::Tensor avgpool2x_quantize_pack_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv) {
    TORCH_CHECK(x.dim() == 4, "avgpool2x_quantize_pack_noahat_fprop: x must be [N,C,H,W]");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "avgpool2x_quantize_pack_noahat_fprop: H,W must be even");
    TORCH_CHECK(C % 2 == 0, "avgpool2x_quantize_pack_noahat_fprop: C must be even for int4 packing");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    long num_elements_out = (long)N * C * (H / 2) * (W / 2);
    long num_output = num_elements_out / 2;
    auto x_packed = torch::empty({num_output}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    int block_size = 256;
    long num_work_items = (num_elements_out + 3) / 4;
    int grid_size = (int)std::min<long>((num_work_items + block_size - 1) / block_size, 2147483647L);
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    int num_channels = (smooth_inv.numel() > 0) ? (int)smooth_inv.numel() : C;
    if (x.scalar_type() == torch::kHalf) {
        avgpool2x_quantize_pack_noahat_kernel<__half><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    } else {
        avgpool2x_quantize_pack_noahat_kernel<float><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<float>(), x_packed.data_ptr<int8_t>(),
            scale_buf.data_ptr<float>(), smooth_ptr, C, H, W, num_channels, num_elements_out);
    }
    return x_packed.view({N, H / 2, W / 2, C / 2});
}
