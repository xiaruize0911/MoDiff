// =========================================================================
// GroupNorm (+ optional fused SiLU), native channels_last (NHWC-physical).
//
// F.group_norm always returns a plain NCHW-contiguous tensor regardless of
// its input's memory format (profiled and confirmed against this project's
// actual pipeline -- see FusedGroupNormSiLU's docstring in
// integration/fused_ops/fused_resblock.py). Every quantized conv in this
// pipeline requires channels_last for its CUTLASS kernel, so a ResBlock
// running channels_last activations was paying two avoidable round-trip
// copies per GroupNorm call: channels_last -> (GroupNorm forces) -> NCHW ->
// (conv forces back) -> channels_last.
//
// This kernel reads and writes NHWC-physical memory directly and never
// materializes an NCHW intermediate. One block handles one (sample, group)
// pair; within a group, channels are contiguous in memory (channels-last
// layout), so indexing with the channel-in-group as the fast-varying
// dimension (idx = hw * channels_per_group + c_local) keeps consecutive
// threads' accesses contiguous. (A prior Triton attempt at this fusion --
// integration/fused_ops/fused_gn_silu.py, not used in production -- indexed
// the opposite way for its channels_last branch, giving it a stride-C access
// pattern; that is very likely why it measured slower than plain
// F.group_norm+F.silu despite doing less total work.)
//
// Reduction is single-pass (running sum + sum-of-squares, accumulated in
// fp32 regardless of input dtype) rather than the textbook two-pass
// mean-then-variance, trading a small amount of numerical headroom for one
// fewer full pass over the group's data; group sizes here (few hundred to
// ~25k elements) keep that fp32 accumulation error negligible.
// =========================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../common.cuh"

__device__ __forceinline__ float gn_load(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float gn_load(const __half* p, long i) { return __half2float(p[i]); }
__device__ __forceinline__ void gn_store(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void gn_store(__half* p, long i, float v) { p[i] = __float2half(v); }

template <typename T>
__global__ void group_norm_silu_nhwc_kernel(
    const T* __restrict__ X,      // [N, H, W, C] physical (channels_last NCHW logical)
    T* __restrict__ Y,            // same shape/layout as X
    const T* __restrict__ gamma,  // [C], affine weight
    const T* __restrict__ beta,   // [C], affine bias
    const T* __restrict__ mod_scale, // [N, C] scale-shift modulation, or nullptr
    const T* __restrict__ mod_shift, // [N, C] scale-shift modulation, or nullptr
    int C,
    long HW,
    int G,
    float eps,
    bool apply_silu
) {
    const int CPG = C / G;                  // channels per group
    const long group_size = (long)CPG * HW;

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const T* x_base = X + (long)n * HW * C;
    T* y_base = Y + (long)n * HW * C;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // Pass 1: accumulate sum and sum-of-squares over this (sample, group).
    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        long mem_idx = hw * C + c_start + c_local;
        float v = gn_load(x_base, mem_idx);
        local_sum += v;
        local_sumsq += v * v;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }

    __shared__ float mean_s, inv_std_s;
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / (float)group_size;
        float var = s_sumsq[0] / (float)group_size - mean * mean;
        var = fmaxf(var, 0.0f);  // guard against tiny negative values from fp rounding
        mean_s = mean;
        inv_std_s = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;

    // Pass 2: normalize, apply affine, optionally apply SiLU, write out.
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        int c_global = c_start + c_local;
        long mem_idx = hw * C + c_global;

        float v = gn_load(x_base, mem_idx);
        float w = gn_load(gamma, c_global);
        float b = gn_load(beta, c_global);
        float normed = (v - mean) * inv_std * w + b;
        // Optional scale-shift modulation (use_scale_shift_norm) before SiLU.
        if (mod_scale != nullptr) {
            long midx = (long)n * C + c_global;
            normed = normed * (1.0f + gn_load(mod_scale, midx)) + gn_load(mod_shift, midx);
        }
        float out = apply_silu ? (normed / (1.0f + expf(-normed))) : normed;
        gn_store(y_base, mem_idx, out);
    }
}

// x must be channels_last-contiguous (NHWC physical); weight/bias must match
// x's dtype (fp32 or fp16). Returns a new channels_last tensor of the same
// shape/dtype as x -- GroupNorm normalized, optionally with SiLU fused in.
torch::Tensor group_norm_silu_nhwc(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor mod_scale,   // [N, C] scale-shift modulation, or empty for none
    torch::Tensor mod_shift
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_nhwc: only float32 and float16 are supported");
    const bool has_mod = mod_scale.numel() > 0;
    TORCH_CHECK(!has_mod || (mod_scale.scalar_type() == x.scalar_type() && mod_shift.scalar_type() == x.scalar_type()),
                "group_norm_silu_nhwc: mod_scale/mod_shift dtype must match input dtype");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "group_norm_silu_nhwc: num_channels must be divisible by num_groups");
    const long HW = (long)H * W;
    const int CPG = C / (int)num_groups;
    const long group_size = (long)CPG * HW;

    auto y = torch::empty_like(x);

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;

    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat32) {
        group_norm_silu_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), y.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            C, HW, (int)num_groups, (float)eps, apply_silu
        );
    } else {
        group_norm_silu_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            C, HW, (int)num_groups, (float)eps, apply_silu
        );
    }

    return y;
}

// INT8-emitting variant of the kernel above: identical GroupNorm(+SiLU) math,
// but pass 2 quantizes the result to int8 (out * scale, clamped/rounded) so the
// following calibrated INT8 conv can read it directly -- fusing away the separate
// per-conv quantize kernel. Optional per-channel `smooth_inv` (SmoothQuant) is
// applied before quantize, mirroring modiff_delta_quantize's static quantize.
template <typename TIn>
__global__ void group_norm_silu_quantize_nhwc_kernel(
    const TIn* __restrict__ X,
    int8_t* __restrict__ Yq,          // [N, H, W, C] physical int8, same layout as X
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale, // [N, C] scale-shift modulation, or nullptr
    const TIn* __restrict__ mod_shift, // [N, C] scale-shift modulation, or nullptr
    const float* __restrict__ scale_ptr,    // scalar quant multiplier = 127/absmax
    const float* __restrict__ smooth_inv,   // [C] SmoothQuant, or nullptr
    int C,
    long HW,
    int G,
    float eps,
    bool apply_silu
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    int8_t* yq_base = Yq + (long)n * HW * C;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // Pass 1: sum + sum-of-squares over this (sample, group).
    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        long mem_idx = hw * C + c_start + c_local;
        float v = gn_load(x_base, mem_idx);
        local_sum += v;
        local_sumsq += v * v;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }
    __shared__ float mean_s, inv_std_s;
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / (float)group_size;
        float var = s_sumsq[0] / (float)group_size - mean * mean;
        var = fmaxf(var, 0.0f);
        mean_s = mean;
        inv_std_s = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;

    // Pass 2: normalize, affine, optional SiLU, optional SmoothQuant, quantize -> int8.
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        int c_global = c_start + c_local;
        long mem_idx = hw * C + c_global;

        float v = gn_load(x_base, mem_idx);
        float w = gn_load(gamma, c_global);
        float b = gn_load(beta, c_global);
        float normed = (v - mean) * inv_std * w + b;
        // Optional scale-shift modulation (use_scale_shift_norm): per-(sample,channel)
        // affine from the timestep embedding, applied before SiLU. normed*(1+s)+sh.
        if (mod_scale != nullptr) {
            long midx = (long)n * C + c_global;
            normed = normed * (1.0f + gn_load(mod_scale, midx)) + gn_load(mod_shift, midx);
        }
        float out = apply_silu ? (normed / (1.0f + expf(-normed))) : normed;
        if (smooth_inv != nullptr) out *= smooth_inv[c_global];
        yq_base[mem_idx] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(out * scale)));
    }
}

// Host wrapper for the INT8-emitting GroupNorm+SiLU. Returns an int8 tensor with
// the same NHWC (channels_last) layout as x. `scale` is a 1-element device tensor
// (127/absmax); `smooth_inv` is [C] or an empty tensor for identity.
torch::Tensor group_norm_silu_quantize_nhwc(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,   // [N, C] scale-shift modulation, or empty for none
    torch::Tensor mod_shift
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_quantize_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_quantize_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_quantize_nhwc: only float32 and float16 are supported");
    const bool has_mod = mod_scale.numel() > 0;
    TORCH_CHECK(!has_mod || (mod_scale.scalar_type() == x.scalar_type() && mod_shift.scalar_type() == x.scalar_type()),
                "group_norm_silu_quantize_nhwc: mod_scale/mod_shift dtype must match input dtype");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "group_norm_silu_quantize_nhwc: num_channels must be divisible by num_groups");
    const long HW = (long)H * W;
    const int CPG = C / (int)num_groups;
    const long group_size = (long)CPG * HW;

    auto yq = torch::empty_like(x, x.options().dtype(torch::kInt8));

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;

    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;

    if (x.scalar_type() == torch::kFloat32) {
        group_norm_silu_quantize_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu
        );
    } else {
        group_norm_silu_quantize_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu
        );
    }

    return yq;
}

// ============================================================================
// INT8-INPUT variant of group_norm_silu_quantize_nhwc: reads an int8 activation
// (the upstream conv's int8 output) + a scalar dequant scale, so the conv->GN
// handoff never materializes an fp16 tensor (halves that write + this read). The
// GN stats are computed from the dequantized values (v = X_i8 * in_dequant); the
// rest (affine, scale-shift mod, SiLU, requantize->int8 for the next conv) is
// identical to the fp16 path. `TAff` is the gamma/beta/mod dtype (fp16 or fp32).
// ============================================================================
template <typename TAff>
__global__ void group_norm_silu_dequant_quantize_nhwc_kernel(
    const int8_t* __restrict__ X,
    int8_t* __restrict__ Yq,
    const TAff* __restrict__ gamma,
    const TAff* __restrict__ beta,
    const TAff* __restrict__ mod_scale,
    const TAff* __restrict__ mod_shift,
    float in_dequant,                       // upstream conv's dequant scale (absmax/127)
    const float* __restrict__ scale_ptr,    // 127/absmax for THIS output
    const float* __restrict__ smooth_inv,
    int C, long HW, int G, float eps, bool apply_silu
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;
    const int8_t* x_base = X + (long)n * HW * C;
    int8_t* yq_base = Yq + (long)n * HW * C;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        float v = (float)x_base[hw * C + c_start + c_local] * in_dequant;
        local_sum += v; local_sumsq += v * v;
    }
    s_sum[threadIdx.x] = local_sum; s_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { s_sum[threadIdx.x] += s_sum[threadIdx.x + s]; s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s]; }
        __syncthreads();
    }
    __shared__ float mean_s, inv_std_s;
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / (float)group_size;
        float var = fmaxf(s_sumsq[0] / (float)group_size - mean * mean, 0.0f);
        mean_s = mean; inv_std_s = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = mean_s, inv_std = inv_std_s, scale = *scale_ptr;

    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        int c_global = c_start + c_local;
        long mem_idx = hw * C + c_global;
        float v = (float)x_base[mem_idx] * in_dequant;
        float normed = (v - mean) * inv_std * gn_load(gamma, c_global) + gn_load(beta, c_global);
        if (mod_scale != nullptr) {
            long midx = (long)n * C + c_global;
            normed = normed * (1.0f + gn_load(mod_scale, midx)) + gn_load(mod_shift, midx);
        }
        float out = apply_silu ? (normed / (1.0f + expf(-normed))) : normed;
        if (smooth_inv != nullptr) out *= smooth_inv[c_global];
        yq_base[mem_idx] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(out * scale)));
    }
}

// Host wrapper: int8-in (x_int8 NHWC + in_dequant scalar) -> int8-out GroupNorm+SiLU.
torch::Tensor group_norm_silu_dequant_quantize_nhwc(
    torch::Tensor x_int8, double in_dequant,
    torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift
) {
    CHECK_CUDA(x_int8); CHECK_CONTIGUOUS(x_int8);
    TORCH_CHECK(x_int8.dim() == 4 && x_int8.scalar_type() == torch::kInt8, "x_int8 must be 4D int8");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16 || weight.scalar_type() == torch::kFloat32,
                "weight/bias must be fp16 or fp32");
    const bool has_mod = mod_scale.numel() > 0;
    const int N = x_int8.size(0), C = x_int8.size(1), H = x_int8.size(2), W = x_int8.size(3);
    TORCH_CHECK(C % num_groups == 0, "num_channels must be divisible by num_groups");
    const long HW = (long)H * W;
    const long group_size = (long)(C / (int)num_groups) * HW;
    auto yq = torch::empty_like(x_int8);
    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups)), block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    const int8_t* xp = reinterpret_cast<const int8_t*>(x_int8.data_ptr<int8_t>());
    int8_t* yp = reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>());
    if (weight.scalar_type() == torch::kFloat32) {
        group_norm_silu_dequant_quantize_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            xp, yp, weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            (float)in_dequant, scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu);
    } else {
        group_norm_silu_dequant_quantize_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            xp, yp, reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            (float)in_dequant, scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu);
    }
    return yq;
}

// INT4-emitting variant: identical GroupNorm(+SiLU) math, but pass 2 quantizes to
// int4 codes in [-7,7] and packs adjacent-channel pairs into one byte, producing
// output byte-identical to scale_quantize_and_pack(SiLU(GN(x)), scale) so the
// following calibrated INT4 conv can read it directly. Packing is along the
// (contiguous) NHWC channel dim: byte = flat_element/2, low nibble = even channel,
// high nibble = odd channel -- exactly quantize.cu::scale_quantize_pack_kernel's
// convention. Requires channels-per-group (CPG) even so a channel pair never
// straddles a group boundary (both channels then share one group's mean/inv_std);
// the host wrapper enforces this and the Python caller gates on it, falling back
// to the two-kernel path otherwise.
template <typename TIn>
__global__ void group_norm_silu_quantize_pack_nhwc_kernel(
    const TIn* __restrict__ X,
    int8_t* __restrict__ Yqp,         // [N, H, W, C/2] packed int4, channels_last-flat
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale, // [N, C] scale-shift modulation, or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int C,
    long HW,
    int G,
    float eps,
    bool apply_silu
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    int8_t* yqp_base = Yqp + (long)n * ((HW * (long)C) / 2);  // packed bytes per sample

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // Pass 1: sum + sum-of-squares over this (sample, group).
    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        long mem_idx = hw * C + c_start + c_local;
        float v = gn_load(x_base, mem_idx);
        local_sum += v;
        local_sumsq += v * v;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }
    __shared__ float mean_s, inv_std_s;
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / (float)group_size;
        float var = s_sumsq[0] / (float)group_size - mean * mean;
        var = fmaxf(var, 0.0f);
        mean_s = mean;
        inv_std_s = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;

    // Pass 2: one thread per output byte = one (even,odd) channel pair at a
    // spatial position. c_start and C are both even (CPG even), so the even
    // channel's flat index mem_idx0 is even and byte = mem_idx0/2.
    const int HALF_CPG = CPG / 2;
    const long pairs = group_size / 2;   // = HALF_CPG * HW
    for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
        int cpair = pidx % HALF_CPG;
        long hw = pidx / HALF_CPG;
        int c_global0 = c_start + 2 * cpair;
        long mem_idx0 = hw * (long)C + c_global0;

        float v0 = gn_load(x_base, mem_idx0);
        float v1 = gn_load(x_base, mem_idx0 + 1);
        float w0 = gn_load(gamma, c_global0),     b0 = gn_load(beta, c_global0);
        float w1 = gn_load(gamma, c_global0 + 1), b1 = gn_load(beta, c_global0 + 1);
        float n0 = (v0 - mean) * inv_std * w0 + b0;
        float n1 = (v1 - mean) * inv_std * w1 + b1;
        if (mod_scale != nullptr) {
            long midx0 = (long)n * C + c_global0;
            n0 = n0 * (1.0f + gn_load(mod_scale, midx0))     + gn_load(mod_shift, midx0);
            n1 = n1 * (1.0f + gn_load(mod_scale, midx0 + 1)) + gn_load(mod_shift, midx0 + 1);
        }
        float o0 = apply_silu ? (n0 / (1.0f + expf(-n0))) : n0;
        float o1 = apply_silu ? (n1 / (1.0f + expf(-n1))) : n1;
        if (smooth_inv != nullptr) {
            o0 *= smooth_inv[c_global0];
            o1 *= smooth_inv[c_global0 + 1];
        }
        int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o0 * scale)));
        int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o1 * scale)));
        yqp_base[mem_idx0 / 2] = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
    }
}

// Host wrapper for the INT4-packed GroupNorm+SiLU. Returns a [N, H, W, C/2] int8
// tensor holding packed int4 codes, matching scale_quantize_and_pack's layout.
// Requires C and channels-per-group both even.
torch::Tensor group_norm_silu_quantize_pack_nhwc(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,   // [N, C] scale-shift modulation, or empty for none
    torch::Tensor mod_shift
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_quantize_pack_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_quantize_pack_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_quantize_pack_nhwc: only float32 and float16 are supported");
    const bool has_mod = mod_scale.numel() > 0;
    TORCH_CHECK(!has_mod || (mod_scale.scalar_type() == x.scalar_type() && mod_shift.scalar_type() == x.scalar_type()),
                "group_norm_silu_quantize_pack_nhwc: mod_scale/mod_shift dtype must match input dtype");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "group_norm_silu_quantize_pack_nhwc: num_channels must be divisible by num_groups");
    const int CPG = C / (int)num_groups;
    TORCH_CHECK(C % 2 == 0 && CPG % 2 == 0,
                "group_norm_silu_quantize_pack_nhwc: channels and channels-per-group must be even "
                "(int4 packs channel pairs within a group)");
    const long HW = (long)H * W;
    const long group_size = (long)CPG * HW;

    // [N, H, W, C/2] contiguous == the same flat byte order as scale_quantize_and_pack.
    auto yqp = torch::empty({N, H, W, C / 2},
                            torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;

    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;

    if (x.scalar_type() == torch::kFloat32) {
        group_norm_silu_quantize_pack_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu
        );
    } else {
        group_norm_silu_quantize_pack_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu
        );
    }

    return yqp;
}
