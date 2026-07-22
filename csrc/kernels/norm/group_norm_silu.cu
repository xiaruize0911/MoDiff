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

#include "common.cuh"

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

// -----------------------------------------------------------------------------
//   Op:       GroupNorm(+SiLU) NHWC (channels_last-native)
//   Inputs:   x          fp16|fp32 [N,C,H,W] channels_last-contiguous (NHWC physical)
//             weight     fp16|fp32 [C]   affine gamma (dtype must match x)
//             bias       fp16|fp32 [C]   affine beta  (dtype must match x)
//             num_groups int64; eps double; apply_silu bool
//             mod_scale  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//             mod_shift  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//   Outputs:  fp16|fp32 [N,C,H,W] channels_last, same shape/dtype as x
//   Computes: per-(sample,group) mean/var over (channels-in-group x H x W), fp32 accum;
//             y = (x-mean)*rsqrt(var+eps)*gamma + beta;
//             if mod:  y = y*(1+mod_scale)+mod_shift  (before SiLU);
//             if apply_silu:  y = y*sigmoid(y)
//   Fuses:    SiLU + optional scale-shift modulation into the norm; reads/writes NHWC
//             directly (no NCHW intermediate)
//   Constraints: 4D; C%num_groups==0; x channels_last-contiguous; weight/bias/mod dtype==x;
//             fp16 or fp32 only
//   vs fp16:  native NHWC GroupNorm(+SiLU) 1.58-2.11x vs F.group_norm at churches spatial
//             sizes (win = reads/writes channels_last directly, avoiding F.group_norm's
//             forced NCHW round-trip). ~1.0x at tiny spatial (768/4x4).
//
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

// -----------------------------------------------------------------------------
//   Op:       GroupNorm(+SiLU)+quantize-to-INT8 NHWC (channels_last-native)
//   Inputs:   x          fp16|fp32 [N,C,H,W] channels_last-contiguous (NHWC physical)
//             weight     fp16|fp32 [C]   affine gamma (dtype must match x)
//             bias       fp16|fp32 [C]   affine beta  (dtype must match x)
//             num_groups int64; eps double; apply_silu bool
//             scale      fp32 [1] device scalar = 127/absmax (quant multiplier)
//             smooth_inv fp32 [C]  SmoothQuant per-channel inverse, or empty for identity
//             mod_scale  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//             mod_shift  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//   Outputs:  int8 [N,C,H,W] channels_last (same NHWC layout as x)
//   Computes: same GN(+mod)(+SiLU) as group_norm_silu_nhwc, then
//             out *= smooth_inv[c] (if given); q = clamp(round(out*scale), -127, 127)
//   Fuses:    SiLU + scale-shift mod + SmoothQuant + int8 quantize into one kernel,
//             feeding the downstream INT8 GEMM (no standalone quantize pass)
//   Constraints: 4D; C%num_groups==0; x channels_last-contiguous; weight/bias/mod dtype==x;
//             fp16 or fp32 input only
//   vs fp16:  same GN win as group_norm_silu_nhwc plus the activation quantize fused into
//             the same kernel (feeds the downstream int GEMM), so the quantize is
//             effectively free vs a standalone pass.
//
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
// TWO-SOURCE variant of group_norm_silu_quantize_nhwc: reads the GN input from
// TWO channel-concatenated NHWC sources (XA [N,H,W,C1], XB [N,H,W,C2]) instead of
// one pre-concatenated tensor -- channels [0,C1) come from XA, [C1,C1+C2) from XB.
// Lets the UNet decoder skip concat (torch.cat([h, skip], dim=1)) be READ in-place
// by this GN, so the CatArrayBatchedCopy that would materialize the concat is
// eliminated on the GN side. Output is the usual contiguous int8 [N,C,H,W] the
// downstream conv reads. Math is identical to concat-then-GN (same per-element
// values, same reduction order) -> bit-identical.
// ============================================================================
template <typename TIn>
__global__ void group_norm_silu_quantize_2src_nhwc_kernel(
    const TIn* __restrict__ XA,        // [N,H,W,C1] physical (channels_last)
    const TIn* __restrict__ XB,        // [N,H,W,C2] physical
    int C1,
    int8_t* __restrict__ Yq,           // [N,H,W,C] int8, C = C1+C2
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale, // [N,C] or nullptr
    const TIn* __restrict__ mod_shift, // [N,C] or nullptr
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int C, long HW, int G, float eps, bool apply_silu
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;
    const int C2 = C - C1;
    const TIn* xa_base = XA + (long)n * HW * C1;
    const TIn* xb_base = XB + (long)n * HW * C2;
    int8_t* yq_base = Yq + (long)n * HW * C;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // Pass 1: stats over this (sample, group), reading each channel from XA or XB.
    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        int c_global = c_start + c_local;
        float v = (c_global < C1) ? gn_load(xa_base, hw * C1 + c_global)
                                  : gn_load(xb_base, hw * C2 + (c_global - C1));
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

    // Pass 2: normalize, affine, optional mod/SiLU/SmoothQuant, quantize -> int8.
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        int c_global = c_start + c_local;
        float v = (c_global < C1) ? gn_load(xa_base, hw * C1 + c_global)
                                  : gn_load(xb_base, hw * C2 + (c_global - C1));
        float w = gn_load(gamma, c_global);
        float b = gn_load(beta, c_global);
        float normed = (v - mean) * inv_std * w + b;
        if (mod_scale != nullptr) {
            long midx = (long)n * C + c_global;
            normed = normed * (1.0f + gn_load(mod_scale, midx)) + gn_load(mod_shift, midx);
        }
        float out = apply_silu ? (normed / (1.0f + expf(-normed))) : normed;
        if (smooth_inv != nullptr) out *= smooth_inv[c_global];
        yq_base[hw * C + c_global] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(out * scale)));
    }
}

// Host wrapper for the two-source INT8-emitting GroupNorm+SiLU. xa=[N,C1,H,W],
// xb=[N,C2,H,W] (both channels_last); returns int8 [N,C1+C2,H,W] channels_last.
// gamma/beta/scale/smooth_inv/mod are indexed over the full C=C1+C2 (as if xa,xb
// were concatenated). split_c1 must equal xa.size(1) (passed for clarity/checks).
torch::Tensor group_norm_silu_quantize_2src_nhwc(
    torch::Tensor xa,
    torch::Tensor xb,
    int64_t split_c1,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,
    torch::Tensor mod_shift
) {
    CHECK_CUDA(xa); CHECK_CONTIGUOUS(xa);
    CHECK_CUDA(xb); CHECK_CONTIGUOUS(xb);
    TORCH_CHECK(xa.dim() == 4 && xb.dim() == 4, "group_norm_silu_quantize_2src_nhwc: expects 4D tensors");
    TORCH_CHECK(xa.scalar_type() == xb.scalar_type() && xa.scalar_type() == weight.scalar_type()
                && xa.scalar_type() == bias.scalar_type(),
                "group_norm_silu_quantize_2src_nhwc: xa/xb/weight/bias dtype must match");
    TORCH_CHECK(xa.scalar_type() == torch::kFloat32 || xa.scalar_type() == torch::kFloat16,
                "group_norm_silu_quantize_2src_nhwc: only float32/float16 supported");
    TORCH_CHECK(xa.size(0) == xb.size(0) && xa.size(2) == xb.size(2) && xa.size(3) == xb.size(3),
                "group_norm_silu_quantize_2src_nhwc: xa/xb must share N,H,W");
    TORCH_CHECK(split_c1 == xa.size(1), "group_norm_silu_quantize_2src_nhwc: split_c1 must equal xa channels");
    const bool has_mod = mod_scale.numel() > 0;

    const int N = xa.size(0), C1 = xa.size(1), H = xa.size(2), W = xa.size(3);
    const int C2 = xb.size(1);
    const int C = C1 + C2;
    TORCH_CHECK(C % num_groups == 0, "group_norm_silu_quantize_2src_nhwc: C must be divisible by num_groups");
    const long HW = (long)H * W;
    const int CPG = C / (int)num_groups;
    const long group_size = (long)CPG * HW;

    auto yq = torch::empty({N, C, H, W}, xa.options().dtype(torch::kInt8), torch::MemoryFormat::ChannelsLast);

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;

    if (xa.scalar_type() == torch::kFloat32) {
        group_norm_silu_quantize_2src_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            xa.data_ptr<float>(), xb.data_ptr<float>(), C1,
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr, C, HW, (int)num_groups, (float)eps, apply_silu);
    } else {
        group_norm_silu_quantize_2src_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(xa.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(xb.data_ptr<at::Half>()), C1,
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr, C, HW, (int)num_groups, (float)eps, apply_silu);
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

// -----------------------------------------------------------------------------
//   Op:       INT8-in GroupNorm(+SiLU)+requantize-to-INT8 NHWC (channels_last-native)
//   Inputs:   x_int8     int8 [N,C,H,W] channels_last (upstream conv's int8 output)
//             in_dequant double  upstream conv's dequant scale (absmax/127)
//             weight     fp16|fp32 [C] affine gamma
//             bias       fp16|fp32 [C] affine beta (gamma/beta dtype = TAff)
//             num_groups int64; eps double; apply_silu bool
//             scale      fp32 [1] device scalar = 127/absmax for THIS output
//             smooth_inv fp32 [C]  SmoothQuant per-channel inverse, or empty for identity
//             mod_scale  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//             mod_shift  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//   Outputs:  int8 [N,C,H,W] channels_last (same NHWC layout as x_int8)
//   Computes: v = x_int8 * in_dequant; GN stats computed from the dequantized v;
//             then affine + optional mod + optional SiLU + optional SmoothQuant;
//             q = clamp(round(out*scale), -127, 127)
//   Fuses:    dequant of the int8 input + GN + SiLU + mod + SmoothQuant + int8 requantize
//             into one kernel, so the conv->GN handoff never materializes an fp16 tensor
//             (halves that write + this read)
//   Constraints: 4D int8 input; C%num_groups==0; x_int8 channels_last; weight/bias fp16 or fp32
//   vs fp16:  same GN win as group_norm_silu_nhwc plus the activation quantize fused into
//             the same kernel (feeds the downstream int GEMM), so the quantize is
//             effectively free vs a standalone pass.
//
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

// -----------------------------------------------------------------------------
//   Op:       GroupNorm(+SiLU)+quantize-to-INT4+pack NHWC (channels_last-native)
//   Inputs:   x          fp16|fp32 [N,C,H,W] channels_last-contiguous (NHWC physical)
//             weight     fp16|fp32 [C]   affine gamma (dtype must match x)
//             bias       fp16|fp32 [C]   affine beta  (dtype must match x)
//             num_groups int64; eps double; apply_silu bool
//             scale      fp32 [1] device scalar quant multiplier
//             smooth_inv fp32 [C]  SmoothQuant per-channel inverse, or empty for identity
//             mod_scale  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//             mod_shift  fp16|fp32 [N,C] scale-shift modulation, or empty for none
//   Outputs:  int8 [N,H,W,C/2] packed int4 codes (byte layout matches scale_quantize_and_pack)
//   Computes: same GN(+mod)(+SiLU)(+SmoothQuant) as the int8 variant, but quantizes to int4
//             codes in [-7,7] and packs adjacent-channel pairs into one byte along the
//             contiguous NHWC channel dim (low nibble = even channel, high nibble = odd)
//   Fuses:    SiLU + scale-shift mod + SmoothQuant + int4 quantize + pack into one kernel,
//             feeding the downstream INT4 conv directly (no standalone quantize+pack pass)
//   Constraints: 4D; C%num_groups==0; C even AND channels-per-group even (a channel pair must
//             not straddle a group boundary); x channels_last-contiguous; weight/bias/mod
//             dtype==x; fp16 or fp32 input only
//   vs fp16:  same GN win as group_norm_silu_nhwc plus the activation quantize fused into
//             the same kernel (feeds the downstream int GEMM), so the quantize is
//             effectively free vs a standalone pass.
//
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

// =========================================================================
// MoDiff-fused GroupNorm(+mod)+SiLU + temporal-DELTA quantize (int8 / int4).
//
// These are the modiff-path counterparts of group_norm_silu_quantize[_pack]_nhwc
// above: same GroupNorm(+scale-shift mod)+SiLU(+SmoothQuant) math, but instead of a
// static quantize they perform the MoDiff temporal-delta quantize + in-place a_hat
// cache update -- exactly the epilogue of
// static_quantize_and_update_ahat_kernel_int8_half_cache_silu in
// kernels/quantize/modiff_delta_quantize.cu. This fuses away the standalone
// GroupNorm kernel + the separate step1_static_quantize_fprop_silu pass that the
// modiff ResBlock path (_forward_modulated_static_fused_silu, fuse_input_silu=True)
// otherwise runs back-to-back, removing the fp16 `normed` round-trip between them.
//
// Bit-exactness vs that two-kernel path: the reference materializes `normed` as an
// fp16 tensor (group_norm_silu_nhwc output, apply_silu=False) which step1 then reads
// and SiLU's. We replicate that fp16 rounding of `normed` BEFORE SiLU
// (__float2half then back) so the SiLU input -- hence the int8/int4 code and the
// a_hat update (cache += q/scale, stored fp16) -- matches element-for-element. The
// a_hat_cache is fp16 (the only dtype the calibrated production path uses; enforced
// by the step1_silu reference and TORCH_CHECK'd here).
// =========================================================================
__device__ __forceinline__ float gns_silu(float v) { return v / (1.0f + expf(-v)); }

template <typename TIn>
__global__ void group_norm_silu_delta_quantize_nhwc_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,   // [N,H,W,C] fp16 channels_last, updated in place
    int8_t* __restrict__ Yq,            // [N,H,W,C] int8, quantized delta
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,  // [N, C] scale-shift modulation, or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ scale_ptr,   // scalar quant multiplier = 127/absmax
    const float* __restrict__ smooth_inv,  // [C] SmoothQuant, or nullptr
    int C, long HW, int G, float eps, bool apply_silu
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    __half* cache_base = a_hat_cache + (long)n * HW * C;
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
        float var = fmaxf(s_sumsq[0] / (float)group_size - mean * mean, 0.0f);
        mean_s = mean;
        inv_std_s = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;

    // Pass 2: normalize, affine, optional mod, (fp16-round) SiLU, SmoothQuant,
    // delta-quantize against a_hat, update a_hat in place.
    for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx % CPG;
        long hw = idx / CPG;
        int c_global = c_start + c_local;
        long mem_idx = hw * C + c_global;

        float v = gn_load(x_base, mem_idx);
        float w = gn_load(gamma, c_global);
        float b = gn_load(beta, c_global);
        float normed = (v - mean) * inv_std * w + b;
        if (mod_scale != nullptr) {
            long midx = (long)n * C + c_global;
            normed = normed * (1.0f + gn_load(mod_scale, midx)) + gn_load(mod_shift, midx);
        }
        // Mirror the reference's fp16 `normed` intermediate before SiLU.
        float normed_h = __half2float(__float2half(normed));
        float out = apply_silu ? gns_silu(normed_h) : normed_h;
        if (smooth_inv != nullptr) out *= smooth_inv[c_global];
        float cache = __half2float(cache_base[mem_idx]);
        float q = fmaxf(-127.0f, fminf(127.0f, roundf((out - cache) * scale)));
        cache_base[mem_idx] = __float2half_rn(cache + q * inv_scale);
        yq_base[mem_idx] = (int8_t)q;
    }
}

// Host wrapper: MoDiff GN(+mod)+SiLU + int8 delta-quantize + a_hat update.
// a_hat_cache is fp16 [N,C,H,W] channels_last, modified in place. Returns int8
// [N,C,H,W] channels_last (the quantized delta the o_hat conv consumes).
torch::Tensor group_norm_silu_delta_quantize_nhwc(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor a_hat_cache,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,
    torch::Tensor mod_shift
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_delta_quantize_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_delta_quantize_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_nhwc: only float32 and float16 are supported");
    TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_nhwc: a_hat_cache must be fp16 (calibrated modiff path)");
    TORCH_CHECK(a_hat_cache.sizes() == x.sizes(),
                "group_norm_silu_delta_quantize_nhwc: a_hat_cache must match x shape");
    const bool has_mod = mod_scale.numel() > 0;
    TORCH_CHECK(!has_mod || (mod_scale.scalar_type() == x.scalar_type() && mod_shift.scalar_type() == x.scalar_type()),
                "group_norm_silu_delta_quantize_nhwc: mod_scale/mod_shift dtype must match input dtype");

    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "group_norm_silu_delta_quantize_nhwc: num_channels must be divisible by num_groups");
    const long HW = (long)H * W;
    const long group_size = (long)(C / (int)num_groups) * HW;

    auto yq = torch::empty_like(x, x.options().dtype(torch::kInt8));

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    __half* cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());

    if (x.scalar_type() == torch::kFloat32) {
        group_norm_silu_delta_quantize_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu);
    } else {
        group_norm_silu_delta_quantize_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu);
    }
    return yq;
}

// INT4-packed MoDiff-fused GN+SiLU+delta-quantize: as above but quantizes the
// delta to [-7,7] and packs adjacent-channel pairs into one byte (low nibble =
// even channel, high nibble = odd), matching group_norm_silu_quantize_pack_nhwc's
// layout and step1_static_quantize_pack_int4_fprop_silu's semantics. Requires
// channels-per-group even so a pair never straddles a group boundary.
template <typename TIn>
__global__ void group_norm_silu_delta_quantize_pack_nhwc_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,   // [N,H,W,C] fp16 channels_last, in place
    int8_t* __restrict__ Yqp,           // [N,H,W,C/2] packed int4
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int C, long HW, int G, float eps, bool apply_silu
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    __half* cache_base = a_hat_cache + (long)n * HW * C;
    int8_t* yqp_base = Yqp + (long)n * ((HW * (long)C) / 2);

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

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
        float var = fmaxf(s_sumsq[0] / (float)group_size - mean * mean, 0.0f);
        mean_s = mean;
        inv_std_s = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;

    const int HALF_CPG = CPG / 2;
    const long pairs = group_size / 2;
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
        float o0 = apply_silu ? gns_silu(__half2float(__float2half(n0))) : __half2float(__float2half(n0));
        float o1 = apply_silu ? gns_silu(__half2float(__float2half(n1))) : __half2float(__float2half(n1));
        if (smooth_inv != nullptr) {
            o0 *= smooth_inv[c_global0];
            o1 *= smooth_inv[c_global0 + 1];
        }
        float c0 = __half2float(cache_base[mem_idx0]);
        float c1 = __half2float(cache_base[mem_idx0 + 1]);
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((o0 - c0) * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf((o1 - c1) * scale)));
        cache_base[mem_idx0]     = __float2half_rn(c0 + q0 * inv_scale);
        cache_base[mem_idx0 + 1] = __float2half_rn(c1 + q1 * inv_scale);
        int8_t i0 = (int8_t)q0;
        int8_t i1 = (int8_t)q1;
        yqp_base[mem_idx0 / 2] = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
    }
}

// Host wrapper: MoDiff GN(+mod)+SiLU + int4 delta-quantize+pack + a_hat update.
torch::Tensor group_norm_silu_delta_quantize_pack_nhwc(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor a_hat_cache,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,
    torch::Tensor mod_shift
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_delta_quantize_pack_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_delta_quantize_pack_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_pack_nhwc: only float32 and float16 are supported");
    TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_pack_nhwc: a_hat_cache must be fp16 (calibrated modiff path)");
    TORCH_CHECK(a_hat_cache.sizes() == x.sizes(),
                "group_norm_silu_delta_quantize_pack_nhwc: a_hat_cache must match x shape");
    const bool has_mod = mod_scale.numel() > 0;
    TORCH_CHECK(!has_mod || (mod_scale.scalar_type() == x.scalar_type() && mod_shift.scalar_type() == x.scalar_type()),
                "group_norm_silu_delta_quantize_pack_nhwc: mod_scale/mod_shift dtype must match input dtype");

    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "group_norm_silu_delta_quantize_pack_nhwc: num_channels must be divisible by num_groups");
    const int CPG = C / (int)num_groups;
    TORCH_CHECK(C % 2 == 0 && CPG % 2 == 0,
                "group_norm_silu_delta_quantize_pack_nhwc: channels and channels-per-group must be even");
    const long HW = (long)H * W;
    const long group_size = (long)CPG * HW;

    auto yqp = torch::empty({N, H, W, C / 2},
                            torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    __half* cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());

    if (x.scalar_type() == torch::kFloat32) {
        group_norm_silu_delta_quantize_pack_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu);
    } else {
        group_norm_silu_delta_quantize_pack_nhwc_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu);
    }
    return yqp;
}
