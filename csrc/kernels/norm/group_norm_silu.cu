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
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>

#include "common.cuh"

__device__ __forceinline__ float gn_load(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float gn_load(const __half* p, long i) { return __half2float(p[i]); }
__device__ __forceinline__ void gn_store(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void gn_store(__half* p, long i, float v) { p[i] = __float2half(v); }

// 2-wide (vectorized) counterparts of gn_load/gn_store. Caller guarantees `i` is even
// relative to `p`'s own base pointer, so the reinterpret-cast to float2/__half2 lands on
// a naturally-aligned address (every offset used by the vec2 kernels below -- n*HW*C,
// c_start, pair bases -- is a multiple of an even quantity given C/CPG are even).
__device__ __forceinline__ float2 gn_load2(const float* p, long i) {
    return reinterpret_cast<const float2*>(p)[i >> 1];
}
__device__ __forceinline__ float2 gn_load2(const __half* p, long i) {
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}
__device__ __forceinline__ void gn_store2(float* p, long i, float2 v) {
    reinterpret_cast<float2*>(p)[i >> 1] = v;
}
__device__ __forceinline__ void gn_store2(__half* p, long i, float2 v) {
    reinterpret_cast<__half2*>(p)[i >> 1] = __float22half2_rn(v);
}

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

// Vectorized (half2/float2) counterpart of group_norm_silu_quantize_nhwc_kernel. Pass 1
// (reduction) is byte-for-byte identical to the scalar kernel above -- deliberately NOT
// vectorized here (see the plan's Cycle-3 note: vectorizing the read side would change
// the per-thread index partition and hence fp32 summation order, which is a real
// numerics risk kept isolated to its own gated cycle). Only pass 2 (apply+quantize,
// order-independent) is vectorized, using the same pair-major index math
// group_norm_silu_quantize_pack_nhwc_vec2_kernel's pass 2 already uses. Caller (the host
// wrapper) only dispatches here when CPG is even, so a channel pair never straddles a
// group boundary and shares one mean/inv_std.
template <typename TIn, bool VEC_REDUCE = false>
__global__ void group_norm_silu_quantize_nhwc_vec2_kernel(
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

    float local_sum = 0.0f, local_sumsq = 0.0f;
    if constexpr (VEC_REDUCE) {
        // Attention's static-quantized QKV producer tolerates the normal one-code rounding
        // freedom of INT8 quantization. Pair-major reduction halves address/loop/load work;
        // CPG is even, so no pair crosses a GroupNorm group boundary.
        const int HALF_CPG = CPG / 2;
        const long pairs = group_size / 2;
        for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
            const int cpair = pidx % HALF_CPG;
            const long hw = pidx / HALF_CPG;
            const long mem_idx0 = hw * C + c_start + 2 * cpair;
            const float2 v = gn_load2(x_base, mem_idx0);
            local_sum += v.x + v.y;
            local_sumsq += v.x * v.x + v.y * v.y;
        }
    } else {
        // Compatibility path: preserve the scalar reference's summation partition exactly.
        for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
            int c_local = idx % CPG;
            long hw = idx / CPG;
            long mem_idx = hw * C + c_start + c_local;
            float v = gn_load(x_base, mem_idx);
            local_sum += v;
            local_sumsq += v * v;
        }
    }
    __shared__ float mean_s, inv_std_s;
    if constexpr (VEC_REDUCE) {
        // The fast attention-only path already permits reduction-order freedom. Reduce within
        // each warp in registers, then reduce the at-most-32 warp partials with one warp. The
        // generic shared-memory tree needs log2(blockDim) full-CTA barriers (10 at the production
        // 1024-thread blocks); this needs two.
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
            local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, off);
        }
        if (lane == 0) {
            s_sum[warp] = local_sum;
            s_sumsq[warp] = local_sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float block_sum = lane < nwarp ? s_sum[lane] : 0.0f;
            float block_sumsq = lane < nwarp ? s_sumsq[lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                block_sum += __shfl_down_sync(0xffffffff, block_sum, off);
                block_sumsq += __shfl_down_sync(0xffffffff, block_sumsq, off);
            }
            if (lane == 0) {
                float mean = block_sum / (float)group_size;
                float var = block_sumsq / (float)group_size - mean * mean;
                var = fmaxf(var, 0.0f);
                mean_s = mean;
                inv_std_s = rsqrtf(var + eps);
            }
        }
    } else {
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
        if (threadIdx.x == 0) {
            float mean = s_sum[0] / (float)group_size;
            float var = s_sumsq[0] / (float)group_size - mean * mean;
            var = fmaxf(var, 0.0f);
            mean_s = mean;
            inv_std_s = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;

    // Pass 2: pair-major (vectorized). One thread handles a (even,odd) channel pair at
    // one spatial position -- same index math as group_norm_silu_quantize_pack_nhwc_vec2_kernel's
    // pass 2, but writes two UNPACKED int8 bytes (one int16 store) instead of one nibble byte.
    const int HALF_CPG = CPG / 2;
    const long pairs = group_size / 2;
    for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
        int cpair = pidx % HALF_CPG;
        long hw = pidx / HALF_CPG;
        int c_global0 = c_start + 2 * cpair;
        long mem_idx0 = hw * (long)C + c_global0;

        float2 v = gn_load2(x_base, mem_idx0);
        float2 w = gn_load2(gamma, c_global0);
        float2 b = gn_load2(beta, c_global0);
        float n0 = (v.x - mean) * inv_std * w.x + b.x;
        float n1 = (v.y - mean) * inv_std * w.y + b.y;
        if (mod_scale != nullptr) {
            long midx0 = (long)n * C + c_global0;
            float2 ms = gn_load2(mod_scale, midx0);
            float2 sh = gn_load2(mod_shift, midx0);
            n0 = n0 * (1.0f + ms.x) + sh.x;
            n1 = n1 * (1.0f + ms.y) + sh.y;
        }
        float o0 = apply_silu ? (n0 / (1.0f + expf(-n0))) : n0;
        float o1 = apply_silu ? (n1 / (1.0f + expf(-n1))) : n1;
        if (smooth_inv != nullptr) {
            o0 *= smooth_inv[c_global0];
            o1 *= smooth_inv[c_global0 + 1];
        }
        int8_t i0 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(o0 * scale)));
        int8_t i1 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(o1 * scale)));
        reinterpret_cast<int16_t*>(yq_base)[mem_idx0 >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
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
static torch::Tensor group_norm_silu_quantize_nhwc_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,   // [N, C] scale-shift modulation, or empty for none
    torch::Tensor mod_shift,
    bool fast_reduce
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
    if (fast_reduce) {
        // Pair-major pass 1 has group_size/2 elements. About six pairs/thread gives the best
        // latency/occupancy balance on A40: 512 threads at group_size=6144 (T=1024),
        // 256 at 3072 (T=256), and 128 for the smaller attention blocks. The old generic
        // heuristic launched 1024 threads and was 1.27-4.3x slower after warp reductions.
        block_size = 128;
        while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;
        const char* ft = getenv("MODIFF_GN_FAST_THREADS");
        if (ft) {
            const int v = atoi(ft);
            if (v == 128 || v == 256 || v == 512 || v == 1024) block_size = v;
        }
    }

    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    // Pass-2 vectorization requires a channel pair to never straddle a group boundary
    // (both channels then share one mean/inv_std) -- no existing TORCH_CHECK enforces
    // this (unlike the int4-pack sibling), so this is a genuine runtime fallback, not
    // just documentation. Real model configs always have even CPG; this dispatch is
    // exercised by gn_static_quantize_verify.py's synthetic odd-CPG case.
    const bool use_vec2 = (CPG % 2 == 0);

    if (x.scalar_type() == torch::kFloat32) {
        if (use_vec2) {
            if (fast_reduce)
                group_norm_silu_quantize_nhwc_vec2_kernel<float, true><<<grid, block, shmem_bytes, stream>>>(
                    x.data_ptr<float>(), reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    weight.data_ptr<float>(), bias.data_ptr<float>(),
                    has_mod ? mod_scale.data_ptr<float>() : nullptr,
                    has_mod ? mod_shift.data_ptr<float>() : nullptr,
                    scale.data_ptr<float>(), smooth_ptr,
                    C, HW, (int)num_groups, (float)eps, apply_silu);
            else
                group_norm_silu_quantize_nhwc_vec2_kernel<float, false><<<grid, block, shmem_bytes, stream>>>(
                    x.data_ptr<float>(), reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    weight.data_ptr<float>(), bias.data_ptr<float>(),
                    has_mod ? mod_scale.data_ptr<float>() : nullptr,
                    has_mod ? mod_shift.data_ptr<float>() : nullptr,
                    scale.data_ptr<float>(), smooth_ptr,
                    C, HW, (int)num_groups, (float)eps, apply_silu);
        } else {
            group_norm_silu_quantize_nhwc_kernel<float><<<grid, block, shmem_bytes, stream>>>(
                x.data_ptr<float>(), reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                scale.data_ptr<float>(), smooth_ptr,
                C, HW, (int)num_groups, (float)eps, apply_silu
            );
        }
    } else {
        if (use_vec2) {
            if (fast_reduce)
                group_norm_silu_quantize_nhwc_vec2_kernel<__half, true><<<grid, block, shmem_bytes, stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                    has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                    has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                    scale.data_ptr<float>(), smooth_ptr,
                    C, HW, (int)num_groups, (float)eps, apply_silu);
            else
                group_norm_silu_quantize_nhwc_vec2_kernel<__half, false><<<grid, block, shmem_bytes, stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                    has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                    has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                    scale.data_ptr<float>(), smooth_ptr,
                    C, HW, (int)num_groups, (float)eps, apply_silu);
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
    }

    return yq;
}

torch::Tensor group_norm_silu_quantize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift) {
    return group_norm_silu_quantize_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, false);
}

torch::Tensor group_norm_silu_quantize_nhwc_fast(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift) {
    return group_norm_silu_quantize_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, true);
}

// ============================================================================
// STATUS: built, verified correct, measured faster -- but NOT currently reachable from
// Python, by design. Not dead code; do not delete.
//
//   Validated (docs/comprehensive_benchmark_2026-07-15/REPORT.md §"int8-conv-output"):
//   matches the fp16-input path to <=1 int8 code on 100% of elements, and runs
//   ~1.03-1.09x faster (it reads half the bytes). Quality of the conv->int8->GN handoff
//   was separately measured via the MODIFF_CONV_INT8_OUT fake-quant probe in
//   integration/fused_ops/fused_resblock.py: only +0.0023..0.0033 rel-err, far inside the
//   0.02 gate. So both halves of the idea check out.
//
//   Blocked on the CONV side, not here: to feed this kernel, the upstream conv must write
//   int8 directly. The existing int8-output path (forward_to_int8 -> relu_requant)
//   materializes an fp16 scratch tensor first, so end-to-end it moves MORE bytes, not fewer
//   (measured 0.83-0.97x, i.e. a slowdown). Realizing the win needs a direct-int8-output
//   CUTLASS conv epilogue (int32 acc -> dequant -> round/clamp -> int8, no fp16 scratch),
//   whose mixed-type output is non-trivial in CUTLASS. Projected e2e gain ~2%; deferred as
//   a low effort/payoff item. This kernel is the finished half, kept ready for that day.
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

// INT4-emitting GroupNorm(+SiLU)+quantize+pack, half2/float2-vectorized.
//
// Identical GroupNorm(+SiLU) math to group_norm_silu_quantize_nhwc_vec2_kernel, but pass 2
// quantizes to int4 codes in [-7,7] and packs adjacent-channel pairs into one byte, producing
// output byte-identical to scale_quantize_and_pack(SiLU(GN(x)), scale) so the following
// calibrated INT4 conv can read it directly. Packing is along the (contiguous) NHWC channel
// dim: byte = flat_element/2, low nibble = even channel, high nibble = odd channel -- exactly
// quantize.cu::scale_quantize_pack_kernel's convention.
//
// Requires channels-per-group (CPG) even, so a channel pair never straddles a group boundary
// (both channels then share one group's mean/inv_std); the host wrapper enforces this and the
// Python caller gates on it, falling back to the two-kernel path otherwise. Pass 2 is
// naturally pair-major (one thread per output byte = one channel pair), so vectorizing it
// needed no loop restructuring -- just gn_load2 in place of per-element loads. Pass 1 (the
// stats reduction) is deliberately NOT vectorized: see the Cycle-3 note on
// gn_group_stats_vec2_kernel below.
template <typename TIn, bool FAST_REDUCE = false>
__global__ void group_norm_silu_quantize_pack_nhwc_vec2_kernel(
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
    bool apply_silu,
    int Kpad                       // padded row width in CHANNELS (>= C, even); == C for no padding
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int KpadH = Kpad / 2;    // bytes per spatial position in the output

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    int8_t* yqp_base = Yqp + (long)n * (HW * (long)KpadH);

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    float local_sum = 0.0f, local_sumsq = 0.0f;
    if constexpr (FAST_REDUCE) {
        const int HALF_CPG = CPG / 2;
        const long pairs = group_size / 2;
        for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
            const int cpair = pidx % HALF_CPG;
            const long hw = pidx / HALF_CPG;
            const long mem_idx0 = hw * C + c_start + 2 * cpair;
            const float2 v = gn_load2(x_base, mem_idx0);
            local_sum += v.x + v.y;
            local_sumsq += v.x * v.x + v.y * v.y;
        }
    } else {
        for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
            int c_local = idx % CPG;
            long hw = idx / CPG;
            long mem_idx = hw * C + c_start + c_local;
            float v = gn_load(x_base, mem_idx);
            local_sum += v;
            local_sumsq += v * v;
        }
    }
    __shared__ float mean_s, inv_std_s;
    if constexpr (FAST_REDUCE) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
            local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, off);
        }
        if (lane == 0) {
            s_sum[warp] = local_sum;
            s_sumsq[warp] = local_sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float block_sum = lane < nwarp ? s_sum[lane] : 0.0f;
            float block_sumsq = lane < nwarp ? s_sumsq[lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                block_sum += __shfl_down_sync(0xffffffff, block_sum, off);
                block_sumsq += __shfl_down_sync(0xffffffff, block_sumsq, off);
            }
            if (lane == 0) {
                float mean = block_sum / (float)group_size;
                float var = block_sumsq / (float)group_size - mean * mean;
                mean_s = mean;
                inv_std_s = rsqrtf(fmaxf(var, 0.0f) + eps);
            }
        }
    } else {
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
        if (threadIdx.x == 0) {
            float mean = s_sum[0] / (float)group_size;
            float var = s_sumsq[0] / (float)group_size - mean * mean;
            mean_s = mean;
            inv_std_s = rsqrtf(fmaxf(var, 0.0f) + eps);
        }
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;

    const int HALF_CPG = CPG / 2;
    const long pairs = group_size / 2;
    for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
        int cpair = pidx % HALF_CPG;
        long hw = pidx / HALF_CPG;
        int c_global0 = c_start + 2 * cpair;
        long mem_idx0 = hw * (long)C + c_global0;

        float2 v = gn_load2(x_base, mem_idx0);
        float2 w = gn_load2(gamma, c_global0);
        float2 b = gn_load2(beta, c_global0);
        float n0 = (v.x - mean) * inv_std * w.x + b.x;
        float n1 = (v.y - mean) * inv_std * w.y + b.y;
        if (mod_scale != nullptr) {
            long midx0 = (long)n * C + c_global0;
            float2 ms = gn_load2(mod_scale, midx0);
            float2 sh = gn_load2(mod_shift, midx0);
            n0 = n0 * (1.0f + ms.x) + sh.x;
            n1 = n1 * (1.0f + ms.y) + sh.y;
        }
        float o0 = apply_silu ? (n0 / (1.0f + expf(-n0))) : n0;
        float o1 = apply_silu ? (n1 / (1.0f + expf(-n1))) : n1;
        if (smooth_inv != nullptr) {
            o0 *= smooth_inv[c_global0];
            o1 *= smooth_inv[c_global0 + 1];
        }
        int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o0 * scale)));
        int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o1 * scale)));
        // Row width is KpadH bytes, not C/2: with Kpad == C this is exactly the old mem_idx0/2
        // (mem_idx0 = hw*C + c_global0 and c_global0 is even), so the no-pad path is bit-identical.
        // Channels >= C are never visited by any group, so the pad bytes are left at the zero the
        // host pre-filled them with.
        yqp_base[hw * (long)KpadH + (c_global0 >> 1)] = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
    }
    // Only group zero clears the padded tail. This replaces a full-output torch::zeros
    // initialization with writes to the bytes that actually require zeroing.
    if (g == 0 && Kpad > C) {
        const int tail_bytes = KpadH - C / 2;
        for (long idx = threadIdx.x; idx < HW * (long)tail_bytes; idx += blockDim.x) {
            const long hw = idx / tail_bytes;
            const int pb = idx % tail_bytes;
            yqp_base[hw * (long)KpadH + C / 2 + pb] = 0;
        }
    }
}

// Fused GroupNorm+SiLU+quantize+2x resize, one launch, for the updown ResBlock path.
// Pass 1 (the stats reduction) is copied verbatim from the non-resizing sibling; only pass 2
// differs. UP=true does nearest 2x upsample, which is an exact index-select, so a value is
// computed once and stored to the four output positions. UP=false does 2x2 average pooling
// and is the reason this has to be ONE kernel: averaging must happen on the fp32 post-SiLU
// values BEFORE quantization. Splitting it (quantize, then average the int4 codes) rounds
// each of the four inputs first and averages the rounded values, which is a different and
// worse result -- that non-commutation is exactly why the shipped pipeline keeps GroupNorm at
// fp16 and defers the quantize to the resize kernel instead.
template <typename TIn, bool FAST_REDUCE, bool UP>
__global__ void group_norm_silu_quantize_pack_resize_nhwc_kernel(
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
    bool apply_silu,
    int Kpad,                      // padded row width in CHANNELS (>= C, even); == C for no padding
    int W                          // INPUT width, needed to map hw -> (h, w) for the resize
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int KpadH = Kpad / 2;    // bytes per spatial position in the output

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    const long HW_OUT = UP ? (HW * 4) : (HW / 4);
    int8_t* yqp_base = Yqp + (long)n * (HW_OUT * (long)KpadH);

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    float local_sum = 0.0f, local_sumsq = 0.0f;
    if constexpr (FAST_REDUCE) {
        const int HALF_CPG = CPG / 2;
        const long pairs = group_size / 2;
        for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
            const int cpair = pidx % HALF_CPG;
            const long hw = pidx / HALF_CPG;
            const long mem_idx0 = hw * C + c_start + 2 * cpair;
            const float2 v = gn_load2(x_base, mem_idx0);
            local_sum += v.x + v.y;
            local_sumsq += v.x * v.x + v.y * v.y;
        }
    } else {
        for (long idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
            int c_local = idx % CPG;
            long hw = idx / CPG;
            long mem_idx = hw * C + c_start + c_local;
            float v = gn_load(x_base, mem_idx);
            local_sum += v;
            local_sumsq += v * v;
        }
    }
    __shared__ float mean_s, inv_std_s;
    if constexpr (FAST_REDUCE) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
            local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, off);
        }
        if (lane == 0) {
            s_sum[warp] = local_sum;
            s_sumsq[warp] = local_sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float block_sum = lane < nwarp ? s_sum[lane] : 0.0f;
            float block_sumsq = lane < nwarp ? s_sumsq[lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                block_sum += __shfl_down_sync(0xffffffff, block_sum, off);
                block_sumsq += __shfl_down_sync(0xffffffff, block_sumsq, off);
            }
            if (lane == 0) {
                float mean = block_sum / (float)group_size;
                float var = block_sumsq / (float)group_size - mean * mean;
                mean_s = mean;
                inv_std_s = rsqrtf(fmaxf(var, 0.0f) + eps);
            }
        }
    } else {
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
        if (threadIdx.x == 0) {
            float mean = s_sum[0] / (float)group_size;
            float var = s_sumsq[0] / (float)group_size - mean * mean;
            mean_s = mean;
            inv_std_s = rsqrtf(fmaxf(var, 0.0f) + eps);
        }
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;

    const int HALF_CPG = CPG / 2;
    const int Wi = W, Hi = (int)(HW / W);

    // Compute one channel pair's post-SiLU fp32 values at input position hw.
    auto compute_pair = [&](long hw, int c_global0, float& o0, float& o1) {
        const long mem_idx0 = hw * (long)C + c_global0;
        float2 v = gn_load2(x_base, mem_idx0);
        float2 wgt = gn_load2(gamma, c_global0);
        float2 b = gn_load2(beta, c_global0);
        float n0 = (v.x - mean) * inv_std * wgt.x + b.x;
        float n1 = (v.y - mean) * inv_std * wgt.y + b.y;
        if (mod_scale != nullptr) {
            const long midx0 = (long)n * C + c_global0;
            float2 ms = gn_load2(mod_scale, midx0);
            float2 sh = gn_load2(mod_shift, midx0);
            n0 = n0 * (1.0f + ms.x) + sh.x;
            n1 = n1 * (1.0f + ms.y) + sh.y;
        }
        o0 = apply_silu ? (n0 / (1.0f + expf(-n0))) : n0;
        o1 = apply_silu ? (n1 / (1.0f + expf(-n1))) : n1;
        if (smooth_inv != nullptr) {
            o0 *= smooth_inv[c_global0];
            o1 *= smooth_inv[c_global0 + 1];
        }
    };

    if constexpr (UP) {
        const int Wo = Wi * 2;
        const long pairs = group_size / 2;                 // iterate INPUT positions
        for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
            const int cpair = pidx % HALF_CPG;
            const long hw = pidx / HALF_CPG;
            const int c_global0 = c_start + 2 * cpair;
            float o0, o1;
            compute_pair(hw, c_global0, o0, o1);
            const int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o0 * scale)));
            const int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o1 * scale)));
            const int8_t packed = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
            const int h = (int)(hw / Wi), w = (int)(hw % Wi);
            const int byte_c = c_global0 >> 1;
#pragma unroll
            for (int dy = 0; dy < 2; ++dy)
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                    const long hw_out = (long)(2 * h + dy) * Wo + (2 * w + dx);
                    yqp_base[hw_out * (long)KpadH + byte_c] = packed;
                }
        }
    } else {
        const int Wo = Wi / 2, Ho = Hi / 2;
        const long pairs_out = (long)Ho * Wo * HALF_CPG;   // iterate OUTPUT positions
        for (long pidx = threadIdx.x; pidx < pairs_out; pidx += blockDim.x) {
            const int cpair = pidx % HALF_CPG;
            const long hwo = pidx / HALF_CPG;
            const int ho = (int)(hwo / Wo), wo = (int)(hwo % Wo);
            const int c_global0 = c_start + 2 * cpair;
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int dy = 0; dy < 2; ++dy)
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                    float o0, o1;
                    compute_pair((long)(2 * ho + dy) * Wi + (2 * wo + dx), c_global0, o0, o1);
                    a0 += o0;
                    a1 += o1;
                }
            a0 *= 0.25f;                                   // average BEFORE quantizing
            a1 *= 0.25f;
            const int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(a0 * scale)));
            const int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(a1 * scale)));
            yqp_base[hwo * (long)KpadH + (c_global0 >> 1)] =
                (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
        }
    }

    if (g == 0 && Kpad > C) {
        const int tail_bytes = KpadH - C / 2;
        for (long idx = threadIdx.x; idx < HW_OUT * (long)tail_bytes; idx += blockDim.x) {
            const long hw = idx / tail_bytes;
            const int pb = idx % tail_bytes;
            yqp_base[hw * (long)KpadH + C / 2 + pb] = 0;
        }
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
static torch::Tensor group_norm_silu_quantize_pack_nhwc_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps,
    bool apply_silu,
    torch::Tensor scale,
    torch::Tensor smooth_inv,
    torch::Tensor mod_scale,   // [N, C] scale-shift modulation, or empty for none
    torch::Tensor mod_shift,
    int64_t k_pad,             // padded row width in channels for the int4 GEMM; <=0 or ==C -> no pad
    bool fast_reduce
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

    // [N, H, W, Kpad/2] contiguous. With Kpad == C this is the same flat byte order as
    // scale_quantize_and_pack (the historical behaviour). With Kpad > C the extra channels are the
    // int4 GEMM's K zero-padding, which lets the C=192 attention blocks (K 192 -> 256) stay on the
    // fused GN->pack path instead of paying group_norm_silu_nhwc + F.pad + a standalone
    // quantize_act_int4_pack -- mirrors quantize_attn_out_int4_pack's k_pad. The g==0 CTA clears
    // only the padded tail of every row, so a full-output zero fill is unnecessary.
    const int Kpad = (k_pad > (int64_t)C) ? (int)k_pad : C;
    TORCH_CHECK(Kpad % 2 == 0, "group_norm_silu_quantize_pack_nhwc: k_pad must be even");
    auto opts = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto yqp = torch::empty({N, H, W, Kpad / 2}, opts);

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;
    if (fast_reduce) {
        block_size = 128;
        while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;
    }

    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;

    // C%2==0 && CPG%2==0 already TORCH_CHECK'd above -> always safe to use the
    // vectorized kernel here (unlike the non-pack sibling, no scalar fallback needed).
    if (x.scalar_type() == torch::kFloat32) {
        if (fast_reduce)
          group_norm_silu_quantize_pack_nhwc_vec2_kernel<float, true><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad);
        else
          group_norm_silu_quantize_pack_nhwc_vec2_kernel<float, false><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad
        );
    } else {
        if (fast_reduce)
          group_norm_silu_quantize_pack_nhwc_vec2_kernel<__half, true><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad);
        else
          group_norm_silu_quantize_pack_nhwc_vec2_kernel<__half, false><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return yqp;
}

// Host side of the fused GN+SiLU+quantize+resize kernel. `resize` is +1 for a nearest 2x
// upsample and -1 for a 2x2 average pool; the output is [N, H*2, W*2, Kpad/2] or
// [N, H/2, W/2, Kpad/2]. Deliberately a separate entry point from the non-resizing sibling: it
// is a prototype for the updown ResBlock path and is not wired into the pipeline.
torch::Tensor group_norm_silu_quantize_pack_resize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "gn_quantize_pack_resize expects [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kFloat32,
                "gn_quantize_pack_resize: only fp16/fp32 supported");
    TORCH_CHECK(resize == 1 || resize == -1, "gn_quantize_pack_resize: resize must be +1 or -1");
    const bool has_mod = mod_scale.numel() > 0;
    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "channels must divide into groups");
    const int CPG = C / (int)num_groups;
    TORCH_CHECK(C % 2 == 0 && CPG % 2 == 0, "channels and channels-per-group must be even");
    const bool up = (resize == 1);
    if (!up) TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "downsample needs even H and W");
    const long HW = (long)H * W;
    const long group_size = (long)CPG * HW;
    const int Kpad = (k_pad > (int64_t)C) ? (int)k_pad : C;
    TORCH_CHECK(Kpad % 2 == 0, "k_pad must be even");
    const int Ho = up ? H * 2 : H / 2, Wo = up ? W * 2 : W / 2;
    auto yqp = torch::empty({N, Ho, Wo, Kpad / 2},
                            torch::TensorOptions().dtype(torch::kInt8).device(x.device()));

    int block_size = 128;
    while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups)), block((unsigned int)block_size);
    size_t shmem = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;

    // gn_load2 is overloaded for `const float*` and `const __half*` only, so the fp16 launch has
    // to reinterpret at::Half -- instantiating the kernel on at::Half itself does not compile.
#define MODIFF_GNQR_LAUNCH(T, ATT, UPV)                                                     \
    group_norm_silu_quantize_pack_resize_nhwc_kernel<T, true, UPV>                          \
        <<<grid, block, shmem, stream>>>(                                                   \
            reinterpret_cast<const T*>(x.data_ptr<ATT>()),                                  \
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),                              \
            reinterpret_cast<const T*>(weight.data_ptr<ATT>()),                             \
            reinterpret_cast<const T*>(bias.data_ptr<ATT>()),                               \
            has_mod ? reinterpret_cast<const T*>(mod_scale.data_ptr<ATT>()) : nullptr,      \
            has_mod ? reinterpret_cast<const T*>(mod_shift.data_ptr<ATT>()) : nullptr,      \
            scale.data_ptr<float>(), smooth_ptr, C, HW, (int)num_groups, (float)eps,        \
            apply_silu, Kpad, W)

    if (x.scalar_type() == torch::kFloat32) {
        if (up) MODIFF_GNQR_LAUNCH(float, float, true);
        else    MODIFF_GNQR_LAUNCH(float, float, false);
    } else {
        if (up) MODIFF_GNQR_LAUNCH(__half, at::Half, true);
        else    MODIFF_GNQR_LAUNCH(__half, at::Half, false);
    }
#undef MODIFF_GNQR_LAUNCH
    C10_CUDA_CHECK(cudaGetLastError());
    return yqp;
}

torch::Tensor group_norm_silu_quantize_pack_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad) {
    return group_norm_silu_quantize_pack_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, false);
}

torch::Tensor group_norm_silu_quantize_pack_nhwc_fast(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad) {
    return group_norm_silu_quantize_pack_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, true);
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
// SPLIT INTO TWO KERNELS (stats + flat apply) rather than one group-major kernel.
// The earlier single fused kernel did its whole pass 2 -- including the fp16 a_hat
// read-modify-write -- group-major (one block per (sample,group)), so consecutive
// threads walked contiguous runs of only CPG channels before jumping C elements to
// the next spatial position. At the dominant low-CPG / high-spatial shapes (CPG=4
// at C=128, 8 at C=256) that fragments every a_hat load+store into ~4-8x the DRAM
// sectors; the penalty on the fp16 a_hat traffic (read AND write, tensor-sized) beat
// the `normed` round-trip the fusion saved, so it measured a ~2-3 ms/step e2e
// REGRESSION (docs/benchmark_5mode_2026-07-20, fused_resblock.py).
//
// The split fixes the coalescing while still removing the `normed` intermediate:
//   1. gn_group_stats_kernel -- group-major reduction, reads x once, emits only a
//      tiny [N,G] mean/inv_std buffer (the strided read is inherent to any GN
//      reduction and present in every variant).
//   2. gn_apply_delta_quantize[_pack]_flat_kernel -- element-major grid-stride, so
//      x / a_hat / Yq are all contiguous per warp exactly like the standalone step1
//      kernel it subsumes, while doing the GN affine + mod + SiLU + delta-quantize
//      inline (no fp16 `normed` tensor materialized).
//
// Bit-exactness vs the two-kernel default (group_norm_silu_nhwc(apply_silu=False) ->
// step1_static_quantize[_pack]_fprop_silu): kernel 1's reduction is identical to
// group_norm_silu_nhwc_kernel (same block_size, same fp32 tree -> same mean/inv_std,
// exchanged losslessly through the fp32 buffer), and kernel 2 replicates the fp16
// rounding of `normed` BEFORE SiLU (__float2half then back) so the SiLU input --
// hence the int8/int4 code and the a_hat update (cache += q/scale, stored fp16) --
// matches element-for-element. a_hat_cache is fp16 (the only dtype the calibrated
// production path uses; enforced by the step1_silu reference and TORCH_CHECK'd here).
// =========================================================================
__device__ __forceinline__ float gns_silu(float v) { return v / (1.0f + expf(-v)); }

// Kernel 1: per-(sample,group) mean + inv_std. Reduction is byte-for-byte identical
// to group_norm_silu_nhwc_kernel's pass 1 (must be, for bit-exact means). grid = N*G,
// so blockIdx.x indexes the [N*G] mean/inv_std outputs directly.
template <typename TIn>
__global__ void gn_group_stats_kernel(
    const TIn* __restrict__ X,
    float* __restrict__ mean_out,      // [N*G]
    float* __restrict__ inv_std_out,   // [N*G]
    int C, long HW, int G, float eps
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;
    const TIn* x_base = X + (long)n * HW * C;

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
    if (threadIdx.x == 0) {
        // Statement sequence kept textually identical to group_norm_silu_nhwc_kernel
        // (same translation unit) so nvcc emits the same fp32 codegen / FMA
        // contraction -> bit-identical mean/inv_std, hence a bit-identical `normed`
        // downstream. (A one-liner fmaxf(... , 0) here perturbed var by ~1 ULP at
        // large group sizes, flipping the occasional fine int8 code.)
        float mean = s_sum[0] / (float)group_size;
        float var = s_sumsq[0] / (float)group_size - mean * mean;
        var = fmaxf(var, 0.0f);
        mean_out[blockIdx.x] = mean;
        inv_std_out[blockIdx.x] = rsqrtf(var + eps);
    }
}

// ============================================================================
// UNREFERENCED ON PURPOSE -- do not wire this in. This is a FAILED experiment kept as
// executable documentation of *why* the GN stats reduction stays scalar.
//
// Retention rule for this codebase: a superseded-but-correct scalar kernel is deleted once
// its vectorized replacement ships (git history has it). A kernel is only kept unreferenced
// when the *reason it isn't used* is a correctness finding worth not rediscovering -- this
// one, and it alone, meets that bar.
//
// What it is: pair-major vectorized counterpart of gn_group_stats_kernel. Reassigns which elements each thread
// sums (pair-major instead of strided-across-threads), which changes fp32 addition
// order vs the scalar kernel. This file's OWN comment above already documented that a
// MUCH smaller perturbation (a one-line fmaxf reordering) previously flipped occasional
// int8 codes via a ~1 ULP variance change -- and sure enough, wiring this kernel in
// (gated on CPG%2==0) passed gn_modiff_verify_kernel.py's random-data check but FAILED
// gn_modiff_verify_realinput.py with max_code_diff=1. Reverted; gn_launch_group_stats
// below unconditionally calls the scalar gn_group_stats_kernel again.
template <typename TIn>
__global__ void gn_group_stats_vec2_kernel(
    const TIn* __restrict__ X,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int C, long HW, int G, float eps
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;
    const TIn* x_base = X + (long)n * HW * C;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    float local_sum = 0.0f, local_sumsq = 0.0f;
    const long pairs = group_size / 2;
    for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
        long idx0 = pidx * 2;
        int c_local0 = (int)(idx0 % CPG);
        long hw = idx0 / CPG;
        long mem_idx0 = hw * C + c_start + c_local0;
        float2 v = gn_load2(x_base, mem_idx0);
        local_sum += v.x;
        local_sumsq += v.x * v.x;
        local_sum += v.y;
        local_sumsq += v.y * v.y;
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
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / (float)group_size;
        float var = s_sumsq[0] / (float)group_size - mean * mean;
        var = fmaxf(var, 0.0f);
        mean_out[blockIdx.x] = mean;
        inv_std_out[blockIdx.x] = rsqrtf(var + eps);
    }
}

// Launch kernel 1 (dtype-dispatched). block_size formula MUST match
// group_norm_silu_nhwc / group_norm_silu_nhwc_kernel so the fp32 reduction tree --
// and therefore the mean/inv_std -- is bit-identical to the two-kernel reference.
// --- Alternate-order (atomic) group stats: element-major grid-stride atomicAdd of sum/sumsq per
// (sample,group), then finalize. Same math, DIFFERENT fp32 summation order than the group-major tree
// -- used (MODIFF_GN_STATS_ALT=1) to measure the o_hat DDIM drift a conv-epilogue-fused reduction
// would introduce, before committing to that fusion. ---
template <typename TIn>
__global__ void gn_stats_sum_kernel(const TIn* __restrict__ X, float* __restrict__ sum,
                                    int C, int G, long sample_stride, long num_elements) {
    const int CPG = C / G;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
         i += (long)blockDim.x * gridDim.x)
        atomicAdd(&sum[(i / sample_stride) * G + ((int)(i % C) / CPG)], gn_load(X, i));
}
template <typename TIn>
__global__ void gn_stats_var_kernel(const TIn* __restrict__ X, const float* __restrict__ mean,
                                    float* __restrict__ var, int C, int G, long sample_stride, long num_elements) {
    const int CPG = C / G;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
         i += (long)blockDim.x * gridDim.x) {
        long s = (i / sample_stride) * G + ((int)(i % C) / CPG);
        float d = gn_load(X, i) - mean[s];        // subtract mean BEFORE squaring -> stable
        atomicAdd(&var[s], d * d);
    }
}
__global__ void gn_mean_kernel(const float* __restrict__ sum, float* __restrict__ mean, long gs, int NG) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < NG) mean[i] = sum[i] / (float)gs;
}
__global__ void gn_invstd_kernel(const float* __restrict__ var, float* __restrict__ inv_std,
                                 long gs, float eps, int NG, float perturb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < NG) inv_std[i] = perturb / sqrtf(var[i] / (float)gs + eps);
}

// Single-pass merged variant (MODIFF_GN_STATS_ALT=2): element-major grid-stride, atomicAdd BOTH
// sum and sumsq for a given element in the same pass -- avoids gn_stats_sum_kernel + gn_stats_var_kernel's
// second read of X (the mean-first-then-resweep two-pass approach above), at the cost of the
// classic (less numerically stable, but here negligible per this file's header docstring)
// sum/sumsq -> var = E[x^2]-E[x]^2 formula instead of subtract-mean-before-square. Same reduction
// order (atomic, not the group-major tree) as the two-pass ALT variant, so drift vs the default
// tree reduction is bounded by that already-validated (07a99ca) probe.
template <typename TIn>
__global__ void gn_stats_sumsq_kernel(const TIn* __restrict__ X, float* __restrict__ sum,
                                      float* __restrict__ sumsq, int C, int G, long sample_stride,
                                      long num_elements) {
    const int CPG = C / G;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
         i += (long)blockDim.x * gridDim.x) {
        long s = (i / sample_stride) * G + ((int)(i % C) / CPG);
        float v = gn_load(X, i);
        atomicAdd(&sum[s], v);
        atomicAdd(&sumsq[s], v * v);
    }
}
__global__ void gn_finalize_sumsq_kernel(const float* __restrict__ sum, const float* __restrict__ sumsq,
                                         float* __restrict__ mean_out, float* __restrict__ inv_std_out,
                                         long gs, float eps, int NG) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NG) {
        float mean = sum[i] / (float)gs;
        float var = sumsq[i] / (float)gs - mean * mean;
        var = fmaxf(var, 0.0f);
        mean_out[i] = mean;
        inv_std_out[i] = rsqrtf(var + eps);
    }
}

static void gn_launch_group_stats(
    const torch::Tensor& x, int N, int C, long HW, int num_groups, double eps,
    torch::Tensor& mean, torch::Tensor& inv_std
) {
    static const char* _alt = std::getenv("MODIFF_GN_STATS_ALT");
    if (_alt != nullptr && _alt[0] == '2') {
        // Single-pass merged atomic variant: one grid-stride sweep over X atomicAdd'ing both sum
        // and sumsq (see gn_stats_sumsq_kernel) -- half the reads of the two-pass ALT=1 variant
        // below, since it never re-reads X to compute (x-mean)^2. Candidate replacement for the
        // default group-major tree kernel (measured ~9.4ms/step of the modiff-vs-baseline
        // GroupNorm gap in docs/benchmark_5mode_2026-07-23); kept opt-in until A/B'd across shapes.
        auto sopt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
        auto sum = torch::zeros({N * num_groups}, sopt);
        auto sumsq = torch::zeros({N * num_groups}, sopt);
        long num_elements = (long)N * C * HW, sample_stride = (long)C * HW;
        long group_size = (long)(C / num_groups) * HW;
        int ab = 256; unsigned int ag = (unsigned int)((num_elements + ab - 1) / ab);
        int NG = N * num_groups, fb = 128, fg = (NG + fb - 1) / fb;
        cudaStream_t st = at::cuda::getCurrentCUDAStream();
        if (x.scalar_type() == torch::kFloat32) {
            gn_stats_sumsq_kernel<float><<<ag, ab, 0, st>>>(x.data_ptr<float>(), sum.data_ptr<float>(), sumsq.data_ptr<float>(), C, num_groups, sample_stride, num_elements);
        } else {
            const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
            gn_stats_sumsq_kernel<__half><<<ag, ab, 0, st>>>(xp, sum.data_ptr<float>(), sumsq.data_ptr<float>(), C, num_groups, sample_stride, num_elements);
        }
        gn_finalize_sumsq_kernel<<<fg, fb, 0, st>>>(sum.data_ptr<float>(), sumsq.data_ptr<float>(),
                                                     mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                                                     group_size, (float)eps, NG);
        return;
    }
    if (_alt != nullptr && _alt[0] != '0') {
        // Stable two-pass, element-major atomic order (different fp32 order than the group-major
        // tree): pass1 sum->mean, pass2 sum of (x-mean)^2 -> var. Measures reorder drift without
        // the one-pass sumsq-mean^2 cancellation.
        auto sopt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
        auto sum = torch::zeros({N * num_groups}, sopt);
        auto var = torch::zeros({N * num_groups}, sopt);
        long num_elements = (long)N * C * HW, sample_stride = (long)C * HW;
        long group_size = (long)(C / num_groups) * HW;
        int ab = 256; unsigned int ag = (unsigned int)((num_elements + ab - 1) / ab);
        int NG = N * num_groups, fb = 128, fg = (NG + fb - 1) / fb;
        cudaStream_t st = at::cuda::getCurrentCUDAStream();
        if (x.scalar_type() == torch::kFloat32) {
            gn_stats_sum_kernel<float><<<ag, ab, 0, st>>>(x.data_ptr<float>(), sum.data_ptr<float>(), C, num_groups, sample_stride, num_elements);
            gn_mean_kernel<<<fg, fb, 0, st>>>(sum.data_ptr<float>(), mean.data_ptr<float>(), group_size, NG);
            gn_stats_var_kernel<float><<<ag, ab, 0, st>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), C, num_groups, sample_stride, num_elements);
        } else {
            const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
            gn_stats_sum_kernel<__half><<<ag, ab, 0, st>>>(xp, sum.data_ptr<float>(), C, num_groups, sample_stride, num_elements);
            gn_mean_kernel<<<fg, fb, 0, st>>>(sum.data_ptr<float>(), mean.data_ptr<float>(), group_size, NG);
            gn_stats_var_kernel<__half><<<ag, ab, 0, st>>>(xp, mean.data_ptr<float>(), var.data_ptr<float>(), C, num_groups, sample_stride, num_elements);
        }
        const char* _pf = std::getenv("MODIFF_GN_STATS_PERTURB");   // sanity: >1.0 deliberately perturbs inv_std
        float perturb = (_pf != nullptr) ? (float)atof(_pf) : 1.0f;
        gn_invstd_kernel<<<fg, fb, 0, st>>>(var.data_ptr<float>(), inv_std.data_ptr<float>(), group_size, (float)eps, NG, perturb);
        return;
    }
    const long group_size = (long)(C / num_groups) * HW;
    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups));
    dim3 block((unsigned int)block_size);
    size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Cycle 3 attempted a CPG-even-gated dispatch to gn_group_stats_vec2_kernel here;
    // reverted after it failed gn_modiff_verify_realinput.py's zero-tolerance gate
    // (max_code_diff=1) -- see gn_group_stats_vec2_kernel's comment above.
    if (x.scalar_type() == torch::kFloat32) {
        gn_group_stats_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            C, HW, num_groups, (float)eps);
    } else {
        gn_group_stats_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            C, HW, num_groups, (float)eps);
    }
}

// Kernel 2 (int8): flat, fully-coalesced GN-affine(+mod)+fp16-round+SiLU(+smooth) +
// delta-quantize + in-place a_hat update. One thread per element, grid-stride over
// the physical NHWC tensor -- so x, a_hat and Yq loads/stores are contiguous per
// warp. c = i%C, n = i/sample_stride, group = c/CPG (index into the [N*G] stats).
template <typename TIn>
__global__ void gn_apply_delta_quantize_flat_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,     // [N,H,W,C] fp16 channels_last, in place
    int8_t* __restrict__ Yq,              // [N,H,W,C] int8 quantized delta
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,    // [N,C] or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ mean_in,    // [N*G]
    const float* __restrict__ inv_std_in, // [N*G]
    const float* __restrict__ scale_ptr,  // scalar quant multiplier = 127/absmax
    const float* __restrict__ smooth_inv, // [C] or nullptr
    int C, int G, long sample_stride, long num_elements, bool apply_silu
) {
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
         i += (long)blockDim.x * gridDim.x) {
        int c = (int)(i % C);
        long n = i / sample_stride;
        long stats_idx = n * G + (c / CPG);
        float mean = mean_in[stats_idx];
        float inv_std = inv_std_in[stats_idx];
        // Same three-temp form as group_norm_silu_nhwc_kernel's pass 2 (same TU) so the
        // fp32 `normed` -- and thus its fp16 round feeding SiLU -- matches bit-for-bit.
        float v = gn_load(X, i);
        float w = gn_load(gamma, c);
        float b = gn_load(beta, c);
        float normed = (v - mean) * inv_std * w + b;
        if (mod_scale != nullptr) {
            long midx = n * C + c;
            normed = normed * (1.0f + gn_load(mod_scale, midx)) + gn_load(mod_shift, midx);
        }
        float normed_h = __half2float(__float2half(normed));
        float out = apply_silu ? gns_silu(normed_h) : normed_h;
        if (smooth_inv != nullptr) out *= smooth_inv[c];
        float cache = __half2float(a_hat_cache[i]);
        float q = fmaxf(-127.0f, fminf(127.0f, roundf((out - cache) * scale)));
        a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
        Yq[i] = (int8_t)q;
    }
}

// Vectorized (half2/float2) counterpart of gn_apply_delta_quantize_flat_kernel. Pair-major
// grid-stride loop, mirroring the structure gn_apply_delta_quantize_pack_flat_vec2_kernel below
// already uses. gamma/beta/mod_scale/mod_shift/a_hat_cache all read/written via one gn_load2/
// gn_store2 call per pair instead of two scalar calls; the two output int8 codes are packed
// into one int16 store. Requires CPG even (so a pair's c0/c0+1 always share one group and
// hence one mean/inv_std, exactly like the pack kernel below) -- the caller (host wrapper)
// only dispatches here when that holds, else falls back to the scalar kernel above.
template <typename TIn>
__global__ void gn_apply_delta_quantize_flat_vec2_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,     // [N,H,W,C] fp16 channels_last, in place
    int8_t* __restrict__ Yq,              // [N,H,W,C] int8 quantized delta
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,    // [N,C] or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ mean_in,    // [N*G]
    const float* __restrict__ inv_std_in, // [N*G]
    const float* __restrict__ scale_ptr,  // scalar quant multiplier = 127/absmax
    const float* __restrict__ smooth_inv, // [C] or nullptr
    int C, int G, long sample_stride, long num_elements, bool apply_silu
) {
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const long stride = (long)blockDim.x * gridDim.x;
    for (long base = 2 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < num_elements; base += 2 * stride) {
        int c0 = (int)(base % C);          // even; c0 and c0+1 are in the same group
        long n = base / sample_stride;
        long stats_idx = n * G + (c0 / CPG);
        float mean = mean_in[stats_idx];
        float inv_std = inv_std_in[stats_idx];

        float2 v = gn_load2(X, base);
        float2 w = gn_load2(gamma, c0);
        float2 b = gn_load2(beta, c0);
        float n0 = (v.x - mean) * inv_std * w.x + b.x;
        float n1 = (v.y - mean) * inv_std * w.y + b.y;
        if (mod_scale != nullptr) {
            long midx = n * C + c0;
            float2 ms = gn_load2(mod_scale, midx);
            float2 sh = gn_load2(mod_shift, midx);
            n0 = n0 * (1.0f + ms.x) + sh.x;
            n1 = n1 * (1.0f + ms.y) + sh.y;
        }
        float n0h = __half2float(__float2half(n0));
        float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        if (smooth_inv != nullptr) { o0 *= smooth_inv[c0]; o1 *= smooth_inv[c0 + 1]; }
        float2 cache = gn_load2(a_hat_cache, base);
        float q0 = fmaxf(-127.0f, fminf(127.0f, roundf((o0 - cache.x) * scale)));
        float q1 = fmaxf(-127.0f, fminf(127.0f, roundf((o1 - cache.y) * scale)));
        gn_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        reinterpret_cast<int16_t*>(Yq)[base >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
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
    const int CPG = C / (int)num_groups;
    // Vectorized (vec2) pass requires a channel pair to never straddle a group boundary
    // (both channels then share one mean/inv_std) -- no existing TORCH_CHECK enforces this
    // (unlike the int4-pack sibling below), so this is a genuine runtime fallback. Real
    // model configs always have even CPG; the odd-CPG branch is exercised by
    // gn_static_quantize_verify.py's synthetic shape (via the sibling static kernel; this
    // delta-quantize path is covered structurally by the same CPG invariant).
    const bool use_vec2 = (CPG % 2 == 0);

    auto yq = torch::empty_like(x, x.options().dtype(torch::kInt8));
    auto stats_opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto mean = torch::empty({N * (int)num_groups}, stats_opts);
    auto inv_std = torch::empty({N * (int)num_groups}, stats_opts);
    gn_launch_group_stats(x, N, C, HW, (int)num_groups, eps, mean, inv_std);

    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    __half* cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    const long num_elements = (long)N * C * HW;
    const long sample_stride = (long)C * HW;
    const int ablock = 256;
    const unsigned int agrid_scalar = (unsigned int)((num_elements + ablock - 1) / ablock);
    const unsigned int agrid_vec2 = (unsigned int)((num_elements / 2 + ablock - 1) / ablock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat32) {
        if (use_vec2) {
            gn_apply_delta_quantize_flat_vec2_kernel<float><<<agrid_vec2, ablock, 0, stream>>>(
                x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                scale.data_ptr<float>(), smooth_ptr,
                C, (int)num_groups, sample_stride, num_elements, apply_silu);
        } else {
            gn_apply_delta_quantize_flat_kernel<float><<<agrid_scalar, ablock, 0, stream>>>(
                x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                scale.data_ptr<float>(), smooth_ptr,
                C, (int)num_groups, sample_stride, num_elements, apply_silu);
        }
    } else {
        if (use_vec2) {
            gn_apply_delta_quantize_flat_vec2_kernel<__half><<<agrid_vec2, ablock, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                scale.data_ptr<float>(), smooth_ptr,
                C, (int)num_groups, sample_stride, num_elements, apply_silu);
        } else {
            gn_apply_delta_quantize_flat_kernel<__half><<<agrid_scalar, ablock, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                scale.data_ptr<float>(), smooth_ptr,
                C, (int)num_groups, sample_stride, num_elements, apply_silu);
        }
    }
    return yq;
}

// Kernel 2 (int4), half2/float2-vectorized: flat, coalesced MoDiff delta-quantize that
// packs adjacent channel pairs (even -> low nibble, odd -> high) into one byte, matching
// group_norm_silu_quantize_pack_nhwc's layout and
// step1_static_quantize_pack_int4_fprop_silu's semantics. One thread per pair; base is even
// and channels-per-group is even, so a pair never straddles a group boundary (both channels
// share group c0/CPG). The loop is naturally pair-major (one thread per output byte), so
// vectorizing it needed no restructuring -- just gn_load2/gn_store2 in place of per-element
// accesses.
template <typename TIn>
__global__ void gn_apply_delta_quantize_pack_flat_vec2_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,     // [N,H,W,C] fp16 channels_last, in place
    int8_t* __restrict__ Yqp,             // [N,H,W,C/2] packed int4
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,    // [N,C] or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ mean_in,    // [N*G]
    const float* __restrict__ inv_std_in, // [N*G]
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv, // [C] or nullptr
    int C, int G, long sample_stride, long num_elements, bool apply_silu
) {
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const long stride = (long)blockDim.x * gridDim.x;
    for (long base = 2 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < num_elements; base += 2 * stride) {
        int c0 = (int)(base % C);
        long n = base / sample_stride;
        long stats_idx = n * G + (c0 / CPG);
        float mean = mean_in[stats_idx];
        float inv_std = inv_std_in[stats_idx];

        float2 v = gn_load2(X, base);
        float2 w = gn_load2(gamma, c0);
        float2 b = gn_load2(beta, c0);
        float n0 = (v.x - mean) * inv_std * w.x + b.x;
        float n1 = (v.y - mean) * inv_std * w.y + b.y;
        if (mod_scale != nullptr) {
            long midx = n * C + c0;
            float2 ms = gn_load2(mod_scale, midx);
            float2 sh = gn_load2(mod_shift, midx);
            n0 = n0 * (1.0f + ms.x) + sh.x;
            n1 = n1 * (1.0f + ms.y) + sh.y;
        }
        float o0 = apply_silu ? gns_silu(__half2float(__float2half(n0))) : __half2float(__float2half(n0));
        float o1 = apply_silu ? gns_silu(__half2float(__float2half(n1))) : __half2float(__float2half(n1));
        if (smooth_inv != nullptr) { o0 *= smooth_inv[c0]; o1 *= smooth_inv[c0 + 1]; }
        float2 cache = gn_load2(a_hat_cache, base);
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf((o0 - cache.x) * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf((o1 - cache.y) * scale)));
        gn_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        Yqp[base / 2] = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
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
    auto yqp = torch::empty({N, H, W, C / 2},
                            torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    auto stats_opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto mean = torch::empty({N * (int)num_groups}, stats_opts);
    auto inv_std = torch::empty({N * (int)num_groups}, stats_opts);
    gn_launch_group_stats(x, N, C, HW, (int)num_groups, eps, mean, inv_std);

    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    __half* cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());
    const long num_elements = (long)N * C * HW;
    const long sample_stride = (long)C * HW;
    const int ablock = 256;
    const long num_pairs = num_elements / 2;
    const unsigned int agrid = (unsigned int)((num_pairs + ablock - 1) / ablock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // C%2==0 && CPG%2==0 already TORCH_CHECK'd above -> always safe to use the
    // vectorized kernel here, no scalar fallback needed.
    if (x.scalar_type() == torch::kFloat32) {
        gn_apply_delta_quantize_pack_flat_vec2_kernel<float><<<agrid, ablock, 0, stream>>>(
            x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            scale.data_ptr<float>(), smooth_ptr,
            C, (int)num_groups, sample_stride, num_elements, apply_silu);
    } else {
        gn_apply_delta_quantize_pack_flat_vec2_kernel<__half><<<agrid, ablock, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            scale.data_ptr<float>(), smooth_ptr,
            C, (int)num_groups, sample_stride, num_elements, apply_silu);
    }
    return yqp;
}
