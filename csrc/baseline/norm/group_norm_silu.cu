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

#include "../common/common.cuh"

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

// Forward declaration only. The definition, and the measurements that motivate it, live further
// down next to the flat delta-quantize kernels that introduced it. The resize delta kernel is
// defined above those and reports its delta range through the same helper, so it needs the
// declaration here.
__device__ __forceinline__ void gn_report_delta_absmax(
    float local_max, float* __restrict__ sdata, float* __restrict__ absmax_buf,
    float* __restrict__ next_scale_out, float* __restrict__ next_inv_out,
    unsigned int* __restrict__ retire_count, float Q_level, float safety);

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
    int Kpad,                      // padded row width in CHANNELS (>= C, even); == C for no padding
    // ACTIVATION ZERO POINT (plan fix #2). a_q = clamp(round(a*s) + z, -7, 7), so the 15 available
    // codes can straddle an asymmetric range instead of being centred on 0. silu(gn(x)) is one-sided
    // -- measured |max|/|min| = 19.91x, with only 5 of 15 codes carrying >0.1% of the mass, an
    // effective 2.32 bits of a nominal 3.91 -- which is what this exists to recover.
    //
    // The dequantization's -z*sum(w_q) term is folded into the conv bias at CALIBRATION time
    // (OptimizedInt4Conv2d._refold_zp_bias), so neither the GEMM nor the epilogue sees z, and this
    // kernel needs nothing but the addition below.
    //
    // z == 0 REPRODUCES THE OLD KERNEL EXACTLY -- the added term is 0.0f, not a differently-rounded
    // path -- which is what lets an asymmetric-capable build keep every committed symmetric number.
    float zp = 0.0f
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
        int8_t i0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o0 * scale) + zp));
        int8_t i1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(o1 * scale) + zp));
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

// Fill ONLY the spatial halo of a padded packed-int4 output with a constant byte. Launched alongside
// the GN kernel above when a halo is requested: the GN kernel writes the interior at an offset, this
// writes the border, and neither traverses the other's bytes. The border is O((H+W)*pad) positions
// against the interior's O(H*W), so this is a rounding error next to the GN pass itself.
// Fused GroupNorm+SiLU+quantize+2x resize, one launch, for the updown ResBlock path.
// Pass 1 (the stats reduction) is copied verbatim from the non-resizing sibling; only pass 2
// differs. UP=true does nearest 2x upsample, which is an exact index-select, so a value is
// computed once and stored to the four output positions. UP=false does 2x2 average pooling
// and is the reason this has to be ONE kernel: averaging must happen on the fp32 post-SiLU
// values BEFORE quantization. Splitting it (quantize, then average the int4 codes) rounds
// each of the four inputs first and averages the rounded values, which is a different and
// worse result -- that non-commutation is exactly why the shipped pipeline keeps GroupNorm at
// fp16 and defers the quantize to the resize kernel instead.
// PACK=true writes one byte per channel PAIR (two int4 nibbles, range +-7); PACK=false
// writes two int8 bytes as one int16 store (range +-127), matching the non-pack sibling.
template <typename TIn, bool FAST_REDUCE, bool UP, bool PACK>
__global__ void group_norm_silu_quantize_resize_nhwc_kernel(
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
    int W,                         // INPUT width, needed to map hw -> (h, w) for the resize
    // ACTIVATION ZERO POINT (plan fix #2), identical contract to
    // group_norm_silu_quantize_pack_nhwc_vec2_kernel: a_q = clamp(round(a*s) + z, -7, 7), with the
    // -z*sum(w_q) term folded into the conv bias at calibration time.
    //
    // PACK=false (int8) NEVER receives a non-zero z -- the host wrapper TORCH_CHECKs it. The int4
    // zero point comes from an int4 calibration file and the int8 path's bias carries no matching
    // correction, so letting it through would create the exact mismatch this feature guards against.
    //
    // In the DOWN direction z is added AFTER the 2x2 average, which is the only correct place: the
    // average is taken on fp32 post-SiLU values and z shifts the CODE, not the activation. Adding it
    // to each of the four contributions would scale it by 4.
    //
    // z == 0 reproduces the old kernel exactly in both directions.
    float zp = 0.0f
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int KpadH = Kpad / 2;    // bytes per spatial position in the output

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    const TIn* x_base = X + (long)n * HW * C;
    const long HW_OUT = UP ? (HW * 4) : (HW / 4);
    const long row_bytes = PACK ? (long)KpadH : (long)C;
    int8_t* yqp_base = Yqp + (long)n * (HW_OUT * row_bytes);

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
            const float lim = PACK ? 7.0f : 127.0f;
            const int8_t i0 = (int8_t)fmaxf(-lim, fminf(lim, roundf(o0 * scale) + zp));
            const int8_t i1 = (int8_t)fmaxf(-lim, fminf(lim, roundf(o1 * scale) + zp));
            const int8_t packed = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
            const int16_t pair16 = (int16_t)(((uint8_t)i0) | (((uint16_t)(uint8_t)i1) << 8));
            const int h = (int)(hw / Wi), w = (int)(hw % Wi);
#pragma unroll
            for (int dy = 0; dy < 2; ++dy)
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                    const long hw_out = (long)(2 * h + dy) * Wo + (2 * w + dx);
                    if constexpr (PACK)
                        yqp_base[hw_out * row_bytes + (c_global0 >> 1)] = packed;
                    else
                        reinterpret_cast<int16_t*>(yqp_base)[
                            (hw_out * row_bytes + c_global0) >> 1] = pair16;
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
            const float lim = PACK ? 7.0f : 127.0f;
            const int8_t i0 = (int8_t)fmaxf(-lim, fminf(lim, roundf(a0 * scale) + zp));
            const int8_t i1 = (int8_t)fmaxf(-lim, fminf(lim, roundf(a1 * scale) + zp));
            if constexpr (PACK)
                yqp_base[hwo * row_bytes + (c_global0 >> 1)] =
                    (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
            else
                reinterpret_cast<int16_t*>(yqp_base)[(hwo * row_bytes + c_global0) >> 1] =
                    (int16_t)(((uint8_t)i0) | (((uint16_t)(uint8_t)i1) << 8));
        }
    }

    if constexpr (PACK) {
        if (g == 0 && Kpad > C) {
            const int tail_bytes = KpadH - C / 2;
            for (long idx = threadIdx.x; idx < HW_OUT * (long)tail_bytes; idx += blockDim.x) {
                const long hw = idx / tail_bytes;
                const int pb = idx % tail_bytes;
                yqp_base[hw * (long)KpadH + C / 2 + pb] = 0;
            }
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
    bool fast_reduce,
    double zero_point          // activation zero point; 0.0 reproduces the symmetric kernel exactly
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
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad, (float)zero_point);
        else
          group_norm_silu_quantize_pack_nhwc_vec2_kernel<float, false><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad, (float)zero_point
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
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad, (float)zero_point);
        else
          group_norm_silu_quantize_pack_nhwc_vec2_kernel<__half, false><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, Kpad, (float)zero_point
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return yqp;
}

// Host side of the fused GN+SiLU+quantize+resize kernel. `resize` is +1 for a nearest 2x
// upsample and -1 for a 2x2 average pool; the output is [N, H*2, W*2, Kpad/2] or
// [N, H/2, W/2, Kpad/2]. Deliberately a separate entry point from the non-resizing sibling: it
// is a prototype for the updown ResBlock path and is not wired into the pipeline.
static torch::Tensor group_norm_silu_quantize_resize_nhwc_impl(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack, double zero_point
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    // The int4 zero point is meaningless on the int8 output: that path's bias carries no matching
    // -z*sum(w_q) correction, so honouring z there would BE the mismatch this feature exists to
    // avoid. Refuse instead of silently applying or silently dropping it.
    TORCH_CHECK(pack || zero_point == 0.0,
                "gn_quantize_resize: a non-zero activation zero point is only defined for the "
                "packed int4 output (pack=true); the int8 path has no bias correction for it");
    TORCH_CHECK(x.dim() == 4, "gn_quantize_resize expects [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kFloat32,
                "gn_quantize_resize: only fp16/fp32 supported");
    TORCH_CHECK(resize == 1 || resize == -1, "gn_quantize_resize: resize must be +1 or -1");
    const bool has_mod = mod_scale.numel() > 0;
    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "channels must divide into groups");
    const int CPG = C / (int)num_groups;
    TORCH_CHECK(C % 2 == 0 && CPG % 2 == 0, "channels and channels-per-group must be even");
    const bool up = (resize == 1);
    if (!up) TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "downsample needs even H and W");
    const long HW = (long)H * W;
    const long group_size = (long)CPG * HW;
    const int Kpad = (pack && k_pad > (int64_t)C) ? (int)k_pad : C;
    TORCH_CHECK(Kpad % 2 == 0, "k_pad must be even");
    const int Ho = up ? H * 2 : H / 2, Wo = up ? W * 2 : W / 2;
    auto opts = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    // The two outputs have the same NHWC-physical bytes but deliberately different LOGICAL
    // shapes, because their consumers disagree about where the spatial extents come from.
    // int4 goes to _conv_from_int4(x_q, Ho, Wo), which is told them; the tensor is a literal
    // [N, Ho, Wo, Kpad/2] byte buffer (Kpad/2 != C, so an NCHW shape would be a lie anyway).
    // int8 goes to _conv_from_int8(x_q), which reads H and W off x_q.shape[2] and [3] -- so it
    // must be [N, C, Ho, Wo] channels_last, matching avgpool2x_quantize_noahat_fprop. Returning
    // a literal [N, Ho, Wo, C] here made the conv read Wo as its height and C as its width and
    // walk ~128 KiB off the end of the activation.
    auto yq = pack
        ? torch::empty({N, Ho, Wo, Kpad / 2}, opts)
        : torch::empty({N, C, Ho, Wo},
                       opts.memory_format(torch::MemoryFormat::ChannelsLast));

    int block_size = 128;
    while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;
    dim3 grid((unsigned int)(N * num_groups)), block((unsigned int)block_size);
    size_t shmem = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;

    // gn_load2 is overloaded for `const float*` and `const __half*` only, so the fp16 launch has
    // to reinterpret at::Half -- instantiating the kernel on at::Half itself does not compile.
#define MODIFF_GNQR_LAUNCH(T, ATT, UPV, PK)                                                 \
    group_norm_silu_quantize_resize_nhwc_kernel<T, true, UPV, PK>                           \
        <<<grid, block, shmem, stream>>>(                                                   \
            reinterpret_cast<const T*>(x.data_ptr<ATT>()),                                  \
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),                               \
            reinterpret_cast<const T*>(weight.data_ptr<ATT>()),                             \
            reinterpret_cast<const T*>(bias.data_ptr<ATT>()),                               \
            has_mod ? reinterpret_cast<const T*>(mod_scale.data_ptr<ATT>()) : nullptr,      \
            has_mod ? reinterpret_cast<const T*>(mod_shift.data_ptr<ATT>()) : nullptr,      \
            scale.data_ptr<float>(), smooth_ptr, C, HW, (int)num_groups, (float)eps,        \
            apply_silu, Kpad, W, (float)zero_point)
#define MODIFF_GNQR_DISPATCH(T, ATT)                                                        \
    do {                                                                                    \
        if (up &&  pack) MODIFF_GNQR_LAUNCH(T, ATT, true,  true);                           \
        if (up && !pack) MODIFF_GNQR_LAUNCH(T, ATT, true,  false);                          \
        if (!up &&  pack) MODIFF_GNQR_LAUNCH(T, ATT, false, true);                          \
        if (!up && !pack) MODIFF_GNQR_LAUNCH(T, ATT, false, false);                         \
    } while (0)

    if (x.scalar_type() == torch::kFloat32) MODIFF_GNQR_DISPATCH(float, float);
    else                                    MODIFF_GNQR_DISPATCH(__half, at::Half);
#undef MODIFF_GNQR_DISPATCH
#undef MODIFF_GNQR_LAUNCH
    C10_CUDA_CHECK(cudaGetLastError());
    return yq;
}

// Two arities, same reason as group_norm_silu_quantize_pack_nhwc{,_zp}: pybind11 does not inherit
// C++ defaults, and the 13-argument callers (integration/fused_ops/fused_resblock.py and the
// archived prototypes in docs/*/scripts) must keep working byte-for-byte.
torch::Tensor group_norm_silu_quantize_resize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack
) {
    return group_norm_silu_quantize_resize_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, resize, pack, 0.0);
}

torch::Tensor group_norm_silu_quantize_resize_nhwc_zp(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack, double zero_point
) {
    return group_norm_silu_quantize_resize_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, resize, pack, zero_point);
}


// TWO ARITIES EACH, not a default argument: pybind11 does not inherit C++ defaults, and this file
// annotates no argument names, so the existing 11-argument callers in integration/ and ~8 archived
// docs/*/scripts keep working untouched while the 12-argument form names the activation zero point.
// Exactly the pattern step1_static_quantize_fprop already uses (see pybind.cpp).
torch::Tensor group_norm_silu_quantize_pack_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad) {
    return group_norm_silu_quantize_pack_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, false, 0.0);
}

torch::Tensor group_norm_silu_quantize_pack_nhwc_zp(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad, double zero_point) {
    return group_norm_silu_quantize_pack_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, false, zero_point);
}

torch::Tensor group_norm_silu_quantize_pack_nhwc_fast(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad) {
    return group_norm_silu_quantize_pack_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, true, 0.0);
}

torch::Tensor group_norm_silu_quantize_pack_nhwc_fast_zp(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad, double zero_point) {
    return group_norm_silu_quantize_pack_nhwc_impl(
        x, weight, bias, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, k_pad, true, zero_point);
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

// =========================================================================
// Free absmax reporting for the delta-quantize kernels.
//
// A delta-quantize kernel already evaluates every |delta| on its way to a code. So it can also
// reduce their max and, in its retirement election, write the scale the NEXT step should use --
// at the cost of one shared-memory reduction and one atomic per block, with no extra pass over
// memory. The separate absmax pass then disappears entirely.
//
// Measured motivation (batch 128, 2026-08-04): after the GN-stats and resize fusions landed,
// MoDiff's remaining kernel-time overhead against its own baseline was +4.58 ms/step, of which
// the standalone absmax reduction was +1.57 -- the largest single ADDRESSABLE item (the conv
// o_hat RMW at +1.31 and the a_hat traffic at +0.93 are required by Eqs 9-10).
//
// The scale is therefore one step stale. That is the mildest possible staleness, and the measured
// tolerance is far wider: refreshing only every 8th step cost nothing (0.97x-1.06x relative to
// exact per-step), and only every 25th step broke down. `safety` gives headroom for the range
// growing between steps; the delta range evolves smoothly along a DDIM trajectory.
//
// Call from ONE thread per block after a __syncthreads(), with `sdata` sized blockDim.x floats.
// Null absmax_buf => no-op, so the same kernel serves the reporting and non-reporting paths.
__device__ __forceinline__ void gn_report_delta_absmax(
    float local_max,                     // this thread's max |delta|
    float* __restrict__ sdata,           // [blockDim.x] scratch
    float* __restrict__ absmax_buf,      // [1], 0 on entry (self-resetting); nullptr => skip
    float* __restrict__ next_scale_out,  // [1] out: Q_level/(safety*absmax) for the NEXT step
    float* __restrict__ next_inv_out,    // [1] out: its reciprocal (CUTLASS alpha)
    unsigned int* __restrict__ retire_count,  // [1], 0 on entry (self-resetting)
    float Q_level, float safety
) {
    if (absmax_buf == nullptr) return;
    const int tid = threadIdx.x;
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid != 0) return;
    float val = sdata[0];
    unsigned int* addr = (unsigned int*)absmax_buf;
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);
    __threadfence();
    unsigned int ticket = atomicAdd(retire_count, 1u);
    if (ticket == gridDim.x - 1) {
        float am = fmaxf(*absmax_buf * safety, 1e-6f);
        *next_scale_out = Q_level / am;
        *next_inv_out = am / Q_level;
        *absmax_buf = 0.0f;
        *retire_count = 0;
    }
}














