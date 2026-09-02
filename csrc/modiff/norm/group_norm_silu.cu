// =========================================================================
// MoDiff temporal-delta GroupNorm kernels. Baseline twin: csrc/baseline/norm/group_norm_silu.cu.
//
// The baseline file computes GN(+SiLU)(+quantize) directly. This file's kernels instead quantize
// the DELTA against the previous timestep's a_hat cache, which is why they need mean/inv_std
// precomputed: the delta path is FLAT (grid-strided over elements, because the delta absmax is a
// whole-tensor reduction) and so cannot use the baseline's block-per-(n,group) shape that computes
// stats and applies the norm in one launch. `gn_launch_group_stats` below is that separate stats
// pass, and `gn_stats_from_tiles` is the Stage-A prototype for folding it into the producing conv's
// epilogue (see docs/gn_stats_in_epilogue_2026-08-11 and docs/aq_fusion_2026-08-12).
//
// Family 3 of the csrc/ datapath split (2026-08-12). The partition was CLEAN: 15 kernels and
// `gn_launch_group_stats` are reached only from the delta entry points, 5 kernels and the two
// `*_impl` helpers only from the baseline ones, and ZERO kernels are shared. What is duplicated
// below is only the small dtype-dispatch device helpers (gn_load/gn_load2/gn_store2, gns_silu) and
// gn_report_delta_absmax, which both trees' kernels call.
//
// KEEP THE COPIES IDENTICAL to their twins -- every A/B in docs/ compares the two datapaths.
// =========================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "gn_block_size.h"
#include <cuda_fp16.h>
#include <cstdlib>
#include <type_traits>

// Explicit relative path, NOT bare: a bare include resolves through the global -I csrc and would
// pick a different tree's copy. See csrc/README.md.
#include "../common/common.cuh"

// ==== COPIED device helpers (also used by the baseline twin) ====

// ---- COPY of gn_load ----
__device__ __forceinline__ float gn_load(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float gn_load(const __half* p, long i) { return __half2float(p[i]); }

// ---- COPY of gn_load2 ----
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

// ---- COPY of gn_store2 ----
__device__ __forceinline__ void gn_store2(float* p, long i, float2 v) {
    reinterpret_cast<float2*>(p)[i >> 1] = v;
}
__device__ __forceinline__ void gn_store2(__half* p, long i, float2 v) {
    reinterpret_cast<__half2*>(p)[i >> 1] = __float22half2_rn(v);
}

#include "../common/ahat_cache.cuh"

// ---- COPY of gns_silu ----
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

// ---- COPY of gn_report_delta_absmax_decl ----
// Forward declaration only. The definition, and the measurements that motivate it, live further
// down next to the flat delta-quantize kernels that introduced it. The resize delta kernel is
// defined above those and reports its delta range through the same helper, so it needs the
// declaration here.
__device__ __forceinline__ void gn_report_delta_absmax(
    float local_max, float* __restrict__ sdata, float* __restrict__ absmax_buf,
    float* __restrict__ next_scale_out, float* __restrict__ next_inv_out,
    unsigned int* __restrict__ retire_count, float Q_level, float safety);

// ---- COPY of gn_report_delta_absmax ----
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

// ==== MoDiff delta-path kernels and entry points (moved from the baseline file) ====

// =========================================================================
// MoDiff delta twin of group_norm_silu_quantize_resize_nhwc_kernel.
//
// Why it exists: the eight updown ResBlocks get ZERO fusion under MoDiff. The baseline fuses
// GN+SiLU+resize+quantize into this one kernel (self-documented at 1.45-5.6x, median ~2.9x on the
// eight real updown shapes), but _prequant_gn_resize_conv gates on `not modiff`, so MoDiff falls
// back to a standalone PyTorch resize followed by a separate delta-quantize. Measured cost of that
// fallback at batch 128 (2026-08-04): +1.20 ms/step nearest upsample, +0.44 avg_pool, +0.71
// GN+SiLU-only -- 2.35 ms/step, the largest remaining NON-INTRINSIC MoDiff overhead.
//
// Everything above the quantize is copied verbatim from the baseline kernel, including the stats
// reduction, `compute_pair`, and the pre-quantize 2x2 average in the DOWN branch (which is the
// reason this has to be one kernel at all: averaging must happen on the fp32 post-SiLU values
// BEFORE quantization).
//
// What changes: a_hat is subtracted before quantizing and updated in place afterwards. a_hat is
// cached at the POST-resize (conv input) resolution, exactly as the unfused path already does, so
// this is a pure fusion with no change to the state layout or to MoDiff's semantics.
//
// The one subtlety, in the UP branch: nearest 2x upsample sends one input value to four output
// positions, and those four positions have FOUR DIFFERENT a_hat entries. So unlike the baseline --
// which computes one code and stores it four times -- the delta must be formed and quantized once
// per output position. The loop still grids over INPUT positions (so the GN affine and SiLU are
// evaluated once, as in the baseline); only the subtract/quantize/update is done four times.
//
// DYNAMIC SCALE (added 2026-08-10). Originally this kernel took its scale as a device pointer and
// nothing else, so it could not serve a step that has to MEASURE the delta range. The caller
// therefore declined on every refresh step -- which at MODIFF_DELTA_REFRESH=1, i.e. the paper's
// own configuration, is every step, so the fusion never fired at all and the eight updown
// ResBlocks fell back to a standalone resize + separate delta-quantize (measured 0/8 fused at K=1
// against 6/8 at K=4, docs/component_attribution_2026-08-07). Two additions fix that, both
// mirroring what group_norm_silu_delta_quantize_nhwc already does for the other 62 convs:
//
//   * `reduce_only` turns this same kernel into its own reduction-only twin -- identical GN, mod,
//     SiLU, SmoothQuant, resize and a_hat arithmetic, but it stores nothing and instead reduces
//     max|delta| and publishes Q_level/absmax through gn_report_delta_absmax. Sharing one body
//     rather than writing a second kernel is what guarantees the measured range is the range the
//     quantize pass then sees; a hand-copied twin would drift the first time either changed.
//   * on a quantizing pass, the same reduce-and-publish runs for free at the end (report_next),
//     so the separate pass can be skipped entirely where the caller wants that trade.
//
// The delta is reduced BEFORE the clamp, so the report is the true range rather than a clipped
// lower bound -- the same convention as gn_apply_delta_quantize_flat_vec2_kernel.
template <typename TIn, bool FAST_REDUCE, bool UP, bool PACK>
__global__ void group_norm_silu_delta_quantize_resize_nhwc_kernel(
    const TIn* __restrict__ X,
    int8_t* __restrict__ Yqp,         // [N, H_out, W_out, C or C/2] codes, channels_last-flat
    __half* __restrict__ a_hat_cache, // [N, H_out, W_out, C] fp16, POST-resize, updated in place
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
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument, which THIS kernel did until 2026-08-10) and a 4-bit layer
    // silently stayed 8-bit. A bool has no such value to get wrong; the saturation limit
    // is derived from the datapath below.
    bool a4,
    // --- delta-range reporting (absmax_buf == nullptr => none, the original behaviour) ---
    float* __restrict__ absmax_buf,      // [1], 0 on entry (self-resetting)
    float* __restrict__ next_scale_out,  // [1] out: Q_level/(safety*absmax)
    float* __restrict__ next_inv_out,    // [1] out: its reciprocal (the conv's CUTLASS alpha)
    unsigned int* __restrict__ retire_count,  // [1], 0 on entry (self-resetting)
    float Q_level,
    float safety,
    // reduce_only: measure the range and publish, store NOTHING -- neither codes nor a_hat.
    // scale_ptr is not dereferenced, which is what lets this pass run BEFORE the step has a scale.
    bool reduce_only,
    // skip-K: still form codes against a_hat, but do not commit the cache (naive freeze).
    bool write_ahat,
    bool ahat_i8 = false,
    const float* ahat_qscale = nullptr, int ahat_ng = 0
) {
    float ahat_s, ahat_inv, ahat_lim;
    ahat_qparams(ahat_i8, ahat_qscale, ahat_s, ahat_inv, ahat_lim, ahat_ng);
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
    // Not dereferenced on the reduction pass: it runs precisely when no scale for this step
    // exists yet, and scale_ptr may be pointing at an uninitialised buffer.
    const float scale = reduce_only ? 1.0f : *scale_ptr;
    const float inv_scale = 1.0f / scale;   // a_hat += q * inv_scale, MoDiff Eq 10
    float local_delta_max = 0.0f;           // max |delta| seen by this thread, pre-clamp

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
            const float lim = (PACK || a4) ? 7.0f : 127.0f;
            const int h = (int)(hw / Wi), w = (int)(hw % Wi);
            // The four output positions share this input value but NOT its a_hat entry, so the
            // delta is formed and quantized four times. compute_pair ran once.
#pragma unroll
            for (int dy = 0; dy < 2; ++dy)
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                    const long hw_out = (long)(2 * h + dy) * Wo + (2 * w + dx);
                    const long ci = (long)n * HW_OUT * C + hw_out * (long)C + c_global0;
                    // vec2: ci is even (c_start = g*CPG with CPG even, plus 2*cpair), so this
                    // is one naturally-aligned 4-byte load instead of two 2-byte ones. Same values.
                    float q0, q1, d0, d1;
                    if (reduce_only) {
                        // Match ahat_quant_update2's pre-clamp residual without writing codes.
                        if (ahat_is_imode(ahat_i8, ahat_s)) {
                            const int a0 = ahat_load_int(a_hat_cache, ci, ahat_lim);
                            const int a1 = ahat_load_int(a_hat_cache, ci + 1, ahat_lim);
                            d0 = o0 - (float)a0 * inv_scale;
                            d1 = o1 - (float)a1 * inv_scale;
                        } else {
                            float bs, binv, blim;
                            ahat_resolve(ahat_i8, ahat_qscale, ci, C, ahat_ng,
                                         ahat_s, ahat_inv, ahat_lim, bs, binv, blim);
                            const float2 cpv = ahat_load2(a_hat_cache, ci, ahat_i8, bs);
                            d0 = o0 - cpv.x; d1 = o1 - cpv.y;
                        }
                        local_delta_max = fmaxf(local_delta_max, fmaxf(fabsf(d0), fabsf(d1)));
                        continue;
                    }
                    ahat_quant_update2(a_hat_cache, ci, o0, o1, scale, inv_scale, lim,
                                       ahat_i8, ahat_s, ahat_inv, ahat_lim, write_ahat,
                                       q0, q1, d0, d1, ahat_qscale, C, ahat_ng);
                    local_delta_max = fmaxf(local_delta_max, fmaxf(fabsf(d0), fabsf(d1)));
                    const int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
                    if constexpr (PACK)
                        yqp_base[hw_out * row_bytes + (c_global0 >> 1)] =
                            (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
                    else
                        reinterpret_cast<int16_t*>(yqp_base)[
                            (hw_out * row_bytes + c_global0) >> 1] =
                            (int16_t)(((uint8_t)i0) | (((uint16_t)(uint8_t)i1) << 8));
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
            const float lim = (PACK || a4) ? 7.0f : 127.0f;
            const long ci = (long)n * HW_OUT * C + hwo * (long)C + c_global0;
            float q0, q1, d0, d1;
            if (reduce_only) {
                if (ahat_is_imode(ahat_i8, ahat_s)) {
                    const int ia0 = ahat_load_int(a_hat_cache, ci, ahat_lim);
                    const int ia1 = ahat_load_int(a_hat_cache, ci + 1, ahat_lim);
                    d0 = a0 - (float)ia0 * inv_scale;
                    d1 = a1 - (float)ia1 * inv_scale;
                } else {
                    float bs, binv, blim;
                    ahat_resolve(ahat_i8, ahat_qscale, ci, C, ahat_ng,
                                 ahat_s, ahat_inv, ahat_lim, bs, binv, blim);
                    const float2 cpv = ahat_load2(a_hat_cache, ci, ahat_i8, bs);
                    d0 = a0 - cpv.x; d1 = a1 - cpv.y;
                }
                local_delta_max = fmaxf(local_delta_max, fmaxf(fabsf(d0), fabsf(d1)));
                continue;
            }
            ahat_quant_update2(a_hat_cache, ci, a0, a1, scale, inv_scale, lim,
                               ahat_i8, ahat_s, ahat_inv, ahat_lim, write_ahat,
                               q0, q1, d0, d1, ahat_qscale, C, ahat_ng);
            local_delta_max = fmaxf(local_delta_max, fmaxf(fabsf(d0), fabsf(d1)));
            const int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
            if constexpr (PACK)
                yqp_base[hwo * row_bytes + (c_global0 >> 1)] =
                    (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
            else
                reinterpret_cast<int16_t*>(yqp_base)[(hwo * row_bytes + c_global0) >> 1] =
                    (int16_t)(((uint8_t)i0) | (((uint16_t)(uint8_t)i1) << 8));
        }
    }

    if constexpr (PACK) {
        if (!reduce_only && g == 0 && Kpad > C) {
            const int tail_bytes = KpadH - C / 2;
            for (long idx = threadIdx.x; idx < HW_OUT * (long)tail_bytes; idx += blockDim.x) {
                const long hw = idx / tail_bytes;
                const int pb = idx % tail_bytes;
                yqp_base[hw * (long)KpadH + C / 2 + pb] = 0;
            }
        }
    }

    // Every thread reaches this: the loops above are block-strided and nothing returns early, so
    // the __syncthreads() inside the helper is safe. absmax_buf == nullptr makes it a no-op (and
    // it returns before its first barrier, uniformly across the grid).
    gn_report_delta_absmax(local_delta_max, sdata, absmax_buf, next_scale_out, next_inv_out,
                           retire_count, Q_level, safety);
}

// -----------------------------------------------------------------------------
//   Op:       MoDiff GroupNorm(+mod)(+SiLU)+2x resize + temporal-delta quantize + in-place a_hat
//   Inputs:   same as group_norm_silu_quantize_resize_nhwc, plus a_hat_cache fp16 [N,C,Ho,Wo]
//             (POST-resize resolution, modified in place)
//   Outputs:  int8/packed-int4 codes of Q(a_t - a_hat_{t+1}) at the resized resolution;
//             a_hat_cache advanced to a_hat_t
//   Computes: MoDiff Eqs 9-10 with A(.) = resize o GN o SiLU, all in one launch
//   Fuses:    the eight updown ResBlocks' GN+SiLU+resize+delta-quantize+cache-update, which
//             previously ran as a standalone PyTorch resize plus a separate delta-quantize
//             because _prequant_gn_resize_conv gates on `not modiff` (measured 2.35 ms/step at
//             batch 128: +1.20 upsample, +0.44 avg_pool, +0.71 GN+SiLU-only)
//   Constraints: as the baseline twin, plus a_hat_cache fp16 with N*C*Ho*Wo elements. The scale
//             is a device pointer, so it works with the static per-step table and with the
//             retained dynamic scale alike.
//   Dynamic: pass real 1-element absmax_buf/scale_out/inv_scale_out/retire_count to have the
//             per-call scale discovered here instead of supplied. With report_next=false that is
//             a separate reduction-only launch of this same kernel followed by the quantizing one
//             (a fresh scale, used immediately -- what the other 62 convs do by default); with
//             report_next=true the quantizing launch publishes for a later step at no extra pass.
//             All four empty => static, the original behaviour, bit-identical.
// -----------------------------------------------------------------------------
torch::Tensor group_norm_silu_delta_quantize_resize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack,
    torch::Tensor a_hat_cache,         // fp16 or int8, POST-resize shape, updated in place
    torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
    torch::Tensor retire_count, double Q_level, bool report_next, double safety,
    bool a4,
    bool write_ahat,
    torch::Tensor ahat_scale
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "gn_quantize_resize expects [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kFloat32,
                "gn_quantize_resize: only fp16/fp32 supported");
    TORCH_CHECK(resize == 1 || resize == -1, "gn_quantize_resize: resize must be +1 or -1");
    bool ahat_i8 = false;
    const float* ahat_qscale_ptr = nullptr;
    int ahat_ng = 0;
    __half* cache_ptr = nullptr;
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
    // a_hat is cached at the POST-resize (conv input) resolution -- the same shape the unfused
    // path already uses -- so this fusion changes no state layout.
    TORCH_CHECK(a_hat_cache.numel() == (long)N * C * Ho * Wo,
                "gn_delta_quantize_resize: a_hat_cache must have N*C*Ho*Wo elements (post-resize)");
    // The a_hat read/write in both the UP and DOWN paths is a vec2 at `ci`, which is even only if
    // the group's first channel is: ci = n*HW_OUT*C + hw_out*C + c_start + 2*cpair, and C is a
    // multiple of the alignment already required elsewhere. An odd CPG would make c_start odd on
    // odd groups and misalign the __half2. Assert it rather than comment it -- a misaligned
    // reinterpret_cast is undefined behaviour that can appear to work.
    TORCH_CHECK((C / num_groups) % 2 == 0,
                "gn_delta_quantize_resize: channels-per-group must be even for the vec2 a_hat "
                "access (C=", C, ", groups=", num_groups, ")");
    bind_ahat_cache(a_hat_cache, ahat_scale, cache_ptr, ahat_i8, ahat_qscale_ptr,
                    "group_norm_silu_delta_quantize_resize_nhwc", &ahat_ng);
    const bool block_commit = ahat_i8 && ahat_ng > 0 && !pack;
    TORCH_CHECK(!(pack && ahat_i8 && ahat_ng > 0),
                "gn_delta_quantize_resize: along-C int8 a_hat is int8-only (not packed int4)");

    // Dynamic scale, exactly as group_norm_silu_delta_quantize_nhwc defines it:
    //   dynamic -- a reduction-only launch measures THIS call's delta range and publishes
    //              Q_level/absmax into scale_out, which the quantizing launch then reads. The
    //              range is fresh and used immediately. Costs one extra pass.
    //   report  -- no extra pass; the quantizing launch reduces the range it is already computing
    //              and publishes it for a LATER step, quantizing meanwhile with `scale` as given.
    const bool have_bufs = absmax_buf.numel() > 0;
    const bool dynamic = have_bufs && !report_next;
    const bool report = have_bufs && report_next;
    if (have_bufs) {
        TORCH_CHECK(scale_out.numel() > 0 && inv_scale_out.numel() > 0 && retire_count.numel() > 0,
                    "gn_delta_quantize_resize: dynamic mode needs absmax_buf, scale_out, "
                    "inv_scale_out and retire_count together");
        TORCH_CHECK(absmax_buf.scalar_type() == torch::kFloat32
                        && scale_out.scalar_type() == torch::kFloat32
                        && inv_scale_out.scalar_type() == torch::kFloat32,
                    "gn_delta_quantize_resize: absmax_buf/scale_out/inv_scale_out must be fp32");
        TORCH_CHECK(retire_count.scalar_type() == torch::kInt32,
                    "gn_delta_quantize_resize: retire_count must be int32");
    }
    float* absmax_ptr = have_bufs ? absmax_buf.data_ptr<float>() : nullptr;
    float* nscale_ptr = have_bufs ? scale_out.data_ptr<float>() : nullptr;
    float* ninv_ptr = have_bufs ? inv_scale_out.data_ptr<float>() : nullptr;
    unsigned int* retire_ptr =
        have_bufs ? (unsigned int*)retire_count.data_ptr<int>() : nullptr;

    // gn_load2 is overloaded for `const float*` and `const __half*` only, so the fp16 launch has
    // to reinterpret at::Half -- instantiating the kernel on at::Half itself does not compile.
#define MODIFF_GNDQR_LAUNCH(T, ATT, UPV, PK, SCALEP, AMAX, NSC, NIV, RET, RONLY)             \
    group_norm_silu_delta_quantize_resize_nhwc_kernel<T, true, UPV, PK>                     \
        <<<grid, block, shmem, stream>>>(                                                   \
            reinterpret_cast<const T*>(x.data_ptr<ATT>()),                                  \
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),                               \
            cache_ptr,                                                                      \
            reinterpret_cast<const T*>(weight.data_ptr<ATT>()),                             \
            reinterpret_cast<const T*>(bias.data_ptr<ATT>()),                               \
            has_mod ? reinterpret_cast<const T*>(mod_scale.data_ptr<ATT>()) : nullptr,      \
            has_mod ? reinterpret_cast<const T*>(mod_shift.data_ptr<ATT>()) : nullptr,      \
            (SCALEP), smooth_ptr, C, HW, (int)num_groups, (float)eps,                       \
            apply_silu, Kpad, W, a4,                                              \
            (AMAX), (NSC), (NIV), (RET), (float)Q_level, (float)safety, (RONLY),   \
            write_ahat && !block_commit, ahat_i8, ahat_qscale_ptr, ahat_ng)
#define MODIFF_GNDQR_DISPATCH(T, ATT, SCALEP, AMAX, NSC, NIV, RET, RONLY)                    \
    do {                                                                                    \
        if (up &&  pack) MODIFF_GNDQR_LAUNCH(T, ATT, true,  true,  SCALEP, AMAX, NSC, NIV, RET, RONLY); \
        if (up && !pack) MODIFF_GNDQR_LAUNCH(T, ATT, true,  false, SCALEP, AMAX, NSC, NIV, RET, RONLY); \
        if (!up &&  pack) MODIFF_GNDQR_LAUNCH(T, ATT, false, true,  SCALEP, AMAX, NSC, NIV, RET, RONLY); \
        if (!up && !pack) MODIFF_GNDQR_LAUNCH(T, ATT, false, false, SCALEP, AMAX, NSC, NIV, RET, RONLY); \
    } while (0)
#define MODIFF_GNDQR_BOTH(SCALEP, AMAX, NSC, NIV, RET, RONLY)                                \
    do {                                                                                    \
        if (x.scalar_type() == torch::kFloat32)                                             \
            MODIFF_GNDQR_DISPATCH(float, float, SCALEP, AMAX, NSC, NIV, RET, RONLY);        \
        else                                                                                \
            MODIFF_GNDQR_DISPATCH(__half, at::Half, SCALEP, AMAX, NSC, NIV, RET, RONLY);    \
    } while (0)

    if (dynamic) {
        // Reduction-only: same body, stores nothing, publishes Q_level/absmax into scale_out.
        // `scale` is not read, so it may legitimately be an uninitialised buffer here.
        MODIFF_GNDQR_BOTH(nullptr, absmax_ptr, nscale_ptr, ninv_ptr, retire_ptr, true);
        C10_CUDA_CHECK(cudaGetLastError());
    }
    // The quantizing launch. In `dynamic` it reads the scale the pass above just wrote (and must
    // NOT report again, or it would re-run the election and overwrite that scale mid-step); in
    // `report` it quantizes with the caller's scale and publishes for a later step.
    const float* scale_ptr_eff = dynamic ? (const float*)nscale_ptr : scale.data_ptr<float>();
    MODIFF_GNDQR_BOTH(scale_ptr_eff,
                      report ? absmax_ptr : nullptr, report ? nscale_ptr : nullptr,
                      report ? ninv_ptr : nullptr, report ? retire_ptr : nullptr, false);
#undef MODIFF_GNDQR_BOTH
#undef MODIFF_GNDQR_DISPATCH
#undef MODIFF_GNDQR_LAUNCH
    C10_CUDA_CHECK(cudaGetLastError());
    if (write_ahat && block_commit) {
        const long numel_out = (long)N * C * Ho * Wo;
        ahat_commit_block(cache_ptr, const_cast<float*>(ahat_qscale_ptr),
                          yq.data_ptr<int8_t>(), scale_ptr_eff, C, ahat_ng,
                          numel_out, stream);
        C10_CUDA_CHECK(cudaGetLastError());
    }
    return yq;
}

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

// =========================================================================
// Channel-major GN statistics: coalesced AND deterministic.
//
// Why a fourth variant. Measured 2026-08-04 (real checkpoint, batch 8, dynamic delta):
//     default gn_group_stats_kernel   17.69 ms/step   deterministic
//     MODIFF_GN_STATS_ALT=1           45.48 ms/step   NONdeterministic  (2.6x slower)
//     MODIFF_GN_STATS_ALT=2           29.45 ms/step   NONdeterministic  (1.7x slower)
// So both pre-existing alternatives lose on both axes, and gn_launch_group_stats' comment calling
// ALT=2 a "candidate replacement" for the group-major tree is wrong -- it is 11.8 ms/step slower.
// Their problem is atomicAdd: it serializes G-way contention AND makes the fp32 summation order
// irreproducible, which showed up directly as latents differing by up to 1.3e-1 between two
// replays of the same seed.
//
// Meanwhile the default kernel's problem is coalescing. It reads group-major: thread t handles
// (c_local = t % CPG, hw = t / CPG), so at CPG=6 (C=192, G=32) a warp reads 12-byte runs strided
// by C*2 bytes -- it touches ~5 sectors per warp and uses 12 B of each. That is the 9.51 ms/step
// which the bucket breakdown shows is MoDiff's ENTIRE overhead against its own baseline.
//
// This variant fixes coalescing without atomics, by choosing the thread->element map so that a
// thread's GROUP is invariant across the whole loop:
//
//   blockDim.x == C, thread t owns channel t for the entire kernel. The loop steps over spatial
//   positions, so every read is x[(n*HW + hw)*C + t] -- consecutive threads, consecutive addresses,
//   fully coalesced 128 B/warp. Because t is fixed, t's group (t / CPG) is fixed too, so each
//   thread accumulates privately into registers with no contention and no atomics.
//
// The per-group combine is then a fixed-order shared-memory pass (lane g sums its CPG entries in
// index order), and the cross-block combine is a second kernel over a [N,G,nblocks] partials
// buffer, also in index order. Every summation order is a pure function of the shapes, so the
// result is bit-reproducible across launches and grid sizes.
//
// It is NOT bit-identical to the group-major tree -- a different (equally valid) fp32 summation
// order changes mean/var by ~1 ULP, which can flip the occasional int8 code. That was the reason
// gn_group_stats_vec2_kernel was reverted, but it no longer disqualifies anything: the default
// delta quantizer is now dynamic, so its scale is recomputed per call and bit-exactness against
// the old two-kernel path was already given up by design. The acceptance criterion is agreement
// with an fp64 reference, not agreement with the old kernel.
//
// K = channels per thread, so a block is C/K threads and thread t owns channels t, t+B, ... with
// B = blockDim.x = C/K. K=1 is the original one-thread-per-channel form and generates the same code.
//
// K>1 exists because C <= 1024 is not "every channel count in this UNet", which is what an earlier
// version of this comment claimed. GroupNorm in a DECODER ResBlock sees the concatenated
// cat([h, hs.pop()]) width -- 1152 and 1536 here -- so those blocks silently fell back to the
// group-major tree. Measured 2026-08-04: gn_group_stats_kernel was still running at 142.3 ms/batch
// (0.71 ms/step) on the MoDiff path for exactly those layers.
//
// The spatial loop stays outermost so X is read ONCE regardless of K; putting the channel loop
// outside would re-walk the whole tensor K times. For a fixed hw the K inner loads are K separate
// runs of B consecutive threads over B consecutive channels, so each is still fully coalesced.
//
// Determinism is unchanged: each channel's spatial sum accumulates in the same hw order as K=1, and
// the group combine still reads shared memory in ascending channel index. The result is a pure
// function of the shapes, so it is reproducible across launches and grid sizes -- it is simply not
// bit-identical to the group-major tree, which the delta quantizer no longer requires.
//
// Requires C % K == 0 and C/K <= 1024 and C/K >= G; gn_launch_group_stats checks all three.
template <typename TIn, int K>
__global__ void gn_stats_partials_chanmajor_kernel(
    const TIn* __restrict__ X,
    float* __restrict__ part_sum,      // [N, G, nblocks]
    float* __restrict__ part_sumsq,
    int C, long HW, int G, int nblocks,
    // ---- OPTIONAL SPLIT INPUT (the decoder skip-concat fold) ----------------------------------
    // When X2 != nullptr the tensor is not materialized: X holds the first C1 channels and X2 the
    // remaining C - C1, each channels_last in its own buffer, and this kernel reads them in place
    // instead of reading a concatenation someone else had to write first. When OutCat != nullptr it
    // ALSO writes that concatenation as it goes, for the consumers that still need it (the apply
    // kernel and the ResBlock's 1x1 skip conv).
    //
    // WHY THIS IS THE WHOLE OPTIMISATION. Today the decoder pays cat2 (read C + write C) and then
    // this kernel reads the result (C) -- 3C of traffic. Reading the halves directly and emitting the
    // concatenation here costs read C + write C = 2C, because the read was already being paid. cat2
    // measures at 81% of peak on the shapes that dominate, so it is pure traffic and removing a pass
    // over it is the entire win. Measured ceiling for the full fold (this plus splitting the skip
    // conv) was 1.45-2.01% end to end; this half needs no change to the apply kernel or the conv.
    //
    // COALESCING IS PRESERVED, and not by luck: c is t + k*B, so within one k every consecutive
    // thread reads a consecutive channel, and a warp only straddles the C1 boundary if C1 % 32 != 0.
    // Every C1 this UNet concatenates is 192, 384 or 768 -- all multiples of 32 -- so no warp splits.
    // The concat store is at the output's own stride C and is fully coalesced by construction.
    //
    // DETERMINISM AND BIT-EXACTNESS ARE UNCHANGED. Each channel's spatial sum still accumulates in
    // ascending hw order in one thread's register, and the group combine still reads shared memory in
    // ascending channel index. Only the ADDRESS a value is loaded from changes, never the value or
    // the order it is added in -- so the partials are bit-identical to the concatenated path, which
    // the gate in integration/tests/test_cat2_gn_fold.py asserts rather than assumes.
    const TIn* __restrict__ X2 = nullptr,
    TIn* __restrict__ OutCat = nullptr,
    int C1 = 0
) {
    const int CPG = C / G;
    const int B = blockDim.x;                  // == C / K
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const bool split = (X2 != nullptr);
    const int c2 = C - C1;
    // Per-buffer row strides: the halves are channels_last in their OWN widths, so their rows are
    // C1 and c2 wide while the concatenation's row is C wide.
    const TIn* a_base = X + (long)n * HW * (split ? C1 : C);
    const TIn* b_base = split ? (X2 + (long)n * HW * c2) : nullptr;
    TIn* cat_base = (OutCat != nullptr) ? (OutCat + (long)n * HW * C) : nullptr;

    float s[K], sq[K];
#pragma unroll
    for (int k = 0; k < K; ++k) { s[k] = 0.0f; sq[k] = 0.0f; }

    for (long hw = blockIdx.x; hw < HW; hw += nblocks) {
        const long row = hw * (long)C;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int c = t + k * B;
            // The raw element is kept so the concat store is a COPY, not a round-trip through
            // float: __half -> float -> __half is lossless, but copying the bits cannot even raise
            // the question, and cat2_channels_last_fp16 is a pure copy that this must match.
            TIn raw;
            if (!split) {
                raw = a_base[row + c];
            } else if (c < C1) {
                raw = a_base[hw * (long)C1 + c];
            } else {
                raw = b_base[hw * (long)c2 + (c - C1)];
            }
            if (cat_base != nullptr) cat_base[row + c] = raw;
            const float v = gn_load(&raw, 0);
            s[k] += v;
            sq[k] += v * v;
        }
    }

    extern __shared__ float sdata[];
    float* ss = sdata;                         // [C]
    float* sq_s = sdata + C;                   // [C]
#pragma unroll
    for (int k = 0; k < K; ++k) {
        ss[t + k * B] = s[k];
        sq_s[t + k * B] = sq[k];
    }
    __syncthreads();

    // One lane per group sums its CPG channels in ascending index order -- fixed order, so
    // reproducible. G is 32 here, so this costs 32 active lanes for CPG steps, once per block.
    if (t < G) {
        float gs = 0.0f, gsq = 0.0f;
        const int c0 = t * CPG;
        for (int k = 0; k < CPG; ++k) {
            gs += ss[c0 + k];
            gsq += sq_s[c0 + k];
        }
        const long o = ((long)n * G + t) * nblocks + blockIdx.x;
        part_sum[o] = gs;
        part_sumsq[o] = gsq;
    }
}

// ============================================================================================
// gn_stats_partials_chanmajor_vec2_kernel -- the same statistics, BIT-IDENTICAL, at 1.21x.
//
// WHAT IT CHANGES, and nothing else. The kernel above is LATENCY-bound, not bandwidth-bound: it
// reached only 50-72% of peak on this UNet's shapes (49.8% in-model, measured 2026-08-26 by nsys
// against the apply kernel's 84.5% on the same tensors). Two causes, one fix each:
//
//   1. one 2-byte load per thread per hw iteration. A warp's 32 lanes therefore request 64 B --
//      half a sector -- and the loop is a serial accumulate, so exactly one load per thread is in
//      flight. FIX: a thread owns two ADJACENT channels and loads them as one __half2 (4 B, so a
//      warp requests a full 128 B).
//   2. the hw loop is not unrolled, so nothing hides the load latency. FIX: process four hw values
//      per iteration with the four loads issued before the first dependent add.
//
// WHY IT IS BIT-IDENTICAL, which is the whole point -- the delta quantizer's a_hat invariant and
// every committed FID number depend on these partials not moving:
//   (i)   each channel still has its OWN fp32 accumulator (2t and 2t+1 are separate registers);
//   (ii)  that accumulator still walks hw in ASCENDING order -- unrolling reorders no adds, it only
//         issues the loads earlier;
//   (iii) shared memory is still indexed by channel and the group combine still sums CPG channels
//         in ascending channel index;
//   (iv)  the fp16 -> fp32 conversion is the same __half2float, applied per element.
// Only the ADDRESS a value is loaded from and the INSTRUCTION that loads it change, never a value
// and never the order it is added in. integration/tests/test_gn_stats_vec2.py asserts equality of
// both partial arrays against the kernel above on all 18 real shapes rather than assuming it.
//
// MEASURED, freq-weighted over all 18 real shapes at batch 128 (A40):
//   3.305 -> 2.603 ms/step, 1.27x, and 16-72% -> 22-84% of peak bandwidth per shape.
// Wins or ties on 16 of 18; the two ties are C=1536, whose scalar path already uses K=2.
//
// It also removes the need for K > 1 on this UNet: the block is C/2 threads, so C <= 2048 fits
// under the 1024-thread cap where the scalar kernel needs the K split from C = 1152 upwards.
//
// NOT used for the decoder skip-concat fold (the X2/OutCat path above): that kernel also WRITES a
// concatenation, and folding a vectorized store into it is a separate change with its own gate.
// Requires C even, C/2 <= 1024, C/2 >= G, C % G == 0 -- gn_launch_group_stats checks all four.
template <typename TIn>
__global__ void gn_stats_partials_chanmajor_vec2_kernel(
    const TIn* __restrict__ X,
    float* __restrict__ part_sum,      // [N, G, nblocks]
    float* __restrict__ part_sumsq,
    int C, long HW, int G, int nblocks,
    // Same optional split / concat-emitting contract as the scalar twin above. A vec2 pair at an
    // even channel NEVER straddles the C1 boundary, because every C1 this UNet concatenates is a
    // multiple of 32: an even c < C1 has c+1 <= C1-1, and an even c >= C1 has both halves past it.
    // gn_stats_from_cat2 TORCH_CHECKs C1 % 2 rather than trusting that.
    const TIn* __restrict__ X2 = nullptr,
    TIn* __restrict__ OutCat = nullptr,
    int C1 = 0
) {
    const int CPG = C / G;
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const long C2 = C / 2;
    const int c = 2 * t;                       // this thread's first channel, always even
    const bool split = (X2 != nullptr);
    const int c2w = C - C1;
    // Both halves of the pair live in the same group whenever CPG is even; when CPG is odd the
    // pair can straddle a group boundary, which is FINE -- the two accumulators are written to
    // their own channel slots in shared memory and the group combine reads by channel index.
    const __half2* a_base = reinterpret_cast<const __half2*>(
        X + (long)n * HW * (split ? C1 : C));
    const __half2* b_base = split
        ? reinterpret_cast<const __half2*>(X2 + (long)n * HW * c2w) : nullptr;
    __half2* cat_base = (OutCat != nullptr)
        ? reinterpret_cast<__half2*>(OutCat + (long)n * HW * C) : nullptr;
    // Which buffer this thread reads, and at what pair offset, is loop-invariant.
    const __half2* src = (!split) ? a_base : (c < C1 ? a_base : b_base);
    const long src_pairs = (!split) ? C2 : (c < C1 ? (long)C1 / 2 : (long)c2w / 2);
    const long src_t = (!split || c < C1) ? (long)t : (long)(c - C1) / 2;
    float sa = 0.0f, sqa = 0.0f, sb = 0.0f, sqb = 0.0f;
    long hw = blockIdx.x;
    for (; hw + 3L * nblocks < HW; hw += 4L * nblocks) {
        const __half2 r0 = src[(hw) * src_pairs + src_t];
        const __half2 r1 = src[(hw + nblocks) * src_pairs + src_t];
        const __half2 r2 = src[(hw + 2L * nblocks) * src_pairs + src_t];
        const __half2 r3 = src[(hw + 3L * nblocks) * src_pairs + src_t];
        if (cat_base != nullptr) {             // a pure bit copy, exactly as the scalar twin
            cat_base[(hw) * C2 + t] = r0;
            cat_base[(hw + nblocks) * C2 + t] = r1;
            cat_base[(hw + 2L * nblocks) * C2 + t] = r2;
            cat_base[(hw + 3L * nblocks) * C2 + t] = r3;
        }
        const float2 f0 = __half22float2(r0), f1 = __half22float2(r1);
        const float2 f2 = __half22float2(r2), f3 = __half22float2(r3);
        sa += f0.x; sqa += f0.x * f0.x; sb += f0.y; sqb += f0.y * f0.y;
        sa += f1.x; sqa += f1.x * f1.x; sb += f1.y; sqb += f1.y * f1.y;
        sa += f2.x; sqa += f2.x * f2.x; sb += f2.y; sqb += f2.y * f2.y;
        sa += f3.x; sqa += f3.x * f3.x; sb += f3.y; sqb += f3.y * f3.y;
    }
    for (; hw < HW; hw += nblocks) {
        const __half2 r = src[hw * src_pairs + src_t];
        if (cat_base != nullptr) cat_base[hw * C2 + t] = r;
        const float2 v = __half22float2(r);
        sa += v.x; sqa += v.x * v.x;
        sb += v.y; sqb += v.y * v.y;
    }
    extern __shared__ float sdata[];
    float* ss = sdata;                         // [C]
    float* sq_s = sdata + C;                   // [C]
    ss[2 * t] = sa;     sq_s[2 * t] = sqa;
    ss[2 * t + 1] = sb; sq_s[2 * t + 1] = sqb;
    __syncthreads();
    if (t < G) {
        float gs = 0.0f, gsq = 0.0f;
        const int c0 = t * CPG;
        for (int k = 0; k < CPG; ++k) { gs += ss[c0 + k]; gsq += sq_s[c0 + k]; }
        const long o = ((long)n * G + t) * nblocks + blockIdx.x;
        part_sum[o] = gs;
        part_sumsq[o] = gsq;
    }
}


// Cross-block combine, also in fixed index order. One thread per (sample, group).
__global__ void gn_stats_reduce_partials_kernel(
    const float* __restrict__ part_sum,
    const float* __restrict__ part_sumsq,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int nblocks, long group_size, float eps, int NG
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NG) return;
    float s = 0.0f, sq = 0.0f;
    const long base = (long)i * nblocks;
    for (int b = 0; b < nblocks; ++b) {
        s += part_sum[base + b];
        sq += part_sumsq[base + b];
    }
    // Same statement sequence as gn_group_stats_kernel's finalize, for like-for-like fp32 codegen.
    float mean = s / (float)group_size;
    float var = sq / (float)group_size - mean * mean;
    var = fmaxf(var, 0.0f);
    mean_out[i] = mean;
    inv_std_out[i] = rsqrtf(var + eps);
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

// =========================================================================
// PROTOTYPE (Stage A of docs/gn_stats_in_epilogue_2026-08-11). Not wired to anything.
//
// Mimics what a conv EVT epilogue would do if it emitted the GroupNorm partial sums as an auxiliary
// output: grid the tensor exactly as the conv's 128x128 threadblock tiles do (verified shape,
// conv2d_evt.cu instantiates GemmShape<128,128,128>), and have each tile accumulate per-(n, group)
// sum/sumsq into shared memory before writing its own slot.
//
// The question it exists to answer is NOT "does it fit" -- the accumulator is at most 56 (n,g) pairs,
// 448 B, worked out in the design doc. It is whether the per-thread scatter into that shared array
// costs less than the 4.75 ms/step full read (gn_stats_partials_chanmajor) it would replace.
//
// ANSWER, 2026-08-11: not like this. Weighted over the model's conv-output shapes the prototype is
// 30.83 ms/step against 4.75, worst at the shapes with fewest slots (768x4x4: 4.51x), and
// nondeterministic. The arithmetic is right (max rel err 1.4e-3 vs an fp32 reference), so this is a
// cost and determinism result, not a correctness one.
//
// WHAT IT DOES AND DOES NOT REFUTE. It refutes the SCATTER-VIA-SHARED-ATOMICS implementation, which is
// what a straightforward epilogue node would reach for. It does not by itself refute the concept: in a
// real EVT epilogue each thread's fragment is already in registers with a known (n, g), so a warp-level
// tree reduction per group could replace the atomics entirely. But note the gap is 6.5x and the
// epilogue version pays its reduction ON TOP of the conv's existing epilogue work, so a tree version
// has to beat a large margin, and the contention it must avoid is structural: few slots, many threads.
// Kept unreferenced on the dead-code policy's one admissible ground -- the reason it is unused is the
// finding. Harness: the numbers above are reproducible from the kernel plus group_norm_silu_nhwc.
//
// SECOND ANSWER, 2026-08-13 (docs/gn_stats_in_epilogue_2026-08-11/FINDINGS.md). The warp-tree rewrite
// below had never been re-measured after it was written. Two of its three gates pass, and they are the
// two the rewrite was FOR: max rel err 5.7e-07 against an fp32 reference, and DETERMINISTIC on every
// shape, where the atomics version was non-deterministic on all of them.
//
// IT FAILS THE SPEED GATE BY 3.6x. bench_gn_stats_tiles.py grades against a "shipped us" column
// inherited from the 2026-08-11 report -- 11.94 ms weighted -- because this kernel has no pybind entry.
// That column is 3.8x TOO LARGE: measured directly through group_norm_silu_delta_quantize_pack_nhwc
// with CUPTI self-time, the stats pass is 3.14 ms weighted (it is the FULL GN op that costs ~11 ms).
// So the prototype's 11.43 ms loses to 3.14 ms, and the earlier claim in this comment that it was
// 0.96x was an artifact of the wrong baseline. Retracted.
//
// AND THERE IS NO HEADROOM HERE TO CHASE. This kernel achieves 444/487/397 GB/s on the three shapes
// that carry weight -- 64%/70%/57% of the A40's 696 -- reading X exactly once. Only 768x4x4 is poor
// (144 GB/s, 21%), a 3 MiB tensor where the launch dominates. The earlier "9-19% of peak, 6.2x off
// roofline, ~3.8% of end-to-end available" was computed from the inherited column and is withdrawn.
//
// Which also explains why the GN family is only 1.19x faster than fp16 at W4A4 and why that is not a
// defect: GroupNorm's input is fp16 in EVERY mode, so quantization never shrinks this pass's traffic.
// A memory-bound pass over X at 57-70% of peak is the answer, not a symptom.
//
// Slots are per (tile_m, tile_n), never atomics: MODIFF_GN_STATS_ALT=2 measured an atomic GN
// reduction 1.7x slower AND nondeterministic (1.27e-1 latent drift between replays of one seed).
template <typename TIn>
__global__ void gn_stats_from_tiles_kernel(
    const TIn* __restrict__ X, float* __restrict__ part_sum, float* __restrict__ part_sumsq,
    int C, long HW, int G, int Mt, int Nt, int n_tiles_n, long M) {
  // [2 * WARPS * (PAIRS+1)]: sum then sumsq, PRIVATE PER WARP, one trailing sentinel slot per warp.
  //
  // Rewritten 2026-08-12. The first version accumulated with two shared atomicAdd per element into a
  // single [2*PAIRS] array, and failed both of its gates: 30.83 ms/step against the 4.75 ms pass it
  // replaces (6.5x the wrong way) and det=False on every shape. Both came from the same place --
  // 23-56 slots with 256 threads contending, and float atomicAdd being order-dependent.
  //
  // This version has NO atomics. Each element is reduced within its warp among exactly the lanes that
  // share its (n, g) slot -- `__match_any_sync` finds them, a masked butterfly sums them -- and only
  // that group's leader lane touches shared memory, in its own warp's private slot array. The
  // cross-warp sum is then a fixed w = 0..WARPS-1 loop. Every addition happens in a fixed order, so
  // determinism is structural rather than hoped for.
  //
  // Coalescing is preserved: the element loop still walks row-major, so consecutive lanes read
  // consecutive channels. A slot-outer loop would have been simpler but reads only CPG (6..48)
  // contiguous floats per row, which is worse exactly where C is smallest.
  extern __shared__ float sacc[];
  const int CPG = C / G;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int WARPS = (int)(blockDim.x >> 5);
  const int tm = blockIdx.x, tn = blockIdx.y;
  const long m0 = (long)tm * Mt;
  const int  n0 = tn * Nt;
  // (n, g) range this tile can touch. Derived per tile, because along M a tile spans several samples
  // once HW < Mt, and along N it straddles groups whenever Nt % CPG != 0 -- both true in this model.
  const int n_first = (int)(m0 / HW), n_last = (int)(min(m0 + Mt - 1, M - 1) / HW);
  const int g_first = n0 / CPG, g_last = min(n0 + Nt - 1, C - 1) / CPG;
  const int ng = n_last - n_first + 1, gg = g_last - g_first + 1;
  const int PAIRS = ng * gg;
  const int SLOTS = PAIRS + 1;                    // +1: the sentinel the tail iteration writes into
  float* wsum = sacc;                             // [WARPS * SLOTS]
  float* wsq = sacc + (long)WARPS * SLOTS;        // [WARPS * SLOTS]
  for (int i = threadIdx.x; i < 2 * WARPS * SLOTS; i += blockDim.x) sacc[i] = 0.f;
  __syncthreads();
  const long rows = min((long)Mt, M - m0);
  const int cols = min(Nt, C - n0);
  // Every lane runs the SAME number of iterations, out-of-range ones carrying v=0 and the sentinel
  // slot. Uniform trip count is what makes the full 0xffffffff mask below correct -- taking
  // __activemask() instead would be reading a value the compiler is free to change under
  // reconvergence, which is the class of bug this kernel already paid for once.
  const long total = rows * cols;
  const long iters = (total + blockDim.x - 1) / blockDim.x;
  for (long it = 0; it < iters; ++it) {
    const long idx = it * blockDim.x + threadIdx.x;
    const bool ok = idx < total;
    float v = 0.f;
    int slot = PAIRS;                             // sentinel
    if (ok) {
      const long r = idx / cols;  const int cc = (int)(idx % cols);
      const long m = m0 + r;      const int c = n0 + cc;
      v = gn_load(X, m * (long)C + c);
      slot = ((int)(m / HW) - n_first) * gg + (c / CPG - g_first);
    }
    // Segmented reduction over the lanes sharing `slot`. `__match_any_sync` gives a bit per such lane.
    //
    // An INCLUSIVE SCAN upward (Hillis-Steele) rather than a `__shfl_down_sync` halving tree, and the
    // reason is SPEED, not correctness. Both forms were built and measured: the down-shift tree is
    // equally accurate (5.5e-7 against an fp32 reference, same as this) and ~9% slower per shape, which
    // is the difference between 1.04x the shipped pass -- still losing -- and 0.95x. An earlier
    // comment here claimed the down-shift form dropped and double-counted lanes; that was wrong, and
    // the 10.4% error it cited came from a test feeding this kernel a contiguous NCHW tensor when it
    // reads channels_last.
    //
    // The scan is correct here because equal slots within a warp are always CONTIGUOUS: lanes take
    // consecutive `idx`, so they walk consecutive channels, and a warp spans at most one row boundary
    // (cols >= 64 > 32 for every tile in this model) across which the group index necessarily changes
    // (cols > CPG). Contiguity is what makes the single test "is lane-off in my group?" sufficient:
    // if it is, every lane between us is too, so the partial it carries is entirely mine to absorb.
    const unsigned mask = __match_any_sync(0xffffffffu, slot);
    float s = v, q = v * v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float os = __shfl_up_sync(0xffffffffu, s, off);
      const float oq = __shfl_up_sync(0xffffffffu, q, off);
      if (lane >= off && ((mask >> (lane - off)) & 1u)) { s += os; q += oq; }
    }
    // After an inclusive scan the segment total sits in its HIGHEST lane, so that lane writes. Groups
    // have distinct slots by construction, so no two writers collide in this iteration, and the array
    // belongs to this warp alone -- no atomic, and a fixed addition order.
    if (lane == (31 - __clz(mask)) && slot != PAIRS) {
      wsum[warp * SLOTS + slot] += s;
      wsq[warp * SLOTS + slot] += q;
    }
  }
  __syncthreads();
  const long nblk = (long)gridDim.x * n_tiles_n;
  const long slot0 = (long)tm * n_tiles_n + tn;
  for (int i = threadIdx.x; i < PAIRS; i += blockDim.x) {
    const int nn = n_first + i / gg, g = g_first + i % gg;
    const long o = ((long)nn * G + g) * nblk + slot0;
    // Fixed order, w ascending: the one place the warps' partials meet, and the reason the whole
    // kernel is bit-reproducible across launches.
    float s = 0.f, q = 0.f;
    for (int w = 0; w < WARPS; ++w) { s += wsum[w * SLOTS + i]; q += wsq[w * SLOTS + i]; }
    part_sum[o] = s;  part_sumsq[o] = q;
  }
}

std::vector<torch::Tensor> gn_stats_from_tiles(torch::Tensor x, int64_t num_groups,
                                               int64_t Mt, int64_t Nt) {
  CHECK_CUDA(x);
  const int N = x.size(0), C = x.size(1);
  const long HW = (long)x.size(2) * x.size(3), M = (long)N * HW;
  const int tm = (int)((M + Mt - 1) / Mt), tn = (int)((C + Nt - 1) / Nt);
  auto o = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  auto ps = torch::zeros({(long)N * num_groups * tm * tn}, o);
  auto pq = torch::zeros({(long)N * num_groups * tm * tn}, o);
  // Per-WARP private slots now, so the buffer is 2 * WARPS * (PAIRS + 1) rather than 2 * PAIRS.
  // Worst case over this model's shapes is 56 pairs (design doc), +1 sentinel, x 8 warps x 2 arrays =
  // 912 floats = 3.6 KB. 256 pairs of slack is kept for the same reason it was before -- the bound is
  // this model's, not the kernel's -- and 3.6 KB was never the constraint; the atomics were.
  const int GNS_THREADS = 256, GNS_WARPS = GNS_THREADS / 32;
  const size_t shmem = 2 * (size_t)GNS_WARPS * 257 * sizeof(float);
  dim3 grid((unsigned)tm, (unsigned)tn);
  auto st = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat32)
    gn_stats_from_tiles_kernel<float><<<grid, GNS_THREADS, shmem, st>>>(
        x.data_ptr<float>(), ps.data_ptr<float>(), pq.data_ptr<float>(),
        C, HW, (int)num_groups, (int)Mt, (int)Nt, tn, M);
  else
    gn_stats_from_tiles_kernel<__half><<<grid, GNS_THREADS, shmem, st>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), ps.data_ptr<float>(),
        pq.data_ptr<float>(), C, HW, (int)num_groups, (int)Mt, (int)Nt, tn, M);
  C10_CUDA_CHECK(cudaGetLastError());
  return {ps, pq};
}

// =========================================================================
// cat2_gn_stats_fp16 -- the decoder skip-concat fold, standalone.
//
// Reads the two halves in place, emits their channel concatenation, and returns the GroupNorm
// (mean, inv_std) computed over it -- all in ONE pass over the data. It replaces
// `cat2_channels_last_fp16` followed by the stats pass inside a GN prologue: 3C of traffic becomes
// 2C, because the read the stats pass was going to do anyway is the read that feeds the copy.
//
// Deliberately standalone rather than wired into group_norm_silu_delta_quantize_pack_nhwc yet: this
// is the half with no arithmetic risk, it is separately gateable (test_cat2_gn_fold.py asserts the
// partials are BIT-IDENTICAL to the concatenated path and the emitted concat matches
// cat2_channels_last_fp16 exactly), and landing it before the wiring keeps a rebuild that touches the
// model's hottest prologue out of the same change as the kernel itself.
//
// Only the channel-major variant is offered. The group-major tree and the two atomic variants are
// kept in this file as recorded negative results (see gn_stats_partials_chanmajor_kernel's header);
// none of them is what runs, so none of them needs a split-input form.
std::vector<torch::Tensor> cat2_gn_stats_fp16(
    torch::Tensor a, torch::Tensor b, int64_t num_groups, double eps
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cat2_gn_stats_fp16: expected CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf,
                "cat2_gn_stats_fp16: expected FP16 tensors");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "cat2_gn_stats_fp16: expected 4D [N,C,H,W]");
    TORCH_CHECK(a.size(0) == b.size(0) && a.size(2) == b.size(2) && a.size(3) == b.size(3),
                "cat2_gn_stats_fp16: N,H,W must match");
    TORCH_CHECK(a.is_contiguous(at::MemoryFormat::ChannelsLast)
                && b.is_contiguous(at::MemoryFormat::ChannelsLast),
                "cat2_gn_stats_fp16: both inputs must be channels_last contiguous");
    const int N = (int)a.size(0), C1 = (int)a.size(1), C2 = (int)b.size(1);
    const int C = C1 + C2, H = (int)a.size(2), W = (int)a.size(3);
    const long HW = (long)H * W;
    TORCH_CHECK(C % num_groups == 0, "cat2_gn_stats_fp16: C must be divisible by num_groups");
    // The kernel's coalescing argument rests on no warp straddling the C1 boundary. Checked rather
    // than assumed -- every width this UNet concatenates satisfies it, and a future one that does not
    // should fail loudly here instead of silently reading at half efficiency.
    TORCH_CHECK(C1 % 32 == 0, "cat2_gn_stats_fp16: C1 must be a multiple of 32 (got ", C1,
                ") -- otherwise a warp straddles the two buffers and the loads stop coalescing");

    const int K = (C + 1023) / 1024;
    const int BLK = (K > 0) ? C / K : 0;
    TORCH_CHECK(K >= 1 && K <= 4 && (C % K) == 0 && BLK <= 1024 && BLK >= num_groups,
                "cat2_gn_stats_fp16: unsupported C=", C, " for the channel-major stats kernel");

    auto cat = torch::empty({N, C, H, W},
                            a.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto sopt = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    const int nblocks = (int)std::min<long>(HW, 32);
    const int NG = N * (int)num_groups;
    const long group_size = (long)(C / num_groups) * HW;
    auto part_sum = torch::empty({(long)NG * nblocks}, sopt);
    auto part_sumsq = torch::empty({(long)NG * nblocks}, sopt);
    auto mean = torch::empty({NG}, sopt);
    auto inv_std = torch::empty({NG}, sopt);

    const size_t shmem = (size_t)2 * C * sizeof(float);
    dim3 grid((unsigned)nblocks, (unsigned)N);
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define GN_CAT2(KK)                                                                                    gn_stats_partials_chanmajor_kernel<__half, KK><<<grid, BLK, shmem, st>>>(                              reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),                                           part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),                                          C, HW, (int)num_groups, nblocks,                                                                   reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),                                           reinterpret_cast<__half*>(cat.data_ptr<at::Half>()), C1)
    // VEC2 FAST PATH for the fold, same kernel and same invariants as the non-split dispatch in
    // gn_launch_group_stats. Requires C1 even so a pair never straddles the halves' boundary.
    static const char* _v2c = std::getenv("MODIFF_GN_STATS_VEC2");
    const int BLK2 = C / 2;
    if (((_v2c == nullptr) || (_v2c[0] != '0')) && (C % 2) == 0 && (C1 % 2) == 0
        && BLK2 <= 1024 && BLK2 >= (int)num_groups) {
        gn_stats_partials_chanmajor_vec2_kernel<__half><<<grid, BLK2, shmem, st>>>(
            reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
            part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),
            C, HW, (int)num_groups, nblocks,
            reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(cat.data_ptr<at::Half>()), C1);
    } else
    switch (K) {
        case 1: GN_CAT2(1); break;
        case 2: GN_CAT2(2); break;
        case 3: GN_CAT2(3); break;
        default: GN_CAT2(4); break;
    }
#undef GN_CAT2
    C10_CUDA_CHECK(cudaGetLastError());
    const int fb = 128, fg = (NG + fb - 1) / fb;
    gn_stats_reduce_partials_kernel<<<fg, fb, 0, st>>>(
        part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        nblocks, group_size, (float)eps, NG);
    C10_CUDA_CHECK(cudaGetLastError());
    return {cat, mean, inv_std};
}

// Single-input twin of cat2_gn_stats_fp16: the SHIPPED stats pass, exposed on its own.
//
// This entry point exists because its ABSENCE caused the worst measurement error of 2026-08-13. With
// no way to time gn_stats_partials_chanmajor_kernel directly, its cost was inherited from an earlier
// report -- 11.94 ms weighted, which turned out to be 3.8x too large and to be the FULL GN op rather
// than the stats pass. A roofline claim was published from it and had to be retracted. A hot kernel
// with no way to measure it in isolation is a kernel whose cost will eventually be guessed.
//
// It is also what makes test_cat2_gn_fold.py's stats comparison mean anything: the split path has to
// agree with the CONTIGUOUS path through this entry. Comparing it against itself re-split (the first
// draft of that gate) tests determinism and calls it equivalence.
std::vector<torch::Tensor> gn_stats_fp16(torch::Tensor x, int64_t num_groups, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf,
                "gn_stats_fp16: expected a CUDA FP16 tensor");
    TORCH_CHECK(x.dim() == 4, "gn_stats_fp16: expected 4D [N,C,H,W]");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast),
                "gn_stats_fp16: expected channels_last contiguous");
    const int N = (int)x.size(0), C = (int)x.size(1);
    const long HW = (long)x.size(2) * x.size(3);
    TORCH_CHECK(C % num_groups == 0, "gn_stats_fp16: C must be divisible by num_groups");
    const int K = (C + 1023) / 1024;
    const int BLK = (K > 0) ? C / K : 0;
    TORCH_CHECK(K >= 1 && K <= 4 && (C % K) == 0 && BLK <= 1024 && BLK >= num_groups,
                "gn_stats_fp16: unsupported C=", C);
    auto sopt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    const int nblocks = (int)std::min<long>(HW, 32);
    const int NG = N * (int)num_groups;
    const long group_size = (long)(C / num_groups) * HW;
    auto part_sum = torch::empty({(long)NG * nblocks}, sopt);
    auto part_sumsq = torch::empty({(long)NG * nblocks}, sopt);
    auto mean = torch::empty({NG}, sopt);
    auto inv_std = torch::empty({NG}, sopt);
    const size_t shmem = (size_t)2 * C * sizeof(float);
    dim3 grid((unsigned)nblocks, (unsigned)N);
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define GN_ONE(KK)                                                                                 \
    gn_stats_partials_chanmajor_kernel<__half, KK><<<grid, BLK, shmem, st>>>(                      \
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),                                   \
        part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),                                  \
        C, HW, (int)num_groups, nblocks)
    switch (K) {
        case 1: GN_ONE(1); break;
        case 2: GN_ONE(2); break;
        case 3: GN_ONE(3); break;
        default: GN_ONE(4); break;
    }
#undef GN_ONE
    C10_CUDA_CHECK(cudaGetLastError());
    const int fb = 128, fg = (NG + fb - 1) / fb;
    gn_stats_reduce_partials_kernel<<<fg, fb, 0, st>>>(
        part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        nblocks, group_size, (float)eps, NG);
    C10_CUDA_CHECK(cudaGetLastError());
    return {mean, inv_std};
}

static void gn_launch_group_stats(
    const torch::Tensor& x, int N, int C, long HW, int num_groups, double eps,
    torch::Tensor& mean, torch::Tensor& inv_std
) {
    // NOTE ON THE VARIANT SELECTOR: `_alt` is a function-local static, so it is captured ONCE per
    // process at the first call. Setting MODIFF_GN_STATS_ALT between models in the same process is
    // silently ineffective -- every A/B of these variants must fork a fresh process per variant
    // (docs/modiff_correctness_2026-08-03/scripts/gn_stats_ab.py does).
    static const char* _alt = std::getenv("MODIFF_GN_STATS_ALT");

    // ALT=3 / default-when-eligible: channel-major partials, coalesced and deterministic.
    // See gn_stats_partials_chanmajor_kernel for the design and for the measurements that ruled
    // out ALT=1 and ALT=2 (both slower than the group-major tree AND nondeterministic).
    // ALT=0 forces the historical group-major tree, so the channel-major kernel can be A/B'd
    // against what it replaced without rebuilding.
    //
    // K = channels per thread. C <= 1024 gives K=1 (one thread per channel, as before); the
    // decoder ResBlocks, whose GroupNorm sees the concatenated 1152/1536 width, get K=2 instead of
    // falling back. Splitting C into K equal parts keeps every thread's work uniform, which needs
    // C % K == 0 -- true for every even C, and all of this UNet's are multiples of 32.
    const bool want_chanmajor = (_alt == nullptr) || (_alt[0] == '3');
    const int K = (C + 1023) / 1024;
    const int BLK = (K > 0) ? C / K : 0;
    if (want_chanmajor && K >= 1 && K <= 4 && (C % K) == 0 && BLK <= 1024
        && BLK >= num_groups && (C % num_groups) == 0) {
        const int nblocks = (int)std::min<long>(HW, 32);
        auto sopt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
        auto part_sum = torch::empty({(long)N * num_groups * nblocks}, sopt);
        auto part_sumsq = torch::empty({(long)N * num_groups * nblocks}, sopt);
        const int NG = N * num_groups;
        const long group_size = (long)(C / num_groups) * HW;
        const size_t shmem = (size_t)2 * C * sizeof(float);
        dim3 grid((unsigned)nblocks, (unsigned)N);
        cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define GN_CHANMAJOR(KK)                                                                           \
        do {                                                                                       \
            if (x.scalar_type() == torch::kFloat32) {                                              \
                gn_stats_partials_chanmajor_kernel<float, KK><<<grid, BLK, shmem, st>>>(           \
                    x.data_ptr<float>(), part_sum.data_ptr<float>(),                               \
                    part_sumsq.data_ptr<float>(), C, HW, num_groups, nblocks);                     \
            } else {                                                                               \
                gn_stats_partials_chanmajor_kernel<__half, KK><<<grid, BLK, shmem, st>>>(          \
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),                       \
                    part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),                      \
                    C, HW, num_groups, nblocks);                                                   \
            }                                                                                      \
        } while (0)
        // VEC2 FAST PATH, default ON, BIT-IDENTICAL to the scalar switch below (see the kernel's
        // header for the invariants and the gate). Set MODIFF_GN_STATS_VEC2=0 to force the scalar
        // path -- captured once per process, like _alt above, so an A/B must fork per variant.
        static const char* _vec2env = std::getenv("MODIFF_GN_STATS_VEC2");
        const bool want_vec2 = (_vec2env == nullptr) || (_vec2env[0] != '0');
        const int BLK2 = C / 2;
        if (want_vec2 && (C % 2) == 0 && BLK2 <= 1024 && BLK2 >= num_groups) {
            if (x.scalar_type() == torch::kFloat32) {
                gn_stats_partials_chanmajor_vec2_kernel<float><<<grid, BLK2, shmem, st>>>(
                    x.data_ptr<float>(), part_sum.data_ptr<float>(),
                    part_sumsq.data_ptr<float>(), C, HW, num_groups, nblocks);
            } else {
                gn_stats_partials_chanmajor_vec2_kernel<__half><<<grid, BLK2, shmem, st>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),
                    C, HW, num_groups, nblocks);
            }
        } else {
        switch (K) {
            case 1: GN_CHANMAJOR(1); break;
            case 2: GN_CHANMAJOR(2); break;
            case 3: GN_CHANMAJOR(3); break;
            default: GN_CHANMAJOR(4); break;
        }
        }
#undef GN_CHANMAJOR
        const int fb = 128, fg = (NG + fb - 1) / fb;
        gn_stats_reduce_partials_kernel<<<fg, fb, 0, st>>>(
            part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            nblocks, group_size, (float)eps, NG);
        return;
    }

    if (_alt != nullptr && _alt[0] == '2') {
        // Single-pass merged atomic variant: one grid-stride sweep over X atomicAdd'ing both sum
        // and sumsq (see gn_stats_sumsq_kernel) -- half the reads of the two-pass ALT=1 variant
        // below, since it never re-reads X to compute (x-mean)^2.
        //
        // MEASURED AND REJECTED 2026-08-04: 29.45 ms/step vs the group-major tree's 17.69 (1.7x
        // SLOWER, not the ~9.4 ms/step saving an earlier version of this comment predicted) and
        // nondeterministic -- two replays of one seed gave latents differing by 1.27e-1. The
        // atomicAdd both serializes G-way contention and destroys summation-order reproducibility.
        // Kept only as the executable record of that result; use ALT=3 / the default instead.
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
    // ONE policy, shared with the launcher that must produce identical mean/inv_std -- see
    // csrc/gn_block_size.h. The formula used to be duplicated here and there, which is how two things
    // that must agree bit-for-bit stop agreeing.
    int block_size = modiff_gn_stats_block_size(group_size);
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
    int C, int G, long sample_stride, long num_elements, bool apply_silu,
    // Symmetric code ceiling; <= 0 means the 127 this kernel used to clamp at unconditionally.
    // Only differs from 127 on a 4-bit datapath (a4), which is the only case where a delta that
    // outgrew a reused scale has anything to saturate against.
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument, which THIS kernel did until 2026-08-10) and a 4-bit layer
    // silently stayed 8-bit. A bool has no such value to get wrong; the saturation limit
    // is derived from the datapath below.
    bool a4,
    bool write_ahat,
    bool ahat_i8 = false,
    const float* ahat_qscale = nullptr, int ahat_ng = 0
) {
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. Derived, not passed, so it cannot
    // disagree with the bit-width the scale was built for.
    const float a4_lim = a4 ? 7.0f : 127.0f;
    float ahat_s, ahat_inv, ahat_lim;
    ahat_qparams(ahat_i8, ahat_qscale, ahat_s, ahat_inv, ahat_lim, ahat_ng);
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
        float q = ahat_quant_update(a_hat_cache, i, out, scale, inv_scale, a4_lim,
                                    ahat_i8, ahat_s, ahat_inv, ahat_lim, write_ahat,
                                    ahat_qscale, C, ahat_ng);
        Yq[i] = (int8_t)q;
    }
}

// Reduction-only twin of gn_apply_delta_quantize_flat_kernel: computes the absmax of the
// very same delta expression instead of quantizing it, so the caller can derive a *dynamic*
// per-call scale that provably cannot clip.
//
// Everything from `float v = gn_load(X, i)` down to the smooth multiply is copied verbatim
// from the kernel above, including the deliberate `__half2float(__float2half(normed))`
// round-trip. That is load-bearing: the scale is only guaranteed non-clipping if the
// expression reduced here is bit-identical to the expression the quantizer later evaluates.
// Any simplification (skipping the fp16 round, reordering silu and smooth) breaks that
// guarantee in exactly the cases that matter -- the tail elements that set the absmax.
//
// It reuses the mean/inv_std the caller already computed, so the cost is one extra
// elementwise read pass over X and a_hat, not another GroupNorm statistics pass.
template <typename TIn>
__global__ void gn_delta_absmax_flat_kernel(
    const TIn* __restrict__ X,
    const __half* __restrict__ a_hat_cache,  // read-only here
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,    // [N,C] or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ mean_in,    // [N*G]
    const float* __restrict__ inv_std_in, // [N*G]
    float* __restrict__ absmax_buf,       // [1], must be 0 on entry (self-resetting)
    float* __restrict__ scale_out,        // [1] out: Q_level/max(absmax,eps)
    float* __restrict__ inv_scale_out,    // [1] out: its reciprocal (CUTLASS alpha)
    unsigned int* __restrict__ retire_count, // [1], must be 0 on entry (self-resetting)
    const float* __restrict__ smooth_inv, // [C] or nullptr
    float Q_level,                        // 7.0 for INT4, 127.0 for INT8
    int C, int G, long sample_stride, long num_elements, bool apply_silu,
    bool ahat_i8 = false,
    const float* ahat_qscale = nullptr, int ahat_ng = 0
) {
    extern __shared__ float sdata[];
    const int CPG = C / G;
    float ahat_s, ahat_inv, ahat_lim;
    ahat_qparams(ahat_i8, ahat_qscale, ahat_s, ahat_inv, ahat_lim, ahat_ng);
    float local_max = 0.0f;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
         i += (long)blockDim.x * gridDim.x) {
        int c = (int)(i % C);
        long n = i / sample_stride;
        long stats_idx = n * G + (c / CPG);
        float mean = mean_in[stats_idx];
        float inv_std = inv_std_in[stats_idx];
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
        float bs, binv, blim;
        ahat_resolve(ahat_i8, ahat_qscale, i, C, ahat_ng, ahat_s, ahat_inv, ahat_lim, bs, binv, blim);
        local_max = fmaxf(local_max, fabsf(out - ahat_load(a_hat_cache, i, ahat_i8, bs)));
    }

    // Same block reduction + atomic float-max + last-block-retires election as
    // sub_absmax_scale_kernel / delta_absmax_fp16_kernel in modiff_delta_quantize.cu.
    const int tid = threadIdx.x;
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
            float am = fmaxf(*absmax_buf, 1e-6f);
            *scale_out = Q_level / am;
            *inv_scale_out = am / Q_level;
            *absmax_buf = 0.0f;
            *retire_count = 0;
        }
    }
}

// Vectorized counterpart of gn_delta_absmax_flat_kernel, pair-major exactly like
// gn_apply_delta_quantize_flat_vec2_kernel below.
//
// This is not a micro-optimization. The scalar version above issues 2-byte fp16 loads, i.e. 64 B
// per warp -- half a 128 B sector, so it wastes half of every memory transaction on a kernel that
// is purely bandwidth-bound. Measured cost of the scalar form at batch 128 / 200 steps: the
// dynamic delta scale added +8.62 ms/step (int8) and +10.84 (int4) over static, which made the
// reduction pass the single largest remaining MoDiff overhead.
//
// Requires CPG even, same condition as the vec2 quantize kernel it must agree with: a pair's
// c0/c0+1 then always share one group and hence one mean/inv_std.
template <typename TIn>
__global__ void gn_delta_absmax_flat_vec2_kernel(
    const TIn* __restrict__ X,
    const __half* __restrict__ a_hat_cache,
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ mean_in,
    const float* __restrict__ inv_std_in,
    float* __restrict__ absmax_buf,
    float* __restrict__ scale_out,
    float* __restrict__ inv_scale_out,
    unsigned int* __restrict__ retire_count,
    const float* __restrict__ smooth_inv,
    float Q_level,
    int C, int G, long sample_stride, long num_elements, bool apply_silu,
    bool ahat_i8 = false,
    const float* ahat_qscale = nullptr, int ahat_ng = 0
) {
    extern __shared__ float sdata[];
    const int CPG = C / G;
    float ahat_s, ahat_inv, ahat_lim;
    ahat_qparams(ahat_i8, ahat_qscale, ahat_s, ahat_inv, ahat_lim, ahat_ng);
    float local_max = 0.0f;
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
        // Same fp16 round-trip as the quantize kernels; see the scalar twin's comment for why
        // this must not be simplified away.
        n0 = __half2float(__float2half(n0));
        n1 = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0) : n0;
        float o1 = apply_silu ? gns_silu(n1) : n1;
        if (smooth_inv != nullptr) {
            o0 *= smooth_inv[c0];
            o1 *= smooth_inv[c0 + 1];
        }
        float bs, binv, blim;
        ahat_resolve(ahat_i8, ahat_qscale, base, C, ahat_ng, ahat_s, ahat_inv, ahat_lim, bs, binv, blim);
        float2 c = ahat_load2(a_hat_cache, base, ahat_i8, bs);
        local_max = fmaxf(local_max, fmaxf(fabsf(o0 - c.x), fabsf(o1 - c.y)));
    }

    const int tid = threadIdx.x;
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
            float am = fmaxf(*absmax_buf, 1e-6f);
            *scale_out = Q_level / am;
            *inv_scale_out = am / Q_level;
            *absmax_buf = 0.0f;
            *retire_count = 0;
        }
    }
}

// Vectorized (half2/float2) counterpart of gn_apply_delta_quantize_flat_kernel. Pair-major
// grid-stride loop, mirroring the structure gn_apply_delta_quantize_pack_flat_vec2_kernel below
// already uses. gamma/beta/mod_scale/mod_shift/a_hat_cache all read/written via one gn_load2/
// gn_store2 call per pair instead of two scalar calls; the two output int8 codes are packed
// into one int16 store. Requires CPG even (so a pair's c0/c0+1 always share one group and
// hence one mean/inv_std, exactly like the pack kernel below) -- the caller (host wrapper)
// only dispatches here when that holds, else falls back to the scalar kernel above.
// AhatB32 (implies AhatI8) selects the along-C B=32 fast path at compile time. As a
// runtime branch the dead generic a_hat code cost 10 registers -- 44 vs 34 for the fp16
// instantiation -- which dropped the block/SM limit from 7 to 5 and made this kernel
// slower than fp16 despite moving 25% fewer bytes.
template <typename TIn, bool AhatI8, bool WriteAhat, bool AhatB32 = false>
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
    int C, int G, long sample_stride, long num_elements, bool apply_silu,
    // Free absmax reporting for the NEXT step's scale (all nullptr => disabled).
    // See gn_report_delta_absmax: this removes the separate absmax pass at the cost of a
    // one-step-stale scale, which the staleness sweep showed costs nothing.
    float* __restrict__ absmax_buf, float* __restrict__ next_scale_out,
    float* __restrict__ next_inv_out, unsigned int* __restrict__ retire_count,
    float Q_level, float safety,
    // Symmetric code ceiling for THIS step's quantization; <= 0 means the 127 this kernel used
    // to clamp at unconditionally. Distinct from Q_level above, which sets the NEXT step's
    // scale: Q_level/absmax is a scale, code_ceiling is where codes saturate, and a clip ratio
    // is exactly the case where the two stop agreeing.
    // Activation bit-width of THIS datapath, not a magnitude. It replaced a
    // `float code_ceiling`, whose failure mode was a plausible-but-wrong number: pass 127
    // (or forget the argument, which THIS kernel did until 2026-08-10) and a 4-bit layer
    // silently stayed 8-bit. A bool has no such value to get wrong; the saturation limit
    // is derived from the datapath below.
    bool a4,
    const float* ahat_qscale = nullptr, int ahat_ng = 0
) {
    extern __shared__ float sdata[];
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    // Q_b for this datapath: 7 at 4 bits, 127 at 8. Derived, not passed, so it cannot
    // disagree with the bit-width the scale was built for.
    const float a4_lim = a4 ? 7.0f : 127.0f;
    float ahat_s = 1.0f, ahat_inv = 1.0f, ahat_lim = 127.0f;
    if constexpr (!AhatB32)
        ahat_qparams(AhatI8, ahat_qscale, ahat_s, ahat_inv, ahat_lim, ahat_ng);
    float local_max = 0.0f;
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
        float q0, q1, d0, d1;
        if constexpr (AhatB32) {
            if constexpr (WriteAhat)
                ahat_b32_update2(reinterpret_cast<int8_t*>(a_hat_cache),
                                 const_cast<float*>(ahat_qscale), base,
                                 o0, o1, scale, inv_scale, a4_lim, q0, q1, d0, d1);
            else
                ahat_b32_read2(reinterpret_cast<const int8_t*>(a_hat_cache), ahat_qscale, base,
                               o0, o1, scale, a4_lim, q0, q1, d0, d1);
        } else if (AhatI8 && ahat_ng > 0) {
            // Blockwise but not B=32: quantize without storing, then resnap the group.
            ahat_quant_update2_w<false>(a_hat_cache, base, o0, o1, scale, inv_scale, a4_lim,
                               AhatI8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1,
                               ahat_qscale, C, ahat_ng);
            if constexpr (WriteAhat)
                ahat_block_resnap2(a_hat_cache, const_cast<float*>(ahat_qscale), base, C, ahat_ng,
                                   (o0 - d0) + q0 * inv_scale, (o1 - d1) + q1 * inv_scale);
        } else {
            // fp16, per-tensor int8 and I-MoDiff all store inside the helper. I-MoDiff in
            // particular cannot be split into quantize-then-ahat_store2: its ahat_inv is 0
            // (scale[0]==0 is how the integer datapath is signalled), so the external store
            // wrote zeros and the cache stopped accumulating.
            ahat_quant_update2_w<WriteAhat>(a_hat_cache, base, o0, o1, scale, inv_scale, a4_lim,
                               AhatI8, ahat_s, ahat_inv, ahat_lim, q0, q1, d0, d1,
                               ahat_qscale, C, ahat_ng);
        }
        local_max = fmaxf(local_max, fmaxf(fabsf(d0), fabsf(d1)));
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        reinterpret_cast<int16_t*>(Yq)[base >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
    }
    gn_report_delta_absmax(local_max, sdata, absmax_buf, next_scale_out, next_inv_out,
                           retire_count, Q_level, safety);
}

// 4-channels-per-thread twin of the kernel above, for along-C B=32 int8 a_hat only.
// Same math and the same free-absmax epilogue; the only difference is that one thread
// owns four consecutive channels, so eight lanes cover a B=32 group instead of sixteen.
// See ahat_b32_update4 for why the fp16 a_hat path does not get the same treatment.
// Requires CPG % 4 == 0 so all four channels share one mean/inv_std -- the host falls
// back to the vec2 kernel otherwise (churches C=192 and C=576 have CPG 6 and 18).
template <typename TIn, bool WriteAhat>
__global__ void gn_apply_delta_quantize_flat_vec4_b32_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ Yq,
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ mean_in,
    const float* __restrict__ inv_std_in,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ smooth_inv,
    int C, int G, long sample_stride, long num_elements, bool apply_silu,
    float* __restrict__ absmax_buf, float* __restrict__ next_scale_out,
    float* __restrict__ next_inv_out, unsigned int* __restrict__ retire_count,
    float Q_level, float safety, bool a4,
    const float* ahat_qscale, int ahat_ng
) {
    extern __shared__ float sdata[];
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float a4_lim = a4 ? 7.0f : 127.0f;
    int8_t* cache = reinterpret_cast<int8_t*>(a_hat_cache);
    float* qscale = const_cast<float*>(ahat_qscale);
    float local_max = 0.0f;
    const long stride = (long)blockDim.x * gridDim.x;
    for (long base = 4 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < num_elements; base += 4 * stride) {
        const int c0 = (int)(base % C);    // multiple of 4; all four share one group
        const long n = base / sample_stride;
        const long stats_idx = n * G + (c0 / CPG);
        const float mean = mean_in[stats_idx];
        const float inv_std = inv_std_in[stats_idx];

        const float2 v0 = gn_load2(X, base), v1 = gn_load2(X, base + 2);
        const float2 w0 = gn_load2(gamma, c0), w1 = gn_load2(gamma, c0 + 2);
        const float2 b0 = gn_load2(beta, c0), b1 = gn_load2(beta, c0 + 2);
        float o[4];
        o[0] = (v0.x - mean) * inv_std * w0.x + b0.x;
        o[1] = (v0.y - mean) * inv_std * w0.y + b0.y;
        o[2] = (v1.x - mean) * inv_std * w1.x + b1.x;
        o[3] = (v1.y - mean) * inv_std * w1.y + b1.y;
        if (mod_scale != nullptr) {
            const long midx = n * C + c0;
            const float2 ms0 = gn_load2(mod_scale, midx), ms1 = gn_load2(mod_scale, midx + 2);
            const float2 sh0 = gn_load2(mod_shift, midx), sh1 = gn_load2(mod_shift, midx + 2);
            o[0] = o[0] * (1.0f + ms0.x) + sh0.x;
            o[1] = o[1] * (1.0f + ms0.y) + sh0.y;
            o[2] = o[2] * (1.0f + ms1.x) + sh1.x;
            o[3] = o[3] * (1.0f + ms1.y) + sh1.y;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float h = __half2float(__float2half(o[k]));
            o[k] = apply_silu ? gns_silu(h) : h;
        }
        if (smooth_inv != nullptr) {
            const float4 sm = *reinterpret_cast<const float4*>(smooth_inv + c0);
            o[0] *= sm.x; o[1] *= sm.y; o[2] *= sm.z; o[3] *= sm.w;
        }
        float q[4], d[4];
        if constexpr (WriteAhat)
            ahat_b32_update4(cache, qscale, base, o, scale, inv_scale, a4_lim, q, d);
        else
            ahat_b32_read4(cache, ahat_qscale, base, o, scale, a4_lim, q, d);
#pragma unroll
        for (int k = 0; k < 4; ++k) local_max = fmaxf(local_max, fabsf(d[k]));
        *reinterpret_cast<unsigned*>(Yq + base) = __byte_perm(
            __byte_perm(ahat_f_to_byte(q[0]), ahat_f_to_byte(q[1]), 0x4040u),
            __byte_perm(ahat_f_to_byte(q[2]), ahat_f_to_byte(q[3]), 0x4040u), 0x5410u);
    }
    gn_report_delta_absmax(local_max, sdata, absmax_buf, next_scale_out, next_inv_out,
                           retire_count, Q_level, safety);
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
    torch::Tensor mod_shift,
    // --- optional dynamic-scale mode (all four empty => static, the original behaviour) ---
    // When supplied, the per-call delta scale is discovered on device between the statistics
    // pass and the quantize pass (gn_delta_absmax_flat_kernel) and `scale` is ignored. The
    // extra cost is one elementwise read pass; the benefit is a scale that cannot clip.
    // Measured 2026-08-04 on the real LSUN-churches checkpoint, the static setting clips on
    // 49 of 70 conv layers.
    torch::Tensor absmax_buf,
    torch::Tensor scale_out,
    torch::Tensor inv_scale_out,
    torch::Tensor retire_count,
    double Q_level,
    // report_next: skip the separate absmax pass and instead have the QUANTIZE kernel record the
    // delta range and publish the next step's scale for free (gn_report_delta_absmax). `scale` then
    // has to be the scale a previous step published. safety gives headroom for the range growing.
    bool report_next,
    double safety,
    // Symmetric code ceiling, i.e. Q_b for a b-bit activation datapath. <= 0 (the default) keeps the
    // 127 these kernels clamped at unconditionally, so every pre-existing caller is bit-identical.
    // Only matters when the scale is deliberately larger than Q_b/absmax -- which is what a clip
    // ratio is, and why the ratio was not a clip below A8 before this existed
    // (docs/delta_clip_2026-08-06/FINDINGS.md).
    bool a4,
    bool write_ahat,
    torch::Tensor ahat_scale
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_delta_quantize_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_delta_quantize_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_nhwc: only float32 and float16 are supported");
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
    bool ahat_i8 = false;
    const float* ahat_qscale_ptr = nullptr;
    int ahat_ng = 0;
    __half* cache_ptr = nullptr;
    bind_ahat_cache(a_hat_cache, ahat_scale, cache_ptr, ahat_i8, ahat_qscale_ptr,
                    "group_norm_silu_delta_quantize_nhwc", &ahat_ng);
    const long num_elements = (long)N * C * HW;
    const long sample_stride = (long)C * HW;
    const int ablock = 256;
    const unsigned int agrid_scalar = (unsigned int)((num_elements + ablock - 1) / ablock);
    const unsigned int agrid_vec2 = (unsigned int)((num_elements / 2 + ablock - 1) / ablock);
    const unsigned int agrid_vec4 = (unsigned int)((num_elements / 4 + ablock - 1) / ablock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dynamic mode: discover the scale from this call's own delta, between the statistics pass
    // above and the quantize pass below. Reuses mean/inv_std, so no second GN reduction.
    const bool have_bufs = absmax_buf.numel() > 0;
    const bool dynamic = have_bufs && !report_next;   // separate absmax pass
    const bool report = have_bufs && report_next;     // free reporting from the quantize kernel
    const float* scale_ptr_eff = scale.data_ptr<float>();
    if (have_bufs) {
        TORCH_CHECK(scale_out.numel() > 0 && inv_scale_out.numel() > 0 && retire_count.numel() > 0,
                    "group_norm_silu_delta_quantize_nhwc: dynamic mode needs absmax_buf, "
                    "scale_out, inv_scale_out and retire_count together");
    }
    if (dynamic) {
        // Grid is capped so the retirement election stays cheap; the grid-stride loop covers
        // the tensor regardless of grid size.
        // Pair-major grid when CPG is even (always, for real configs) -- the scalar fallback
        // wastes half of every 128 B sector on fp16 input. Grid is halved to match the 2-wide step.
        const bool rvec2 = (CPG % 2 == 0);
        const long rwork = rvec2 ? (num_elements / 2) : num_elements;
        const unsigned int rgrid = (unsigned int)std::max<long>(
            1, std::min<long>(1024, (rwork + ablock - 1) / ablock));
        if (x.scalar_type() == torch::kFloat32) {
            if (rvec2)
            gn_delta_absmax_flat_vec2_kernel<float><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                x.data_ptr<float>(), cache_ptr,
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
            else
            gn_delta_absmax_flat_kernel<float><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                x.data_ptr<float>(), cache_ptr,
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
        } else {
            if (rvec2)
            gn_delta_absmax_flat_vec2_kernel<__half><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
            else
            gn_delta_absmax_flat_kernel<__half><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
        }
        scale_ptr_eff = scale_out.data_ptr<float>();
    }

    const bool fuse_block = use_vec2 && ahat_block_shuffle_ok(C, ahat_ng);
    const bool wa = write_ahat && !(ahat_i8 && ahat_ng > 0 && !fuse_block);
    const bool blk32 = use_vec2 && ahat_i8 && ahat_is_b32(C, ahat_ng);
    // vec4 needs all four channels of a thread in one GN group and the a_hat/Yq words
    // 4-byte aligned; C is a multiple of 32 whenever blk32 holds, so only CPG matters.
    const bool blk32_vec4 = blk32 && (CPG % 4 == 0);
    if (x.scalar_type() == torch::kFloat32) {
        if (use_vec2) {
            // Shared memory sized for the free-absmax reduction (gn_report_delta_absmax). Always
            // allocated: it is one float per thread, and the kernel's extern __shared__ must be
            // backed even on the non-reporting path where the helper returns immediately.
            auto launch_f = [&](auto a8tag, auto wtag, auto btag) {
                constexpr bool A8 = decltype(a8tag)::value;
                constexpr bool WA = decltype(wtag)::value;
                constexpr bool B32 = decltype(btag)::value;
                gn_apply_delta_quantize_flat_vec2_kernel<float, A8, WA, B32>
                    <<<agrid_vec2, ablock, ablock * sizeof(float), stream>>>(
                    x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    weight.data_ptr<float>(), bias.data_ptr<float>(),
                    has_mod ? mod_scale.data_ptr<float>() : nullptr,
                    has_mod ? mod_shift.data_ptr<float>() : nullptr,
                    mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                    scale_ptr_eff, smooth_ptr,
                    C, (int)num_groups, sample_stride, num_elements, apply_silu,
                    report ? absmax_buf.data_ptr<float>() : nullptr,
                    report ? scale_out.data_ptr<float>() : nullptr,
                    report ? inv_scale_out.data_ptr<float>() : nullptr,
                    report ? (unsigned int*)retire_count.data_ptr<int>() : nullptr,
                    (float)Q_level, (float)safety, a4, ahat_qscale_ptr, ahat_ng);
            };
            auto launch_f4 = [&](auto wtag) {
                constexpr bool WA = decltype(wtag)::value;
                gn_apply_delta_quantize_flat_vec4_b32_kernel<float, WA>
                    <<<agrid_vec4, ablock, ablock * sizeof(float), stream>>>(
                    x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    weight.data_ptr<float>(), bias.data_ptr<float>(),
                    has_mod ? mod_scale.data_ptr<float>() : nullptr,
                    has_mod ? mod_shift.data_ptr<float>() : nullptr,
                    mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                    scale_ptr_eff, smooth_ptr,
                    C, (int)num_groups, sample_stride, num_elements, apply_silu,
                    report ? absmax_buf.data_ptr<float>() : nullptr,
                    report ? scale_out.data_ptr<float>() : nullptr,
                    report ? inv_scale_out.data_ptr<float>() : nullptr,
                    report ? (unsigned int*)retire_count.data_ptr<int>() : nullptr,
                    (float)Q_level, (float)safety, a4, ahat_qscale_ptr, ahat_ng);
            };
            if (blk32_vec4) {
                if (wa) launch_f4(std::true_type{});
                else    launch_f4(std::false_type{});
            } else if (wa) {
                if (blk32)        launch_f(std::true_type{}, std::true_type{}, std::true_type{});
                else if (ahat_i8) launch_f(std::true_type{}, std::true_type{}, std::false_type{});
                else              launch_f(std::false_type{}, std::true_type{}, std::false_type{});
            } else {
                if (blk32)        launch_f(std::true_type{}, std::false_type{}, std::true_type{});
                else if (ahat_i8) launch_f(std::true_type{}, std::false_type{}, std::false_type{});
                else              launch_f(std::false_type{}, std::false_type{}, std::false_type{});
            }
        } else {
            gn_apply_delta_quantize_flat_kernel<float><<<agrid_scalar, ablock, 0, stream>>>(
                x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                scale_ptr_eff, smooth_ptr,
                C, (int)num_groups, sample_stride, num_elements, apply_silu, a4,
                write_ahat && !(ahat_i8 && ahat_ng > 0),
                ahat_i8, ahat_qscale_ptr, ahat_ng);
        }
    } else {
        if (use_vec2) {
            auto launch_h = [&](auto a8tag, auto wtag, auto btag) {
                constexpr bool A8 = decltype(a8tag)::value;
                constexpr bool WA = decltype(wtag)::value;
                constexpr bool B32 = decltype(btag)::value;
                gn_apply_delta_quantize_flat_vec2_kernel<__half, A8, WA, B32>
                    <<<agrid_vec2, ablock, ablock * sizeof(float), stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                    reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                    has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                    has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                    mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                    scale_ptr_eff, smooth_ptr,
                    C, (int)num_groups, sample_stride, num_elements, apply_silu,
                    report ? absmax_buf.data_ptr<float>() : nullptr,
                    report ? scale_out.data_ptr<float>() : nullptr,
                    report ? inv_scale_out.data_ptr<float>() : nullptr,
                    report ? (unsigned int*)retire_count.data_ptr<int>() : nullptr,
                    (float)Q_level, (float)safety, a4, ahat_qscale_ptr, ahat_ng);
            };
            auto launch_h4 = [&](auto wtag) {
                constexpr bool WA = decltype(wtag)::value;
                gn_apply_delta_quantize_flat_vec4_b32_kernel<__half, WA>
                    <<<agrid_vec4, ablock, ablock * sizeof(float), stream>>>(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                    reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                    reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                    has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                    has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                    mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                    scale_ptr_eff, smooth_ptr,
                    C, (int)num_groups, sample_stride, num_elements, apply_silu,
                    report ? absmax_buf.data_ptr<float>() : nullptr,
                    report ? scale_out.data_ptr<float>() : nullptr,
                    report ? inv_scale_out.data_ptr<float>() : nullptr,
                    report ? (unsigned int*)retire_count.data_ptr<int>() : nullptr,
                    (float)Q_level, (float)safety, a4, ahat_qscale_ptr, ahat_ng);
            };
            if (blk32_vec4) {
                if (wa) launch_h4(std::true_type{});
                else    launch_h4(std::false_type{});
            } else if (wa) {
                if (blk32)        launch_h(std::true_type{}, std::true_type{}, std::true_type{});
                else if (ahat_i8) launch_h(std::true_type{}, std::true_type{}, std::false_type{});
                else              launch_h(std::false_type{}, std::true_type{}, std::false_type{});
            } else {
                if (blk32)        launch_h(std::true_type{}, std::false_type{}, std::true_type{});
                else if (ahat_i8) launch_h(std::true_type{}, std::false_type{}, std::false_type{});
                else              launch_h(std::false_type{}, std::false_type{}, std::false_type{});
            }
        } else {
            gn_apply_delta_quantize_flat_kernel<__half><<<agrid_scalar, ablock, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                scale_ptr_eff, smooth_ptr,
                C, (int)num_groups, sample_stride, num_elements, apply_silu, a4,
                write_ahat && !(ahat_i8 && ahat_ng > 0),
                ahat_i8, ahat_qscale_ptr, ahat_ng);
        }
    }
    if (write_ahat && ahat_i8 && ahat_ng > 0 && !fuse_block) {
        ahat_commit_block(cache_ptr, const_cast<float*>(ahat_qscale_ptr),
                          yq.data_ptr<int8_t>(), scale_ptr_eff, C, ahat_ng,
                          num_elements, stream);
    }
    return yq;
}

// ---------------------------------------------------------------------------------------------
// SINGLE-KERNEL (group-major) MoDiff delta-quantize: the structural counterpart of the BASELINE's
// group_norm_silu_quantize_nhwc_vec2_kernel, which does its statistics reduction and its apply pass
// in ONE kernel with mean/inv_std never leaving shared memory.
//
// A FAILED EXPERIMENT, KEPT AS EXECUTABLE EVIDENCE (see the dead-code policy in
// csrc/modiff_kernels_api.h: unreferenced code stays only when the reason it is unused is itself a
// finding worth not rediscovering). THE MERGE IS A REGRESSION -- do not wire it into the ResBlock
// path without re-reading this comment.
//
// The premise. The two-kernel MoDiff path above (gn_launch_group_stats writing mean/inv_std to
// GLOBAL memory, then gn_apply_delta_quantize_flat_* reading them back) looked like it was paying
// two things the baseline's one-kernel GN does not: a second launch, and a global round-trip for the
// statistics. The split had also never itself been A/B'd -- gn_stats_ab.py compared four *stats
// algorithms* with the split held fixed, and 2026-08-16's fast_reduce swap explicitly "does not
// touch" this path.
//
// THE ARITHMETIC THAT SHOULD HAVE BEEN DONE FIRST. Both paths move the SAME 9 bytes/element (x read
// twice + a_hat read + a_hat write + int8 out); mean/inv_std is only N*G floats (16 KB against
// 226 MB), so the "round-trip" was never real traffic. The entire available saving was ONE kernel
// launch -- ~5 us against a 150-430 us kernel, i.e. 1-3% -- plus whatever L2 reuse of x a single
// block might get between its two passes.
//
// WHAT IT COST INSTEAD, and the mechanism, from a CONTROLLED sweep (b128, A40, production
// modulation, independent-layers chained). Every confound is pinned: N=128, G=32 and C*H*W=196608
// throughout, so total elements, total nominal bytes (9 B/elem) and grid size (N*G = 4096 blocks)
// are IDENTICAL on every row. Only C and HW trade off, which varies CPG and hence the contiguous
// run length -- the one thing under test:
//
//   CPG  run bytes   two-kernel        this kernel       merged/two
//     6      12 B    0.4305 ms (76%)   0.9993 ms (33%)     2.321x
//    12      24 B    0.4328 ms (75%)   0.7675 ms (42%)     1.773x
//    24      48 B    0.4545 ms (72%)   0.6107 ms (53%)     1.344x
//    48      96 B    0.4727 ms (69%)   0.4921 ms (66%)     1.041x
//    96     192 B    0.8127 ms (40%)   0.3982 ms (82%)     0.490x   <- merged wins 2x here
//
// Monotone in the run length, and nothing else moved: the mechanism IS the access pattern.
// One-block-per-group forces GROUP-MAJOR access, and in NHWC a group is CPG contiguous channels
// inside one pixel (CPG*2 bytes) and then a jump of C*2. `cpair` varies fastest here, so a 32-thread
// warp spans ceil(64/CPG) pixels -- ~11 of them at CPG=6, each contributing 12 useful bytes out of a
// 32-byte sector. 12/32 = 37.5% predicted utilisation against 33% measured. The flat apply kernel it
// replaces has no runs at all: consecutive threads take consecutive addresses, so a warp covers 128
// contiguous bytes = 4 full sectors.
//
// The strided a_hat WRITE is the expensive half: a partial-sector store is a read-modify-write in
// L2, so 12 useful bytes cost a fetch plus a store. That is why the baseline's group-major single
// kernel tolerates the identical pattern at 60-74% -- it carries 5 B/elem, x re-reads that hit L2,
// and a 1 B/elem int8 store, but NO strided fp16 state write. Adding a_hat's read+write under
// group-major access is what collapses.
//
// SO THE DESIGN IS NOT WRONG, IT IS MISMATCHED. Past CPG~48 it wins, and at CPG=96 it reaches 82% of
// peak -- the best efficiency of any kernel in this family -- while the two-kernel path degrades
// there (its channel-major stats kernel wants 2*C*sizeof(float) = 24 KB of shared memory at C=3072).
// A net with wide channels or few groups should re-measure. THIS UNet runs G=32 throughout and its
// hot shapes are CPG=6 and 12 (7 calls/step each), which is the worst case for this kernel.
//
// VERDICT: the existing two-kernel split is correct FOR THIS NET, and now measured rather than
// inherited. What was never true is the premise -- both paths move the same 9 B/elem, so there was
// only ever ~1-3% (one launch) to win, against a 2.3x coalescing penalty to lose.
//
// WHAT IS AND IS NOT PRESERVED. The delta-quantize arithmetic is copied verbatim from
// gn_apply_delta_quantize_flat_vec2_kernel, including the `__half2float(__float2half(n))` rounding
// of `normed` before SiLU that keeps this path matching the original standalone-GN + step1 pair. The
// REDUCTION ORDER changes (channel-major partials -> pair-major warp-shuffle within one block),
// exactly as the baseline's fast_reduce swap changed it -- so mean/inv_std move in the last fp32
// bits and a value sitting on a code boundary can land either side. Same class and same magnitude
// as docs/gn_fast_reduce_2026-08-16 measured for the baseline (<=1 code on ~1e-5% of elements);
// measured for this kernel in the same doc. NOT bit-identical, and the caller keeps the two-kernel
// path available so that is falsifiable rather than asserted.
//
// SCOPE: static delta scale only. The dynamic/report modes publish a scale via a separate absmax
// pass (or gn_report_delta_absmax's cross-block retirement election), neither of which composes with
// a one-block-per-group launch; those callers keep the two-kernel path.
template <typename TIn>
__global__ void gn_delta_quantize_fused_groupmajor_vec2_kernel(
    const TIn* __restrict__ X,
    __half* __restrict__ a_hat_cache,     // [N,H,W,C] fp16 channels_last, in place
    int8_t* __restrict__ Yq,              // [N,H,W,C] int8 quantized delta
    const TIn* __restrict__ gamma,
    const TIn* __restrict__ beta,
    const TIn* __restrict__ mod_scale,    // [N,C] or nullptr
    const TIn* __restrict__ mod_shift,
    const float* __restrict__ scale_ptr,  // scalar quant multiplier = Q_b/absmax
    const float* __restrict__ smooth_inv, // [C] or nullptr
    int C, long HW, int G, float eps, bool apply_silu,
    // Activation bit-width of THIS datapath, not a magnitude -- same contract as the flat kernel.
    bool a4,
    bool write_ahat
) {
    const int CPG = C / G;
    const long group_size = (long)CPG * HW;
    const int HALF_CPG = CPG / 2;
    const long pairs = group_size / 2;

    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    const int c_start = g * CPG;

    // Per-sample bases: n*HW*C is even (C is a multiple of the group count and of 2 here), so every
    // gn_load2/gn_store2 below stays naturally aligned -- the same argument gn_load2's comment makes.
    const TIn* x_base = X + (long)n * HW * C;
    int8_t* yq_base = Yq + (long)n * HW * C;
    __half* cache_base = a_hat_cache + (long)n * HW * C;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // ---- Pass 1: pair-major reduction, identical partition to the baseline's fast_reduce path ----
    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
        const int cpair = pidx % HALF_CPG;
        const long hw = pidx / HALF_CPG;
        const long mem_idx0 = hw * C + c_start + 2 * cpair;
        const float2 v = gn_load2(x_base, mem_idx0);
        local_sum += v.x + v.y;
        local_sumsq += v.x * v.x + v.y * v.y;
    }

    __shared__ float mean_s, inv_std_s;
    {
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
    }
    __syncthreads();
    const float mean = mean_s;
    const float inv_std = inv_std_s;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float a4_lim = a4 ? 7.0f : 127.0f;

    // ---- Pass 2: apply + delta-quantize + in-place a_hat update (math copied from the flat kernel)
    for (long pidx = threadIdx.x; pidx < pairs; pidx += blockDim.x) {
        const int cpair = pidx % HALF_CPG;
        const long hw = pidx / HALF_CPG;
        const int c_global0 = c_start + 2 * cpair;
        const long mem_idx0 = hw * (long)C + c_global0;

        float2 v = gn_load2(x_base, mem_idx0);
        float2 w = gn_load2(gamma, c_global0);
        float2 b = gn_load2(beta, c_global0);
        float n0 = (v.x - mean) * inv_std * w.x + b.x;
        float n1 = (v.y - mean) * inv_std * w.y + b.y;
        if (mod_scale != nullptr) {
            const long midx0 = (long)n * C + c_global0;
            float2 ms = gn_load2(mod_scale, midx0);
            float2 sh = gn_load2(mod_shift, midx0);
            n0 = n0 * (1.0f + ms.x) + sh.x;
            n1 = n1 * (1.0f + ms.y) + sh.y;
        }
        // Same fp16 round-trip of `normed` the flat kernel performs, so SiLU sees the value the
        // original standalone-GN path would have materialized.
        const float n0h = __half2float(__float2half(n0));
        const float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        if (smooth_inv != nullptr) { o0 *= smooth_inv[c_global0]; o1 *= smooth_inv[c_global0 + 1]; }
        float2 cache = gn_load2(cache_base, mem_idx0);
        const float d0 = o0 - cache.x, d1 = o1 - cache.y;
        const float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        const float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        if (write_ahat)
            gn_store2(cache_base, mem_idx0,
                      make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
        const int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        reinterpret_cast<int16_t*>(yq_base)[mem_idx0 >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
    }
}

// Host wrapper for the single-kernel (group-major) MoDiff delta-quantize above. Same signature and
// same return contract as group_norm_silu_delta_quantize_nhwc's static path, minus the six
// dynamic-scale arguments it cannot serve (see the kernel's SCOPE note). Eligibility is checked
// here and reported by THROWING rather than by silently falling back, so a caller that thinks it is
// on the fused path always is -- the Python side probes eligibility itself.
torch::Tensor group_norm_silu_delta_quantize_nhwc_fused(
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
    torch::Tensor mod_shift,
    bool a4,
    bool write_ahat = true
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_delta_quantize_nhwc_fused expects a 4D [N,C,H,W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_delta_quantize_nhwc_fused: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_nhwc_fused: only float32 and float16 are supported");
    TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_nhwc_fused: a_hat_cache must be fp16");
    TORCH_CHECK(a_hat_cache.sizes() == x.sizes(),
                "group_norm_silu_delta_quantize_nhwc_fused: a_hat_cache must match x shape");
    const bool has_mod = mod_scale.numel() > 0;
    TORCH_CHECK(!has_mod || (mod_scale.scalar_type() == x.scalar_type() &&
                             mod_shift.scalar_type() == x.scalar_type()),
                "group_norm_silu_delta_quantize_nhwc_fused: mod dtype must match input dtype");

    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C % num_groups == 0,
                "group_norm_silu_delta_quantize_nhwc_fused: C must be divisible by num_groups");
    const long HW = (long)H * W;
    const int CPG = C / (int)num_groups;
    // vec2 (and hence this kernel) needs a channel pair to stay inside one group, so both channels
    // share one mean/inv_std. The two-kernel path has a scalar fallback for odd CPG; this one does
    // not, and says so instead of quietly computing the wrong normalization.
    TORCH_CHECK(CPG % 2 == 0,
                "group_norm_silu_delta_quantize_nhwc_fused: channels-per-group must be even");
    const long group_size = (long)CPG * HW;

    auto yq = torch::empty_like(x, x.options().dtype(torch::kInt8));

    // Same block-size heuristic the baseline's fast_reduce path uses (~six pairs/thread on A40).
    int block_size = 128;
    while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;

    dim3 grid((unsigned int)(N * (int)num_groups));
    dim3 block((unsigned int)block_size);
    const size_t shmem_bytes = 2 * (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    __half* cache_ptr = reinterpret_cast<__half*>(a_hat_cache.data_ptr<at::Half>());

    if (x.scalar_type() == torch::kFloat32) {
        gn_delta_quantize_fused_groupmajor_vec2_kernel<float><<<grid, block, shmem_bytes, stream>>>(
            x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, a4, write_ahat);
    } else {
        gn_delta_quantize_fused_groupmajor_vec2_kernel<__half><<<grid, block, shmem_bytes, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
            reinterpret_cast<int8_t*>(yq.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            scale.data_ptr<float>(), smooth_ptr,
            C, HW, (int)num_groups, (float)eps, apply_silu, a4, write_ahat);
    }
    C10_CUDA_CHECK(cudaGetLastError());
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
    int C, int G, long sample_stride, long num_elements, bool apply_silu,
    // Free absmax reporting, INT4 twin of the int8 sibling. See gn_report_delta_absmax.
    float* __restrict__ absmax_buf, float* __restrict__ next_scale_out,
    float* __restrict__ next_inv_out, unsigned int* __restrict__ retire_count,
    float Q_level, float safety,
    bool write_ahat,
    bool ahat_i8 = false,
    const float* ahat_qscale = nullptr, int ahat_ng = 0
) {
    extern __shared__ float sdata[];
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    float ahat_s, ahat_inv, ahat_lim;
    ahat_qparams(ahat_i8, ahat_qscale, ahat_s, ahat_inv, ahat_lim, ahat_ng);
    float local_max = 0.0f;
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
        float bs, binv, blim;
        ahat_resolve(ahat_i8, ahat_qscale, base, C, ahat_ng, ahat_s, ahat_inv, ahat_lim, bs, binv, blim);
        float2 cache = ahat_load2(a_hat_cache, base, ahat_i8, bs);
        const float d0 = o0 - cache.x, d1 = o1 - cache.y;
        // Reduced BEFORE the clamp, so the report is the true range and not a clipped lower bound.
        local_max = fmaxf(local_max, fmaxf(fabsf(d0), fabsf(d1)));
        float q0 = fmaxf(-7.0f, fminf(7.0f, roundf(d0 * scale)));
        float q1 = fmaxf(-7.0f, fminf(7.0f, roundf(d1 * scale)));
        if (write_ahat)
            ahat_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale),
                        ahat_i8, binv, blim);
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        Yqp[base / 2] = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
    }
    gn_report_delta_absmax(local_max, sdata, absmax_buf, next_scale_out, next_inv_out,
                           retire_count, Q_level, safety);
}

// Host wrapper: MoDiff GN(+mod)+SiLU + int4 delta-quantize+pack + a_hat update.
// Implementation. Split out on 2026-08-13 so the decoder skip-concat fold can supply the
// GroupNorm statistics it already computed instead of this function recomputing them --
// which IS the saving. The only behavioural change inside this body is at the stats call
// site above; every use of `x` is untouched, because the fold hands over the materialized
// concatenation and `x` is that tensor.
static torch::Tensor gn_delta_pack_impl(
    torch::Tensor mean_in,          // empty => compute the stats here, exactly as before
    torch::Tensor inv_std_in,
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
    torch::Tensor mod_shift,
    // Optional dynamic-scale mode, identical contract to the INT8 sibling above (all four
    // empty => static). Q_level is 7.0 here. gn_delta_absmax_flat_kernel is reused verbatim:
    // it reduces the pre-quantization delta and so is indifferent to int4 packing.
    torch::Tensor absmax_buf,
    torch::Tensor scale_out,
    torch::Tensor inv_scale_out,
    torch::Tensor retire_count,
    double Q_level,
    bool report_next,        // see the INT8 sibling
    double safety,
    bool write_ahat = true,
    torch::Tensor ahat_scale = {}
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "group_norm_silu_delta_quantize_pack_nhwc expects a 4D [N, C, H, W] tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
                "group_norm_silu_delta_quantize_pack_nhwc: weight/bias dtype must match input dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "group_norm_silu_delta_quantize_pack_nhwc: only float32 and float16 are supported");
    TORCH_CHECK(a_hat_cache.scalar_type() == torch::kFloat16
                    || a_hat_cache.scalar_type() == torch::kInt8,
                "group_norm_silu_delta_quantize_pack_nhwc: a_hat_cache must be fp16 or int8");
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
    torch::Tensor mean, inv_std;
    if (mean_in.defined() && mean_in.numel() > 0) {
        // Supplied by cat2_gn_stats_fp16, which computed them on THIS tensor while writing it.
        // Checked rather than trusted: stats for the wrong shape would not crash, they would
        // silently normalise with someone else's mean.
        TORCH_CHECK(inv_std_in.defined() && inv_std_in.numel() == mean_in.numel(),
                    "gn_delta_pack_impl: inv_std_in must match mean_in");
        TORCH_CHECK(mean_in.numel() == (long)N * num_groups,
                    "gn_delta_pack_impl: precomputed stats have ", mean_in.numel(),
                    " entries, expected N*G = ", (long)N * num_groups);
        TORCH_CHECK(mean_in.scalar_type() == torch::kFloat32
                    && inv_std_in.scalar_type() == torch::kFloat32,
                    "gn_delta_pack_impl: precomputed stats must be float32");
        mean = mean_in;
        inv_std = inv_std_in;
    } else {
        mean = torch::empty({N * (int)num_groups}, stats_opts);
        inv_std = torch::empty({N * (int)num_groups}, stats_opts);
        gn_launch_group_stats(x, N, C, HW, (int)num_groups, eps, mean, inv_std);
    }

    const float* smooth_ptr = (smooth_inv.numel() > 0) ? smooth_inv.data_ptr<float>() : nullptr;
    __half* cache_ptr = nullptr;
    bool ahat_i8 = false;
    const float* ahat_qscale_ptr = nullptr;
    int ahat_ng = 0;
    bind_ahat_cache(a_hat_cache, ahat_scale, cache_ptr, ahat_i8, ahat_qscale_ptr,
                    "group_norm_silu_delta_quantize_pack_nhwc", &ahat_ng);
    const long num_elements = (long)N * C * HW;
    const long sample_stride = (long)C * HW;
    const int ablock = 256;
    const long num_pairs = num_elements / 2;
    const unsigned int agrid = (unsigned int)((num_pairs + ablock - 1) / ablock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const bool have_bufs = absmax_buf.numel() > 0;
    const bool dynamic = have_bufs && !report_next;
    const bool report = have_bufs && report_next;
    const float* scale_ptr_eff = scale.data_ptr<float>();
    if (have_bufs) {
        TORCH_CHECK(scale_out.numel() > 0 && inv_scale_out.numel() > 0 && retire_count.numel() > 0,
                    "group_norm_silu_delta_quantize_pack_nhwc: dynamic mode needs absmax_buf, "
                    "scale_out, inv_scale_out and retire_count together");
    }
    if (dynamic) {
        // Pair-major grid when CPG is even (always, for real configs) -- the scalar fallback
        // wastes half of every 128 B sector on fp16 input. Grid is halved to match the 2-wide step.
        const bool rvec2 = (CPG % 2 == 0);
        const long rwork = rvec2 ? (num_elements / 2) : num_elements;
        const unsigned int rgrid = (unsigned int)std::max<long>(
            1, std::min<long>(1024, (rwork + ablock - 1) / ablock));
        if (x.scalar_type() == torch::kFloat32) {
            if (rvec2)
            gn_delta_absmax_flat_vec2_kernel<float><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                x.data_ptr<float>(), cache_ptr,
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
            else
            gn_delta_absmax_flat_kernel<float><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                x.data_ptr<float>(), cache_ptr,
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                has_mod ? mod_scale.data_ptr<float>() : nullptr,
                has_mod ? mod_shift.data_ptr<float>() : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
        } else {
            if (rvec2)
            gn_delta_absmax_flat_vec2_kernel<__half><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
            else
            gn_delta_absmax_flat_kernel<__half><<<rgrid, ablock, ablock * sizeof(float), stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
                has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
                mean.data_ptr<float>(), inv_std.data_ptr<float>(),
                absmax_buf.data_ptr<float>(), scale_out.data_ptr<float>(),
                inv_scale_out.data_ptr<float>(),
                (unsigned int*)retire_count.data_ptr<int>(),
                smooth_ptr, (float)Q_level,
                C, (int)num_groups, sample_stride, num_elements, apply_silu,
                ahat_i8, ahat_qscale_ptr, ahat_ng);
        }
        scale_ptr_eff = scale_out.data_ptr<float>();
    }

    // C%2==0 && CPG%2==0 already TORCH_CHECK'd above -> always safe to use the
    // vectorized kernel here, no scalar fallback needed.
    if (x.scalar_type() == torch::kFloat32) {
        gn_apply_delta_quantize_pack_flat_vec2_kernel<float><<<agrid, ablock, ablock * sizeof(float), stream>>>(
            x.data_ptr<float>(), cache_ptr, reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            has_mod ? mod_scale.data_ptr<float>() : nullptr,
            has_mod ? mod_shift.data_ptr<float>() : nullptr,
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            scale_ptr_eff, smooth_ptr,
            C, (int)num_groups, sample_stride, num_elements, apply_silu,
            report ? absmax_buf.data_ptr<float>() : nullptr,
            report ? scale_out.data_ptr<float>() : nullptr,
            report ? inv_scale_out.data_ptr<float>() : nullptr,
            report ? (unsigned int*)retire_count.data_ptr<int>() : nullptr,
            (float)Q_level, (float)safety, write_ahat, ahat_i8, ahat_qscale_ptr, ahat_ng);
    } else {
        gn_apply_delta_quantize_pack_flat_vec2_kernel<__half><<<agrid, ablock, ablock * sizeof(float), stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), cache_ptr,
            reinterpret_cast<int8_t*>(yqp.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            has_mod ? reinterpret_cast<const __half*>(mod_scale.data_ptr<at::Half>()) : nullptr,
            has_mod ? reinterpret_cast<const __half*>(mod_shift.data_ptr<at::Half>()) : nullptr,
            mean.data_ptr<float>(), inv_std.data_ptr<float>(),
            scale_ptr_eff, smooth_ptr,
            C, (int)num_groups, sample_stride, num_elements, apply_silu,
            report ? absmax_buf.data_ptr<float>() : nullptr,
            report ? scale_out.data_ptr<float>() : nullptr,
            report ? inv_scale_out.data_ptr<float>() : nullptr,
            report ? (unsigned int*)retire_count.data_ptr<int>() : nullptr,
            (float)Q_level, (float)safety, write_ahat, ahat_i8, ahat_qscale_ptr, ahat_ng);
    }
    return yqp;
}

// Unchanged public entry: computes its own statistics, exactly as before this refactor.
torch::Tensor group_norm_silu_delta_quantize_pack_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor a_hat_cache,
    int64_t num_groups, double eps, bool apply_silu, torch::Tensor scale,
    torch::Tensor smooth_inv, torch::Tensor mod_scale, torch::Tensor mod_shift,
    torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
    torch::Tensor retire_count, double Q_level, bool report_next, double safety,
    bool write_ahat, torch::Tensor ahat_scale
) {
    return gn_delta_pack_impl(torch::Tensor(), torch::Tensor(), x, weight, bias, a_hat_cache, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, absmax_buf, scale_out, inv_scale_out, retire_count,
        Q_level, report_next, safety, write_ahat, ahat_scale);
}

// THE FOLD, wired: takes the two halves the decoder would have concatenated, produces the
// concatenation AND the GroupNorm statistics in one pass (cat2_gn_stats_fp16), then runs the ordinary
// delta-quantize apply over that concatenation with those statistics handed to it. So the tensor is
// read twice in total instead of three times.
//
// Returns {packed, cat}. The caller needs `cat` because the ResBlock consumes it twice more -- the
// 1x1 skip conv and the out-conv's residual -- and that is precisely why this design does NOT require
// splitting the skip conv into two accumulating halves. That was the risky half of the original plan,
// and the half whose measurement was dominated by rows where one GEMM timed slower than two.
//
// Measured: 51% of cat2 saved, 1.65% projected end to end (bench_cat2_gn_fold.py). Gated to
// BIT-EXACTNESS against cat2 + the contiguous stats path on all 9 real shapes (test_cat2_gn_fold.py),
// because 1.65% cannot justify moving any number the model produces.
std::vector<torch::Tensor> group_norm_silu_delta_quantize_pack_cat2_nhwc(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor a_hat_cache, int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv, torch::Tensor mod_scale,
    torch::Tensor mod_shift, torch::Tensor absmax_buf, torch::Tensor scale_out,
    torch::Tensor inv_scale_out, torch::Tensor retire_count, double Q_level,
    bool report_next, double safety, bool write_ahat, torch::Tensor ahat_scale
) {
    auto fold = cat2_gn_stats_fp16(a, b, num_groups, eps);
    torch::Tensor x = fold[0];
    auto yqp = gn_delta_pack_impl(fold[1], fold[2], x, weight, bias, a_hat_cache, num_groups, eps, apply_silu, scale, smooth_inv,
        mod_scale, mod_shift, absmax_buf, scale_out, inv_scale_out, retire_count,
        Q_level, report_next, safety, write_ahat, ahat_scale);
    return {yqp, x};
}
