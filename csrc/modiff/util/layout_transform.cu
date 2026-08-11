// =========================================================================
// MoDiff delta-quantize layout transform. Baseline twin: csrc/baseline/util/layout_transform.cu.
//
// The attention blocks run their QKV/output projections as Conv1d(kernel=1), executed as a
// Conv2d GEMM by reshaping to channels-last -- a real transpose (NCW -> NHWC-like). The baseline
// file fuses transpose+cast; this file's kernel additionally fuses in MoDiff's delta-quantize
// step (K3), so the whole pre-GEMM pipeline for MoDiff's INT8 attention path is one launch:
//   residual = x - a_hat_cache  ->  quantize  ->  transposed channels-last int8 store
//
// Family 6 of the csrc/ datapath split (2026-08-12). Nothing is shared with the baseline file
// beyond the TILE_T tile width, which is a #define and is duplicated below -- the four delta
// kernels are launched only by the host function here, so this is a clean move with no static
// twins to keep in sync.
// =========================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// COPY of the tile width from the baseline file: same TILE_T x TILE_T shared-memory tile
// transpose (with a +1 column pad against bank conflicts), so both sides stay coalesced.
#define TILE_T 32   // match warp width

// =========================================================================
// Fused K1+K2+K3: FP16 NCW -> INT8 CL with MoDiff delta subtraction
//
// Single kernel replaces:
//   K1+K2: fp16_ncw_to_fp32_cl   (FP16 [N,C,L] -> FP32 [N*L,C,1,1] CL)
//   K3:    step1_static_quantize  (FP32 CL, a_hat -> INT8 CL, a_hat updated)
//
// Memory layout:
//   x_fp16   : [N, C, L]       FP16  NCW  - input activations
//   a_hat    : [N*L, C]        FP32  CL   - MoDiff cache (updated in-place)
//   out_int8 : [N*L, C]        INT8  CL   - CUTLASS GEMM input
//
// Tiling (same TILE_T=32 as the kernels above):
//   Phase 1: coalesced FP16 reads  (threadIdx.x -> NL) -> tile[c][nl]
//   Phase 2: threadIdx.x -> C      -> read tile (transposed), delta, quantize,
//            update a_hat, write INT8 CL  (coalesced writes)
// =========================================================================
__global__ void fp16_ncw_delta_to_int8_cl_kernel(
    const __half* __restrict__ x,      // [N, C, L]  FP16 NCW
    float*        __restrict__ a_hat,  // [N*L, C]   FP32 CL  (in-place)
    int8_t*       __restrict__ out,    // [N*L, C]   INT8 CL
    float scale,                       // static_input_scale = 127/max_abs  (quantize)
    float inv_scale,                   // 1/scale = max_abs/127              (dequantize)
    int N, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int NL      = N * L;
    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: coalesced FP16 reads (threadIdx.x varies NL direction)
    {
        int nl = nl_base + threadIdx.x;
        int c  = c_base  + threadIdx.y;
        if (nl < NL && c < C) {
            int n = nl / L, l = nl % L;
            tile[threadIdx.y][threadIdx.x] = __half2float(x[n * C * L + c * L + l]);
        } else {
            tile[threadIdx.y][threadIdx.x] = 0.f;
        }
    }
    __syncthreads();

    // Phase 2: subtract a_hat, quantize, update cache, write INT8
    //          (threadIdx.x varies C direction -> coalesced a_hat/out accesses)
    {
        int nl  = nl_base + threadIdx.y;
        int c   = c_base  + threadIdx.x;
        if (nl < NL && c < C) {
            float xval = tile[threadIdx.x][threadIdx.y];  // transposed access
            int   idx  = nl * C + c;                      // CL index = [N*L, C]
            float r    = xval - a_hat[idx];
            float q    = fmaxf(-127.f, fminf(127.f, rintf(r * scale)));
            a_hat[idx] += q * inv_scale;                  // update cache
            out[idx]    = (int8_t)q;                      // write INT8
        }
    }
}

// Same as fp16_ncw_delta_to_int8_cl_kernel above, but a_hat is stored as FP16
// (halves cache memory, matches OptimizedInt8Conv2d switching cache_dtype to
// FP16 once calibration finishes -- see int8_optimized.py's cache_dtype). The
// cache is read/written in FP32 arithmetic internally for the same numerics
// as the FP32-cache kernel; only the resident storage differs.
__global__ void fp16_ncw_delta_to_int8_cl_kernel_half_cache(
    const __half* __restrict__ x,      // [N, C, L]  FP16 NCW
    __half*       __restrict__ a_hat,  // [N*L, C]   FP16 CL  (in-place)
    int8_t*       __restrict__ out,    // [N*L, C]   INT8 CL
    float scale,                       // static_input_scale = 127/max_abs  (quantize)
    float inv_scale,                   // 1/scale = max_abs/127              (dequantize)
    int N, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int NL      = N * L;
    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: coalesced FP16 reads (threadIdx.x varies NL direction)
    {
        int nl = nl_base + threadIdx.x;
        int c  = c_base  + threadIdx.y;
        if (nl < NL && c < C) {
            int n = nl / L, l = nl % L;
            tile[threadIdx.y][threadIdx.x] = __half2float(x[n * C * L + c * L + l]);
        } else {
            tile[threadIdx.y][threadIdx.x] = 0.f;
        }
    }
    __syncthreads();

    // Phase 2: subtract a_hat, quantize, update cache, write INT8
    //          (threadIdx.x varies C direction -> coalesced a_hat/out accesses)
    {
        int nl  = nl_base + threadIdx.y;
        int c   = c_base  + threadIdx.x;
        if (nl < NL && c < C) {
            float xval = tile[threadIdx.x][threadIdx.y];  // transposed access
            int   idx  = nl * C + c;                      // CL index = [N*L, C]
            float a_v  = __half2float(a_hat[idx]);
            float r    = xval - a_v;
            float q    = fmaxf(-127.f, fminf(127.f, rintf(r * scale)));
            a_hat[idx] = __float2half(a_v + q * inv_scale);  // update cache
            out[idx]   = (int8_t)q;                          // write INT8
        }
    }
}

// Vectorized (half2) phase-1-only counterpart of fp16_ncw_delta_to_int8_cl_kernel. Phase 2 (the
// delta/quantize/cache-update/int8-write) is left scalar and untouched here -- it interleaves an
// a_hat read-modify-write with the int8 quantize, more moving parts to re-verify bit-exactly than
// the plain transpose+cast kernels above, so this round only closes the same fp16-read
// under-utilization gap (32 threads x 2B = 64B/warp) that fp16_ncw_to_fp32_cl_vec2_kernel does.
// Same L % 2 == 0 safety gate.
__global__ void fp16_ncw_delta_to_int8_cl_vec2_kernel(
    const __half* __restrict__ x,      // [N, C, L]  FP16 NCW
    float*        __restrict__ a_hat,  // [N*L, C]   FP32 CL  (in-place)
    int8_t*       __restrict__ out,    // [N*L, C]   INT8 CL
    float scale,
    float inv_scale,
    int N, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int NL      = N * L;
    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: vectorized coalesced reads, 2 nl per thread
    if (threadIdx.x < TILE_T / 2) {
        int nl0 = nl_base + threadIdx.x * 2;
        int c   = c_base  + threadIdx.y;
        float v0 = 0.f, v1 = 0.f;
        if (c < C && nl0 < NL) {
            int n = nl0 / L, l0 = nl0 % L;
            if (nl0 + 1 < NL) {
                float2 v = __half22float2(*reinterpret_cast<const __half2*>(&x[(long)n * C * L + (long)c * L + l0]));
                v0 = v.x; v1 = v.y;
            } else {
                v0 = __half2float(x[(long)n * C * L + (long)c * L + l0]);
            }
        }
        tile[threadIdx.y][threadIdx.x * 2]     = v0;
        tile[threadIdx.y][threadIdx.x * 2 + 1] = v1;
    }
    __syncthreads();

    // Phase 2: unchanged scalar delta/quantize/cache-update/int8-write
    {
        int nl  = nl_base + threadIdx.y;
        int c   = c_base  + threadIdx.x;
        if (nl < NL && c < C) {
            float xval = tile[threadIdx.x][threadIdx.y];
            int   idx  = nl * C + c;
            float r    = xval - a_hat[idx];
            float q    = fmaxf(-127.f, fminf(127.f, rintf(r * scale)));
            a_hat[idx] += q * inv_scale;
            out[idx]    = (int8_t)q;
        }
    }
}

// Vectorized (half2) phase-1-only counterpart of fp16_ncw_delta_to_int8_cl_kernel_half_cache.
// Same rationale/gate as fp16_ncw_delta_to_int8_cl_vec2_kernel above; phase 2 (fp16-cache
// read-modify-write + int8 quantize) is left scalar.
__global__ void fp16_ncw_delta_to_int8_cl_kernel_half_cache_vec2(
    const __half* __restrict__ x,      // [N, C, L]  FP16 NCW
    __half*       __restrict__ a_hat,  // [N*L, C]   FP16 CL  (in-place)
    int8_t*       __restrict__ out,    // [N*L, C]   INT8 CL
    float scale,
    float inv_scale,
    int N, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int NL      = N * L;
    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: vectorized coalesced reads, 2 nl per thread
    if (threadIdx.x < TILE_T / 2) {
        int nl0 = nl_base + threadIdx.x * 2;
        int c   = c_base  + threadIdx.y;
        float v0 = 0.f, v1 = 0.f;
        if (c < C && nl0 < NL) {
            int n = nl0 / L, l0 = nl0 % L;
            if (nl0 + 1 < NL) {
                float2 v = __half22float2(*reinterpret_cast<const __half2*>(&x[(long)n * C * L + (long)c * L + l0]));
                v0 = v.x; v1 = v.y;
            } else {
                v0 = __half2float(x[(long)n * C * L + (long)c * L + l0]);
            }
        }
        tile[threadIdx.y][threadIdx.x * 2]     = v0;
        tile[threadIdx.y][threadIdx.x * 2 + 1] = v1;
    }
    __syncthreads();

    // Phase 2: unchanged scalar delta/quantize/cache-update/int8-write
    {
        int nl  = nl_base + threadIdx.y;
        int c   = c_base  + threadIdx.x;
        if (nl < NL && c < C) {
            float xval = tile[threadIdx.x][threadIdx.y];
            int   idx  = nl * C + c;
            float a_v  = __half2float(a_hat[idx]);
            float r    = xval - a_v;
            float q    = fmaxf(-127.f, fminf(127.f, rintf(r * scale)));
            a_hat[idx] = __float2half(a_v + q * inv_scale);
            out[idx]   = (int8_t)q;
        }
    }
}

// C++ wrapper: FP16 NCW + a_hat (FP32 or FP16) CL -> INT8 CL, a_hat updated
// in-place. Fuses K1+K2 (layout transpose) + K3 (MoDiff delta quantize).
//   Op:       Layout transform (FP16 NCW -> INT8 channels-last) + MoDiff delta-quantize + cache update
//   Inputs:   x FP16 [N,C,L] NCW input activations; a_hat FP32 or FP16 [N*L, C, 1, 1]
//             channels-last MoDiff cache (in-place); scale_t FP32 [1] = static_input_scale
//             = 127/max_abs; N, C, L ints
//   Outputs:  INT8 [N*L, C, 1, 1] channels-last (== [N*L, C], CUTLASS GEMM input);
//             a_hat updated in place
//   Computes: per (nl, c): r = x_transposed - a_hat; q = clamp(round(r*scale), -127, 127);
//             a_hat += q/scale; out = q  (temporal delta against the cache, then int8 quantize)
//   Fuses:    K1+K2 (NCW->CL transpose + fp16->fp32) + K3 (delta subtract + int8 quantize +
//             cache update) into ONE launch — the whole pre-GEMM pipeline for MoDiff's INT8
//             attention path; TILE_T=32 shared-memory tile transpose, coalesced both sides
//   Constraints: CUDA FP16 x; a_hat FP32 or FP16 (half-cache read/written in fp32 arithmetic
//                for identical numerics); C % 4 == 0; scale read on host via .item() (one sync)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
torch::Tensor fp16_ncw_delta_to_int8_cl(
    torch::Tensor x,        // FP16 [N, C, L]
    torch::Tensor a_hat,    // FP32 or FP16 [N*L, C, 1, 1] channels-last  (updated in-place)
    torch::Tensor scale_t,  // FP32 [1]  = static_input_scale = 127/max_abs
    int N, int C, int L
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf,
                "fp16_ncw_delta_to_int8_cl: expected CUDA FP16 tensor for x");
    TORCH_CHECK(a_hat.is_cuda() && (a_hat.scalar_type() == at::kFloat || a_hat.scalar_type() == at::kHalf),
                "fp16_ncw_delta_to_int8_cl: expected CUDA FP32 or FP16 tensor for a_hat");
    TORCH_CHECK(C % 4 == 0, "fp16_ncw_delta_to_int8_cl: C must be divisible by 4");

    int NL = N * L;
    float scale_val     = scale_t.item<float>();
    float inv_scale_val = 1.0f / scale_val;

    // Output: INT8 [N*L, C, 1, 1] channels-last (H=W=1 -> identical to [N*L, C])
    auto out = torch::empty(
        {NL, C, 1, 1},
        x.options()
            .dtype(torch::kInt8)
            .memory_format(torch::MemoryFormat::ChannelsLast)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 block(TILE_T, TILE_T);
    dim3 grid((C  + TILE_T - 1) / TILE_T,
              (NL + TILE_T - 1) / TILE_T);

    const bool use_vec2 = (L % 2 == 0);
    if (a_hat.scalar_type() == at::kHalf) {
        if (use_vec2) {
            fp16_ncw_delta_to_int8_cl_kernel_half_cache_vec2<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>()),
                out.data_ptr<int8_t>(),
                scale_val, inv_scale_val,
                N, C, L
            );
        } else {
            fp16_ncw_delta_to_int8_cl_kernel_half_cache<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>()),
                out.data_ptr<int8_t>(),
                scale_val, inv_scale_val,
                N, C, L
            );
        }
    } else {
        if (use_vec2) {
            fp16_ncw_delta_to_int8_cl_vec2_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                a_hat.data_ptr<float>(),
                out.data_ptr<int8_t>(),
                scale_val, inv_scale_val,
                N, C, L
            );
        } else {
            fp16_ncw_delta_to_int8_cl_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                a_hat.data_ptr<float>(),
                out.data_ptr<int8_t>(),
                scale_val, inv_scale_val,
                N, C, L
            );
        }
    }
    return out;
}
