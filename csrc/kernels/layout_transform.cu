// =========================================================================
// Fused layout-transpose + dtype-cast kernels for MoDiffConv1dCUTLASS.
//
// The attention blocks run their QKV/output projections as Conv1d(kernel=1),
// which this project executes as a Conv2d GEMM by reshaping to channels-last.
// That reshape is a real transpose (NCW -> NHWC-like), not just a view, so
// naively it costs two separate kernels each way:
//   before: x.permute(0,2,1).contiguous().float()   (transpose + cast)
//   after:  out.permute(0,2,1).contiguous().half()   (transpose + cast)
// The kernels below fuse each pair into one kernel, and fp16_ncw_delta_to_int8_cl
// additionally fuses in the MoDiff delta-quantize step (K3), so the whole
// pre-GEMM pipeline for MoDiff's INT8 attention path is a single kernel launch.
//
// All three use the same TILE_T x TILE_T shared-memory tile transpose so that
// both the read and write side are fully coalesced regardless of the L
// stride, with a +1 column pad to avoid shared-memory bank conflicts on the
// transposed access.
// =========================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_T 32   // match warp width

// FP16 [N,C,L] -> FP32 [N*L,C,1,1] channels-last (fuses K1+K2).
//
// Phase 1 (coalesced FP16 reads, threadIdx.x varies NL i.e. l-within-n):
//   src addr = n*C*L + c*L + l  -- adjacent l -> adjacent addresses
// Phase 2 (coalesced FP32 writes, threadIdx.x varies C):
//   dst addr = nl*C + c         -- adjacent c -> adjacent addresses
__global__ void fp16_ncw_to_fp32_cl_kernel(
    const __half* __restrict__ src,   // [N, C, L]
    float*        __restrict__ dst,   // [N*L, C]
    int N, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];  // +1 avoids bank conflicts

    int NL      = N * L;
    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: coalesced reads (threadIdx.x -> NL direction)
    {
        int nl = nl_base + threadIdx.x;
        int c  = c_base  + threadIdx.y;
        if (nl < NL && c < C) {
            int n = nl / L, l = nl % L;
            tile[threadIdx.y][threadIdx.x] = __half2float(src[n * C * L + c * L + l]);
        } else {
            tile[threadIdx.y][threadIdx.x] = 0.f;
        }
    }
    __syncthreads();

    // Phase 2: coalesced writes (threadIdx.x -> C direction)
    {
        int nl = nl_base + threadIdx.y;
        int c  = c_base  + threadIdx.x;
        if (nl < NL && c < C)
            dst[nl * C + c] = tile[threadIdx.x][threadIdx.y];
    }
}

// FP32 [N*L,C,1,1] channels-last -> FP16 [N,C,L] (fuses K7+K8).
//
// Phase 1 (coalesced FP32 reads, threadIdx.x varies C):
//   src addr = nl*C + c        -- adjacent c -> adjacent addresses
// Phase 2 (coalesced FP16 writes, threadIdx.x varies NL i.e. l-within-n):
//   dst addr = n*C*L + c*L + l -- adjacent l -> adjacent addresses
__global__ void fp32_cl_to_fp16_ncw_kernel(
    const float* __restrict__ src,   // [N*L, C]
    __half*      __restrict__ dst,   // [N, C, L]
    int NL, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: coalesced reads (threadIdx.x -> C direction)
    {
        int nl = nl_base + threadIdx.y;
        int c  = c_base  + threadIdx.x;
        tile[threadIdx.y][threadIdx.x] = (nl < NL && c < C) ? src[nl * C + c] : 0.f;
    }
    __syncthreads();

    // Phase 2: coalesced writes (threadIdx.x -> NL direction)
    {
        int nl = nl_base + threadIdx.x;
        int c  = c_base  + threadIdx.y;
        if (nl < NL && c < C) {
            int n = nl / L, l = nl % L;
            dst[n * C * L + c * L + l] = __float2half(tile[threadIdx.x][threadIdx.y]);
        }
    }
}

// C++ wrapper: FP16 NCW -> FP32 channels-last (fuses K1+K2)
torch::Tensor fp16_ncw_to_fp32_cl(
    torch::Tensor src,
    int N, int C, int L
) {
    TORCH_CHECK(src.is_cuda() && src.scalar_type() == at::kHalf,
                "fp16_ncw_to_fp32_cl: expected CUDA FP16 tensor");
    TORCH_CHECK(C % 4 == 0, "fp16_ncw_to_fp32_cl: C must be divisible by 4");

    auto dst = torch::empty(
        {N * L, C, 1, 1},
        src.options()
            .dtype(torch::kFloat32)
            .memory_format(torch::MemoryFormat::ChannelsLast)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int NL = N * L;
    dim3 block(TILE_T, TILE_T);
    dim3 grid((C  + TILE_T - 1) / TILE_T,
              (NL + TILE_T - 1) / TILE_T);

    fp16_ncw_to_fp32_cl_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
        dst.data_ptr<float>(),
        N, C, L
    );
    return dst;
}

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

// C++ wrapper: FP16 NCW + a_hat FP32 CL -> INT8 CL, a_hat updated in-place.
// Fuses K1+K2 (layout transpose) + K3 (MoDiff delta quantize).
torch::Tensor fp16_ncw_delta_to_int8_cl(
    torch::Tensor x,        // FP16 [N, C, L]
    torch::Tensor a_hat,    // FP32 [N*L, C, 1, 1] channels-last  (updated in-place)
    torch::Tensor scale_t,  // FP32 [1]  = static_input_scale = 127/max_abs
    int N, int C, int L
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf,
                "fp16_ncw_delta_to_int8_cl: expected CUDA FP16 tensor for x");
    TORCH_CHECK(a_hat.is_cuda() && a_hat.scalar_type() == at::kFloat,
                "fp16_ncw_delta_to_int8_cl: expected CUDA FP32 tensor for a_hat");
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

    fp16_ncw_delta_to_int8_cl_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        a_hat.data_ptr<float>(),
        out.data_ptr<int8_t>(),
        scale_val, inv_scale_val,
        N, C, L
    );
    return out;
}

// C++ wrapper: FP32 channels-last -> FP16 NCW (fuses K7+K8)
torch::Tensor fp32_cl_to_fp16_ncw(
    torch::Tensor src,
    int N, int C, int L
) {
    TORCH_CHECK(src.is_cuda() && src.scalar_type() == at::kFloat,
                "fp32_cl_to_fp16_ncw: expected CUDA FP32 tensor");
    TORCH_CHECK(C % 4 == 0, "fp32_cl_to_fp16_ncw: C must be divisible by 4");

    auto dst = torch::empty(
        {N, C, L},
        src.options().dtype(torch::kHalf)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int NL = N * L;
    dim3 block(TILE_T, TILE_T);
    dim3 grid((C  + TILE_T - 1) / TILE_T,
              (NL + TILE_T - 1) / TILE_T);

    fp32_cl_to_fp16_ncw_kernel<<<grid, block, 0, stream>>>(
        src.data_ptr<float>(),
        reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
        NL, C, L
    );
    return dst;
}
