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

// Vectorized (half2) counterpart of fp16_ncw_to_fp32_cl_kernel. The scalar phase-1 read above
// moves only 2 bytes/thread (32 threads x 2B = 64B/warp, under a full 128B transaction); this
// widens it to a half2 load, 2 elements/thread, halving the active thread count for phase 1 only
// (phase 2, the FP32 write, is already 128B/warp and untouched). Safe when L is even: nl and nl+1
// share the same n whenever nl is even (since l = nl % L is then even too, and L even means the
// last element of any n, l=L-1, is always odd -- so an even-aligned pair never straddles the n
// boundary). The caller only dispatches here when L % 2 == 0, with the scalar kernel as fallback.
__global__ void fp16_ncw_to_fp32_cl_vec2_kernel(
    const __half* __restrict__ src,   // [N, C, L]
    float*        __restrict__ dst,   // [N*L, C]
    int N, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int NL      = N * L;
    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: vectorized coalesced reads, 2 nl per thread (threadIdx.x in [0, TILE_T/2))
    if (threadIdx.x < TILE_T / 2) {
        int nl0 = nl_base + threadIdx.x * 2;
        int c   = c_base  + threadIdx.y;
        float v0 = 0.f, v1 = 0.f;
        if (c < C && nl0 < NL) {
            int n = nl0 / L, l0 = nl0 % L;
            if (nl0 + 1 < NL) {
                float2 v = __half22float2(*reinterpret_cast<const __half2*>(&src[(long)n * C * L + (long)c * L + l0]));
                v0 = v.x; v1 = v.y;
            } else {
                v0 = __half2float(src[(long)n * C * L + (long)c * L + l0]);
            }
        }
        tile[threadIdx.y][threadIdx.x * 2]     = v0;
        tile[threadIdx.y][threadIdx.x * 2 + 1] = v1;
    }
    __syncthreads();

    // Phase 2: unchanged, coalesced FP32 writes (threadIdx.x -> C direction)
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

// Vectorized (half2) counterpart of fp32_cl_to_fp16_ncw_kernel. Phase 1 (FP32 read) is already
// 128B/warp and untouched; phase 2 (FP16 write) moves only 2 bytes/thread, widened here to a
// half2 store, 2 elements/thread. Same L % 2 == 0 safety argument as fp16_ncw_to_fp32_cl_vec2_kernel
// above (an even-aligned nl pair never straddles an n boundary).
__global__ void fp32_cl_to_fp16_ncw_vec2_kernel(
    const float* __restrict__ src,   // [N*L, C]
    __half*      __restrict__ dst,   // [N, C, L]
    int NL, int C, int L
) {
    __shared__ float tile[TILE_T][TILE_T + 1];

    int c_base  = blockIdx.x * TILE_T;
    int nl_base = blockIdx.y * TILE_T;

    // Phase 1: unchanged, coalesced FP32 reads (threadIdx.x -> C direction)
    {
        int nl = nl_base + threadIdx.y;
        int c  = c_base  + threadIdx.x;
        tile[threadIdx.y][threadIdx.x] = (nl < NL && c < C) ? src[nl * C + c] : 0.f;
    }
    __syncthreads();

    // Phase 2: vectorized coalesced writes, 2 nl per thread (threadIdx.x in [0, TILE_T/2))
    if (threadIdx.x < TILE_T / 2) {
        int nl0 = nl_base + threadIdx.x * 2;
        int c   = c_base  + threadIdx.y;
        if (c < C && nl0 < NL) {
            int n = nl0 / L, l0 = nl0 % L;
            float v0 = tile[threadIdx.x * 2][threadIdx.y];
            if (nl0 + 1 < NL) {
                float v1 = tile[threadIdx.x * 2 + 1][threadIdx.y];
                *reinterpret_cast<__half2*>(&dst[(long)n * C * L + (long)c * L + l0]) = __floats2half2_rn(v0, v1);
            } else {
                dst[(long)n * C * L + (long)c * L + l0] = __float2half(v0);
            }
        }
    }
}

// C++ wrapper: FP16 NCW -> FP32 channels-last (fuses K1+K2)
//   Op:       Layout transform (FP16 NCW -> FP32 channels-last) + dtype cast
//   Inputs:   src FP16 [N,C,L] (NCW); N, C, L ints (logical dims)
//   Outputs:  FP32 [N*L, C, 1, 1] channels-last (== [N*L, C], the Conv2d-GEMM input layout)
//   Computes: dst[nl, c] = float(src[n, c, l]) with nl = n*L + l  (transpose NCW->[N*L,C] + fp16->fp32 cast)
//   Fuses:    transpose + dtype cast (K1+K2) in one launch, replacing
//             x.permute(0,2,1).contiguous().float(); TILE_T=32 shared-memory tile transpose
//             keeps both read and write coalesced (+1 pad avoids bank conflicts)
//   Constraints: CUDA FP16 input; C % 4 == 0 (TORCH_CHECK)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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

    // Vec2 phase-1 read is safe when L is even (see fp16_ncw_to_fp32_cl_vec2_kernel's comment);
    // the file's own C%4==0 check already guarantees L%2==0 doesn't need a separate assumption
    // check here since L is an independent dimension (sequence length), not derived from C.
    if (L % 2 == 0) {
        fp16_ncw_to_fp32_cl_vec2_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
            dst.data_ptr<float>(),
            N, C, L
        );
    } else {
        fp16_ncw_to_fp32_cl_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
            dst.data_ptr<float>(),
            N, C, L
        );
    }
    return dst;
}






// C++ wrapper: FP32 channels-last -> FP16 NCW (fuses K7+K8)
//   Op:       Layout transform (FP32 channels-last -> FP16 NCW) + dtype cast
//   Inputs:   src FP32 [N*L, C] channels-last (Conv2d-GEMM output); N, C, L ints
//   Outputs:  FP16 [N, C, L] (NCW)
//   Computes: dst[n, c, l] = half(src[nl, c]) with nl = n*L + l  (transpose [N*L,C]->NCW + fp32->fp16 cast)
//   Fuses:    transpose + dtype cast (K7+K8) in one launch, replacing
//             out.permute(0,2,1).contiguous().half(); TILE_T=32 shared-memory tile transpose,
//             coalesced both sides
//   Constraints: CUDA FP32 input; C % 4 == 0 (TORCH_CHECK)
//   vs fp16:  n/a (quantization / layout / MoDiff-caching support op — no fp16 equivalent; these are the overhead ops that fusion tries to hide)
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

    if (L % 2 == 0) {
        fp32_cl_to_fp16_ncw_vec2_kernel<<<grid, block, 0, stream>>>(
            src.data_ptr<float>(),
            reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
            NL, C, L
        );
    } else {
        fp32_cl_to_fp16_ncw_kernel<<<grid, block, 0, stream>>>(
            src.data_ptr<float>(),
            reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
            NL, C, L
        );
    }
    return dst;
}

// =========================================================================
// Specialized 2-tensor channels-last concat, replacing openaimodel.py's
// `torch.cat([h, hs.pop()], dim=1)` (UNetModel.forward's decoder skip-concat).
// Pure data movement -- no arithmetic at all, so this is bit-identical to
// torch.cat([a, b], dim=1) by construction (not something that needs a
// tolerance-based check; a `torch.equal` capture-compare is enough).
//
// ATen's generic N-way `CatArrayBatchedCopy` doesn't know at compile time
// that there are exactly 2 inputs, both channels_last fp16 -- it moves 2
// bytes/thread (measured as the `OpaqueType<2u>` scalar path). Since both C1
// and C2 are even for every real shape in this model (192/384/768/1536,
// confirmed by tracing every skip-concat call site this session), every
// output half2 pair falls entirely within one source tensor's channel range
// (a pair can only straddle the A/B boundary if C1 is odd), so this can
// always vectorize with a plain per-pair branch and never needs a scalar
// tail/boundary case for this model's shapes -- verified by the C1%2==0/
// C2%2==0 TORCH_CHECK below; the Python call site falls back to plain
// `torch.cat` whenever a shape wouldn't satisfy it.
//   Op:       Channels-last concat along dim=1, specialized for exactly 2 FP16 inputs
//   Inputs:   a [N,C1,H,W], b [N,C2,H,W], both channels_last FP16 CUDA, same N,H,W
//   Outputs:  [N,C1+C2,H,W] channels_last FP16, identical to torch.cat([a,b],dim=1)
//   Fuses:    n/a (not a fusion -- a specialized vectorized replacement for a generic ATen op)
//   Constraints: C1 % 2 == 0 && C2 % 2 == 0 (TORCH_CHECK'd; every real shape in this model satisfies this)
__global__ void cat2_channels_last_fp16_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b,
    __half* __restrict__ out, long num_positions, int C1, int C2
) {
    int Ctot = C1 + C2;
    int pairs_per_pos = Ctot / 2;
    int a_pairs = C1 / 2;
    long total_pairs = num_positions * (long)pairs_per_pos;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    for (long i = idx; i < total_pairs; i += stride) {
        long p = i / pairs_per_pos;
        int pair_c = (int)(i % pairs_per_pos);
        __half2 v;
        if (pair_c < a_pairs) {
            v = *reinterpret_cast<const __half2*>(&a[p * C1 + pair_c * 2]);
        } else {
            int pair_b = pair_c - a_pairs;
            v = *reinterpret_cast<const __half2*>(&b[p * C2 + pair_b * 2]);
        }
        *reinterpret_cast<__half2*>(&out[p * Ctot + pair_c * 2]) = v;
    }
}

torch::Tensor cat2_channels_last_fp16(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cat2_channels_last_fp16: expected CUDA tensors");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
                "cat2_channels_last_fp16: expected FP16 tensors");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "cat2_channels_last_fp16: expected 4D [N,C,H,W] tensors");
    TORCH_CHECK(a.size(0) == b.size(0) && a.size(2) == b.size(2) && a.size(3) == b.size(3),
                "cat2_channels_last_fp16: N,H,W must match");
    TORCH_CHECK(a.is_contiguous(at::MemoryFormat::ChannelsLast) &&
                b.is_contiguous(at::MemoryFormat::ChannelsLast),
                "cat2_channels_last_fp16: both inputs must be channels_last contiguous");
    int N = a.size(0), C1 = a.size(1), H = a.size(2), W = a.size(3), C2 = b.size(1);
    TORCH_CHECK(C1 % 2 == 0 && C2 % 2 == 0, "cat2_channels_last_fp16: C1 and C2 must be even");

    auto out = torch::empty({N, C1 + C2, H, W},
        a.options().memory_format(at::MemoryFormat::ChannelsLast));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    long num_positions = (long)N * H * W;
    int pairs_per_pos = (C1 + C2) / 2;
    long total_pairs = num_positions * (long)pairs_per_pos;
    int block_size = 256;
    int grid_size = (int)std::min<long>((total_pairs + block_size - 1) / block_size, 2147483647L);

    cat2_channels_last_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        num_positions, C1, C2
    );
    return out;
}


// -----------------------------------------------------------------------------
//   Op:       packed-INT4 -> INT8 widening, NHWC-physical
//   Inputs:   packed  int8 [N, Ho, Wo, C/2] (or any buffer whose bytes are NHWC-physical with
//                    Kpad/2 bytes per spatial position), the nibble layout every int4 quantize
//                    kernel in this tree writes: byte c/2 holds channel c in the LOW nibble and
//                    channel c+1 in the HIGH nibble, each a signed 4-bit two's-complement code
//             C       logical channel count (even)
//   Outputs:  int8 [N, C, Ho, Wo] channels_last, codes sign-extended to 8 bits
//   Computes: value-preserving widening. A nibble code q in [-7,7] becomes the int8 q, so the
//             GEMM's `alpha` (the reciprocal of the scale that produced q) is UNCHANGED and the
//             dequantized result is bit-identical to what an int4 GEMM would have produced.
//
//   Why this exists: W8A4 -- int8 weights, 4-bit activations -- has no hardware path. Both int4
//   tensor-core MMAs take BOTH operands at 4 bits, and no mainstream ISA has a mixed s8xs4 MMA, so
//   the activation cannot be fed to a GEMM as nibbles alongside int8 weights. Routing the
//   activation through int4 STORAGE and widening it here makes the 4-bitness a property of the
//   format rather than of a clamp parameter: a nibble physically cannot hold a code above 7, so
//   there is nothing left for a code ceiling to enforce. That is what retires `clamp_code`.
//
//   UNREFERENCED, AND THE REASON IS THE FINDING (see the DEAD-CODE POLICY at the top of
//   csrc/modiff_kernels_api.h, which keeps a kernel only on exactly these grounds). Built and
//   verified 2026-08-10 to answer whether W8A4 should route its activation through int4 storage
//   rather than an int8 container with a code ceiling. Both halves were measured:
//     * EQUIVALENCE: pack+widen vs the int8 quantize at Q_b=7 is BIT-IDENTICAL, on all ten cases
//       including the scale-4x-too-fine regime that is the only place the two could differ (a stale
//       MODIFF_DELTA_REFRESH>1 scale, or a clip ratio below 1). So it changes no number.
//     * COST: +4.74 ms/step at batch 128 over the 70 conv layers -- 4.5% of the ~105 ms/step the
//       configuration runs at, and 3.8x the 1.24 ms the updown fusion had just recovered.
//   Paying 4.5% for a bit-identical re-encoding was refused. W8A4 instead names its datapath with
//   the `a4` bool the quantize kernels now take, which gets the same "4 bits is not a magnitude a
//   caller can pass wrongly" property for nothing. Kept because the equivalence result is what
//   makes that choice defensible, and rebuilding this to re-derive it would cost a rebuild plus a
//   GPU hour. Harness: integration/tests/bench_unpack_int4_widen.py.
//
//   Cost detail: one extra pass over the activation (read C/2 bytes, write C). W8A4 was already a
//   quality-only configuration -- "not a speed configuration on any hardware"
//   (docs/act_bits_2026-08-05/FINDINGS.md) -- which is why the trade was worth measuring at all.
//
//   Constraints: C even; `packed` contiguous with kpad_bytes (>= C/2) bytes per spatial position.
// -----------------------------------------------------------------------------
__global__ void unpack_int4_to_int8_cl_kernel(
    const int8_t* __restrict__ packed,
    int8_t* __restrict__ out,
    long num_positions,          // N * Ho * Wo
    int C,
    int kpad_bytes               // bytes per spatial position in `packed` (>= C/2)
) {
    const int pairs_per_pos = C / 2;
    const long total_pairs = num_positions * (long)pairs_per_pos;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total_pairs;
         idx += (long)blockDim.x * gridDim.x) {
        const long pos = idx / pairs_per_pos;
        const int pair = (int)(idx % pairs_per_pos);
        const uint8_t b = (uint8_t)packed[pos * (long)kpad_bytes + pair];
        // Sign-extend each nibble: 0..7 stay, 8..15 map to -8..-1. The quantizers clamp at +-7 so
        // -8 is unreachable in practice, but decoding it correctly costs nothing and keeps this a
        // faithful inverse of the packing rather than a partial one.
        int lo = (int)(b & 0x0F); if (lo > 7) lo -= 16;
        int hi = (int)((b >> 4) & 0x0F); if (hi > 7) hi -= 16;
        // Adjacent output channels are adjacent bytes in NHWC, so this is one aligned 2-byte store.
        const long o = pos * (long)C + 2 * pair;
        out[o]     = (int8_t)lo;
        out[o + 1] = (int8_t)hi;
    }
}

torch::Tensor unpack_int4_to_int8_cl(torch::Tensor packed, int64_t N, int64_t C,
                                     int64_t H, int64_t W) {
    // This translation unit does not include common.cuh, so the checks are spelled out the way
    // every other entry point here spells them.
    TORCH_CHECK(packed.is_cuda(), "unpack_int4_to_int8_cl: packed must be a CUDA tensor");
    TORCH_CHECK(packed.is_contiguous(), "unpack_int4_to_int8_cl: packed must be contiguous");
    TORCH_CHECK(packed.scalar_type() == torch::kInt8,
                "unpack_int4_to_int8_cl: packed must be int8 (nibble pairs)");
    TORCH_CHECK(C % 2 == 0, "unpack_int4_to_int8_cl: C must be even");
    const long num_positions = (long)N * H * W;
    TORCH_CHECK(num_positions > 0, "unpack_int4_to_int8_cl: empty shape");
    TORCH_CHECK(packed.numel() % num_positions == 0,
                "unpack_int4_to_int8_cl: packed numel must divide by N*H*W");
    const long kpad_bytes = packed.numel() / num_positions;
    TORCH_CHECK(kpad_bytes >= C / 2,
                "unpack_int4_to_int8_cl: packed has fewer than C/2 bytes per position");

    auto out = torch::empty({N, C, H, W},
        packed.options().memory_format(at::MemoryFormat::ChannelsLast));
    const long total_pairs = num_positions * (C / 2);
    const int block = 256;
    const int grid = (int)std::min<long>((total_pairs + block - 1) / block, 2147483647L);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unpack_int4_to_int8_cl_kernel<<<grid, block, 0, stream>>>(
        packed.data_ptr<int8_t>(), out.data_ptr<int8_t>(),
        num_positions, (int)C, (int)kpad_bytes);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
