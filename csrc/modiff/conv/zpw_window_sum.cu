// ============================================================================================
// Fix #4 (weight zero point), step 1 of 2: the windowed reduction the record prices as a missing
// capability. It is a channel sum plus an R x S box filter.
//
//   S[n,ho,wo] = sum over the conv window, over ALL input channels, of the int4 activation code
//
// WHY ONE SCALAR PER OUTPUT PIXEL SUFFICES. With asymmetric int4 weights and a SYMMETRIC activation
// grid (z_a = 0, which is what ships):
//
//   out[k,p] = sum_i (w_q[k,i] - z_w[k]) * ws[k] * a_q[i]/s
//            = (ws[k]/s) * ( ACC[k,p] - z_w[k] * S[p] )
//
// S[p] does not depend on k, so ONE reduction serves every output channel, and the epilogue term is a
// rank-1 outer product (per-channel z_w) x (per-pixel S). Verified exact to float64 round-off on the
// eight real conv shapes, including strided, dilated and 1x1, by
// docs/w4a4_quality_2026-08-17/scripts/verify_zpw_decomposition.py.
//
// PADDING IS CLEAN HERE, AND THAT IS WHAT SEPARATES THIS FROM FIX #2. A padded tap has a_q = 0, and
// with z_a = 0 that code means the value 0.0 exactly -- so summing over valid taps only, which is what
// the bounds check below does, needs NO border correction. Fix #2's activation zero point had a
// `-z_a * sum(missing w_q)` term per output pixel and that is what made its cheap route lose
// (docs/zp_coverage_2026-08-13/FINDINGS.md section 3). The same script asserts the analogue is absent.
//
// TWO KERNELS, NOT ONE, on purpose. Stage 1 reads the packed activation once (C/2 bytes per pixel);
// stage 2 then re-reads only T, which is [N,H,W] int32 -- a few hundred KB against tens of MB. Fusing
// them would make every output pixel re-read R*S*C/2 activation bytes.
// ============================================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "../common/common.cuh"

namespace {

// int4 storage: low nibble = channel 2i, high nibble = channel 2i+1, packed along the innermost
// (channel) axis of NHWC. Matches csrc/modiff/quantize/delta_quantize.cu:153.
__device__ __forceinline__ int lo_nib(int8_t b) { int q = b & 0x0F; return q > 7 ? q - 16 : q; }
__device__ __forceinline__ int hi_nib(int8_t b) { int q = (b >> 4) & 0x0F; return q > 7 ? q - 16 : q; }

// Stage 1: T[n,h,w] = sum over channels. One block per pixel; the C/2 bytes of a pixel are contiguous
// in NHWC, so the strided loop below is coalesced within the block.
__global__ void zpw_channel_sum_kernel(const int8_t* __restrict__ x, int32_t* __restrict__ t,
                                       int64_t n_pix, int cp) {
    const int64_t pix = blockIdx.x;
    if (pix >= n_pix) return;
    const int8_t* row = x + pix * (int64_t)cp;
    int acc = 0;
    for (int c = threadIdx.x; c < cp; c += blockDim.x) {
        const int8_t b = row[c];
        acc += lo_nib(b) + hi_nib(b);
    }
    // block reduction in shared memory; blockDim.x is a power of two (set by the launcher)
    extern __shared__ int smem[];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) t[pix] = smem[0];
}

// Stage 2: the R x S box filter, with out-of-bounds taps contributing 0 -- correct because a padded
// tap's code is 0 and z_a = 0. Output is fp32 because the epilogue visitor consumes floats, and
// because S can reach C*R*S*7 ~ 1e5, which fp16 cannot hold.
__global__ void zpw_box_filter_kernel(const int32_t* __restrict__ t, float* __restrict__ s_out,
                                      int N, int H, int W, int Ho, int Wo,
                                      int R, int S, int stride_h, int stride_w,
                                      int pad_h, int pad_w, int dil_h, int dil_w) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * Ho * Wo;
    if (idx >= total) return;
    const int wo = idx % Wo;
    const int ho = (idx / Wo) % Ho;
    const int n = idx / ((int64_t)Ho * Wo);

    const int h0 = ho * stride_h - pad_h;
    const int w0 = wo * stride_w - pad_w;
    int acc = 0;
    for (int r = 0; r < R; ++r) {
        const int h = h0 + r * dil_h;
        if (h < 0 || h >= H) continue;
        const int32_t* trow = t + ((int64_t)n * H + h) * W;
        for (int c = 0; c < S; ++c) {
            const int w = w0 + c * dil_w;
            if (w < 0 || w >= W) continue;
            acc += trow[w];
        }
    }
    s_out[idx] = (float)acc;
}

}  // namespace

// x_packed: int8 [N, H, W, C/2], channels_last-contiguous int4 activation codes.
// Returns fp32 [N, Ho, Wo].
torch::Tensor int4_window_sum(torch::Tensor x_packed, int64_t R, int64_t S,
                              int64_t stride_h, int64_t stride_w,
                              int64_t pad_h, int64_t pad_w,
                              int64_t dil_h, int64_t dil_w) {
    CHECK_CUDA(x_packed);
    TORCH_CHECK(x_packed.dim() == 4, "x_packed must be [N,H,W,C/2], got dim ", x_packed.dim());
    TORCH_CHECK(x_packed.scalar_type() == torch::kChar,
                "x_packed must be int8 (two int4 codes per byte)");
    TORCH_CHECK(R > 0 && S > 0 && stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0,
                "R,S,stride,dilation must be positive");
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "padding must be non-negative");
    auto x = x_packed.contiguous();

    const int N = (int)x.size(0), H = (int)x.size(1), W = (int)x.size(2), cp = (int)x.size(3);
    const int Ho = (int)((H + 2 * pad_h - dil_h * (R - 1) - 1) / stride_h + 1);
    const int Wo = (int)((W + 2 * pad_w - dil_w * (S - 1) - 1) / stride_w + 1);
    TORCH_CHECK(Ho > 0 && Wo > 0, "empty output: Ho=", Ho, " Wo=", Wo);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt).device(x.device());
    auto t = torch::empty({N, H, W}, opts_i32);

    // power of two, and no larger than the row it reduces -- the shared-memory tree below assumes both
    int threads = 32;
    while (threads < cp && threads < 256) threads <<= 1;
    const int64_t n_pix = (int64_t)N * H * W;
    zpw_channel_sum_kernel<<<(unsigned)n_pix, threads, threads * sizeof(int), stream>>>(
        x.data_ptr<int8_t>(), t.data_ptr<int32_t>(), n_pix, cp);

    auto s_out = torch::empty({N, Ho, Wo},
                              torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
    const int64_t total = (int64_t)N * Ho * Wo;
    const int tb = 256;
    zpw_box_filter_kernel<<<(unsigned)((total + tb - 1) / tb), tb, 0, stream>>>(
        t.data_ptr<int32_t>(), s_out.data_ptr<float>(), N, H, W, Ho, Wo,
        (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)dil_h, (int)dil_w);

    return s_out;
}
