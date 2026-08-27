// K=2 deferred-write a_hat probe -- production-fidelity version.
//
// Adds mod_scale/mod_shift (modulation) and smooth_inv (SmoothQuant) support, and a GroupNorm
// stats launcher, so this can be wired into a REAL generation run (fp16, C<=1024, K=1 chanmajor
// vec2 stats -- the shipped default path with no MODIFF_GN_STATS_ALT override), not just synthetic
// isolated-kernel timing.
//
// Kernels below are copied from csrc/modiff/norm/group_norm_silu.cu:
//   gn_apply_delta_quantize_flat_vec2_kernel (:1858)      -> probe_standard (reference)
//   same kernel with the a_hat store removed              -> probe_skip (= ahat_overlap's w0c1)
//   NEW: reconstructs from checkpoint + resident code      -> probe_catchup
//   gn_stats_partials_chanmajor_vec2_kernel (:978)         -> stats_partials (fp16 only, K=1, no cat2)
//   gn_stats_reduce_partials_kernel (:1056)                -> stats_reduce
//
// Scope: fp16 input, static delta scale (no dynamic per-call scale), C <= 1024 and C % 2 == 0
// (the K=1 chanmajor path -- covers every shape this UNet's encoder/mid/most-decoder blocks use;
// excludes only the two decoder concat blocks with C=1152/1536, which take gn_launch_group_stats'
// K=2 split/cat2 branch, not reproduced here).
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

__device__ __forceinline__ float2 gn_load2(const __half* p, long i) {
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}
__device__ __forceinline__ void gn_store2(__half* p, long i, float2 v) {
    reinterpret_cast<__half2*>(p)[i >> 1] = __float22half2_rn(v);
}
__device__ __forceinline__ float gns_silu(float v) { return v / (1.0f + expf(-v)); }

// ---------------------------------------------------------------------------
// GroupNorm stats (K=1 chanmajor vec2 path, fp16 only, no cat2 split)
// ---------------------------------------------------------------------------
__global__ void stats_partials(
    const __half* __restrict__ X,
    float* __restrict__ part_sum, float* __restrict__ part_sumsq,
    int C, long HW, int G, int nblocks)
{
    const int CPG = C / G;
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const int c = 2 * t;
    const __half2* src = reinterpret_cast<const __half2*>(X + (long)n * HW * C);
    const long C2 = C / 2;
    float sa = 0.0f, sqa = 0.0f, sb = 0.0f, sqb = 0.0f;
    long hw = blockIdx.x;
    for (; hw + 3L * nblocks < HW; hw += 4L * nblocks) {
        const __half2 r0 = src[(hw) * C2 + t];
        const __half2 r1 = src[(hw + nblocks) * C2 + t];
        const __half2 r2 = src[(hw + 2L * nblocks) * C2 + t];
        const __half2 r3 = src[(hw + 3L * nblocks) * C2 + t];
        const float2 f0 = __half22float2(r0), f1 = __half22float2(r1);
        const float2 f2 = __half22float2(r2), f3 = __half22float2(r3);
        sa += f0.x; sqa += f0.x * f0.x; sb += f0.y; sqb += f0.y * f0.y;
        sa += f1.x; sqa += f1.x * f1.x; sb += f1.y; sqb += f1.y * f1.y;
        sa += f2.x; sqa += f2.x * f2.x; sb += f2.y; sqb += f2.y * f2.y;
        sa += f3.x; sqa += f3.x * f3.x; sb += f3.y; sqb += f3.y * f3.y;
    }
    for (; hw < HW; hw += nblocks) {
        const __half2 r = src[hw * C2 + t];
        const float2 v = __half22float2(r);
        sa += v.x; sqa += v.x * v.x;
        sb += v.y; sqb += v.y * v.y;
    }
    (void)c;
    extern __shared__ float sdata[];
    float* ss = sdata;
    float* sq_s = sdata + C;
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

__global__ void stats_reduce(
    const float* __restrict__ part_sum, const float* __restrict__ part_sumsq,
    float* __restrict__ mean_out, float* __restrict__ inv_std_out,
    int nblocks, long group_size, float eps, int NG)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NG) return;
    float s = 0.0f, sq = 0.0f;
    const long base = (long)i * nblocks;
    for (int b = 0; b < nblocks; ++b) { s += part_sum[base + b]; sq += part_sumsq[base + b]; }
    float mean = s / (float)group_size;
    float var = sq / (float)group_size - mean * mean;
    var = fmaxf(var, 0.0f);
    mean_out[i] = mean;
    inv_std_out[i] = rsqrtf(var + eps);
}

void stats_launch(torch::Tensor x, torch::Tensor mean, torch::Tensor inv_std,
                  int64_t C, int64_t G, int64_t HW, double eps) {
    TORCH_CHECK(C <= 1024 && (C % 2) == 0 && (C / 2) >= G,
                "K=2 probe stats: only C<=1024, C even, C/2>=G supported (K=1 chanmajor path)");
    const int N = (int)x.size(0);
    const int nblocks = (int)std::min<long>(HW, 32);
    auto sopt = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto part_sum = torch::empty({(long)N * G * nblocks}, sopt);
    auto part_sumsq = torch::empty({(long)N * G * nblocks}, sopt);
    const int NG = N * (int)G;
    const long group_size = (C / G) * HW;
    const size_t shmem = (size_t)2 * C * sizeof(float);
    dim3 grid((unsigned)nblocks, (unsigned)N);
    const int BLK2 = (int)C / 2;
    auto stream = c10::cuda::getCurrentCUDAStream();
    stats_partials<<<grid, BLK2, shmem, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(), (int)C, HW, (int)G, nblocks);
    const int fb = 128, fg = (NG + fb - 1) / fb;
    stats_reduce<<<fg, fb, 0, stream>>>(
        part_sum.data_ptr<float>(), part_sumsq.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(), nblocks, group_size, (float)eps, NG);
}

// ---------------------------------------------------------------------------
// Apply kernels
// ---------------------------------------------------------------------------
__global__ void probe_skip(
    const __half* __restrict__ X, __half* __restrict__ a_hat_cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const __half* __restrict__ mod_scale, const __half* __restrict__ mod_shift,
    const __half* __restrict__ smooth_inv,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    const float* __restrict__ scale_ptr,
    int C, int G, long sample_stride, long num_elements, bool apply_silu, bool a4)
{
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float a4_lim = a4 ? 7.0f : 127.0f;
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
        float n0h = __half2float(__float2half(n0));
        float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        if (smooth_inv != nullptr) {
            o0 *= __half2float(smooth_inv[c0]);
            o1 *= __half2float(smooth_inv[c0 + 1]);
        }
        float2 cache = gn_load2(a_hat_cache, base);
        const float d0 = o0 - cache.x, d1 = o1 - cache.y;
        float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        // NO a_hat write here -- this is the whole point.
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        reinterpret_cast<int16_t*>(Yq)[base >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
    }
}

__global__ void probe_catchup(
    const __half* __restrict__ X, __half* __restrict__ a_hat_cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const __half* __restrict__ mod_scale, const __half* __restrict__ mod_shift,
    const __half* __restrict__ smooth_inv,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    const float* __restrict__ scale_ptr, const float* __restrict__ prev_inv_scale_ptr,
    int C, int G, long sample_stride, long num_elements, bool apply_silu, bool a4)
{
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float prev_inv_scale = *prev_inv_scale_ptr;
    const float a4_lim = a4 ? 7.0f : 127.0f;
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
        float n0h = __half2float(__float2half(n0));
        float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        if (smooth_inv != nullptr) {
            o0 *= __half2float(smooth_inv[c0]);
            o1 *= __half2float(smooth_inv[c0 + 1]);
        }

        float2 A = gn_load2(a_hat_cache, base);
        int16_t packed_c1 = reinterpret_cast<const int16_t*>(Yq)[base >> 1];
        int8_t c1_0 = (int8_t)(packed_c1 & 0xFF);
        int8_t c1_1 = (int8_t)((packed_c1 >> 8) & 0xFF);
        // Round-trip through __half: the standard kernel would have STORED this sum to a_hat's
        // fp16 buffer and READ it back; reconstructing in pure float32 carries more precision
        // than the standard scheme's a_hat_1 actually has (the bug the real build caught).
        float a1_0 = __half2float(__float2half_rn(A.x + (float)c1_0 * prev_inv_scale));
        float a1_1 = __half2float(__float2half_rn(A.y + (float)c1_1 * prev_inv_scale));

        const float d0 = o0 - a1_0, d1 = o1 - a1_1;
        float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        gn_store2(a_hat_cache, base, make_float2(a1_0 + q0 * inv_scale, a1_1 + q1 * inv_scale));
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        reinterpret_cast<int16_t*>(Yq)[base >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
    }
}

__global__ void probe_standard(
    const __half* __restrict__ X, __half* __restrict__ a_hat_cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const __half* __restrict__ mod_scale, const __half* __restrict__ mod_shift,
    const __half* __restrict__ smooth_inv,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    const float* __restrict__ scale_ptr,
    int C, int G, long sample_stride, long num_elements, bool apply_silu, bool a4)
{
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float a4_lim = a4 ? 7.0f : 127.0f;
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
        float n0h = __half2float(__float2half(n0));
        float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        if (smooth_inv != nullptr) {
            o0 *= __half2float(smooth_inv[c0]);
            o1 *= __half2float(smooth_inv[c0 + 1]);
        }
        float2 cache = gn_load2(a_hat_cache, base);
        const float d0 = o0 - cache.x, d1 = o1 - cache.y;
        float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        gn_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        reinterpret_cast<int16_t*>(Yq)[base >> 1] =
            (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
    }
}

static void launch_common(dim3& grid, dim3& block, long num_elements) {
    block = dim3(256);
    grid = dim3((unsigned int)((num_elements / 2 + 255) / 256));
}

static const __half* opt_half(torch::Tensor t) {
    return t.numel() > 0 ? reinterpret_cast<const __half*>(t.data_ptr<at::Half>()) : nullptr;
}

void probe_standard_launch(torch::Tensor x, torch::Tensor a_hat, torch::Tensor yq,
                           torch::Tensor gamma, torch::Tensor beta,
                           torch::Tensor mod_scale, torch::Tensor mod_shift, torch::Tensor smooth_inv,
                           torch::Tensor mean, torch::Tensor inv_std, torch::Tensor scale,
                           int64_t C, int64_t G, int64_t sample_stride, int64_t num_elements,
                           bool apply_silu, bool a4) {
    dim3 grid, block;
    launch_common(grid, block, num_elements);
    auto stream = c10::cuda::getCurrentCUDAStream();
    probe_standard<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>()), yq.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        opt_half(mod_scale), opt_half(mod_shift), opt_half(smooth_inv),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(), scale.data_ptr<float>(),
        (int)C, (int)G, sample_stride, num_elements, apply_silu, a4);
}

void probe_skip_launch(torch::Tensor x, torch::Tensor a_hat, torch::Tensor yq,
                       torch::Tensor gamma, torch::Tensor beta,
                       torch::Tensor mod_scale, torch::Tensor mod_shift, torch::Tensor smooth_inv,
                       torch::Tensor mean, torch::Tensor inv_std, torch::Tensor scale,
                       int64_t C, int64_t G, int64_t sample_stride, int64_t num_elements,
                       bool apply_silu, bool a4) {
    dim3 grid, block;
    launch_common(grid, block, num_elements);
    auto stream = c10::cuda::getCurrentCUDAStream();
    probe_skip<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>()), yq.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        opt_half(mod_scale), opt_half(mod_shift), opt_half(smooth_inv),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(), scale.data_ptr<float>(),
        (int)C, (int)G, sample_stride, num_elements, apply_silu, a4);
}

void probe_catchup_launch(torch::Tensor x, torch::Tensor a_hat, torch::Tensor yq,
                          torch::Tensor gamma, torch::Tensor beta,
                          torch::Tensor mod_scale, torch::Tensor mod_shift, torch::Tensor smooth_inv,
                          torch::Tensor mean, torch::Tensor inv_std, torch::Tensor scale,
                          torch::Tensor prev_inv_scale,
                          int64_t C, int64_t G, int64_t sample_stride, int64_t num_elements,
                          bool apply_silu, bool a4) {
    dim3 grid, block;
    launch_common(grid, block, num_elements);
    auto stream = c10::cuda::getCurrentCUDAStream();
    probe_catchup<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>()), yq.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        opt_half(mod_scale), opt_half(mod_shift), opt_half(smooth_inv),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(), scale.data_ptr<float>(),
        prev_inv_scale.data_ptr<float>(),
        (int)C, (int)G, sample_stride, num_elements, apply_silu, a4);
}

// ---------------------------------------------------------------------------
// General-K windowed scheme: generalizes probe_skip/probe_catchup (K=2) to any window size.
// At position p (0-indexed) within a K-step window:
//   p == 0:        reference = A (the checkpoint), read directly, no reconstruction.
//   0 < p < K-1:    reference = A, then p SEQUENTIAL fp16-rounded adds of pending_codes[0..p-1]
//                   -- replicating exactly the standard scheme's chain of per-step roundings,
//                   which is why each add is followed by its own __float2half_rn, not one round
//                   at the end (rounding is not linear, so summing first then rounding once
//                   would NOT reproduce the standard scheme's value).
//   p == K-1:       same reconstruction (using all K-1 pending codes), then WRITES a_hat_cache
//                   (the real, final catch-up write) instead of storing into pending_codes.
// Yq is written with this step's code on EVERY position (o_hat needs a fresh code every step
// regardless of window position).
__global__ void probe_window_step(
    const __half* __restrict__ X, __half* __restrict__ a_hat_cache, int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const __half* __restrict__ mod_scale, const __half* __restrict__ mod_shift,
    const __half* __restrict__ smooth_inv,
    const float* __restrict__ mean_in, const float* __restrict__ inv_std_in,
    const float* __restrict__ scale_ptr,
    int8_t* __restrict__ pending_codes,           // [Kmax-1, N,C,H,W]; [0..position-1] valid on entry
    const float* __restrict__ pending_inv_scales, // [Kmax-1], device buffer set from Python
    long numel, int position, bool is_last,
    int C, int G, long sample_stride, long num_elements, bool apply_silu, bool a4)
{
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float a4_lim = a4 ? 7.0f : 127.0f;
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
        float n0h = __half2float(__float2half(n0));
        float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        if (smooth_inv != nullptr) {
            o0 *= __half2float(smooth_inv[c0]);
            o1 *= __half2float(smooth_inv[c0 + 1]);
        }

        float2 A = gn_load2(a_hat_cache, base);
        float r0 = A.x, r1 = A.y;
        for (int j = 0; j < position; ++j) {
            int16_t packed = reinterpret_cast<const int16_t*>(pending_codes + j * numel)[base >> 1];
            int8_t cj0 = (int8_t)(packed & 0xFF);
            int8_t cj1 = (int8_t)((packed >> 8) & 0xFF);
            float pj_inv = pending_inv_scales[j];
            r0 = __half2float(__float2half_rn(r0 + (float)cj0 * pj_inv));
            r1 = __half2float(__float2half_rn(r1 + (float)cj1 * pj_inv));
        }

        const float d0 = o0 - r0, d1 = o1 - r1;
        float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
        int16_t packed_out = (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
        reinterpret_cast<int16_t*>(Yq)[base >> 1] = packed_out;

        if (is_last) {
            gn_store2(a_hat_cache, base, make_float2(r0 + q0 * inv_scale, r1 + q1 * inv_scale));
        } else {
            reinterpret_cast<int16_t*>(pending_codes + position * numel)[base >> 1] = packed_out;
        }
    }
}

void probe_window_step_launch(torch::Tensor x, torch::Tensor a_hat, torch::Tensor yq,
                              torch::Tensor gamma, torch::Tensor beta,
                              torch::Tensor mod_scale, torch::Tensor mod_shift, torch::Tensor smooth_inv,
                              torch::Tensor mean, torch::Tensor inv_std, torch::Tensor scale,
                              torch::Tensor pending_codes, torch::Tensor pending_inv_scales,
                              int64_t numel, int64_t position, bool is_last,
                              int64_t C, int64_t G, int64_t sample_stride, int64_t num_elements,
                              bool apply_silu, bool a4) {
    dim3 grid, block;
    launch_common(grid, block, num_elements);
    auto stream = c10::cuda::getCurrentCUDAStream();
    probe_window_step<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>()), yq.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        opt_half(mod_scale), opt_half(mod_shift), opt_half(smooth_inv),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(), scale.data_ptr<float>(),
        pending_codes.data_ptr<int8_t>(), pending_inv_scales.data_ptr<float>(),
        numel, (int)position, is_last,
        (int)C, (int)G, sample_stride, num_elements, apply_silu, a4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stats_launch", &stats_launch);
    m.def("probe_standard_launch", &probe_standard_launch);
    m.def("probe_skip_launch", &probe_skip_launch);
    m.def("probe_catchup_launch", &probe_catchup_launch);
    m.def("probe_window_step_launch", &probe_window_step_launch);
}
