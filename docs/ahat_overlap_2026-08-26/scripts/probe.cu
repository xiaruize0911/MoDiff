// a_hat write-elision probe.
//
// Body copied VERBATIM from gn_apply_delta_quantize_flat_vec2_kernel
// (csrc/modiff/norm/group_norm_silu.cu:1701), the kernel the shipped MoDiff GN path runs in
// static delta mode, with two changes and nothing else:
//   1. the a_hat store and the code store are behind template flags, so the same body can be
//      timed with and without each;
//   2. gn_report_delta_absmax is dropped -- production passes absmax_buf = nullptr in static
//      mode and the helper's first statement is `if (absmax_buf == nullptr) return;`.
// Launch geometry matches production: block 256, grid = ceil(num_elements/2/256), and the
// 256-float dynamic shared allocation is kept so occupancy is identical.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ float2 gn_load2(const __half* p, long i) {
    return __half22float2(reinterpret_cast<const __half2*>(p)[i >> 1]);
}
__device__ __forceinline__ void gn_store2(__half* p, long i, float2 v) {
    reinterpret_cast<__half2*>(p)[i >> 1] = __float22half2_rn(v);
}
__device__ __forceinline__ float gns_silu(float v) { return v / (1.0f + expf(-v)); }

template <bool WRITE_AHAT, bool WRITE_CODE, bool PACK4>
__global__ void probe_apply_vec2(
    const __half* __restrict__ X,
    __half* __restrict__ a_hat_cache,
    int8_t* __restrict__ Yq,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    const float* __restrict__ mean_in,
    const float* __restrict__ inv_std_in,
    const float* __restrict__ scale_ptr,
    int C, int G, long sample_stride, long num_elements, bool apply_silu, bool a4)
{
    extern __shared__ float sdata[];
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float a4_lim = (a4 || PACK4) ? 7.0f : 127.0f;
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
        float n0h = __half2float(__float2half(n0));
        float n1h = __half2float(__float2half(n1));
        float o0 = apply_silu ? gns_silu(n0h) : n0h;
        float o1 = apply_silu ? gns_silu(n1h) : n1h;
        float2 cache = gn_load2(a_hat_cache, base);
        const float d0 = o0 - cache.x, d1 = o1 - cache.y;
        local_max = fmaxf(local_max, fmaxf(fabsf(d0), fabsf(d1)));
        float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        if (WRITE_AHAT) {
            gn_store2(a_hat_cache, base, make_float2(cache.x + q0 * inv_scale, cache.y + q1 * inv_scale));
        }
        if (WRITE_CODE) {
            int8_t i0 = (int8_t)q0, i1 = (int8_t)q1;
            if (PACK4) {
                // COPY of gn_apply_delta_quantize_pack_flat_vec2_kernel's store (:2337)
                Yq[base / 2] = (int8_t)((i0 & 0x0F) | ((i1 & 0x0F) << 4));
            } else {
                reinterpret_cast<int16_t*>(Yq)[base >> 1] =
                    (int16_t)(((unsigned char)i0) | (((unsigned char)i1) << 8));
            }
        } else {
            // keep the quantize live so the compiler cannot delete the arithmetic
            if (q0 == 12345.0f && q1 == 54321.0f) Yq[0] = 1;
        }
    }
    // absmax reporting disabled in static mode; keep local_max live
    if (local_max < 0.0f) sdata[0] = local_max;
}

void probe_launch(torch::Tensor x, torch::Tensor a_hat, torch::Tensor yq,
                  torch::Tensor gamma, torch::Tensor beta,
                  torch::Tensor mean, torch::Tensor inv_std, torch::Tensor scale,
                  int64_t C, int64_t G, int64_t sample_stride, int64_t num_elements,
                  bool apply_silu, bool a4, bool write_ahat, bool write_code, bool pack4)
{
    const int ablock = 256;
    const unsigned int agrid = (unsigned int)((num_elements / 2 + ablock - 1) / ablock);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* ap = reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>());
    int8_t* yp = yq.data_ptr<int8_t>();
    const __half* gp = reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>());
    const __half* bp = reinterpret_cast<const __half*>(beta.data_ptr<at::Half>());
    const float* mp = mean.data_ptr<float>();
    const float* ip = inv_std.data_ptr<float>();
    const float* sp = scale.data_ptr<float>();
    size_t shm = ablock * sizeof(float);
#define LAUNCH(WA, WC, PK) probe_apply_vec2<WA, WC, PK><<<agrid, ablock, shm, stream>>>( \
        xp, ap, yp, gp, bp, mp, ip, sp, (int)C, (int)G, sample_stride, num_elements, apply_silu, a4)
    if (pack4) {
        if (write_ahat && write_code)        { LAUNCH(true, true, true); }
        else if (!write_ahat && write_code)  { LAUNCH(false, true, true); }
        else if (write_ahat && !write_code)  { LAUNCH(true, false, true); }
        else                                 { LAUNCH(false, false, true); }
    } else {
        if (write_ahat && write_code)        { LAUNCH(true, true, false); }
        else if (!write_ahat && write_code)  { LAUNCH(false, true, false); }
        else if (write_ahat && !write_code)  { LAUNCH(true, false, false); }
        else                                 { LAUNCH(false, false, false); }
    }
#undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe_launch", &probe_launch); }
