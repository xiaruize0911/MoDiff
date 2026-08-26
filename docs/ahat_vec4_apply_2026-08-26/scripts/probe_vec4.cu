// Genuine contiguous-vec4 test for the apply kernel: one thread handles 4 CONSECUTIVE flat-NHWC
// elements (a_hat as one 8-byte __half4-style load/store, code as one 32-bit int8x4 store),
// instead of vec2's 2. This is DIFFERENT from the earlier "ILP" probe, whose U>1 variants used
// idx[u] = tid + u*stride -- separated by the full grid stride, i.e. NOT adjacent in memory. That
// tested "more independent grid-strided work per thread" (measured negative); this tests "one
// wider, single, contiguous load/store instruction per thread" (unexplored until now).
//
// Only safe when CPG (channels per group) is a multiple of 4, so a 4-channel group never
// straddles a GroupNorm group boundary -- true for C in {384,768,1152,1536} (CPG 12/24/36/48) but
// NOT for C in {192,576} (CPG 6/18). Restricted to the safe shapes here; the unsafe ones would
// need a per-half-pair stats lookup within the vec4 group, a fallback not built for this probe.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float gns_silu(float v) { return v / (1.0f + expf(-v)); }

template <bool WRITE_AHAT, bool WRITE_CODE>
__global__ void probe_apply_vec4(
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
    const int CPG = C / G;
    const float scale = *scale_ptr;
    const float inv_scale = 1.0f / scale;
    const float a4_lim = a4 ? 7.0f : 127.0f;
    const long stride = (long)blockDim.x * gridDim.x;
    for (long base = 4 * ((long)blockIdx.x * blockDim.x + threadIdx.x);
         base < num_elements; base += 4 * stride) {
        int c0 = (int)(base % C);
        long n = base / sample_stride;
        long stats_idx = n * G + (c0 / CPG);          // one lookup: c0..c0+3 share a group by construction
        float mean = mean_in[stats_idx];
        float inv_std = inv_std_in[stats_idx];

        // one 8-byte load: 4 contiguous __half values
        float4 v;
        {
            __half2 a = reinterpret_cast<const __half2*>(X + base)[0];
            __half2 b = reinterpret_cast<const __half2*>(X + base)[1];
            float2 fa = __half22float2(a), fb = __half22float2(b);
            v = make_float4(fa.x, fa.y, fb.x, fb.y);
        }
        float2 w0 = __half22float2(reinterpret_cast<const __half2*>(gamma + c0)[0]);
        float2 w1 = __half22float2(reinterpret_cast<const __half2*>(gamma + c0)[1]);
        float2 b0 = __half22float2(reinterpret_cast<const __half2*>(beta + c0)[0]);
        float2 b1 = __half22float2(reinterpret_cast<const __half2*>(beta + c0)[1]);
        float n0 = (v.x - mean) * inv_std * w0.x + b0.x;
        float n1 = (v.y - mean) * inv_std * w0.y + b0.y;
        float n2 = (v.z - mean) * inv_std * w1.x + b1.x;
        float n3 = (v.w - mean) * inv_std * w1.y + b1.y;
        float o0 = apply_silu ? gns_silu(__half2float(__float2half(n0))) : n0;
        float o1 = apply_silu ? gns_silu(__half2float(__float2half(n1))) : n1;
        float o2 = apply_silu ? gns_silu(__half2float(__float2half(n2))) : n2;
        float o3 = apply_silu ? gns_silu(__half2float(__float2half(n3))) : n3;

        __half2 c0v = reinterpret_cast<const __half2*>(a_hat_cache + base)[0];
        __half2 c1v = reinterpret_cast<const __half2*>(a_hat_cache + base)[1];
        float2 ca = __half22float2(c0v), cb = __half22float2(c1v);
        float d0 = o0 - ca.x, d1 = o1 - ca.y, d2 = o2 - cb.x, d3 = o3 - cb.y;
        float q0 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d0 * scale)));
        float q1 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d1 * scale)));
        float q2 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d2 * scale)));
        float q3 = fmaxf(-a4_lim, fminf(a4_lim, roundf(d3 * scale)));

        if (WRITE_AHAT) {
            reinterpret_cast<__half2*>(a_hat_cache + base)[0] =
                __float22half2_rn(make_float2(ca.x + q0 * inv_scale, ca.y + q1 * inv_scale));
            reinterpret_cast<__half2*>(a_hat_cache + base)[1] =
                __float22half2_rn(make_float2(cb.x + q2 * inv_scale, cb.y + q3 * inv_scale));
        }
        if (WRITE_CODE) {
            int8_t i0 = (int8_t)q0, i1 = (int8_t)q1, i2 = (int8_t)q2, i3 = (int8_t)q3;
            reinterpret_cast<int32_t*>(Yq)[base >> 2] =
                (int32_t)((unsigned char)i0) | ((int32_t)(unsigned char)i1 << 8) |
                ((int32_t)(unsigned char)i2 << 16) | ((int32_t)(unsigned char)i3 << 24);
        }
    }
}

void probe_launch_vec4(torch::Tensor x, torch::Tensor a_hat, torch::Tensor yq,
                       torch::Tensor gamma, torch::Tensor beta,
                       torch::Tensor mean, torch::Tensor inv_std, torch::Tensor scale,
                       int64_t C, int64_t G, int64_t sample_stride, int64_t num_elements,
                       bool apply_silu, bool a4, bool write_ahat, bool write_code)
{
    TORCH_CHECK((C / G) % 4 == 0, "probe_launch_vec4: CPG must be a multiple of 4");
    const int ablock = 256;
    const unsigned int agrid = (unsigned int)((num_elements / 4 + ablock - 1) / ablock);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* ap = reinterpret_cast<__half*>(a_hat.data_ptr<at::Half>());
    int8_t* yp = yq.data_ptr<int8_t>();
    const __half* gp = reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>());
    const __half* bp = reinterpret_cast<const __half*>(beta.data_ptr<at::Half>());
    const float* mp = mean.data_ptr<float>();
    const float* ip = inv_std.data_ptr<float>();
    const float* sp = scale.data_ptr<float>();
#define LAUNCH4(WA, WC) probe_apply_vec4<WA, WC><<<agrid, ablock, 0, stream>>>( \
        xp, ap, yp, gp, bp, mp, ip, sp, (int)C, (int)G, sample_stride, num_elements, apply_silu, a4)
    if (write_ahat && write_code) { LAUNCH4(true, true); }
    else if (!write_ahat && write_code) { LAUNCH4(false, true); }
    else if (write_ahat && !write_code) { LAUNCH4(true, false); }
    else { LAUNCH4(false, false); }
#undef LAUNCH4
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe_launch_vec4", &probe_launch_vec4); }
