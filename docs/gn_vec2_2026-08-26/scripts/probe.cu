// gn_stats_partials_chanmajor_kernel variants. V0 is a verbatim copy of the shipped kernel's
// non-split path at K=1 (C <= 1024, which every shape in this UNet satisfies).
//
// The bit-exactness invariants the shipped kernel relies on, and that every variant preserves:
//   (i)   each channel has its OWN fp32 accumulator,
//   (ii)  it accumulates over hw in ASCENDING order (hw = blockIdx.x, +nblocks, ...),
//   (iii) shared memory is indexed by channel, and one lane per group sums its CPG channels in
//         ascending channel index.
// V1 only hoists loads (same values, same order). V2 only changes WHICH channels a thread owns
// (adjacent instead of one), so each channel still has its own accumulator walking hw ascending.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float gn_load(const __half* p, long i) { return __half2float(p[i]); }

// ---------------- V0: verbatim (K=1, non-split) ----------------
template <int K>
__global__ void v0(const __half* __restrict__ X, float* __restrict__ part_sum,
                   float* __restrict__ part_sumsq, int C, long HW, int G, int nblocks) {
    const int CPG = C / G;
    const int B = blockDim.x;
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const __half* a_base = X + (long)n * HW * C;
    float s[K], sq[K];
#pragma unroll
    for (int k = 0; k < K; ++k) { s[k] = 0.0f; sq[k] = 0.0f; }
    for (long hw = blockIdx.x; hw < HW; hw += nblocks) {
        const long row = hw * (long)C;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float v = gn_load(a_base, row + t + k * B);
            s[k] += v; sq[k] += v * v;
        }
    }
    extern __shared__ float sdata[];
    float* ss = sdata; float* sq_s = sdata + C;
#pragma unroll
    for (int k = 0; k < K; ++k) { ss[t + k * B] = s[k]; sq_s[t + k * B] = sq[k]; }
    __syncthreads();
    if (t < G) {
        float gs = 0.0f, gsq = 0.0f;
        const int c0 = t * CPG;
        for (int k = 0; k < CPG; ++k) { gs += ss[c0 + k]; gsq += sq_s[c0 + k]; }
        const long o = ((long)n * G + t) * nblocks + blockIdx.x;
        part_sum[o] = gs; part_sumsq[o] = gsq;
    }
}

// ---------------- V1: hw loop unrolled x4, loads hoisted ----------------
template <int K>
__global__ void v1(const __half* __restrict__ X, float* __restrict__ part_sum,
                   float* __restrict__ part_sumsq, int C, long HW, int G, int nblocks) {
    const int CPG = C / G;
    const int B = blockDim.x;
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const __half* a_base = X + (long)n * HW * C;
    float s[K], sq[K];
#pragma unroll
    for (int k = 0; k < K; ++k) { s[k] = 0.0f; sq[k] = 0.0f; }
    long hw = blockIdx.x;
    for (; hw + 3L * nblocks < HW; hw += 4L * nblocks) {
        float r[K][4];
#pragma unroll
        for (int k = 0; k < K; ++k) {   // four INDEPENDENT loads before the first dependent add
            const int c = t + k * B;
            r[k][0] = gn_load(a_base, (hw) * (long)C + c);
            r[k][1] = gn_load(a_base, (hw + nblocks) * (long)C + c);
            r[k][2] = gn_load(a_base, (hw + 2L * nblocks) * (long)C + c);
            r[k][3] = gn_load(a_base, (hw + 3L * nblocks) * (long)C + c);
        }
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j) { s[k] += r[k][j]; sq[k] += r[k][j] * r[k][j]; }
    }
    for (; hw < HW; hw += nblocks) {
        const long row = hw * (long)C;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float v = gn_load(a_base, row + t + k * B);
            s[k] += v; sq[k] += v * v;
        }
    }
    extern __shared__ float sdata[];
    float* ss = sdata; float* sq_s = sdata + C;
#pragma unroll
    for (int k = 0; k < K; ++k) { ss[t + k * B] = s[k]; sq_s[t + k * B] = sq[k]; }
    __syncthreads();
    if (t < G) {
        float gs = 0.0f, gsq = 0.0f;
        const int c0 = t * CPG;
        for (int k = 0; k < CPG; ++k) { gs += ss[c0 + k]; gsq += sq_s[c0 + k]; }
        const long o = ((long)n * G + t) * nblocks + blockIdx.x;
        part_sum[o] = gs; part_sumsq[o] = gsq;
    }
}

// ---------------- V2: one thread owns two ADJACENT channels, loaded as one __half2 ----------------
// blockDim.x == C/2. Channel 2t and 2t+1 keep separate accumulators, so (i)-(iii) hold.
__global__ void v2(const __half* __restrict__ X, float* __restrict__ part_sum,
                   float* __restrict__ part_sumsq, int C, long HW, int G, int nblocks) {
    const int CPG = C / G;
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const __half2* a_base = reinterpret_cast<const __half2*>(X + (long)n * HW * C);
    const long C2 = C / 2;
    float sa = 0.0f, sqa = 0.0f, sb = 0.0f, sqb = 0.0f;
    for (long hw = blockIdx.x; hw < HW; hw += nblocks) {
        const float2 v = __half22float2(a_base[hw * C2 + t]);
        sa += v.x; sqa += v.x * v.x;
        sb += v.y; sqb += v.y * v.y;
    }
    extern __shared__ float sdata[];
    float* ss = sdata; float* sq_s = sdata + C;
    ss[2 * t] = sa; sq_s[2 * t] = sqa;
    ss[2 * t + 1] = sb; sq_s[2 * t + 1] = sqb;
    __syncthreads();
    if (t < G) {
        float gs = 0.0f, gsq = 0.0f;
        const int c0 = t * CPG;
        for (int k = 0; k < CPG; ++k) { gs += ss[c0 + k]; gsq += sq_s[c0 + k]; }
        const long o = ((long)n * G + t) * nblocks + blockIdx.x;
        part_sum[o] = gs; part_sumsq[o] = gsq;
    }
}

// ---------------- V3: both ----------------
__global__ void v3(const __half* __restrict__ X, float* __restrict__ part_sum,
                   float* __restrict__ part_sumsq, int C, long HW, int G, int nblocks) {
    const int CPG = C / G;
    const int t = threadIdx.x;
    const int n = blockIdx.y;
    const __half2* a_base = reinterpret_cast<const __half2*>(X + (long)n * HW * C);
    const long C2 = C / 2;
    float sa = 0.0f, sqa = 0.0f, sb = 0.0f, sqb = 0.0f;
    long hw = blockIdx.x;
    for (; hw + 3L * nblocks < HW; hw += 4L * nblocks) {
        const __half2 r0 = a_base[(hw) * C2 + t];
        const __half2 r1 = a_base[(hw + nblocks) * C2 + t];
        const __half2 r2 = a_base[(hw + 2L * nblocks) * C2 + t];
        const __half2 r3 = a_base[(hw + 3L * nblocks) * C2 + t];
        const float2 f0 = __half22float2(r0), f1 = __half22float2(r1);
        const float2 f2 = __half22float2(r2), f3 = __half22float2(r3);
        sa += f0.x; sqa += f0.x * f0.x; sb += f0.y; sqb += f0.y * f0.y;
        sa += f1.x; sqa += f1.x * f1.x; sb += f1.y; sqb += f1.y * f1.y;
        sa += f2.x; sqa += f2.x * f2.x; sb += f2.y; sqb += f2.y * f2.y;
        sa += f3.x; sqa += f3.x * f3.x; sb += f3.y; sqb += f3.y * f3.y;
    }
    for (; hw < HW; hw += nblocks) {
        const float2 v = __half22float2(a_base[hw * C2 + t]);
        sa += v.x; sqa += v.x * v.x;
        sb += v.y; sqb += v.y * v.y;
    }
    extern __shared__ float sdata[];
    float* ss = sdata; float* sq_s = sdata + C;
    ss[2 * t] = sa; sq_s[2 * t] = sqa;
    ss[2 * t + 1] = sb; sq_s[2 * t + 1] = sqb;
    __syncthreads();
    if (t < G) {
        float gs = 0.0f, gsq = 0.0f;
        const int c0 = t * CPG;
        for (int k = 0; k < CPG; ++k) { gs += ss[c0 + k]; gsq += sq_s[c0 + k]; }
        const long o = ((long)n * G + t) * nblocks + blockIdx.x;
        part_sum[o] = gs; part_sumsq[o] = gsq;
    }
}

void launch(torch::Tensor x, torch::Tensor psum, torch::Tensor psumsq,
            int64_t N, int64_t C, int64_t HW, int64_t G, int64_t nblocks, int64_t variant) {
    dim3 grid((unsigned)nblocks, (unsigned)N);
    size_t shm = (size_t)2 * C * sizeof(float);
    auto st = c10::cuda::getCurrentCUDAStream();
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    float* ps = psum.data_ptr<float>(); float* pq = psumsq.data_ptr<float>();
    // scalar variants need the shipped kernel's K split once C exceeds the 1024-thread cap;
    // the vec2 variants never do on this UNet, because C/2 <= 1024 for every C <= 2048.
    const int Kk = (int)((C + 1023) / 1024);
    const unsigned Bs = (unsigned)(C / Kk);
    if (variant == 0) { if (Kk == 1) v0<1><<<grid, Bs, shm, st>>>(xp, ps, pq, C, HW, G, nblocks);
                        else         v0<2><<<grid, Bs, shm, st>>>(xp, ps, pq, C, HW, G, nblocks); }
    else if (variant == 1) { if (Kk == 1) v1<1><<<grid, Bs, shm, st>>>(xp, ps, pq, C, HW, G, nblocks);
                             else         v1<2><<<grid, Bs, shm, st>>>(xp, ps, pq, C, HW, G, nblocks); }
    else if (variant == 2) v2<<<grid, (unsigned)C / 2, shm, st>>>(xp, ps, pq, C, HW, G, nblocks);
    else                   v3<<<grid, (unsigned)C / 2, shm, st>>>(xp, ps, pq, C, HW, G, nblocks);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
