// =========================================================================
// Fused int8 flash attention for the diffusion-UNet self-attention blocks.
//
// Replaces the math-SDPA score path (QKᵀ / softmax / AV) which materializes the
// full [N,H,T,T] score matrix in HBM. This kernel keeps the running softmax
// state in registers and never writes the T×T matrix, and contracts int8
// Q/K/V operands (2× the bytes shrink vs fp16). Per (n, head, query-row) one
// CTA streams over all keys once (two light passes: max, then sum+AV).
//
// v1 (this file) uses the portable __dp4a int8x4→int32 dot product for QKᵀ and
// an fp32 accumulation for AV (P kept in fp32 → no P-requant error). The
// tensor-core m16n8k32.s8.s8.s32 path (mma_int8.cuh) is a later perf pass (M7),
// gated per-shape by microbench. Softmax (max/exp/normalize) is always fp32.
//
// Layout contract (all contiguous):
//   q,k,v : [N,H,T,hd_pad] int8   (head_dim padded up to a multiple of 4;
//                                   padded lanes are zero → exact dot product)
//   sq,sk : [N,H,T]        f32     per-token dequant scales (q_fp ≈ q_i8·sq)
//   sv    : [N,H,hd]       f32     per-channel (head_dim) dequant scale for V
//   out   : [N,H,T,hd]     f16
//   softmax_scale = 1/sqrt(hd)
// =========================================================================
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "../common.cuh"
#include "mma_int8.cuh"

#define MODIFF_FA_MAX_HD 96      // max padded head_dim in the churches UNet (hd=96)
#define MODIFF_FA_THREADS 128

// Tiled (flash) kernel tuning. int8 path runs only for T>=64 (head_dim<=48 ->
// hd_pad<=64), so the K/V/S/O smem tiles fit under the 48 KB static limit.
#define FA_BR 64
#define FA_BC 64
#define FA_TILE_HD 64            // smem stride for hd_pad (<= this on the int8 path)
#define FA_TILED_THREADS 256

// ---- naive fallback kernel (one CTA per query row): correct for ANY head_dim,
// used for the hd=96 / tiny-T shapes; the tiled kernel handles hd_pad<=64. ----
__global__ void flash_attn_int8_naive_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hd_pad, float softmax_scale) {
  const int g = blockIdx.x;                 // one CTA per (n, head, query row i)
  const int i = g % T;
  const int hh = (g / T) % H;
  const int nn = g / (T * H);
  const int nh = nn * H + hh;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int8_t* qrow  = q + (size_t)(nh * T + i) * hd_pad;
  const int8_t* kbase = k + (size_t)(nh * T) * hd_pad;
  const int8_t* vbase = v + (size_t)(nh * T) * hd_pad;
  const float*  skb   = sk + (size_t)nh * T;
  const float*  svb   = sv + (size_t)nh * hd;
  const float   sqi   = sq[(size_t)nh * T + i];

  // dynamic smem: red[nthreads] | Os[hd] | qs[hd_pad]
  extern __shared__ float smem[];
  float*  red = smem;
  float*  Os  = red + nthreads;
  int8_t* qs  = reinterpret_cast<int8_t*>(Os + hd);

  for (int d = tid; d < hd_pad; d += nthreads) qs[d] = qrow[d];
  for (int d = tid; d < hd;     d += nthreads) Os[d] = 0.f;
  __syncthreads();

  // ---- Pass 1: running max over keys ----
  float lmax = -INFINITY;
  for (int j = tid; j < T; j += nthreads) {
    int dot = dp4a_i8(qs, kbase + (size_t)j * hd_pad, hd_pad);
    float s = dot * sqi * skb[j] * softmax_scale;
    lmax = fmaxf(lmax, s);
  }
  red[tid] = lmax; __syncthreads();
  for (int st = nthreads >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  const float m = red[0];
  __syncthreads();

  // ---- Pass 2: sum(exp) + AV, all fp32 ----
  float lsum = 0.f;
  float acc[MODIFF_FA_MAX_HD];
#pragma unroll
  for (int d = 0; d < MODIFF_FA_MAX_HD; ++d) acc[d] = 0.f;
  for (int j = tid; j < T; j += nthreads) {
    int dot = dp4a_i8(qs, kbase + (size_t)j * hd_pad, hd_pad);
    float s = dot * sqi * skb[j] * softmax_scale;
    float p = __expf(s - m);
    lsum += p;
    const int8_t* vrow = vbase + (size_t)j * hd_pad;
    for (int d = 0; d < hd; ++d) acc[d] += p * (float)vrow[d];
  }
  red[tid] = lsum; __syncthreads();
  for (int st = nthreads >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] += red[tid + st];
    __syncthreads();
  }
  const float l = red[0];
  __syncthreads();

  for (int d = 0; d < hd; ++d) atomicAdd(&Os[d], acc[d]);
  __syncthreads();

  const float invl = (l > 0.f) ? (1.f / l) : 0.f;
  for (int d = tid; d < hd; d += nthreads) {
    out[(size_t)(nh * T + i) * hd + d] = __float2half(Os[d] * invl * svb[d]);
  }
}

// ---- tiled flash kernel: one CTA per (n,head,query-block of BR rows). K/V tiles
// are staged in shared memory and reused across all BR query rows, so K/V are read
// from global once per query-block instead of once per query row (the naive
// kernel's fatal O(T^2) HBM traffic). QKᵀ via __dp4a on smem operands; online
// softmax + PV accumulation in fp32. ----
__global__ void flash_attn_int8_tiled_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hd_pad, float softmax_scale) {
  const int nh = blockIdx.x;
  const int q0 = blockIdx.y * FA_BR;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int8_t* qbase = q + (size_t)nh * T * hd_pad;
  const int8_t* kbase = k + (size_t)nh * T * hd_pad;
  const int8_t* vbase = v + (size_t)nh * T * hd_pad;
  const float*  sqb = sq + (size_t)nh * T;
  const float*  skb = sk + (size_t)nh * T;
  const float*  svb = sv + (size_t)nh * hd;

  __shared__ int8_t Qs[FA_BR * FA_TILE_HD];
  __shared__ int8_t Ks[FA_BC * FA_TILE_HD];
  __shared__ int8_t Vs[FA_BC * FA_TILE_HD];
  __shared__ float  Ss[FA_BR * FA_BC];
  __shared__ float  Os[FA_BR * FA_TILE_HD];
  __shared__ float  m_s[FA_BR], l_s[FA_BR], sqi_s[FA_BR];

  for (int idx = tid; idx < FA_BR * hd_pad; idx += nthreads) {
    int i = idx / hd_pad, d = idx % hd_pad, gi = q0 + i;
    Qs[i * FA_TILE_HD + d] = (gi < T) ? qbase[(size_t)gi * hd_pad + d] : 0;
  }
  for (int i = tid; i < FA_BR; i += nthreads) {
    int gi = q0 + i;
    m_s[i] = -INFINITY; l_s[i] = 0.f; sqi_s[i] = (gi < T) ? sqb[gi] : 0.f;
  }
  for (int idx = tid; idx < FA_BR * hd; idx += nthreads)
    Os[(idx / hd) * FA_TILE_HD + (idx % hd)] = 0.f;
  __syncthreads();

  const int nkt = (T + FA_BC - 1) / FA_BC;
  for (int kt = 0; kt < nkt; ++kt) {
    const int k0 = kt * FA_BC;
    for (int idx = tid; idx < FA_BC * hd_pad; idx += nthreads) {
      int j = idx / hd_pad, d = idx % hd_pad, gj = k0 + j;
      Ks[j * FA_TILE_HD + d] = (gj < T) ? kbase[(size_t)gj * hd_pad + d] : 0;
      Vs[j * FA_TILE_HD + d] = (gj < T) ? vbase[(size_t)gj * hd_pad + d] : 0;
    }
    __syncthreads();
    // S tile = Q . K^T (dp4a on smem operands)
    for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
      int i = idx / FA_BC, j = idx % FA_BC, gj = k0 + j;
      int dot = dp4a_i8(&Qs[i * FA_TILE_HD], &Ks[j * FA_TILE_HD], hd_pad);
      Ss[i * FA_BC + j] = (gj < T) ? (dot * sqi_s[i] * skb[gj] * softmax_scale) : -INFINITY;
    }
    __syncthreads();
    // online softmax + PV, one thread per query row
    for (int i = tid; i < FA_BR; i += nthreads) {
      if (q0 + i >= T) continue;
      float mprev = m_s[i], mcur = mprev;
      for (int j = 0; j < FA_BC; ++j) mcur = fmaxf(mcur, Ss[i * FA_BC + j]);
      float alpha = __expf(mprev - mcur);
      float lcur = l_s[i] * alpha;
      float* Orow = &Os[i * FA_TILE_HD];
      for (int d = 0; d < hd; ++d) Orow[d] *= alpha;
      for (int j = 0; j < FA_BC; ++j) {
        float p = __expf(Ss[i * FA_BC + j] - mcur);
        lcur += p;
        const int8_t* vr = &Vs[j * FA_TILE_HD];
        for (int d = 0; d < hd; ++d) Orow[d] += p * (float)vr[d];
      }
      m_s[i] = mcur; l_s[i] = lcur;
    }
    __syncthreads();
  }
  for (int i = tid; i < FA_BR; i += nthreads) {
    int gi = q0 + i; if (gi >= T) continue;
    float invl = (l_s[i] > 0.f) ? (1.f / l_s[i]) : 0.f;
    float* Orow = &Os[i * FA_TILE_HD];
    for (int d = 0; d < hd; ++d)
      out[(size_t)(nh * T + gi) * hd + d] = __float2half(Orow[d] * invl * svb[d]);
  }
}

// =========================================================================
// Tensor-core flash kernel: mma.m16n8k32.s8 for QKᵀ and PV, softmax on the
// smem S tile (no global T×T). One warp per (n,head,16-query tile).
//   BR=16 (mma M), BC=32 (key tile = mma K for PV, 4 mma N-tiles for QKᵀ).
// Requires hd_pad<=64, T%16==0, hd%8==0 (all int8-path churches blocks qualify).
// Fragment mapping matches mma_smoke (validated exact).
// =========================================================================
#define FA_MMA_BR 16
#define FA_MMA_BC 32             // key tile (multiple of 32); 32 gave best occupancy/softmax balance
#define FA_MMA_MAXHD 64
#define FA_MMA_WARPS 4            // warps per CTA; each handles its own 16-query tile
#define FA_MMA_MAXNT 8           // max hd/8 N-tiles (hd<=64)

// Multi-warp CTA: FA_MMA_WARPS warps share the K/V smem tiles (loaded once per CTA
// and reused by all warps -> higher occupancy). Each warp owns a 16-query tile, keeps
// its O accumulator in registers (fp32 fragments), and runs its own QKᵀ / softmax / PV.
__global__ void flash_attn_int8_mma_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hd_pad, float softmax_scale) {
  const int nh = blockIdx.x;
  const int w = threadIdx.x >> 5;              // warp id in CTA
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int q0 = (blockIdx.y * FA_MMA_WARPS + w) * FA_MMA_BR;   // this warp's query tile

  const int8_t* kb = k + (size_t)nh * T * hd_pad;
  const int8_t* vb = v + (size_t)nh * T * hd_pad;
  const float*  sqb = sq + (size_t)nh * T;
  const float*  skb = sk + (size_t)nh * T;
  const float*  svb = sv + (size_t)nh * hd;

  __shared__ int8_t Ks[FA_MMA_BC * FA_MMA_MAXHD];              // shared across warps
  __shared__ int8_t Vs[FA_MMA_MAXHD * FA_MMA_BC];             // transposed Vs[d*BC+j]=V[kt+j][d]
  __shared__ int8_t Qs[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_MAXHD];
  __shared__ float  Ss[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_BC];
  __shared__ int8_t Ps[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_BC];
  __shared__ float  mrow[FA_MMA_WARPS * FA_MMA_BR], lrow[FA_MMA_WARPS * FA_MMA_BR], arow[FA_MMA_WARPS * FA_MMA_BR];

  int8_t* Qsw = &Qs[w * FA_MMA_BR * hd_pad];
  float*  Ssw = &Ss[w * FA_MMA_BR * FA_MMA_BC];
  int8_t* Psw = &Ps[w * FA_MMA_BR * FA_MMA_BC];
  float*  mw = &mrow[w * FA_MMA_BR];
  float*  lw = &lrow[w * FA_MMA_BR];
  float*  aw = &arow[w * FA_MMA_BR];

  for (int idx = lane; idx < FA_MMA_BR * hd_pad; idx += 32) {
    int row = idx / hd_pad, col = idx % hd_pad, gq = q0 + row;
    Qsw[idx] = (gq < T) ? q[(size_t)nh * T * hd_pad + (size_t)gq * hd_pad + col] : 0;
  }
  if (lane < FA_MMA_BR) { mw[lane] = -INFINITY; lw[lane] = 0.f; }

  float Oreg[FA_MMA_MAXNT * 4];
#pragma unroll
  for (int i = 0; i < FA_MMA_MAXNT * 4; ++i) Oreg[i] = 0.f;
  const int n_nt2 = hd / 8;
  const int nkt = hd_pad / 32;
  __syncthreads();

  for (int kt = 0; kt < T; kt += FA_MMA_BC) {
    for (int idx = threadIdx.x; idx < FA_MMA_BC * hd_pad; idx += blockDim.x) {
      int j = idx / hd_pad, d = idx % hd_pad, gj = kt + j;
      Ks[j * hd_pad + d] = (gj < T) ? kb[(size_t)gj * hd_pad + d] : 0;
    }
    for (int idx = threadIdx.x; idx < hd * FA_MMA_BC; idx += blockDim.x) {
      int d = idx / FA_MMA_BC, j = idx % FA_MMA_BC, gj = kt + j;
      Vs[d * FA_MMA_BC + j] = (gj < T) ? vb[(size_t)gj * hd_pad + d] : 0;
    }
    __syncthreads();

    // ---- QKᵀ ----
    for (int nt = 0; nt < FA_MMA_BC / 8; ++nt) {
      int acc[4] = {0, 0, 0, 0};
      for (int ks = 0; ks < nkt; ++ks) {
        int base = ks * 32;
        unsigned a[4], b[2];
        a[0] = *(const int*)&Qsw[(gid)     * hd_pad + base + tig * 4];
        a[1] = *(const int*)&Qsw[(gid + 8) * hd_pad + base + tig * 4];
        a[2] = *(const int*)&Qsw[(gid)     * hd_pad + base + tig * 4 + 16];
        a[3] = *(const int*)&Qsw[(gid + 8) * hd_pad + base + tig * 4 + 16];
        b[0] = *(const int*)&Ks[(nt * 8 + gid) * hd_pad + base + tig * 4];
        b[1] = *(const int*)&Ks[(nt * 8 + gid) * hd_pad + base + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, a, b);
      }
      float sqi0 = sqb[q0 + gid], sqi1 = sqb[q0 + gid + 8];
      int c0 = nt * 8 + tig * 2, c1 = c0 + 1;
      float sk0 = (kt + c0 < T) ? skb[kt + c0] : 0.f, sk1 = (kt + c1 < T) ? skb[kt + c1] : 0.f;
      Ssw[gid * FA_MMA_BC + c0]       = (kt + c0 < T) ? acc[0] * sqi0 * sk0 * softmax_scale : -INFINITY;
      Ssw[gid * FA_MMA_BC + c1]       = (kt + c1 < T) ? acc[1] * sqi0 * sk1 * softmax_scale : -INFINITY;
      Ssw[(gid + 8) * FA_MMA_BC + c0] = (kt + c0 < T) ? acc[2] * sqi1 * sk0 * softmax_scale : -INFINITY;
      Ssw[(gid + 8) * FA_MMA_BC + c1] = (kt + c1 < T) ? acc[3] * sqi1 * sk1 * softmax_scale : -INFINITY;
    }
    __syncwarp();

    // ---- online softmax + requant P (one lane per query row) ----
    if (lane < FA_MMA_BR) {
      int r = lane;
      float mprev = mw[r], mcur = mprev;
      for (int j = 0; j < FA_MMA_BC; ++j) mcur = fmaxf(mcur, Ssw[r * FA_MMA_BC + j]);
      float alpha = __expf(mprev - mcur);
      float lcur = lw[r] * alpha;
      for (int j = 0; j < FA_MMA_BC; ++j) {
        float p = __expf(Ssw[r * FA_MMA_BC + j] - mcur);
        lcur += p;
        int pq = (int)(p * 127.f + 0.5f);
        Psw[r * FA_MMA_BC + j] = (int8_t)(pq > 127 ? 127 : pq);
      }
      mw[r] = mcur; lw[r] = lcur; aw[r] = alpha;
    }
    __syncwarp();

    // ---- rescale O (registers) by per-row alpha, then PV accumulate ----
    float a_g = aw[gid], a_g8 = aw[gid + 8];
    for (int nt2 = 0; nt2 < n_nt2; ++nt2) {
      Oreg[nt2 * 4 + 0] *= a_g;  Oreg[nt2 * 4 + 1] *= a_g;
      Oreg[nt2 * 4 + 2] *= a_g8; Oreg[nt2 * 4 + 3] *= a_g8;
      int acc[4] = {0, 0, 0, 0};
      for (int ks = 0; ks < FA_MMA_BC / 32; ++ks) {          // K = BC (one or more mma k-steps)
        int koff = ks * 32;
        unsigned a[4], b[2];
        a[0] = *(const int*)&Psw[(gid)     * FA_MMA_BC + koff + tig * 4];
        a[1] = *(const int*)&Psw[(gid + 8) * FA_MMA_BC + koff + tig * 4];
        a[2] = *(const int*)&Psw[(gid)     * FA_MMA_BC + koff + tig * 4 + 16];
        a[3] = *(const int*)&Psw[(gid + 8) * FA_MMA_BC + koff + tig * 4 + 16];
        b[0] = *(const int*)&Vs[(nt2 * 8 + gid) * FA_MMA_BC + koff + tig * 4];
        b[1] = *(const int*)&Vs[(nt2 * 8 + gid) * FA_MMA_BC + koff + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, a, b);
      }
      int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
      Oreg[nt2 * 4 + 0] += (1.f / 127.f) * svb[d0] * acc[0];
      Oreg[nt2 * 4 + 1] += (1.f / 127.f) * svb[d1] * acc[1];
      Oreg[nt2 * 4 + 2] += (1.f / 127.f) * svb[d0] * acc[2];
      Oreg[nt2 * 4 + 3] += (1.f / 127.f) * svb[d1] * acc[3];
    }
    __syncthreads();   // all warps done with shared Ks/Vs before reload
  }

  float invl_g  = (lw[gid]     > 0.f) ? 1.f / lw[gid]     : 0.f;
  float invl_g8 = (lw[gid + 8] > 0.f) ? 1.f / lw[gid + 8] : 0.f;
  for (int nt2 = 0; nt2 < n_nt2; ++nt2) {
    int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
    int gi0 = q0 + gid, gi8 = q0 + gid + 8;
    out[(size_t)(nh * T + gi0) * hd + d0] = __float2half(Oreg[nt2 * 4 + 0] * invl_g);
    out[(size_t)(nh * T + gi0) * hd + d1] = __float2half(Oreg[nt2 * 4 + 1] * invl_g);
    out[(size_t)(nh * T + gi8) * hd + d0] = __float2half(Oreg[nt2 * 4 + 2] * invl_g8);
    out[(size_t)(nh * T + gi8) * hd + d1] = __float2half(Oreg[nt2 * 4 + 3] * invl_g8);
  }
}

// ---- mma fragment-mapping smoke test: C[16,N] = A[16,K] . B[N,K]^T via
// m16n8k32.s8 tensor cores, plain (non-swizzled) smem layout. Validates the
// documented .s8 fragment thread->element mapping before the flash kernel uses it. ----
__global__ void mma_smoke_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 int* __restrict__ C, int K, int N) {
  int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
  extern __shared__ int8_t sm[];
  int8_t* As = sm;
  int8_t* Bs = sm + 16 * K;
  for (int i = lane; i < 16 * K; i += 32) As[i] = A[i];
  for (int i = lane; i < N * K; i += 32) Bs[i] = B[i];
  __syncwarp();
  int nnt = N / 8, nkt = K / 32;
  for (int nt = 0; nt < nnt; ++nt) {
    int acc[4] = {0, 0, 0, 0};
    for (int ks = 0; ks < nkt; ++ks) {
      int base = ks * 32;
      unsigned a[4], b[2];
      a[0] = *(const int*)&As[(gid)     * K + base + tig * 4];
      a[1] = *(const int*)&As[(gid + 8) * K + base + tig * 4];
      a[2] = *(const int*)&As[(gid)     * K + base + tig * 4 + 16];
      a[3] = *(const int*)&As[(gid + 8) * K + base + tig * 4 + 16];
      b[0] = *(const int*)&Bs[(nt * 8 + gid) * K + base + tig * 4];
      b[1] = *(const int*)&Bs[(nt * 8 + gid) * K + base + tig * 4 + 16];
      modiff_mma_m16n8k32(acc, a, b);
    }
    C[(gid)     * N + nt * 8 + tig * 2 + 0] = acc[0];
    C[(gid)     * N + nt * 8 + tig * 2 + 1] = acc[1];
    C[(gid + 8) * N + nt * 8 + tig * 2 + 0] = acc[2];
    C[(gid + 8) * N + nt * 8 + tig * 2 + 1] = acc[3];
  }
}

torch::Tensor mma_smoke(torch::Tensor A, torch::Tensor B) {
  // A [16,K] int8, B [N,K] int8 -> C [16,N] int32 = A . B^T
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kChar && B.dtype() == torch::kChar);
  int K = A.size(1), N = B.size(0);
  TORCH_CHECK(A.size(0) == 16 && K % 32 == 0 && N % 8 == 0, "need M=16, K%32==0, N%8==0");
  auto C = torch::zeros({16, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
  size_t smem = (size_t)(16 * K + N * K);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mma_smoke_kernel<<<1, 32, smem, stream>>>(A.contiguous().data_ptr<int8_t>(),
      B.contiguous().data_ptr<int8_t>(), C.data_ptr<int>(), K, N);
  return C;
}

torch::Tensor flash_attn_int8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                              double softmax_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flash_attn_int8: q/k/v must be CUDA");
  TORCH_CHECK(q.dim() == 4, "q must be [N,H,T,hd_pad]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
  const int N = q.size(0), H = q.size(1), T = q.size(2), hd_pad = q.size(3);
  const int hd = sv.size(-1);
  TORCH_CHECK(hd <= MODIFF_FA_MAX_HD, "head_dim exceeds MODIFF_FA_MAX_HD");
  TORCH_CHECK(hd_pad % 4 == 0, "hd_pad must be a multiple of 4 for __dp4a");

  auto out = torch::empty({N, H, T, hd},
                          torch::TensorOptions().dtype(torch::kFloat16).device(q.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (hd_pad <= FA_MMA_MAXHD && (T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0) {
    // tensor-core mma flash path (all int8-path churches attention blocks)
    dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
    flash_attn_int8_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
        q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), v.data_ptr<int8_t>(),
        sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, H, T, hd, hd_pad, (float)softmax_scale);
  } else if (hd_pad <= FA_TILE_HD) {
    // dp4a tiled fallback (e.g. T not a multiple of 16)
    dim3 grid(N * H, (T + FA_BR - 1) / FA_BR);
    flash_attn_int8_tiled_kernel<<<grid, FA_TILED_THREADS, 0, stream>>>(
        q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), v.data_ptr<int8_t>(),
        sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, H, T, hd, hd_pad, (float)softmax_scale);
  } else {
    // naive fallback (hd_pad > 64, e.g. head_dim 96 tiny-T blocks)
    const int threads = MODIFF_FA_THREADS;
    const int blocks = N * H * T;
    const size_t smem = (size_t)threads * sizeof(float) + (size_t)hd * sizeof(float) + (size_t)hd_pad;
    flash_attn_int8_naive_kernel<<<blocks, threads, smem, stream>>>(
        q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), v.data_ptr<int8_t>(),
        sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, H, T, hd, hd_pad, (float)softmax_scale);
  }
  return out;
}
