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
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "common.cuh"
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
#define FA_MMA_BC 64             // key tile (multiple of 32); 64 halves tile count/syncs vs 32
#define FA_MMA_MAXHD 64
#define FA_MMA_WARPS 4            // warps per CTA; each handles its own 16-query tile (share K/V smem)
#define FA_MMA_MAXNT 8           // max hd/8 N-tiles (hd<=64)

// Multi-warp CTA: FA_MMA_WARPS warps share the K/V smem tiles (loaded once per CTA
// and reused by all warps -> higher occupancy). Each warp owns a 16-query tile, keeps
// its O accumulator in registers (fp32 fragments), and runs its own QKᵀ / softmax / PV.
__global__ void flash_attn_int8_mma_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hd_pad, float softmax_scale,
    int8_t* __restrict__ out_q, float proj_inv_scale, int qout_stride) {
  // out_q != nullptr => fused proj-quantize store: emit int8 token-major [b*T, qout_stride]
  // (qout_stride = C = H*hd), quantized by proj_inv_scale, instead of the fp16 head-major `out`.
  const int nh = blockIdx.x;
  const int w = threadIdx.x >> 5;              // warp id in CTA
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int q0 = (blockIdx.y * FA_MMA_WARPS + w) * FA_MMA_BR;   // this warp's query tile

  const int8_t* kb = k + (size_t)nh * T * hd_pad;
  const int8_t* vb = v + (size_t)nh * T * hd_pad;
  const float*  sqb = sq + (size_t)nh * T;
  const float*  skb = sk + (size_t)nh * T;
  const float*  svb = sv + (size_t)nh * hd;

  // Triple-buffered cp.async tiles (FA_STAGES-deep pipeline). V is taken PRE-TRANSPOSED
  // [N,H,hd_pad,T] so both K ([BC,hd_pad]) and V ([hd_pad,BC]) tiles are HBM-contiguous.
  #define FA_STAGES 2   // 2-stage double-buffer (3 tried: slower at T1024, occupancy-bound)
  __shared__ int8_t Ks[FA_STAGES][FA_MMA_BC * FA_MMA_MAXHD];
  __shared__ int8_t Vs[FA_STAGES][FA_MMA_MAXHD * FA_MMA_BC];  // Vs[buf][d*BC+j] = V[kt+j][d]
  __shared__ int8_t Qs[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_MAXHD];
  __shared__ int8_t Ps[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_BC];

  int8_t* Qsw = &Qs[w * FA_MMA_BR * hd_pad];
  int8_t* Psw = &Ps[w * FA_MMA_BR * FA_MMA_BC];

  for (int idx = lane; idx < FA_MMA_BR * hd_pad; idx += 32) {
    int row = idx / hd_pad, col = idx % hd_pad, gq = q0 + row;
    Qsw[idx] = (gq < T) ? q[(size_t)nh * T * hd_pad + (size_t)gq * hd_pad + col] : 0;
  }

  // Per-lane running softmax state (registers, no smem). Each lane owns rows gid & gid+8;
  // the 4 lanes sharing a gid (tig=0..3) agree on m/l after the __shfl_xor reduction.
  float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
  float Oreg[FA_MMA_MAXNT * 4];
#pragma unroll
  for (int i = 0; i < FA_MMA_MAXNT * 4; ++i) Oreg[i] = 0.f;
  const int n_nt2 = hd / 8;
  const int nkt = hd_pad / 32;
  const int NNT = FA_MMA_BC / 8;                 // QKᵀ N-tiles (=4 for BC=32)
  const float sqi0 = sqb[q0 + gid], sqi1 = sqb[q0 + gid + 8];
  const int NKV = T / FA_MMA_BC;                  // key tiles (T % BC == 0 on the mma path)
  const int KV16 = (FA_MMA_BC * hd_pad) / 16;     // 16B chunks per K tile (== per V tile)

  // cp.async a K tile ([BC,hd_pad] from k[nh]) and V tile ([hd_pad,BC] from pre-T v[nh]) into buf.
  auto load_kv = [&](int buf, int kt) {
    for (int c = threadIdx.x; c < KV16; c += blockDim.x) {
      int off = c * 16;
      // K: row-major [BC,hd_pad], HBM kb[(kt+j)*hd_pad + d]
      modiff_cp_async_cg(modiff_smem_ptr(&Ks[buf][off]),
                         (const uint4*)(kb + (size_t)kt * hd_pad + off), true);
      // V (pre-transposed [hd_pad,T]): row-major [hd_pad,BC], HBM vb[d*T + kt+j]
      int d = off / FA_MMA_BC, j = off % FA_MMA_BC;
      modiff_cp_async_cg(modiff_smem_ptr(&Vs[buf][off]),
                         (const uint4*)(vb + (size_t)d * T + kt + j), true);
    }
    __pipeline_commit();
  };

  // prime FA_STAGES-1 tiles (commits in flight)
#pragma unroll
  for (int s = 0; s < FA_STAGES - 1; ++s) if (s < NKV) load_kv(s % FA_STAGES, s * FA_MMA_BC);

  for (int ktile = 0; ktile < NKV; ++ktile) {
    const int kt = ktile * FA_MMA_BC;
    const int buf = ktile % FA_STAGES;
    __pipeline_wait_prior(FA_STAGES - 2);   // wait until oldest (tile ktile) has arrived
    __syncthreads();                        // all warps done with prev tile's buffer; tile ktile visible
    const int nxt = ktile + (FA_STAGES - 1);
    if (nxt < NKV) load_kv(nxt % FA_STAGES, nxt * FA_MMA_BC);  // prefetch tile ktile+STAGES-1
    int8_t* Ksb = Ks[buf];
    int8_t* Vsb = Vs[buf];

    // ---- QKᵀ: keep the [BR x BC] score tile in registers (Sreg[nt][0..3]) ----
    // Sreg[nt] = {row gid col c0, row gid col c1, row gid+8 col c0, row gid+8 col c1}
    float Sreg[FA_MMA_BC / 8][4];
    for (int nt = 0; nt < NNT; ++nt) {
      int acc[4] = {0, 0, 0, 0};
      for (int ks = 0; ks < nkt; ++ks) {
        int base = ks * 32;
        unsigned a[4], b[2];
        a[0] = *(const int*)&Qsw[(gid)     * hd_pad + base + tig * 4];
        a[1] = *(const int*)&Qsw[(gid + 8) * hd_pad + base + tig * 4];
        a[2] = *(const int*)&Qsw[(gid)     * hd_pad + base + tig * 4 + 16];
        a[3] = *(const int*)&Qsw[(gid + 8) * hd_pad + base + tig * 4 + 16];
        b[0] = *(const int*)&Ksb[(nt * 8 + gid) * hd_pad + base + tig * 4];
        b[1] = *(const int*)&Ksb[(nt * 8 + gid) * hd_pad + base + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, a, b);
      }
      int c0 = nt * 8 + tig * 2, c1 = c0 + 1;
      float sk0 = (kt + c0 < T) ? skb[kt + c0] : 0.f, sk1 = (kt + c1 < T) ? skb[kt + c1] : 0.f;
      Sreg[nt][0] = (kt + c0 < T) ? acc[0] * sqi0 * sk0 * softmax_scale : -INFINITY;
      Sreg[nt][1] = (kt + c1 < T) ? acc[1] * sqi0 * sk1 * softmax_scale : -INFINITY;
      Sreg[nt][2] = (kt + c0 < T) ? acc[2] * sqi1 * sk0 * softmax_scale : -INFINITY;
      Sreg[nt][3] = (kt + c1 < T) ? acc[3] * sqi1 * sk1 * softmax_scale : -INFINITY;
    }

    // ---- register-parallel online softmax (all 32 lanes active) ----
    // local max over this lane's 8 cols for each of its 2 rows
    float lm0 = -INFINITY, lm1 = -INFINITY;
    for (int nt = 0; nt < NNT; ++nt) {
      lm0 = fmaxf(lm0, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      lm1 = fmaxf(lm1, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    // reduce across the 4 tig-lanes of each gid-group (lanes differ in low 2 bits)
    lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1)); lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
    lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1)); lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
    float mcur0 = fmaxf(m_run0, lm0), mcur1 = fmaxf(m_run1, lm1);
    float a_g  = __expf(m_run0 - mcur0);      // per-row alpha (registers)
    float a_g8 = __expf(m_run1 - mcur1);
    // exp P, requant to int8 (fixed scale 127), partial row-sum
    float ls0 = 0.f, ls1 = 0.f;
    for (int nt = 0; nt < NNT; ++nt) {
      int c0 = nt * 8 + tig * 2, c1 = c0 + 1;
      float p00 = __expf(Sreg[nt][0] - mcur0), p01 = __expf(Sreg[nt][1] - mcur0);
      float p10 = __expf(Sreg[nt][2] - mcur1), p11 = __expf(Sreg[nt][3] - mcur1);
      ls0 += p00 + p01; ls1 += p10 + p11;
      int q00 = (int)(p00 * 127.f + 0.5f), q01 = (int)(p01 * 127.f + 0.5f);
      int q10 = (int)(p10 * 127.f + 0.5f), q11 = (int)(p11 * 127.f + 0.5f);
      Psw[gid       * FA_MMA_BC + c0] = (int8_t)(q00 > 127 ? 127 : q00);
      Psw[gid       * FA_MMA_BC + c1] = (int8_t)(q01 > 127 ? 127 : q01);
      Psw[(gid + 8) * FA_MMA_BC + c0] = (int8_t)(q10 > 127 ? 127 : q10);
      Psw[(gid + 8) * FA_MMA_BC + c1] = (int8_t)(q11 > 127 ? 127 : q11);
    }
    // reduce partial sums across the tig-group, combine with running sum (rescaled)
    ls0 += __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
    l_run0 = l_run0 * a_g + ls0; l_run1 = l_run1 * a_g8 + ls1;
    m_run0 = mcur0; m_run1 = mcur1;
    __syncwarp();

    // ---- rescale O (registers) by per-row alpha, then PV accumulate ----
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
        b[0] = *(const int*)&Vsb[(nt2 * 8 + gid) * FA_MMA_BC + koff + tig * 4];
        b[1] = *(const int*)&Vsb[(nt2 * 8 + gid) * FA_MMA_BC + koff + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, a, b);
      }
      int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
      Oreg[nt2 * 4 + 0] += (1.f / 127.f) * svb[d0] * acc[0];
      Oreg[nt2 * 4 + 1] += (1.f / 127.f) * svb[d1] * acc[1];
      Oreg[nt2 * 4 + 2] += (1.f / 127.f) * svb[d0] * acc[2];
      Oreg[nt2 * 4 + 3] += (1.f / 127.f) * svb[d1] * acc[3];
    }
    // buffer reuse safety handled by next iter's __pipeline_wait_prior + __syncthreads
  }

  float invl_g  = (l_run0 > 0.f) ? 1.f / l_run0 : 0.f;
  float invl_g8 = (l_run1 > 0.f) ? 1.f / l_run1 : 0.f;
  const int h = nh % H, n = nh / H;           // (n,h) for the token-major fused-quant store
  for (int nt2 = 0; nt2 < n_nt2; ++nt2) {
    int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
    int gi0 = q0 + gid, gi8 = q0 + gid + 8;
    float o00 = Oreg[nt2*4+0]*invl_g,  o01 = Oreg[nt2*4+1]*invl_g;
    float o10 = Oreg[nt2*4+2]*invl_g8, o11 = Oreg[nt2*4+3]*invl_g8;
    if (out_q == nullptr) {                    // default: fp16 head-major [N,H,T,hd]
      out[(size_t)(nh * T + gi0) * hd + d0] = __float2half(o00);
      out[(size_t)(nh * T + gi0) * hd + d1] = __float2half(o01);
      out[(size_t)(nh * T + gi8) * hd + d0] = __float2half(o10);
      out[(size_t)(nh * T + gi8) * hd + d1] = __float2half(o11);
    } else {                                   // fused: int8 token-major, quantized by proj scale
      int c0 = h * hd + d0, c1 = h * hd + d1;  // round through fp16 first -> bit-matches
      int q00 = __float2int_rn(__half2float(__float2half(o00)) * proj_inv_scale);  // quantize_attn_out_int8
      int q01 = __float2int_rn(__half2float(__float2half(o01)) * proj_inv_scale);
      int q10 = __float2int_rn(__half2float(__float2half(o10)) * proj_inv_scale);
      int q11 = __float2int_rn(__half2float(__float2half(o11)) * proj_inv_scale);
      q00 = q00>127?127:(q00<-127?-127:q00); q01 = q01>127?127:(q01<-127?-127:q01);
      q10 = q10>127?127:(q10<-127?-127:q10); q11 = q11>127?127:(q11<-127?-127:q11);
      out_q[(size_t)(n * T + gi0) * qout_stride + c0] = (int8_t)q00;
      out_q[(size_t)(n * T + gi0) * qout_stride + c1] = (int8_t)q01;
      out_q[(size_t)(n * T + gi8) * qout_stride + c0] = (int8_t)q10;
      out_q[(size_t)(n * T + gi8) * qout_stride + c1] = (int8_t)q11;
    }
  }
}

// =========================================================================
// int4 flash: QKᵀ in int4 (mma.m16n8k64.s4), PV kept in int8 (P∈[0,1] at int4 is
// too coarse). q4,k4 packed int4 [N,H,T,hdp4/2] (hdp4 mult of 64); v int8 PRE-
// TRANSPOSED [N,H,hdp_v,T] (hdp_v=pad(hd,32)). Documented NEGATIVE result: at hd=24
// the int4 K-pad (24->64, 62% waste) makes QKᵀ no faster than int8, so this loses to
// fp16 flash like the int8 kernel. Same register-parallel softmax + cp.async as int8.
// =========================================================================
__global__ void flash_attn_int4_mma_kernel(
    const int8_t* __restrict__ q4, const int8_t* __restrict__ k4, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hdp4, int hdp_v, float softmax_scale,
    int8_t* __restrict__ out_q, float proj_inv_scale, int qout_bstride) {
  // out_q != nullptr => fused proj-quantize store: emit packed int4 token-major [b*T, qout_bstride]
  // (qout_bstride = K_pad/2 bytes/row), quantized by proj_inv_scale. K-pad tail bytes are pre-zeroed
  // by the caller (torch::zeros), so this only writes the real channel-pair bytes.
  const int nh = blockIdx.x;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int q0 = (blockIdx.y * FA_MMA_WARPS + w) * FA_MMA_BR;
  const int rowb4 = hdp4 / 2;                          // packed bytes per Q/K row
  const int kb_stride = (size_t)T * rowb4;

  const int8_t* kb = k4 + (size_t)nh * T * rowb4;
  const int8_t* vb = v + (size_t)nh * hdp_v * T;       // pre-transposed [hdp_v, T]
  const float*  sqb = sq + (size_t)nh * T;
  const float*  skb = sk + (size_t)nh * T;
  const float*  svb = sv + (size_t)nh * hd;

  __shared__ int8_t Ks[2][FA_MMA_BC * 32];             // packed int4 rows (<=32 bytes, hdp4<=64)
  __shared__ int8_t Vs[2][FA_MMA_MAXHD * FA_MMA_BC];   // int8 transposed Vs[buf][d*BC+j]
  __shared__ int8_t Qs[FA_MMA_WARPS * FA_MMA_BR * 32];
  __shared__ int8_t Ps[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_BC];

  int8_t* Qsw = &Qs[w * FA_MMA_BR * rowb4];
  int8_t* Psw = &Ps[w * FA_MMA_BR * FA_MMA_BC];

  for (int idx = lane; idx < FA_MMA_BR * rowb4; idx += 32) {
    int row = idx / rowb4, col = idx % rowb4, gq = q0 + row;
    Qsw[idx] = (gq < T) ? q4[(size_t)nh * kb_stride + (size_t)gq * rowb4 + col] : 0;
  }

  float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
  float Oreg[FA_MMA_MAXNT * 4];
#pragma unroll
  for (int i = 0; i < FA_MMA_MAXNT * 4; ++i) Oreg[i] = 0.f;
  const int n_nt2 = hd / 8;
  const int nkt4 = hdp4 / 64;                           // int4 k-steps (K=64 each)
  const int NNT = FA_MMA_BC / 8;
  const float sqi0 = sqb[q0 + gid], sqi1 = sqb[q0 + gid + 8];
  const int NKV = T / FA_MMA_BC;
  const int K16 = (FA_MMA_BC * rowb4) / 16;             // 16B chunks per K tile (packed)
  const int V16 = (FA_MMA_BC * hdp_v) / 16;             // 16B chunks per V tile

  auto load_kv = [&](int buf, int kt) {
    for (int c = threadIdx.x; c < K16; c += blockDim.x) {   // K packed [BC,rowb4]
      int off = c * 16;
      modiff_cp_async_cg(modiff_smem_ptr(&Ks[buf][off]),
                         (const uint4*)(kb + (size_t)kt * rowb4 + off), true);
    }
    for (int c = threadIdx.x; c < V16; c += blockDim.x) {   // V int8 transposed [hdp_v,BC] from [hdp_v,T]
      int off = c * 16, d = off / FA_MMA_BC, j = off % FA_MMA_BC;
      modiff_cp_async_cg(modiff_smem_ptr(&Vs[buf][off]),
                         (const uint4*)(vb + (size_t)d * T + kt + j), true);
    }
    __pipeline_commit();
  };

  load_kv(0, 0);
  for (int ktile = 0; ktile < NKV; ++ktile) {
    const int kt = ktile * FA_MMA_BC;
    const int buf = ktile & 1;
    __pipeline_wait_prior(0);
    __syncthreads();
    if (ktile + 1 < NKV) load_kv((ktile + 1) & 1, (ktile + 1) * FA_MMA_BC);
    int8_t* Ksb = Ks[buf];
    int8_t* Vsb = Vs[buf];

    // ---- QKᵀ (int4, m16n8k64.s4) -> Sreg ----
    float Sreg[FA_MMA_BC / 8][4];
    for (int nt = 0; nt < NNT; ++nt) {
      int acc[4] = {0, 0, 0, 0};
      for (int ks = 0; ks < nkt4; ++ks) {
        int base = ks * 32;                              // 64 int4 = 32 bytes
        unsigned a[4], b[2];
        a[0] = *(const int*)&Qsw[(gid)     * rowb4 + base + tig * 4];
        a[1] = *(const int*)&Qsw[(gid + 8) * rowb4 + base + tig * 4];
        a[2] = *(const int*)&Qsw[(gid)     * rowb4 + base + tig * 4 + 16];
        a[3] = *(const int*)&Qsw[(gid + 8) * rowb4 + base + tig * 4 + 16];
        b[0] = *(const int*)&Ksb[(nt * 8 + gid) * rowb4 + base + tig * 4];
        b[1] = *(const int*)&Ksb[(nt * 8 + gid) * rowb4 + base + tig * 4 + 16];
        modiff_mma_m16n8k64_s4(acc, a, b);
      }
      int c0 = nt * 8 + tig * 2, c1 = c0 + 1;
      float sk0 = skb[kt + c0], sk1 = skb[kt + c1];
      Sreg[nt][0] = acc[0] * sqi0 * sk0 * softmax_scale;
      Sreg[nt][1] = acc[1] * sqi0 * sk1 * softmax_scale;
      Sreg[nt][2] = acc[2] * sqi1 * sk0 * softmax_scale;
      Sreg[nt][3] = acc[3] * sqi1 * sk1 * softmax_scale;
    }

    // ---- register-parallel online softmax (identical to int8) ----
    float lm0 = -INFINITY, lm1 = -INFINITY;
    for (int nt = 0; nt < NNT; ++nt) {
      lm0 = fmaxf(lm0, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      lm1 = fmaxf(lm1, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1)); lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
    lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1)); lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
    float mcur0 = fmaxf(m_run0, lm0), mcur1 = fmaxf(m_run1, lm1);
    float a_g = __expf(m_run0 - mcur0), a_g8 = __expf(m_run1 - mcur1);
    float ls0 = 0.f, ls1 = 0.f;
    for (int nt = 0; nt < NNT; ++nt) {
      int c0 = nt * 8 + tig * 2, c1 = c0 + 1;
      float p00 = __expf(Sreg[nt][0] - mcur0), p01 = __expf(Sreg[nt][1] - mcur0);
      float p10 = __expf(Sreg[nt][2] - mcur1), p11 = __expf(Sreg[nt][3] - mcur1);
      ls0 += p00 + p01; ls1 += p10 + p11;
      int q00 = (int)(p00 * 127.f + 0.5f), q01 = (int)(p01 * 127.f + 0.5f);
      int q10 = (int)(p10 * 127.f + 0.5f), q11 = (int)(p11 * 127.f + 0.5f);
      Psw[gid       * FA_MMA_BC + c0] = (int8_t)(q00 > 127 ? 127 : q00);
      Psw[gid       * FA_MMA_BC + c1] = (int8_t)(q01 > 127 ? 127 : q01);
      Psw[(gid + 8) * FA_MMA_BC + c0] = (int8_t)(q10 > 127 ? 127 : q10);
      Psw[(gid + 8) * FA_MMA_BC + c1] = (int8_t)(q11 > 127 ? 127 : q11);
    }
    ls0 += __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
    l_run0 = l_run0 * a_g + ls0; l_run1 = l_run1 * a_g8 + ls1;
    m_run0 = mcur0; m_run1 = mcur1;
    __syncwarp();

    // ---- PV (int8, m16n8k32) ----
    for (int nt2 = 0; nt2 < n_nt2; ++nt2) {
      Oreg[nt2 * 4 + 0] *= a_g;  Oreg[nt2 * 4 + 1] *= a_g;
      Oreg[nt2 * 4 + 2] *= a_g8; Oreg[nt2 * 4 + 3] *= a_g8;
      int acc[4] = {0, 0, 0, 0};
      for (int ks = 0; ks < FA_MMA_BC / 32; ++ks) {
        int koff = ks * 32;
        unsigned a[4], b[2];
        a[0] = *(const int*)&Psw[(gid)     * FA_MMA_BC + koff + tig * 4];
        a[1] = *(const int*)&Psw[(gid + 8) * FA_MMA_BC + koff + tig * 4];
        a[2] = *(const int*)&Psw[(gid)     * FA_MMA_BC + koff + tig * 4 + 16];
        a[3] = *(const int*)&Psw[(gid + 8) * FA_MMA_BC + koff + tig * 4 + 16];
        b[0] = *(const int*)&Vsb[(nt2 * 8 + gid) * FA_MMA_BC + koff + tig * 4];
        b[1] = *(const int*)&Vsb[(nt2 * 8 + gid) * FA_MMA_BC + koff + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, a, b);
      }
      int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
      Oreg[nt2 * 4 + 0] += (1.f / 127.f) * svb[d0] * acc[0];
      Oreg[nt2 * 4 + 1] += (1.f / 127.f) * svb[d1] * acc[1];
      Oreg[nt2 * 4 + 2] += (1.f / 127.f) * svb[d0] * acc[2];
      Oreg[nt2 * 4 + 3] += (1.f / 127.f) * svb[d1] * acc[3];
    }
  }

  float invl_g  = (l_run0 > 0.f) ? 1.f / l_run0 : 0.f;
  float invl_g8 = (l_run1 > 0.f) ? 1.f / l_run1 : 0.f;
  const int h = nh % H, n = nh / H;           // (n,h) for the token-major fused-quant store
  for (int nt2 = 0; nt2 < n_nt2; ++nt2) {
    int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
    int gi0 = q0 + gid, gi8 = q0 + gid + 8;
    float o00 = Oreg[nt2*4+0]*invl_g,  o01 = Oreg[nt2*4+1]*invl_g;
    float o10 = Oreg[nt2*4+2]*invl_g8, o11 = Oreg[nt2*4+3]*invl_g8;
    if (out_q == nullptr) {                    // default: fp16 head-major [N,H,T,hd]
      out[(size_t)(nh * T + gi0) * hd + d0] = __float2half(o00);
      out[(size_t)(nh * T + gi0) * hd + d1] = __float2half(o01);
      out[(size_t)(nh * T + gi8) * hd + d0] = __float2half(o10);
      out[(size_t)(nh * T + gi8) * hd + d1] = __float2half(o11);
    } else {                                   // fused: packed int4 token-major, quantized by proj scale
      int c0 = h * hd + d0;                    // even (hd, d0 even) -> packed byte index = c0/2
      // round through fp16 first -> bit-matches quantize_attn_out_int4_pack (low nibble=even ch)
      int q00 = __float2int_rn(__half2float(__float2half(o00)) * proj_inv_scale); q00 = q00>7?7:(q00<-7?-7:q00);
      int q01 = __float2int_rn(__half2float(__float2half(o01)) * proj_inv_scale); q01 = q01>7?7:(q01<-7?-7:q01);
      int q10 = __float2int_rn(__half2float(__float2half(o10)) * proj_inv_scale); q10 = q10>7?7:(q10<-7?-7:q10);
      int q11 = __float2int_rn(__half2float(__float2half(o11)) * proj_inv_scale); q11 = q11>7?7:(q11<-7?-7:q11);
      out_q[(size_t)(n * T + gi0) * qout_bstride + c0/2] = (int8_t)((q00 & 0x0F) | ((q01 & 0x0F) << 4));
      out_q[(size_t)(n * T + gi8) * qout_bstride + c0/2] = (int8_t)((q10 & 0x0F) | ((q11 & 0x0F) << 4));
    }
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

// ---- ENTRYPOINT (test-only). Kernel: mma_smoke_kernel. ----
//   Op:       Attention W8A8 — mma fragment-mapping smoke test
//   Inputs:   A int8 [16,K], B int8 [N,K]
//   Outputs:  C int32 [16,N]
//   Computes: C = A[16,K] · B[N,K]ᵀ via m16n8k32.s8.s8.s32 tensor cores (plain non-swizzled smem)
//   Fuses:    none — validates the documented .s8 fragment thread→element mapping that the flash
//             kernels rely on (must match before the mma flash path is trusted exact)
//   Constraints: M=16, K%32==0, N%8==0
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

// ---- ENTRYPOINT (fused flash attention, int8). Dispatches to one of three kernels below. ----
//   Op:       Attention W8A8 — fused flash
//   Inputs:   q,k,v int8 [N,H,T,hd_pad] (head_dim padded to a multiple of 4; padded lanes zero →
//             exact dot product), sq,sk f32 [N,H,T] (per-token dequant, q_fp≈q_i8·sq), sv f32
//             [N,H,hd] (per-channel/head-dim V dequant), softmax_scale double (= 1/sqrt(hd))
//   Outputs:  out fp16 [N,H,T,hd]
//   Computes: out = softmax(diag(sq)·(QKᵀ)·diag(sk)·scale) · V, with per-channel sv applied to the V
//             contraction; running softmax (max/exp/normalize) always in fp32
//   Fuses:    flash fuses QKᵀ + online softmax + PV, and the scores stay in SRAM/registers — the
//             [N,H,T,T] score matrix is never written to HBM (vs the 3x round-trip of the
//             materialized attn_qk/softmax/attn_av path). FAST PATH = flash_attn_int8_mma_kernel:
//             tensor-core m16n8k32.s8 QKᵀ AND PV, register-parallel online softmax (P requant to
//             int8, fixed scale 127), 2-stage cp.async, BR=16/BC=64 key tile, V fed PRE-TRANSPOSED
//             [N,H,hd_pad,T]; 4 warps/CTA share the K/V smem. Fallbacks (slower, correctness):
//             flash_attn_int8_tiled_kernel (dp4a on smem tiles, when T not a multiple of 64,
//             hd_pad<=64) and flash_attn_int8_naive_kernel (dp4a, one CTA/query row, any head_dim,
//             used for hd_pad>64 e.g. head_dim 96).
//   Constraints: hd<=96 (MODIFF_FA_MAX_HD), hd_pad%4==0; mma path additionally needs hd_pad<=64,
//             T%64==0 (T%(FA_MMA_WARPS·FA_MMA_BR)), hd%8==0, (BC·hd_pad)%16==0
//   vs fp16:  int8 flash 2.73x / int4 flash 2.78x vs fp16 MATH at hd24/T1024; ~1.2x at hd48/T256;
//             ~0.95-1.16x at hd48/T64. vs fp16 FlashAttention-2: 0.20-0.50x (loses). Fusion (BC=64
//             key tile, 2-stage cp.async, register online softmax) removes the 3x HBM round-trip
//             over the [BH,T,T] score matrix that the materialized path pays.
// V-pre-transposed variant: vt is ALREADY [N,H,hd_pad,T] (e.g. straight from quantize_attn_qkv),
// so we skip the internal v.transpose(2,3).contiguous() copy. mma tensor-core path only.
torch::Tensor flash_attn_int8_vt(torch::Tensor q, torch::Tensor k, torch::Tensor vt,
                                 torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                 double softmax_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && vt.is_cuda(), "flash_attn_int8_vt: q/k/vt must be CUDA");
  TORCH_CHECK(q.dim() == 4 && vt.dim() == 4, "q [N,H,T,hd_pad], vt [N,H,hd_pad,T]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && vt.is_contiguous(), "q/k/vt must be contiguous");
  const int N = q.size(0), H = q.size(1), T = q.size(2), hd_pad = q.size(3);
  const int hd = sv.size(-1);
  TORCH_CHECK(hd_pad <= FA_MMA_MAXHD && (T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0
              && (FA_MMA_BC * hd_pad) % 16 == 0, "flash_attn_int8_vt: mma-eligible shapes only");
  TORCH_CHECK(vt.size(2) == hd_pad && vt.size(3) == T, "vt must be [N,H,hd_pad,T]");
  auto out = torch::empty({N, H, T, hd}, torch::TensorOptions().dtype(torch::kFloat16).device(q.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
  flash_attn_int8_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
      q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      N, H, T, hd, hd_pad, (float)softmax_scale, (int8_t*)nullptr, 0.f, 0);
  return out;
}

// Fused proj-quantize variant: same mma flash attention, but the final store emits the attention
// output as INT8 token-major [b*T, C] (C = H*hd) quantized by the calibrated proj scale, instead of
// the fp16 head-major tensor + a separate quantize_attn_out_int8 pass. Bit-identical to
// quantize_attn_out_int8(flash_attn_int8_vt(...), proj_a_scale) (rounds through fp16 first).
torch::Tensor flash_attn_int8_vt_qout(torch::Tensor q, torch::Tensor k, torch::Tensor vt,
                                      torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                      double softmax_scale, double proj_a_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && vt.is_cuda(), "flash_attn_int8_vt_qout: q/k/vt must be CUDA");
  TORCH_CHECK(q.dim() == 4 && vt.dim() == 4, "q [N,H,T,hd_pad], vt [N,H,hd_pad,T]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && vt.is_contiguous(), "q/k/vt must be contiguous");
  const int N = q.size(0), H = q.size(1), T = q.size(2), hd_pad = q.size(3);
  const int hd = sv.size(-1);
  TORCH_CHECK(hd_pad <= FA_MMA_MAXHD && (T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0
              && (FA_MMA_BC * hd_pad) % 16 == 0, "flash_attn_int8_vt_qout: mma-eligible shapes only");
  TORCH_CHECK(vt.size(2) == hd_pad && vt.size(3) == T, "vt must be [N,H,hd_pad,T]");
  const int C = H * hd;   // int8 proj K == C (C % 64 == 0), no K-pad needed
  auto out_q = torch::empty({(long)N * T, C}, torch::TensorOptions().dtype(torch::kChar).device(q.device()));
  const float proj_inv_scale = 1.f / (float)proj_a_scale;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
  flash_attn_int8_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
      q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      (__half*)nullptr, N, H, T, hd, hd_pad, (float)softmax_scale,
      out_q.data_ptr<int8_t>(), proj_inv_scale, C);
  return out_q;   // int8 [b*T, C] token-major (ready for gemm_w8a8_awq_bias_res)
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

  if (hd_pad <= FA_MMA_MAXHD && (T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0
      && (FA_MMA_BC * hd_pad) % 16 == 0) {
    // tensor-core mma flash path (all int8-path churches attention blocks).
    // V is fed PRE-TRANSPOSED [N,H,hd_pad,T] so the cp.async V-tile read is contiguous
    // (padded channels hd..hd_pad are already zero in v, so the transpose carries zeros).
    auto vt = v.transpose(2, 3).contiguous();       // [N,H,hd_pad,T]
    dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
    flash_attn_int8_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
        q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
        sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, H, T, hd, hd_pad, (float)softmax_scale, (int8_t*)nullptr, 0.f, 0);
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

// ---- ENTRYPOINT (fused flash attention, int4 QKᵀ / int8 PV). Kernel: flash_attn_int4_mma_kernel
//      (mma path only — no dp4a/naive fallback). ----
//   Op:       Attention W4A4 — fused flash
//   Inputs:   q4,k4 packed-int4 [N,H,T,hdp4/2] (hdp4 a multiple of 64), v int8 [N,H,T,hdp_v]
//             (natural layout, transposed to [N,H,hdp_v,T] internally; hdp_v=pad(hd,32)), sq,sk f32
//             [N,H,T] (per-token), sv f32 [N,H,hd] (per-channel), hdp4 int64, softmax_scale double
//   Outputs:  out fp16 [N,H,T,hd]
//   Computes: out = softmax(diag(sq)·(QKᵀ)·diag(sk)·scale) · V; QKᵀ in int4 (mma.m16n8k64.s4), PV
//             in int8 (mma.m16n8k32.s8 — P is requantized to int8 because int4 P∈[0,1] is too coarse)
//   Fuses:    flash fuses QKᵀ + online softmax + PV entirely in SRAM/registers (no [N,H,T,T] in HBM);
//             V transposed internally; same register-parallel softmax + 2-stage cp.async as int8.
//             Documented NEGATIVE result: at hd=24 the int4 K-pad (24→64, 62% waste) makes QKᵀ no
//             faster than int8, so it loses to fp16 flash just like the int8 kernel.
//   Constraints: hdp4%64==0 and hdp4<=64, T%64==0 (T%(FA_MMA_WARPS·FA_MMA_BR)), hd%8==0
//   vs fp16:  int8 flash 2.73x / int4 flash 2.78x vs fp16 MATH at hd24/T1024; ~1.2x at hd48/T256;
//             ~0.95-1.16x at hd48/T64. vs fp16 FlashAttention-2: 0.20-0.50x (loses). Fusion (BC=64
//             key tile, 2-stage cp.async, register online softmax) removes the 3x HBM round-trip
//             over the [BH,T,T] score matrix that the materialized path pays.
// V-pre-transposed variant: vt is already int8 [N,H,hdp_v,T] (e.g. from quantize_attn_qkv_i4qk_i8v),
// so we skip the internal transpose.
torch::Tensor flash_attn_int4_vt(torch::Tensor q4, torch::Tensor k4, torch::Tensor vt,
                                 torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                 int64_t hdp4, double softmax_scale) {
  TORCH_CHECK(q4.is_cuda() && k4.is_cuda() && vt.is_cuda(), "flash_attn_int4_vt: q4/k4/vt CUDA");
  TORCH_CHECK(q4.dim() == 4 && vt.dim() == 4 && q4.is_contiguous() && k4.is_contiguous() && vt.is_contiguous(),
              "q4/k4 [N,H,T,hdp4/2], vt [N,H,hdp_v,T] contiguous");
  const int N = q4.size(0), H = q4.size(1), T = q4.size(2);
  const int hd = sv.size(-1);
  const int hdp_v = ((hd + 31) / 32) * 32;
  TORCH_CHECK(hdp4 % 64 == 0 && hdp4 <= FA_MMA_MAXHD, "hdp4 mult of 64 and <= 64");
  TORCH_CHECK((T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0, "int4 flash: T%64==0, hd%8==0");
  TORCH_CHECK(vt.size(2) == hdp_v && vt.size(3) == T, "vt must be [N,H,hdp_v,T]");
  auto out = torch::empty({N, H, T, hd}, torch::TensorOptions().dtype(torch::kFloat16).device(q4.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
  flash_attn_int4_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
      q4.data_ptr<int8_t>(), k4.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      N, H, T, hd, (int)hdp4, hdp_v, (float)softmax_scale, (int8_t*)nullptr, 0.f, 0);
  return out;
}

// Fused proj-quantize variant of flash_attn_int4_vt: the store emits the attention output as
// packed int4 token-major [b*T, k_pad/2] (real channels 0..C-1 packed, C..k_pad-1 pre-zeroed by the
// torch::zeros alloc) quantized by the calibrated proj scale — bit-identical to
// quantize_attn_out_int4_pack(flash_attn_int4_vt(...), proj_a_scale, k_pad).
torch::Tensor flash_attn_int4_vt_qout(torch::Tensor q4, torch::Tensor k4, torch::Tensor vt,
                                      torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                      int64_t hdp4, double softmax_scale, double proj_a_scale, int64_t k_pad) {
  TORCH_CHECK(q4.is_cuda() && k4.is_cuda() && vt.is_cuda(), "flash_attn_int4_vt_qout: q4/k4/vt CUDA");
  TORCH_CHECK(q4.dim() == 4 && vt.dim() == 4 && q4.is_contiguous() && k4.is_contiguous() && vt.is_contiguous(),
              "q4/k4 [N,H,T,hdp4/2], vt [N,H,hdp_v,T] contiguous");
  const int N = q4.size(0), H = q4.size(1), T = q4.size(2);
  const int hd = sv.size(-1);
  const int hdp_v = ((hd + 31) / 32) * 32;
  TORCH_CHECK(hdp4 % 64 == 0 && hdp4 <= FA_MMA_MAXHD, "hdp4 mult of 64 and <= 64");
  TORCH_CHECK((T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0, "int4 flash: T%64==0, hd%8==0");
  TORCH_CHECK(vt.size(2) == hdp_v && vt.size(3) == T, "vt must be [N,H,hdp_v,T]");
  const int C = H * hd;
  int Kpad = (k_pad > 0) ? (int)k_pad : C;
  TORCH_CHECK(Kpad % 2 == 0 && Kpad >= C, "flash_attn_int4_vt_qout: k_pad must be even and >= C=H*hd");
  const int bstride = Kpad / 2;   // packed bytes per token row
  auto out_q = torch::zeros({(long)N * T, bstride}, torch::TensorOptions().dtype(torch::kChar).device(q4.device()));
  const float proj_inv_scale = 1.f / (float)proj_a_scale;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
  flash_attn_int4_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
      q4.data_ptr<int8_t>(), k4.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      (__half*)nullptr, N, H, T, hd, (int)hdp4, hdp_v, (float)softmax_scale,
      out_q.data_ptr<int8_t>(), proj_inv_scale, bstride);
  return out_q;   // packed int4 [b*T, k_pad/2] token-major (ready for gemm_w4a4_awq_bias_res)
}

torch::Tensor flash_attn_int4(torch::Tensor q4, torch::Tensor k4, torch::Tensor v,
                              torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                              int64_t hdp4, double softmax_scale) {
  TORCH_CHECK(q4.is_cuda() && k4.is_cuda() && v.is_cuda(), "flash_attn_int4: q/k/v CUDA");
  TORCH_CHECK(q4.dim() == 4 && q4.is_contiguous() && k4.is_contiguous(), "q4/k4 [N,H,T,hdp4/2] contiguous");
  const int N = q4.size(0), H = q4.size(1), T = q4.size(2);
  const int hd = sv.size(-1);
  const int hdp_v = ((hd + 31) / 32) * 32;
  TORCH_CHECK(hdp4 % 64 == 0 && hdp4 <= FA_MMA_MAXHD, "hdp4 must be a multiple of 64 and <= 64");
  TORCH_CHECK((T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0, "int4 flash: T%64==0, hd%8==0");
  auto out = torch::empty({N, H, T, hd}, torch::TensorOptions().dtype(torch::kFloat16).device(q4.device()));
  auto vt = v.transpose(2, 3).contiguous();            // [N,H,hdp_v,T]
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
  flash_attn_int4_mma_kernel<<<grid, FA_MMA_WARPS * 32, 0, stream>>>(
      q4.data_ptr<int8_t>(), k4.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      N, H, T, hd, (int)hdp4, hdp_v, (float)softmax_scale, (int8_t*)nullptr, 0.f, 0);
  return out;
}
