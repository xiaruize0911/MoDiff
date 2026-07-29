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
#include <cstdlib>   // getenv/atoi for MODIFF_FA_BC
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
//   BR=16 (mma M), BC = template arg, 32 or 64 (key tile; BC/8 mma N-tiles for QKᵀ,
//   BC/32 mma K-steps for PV). Runtime-selected by MODIFF_FA_BC, default 64.
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
// TEMPLATED on HD_PAD. This matters a lot, not cosmetically:
//   * hd_pad used to be a RUNTIME arg, so Oreg[] was indexed by the runtime n_nt2 and the
//     compiler could not keep it in registers -- cuobjdump showed STACK:128, i.e. all 32
//     fp32 accumulators spilled to LOCAL memory and every PV accumulate did a DRAM-backed
//     round trip. That, not arithmetic, is why int8 and int4 reached the SAME ~19-25 TOPS.
//   * smem was sized for FA_MMA_MAXHD=64 regardless of the real hd_pad, so an hd_pad=32
//     shape wasted half its smem: 24 KB/CTA -> 4 CTA/SM -> 33% occupancy on A40.
// With HD_PAD a compile-time constant: Oreg lives in registers (no spill), smem is exact
// (14 KB at HD_PAD=32 -> 7 CTA/SM), and the k/n-tile loops fully unroll.
// WARPS is also templated: more warps per CTA share ONE K/V smem tile, so the number of CTAs
// per (n,h) -- each of which streams the whole K and V -- drops proportionally. At T=1024,
// WARPS=4 gives grid.y=16 (K/V read 16x, ~1.11 GB, measured 365 GB/s of the ~590 GB/s this
// card sustains, i.e. memory-bound); WARPS=8 halves that to 0.57 GB at the same thread-level
// occupancy (3 CTA x 256 thr == 6 CTA x 128 thr == 768 thr/SM).
template <int HD_PAD, int WARPS, int BC>
__global__ void flash_attn_int8_mma_kernel_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hd_pad_rt, float softmax_scale,
    int8_t* __restrict__ out_q, float proj_inv_scale, int qout_stride) {
  constexpr int hd_pad = HD_PAD;
  (void)hd_pad_rt;
  // out_q != nullptr => fused proj-quantize store: emit int8 token-major [b*T, qout_stride]
  // (qout_stride = C = H*hd), quantized by proj_inv_scale, instead of the fp16 head-major `out`.
  const int nh = blockIdx.x;
  const int w = threadIdx.x >> 5;              // warp id in CTA
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int q0 = (blockIdx.y * WARPS + w) * FA_MMA_BR;   // this warp's query tile

  const int8_t* kb = k + (size_t)nh * T * hd_pad;
  const int8_t* vb = v + (size_t)nh * T * hd_pad;
  const float*  sqb = sq + (size_t)nh * T;
  const float*  skb = sk + (size_t)nh * T;
  const float*  svb = sv + (size_t)nh * hd;

  // Double-buffered cp.async tiles (FA_STAGES-deep pipeline). V is taken PRE-TRANSPOSED
  // [N,H,hd_pad,T] so both K ([BC,hd_pad]) and V ([hd_pad,BC]) tiles are HBM-contiguous.
  // Pipeline depth: 2, unconditionally. It used to be `(BC <= 32) ? 3 : 2` on the reasoning that
  // "deeper prefetch is exactly what a latency-bound kernel wants, and BC=32 is where it is
  // affordable". Both halves of that turned out to be wrong:
  //
  //   * The 3-stage path COMPUTED WRONG ANSWERS. BC=32 was the only config that got 3 stages and
  //     the only one that was numerically broken: rel-L2 vs an fp32 reference was 0.415
  //     (hd_pad=64,T=64), 0.197 (hd_pad=32,T=256), 0.068 (hd_pad=32,T=1024) against ~0.014-0.020
  //     for every 2-stage config. It was a race, not a rounding issue: the error grew with grid.x
  //     (9.6e-3 at N*H=16, 0.415 at N*H=1024, which is why a 16-CTA correctness gate never saw it)
  //     and WARPS=4 vs WARPS=8 gave DIFFERENT results, though warp count cannot change the math.
  //     Forcing 2 stages makes all 12 (BC x WARPS x T) configs agree to 0.0095-0.0203 and makes
  //     W=4 and W=8 bit-identical. int4 hardcoded 2 stages all along and was never affected.
  //   * The third stage bought no speed anyway: BC=32/W=8 at T=1024 (the config the heuristic
  //     picks for the dominant C192 block) measured 1891.8 us with 3 stages and 1881.9 us with 2.
  //     Every other cell moved within +-1%. So this is a pure correctness win, not a trade.
  //
  // The un-diagnosed part is *which* of the 3-stage buffer/commit interactions raced;
  // `__pipeline_wait_prior(FA_STAGES-2)` with three buffers is the obvious suspect. Anyone
  // restoring a deeper pipeline must re-run scripts/qattn_correctness.py at its LARGE (128,8,*)
  // cases -- the small ones pass even when the kernel is wrong.
  constexpr int FA_STAGES = 2;
  __shared__ int8_t Ks[FA_STAGES][BC * HD_PAD];
  __shared__ int8_t Vs[FA_STAGES][HD_PAD * BC];  // Vs[buf][d*BC+j] = V[kt+j][d]
  __shared__ int8_t Qs[WARPS * FA_MMA_BR * HD_PAD];
  __shared__ int8_t Ps[WARPS * FA_MMA_BR * BC];
  // Per-key K scales for the current tile, staged once per CTA (512 B). They used to be read
  // straight from HBM inside the score loop -- 24 LDG per lane PER KEY TILE in the SASS census.
  __shared__ float SKs[FA_STAGES][BC];

  int8_t* Qsw = &Qs[w * FA_MMA_BR * hd_pad];
  int8_t* Psw = &Ps[w * FA_MMA_BR * BC];

  // No bounds mask: grid.y == T/(WARPS*BR) and the host enforces T % (FA_MMA_WARPS*FA_MMA_BR) == 0,
  // so q0 + FA_MMA_BR <= T always. Same reasoning kills the kt + c < T masks in the score loop.
  for (int idx = lane; idx < FA_MMA_BR * hd_pad; idx += 32) {
    int row = idx / hd_pad, col = idx % hd_pad;
    Qsw[idx] = q[(size_t)nh * T * hd_pad + (size_t)(q0 + row) * hd_pad + col];
  }

  // Per-lane running softmax state (registers, no smem). Each lane owns rows gid & gid+8;
  // the 4 lanes sharing a gid (tig=0..3) agree on m/l after the __shfl_xor reduction.
  float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
  // Exactly the accumulators this HD_PAD needs, and a compile-time count so they stay in
  // registers instead of spilling (the whole point of templating -- see the header note).
  constexpr int NT_MAX = HD_PAD / 8;
  float Oreg[NT_MAX * 4];
#pragma unroll
  for (int i = 0; i < NT_MAX * 4; ++i) Oreg[i] = 0.f;
  const int n_nt2 = hd / 8;                      // <= NT_MAX; hd may be < HD_PAD (padding)
  constexpr int nkt = HD_PAD / 32;
  constexpr int NNT = BC / 8;             // QKᵀ N-tiles
  const int NKV = T / BC;                  // key tiles (T % BC == 0 on the mma path)
  const int KV16 = (BC * hd_pad) / 16;     // 16B chunks per K tile (== per V tile)

  // ---- fold every loop-invariant scalar into per-lane constants, in log2 units ----
  // The SASS census showed 222 FMUL per lane per key tile against 16 IMMA: this kernel is
  // issue-bound on the fp32 dequant/softmax/requant epilogue, not on the tensor cores. Each
  // fold below removes 32 FMUL per key tile (one per score element this lane owns):
  //   * softmax_scale and log2(e) folded into the Q scale  -> score comes out already in log2
  //     units, so modiff_ex2() is a single MUFU.EX2 with no log2(e) multiply of its own;
  //   * the P requantize factor 127 folded into the running max (mq = mcur - log2(127)), so
  //     modiff_ex2(S - mq) IS 127*exp(S - mcur) and the separate "* 127.f" disappears;
  //   * consequently l_run and Oreg both carry a factor of 127, which cancels in Oreg/l_run,
  //     so the PV epilogue's "* (1/127)" disappears too.
  constexpr float FA_LOG2E   = 1.4426950408889634f;
  constexpr float FA_LOG2_127 = 6.98868468677217f;      // log2(127)
  const float sqs0 = sqb[q0 + gid]     * softmax_scale * FA_LOG2E;
  const float sqs1 = sqb[q0 + gid + 8] * softmax_scale * FA_LOG2E;
  // This lane's V scales, hoisted out of the key loop (they depend only on nt2 and tig).
  float svr[NT_MAX][2];
#pragma unroll
  for (int i = 0; i < NT_MAX; ++i) {
    const int d0 = i * 8 + tig * 2;
    svr[i][0] = (d0     < hd) ? svb[d0]     : 0.f;
    svr[i][1] = (d0 + 1 < hd) ? svb[d0 + 1] : 0.f;
  }

  // ---- hoist the Q A-fragment into registers, ONCE, outside the key loop ----
  // Q is loop-invariant: the same [BR x hd_pad] tile feeds every key tile and every N-tile.
  // It used to be re-read from Qsw inside the mma loop, i.e. NNT*nkt*4 = 32 (HD_PAD=32) or
  // 64 (HD_PAD=64) redundant smem loads per lane PER KEY TILE -- by far the largest single
  // item of smem traffic in the kernel, and the reason both int8 and int4 plateaued at
  // ~15% of their (very different) peaks: the mma units were starved by the smem pipe,
  // not by arithmetic. nkt is compile-time so this is 4*nkt = 4 or 8 registers, held for
  // the whole kernel, and the loads happen NKV times fewer.
  __syncwarp();                      // Qsw was filled cooperatively by this warp's 32 lanes
  unsigned Qa[nkt][4];
#pragma unroll
  for (int ks = 0; ks < nkt; ++ks) {
    const int base = ks * 32;
    Qa[ks][0] = *(const unsigned*)&Qsw[(gid)     * hd_pad + base + tig * 4];
    Qa[ks][1] = *(const unsigned*)&Qsw[(gid + 8) * hd_pad + base + tig * 4];
    Qa[ks][2] = *(const unsigned*)&Qsw[(gid)     * hd_pad + base + tig * 4 + 16];
    Qa[ks][3] = *(const unsigned*)&Qsw[(gid + 8) * hd_pad + base + tig * 4 + 16];
  }

  // cp.async a K tile ([BC,hd_pad] from k[nh]) and V tile ([hd_pad,BC] from pre-T v[nh]) into buf.
  auto load_kv = [&](int buf, int kt) {
    for (int c = threadIdx.x; c < KV16; c += blockDim.x) {
      int off = c * 16;
      // K: row-major [BC,hd_pad], HBM kb[(kt+j)*hd_pad + d]
      modiff_cp_async_cg(modiff_smem_ptr(&Ks[buf][off]),
                         (const uint4*)(kb + (size_t)kt * hd_pad + off), true);
      // V (pre-transposed [hd_pad,T]): row-major [hd_pad,BC], HBM vb[d*T + kt+j]
      int d = off / BC, j = off % BC;
      modiff_cp_async_cg(modiff_smem_ptr(&Vs[buf][off]),
                         (const uint4*)(vb + (size_t)d * T + kt + j), true);
    }
    // K scales for this tile: 64 floats = 16 chunks of 16 B, same pipeline as K/V.
    for (int c = threadIdx.x; c < BC / 4; c += blockDim.x)
      modiff_cp_async_cg(modiff_smem_ptr(&SKs[buf][c * 4]),
                         (const uint4*)(skb + kt + c * 4), true);
    __pipeline_commit();
  };

  // prime FA_STAGES-1 tiles (commits in flight)
#pragma unroll
  for (int s = 0; s < FA_STAGES - 1; ++s) if (s < NKV) load_kv(s % FA_STAGES, s * BC);

  for (int ktile = 0; ktile < NKV; ++ktile) {
    const int kt = ktile * BC;
    const int buf = ktile % FA_STAGES;
    __pipeline_wait_prior(FA_STAGES - 2);   // wait until oldest (tile ktile) has arrived
    __syncthreads();                        // all warps done with prev tile's buffer; tile ktile visible
    const int nxt = ktile + (FA_STAGES - 1);
    if (nxt < NKV) load_kv(nxt % FA_STAGES, nxt * BC);  // prefetch tile ktile+STAGES-1
    int8_t* Ksb = Ks[buf];
    int8_t* Vsb = Vs[buf];

    // ---- QKᵀ: keep the [BR x BC] score tile in registers (Sreg[nt][0..3]) ----
    // Sreg[nt] = {row gid col c0, row gid col c1, row gid+8 col c0, row gid+8 col c1}
    //
    // DO NOT "tile Sreg to cut registers" -- measured and rejected (data/attn_headroom.json).
    // The online softmax is incremental over columns, so splitting this tile into halves of BC/2
    // computes exactly what a smaller BC computes: the idea IS BC=32, which already exists and
    // already lowers REG 127 -> 96. Two measurements kill it at hd=48 (the shape it was aimed at):
    //   * REG 96 at WARPS=8 still allows only 2 CTA/SM -- the SAME 33.3% occupancy as REG 127, so
    //     the register saving buys no occupancy at all (3 CTA would need REG <= 85).
    //   * The one config that does reach higher occupancy, BC=32/W=4 at 41.7%, is the SLOWEST of
    //     the four: 349.6 us vs 264.4 us for BC=64/W=8 at T=256. This kernel gains more from the
    //     WARPS=8 K/V smem sharing than from occupancy.
    float Sreg[BC / 8][4];
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      int acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ks = 0; ks < nkt; ++ks) {
        int base = ks * 32;
        unsigned b[2];
        b[0] = *(const int*)&Ksb[(nt * 8 + gid) * hd_pad + base + tig * 4];
        b[1] = *(const int*)&Ksb[(nt * 8 + gid) * hd_pad + base + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, Qa[ks], b);   // Q from registers (hoisted), not smem
      }
      // Apply ONLY the per-key scale here: Sreg holds acc*sk[j], not the full score. The
      // per-row Q scale is deliberately left out and applied later inside the exp argument.
      // That is exact, not an approximation: every quantization scale is strictly positive, so
      //     max_j (sq_i * u_ij) == sq_i * max_j u_ij
      // and the running max can be tracked in these unscaled units. Doing it this way turns
      // 2 multiplies per element (sqs*sk, then acc*that) plus the later (S - m) subtract into
      // 1 multiply plus 1 FFMA -- 4 fewer instructions per N-tile. No bounds mask (see above).
      const int c0 = nt * 8 + tig * 2;
      const float sk0 = SKs[buf][c0], sk1 = SKs[buf][c0 + 1];
      Sreg[nt][0] = acc[0] * sk0;
      Sreg[nt][1] = acc[1] * sk1;
      Sreg[nt][2] = acc[2] * sk0;
      Sreg[nt][3] = acc[3] * sk1;
    }

    // ---- register-parallel online softmax (all 32 lanes active) ----
    // local max over this lane's 8 cols for each of its 2 rows
    // Single accumulator chain, deliberately. A 4-way split (chain depth 8 -> 4) was measured
    // and is a REGRESSION here: it pushes the kernel from REG:83 to REG:96, which drops
    // occupancy from 3 CTA/SM (24 warps, 50%) to 2 CTA/SM (16 warps, 33%), and T=1024/hd=24
    // went 1973 -> 2032 us. The kernel is latency-bound, but on this shape occupancy buys more
    // latency hiding than instruction-level parallelism does.
    float lm0 = -INFINITY, lm1 = -INFINITY;
    for (int nt = 0; nt < NNT; ++nt) {
      lm0 = fmaxf(lm0, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      lm1 = fmaxf(lm1, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    // reduce across the 4 tig-lanes of each gid-group (lanes differ in low 2 bits)
    lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1)); lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
    lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1)); lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
    float mcur0 = fmaxf(m_run0, lm0), mcur1 = fmaxf(m_run1, lm1);
    // m_run/mcur are in the unscaled u = acc*sk units, so the per-row Q scale enters here.
    float a_g  = modiff_ex2(sqs0 * (m_run0 - mcur0));   // per-row alpha (registers)
    float a_g8 = modiff_ex2(sqs1 * (m_run1 - mcur1));
    // One FFMA per element does the whole exp argument:
    //     sqs*u + nb  where  nb = log2(127) - sqs*mcur
    //   = sqs*(u - mcur) + log2(127)
    // so modiff_ex2 of it IS 127*exp(score - max): the Q scale, the max subtraction and the
    // int8 requantize factor all ride in one instruction. l_run and Oreg both carry the 127,
    // which cancels in the final Oreg/l_run.
    const float nb0 = FA_LOG2_127 - sqs0 * mcur0, nb1 = FA_LOG2_127 - sqs1 * mcur1;
    float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      const int c0 = nt * 8 + tig * 2;
      float p00 = modiff_ex2(fmaf(sqs0, Sreg[nt][0], nb0));
      float p01 = modiff_ex2(fmaf(sqs0, Sreg[nt][1], nb0));
      float p10 = modiff_ex2(fmaf(sqs1, Sreg[nt][2], nb1));
      float p11 = modiff_ex2(fmaf(sqs1, Sreg[nt][3], nb1));
      ls0 += p00 + p01; ls1 += p10 + p11;
      // p is already in [0,127]. __float2int_rn is a single F2I.RN, so we get round-to-nearest
      // (matching the old "* 127.f + 0.5f") for free and still need no clamp select: rn of a
      // value <= 127.0f cannot reach 128. Plain truncation here cost ~2x the requantize error.
      // c0 is even, so the two adjacent columns go out as one 2-byte store instead of two.
      // cvt.pack.sat.s8.s32 replaces the shift+or (IMAD.SHL + LOP3) with one instruction and
      // saturates on the way, so no clamp select is needed either.
      unsigned r0 = modiff_pack2_s8(__float2int_rn(p00), __float2int_rn(p01));
      unsigned r1 = modiff_pack2_s8(__float2int_rn(p10), __float2int_rn(p11));
      *(short*)&Psw[gid       * BC + c0] = (short)r0;
      *(short*)&Psw[(gid + 8) * BC + c0] = (short)r1;
    }
    // reduce partial sums across the tig-group, combine with running sum (rescaled)
    ls0 += __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
    l_run0 = l_run0 * a_g + ls0; l_run1 = l_run1 * a_g8 + ls1;
    m_run0 = mcur0; m_run1 = mcur1;
    __syncwarp();

    // Hoist the P A-fragment: it depends only on (ks, gid, tig), NOT on nt2, yet the PV loop
    // re-read it from Psw once per output tile -- n_nt2 * (BC/32) * 4 = up to 64 smem loads per
    // lane per key tile where 8 suffice. Together with the Q hoist above this cuts inner-loop
    // smem traffic ~2.6x (hd=48: 168 -> 64 loads/lane/tile).
    unsigned Pa[BC / 32][4];
#pragma unroll
    for (int ks = 0; ks < BC / 32; ++ks) {
      const int koff = ks * 32;
      Pa[ks][0] = *(const unsigned*)&Psw[(gid)     * BC + koff + tig * 4];
      Pa[ks][1] = *(const unsigned*)&Psw[(gid + 8) * BC + koff + tig * 4];
      Pa[ks][2] = *(const unsigned*)&Psw[(gid)     * BC + koff + tig * 4 + 16];
      Pa[ks][3] = *(const unsigned*)&Psw[(gid + 8) * BC + koff + tig * 4 + 16];
    }

    // ---- rescale O (registers) by per-row alpha, then PV accumulate ----
#pragma unroll
    for (int nt2 = 0; nt2 < NT_MAX; ++nt2) {
      if (nt2 >= n_nt2) break;                   // NT_MAX is compile-time; n_nt2 <= NT_MAX
      Oreg[nt2 * 4 + 0] *= a_g;  Oreg[nt2 * 4 + 1] *= a_g;
      Oreg[nt2 * 4 + 2] *= a_g8; Oreg[nt2 * 4 + 3] *= a_g8;
      int acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ks = 0; ks < BC / 32; ++ks) {          // K = BC (one or more mma k-steps)
        int koff = ks * 32;
        unsigned b[2];
        b[0] = *(const int*)&Vsb[(nt2 * 8 + gid) * BC + koff + tig * 4];
        b[1] = *(const int*)&Vsb[(nt2 * 8 + gid) * BC + koff + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, Pa[ks], b);   // P from registers (hoisted out of the nt2 loop)
      }
      // svr is pre-loaded and the 1/127 is folded into l_run (see the mq note above), so this is
      // one FFMA per accumulator instead of two FMUL + one FADD.
      Oreg[nt2 * 4 + 0] += svr[nt2][0] * acc[0];
      Oreg[nt2 * 4 + 1] += svr[nt2][1] * acc[1];
      Oreg[nt2 * 4 + 2] += svr[nt2][0] * acc[2];
      Oreg[nt2 * 4 + 3] += svr[nt2][1] * acc[3];
    }
    // buffer reuse safety handled by next iter's __pipeline_wait_prior + __syncthreads
  }

  float invl_g  = (l_run0 > 0.f) ? 1.f / l_run0 : 0.f;
  float invl_g8 = (l_run1 > 0.f) ? 1.f / l_run1 : 0.f;
  const int h = nh % H, n = nh / H;           // (n,h) for the token-major fused-quant store
#pragma unroll
  for (int nt2 = 0; nt2 < NT_MAX; ++nt2) {
    if (nt2 >= n_nt2) break;
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
// PACKED-INPUT flash: reads the interleaved qkv [b,T,nh,3,hd] (channel order
// (nh,{q,k,v},hd)) DIRECTLY and does the split + hd->hd_pad zero-pad + V-transpose
// (and, for fp16 input, the static-scale quantize) INSIDE its smem staging --
// replacing the separate aq_qtok/aq_vquant (or Route-1 from_i8) reshuffle pass and
// the qi/ki/vt HBM round-trip. The mma compute core (QKᵀ, online softmax, PV, store)
// is byte-identical to flash_attn_int8_mma_kernel.
//   TIn=__half : quantize on load (q_inv=1/sq_c, k_inv=1/sk_c, 1/sv[d] for V) ->
//                bit-identical to quantize_attn_qkv_packed_static -> flash_attn_int8_vt.
//   TIn=int8_t : plain gather (input already int8, scales folded upstream, Route-1) ->
//                bit-identical to quantize_attn_qkv_from_i8 -> flash_attn_int8_vt.
// 2-stage double-buffer: cp.async-prefetch RAW tiles (dynamic smem, stride hd) so the global
// load overlaps compute, then quantize+transpose smem->smem into the int8 mma tiles. Scales are
// the frozen per-tensor sq_c/sk_c + per-channel sv[hd]. Vs uses a padded row stride to soften the
// transpose-write bank conflict (stride FA_MMA_BC would be 0 mod 32 -> 32-way).
// =========================================================================
#define FA_VS_STRIDE (FA_MMA_BC + 4)   // padded Vs row stride (4-aligned for the int PV read)
__device__ __forceinline__ int8_t mfp_stage(__half x, float inv) {   // fp16: quantize
  int q = __float2int_rn(__half2float(x) * inv);
  return (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
}
__device__ __forceinline__ int8_t mfp_stage(int8_t x, float /*inv*/) { return x; }  // int8: copy

// TEMPLATED on HD_PAD, for the same reason the unpacked kernel was (see
// flash_attn_int8_mma_kernel_t's header). This kernel was left behind by that round and was
// still the ONLY production flash kernel that spilled: cuobjdump reported REG:126 STACK:32,
// because Oreg/svr were sized by FA_MMA_MAXNT and indexed against the runtime n_nt2, and the
// smem tiles were sized for FA_MMA_MAXHD=64 whatever the real HD_PAD. That is why the packed
// path measured 8499 us at T=1024/hd=24 against 2595 us for quantize+unpacked-flash (3.3x
// WORSE), which the per-block autotune then correctly refused to use -- so the fusion it exists
// to provide (folding the Q/K/V quantize into flash's staging, worth the 724 us S3+quant pass)
// was never actually available.
template <typename TIn, int HD_PAD>
__global__ void flash_attn_int8_packed_mma_kernel(
    const TIn* __restrict__ qkv, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hd_pad_rt,
    float sq_c, float sk_c, float softmax_scale, float q_inv, float k_inv,
    int8_t* __restrict__ out_q, float proj_inv_scale, int qout_stride) {
  (void)hd_pad_rt;                 // HD_PAD is the compile-time truth; the host asserts they match
  const int nh = blockIdx.x;
  const int w = threadIdx.x >> 5;              // warp id in CTA
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int q0 = (blockIdx.y * FA_MMA_WARPS + w) * FA_MMA_BR;   // this warp's query tile

  // packed base for this CTA's (sample n, head h): X(t,d) = xh[t*pkT + d]
  const int h = nh % H, n = nh / H;
  const size_t pkT = (size_t)H * 3 * hd;                          // elements per token, all heads
  const TIn* qh = qkv + (size_t)n * T * pkT + (size_t)(h * 3 + 0) * hd;
  const TIn* kh = qkv + (size_t)n * T * pkT + (size_t)(h * 3 + 1) * hd;
  const TIn* vh = qkv + (size_t)n * T * pkT + (size_t)(h * 3 + 2) * hd;

  // int8 mma tiles (static). Raw fp16/int8 prefetch tiles live in DYNAMIC smem (2-stage double-buffer,
  // row stride = hd) so the global load overlaps compute via cp.async; the quantize+transpose is a
  // cheap smem->smem pass. Static: Ks[BC,HD_PAD] + Vs[HD_PAD,BC] + Qs + Ps = 16 KB.
  __shared__ int8_t Ks[FA_MMA_BC * HD_PAD];
  __shared__ int8_t Vs[HD_PAD * FA_VS_STRIDE];   // Vs[d*STRIDE + j] = V[kt+j][d] (padded)
  __shared__ int8_t Qs[FA_MMA_WARPS * FA_MMA_BR * HD_PAD];
  __shared__ int8_t Ps[FA_MMA_WARPS * FA_MMA_BR * FA_MMA_BC];
  extern __shared__ char s_raw[];                   // [2][BC*hd] K then [2][BC*hd] V, TIn-typed
  TIn* Kraw = reinterpret_cast<TIn*>(s_raw);
  TIn* Vraw = Kraw + (size_t)2 * FA_MMA_BC * hd;

  int8_t* Qsw = &Qs[w * FA_MMA_BR * HD_PAD];
  int8_t* Psw = &Ps[w * FA_MMA_BR * FA_MMA_BC];

  // Q: per-warp gather from packed qkv + quantize on load (col>=hd or gq>=T -> 0; d-fastest coalesced).
  for (int idx = lane; idx < FA_MMA_BR * HD_PAD; idx += 32) {
    // grid.y == T/(WARPS*BR) and the host gates T % (FA_MMA_WARPS*FA_MMA_BR) == 0, so
    // q0 + FA_MMA_BR <= T: no row mask. col < hd is the real hd-padding mask and stays.
    int row = idx / HD_PAD, col = idx % HD_PAD;
    Qsw[idx] = (col < hd) ? mfp_stage(qh[(size_t)(q0 + row) * pkT + col], q_inv) : (int8_t)0;
  }

  // Per-lane running softmax state (registers). Each lane owns rows gid & gid+8.
  float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
  // Exactly this HD_PAD's accumulators, compile-time counted so they stay in registers.
  constexpr int NT_MAX = HD_PAD / 8;
  float Oreg[NT_MAX * 4];
#pragma unroll
  for (int i = 0; i < NT_MAX * 4; ++i) Oreg[i] = 0.f;
  const int n_nt2 = hd / 8;
  constexpr int nkt = HD_PAD / 32;
  const int NNT = FA_MMA_BC / 8;                 // QKᵀ N-tiles
  const int NKV = T / FA_MMA_BC;                 // key tiles (T % BC == 0 on the mma path)
  const int EPC = 16 / (int)sizeof(TIn);         // elems per 16B cp.async chunk (8 half / 16 int8)
  const int CPT = hd / EPC;                      // chunks per token (hd*sizeof(TIn) % 16 == 0, host-gated)
  // Same instruction-count folds as the other two mma kernels (see the FA_LOG2E block in
  // flash_attn_int8_mma_kernel_t). This path uses calibrated STATIC scales, so the whole
  // dequant chain collapses to one compile-time-invariant constant in log2 units.
  constexpr float FAP_LOG2E    = 1.4426950408889634f;
  constexpr float FAP_LOG2_127 = 6.98868468677217f;
  const float sqk = sq_c * sk_c * softmax_scale * FAP_LOG2E;
  // This lane's V scales, hoisted out of the key loop (1/127 folded into l_run, see below).
  float svr[NT_MAX][2];
#pragma unroll
  for (int i = 0; i < NT_MAX; ++i) {
    const int d0 = i * 8 + tig * 2;
    svr[i][0] = (d0     < hd) ? sv[d0]     : 0.f;
    svr[i][1] = (d0 + 1 < hd) ? sv[d0 + 1] : 0.f;
  }

  // Hoist the loop-invariant Q A-fragment into registers -- same reason as in the templated
  // kernel above: re-reading it inside the mma loop cost NNT*nkt*4 = up to 64 redundant smem
  // loads per lane PER KEY TILE, which starved the mma units. Bound is the compile-time max
  // (FA_MMA_MAXHD/32 = 2) so the array stays in registers even though nkt is a runtime value.
  __syncwarp();                                  // Qsw was filled by this warp's 32 lanes
  constexpr int NKT_MAX = HD_PAD / 32;
  unsigned Qa[NKT_MAX][4];
#pragma unroll
  for (int ks = 0; ks < NKT_MAX; ++ks) {
    if (ks >= nkt) break;
    const int base = ks * 32;
    Qa[ks][0] = *(const unsigned*)&Qsw[(gid)     * HD_PAD + base + tig * 4];
    Qa[ks][1] = *(const unsigned*)&Qsw[(gid + 8) * HD_PAD + base + tig * 4];
    Qa[ks][2] = *(const unsigned*)&Qsw[(gid)     * HD_PAD + base + tig * 4 + 16];
    Qa[ks][3] = *(const unsigned*)&Qsw[(gid + 8) * HD_PAD + base + tig * 4 + 16];
  }

  // cp.async a raw K + raw V tile (token-major, stride hd) into double-buffer `buf`.
  auto load_raw = [&](int buf, int kt) {
    TIn* Kd = Kraw + (size_t)buf * FA_MMA_BC * hd;
    TIn* Vd = Vraw + (size_t)buf * FA_MMA_BC * hd;
    for (int c = threadIdx.x; c < FA_MMA_BC * CPT; c += blockDim.x) {
      int j = c / CPT, off = (c % CPT) * EPC, gj = kt + j;
      modiff_cp_async_cg(modiff_smem_ptr(&Kd[j * hd + off]),
                         (const uint4*)(kh + (size_t)gj * pkT + off), gj < T);
      modiff_cp_async_cg(modiff_smem_ptr(&Vd[j * hd + off]),
                         (const uint4*)(vh + (size_t)gj * pkT + off), gj < T);
    }
    __pipeline_commit();
  };
  // quantize/transpose raw tile `buf` (smem) -> int8 Ks [BC,HD_PAD] + Vs [HD_PAD,BC] (smem->smem).
  auto quant_tile = [&](int buf) {
    const TIn* Ksrc = Kraw + (size_t)buf * FA_MMA_BC * hd;
    const TIn* Vsrc = Vraw + (size_t)buf * FA_MMA_BC * hd;
    for (int idx = threadIdx.x; idx < FA_MMA_BC * HD_PAD; idx += blockDim.x) {
      int j = idx / HD_PAD, d = idx % HD_PAD;
      Ks[j * HD_PAD + d] = (d < hd) ? mfp_stage(Ksrc[j * hd + d], k_inv) : (int8_t)0;
      Vs[d * FA_VS_STRIDE + j] = (d < hd) ? mfp_stage(Vsrc[j * hd + d], 1.f / sv[d]) : (int8_t)0;
    }
  };

  load_raw(0, 0);                                  // prime tile 0 (2-stage double-buffer)
  for (int ktile = 0; ktile < NKV; ++ktile) {
    const int buf = ktile & 1;
    __pipeline_wait_prior(0);   // raw[buf] arrived
    __syncthreads();            // raw visible; prev Ks/Vs consumed; Q visible (iter 0)
    const int nxt = ktile + 1;
    if (nxt < NKV) load_raw(nxt & 1, nxt * FA_MMA_BC);   // prefetch (overlaps quantize+compute)
    quant_tile(buf);
    __syncthreads();            // Ks/Vs visible
    const int kt = ktile * FA_MMA_BC;
    int8_t* Ksb = Ks;
    int8_t* Vsb = Vs;

    // ---- QKᵀ: keep the [BR x BC] score tile in registers (Sreg[nt][0..3]) ----
    float Sreg[FA_MMA_BC / 8][4];
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      int acc[4] = {0, 0, 0, 0};
      for (int ks = 0; ks < nkt; ++ks) {
        int base = ks * 32;
        unsigned b[2];
        b[0] = *(const int*)&Ksb[(nt * 8 + gid) * HD_PAD + base + tig * 4];
        b[1] = *(const int*)&Ksb[(nt * 8 + gid) * HD_PAD + base + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, Qa[ks], b);   // Q from registers (hoisted), not smem
      }
      int c0 = nt * 8 + tig * 2, c1 = c0 + 1;
      // Same multiply order as flash_attn_int8_mma_kernel (acc*sq*sk*scale) for bit-exactness:
      // static sq/sk are per-tensor, so sqi0==sqi1==sq_c and sk0==sk1==sk_c.
      Sreg[nt][0] = acc[0] * sqk;      // one FMUL, already in log2 units; no bounds mask
      Sreg[nt][1] = acc[1] * sqk;
      Sreg[nt][2] = acc[2] * sqk;
      Sreg[nt][3] = acc[3] * sqk;
    }

    // ---- register-parallel online softmax (all 32 lanes active) ----
    float lm0 = -INFINITY, lm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      lm0 = fmaxf(lm0, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      lm1 = fmaxf(lm1, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1)); lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
    lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1)); lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
    float mcur0 = fmaxf(m_run0, lm0), mcur1 = fmaxf(m_run1, lm1);
    float a_g  = modiff_ex2(m_run0 - mcur0);         // one MUFU.EX2, no library range guard
    float a_g8 = modiff_ex2(m_run1 - mcur1);
    // mq = mcur - log2(127) makes modiff_ex2(S - mq) == 127*exp(S - mcur), so the P requantize
    // scale is free; l_run and Oreg then both carry the 127 and it cancels at the end.
    const float mq0 = mcur0 - FAP_LOG2_127, mq1 = mcur1 - FAP_LOG2_127;
    float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
    // NNT is a runtime const here (this kernel is not templated on HD_PAD), so without an
    // explicit unroll Sreg[8][4] gets a runtime subscript and part of it lands on the stack
    // -- cuobjdump showed STACK:32 for this kernel while the templated ones show STACK:0.
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      const int c0 = nt * 8 + tig * 2;
      float p00 = modiff_ex2(Sreg[nt][0] - mq0), p01 = modiff_ex2(Sreg[nt][1] - mq0);
      float p10 = modiff_ex2(Sreg[nt][2] - mq1), p11 = modiff_ex2(Sreg[nt][3] - mq1);
      ls0 += p00 + p01; ls1 += p10 + p11;
      // p is already in [0,127] (127 folded into mq); F2I.RN needs no clamp select, and the
      // two adjacent columns leave as one 2-byte store.
      // cvt.pack.sat.s8.s32 replaces the shift+or (IMAD.SHL + LOP3) with one instruction and
      // saturates on the way, so no clamp select is needed either.
      unsigned r0 = modiff_pack2_s8(__float2int_rn(p00), __float2int_rn(p01));
      unsigned r1 = modiff_pack2_s8(__float2int_rn(p10), __float2int_rn(p11));
      *(short*)&Psw[gid       * FA_MMA_BC + c0] = (short)r0;
      *(short*)&Psw[(gid + 8) * FA_MMA_BC + c0] = (short)r1;
    }
    ls0 += __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
    l_run0 = l_run0 * a_g + ls0; l_run1 = l_run1 * a_g8 + ls1;
    m_run0 = mcur0; m_run1 = mcur1;
    __syncwarp();

    // Hoist the P A-fragment: it depends only on (ks, gid, tig), NOT on nt2, yet the PV loop
    // re-read it from Psw once per output tile -- n_nt2 * (BC/32) * 4 = up to 64 smem loads per
    // lane per key tile where 8 suffice. Together with the Q hoist above this cuts inner-loop
    // smem traffic ~2.6x (hd=48: 168 -> 64 loads/lane/tile).
    unsigned Pa[FA_MMA_BC / 32][4];
#pragma unroll
    for (int ks = 0; ks < FA_MMA_BC / 32; ++ks) {
      const int koff = ks * 32;
      Pa[ks][0] = *(const unsigned*)&Psw[(gid)     * FA_MMA_BC + koff + tig * 4];
      Pa[ks][1] = *(const unsigned*)&Psw[(gid + 8) * FA_MMA_BC + koff + tig * 4];
      Pa[ks][2] = *(const unsigned*)&Psw[(gid)     * FA_MMA_BC + koff + tig * 4 + 16];
      Pa[ks][3] = *(const unsigned*)&Psw[(gid + 8) * FA_MMA_BC + koff + tig * 4 + 16];
    }

    // ---- rescale O (registers) by per-row alpha, then PV accumulate ----
    // static bound + break -> Oreg stays in registers (runtime n_nt2 gave STACK:128)
    #pragma unroll
    for (int nt2 = 0; nt2 < NT_MAX; ++nt2) {
      if (nt2 >= n_nt2) break;
      Oreg[nt2 * 4 + 0] *= a_g;  Oreg[nt2 * 4 + 1] *= a_g;
      Oreg[nt2 * 4 + 2] *= a_g8; Oreg[nt2 * 4 + 3] *= a_g8;
      int acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ks = 0; ks < FA_MMA_BC / 32; ++ks) {
        int koff = ks * 32;
        unsigned b[2];
        b[0] = *(const int*)&Vsb[(nt2 * 8 + gid) * FA_VS_STRIDE + koff + tig * 4];
        b[1] = *(const int*)&Vsb[(nt2 * 8 + gid) * FA_VS_STRIDE + koff + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, Pa[ks], b);   // P from registers (hoisted out of the nt2 loop)
      }
      Oreg[nt2 * 4 + 0] += svr[nt2][0] * acc[0];   // 1/127 folded into l_run
      Oreg[nt2 * 4 + 1] += svr[nt2][1] * acc[1];
      Oreg[nt2 * 4 + 2] += svr[nt2][0] * acc[2];
      Oreg[nt2 * 4 + 3] += svr[nt2][1] * acc[3];
    }
  }

  float invl_g  = (l_run0 > 0.f) ? 1.f / l_run0 : 0.f;
  float invl_g8 = (l_run1 > 0.f) ? 1.f / l_run1 : 0.f;
  // static bound + break -> Oreg stays in registers (runtime n_nt2 gave STACK:128)
  #pragma unroll
  for (int nt2 = 0; nt2 < NT_MAX; ++nt2) {
    if (nt2 >= n_nt2) break;
    int d0 = nt2 * 8 + tig * 2, d1 = d0 + 1;
    int gi0 = q0 + gid, gi8 = q0 + gid + 8;
    float o00 = Oreg[nt2*4+0]*invl_g,  o01 = Oreg[nt2*4+1]*invl_g;
    float o10 = Oreg[nt2*4+2]*invl_g8, o11 = Oreg[nt2*4+3]*invl_g8;
    if (out_q == nullptr) {
      out[(size_t)(nh * T + gi0) * hd + d0] = __float2half(o00);
      out[(size_t)(nh * T + gi0) * hd + d1] = __float2half(o01);
      out[(size_t)(nh * T + gi8) * hd + d0] = __float2half(o10);
      out[(size_t)(nh * T + gi8) * hd + d1] = __float2half(o11);
    } else {
      int c0 = h * hd + d0, c1 = h * hd + d1;   // round through fp16 first -> bit-matches
      int q00 = __float2int_rn(__half2float(__float2half(o00)) * proj_inv_scale);
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
// TEMPLATED on HDP_V (the V / output head dim, = pad(hd,32) -> 32 or 64) for the same reason
// the int8 kernel is templated on HD_PAD: with hdp_v a runtime arg, Oreg[] was indexed by the
// runtime n_nt2 and cuobjdump reported STACK:128, i.e. all 32 fp32 accumulators living in LOCAL
// memory. Vs was also sized for FA_MMA_MAXHD=64 regardless of the real hdp_v.
// TEMPLATED on WARPS as well as (HDP_V, BC). WARPS was previously fixed at FA_MMA_WARPS on the
// claim that it "measured flat on the int8 side" -- the (BC x WARPS) sweep over every real shape
// (data/attn_tile_sweep.json) contradicts that: int8 gains 1.19x at T=1024/hd=24 (2246 -> 1892 us)
// and 1.32x at T=256/hd=48 (336 -> 265 us) from WARPS=8, because more warps per CTA share ONE
// K/V smem tile, so the number of CTAs per (n,h) -- each of which streams all of K and V -- drops
// proportionally. Nothing about that mechanism is int8-specific; the int4 kernel has the same
// per-CTA K/V streaming, and the sweep confirmed int4 was pinned flat (2264 vs 2263 us) purely
// because MODIFF_FA_MMA4_DISPATCH ignored the knob.
template <int HDP_V, int BC, int WARPS>
__global__ void flash_attn_int4_mma_kernel_t(
    const int8_t* __restrict__ q4, const int8_t* __restrict__ k4, const int8_t* __restrict__ v,
    const float* __restrict__ sq, const float* __restrict__ sk, const float* __restrict__ sv,
    __half* __restrict__ out, int N, int H, int T, int hd, int hdp4, int hdp_v_rt, float softmax_scale,
    int8_t* __restrict__ out_q, float proj_inv_scale, int qout_bstride) {
  const int hdp_v = HDP_V;
  (void)hdp_v_rt;
  // out_q != nullptr => fused proj-quantize store: emit packed int4 token-major [b*T, qout_bstride]
  // (qout_bstride = K_pad/2 bytes/row), quantized by proj_inv_scale. K-pad tail bytes are pre-zeroed
  // by the caller (torch::zeros), so this only writes the real channel-pair bytes.
  const int nh = blockIdx.x;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int q0 = (blockIdx.y * WARPS + w) * FA_MMA_BR;
  const int rowb4 = hdp4 / 2;                          // packed bytes per Q/K row
  const int kb_stride = (size_t)T * rowb4;

  const int8_t* kb = k4 + (size_t)nh * T * rowb4;
  const int8_t* vb = v + (size_t)nh * hdp_v * T;       // pre-transposed [hdp_v, T]
  const float*  sqb = sq + (size_t)nh * T;
  const float*  skb = sk + (size_t)nh * T;
  const float*  svb = sv + (size_t)nh * hd;

  __shared__ int8_t Ks[2][BC * 32];             // packed int4 rows (<=32 bytes, hdp4<=64)
  __shared__ int8_t Vs[2][HDP_V * BC];          // int8 transposed Vs[buf][d*BC+j], exact size
  __shared__ int8_t Qs[WARPS * FA_MMA_BR * 32];
  __shared__ int8_t Ps[WARPS * FA_MMA_BR * BC];
  __shared__ float SKs[2][BC];                  // per-key K scales, staged per tile

  int8_t* Qsw = &Qs[w * FA_MMA_BR * rowb4];
  int8_t* Psw = &Ps[w * FA_MMA_BR * BC];

  // host enforces T % (WARPS*FA_MMA_BR) == 0 and grid.y == T/(WARPS*BR), so q0+BR <= T
  for (int idx = lane; idx < FA_MMA_BR * rowb4; idx += 32) {
    int row = idx / rowb4, col = idx % rowb4;
    Qsw[idx] = q4[(size_t)nh * kb_stride + (size_t)(q0 + row) * rowb4 + col];
  }

  float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
  constexpr int NT_MAX = HDP_V / 8;                     // compile-time -> Oreg stays in registers
  float Oreg[NT_MAX * 4];
#pragma unroll
  for (int i = 0; i < NT_MAX * 4; ++i) Oreg[i] = 0.f;
  const int n_nt2 = hd / 8;                             // <= NT_MAX (hd may be < HDP_V)
  const int nkt4 = hdp4 / 64;                           // int4 k-steps (K=64 each)
  const int NNT = BC / 8;
  const int NKV = T / BC;
  const int K16 = (BC * rowb4) / 16;             // 16B chunks per K tile (packed)
  // Same instruction-count folds as the int8 kernel (see its FA_LOG2E block): softmax_scale and
  // log2(e) fold into the Q scale so modiff_ex2 is a single MUFU.EX2, and log2(127) folds into the
  // running max so the P requantize scale and the PV 1/127 both disappear. The SASS census
  // showed this kernel spends ~98% of its issue slots on this fp32 epilogue, not on the mma.
  constexpr float FA4_LOG2E    = 1.4426950408889634f;
  constexpr float FA4_LOG2_127 = 6.98868468677217f;
  const float sqs0 = sqb[q0 + gid]     * softmax_scale * FA4_LOG2E;
  const float sqs1 = sqb[q0 + gid + 8] * softmax_scale * FA4_LOG2E;
  float svr[NT_MAX][2];
#pragma unroll
  for (int i = 0; i < NT_MAX; ++i) {
    const int d0 = i * 8 + tig * 2;
    svr[i][0] = (d0     < hd) ? svb[d0]     : 0.f;
    svr[i][1] = (d0 + 1 < hd) ? svb[d0 + 1] : 0.f;
  }
  const int V16 = (BC * hdp_v) / 16;             // 16B chunks per V tile

  // Hoist the loop-invariant Q A-fragment into registers (see the int8 kernel's note). Ks is
  // sized [BC*32] so rowb4 <= 32, i.e. hdp4 <= 64 and nkt4 == 1 -- one 4-register fragment.
  __syncwarp();                                         // Qsw filled by this warp's 32 lanes
  constexpr int NKT4_MAX = 1;
  unsigned Qa[NKT4_MAX][4];
#pragma unroll
  for (int ks = 0; ks < NKT4_MAX; ++ks) {
    if (ks >= nkt4) break;
    const int base = ks * 32;
    Qa[ks][0] = *(const unsigned*)&Qsw[(gid)     * rowb4 + base + tig * 4];
    Qa[ks][1] = *(const unsigned*)&Qsw[(gid + 8) * rowb4 + base + tig * 4];
    Qa[ks][2] = *(const unsigned*)&Qsw[(gid)     * rowb4 + base + tig * 4 + 16];
    Qa[ks][3] = *(const unsigned*)&Qsw[(gid + 8) * rowb4 + base + tig * 4 + 16];
  }

  auto load_kv = [&](int buf, int kt) {
    for (int c = threadIdx.x; c < K16; c += blockDim.x) {   // K packed [BC,rowb4]
      int off = c * 16;
      modiff_cp_async_cg(modiff_smem_ptr(&Ks[buf][off]),
                         (const uint4*)(kb + (size_t)kt * rowb4 + off), true);
    }
    for (int c = threadIdx.x; c < V16; c += blockDim.x) {   // V int8 transposed [hdp_v,BC] from [hdp_v,T]
      int off = c * 16, d = off / BC, j = off % BC;
      modiff_cp_async_cg(modiff_smem_ptr(&Vs[buf][off]),
                         (const uint4*)(vb + (size_t)d * T + kt + j), true);
    }
    for (int c = threadIdx.x; c < BC / 4; c += blockDim.x)   // 64 K scales = 16 chunks
      modiff_cp_async_cg(modiff_smem_ptr(&SKs[buf][c * 4]),
                         (const uint4*)(skb + kt + c * 4), true);
    __pipeline_commit();
  };

  load_kv(0, 0);
  for (int ktile = 0; ktile < NKV; ++ktile) {
    const int kt = ktile * BC;
    const int buf = ktile & 1;
    __pipeline_wait_prior(0);
    __syncthreads();
    if (ktile + 1 < NKV) load_kv((ktile + 1) & 1, (ktile + 1) * BC);
    int8_t* Ksb = Ks[buf];
    int8_t* Vsb = Vs[buf];

    // ---- QKᵀ (int4, m16n8k64.s4) -> Sreg ----
    float Sreg[BC / 8][4];
    for (int nt = 0; nt < NNT; ++nt) {
      int acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ks = 0; ks < NKT4_MAX; ++ks) {
        if (ks >= nkt4) break;
        int base = ks * 32;                              // 64 int4 = 32 bytes
        unsigned b[2];
        b[0] = *(const int*)&Ksb[(nt * 8 + gid) * rowb4 + base + tig * 4];
        b[1] = *(const int*)&Ksb[(nt * 8 + gid) * rowb4 + base + tig * 4 + 16];
        modiff_mma_m16n8k64_s4(acc, Qa[ks], b);          // Q from registers (hoisted)
      }
      // Only the per-key scale here; the per-row Q scale is applied inside the exp argument
      // below (exact because all scales are positive -- see the int8 kernel's note).
      const int c0 = nt * 8 + tig * 2;
      const float sk0 = SKs[buf][c0], sk1 = SKs[buf][c0 + 1];   // smem, not HBM
      Sreg[nt][0] = acc[0] * sk0;
      Sreg[nt][1] = acc[1] * sk1;
      Sreg[nt][2] = acc[2] * sk0;
      Sreg[nt][3] = acc[3] * sk1;
    }

    // ---- register-parallel online softmax (identical to int8) ----
    // Single accumulator chain, deliberately. A 4-way split (chain depth 8 -> 4) was measured
    // and is a REGRESSION here: it pushes the kernel from REG:83 to REG:96, which drops
    // occupancy from 3 CTA/SM (24 warps, 50%) to 2 CTA/SM (16 warps, 33%), and T=1024/hd=24
    // went 1973 -> 2032 us. The kernel is latency-bound, but on this shape occupancy buys more
    // latency hiding than instruction-level parallelism does.
    float lm0 = -INFINITY, lm1 = -INFINITY;
    for (int nt = 0; nt < NNT; ++nt) {
      lm0 = fmaxf(lm0, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      lm1 = fmaxf(lm1, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1)); lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
    lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1)); lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
    float mcur0 = fmaxf(m_run0, lm0), mcur1 = fmaxf(m_run1, lm1);
    float a_g  = modiff_ex2(sqs0 * (m_run0 - mcur0));
    float a_g8 = modiff_ex2(sqs1 * (m_run1 - mcur1));
    // one FFMA carries the Q scale + max subtraction + the 127 requantize factor (see int8)
    const float nb0 = FA4_LOG2_127 - sqs0 * mcur0, nb1 = FA4_LOG2_127 - sqs1 * mcur1;
    float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      const int c0 = nt * 8 + tig * 2;
      float p00 = modiff_ex2(fmaf(sqs0, Sreg[nt][0], nb0));
      float p01 = modiff_ex2(fmaf(sqs0, Sreg[nt][1], nb0));
      float p10 = modiff_ex2(fmaf(sqs1, Sreg[nt][2], nb1));
      float p11 = modiff_ex2(fmaf(sqs1, Sreg[nt][3], nb1));
      ls0 += p00 + p01; ls1 += p10 + p11;
      // p is already in [0,127] -- the 127 is folded into mq0/mq1, so there is no "* 127.f"
      // here. F2I.RN needs no clamp select (rn of a value <= 127.0f cannot reach 128), and the
      // two adjacent columns leave as a single 2-byte store.
      // cvt.pack.sat.s8.s32 replaces the shift+or (IMAD.SHL + LOP3) with one instruction and
      // saturates on the way, so no clamp select is needed either.
      unsigned r0 = modiff_pack2_s8(__float2int_rn(p00), __float2int_rn(p01));
      unsigned r1 = modiff_pack2_s8(__float2int_rn(p10), __float2int_rn(p11));
      *(short*)&Psw[gid       * BC + c0] = (short)r0;
      *(short*)&Psw[(gid + 8) * BC + c0] = (short)r1;
    }
    ls0 += __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
    l_run0 = l_run0 * a_g + ls0; l_run1 = l_run1 * a_g8 + ls1;
    m_run0 = mcur0; m_run1 = mcur1;
    __syncwarp();

    // Hoist the P A-fragment: it depends only on (ks, gid, tig), NOT on nt2, yet the PV loop
    // re-read it from Psw once per output tile -- n_nt2 * (BC/32) * 4 = up to 64 smem loads per
    // lane per key tile where 8 suffice. Together with the Q hoist above this cuts inner-loop
    // smem traffic ~2.6x (hd=48: 168 -> 64 loads/lane/tile).
    unsigned Pa[BC / 32][4];
#pragma unroll
    for (int ks = 0; ks < BC / 32; ++ks) {
      const int koff = ks * 32;
      Pa[ks][0] = *(const unsigned*)&Psw[(gid)     * BC + koff + tig * 4];
      Pa[ks][1] = *(const unsigned*)&Psw[(gid + 8) * BC + koff + tig * 4];
      Pa[ks][2] = *(const unsigned*)&Psw[(gid)     * BC + koff + tig * 4 + 16];
      Pa[ks][3] = *(const unsigned*)&Psw[(gid + 8) * BC + koff + tig * 4 + 16];
    }

    // ---- PV (int8, m16n8k32) ----
#pragma unroll
    for (int nt2 = 0; nt2 < NT_MAX; ++nt2) {
      if (nt2 >= n_nt2) break;                           // NT_MAX compile-time; n_nt2 <= NT_MAX
      Oreg[nt2 * 4 + 0] *= a_g;  Oreg[nt2 * 4 + 1] *= a_g;
      Oreg[nt2 * 4 + 2] *= a_g8; Oreg[nt2 * 4 + 3] *= a_g8;
      int acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ks = 0; ks < BC / 32; ++ks) {
        int koff = ks * 32;
        unsigned b[2];
        b[0] = *(const int*)&Vsb[(nt2 * 8 + gid) * BC + koff + tig * 4];
        b[1] = *(const int*)&Vsb[(nt2 * 8 + gid) * BC + koff + tig * 4 + 16];
        modiff_mma_m16n8k32(acc, Pa[ks], b);   // P from registers (hoisted out of the nt2 loop)
      }
      Oreg[nt2 * 4 + 0] += svr[nt2][0] * acc[0];   // 1/127 folded into l_run
      Oreg[nt2 * 4 + 1] += svr[nt2][1] * acc[1];
      Oreg[nt2 * 4 + 2] += svr[nt2][0] * acc[2];
      Oreg[nt2 * 4 + 3] += svr[nt2][1] * acc[3];
    }
  }

  float invl_g  = (l_run0 > 0.f) ? 1.f / l_run0 : 0.f;
  float invl_g8 = (l_run1 > 0.f) ? 1.f / l_run1 : 0.f;
  const int h = nh % H, n = nh / H;           // (n,h) for the token-major fused-quant store
#pragma unroll
  for (int nt2 = 0; nt2 < NT_MAX; ++nt2) {
    if (nt2 >= n_nt2) break;                  // keep Oreg register-resident in the epilogue too
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
// Dispatch to the hd_pad-templated kernel. Only 32 and 64 are needed: hd_pad is
// ceil(hd/32)*32 and the mma path requires hd_pad <= FA_MMA_MAXHD (=64), so hd<=32 -> 32 and
// 33..64 -> 64. Anything else must not reach here (callers TORCH_CHECK hd_pad <= FA_MMA_MAXHD).
// Pick WARPS per call: 8 when T is a multiple of 8*BR (halves K/V re-reads), else 4.
// grid.y must be computed with the SAME WARPS -- both go through modiff_fa_warps(T).
// Key-tile width, chosen once per process. 64 is the default; 32 trades ~30% more
// instructions per unit of work (every per-tile fixed cost doubles) for a smaller Sreg/Pa
// register footprint, which is what caps occupancy on this kernel. Only these two are
// instantiated, so anything else falls back to 64.
// Key-tile width, chosen per shape. Measured on A40 (batch 128, heads 8), int8 path:
//
//   hd_pad  T     BC=64    BC=32   winner
//   32      1024  1989.8   1881.5  32  (+6%)
//   64      256    285.1    299.4  64  (+5%)
//   64      64      52.1     49.2  32  (+6%)
//   64      1024  3637.9   3650.8  64  (tie)
//
// BC=32 costs +38% instructions per unit of work (every per-tile fixed cost doubles) and does
// NOT raise occupancy at WARPS=8 -- REG only falls 83 -> 68, and 4 CTAs of 256 threads need
// REG <= 64. It still wins at hd_pad=32 and at small T because NNT = BC/8 halves, which halves
// the depth of the row-max reduction chain that every exp, every P store and the whole PV mma
// wait on. That the win survives a 38% instruction increase is the clearest evidence that this
// kernel is latency-bound rather than issue-bound.
//
// MODIFF_FA_BC overrides for experiments; anything other than 32/64 falls back to the heuristic.
//
// A guard used to live here forcing BC=64 whenever WARPS==4, because BC=32 produced wrong results
// in that combination. That was a symptom of the 3-stage cp.async pipeline (see the FA_STAGES note
// in the kernel); with 2 stages every (BC, WARPS) pair is correct, so the guard is gone and BC=32
// is selectable again -- worth 46.6 vs 50.0 us on the T=64/hd=48 block (x5 instances).
static inline int modiff_fa_bc(int hd_pad, int T) {
  static const int forced = [] {
    const char* e = getenv("MODIFF_FA_BC");
    const int v = e ? atoi(e) : 0;
    return (v == 32 || v == 64) ? v : 0;
  }();
  if (forced) return forced;
  return (hd_pad <= 32 || T < 128) ? 32 : 64;
}

// Warps per CTA. 8 halves the CTA count per (n,h) so K/V smem is shared more widely, but it also
// quadruples register pressure per CTA, and register pressure is what caps occupancy here.
// MODIFF_FA_WARPS forces 4 or 8 for experiments; the default keeps the historical rule.
static inline int modiff_fa_warps(int T) {
  static const int forced = [] {
    const char* e = getenv("MODIFF_FA_WARPS");
    const int v = e ? atoi(e) : 0;
    return (v == 4 || v == 8) ? v : 0;
  }();
  const int w = forced ? forced : ((T % (8 * FA_MMA_BR) == 0) ? 8 : FA_MMA_WARPS);
  return (T % (w * FA_MMA_BR) == 0) ? w : FA_MMA_WARPS;   // grid.y must divide exactly
}

#define MODIFF_FA_MMA_LAUNCH(HD, W, BC, GRID, STREAM, ...)                                       \
  flash_attn_int8_mma_kernel_t<HD, W, BC><<<(GRID), (W)*32, 0, (STREAM)>>>(__VA_ARGS__)

#define MODIFF_FA_MMA_DISPATCH(GRID, W, STREAM, HDPAD, T_, ...)                                   \
  do {                                                                                           \
    const int bc_ = modiff_fa_bc((HDPAD), (T_));                                                  \
    if ((HDPAD) <= 32) {                                                                          \
      if ((W) == 8) {                                                                              \
        if (bc_ == 32) MODIFF_FA_MMA_LAUNCH(32, 8, 32, GRID, STREAM, __VA_ARGS__);                  \
        else           MODIFF_FA_MMA_LAUNCH(32, 8, 64, GRID, STREAM, __VA_ARGS__);                  \
      } else {                                                                                     \
        if (bc_ == 32) MODIFF_FA_MMA_LAUNCH(32, 4, 32, GRID, STREAM, __VA_ARGS__);                  \
        else           MODIFF_FA_MMA_LAUNCH(32, 4, 64, GRID, STREAM, __VA_ARGS__);                  \
      }                                                                                            \
    } else {                                                                                       \
      if ((W) == 8) {                                                                              \
        if (bc_ == 32) MODIFF_FA_MMA_LAUNCH(64, 8, 32, GRID, STREAM, __VA_ARGS__);                  \
        else           MODIFF_FA_MMA_LAUNCH(64, 8, 64, GRID, STREAM, __VA_ARGS__);                  \
      } else {                                                                                     \
        if (bc_ == 32) MODIFF_FA_MMA_LAUNCH(64, 4, 32, GRID, STREAM, __VA_ARGS__);                  \
        else           MODIFF_FA_MMA_LAUNCH(64, 4, 64, GRID, STREAM, __VA_ARGS__);                  \
      }                                                                                            \
    }                                                                                              \
  } while (0)

// int4 counterpart: same three template axes as the int8 path (HDP_V, BC, WARPS). WARPS used to be
// pinned to FA_MMA_WARPS here; see the kernel's header comment for the measurement that overturned
// that. Both paths now share modiff_fa_bc / modiff_fa_warps, so a heuristic change applies to both.
#define MODIFF_FA_MMA4_LAUNCH(HDPV, BC, W, GRID, STREAM, ...)                                     \
  flash_attn_int4_mma_kernel_t<HDPV, BC, W><<<(GRID), (W)*32, 0, (STREAM)>>>(__VA_ARGS__)

// Same per-shape BC choice as the int8 path. The int4 kernel's V/output dim is hdp_v, so that is
// what feeds the heuristic (hdp4 is always 64 -- the m16n8k64.s4 minimum -- and says nothing about
// the score-tile register footprint).
#define MODIFF_FA_MMA4_DISPATCH(GRID, W, STREAM, HDPV, T_, ...)                                    \
  do {                                                                                            \
    const int bc4_ = modiff_fa_bc((HDPV), (T_));                                                     \
    if ((HDPV) <= 32) {                                                                             \
      if ((W) == 8) {                                                                               \
        if (bc4_ == 32) MODIFF_FA_MMA4_LAUNCH(32, 32, 8, GRID, STREAM, __VA_ARGS__);                 \
        else            MODIFF_FA_MMA4_LAUNCH(32, 64, 8, GRID, STREAM, __VA_ARGS__);                 \
      } else {                                                                                      \
        if (bc4_ == 32) MODIFF_FA_MMA4_LAUNCH(32, 32, 4, GRID, STREAM, __VA_ARGS__);                 \
        else            MODIFF_FA_MMA4_LAUNCH(32, 64, 4, GRID, STREAM, __VA_ARGS__);                 \
      }                                                                                             \
    } else {                                                                                        \
      if ((W) == 8) {                                                                               \
        if (bc4_ == 32) MODIFF_FA_MMA4_LAUNCH(64, 32, 8, GRID, STREAM, __VA_ARGS__);                 \
        else            MODIFF_FA_MMA4_LAUNCH(64, 64, 8, GRID, STREAM, __VA_ARGS__);                 \
      } else {                                                                                      \
        if (bc4_ == 32) MODIFF_FA_MMA4_LAUNCH(64, 32, 4, GRID, STREAM, __VA_ARGS__);                 \
        else            MODIFF_FA_MMA4_LAUNCH(64, 64, 4, GRID, STREAM, __VA_ARGS__);                 \
      }                                                                                             \
    }                                                                                               \
  } while (0)

torch::Tensor flash_attn_int8_vt(torch::Tensor q, torch::Tensor k, torch::Tensor vt,
                                 torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                 double softmax_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && vt.is_cuda(), "flash_attn_int8_vt: q/k/vt must be CUDA");
  TORCH_CHECK(q.dim() == 4 && vt.dim() == 4, "q [N,H,T,hd_pad], vt [N,H,hd_pad,T]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && vt.is_contiguous(), "q/k/vt must be contiguous");
  const int N = q.size(0), H = q.size(1), T = q.size(2), hd_pad = q.size(3);
  const int hd = sv.size(-1);
  TORCH_CHECK(hd_pad <= FA_MMA_MAXHD && (T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0
              && (32 * hd_pad) % 16 == 0,   // 32 = smallest instantiated BC
              "flash_attn_int8_vt: mma-eligible shapes only");
  TORCH_CHECK(vt.size(2) == hd_pad && vt.size(3) == T, "vt must be [N,H,hd_pad,T]");
  auto out = torch::empty({N, H, T, hd}, torch::TensorOptions().dtype(torch::kFloat16).device(q.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int fa_w = modiff_fa_warps(T);
  dim3 grid(N * H, T / (fa_w * FA_MMA_BR));
  MODIFF_FA_MMA_DISPATCH(grid, fa_w, stream, hd_pad, T,
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
  const int fa_w = modiff_fa_warps(T);
  dim3 grid(N * H, T / (fa_w * FA_MMA_BR));
  MODIFF_FA_MMA_DISPATCH(grid, fa_w, stream, hd_pad, T,
      q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      (__half*)nullptr, N, H, T, hd, hd_pad, (float)softmax_scale,
      out_q.data_ptr<int8_t>(), proj_inv_scale, C);
  return out_q;   // int8 [b*T, C] token-major (ready for gemm_w8a8_awq_bias_res)
}

// ---- PACKED-input flash host entries. Read interleaved qkv [b,T,nh,3,hd] directly (fp16 ->
//      quantize on load with frozen per-tensor sq_c/sk_c + per-channel sv[d]; int8 -> plain gather).
//      Replace the separate aq_qtok/aq_vquant (or Route-1 from_i8) reshuffle + the qi/ki/vt HBM
//      round-trip. sv is f32 [hd] (per-channel V dequant). Dispatch on qkv.dtype(). ----
// HD_PAD is a template arg so the kernel's smem/registers are exact (see the kernel header);
// run_flash_packed keeps the old runtime-hd_pad signature and dispatches, so callers are unchanged.
template <typename TIn, int HD_PAD>
static inline void run_flash_packed_t(const TIn* qkv, const float* sv, __half* out, int8_t* out_q,
                                    int N, int H, int T, int hd, int hd_pad,
                                    float sq_c, float sk_c, float softmax_scale, float q_inv, float k_inv,
                                    float proj_inv_scale, int qout_stride, cudaStream_t stream) {
  dim3 grid(N * H, T / (FA_MMA_WARPS * FA_MMA_BR));
  // dynamic smem: 2 tensors (K,V) x 2 stages x [BC*hd] raw tiles (row stride hd)
  size_t smem = (size_t)2 * 2 * FA_MMA_BC * hd * sizeof(TIn);
  // Anything past 48 KB of DYNAMIC smem needs an explicit per-kernel opt-in, even though sm_86
  // allows ~100 KB per block. fp16 with hd=64 needs 32 KB dynamic on top of this kernel's 16.6 KB
  // static, which trips the default cap and surfaced only as a bare cudaErrorInvalidValue
  // ("invalid argument") at launch. The model's head dims (24/48) stay under it, so this was a
  // latent failure for hd=64 callers. Function-local static => runs once per instantiation.
  static const bool smem_optin = [] {
    // The opt-in value must fit the DEVICE's per-block total minus this kernel's static
    // allocation: sm_86 caps static+dynamic at 100 KB, and asking for more than that makes
    // cudaFuncSetAttribute fail (a request of 96 KB does, since static is ~16.6 KB). The
    // failure is silent unless checked, after which the launch still uses the 48 KB default
    // and dies with a bare cudaErrorInvalidValue.
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa{};
    const void* fn = reinterpret_cast<const void*>(&flash_attn_int8_packed_mma_kernel<TIn, HD_PAD>);
    if (cudaFuncGetAttributes(&fa, fn) == cudaSuccess) {
      const int req = optin - (int)fa.sharedSizeBytes;
      if (req > 48 * 1024)
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, req);
    }
    cudaGetLastError();          // clear any probe error; the launch reports real failures
    return true;
  }();
  (void)smem_optin;
  flash_attn_int8_packed_mma_kernel<TIn, HD_PAD><<<grid, FA_MMA_WARPS * 32, smem, stream>>>(
      qkv, sv, out, N, H, T, hd, hd_pad, sq_c, sk_c, softmax_scale, q_inv, k_inv,
      out_q, proj_inv_scale, qout_stride);
}

// Runtime-hd_pad entry: pick the instantiation. Only 32 and 64 exist -- hd_pad is ceil(hd/32)*32 and
// the packed mma path requires hd_pad <= FA_MMA_MAXHD (64), same as the unpacked kernel.
template <typename TIn>
static inline void run_flash_packed(const TIn* qkv, const float* sv, __half* out, int8_t* out_q,
                                    int N, int H, int T, int hd, int hd_pad,
                                    float sq_c, float sk_c, float softmax_scale, float q_inv, float k_inv,
                                    float proj_inv_scale, int qout_stride, cudaStream_t stream) {
  if (hd_pad <= 32)
    run_flash_packed_t<TIn, 32>(qkv, sv, out, out_q, N, H, T, hd, hd_pad, sq_c, sk_c,
                                softmax_scale, q_inv, k_inv, proj_inv_scale, qout_stride, stream);
  else
    run_flash_packed_t<TIn, 64>(qkv, sv, out, out_q, N, H, T, hd, hd_pad, sq_c, sk_c,
                                softmax_scale, q_inv, k_inv, proj_inv_scale, qout_stride, stream);
}

static inline void check_packed(torch::Tensor qkv, torch::Tensor sv, int64_t hd_pad,
                                int& N, int& T, int& H, int& hd) {
  TORCH_CHECK(qkv.is_cuda() && sv.is_cuda(), "flash_attn_int8_packed: qkv/sv must be CUDA");
  TORCH_CHECK(qkv.dim() == 5, "qkv must be [b,T,nh,3,hd]");
  TORCH_CHECK(qkv.size(3) == 3, "qkv dim 3 must be 3 (q,k,v)");
  TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
  TORCH_CHECK(qkv.dtype() == torch::kHalf || qkv.dtype() == torch::kChar, "qkv must be fp16 or int8");
  TORCH_CHECK(sv.dtype() == torch::kFloat32 && sv.is_contiguous(), "sv must be f32 contiguous");
  N = qkv.size(0); T = qkv.size(1); H = qkv.size(2); hd = qkv.size(4);
  TORCH_CHECK(sv.numel() == hd, "sv must be [hd]");
  TORCH_CHECK(hd_pad <= FA_MMA_MAXHD && (T % (FA_MMA_WARPS * FA_MMA_BR)) == 0 && (hd % 8) == 0
              && (FA_MMA_BC * hd_pad) % 16 == 0, "flash_attn_int8_packed: mma-eligible shapes only");
  const int elem = (qkv.dtype() == torch::kHalf) ? 2 : 1;
  TORCH_CHECK((hd * elem) % 16 == 0, "flash_attn_int8_packed: per-token bytes (hd*sizeof) must be a "
              "multiple of 16 for cp.async (fp16 always ok; int8 needs hd%16==0)");
}

torch::Tensor flash_attn_int8_packed_vt(torch::Tensor qkv, torch::Tensor sv, int64_t hd_pad,
                                        double sq_c, double sk_c, double softmax_scale) {
  int N, T, H, hd;
  check_packed(qkv, sv, hd_pad, N, T, H, hd);
  auto out = torch::empty({N, H, T, hd}, torch::TensorOptions().dtype(torch::kFloat16).device(qkv.device()));
  const float sqf = (float)sq_c, skf = (float)sk_c, ssf = (float)softmax_scale;
  const float q_inv = 1.f / sqf, k_inv = 1.f / skf;
  auto* op_ = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  const float* svp = sv.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (qkv.dtype() == torch::kHalf)
    run_flash_packed<__half>(reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), svp, op_,
                             nullptr, N, H, T, hd, (int)hd_pad, sqf, skf, ssf, q_inv, k_inv, 0.f, 0, stream);
  else
    run_flash_packed<int8_t>(qkv.data_ptr<int8_t>(), svp, op_,
                             nullptr, N, H, T, hd, (int)hd_pad, sqf, skf, ssf, q_inv, k_inv, 0.f, 0, stream);
  return out;   // fp16 [N,H,T,hd]
}

torch::Tensor flash_attn_int8_packed_vt_qout(torch::Tensor qkv, torch::Tensor sv, int64_t hd_pad,
                                             double sq_c, double sk_c, double softmax_scale,
                                             double proj_a_scale) {
  int N, T, H, hd;
  check_packed(qkv, sv, hd_pad, N, T, H, hd);
  const int C = H * hd;   // int8 proj K == C (C % 64 == 0)
  auto out_q = torch::empty({(long)N * T, C}, torch::TensorOptions().dtype(torch::kChar).device(qkv.device()));
  const float sqf = (float)sq_c, skf = (float)sk_c, ssf = (float)softmax_scale;
  const float q_inv = 1.f / sqf, k_inv = 1.f / skf;
  const float proj_inv_scale = 1.f / (float)proj_a_scale;
  int8_t* oqp = out_q.data_ptr<int8_t>();
  const float* svp = sv.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (qkv.dtype() == torch::kHalf)
    run_flash_packed<__half>(reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), svp, nullptr,
                             oqp, N, H, T, hd, (int)hd_pad, sqf, skf, ssf, q_inv, k_inv, proj_inv_scale, C, stream);
  else
    run_flash_packed<int8_t>(qkv.data_ptr<int8_t>(), svp, nullptr,
                             oqp, N, H, T, hd, (int)hd_pad, sqf, skf, ssf, q_inv, k_inv, proj_inv_scale, C, stream);
  return out_q;   // int8 [b*T, C] token-major
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
    const int fa_w = modiff_fa_warps(T);
    dim3 grid(N * H, T / (fa_w * FA_MMA_BR));
    MODIFF_FA_MMA_DISPATCH(grid, fa_w, stream, hd_pad, T,
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
  const int fa_w4 = modiff_fa_warps(T);                // must match grid.y (see the int8 path)
  dim3 grid(N * H, T / (fa_w4 * FA_MMA_BR));
  MODIFF_FA_MMA4_DISPATCH(grid, fa_w4, stream, hdp_v, T,
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
  const int fa_w4 = modiff_fa_warps(T);                // must match grid.y (see the int8 path)
  dim3 grid(N * H, T / (fa_w4 * FA_MMA_BR));
  MODIFF_FA_MMA4_DISPATCH(grid, fa_w4, stream, hdp_v, T,
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
  const int fa_w4 = modiff_fa_warps(T);                // must match grid.y (see the int8 path)
  dim3 grid(N * H, T / (fa_w4 * FA_MMA_BR));
  MODIFF_FA_MMA4_DISPATCH(grid, fa_w4, stream, hdp_v, T,
      q4.data_ptr<int8_t>(), k4.data_ptr<int8_t>(), vt.data_ptr<int8_t>(),
      sq.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      N, H, T, hd, (int)hdp4, hdp_v, (float)softmax_scale, (int8_t*)nullptr, 0.f, 0);
  return out;
}
