// ============================================================================================
// DUPLICATED HEADER -- MODIFF copy.  Twin: csrc/baseline/common/mma_int8.cuh
//
// int8/int4 tensor-core MMA + async-copy primitives.
//
// This file is device-side only (templates and __device__ inlines), so both datapaths can carry
// their own copy without any symbol collision. The copies were made deliberately: csrc/ is split
// into a MoDiff tree and a baseline tree so each datapath can be read, edited and profiled without
// the other in the way, and a shared include directory would have re-coupled them.
//
// THE COST, stated because it is real: these copies can DIVERGE. Anything numerical changed here
// must be changed in the twin, or the two datapaths stop being comparable -- and every A/B in
// docs/ compares them. `diff csrc/baseline/... csrc/modiff/...` is the check; it should come back
// empty for every file whose header says "identical to twin" below.
//
// STATUS: identical to twin (byte-for-byte at the time of the split, 2026-08-12).
// ============================================================================================
// =========================================================================
// int8/int4 tensor-core MMA + async-copy primitives for the fused flash
// attention kernel (csrc/kernels/flash_attn_int8.cu).
//
// The ldmatrix / cp.async / mma inlines are ported verbatim from AWQ's
// QServe-derived W8A8 GEMM
//   /workspace/llm-awq/awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu
// (Apache-2.0; see that file's header for the AWQ citation). They are kept
// here so the flash kernel can switch its QKᵀ / AV inner products from the
// portable __dp4a path (used for the first correct implementation) to the
// m16n8k32.s8.s8.s32 tensor-core path in the performance milestone (M7),
// without pulling in the whole AWQ GEMM.
//
// Also provides small, always-correct __dp4a helpers used by the v1 kernel.
// =========================================================================
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define MODIFF_L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define MODIFF_L2_CACHEHINT(size)
#endif

// ---- __dp4a int8x4 -> int32 dot product (portable, correctness-first) ----
// Contract two int8 vectors of length K (K a multiple of 4) into int32.
__device__ __forceinline__ int dp4a_i8(const int8_t* a, const int8_t* b, int K) {
  int acc = 0;
  const int* a4 = reinterpret_cast<const int*>(a);
  const int* b4 = reinterpret_cast<const int*>(b);
#pragma unroll 4
  for (int k = 0; k < (K >> 2); ++k) {
    acc = __dp4a(a4[k], b4[k], acc);
  }
  return acc;
}

// ---- AWQ ldmatrix / cp.async / mma inlines (for the M7 tensor-core pass) ----
__device__ __forceinline__ uint32_t modiff_smem_ptr(void const* const ptr) {
  uint32_t smem_int_ptr;
  asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_int_ptr)
      : "l"(ptr));
  return smem_int_ptr;
}

__device__ __forceinline__ void modiff_ldmatrix_x4(int8_t* dst, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(((unsigned*)dst)[0]), "=r"(((unsigned*)dst)[1]),
                 "=r"(((unsigned*)dst)[2]), "=r"(((unsigned*)dst)[3])
               : "r"(addr));
}

__device__ __forceinline__ void modiff_ldmatrix_x4_trans(int8_t* dst, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(((unsigned*)dst)[0]), "=r"(((unsigned*)dst)[1]),
                 "=r"(((unsigned*)dst)[2]), "=r"(((unsigned*)dst)[3])
               : "r"(addr));
}

__device__ __forceinline__ void modiff_cp_async_cg(uint32_t smem_int_ptr,
                                                   const uint4* __restrict__ src, bool mask) {
  const int cp_size = 16;
  asm volatile("{"
               "  .reg .pred p;"
               "  setp.ne.b32 p, %0, 0;"
               "  @p cp.async.cg.shared.global" MODIFF_L2_CACHEHINT(128) " [%1], [%2], %3;"
               "}" ::"r"((int)mask),
               "r"(smem_int_ptr), "l"(src), "n"(cp_size));
}

// 8-byte cp.async, for tensors whose per-token bytes are a multiple of 8 but not of 16.
//
// `.ca` and not `.cg`: the cache-global qualifier is only legal at cp-size 16, so a narrower copy has
// to go through L1. That is the cost of this path, and the reason it is not the default.
//
// Why it exists: the int8 packed flash gather addresses each (head, q|k|v) slice at
// `(h * 3 + j) * hd` elements, so with hd = 24 the odd slices land on 8-byte-but-not-16-byte
// boundaries and a 16 B cp.async is illegal however the loop is written. hd = 24 is this model's
// dominant attention width (5 blocks at T=1024), so the alternative to 8 B copies is not using the
// gather path there at all. See docs/aq_fusion_2026-08-12.
__device__ __forceinline__ void modiff_cp_async_ca8(uint32_t smem_int_ptr,
                                                    const uint2* __restrict__ src, bool mask) {
  const int cp_size = 8;
  asm volatile("{"
               "  .reg .pred p;"
               "  setp.ne.b32 p, %0, 0;"
               "  @p cp.async.ca.shared.global" MODIFF_L2_CACHEHINT(64) " [%1], [%2], %3;"
               "}" ::"r"((int)mask),
               "r"(smem_int_ptr), "l"(src), "n"(cp_size));
}

// ---- softmax/requantize primitives, hand-written because the CUDA library versions cost
// extra instructions that show up clearly in the SASS census of the flash kernels. ----

// A softmax exponential in ONE instruction. Two layers of overhead are stripped here, both
// found by reading the flash kernel's SASS:
//   * exp2f() adds a library range check and an argument fixup around MUFU.EX2;
//   * even ex2.approx.f32 (without .ftz) emits subnormal handling -- ptxas generates
//       FSETP.GEU P, x, -126 ; @!P FMUL x, x, 0.5 ; MUFU.EX2 ; @!P FMUL r, r, r
//     i.e. 1 predicate + 2 conditional multiplies per call, which measured 34 FSETP + 68 FMUL
//     per lane per key tile -- 17% of the loop body.
// .ftz is exact here rather than a tolerance trade: the result feeds an int8 requantize with a
// range of [0,127], and an argument below -126 means 2^x is subnormal, so 127*2^x rounds to 0
// either way. Callers must keep the argument <= 0 (it is: score - running_max).
__device__ __forceinline__ float modiff_ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Pack two s32 into the low two bytes of one register, with saturation, in ONE instruction.
// Replaces the shift+or pair (IMAD.SHL + LOP3) that `lo | (hi << 8)` compiles to, and the
// saturation makes any explicit clamp dead. Byte order per PTX: d[7:0]=sat(b), d[15:8]=sat(a).
__device__ __forceinline__ unsigned modiff_pack2_s8(int lo, int hi) {
  // PTX: the 8-bit form of cvt.pack takes FOUR operands and a .b32 tail type --
  //   cvt.pack.sat.s8.s32.b32 d, a, b, c;  =>  d = { c[15:0], sat_s8(a), sat_s8(b) }
  // so `lo` must be the SECOND source to land in d[7:0]. The 3-operand spelling only exists
  // for the 16-bit conversions and ptxas rejects it here.
  unsigned d, z = 0u;
  asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(hi), "r"(lo), "r"(z));
  return d;
}

// m16n8k32.row.col.s32.s8.s8.s32 : C[16x8] += A[16x32] * B[8x32]^T  (A=4 regs, B=2 regs, each 4 int8/reg)
__device__ __forceinline__ void modiff_mma_m16n8k32(void* C, void* A, void* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=r"(((int*)C)[0]), "=r"(((int*)C)[1]), "=r"(((int*)C)[2]), "=r"(((int*)C)[3])
      : "r"(((unsigned*)A)[0]), "r"(((unsigned*)A)[1]), "r"(((unsigned*)A)[2]),
        "r"(((unsigned*)A)[3]), "r"(((unsigned*)B)[0]), "r"(((unsigned*)B)[1]),
        "r"(((int*)C)[0]), "r"(((int*)C)[1]), "r"(((int*)C)[2]), "r"(((int*)C)[3]));
}

// Same, but with a ZERO addend: D = A*B instead of C += A*B. Uses RZ for the C operand, so the
// first mma of a blockwise group can OVERWRITE its int32 accumulator instead of the kernel first
// zeroing it. That removes one MOV per accumulator per flush -- 64 MOVs per flush per thread in
// the 128x128/8-warp config -- which is pure overhead on the blockwise path's hot loop.
__device__ __forceinline__ void modiff_mma_m16n8k32_zero(void* C, void* A, void* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=r"(((int*)C)[0]), "=r"(((int*)C)[1]), "=r"(((int*)C)[2]), "=r"(((int*)C)[3])
      : "r"(((unsigned*)A)[0]), "r"(((unsigned*)A)[1]), "r"(((unsigned*)A)[2]),
        "r"(((unsigned*)A)[3]), "r"(((unsigned*)B)[0]), "r"(((unsigned*)B)[1]),
        "r"(0));   // ptxas folds a proven-zero addend to RZ
}

// m16n8k64.row.col.s32.s4.s4.s32 : C[16x8] += A[16x64] * B[8x64]^T  (A=4 regs, B=2 regs, each 8 int4/reg)
__device__ __forceinline__ void modiff_mma_m16n8k64_s4(void* C, void* A, void* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=r"(((int*)C)[0]), "=r"(((int*)C)[1]), "=r"(((int*)C)[2]), "=r"(((int*)C)[3])
      : "r"(((unsigned*)A)[0]), "r"(((unsigned*)A)[1]), "r"(((unsigned*)A)[2]),
        "r"(((unsigned*)A)[3]), "r"(((unsigned*)B)[0]), "r"(((unsigned*)B)[1]),
        "r"(((int*)C)[0]), "r"(((int*)C)[1]), "r"(((int*)C)[2]), "r"(((int*)C)[3]));
}
