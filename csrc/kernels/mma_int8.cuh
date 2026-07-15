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

// m16n8k32.row.col.s32.s8.s8.s32 : C[16x8] += A[16x32] * B[8x32]^T
__device__ __forceinline__ void modiff_mma_m16n8k32(void* C, void* A, void* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=r"(((int*)C)[0]), "=r"(((int*)C)[1]), "=r"(((int*)C)[2]), "=r"(((int*)C)[3])
      : "r"(((unsigned*)A)[0]), "r"(((unsigned*)A)[1]), "r"(((unsigned*)A)[2]),
        "r"(((unsigned*)A)[3]), "r"(((unsigned*)B)[0]), "r"(((unsigned*)B)[1]),
        "r"(((int*)C)[0]), "r"(((int*)C)[1]), "r"(((int*)C)[2]), "r"(((int*)C)[3]));
}
