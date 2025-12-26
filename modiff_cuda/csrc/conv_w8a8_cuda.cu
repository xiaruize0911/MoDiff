#include "utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "cp_async.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"
#include "qgemm/w8a8/gemm_utils.cuh"

#define PACK_SIZE_C16 8 
#define PACK_SIZE_C32 4

template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, 
          uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void load_A_conv(
    const int8_t *A_base,
    uint32_t m_base,
    uint32_t M, uint32_t H_out, uint32_t W_out,
    uint32_t stride_n, uint32_t stride_h, uint32_t stride_w, uint32_t stride_c,
    uint32_t K,
    uint32_t offset_K,
    smem_t<swizzle_mode, stride> smem
) {
  // Simplified: use linear GEMM-style loading (treats input as contiguous M×K matrix)
  // TODO: Implement proper im2col-style spatial indexing for convolution
  const int8_t *ptr_base = A_base + m_base * K + offset_K;
  load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp_per_iter, 
                      smem_iters_row, smem_iters_col, swizzle_mode, stride>(ptr_base, K, smem);
}

template <uint32_t CTA_M, uint32_t CTA_N, uint32_t CTA_K, uint32_t WARP_M, uint32_t WARP_N, uint32_t CTA_STRIDE, OutputDtype output_dtype=OutputDtype::kFloat16, uint32_t K_STAGE=2>
__global__ void Conv2dInt8Kernel(
    const int8_t *__restrict__ A, 
    const int8_t *__restrict__ B,
    half *__restrict__ C16, 
    const half *__restrict__ scale_A, 
    const half *__restrict__ scale_B,
    const int M, const int N, const int K,
    const int R, const int S,
    const int H_out, const int W_out,
    const int stride_n, const int stride_ah, const int stride_aw,
    const int stride_br, const int stride_bs
)
{
  static_assert(K_STAGE > 1);

  constexpr uint32_t num_warps_m = CTA_M / WARP_M;
  constexpr uint32_t num_warps_n = CTA_N / WARP_N;
  constexpr uint32_t num_warps = num_warps_m * num_warps_n;
  constexpr uint32_t num_tiles_m = WARP_M / MMA_M;
  constexpr uint32_t num_tiles_n = WARP_N / MMA_N;
  constexpr uint32_t num_tiles_k = CTA_K / MMA_K;

  constexpr uint32_t AB_SMEM_STRIDE = CTA_K;
  constexpr uint32_t C_SMEM_STRIDE = CTA_N;

  uint32_t blockIdx_m = (blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : blockIdx.y;
  uint32_t blockIdx_n = blockIdx.z * gridDim.x + blockIdx.x;

  if (blockIdx_m >= M / CTA_M || blockIdx_n >= N / CTA_N) return;

  // Shared memory: declare as 1D array  
  extern __shared__ int8_t smem_raw[];

  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  int32_t RC[num_tiles_m][num_tiles_n][8];
  #pragma unroll
  for(int i=0; i<num_tiles_m; ++i)
    for(int j=0; j<num_tiles_n; ++j)
      for(int k=0; k<8; ++k) RC[i][j][k] = 0;

  constexpr uint32_t PACK_SIZE_B128 = sizeof(b128_t);  // 16 bytes
  
  // All offsets must be in units of b128_t for smem_t!
  // Raw byte offsets:
  constexpr uint32_t B_smem_byte_off = CTA_M * AB_SMEM_STRIDE;  
  constexpr uint32_t smem_stage_byte_off = (CTA_M + CTA_N) * AB_SMEM_STRIDE;
  
  // Convert to b128_t units for smem_t pointer arithmetic:
  constexpr uint32_t B_smem_idx_off = B_smem_byte_off / PACK_SIZE_B128;
  constexpr uint32_t smem_stage_off = smem_stage_byte_off / PACK_SIZE_B128;

  constexpr SwizzleMode swizzle_mode_AB = (AB_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  
  // smem_t constructor will cast to b128_t*, so pointer arithmetic is in b128_t units
  using b128_ptr = b128_t*;
  b128_ptr smem_base = reinterpret_cast<b128_ptr>(smem_raw);
  
  smem_t<swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE> current_smem_A(smem_base);
  smem_t<swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE> current_smem_B(smem_base + B_smem_idx_off);

  constexpr uint32_t global_to_shared_line_lanes = (AB_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp = (AB_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t A_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t B_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t A_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp);
  constexpr uint32_t B_smem_iters_col = CTA_N / (num_warps * global_to_shared_copy_lines_per_warp);

  uint32_t m_warp_base = blockIdx_m * CTA_M + (CTA_M / num_warps) * warp_id;

  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
        
        const int8_t *A_base = A + (r * stride_ah + s * stride_aw);
        const int8_t *B_base = B + (r * stride_br + s * stride_bs);
        const int8_t *B_warp_ptr = B_base + blockIdx_n * CTA_N * K + CTA_N / num_warps * warp_id * K;

        uint32_t smem_store_idx = K_STAGE - 1, smem_store_off = 0;
        uint32_t smem_load_idx = 0, smem_load_off = 0;

        #pragma unroll
        for (uint32_t stage = 0; stage < K_STAGE; stage++) {
            smem_store_idx = (smem_store_idx + 1) % K_STAGE;
            smem_store_off = smem_store_idx * smem_stage_off;

            current_smem_A.set_base(smem_base + smem_store_off);
            current_smem_B.set_base(smem_base + smem_store_off + B_smem_idx_off);

            load_A_conv<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
                A_base, blockIdx_m * CTA_M, M, H_out, W_out, stride_n, stride_ah, stride_aw, 1, K, stage * CTA_K, current_smem_A);

            load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
                B_warp_ptr + stage * CTA_K, K, current_smem_B);
            
            cp_async::commit_group();
        }

        cp_async::wait_group<K_STAGE - 1>();
        __syncthreads();

        uint32_t reg_store_idx = 0;
        uint32_t reg_load_idx = 1;
        uint32_t RA[2][num_tiles_m][4];
        uint32_t RB[2][num_tiles_n][4];

        current_smem_A.set_base(smem_base + smem_load_off);
        current_smem_B.set_base(smem_base + smem_load_off + B_smem_idx_off);

        share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_A, RA[reg_store_idx], 0);
        share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_B, RB[reg_store_idx], 0);

        reg_store_idx ^= 1; reg_load_idx ^= 1;
        share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_A, RA[reg_store_idx], 2);
        share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_B, RB[reg_store_idx], 2);
        tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

        #pragma unroll
        for (uint32_t offset_K = K_STAGE * CTA_K; offset_K < K; offset_K += CTA_K) {
            smem_store_idx = (smem_store_idx + 1) % K_STAGE;
            smem_store_off = smem_store_idx * smem_stage_off;  // In b128_t units
            current_smem_A.set_base(smem_base + smem_store_off);
            current_smem_B.set_base(smem_base + smem_store_off + B_smem_idx_off);

            load_A_conv<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
                A_base, blockIdx_m * CTA_M, M, H_out, W_out, stride_n, stride_ah, stride_aw, 1, K, offset_K, current_smem_A);

            load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(
                B_warp_ptr + offset_K, K, current_smem_B);
            cp_async::commit_group();
            cp_async::wait_group<K_STAGE - 1>();
            __syncthreads();

            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;  // In b128_t units
            current_smem_A.set_base(smem_base + smem_load_off);
            current_smem_B.set_base(smem_base + smem_load_off + B_smem_idx_off);

            #pragma unroll
            for (uint32_t k = 0; k < num_tiles_k; k += 2) {
                reg_store_idx ^= 1; reg_load_idx ^= 1;
                share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_A, RA[reg_store_idx], 2 * k);
                share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_B, RB[reg_store_idx], 2 * k);
                tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

                reg_store_idx ^= 1; reg_load_idx ^= 1;
                share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_A, RA[reg_store_idx], 2 * k + 2);
                share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_B, RB[reg_store_idx], 2 * k + 2);
                tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
            }
            __syncthreads();
        }

        if constexpr (K_STAGE >= 2) {
            cp_async::wait_group<0>();
            __syncthreads();
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;  // In b128_t units
            current_smem_A.set_base(smem_base + smem_load_off);
            current_smem_B.set_base(smem_base + smem_load_off + B_smem_idx_off);

            #pragma unroll
            for (uint32_t k = 0; k < num_tiles_k; k += 2) {
                reg_store_idx ^= 1; reg_load_idx ^= 1;
                share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_A, RA[reg_store_idx], 2 * k);
                share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_B, RB[reg_store_idx], 2 * k);
                tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);

                reg_store_idx ^= 1; reg_load_idx ^= 1;
                share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_A, RA[reg_store_idx], 2 * k + 2);
                share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE>(current_smem_B, RB[reg_store_idx], 2 * k + 2);
                tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
            }
        }
        __syncthreads();
    }
  }

  half *C16_warp_base_ptr = C16 + blockIdx_m * CTA_M * N + CTA_M / num_warps * warp_id * N + blockIdx_n * CTA_N;
  
  constexpr uint32_t global_to_shared_line_lanes_C16 = (C_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C16 = (C_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t C16_smem_iters_row = C_SMEM_STRIDE / (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
  constexpr uint32_t C16_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp_C16);

  const half *scale_A_warp_ptr = scale_A + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
  const half *scale_B_warp_ptr = scale_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;

  float a_scale = 1.0f;
  float2 b_scale = {1.0f, 1.0f};
  float2 psums = {0.0f, 0.0f};

  #pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; i++) {
    #pragma unroll
    for (uint32_t j = 0; j < num_tiles_n; j++) {
        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4));
        b_scale = __half22float2(*reinterpret_cast<const half2*>(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4)));
        psums = make_float2(__int2float_rn(RC[i][j][0]), __int2float_rn(RC[i][j][1]));
        psums.x = psums.x * a_scale * b_scale.x;
        psums.y = psums.y * a_scale * b_scale.y;
        ((half2*)RC[i][j])[0] = __float22half2_rn(psums);
        
        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4 + 8));
        psums = make_float2(__int2float_rn(RC[i][j][2]), __int2float_rn(RC[i][j][3]));
        psums.x = psums.x * a_scale * b_scale.x;
        psums.y = psums.y * a_scale * b_scale.y;
        ((half2*)RC[i][j])[2] = __float22half2_rn(psums);

        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4));
        b_scale = __half22float2(*reinterpret_cast<const half2*>(scale_B_warp_ptr + j * MMA_N + 2 * (lane_id % 4) + 8));
        psums = make_float2(__int2float_rn(RC[i][j][4]), __int2float_rn(RC[i][j][5]));
        psums.x = psums.x * a_scale * b_scale.x;
        psums.y = psums.y * a_scale * b_scale.y;
        ((half2*)RC[i][j])[4] = __float22half2_rn(psums);

        a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M + lane_id / 4 + 8));
        psums = make_float2(__int2float_rn(RC[i][j][6]), __int2float_rn(RC[i][j][7]));
        psums.x = psums.x * a_scale * b_scale.x;
        psums.y = psums.y * a_scale * b_scale.y;
        ((half2*)RC[i][j])[6] = __float22half2_rn(psums);
    }
  }

  // C buffer can REUSE the A/B buffer space since we're done with matrix multiply
  // We only need space for one CTA tile of output (CTA_M x CTA_N halves)
  constexpr uint32_t C_smem_byte_off = 0;  // Reuse from start
  constexpr uint32_t C_smem_idx_off = C_smem_byte_off / PACK_SIZE_B128;
  
  constexpr SwizzleMode swizzle_mode_C16 = (C_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_C16, C_SMEM_STRIDE / PACK_SIZE_C16> smem_C16(smem_base + C_smem_idx_off);

  #pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; i++) {
    #pragma unroll
    for (uint32_t j = 0; j < num_tiles_n; j++) {
        uint32_t offset_C1 = smem_C16.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N / PACK_SIZE_C16));
        
        ((int32_t*)(smem_C16.base + offset_C1))[lane_id % 4] = RC[i][j][0];
        ((int32_t*)(smem_C16.base + offset_C1 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][2];

        uint32_t offset_C2 = smem_C16.get_permuted_offset(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N / PACK_SIZE_C16) + 1);
        
        ((int32_t*)(smem_C16.base + offset_C2))[lane_id % 4] = RC[i][j][4];
        ((int32_t*)(smem_C16.base + offset_C2 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][6];
    }
  }

  __syncthreads();

  half *C_lane_ptr = C16_warp_base_ptr + lane_id / global_to_shared_line_lanes_C16 * N + lane_id % global_to_shared_line_lanes_C16 * PACK_SIZE_C16;
  uint32_t offset_C = smem_C16.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_C16 * C16_smem_iters_col + lane_id / global_to_shared_line_lanes_C16, lane_id % global_to_shared_line_lanes_C16);

  #pragma unroll
  for (uint32_t i = 0; i < C16_smem_iters_col; i++) {
    #pragma unroll
    for (uint32_t j = 0; j < C16_smem_iters_row; j++) {
        smem_C16.store_128b(offset_C, C_lane_ptr);
        C_lane_ptr += (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
        offset_C = smem_C16.advance_offset_by_column<global_to_shared_line_lanes_C16>(offset_C);
    }
    offset_C = smem_C16.advance_offset_by_row<global_to_shared_copy_lines_per_warp_C16>(offset_C - (C16_smem_iters_row * global_to_shared_line_lanes_C16));
    C_lane_ptr += ((global_to_shared_copy_lines_per_warp_C16 * N) - (C16_smem_iters_row * global_to_shared_line_lanes_C16 * PACK_SIZE_C16));
  }
}

void conv2d_w8a8_cuda_run(
    const int8_t *A, const int8_t *B, half *C,
    const half *scale_A, const half *scale_B,
    int M, int N, int K, int R, int S,
    int H_out, int W_out,
    int stride_n, int stride_ah, int stride_aw,
    int stride_br, int stride_bs
) {
    dim3 grid(N / 128, M / 128, 1); 
    dim3 block(32, 8, 1);  // threadIdx.x = lane (0-31), threadIdx.y = warp (0-7)
    // Shared memory layout:
    // - A/B buffers for 2 stages: 2 * (128 + 128) * 64 = 32KB
    // - C buffer reuses A/B space (written after all loads complete)
    // - Total: 32KB
    constexpr int CTA_M = 128, CTA_N = 128, CTA_K = 64;
    size_t smem_size = 32768;  // 32KB
    
    auto kernel_func = Conv2dInt8Kernel<128, 128, 64, 64, 64, 64, OutputDtype::kFloat16, 2>;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    kernel_func<<<grid, block, smem_size>>>(
        A, B, C, scale_A, scale_B, M, N, K, R, S, H_out, W_out, stride_n, stride_ah, stride_aw, stride_br, stride_bs
    );
}
