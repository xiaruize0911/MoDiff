#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

// Debug kernel to print shared memory layout and offsets
__global__ void debug_smem_offsets() {
    constexpr int CTA_M = 128;
    constexpr int CTA_N = 128;
    constexpr int AB_SMEM_STRIDE = 64;
    constexpr int K_STAGE = 2;
    
    unsigned int B_smem_idx_off = CTA_M * AB_SMEM_STRIDE;
    unsigned int smem_stage_off = (CTA_M + CTA_N) * AB_SMEM_STRIDE;
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        printf("=== Shared Memory Layout Debug ===\n");
        printf("CTA_M = %d, CTA_N = %d, AB_SMEM_STRIDE = %d\n", CTA_M, CTA_N, AB_SMEM_STRIDE);
        printf("K_STAGE = %d\n", K_STAGE);
        printf("\n");
        printf("B_smem_idx_off = %u bytes (0x%x)\n", B_smem_idx_off, B_smem_idx_off);
        printf("smem_stage_off = %u bytes (0x%x)\n", smem_stage_off, smem_stage_off);
        printf("\n");
        printf("Stage 0 A: 0x0000 - 0x%04x (%u bytes)\n", CTA_M * AB_SMEM_STRIDE - 1, CTA_M * AB_SMEM_STRIDE);
        printf("Stage 0 B: 0x%04x - 0x%04x (%u bytes)\n", B_smem_idx_off, B_smem_idx_off + CTA_N * AB_SMEM_STRIDE - 1, CTA_N * AB_SMEM_STRIDE);
        printf("Stage 0 Total: 0x0000 - 0x%04x (%u bytes)\n", smem_stage_off - 1, smem_stage_off);
        printf("\n");
        printf("Stage 1 A: 0x%04x - 0x%04x (%u bytes)\n", smem_stage_off, smem_stage_off + CTA_M * AB_SMEM_STRIDE - 1, CTA_M * AB_SMEM_STRIDE);
        printf("Stage 1 B: 0x%04x - 0x%04x (%u bytes)\n", smem_stage_off + B_smem_idx_off, smem_stage_off + B_smem_idx_off + CTA_N * AB_SMEM_STRIDE - 1, CTA_N * AB_SMEM_STRIDE);
        printf("Stage 1 Total: 0x%04x - 0x%04x (%u bytes)\n", smem_stage_off, 2 * smem_stage_off - 1, smem_stage_off);
        printf("\n");
        printf("Total shared memory needed: %u bytes = %.1f KB\n", 2 * smem_stage_off, 2.0f * smem_stage_off / 1024);
        printf("Allocated: 49152 bytes = 48 KB\n");
        printf("Status: %s\n", (2 * smem_stage_off <= 49152) ? "OK" : "OVERFLOW!");
    }
}

void print_smem_debug() {
    debug_smem_offsets<<<1, dim3(32, 8, 1)>>>();
    cudaDeviceSynchronize();
}

int main() {
    print_smem_debug();
    return 0;
}
