
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_modiff_kernel(
    # Pointers
    X_ptr, X_prev_ptr,      # [M, K]
    W_ptr,                  # [K, N]
    O_prev_ptr,             # [M, N]
    Out_ptr,                # [M, N]
    scale_w_ptr,            # [N] or scalar
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # W is transposed [N, K] in memory for better access? No, usually [K, N] or [N, K]. Let's assume [N, K] for W (Col-Major) or [K, N] (Row-Major). 
                            # Standard GEMM A[M,K] x B[K,N]. B is usually [N, K] for efficient load.
    stride_om, stride_on,
    # Constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # PID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    # X: [M, K]
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    xp_ptrs = X_prev_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    
    # W: [K, N] -> We assume W is stored as [N, K] (transposed) for coalesced load if stride_wn=1
    # w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    # Actually, let's assume standard layout B[K, N].
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # -----------------------------------------------------------
    # Pass 1: Compute Dynamic Scale (Max Abs Residual)
    # -----------------------------------------------------------
    # We need max(|x - xp|) over all K.
    # Since we tile K, we must iterate K.
    
    max_val = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_k = k + offs_k < K
        mask_m = offs_m < M
        
        # Load X and X_prev
        x = tl.load(x_ptrs + k * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        xp = tl.load(xp_ptrs + k * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        residual = x - xp
        block_max = tl.max(tl.abs(residual), 1)
        max_val = tl.maximum(max_val, block_max)
    
    # Compute scale
    scale_a = max_val / 127.0
    scale_a = tl.where(scale_a < 1e-8, 1e-8, scale_a)
    
    # -----------------------------------------------------------
    # Pass 2: Quantize, Update Cache, GEMM
    # -----------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_k = k + offs_k < K
        mask_m = offs_m < M
        
        # Re-load X and X_prev (L2 cache should help)
        x = tl.load(x_ptrs + k * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        xp = tl.load(xp_ptrs + k * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Compute Residual
        residual = x - xp
        
        # Quantize
        res_scaled = residual / scale_a[:, None]
        res_int = tl.floor(res_scaled + 0.5) # Round
        res_int = tl.maximum(tl.minimum(res_int, 127.0), -128.0)
        
        # Dequantize for Cache Update
        res_dequant = res_int * scale_a[:, None]
        x_new = res_dequant + xp
        
        # Update Cache (Only one block column does this to avoid race)
        # We use pid_n == 0. This assumes N is covered by at least one block.
        if pid_n == 0:
            tl.store(xp_ptrs + k * stride_xk, x_new, mask=mask_m[:, None] & mask_k[None, :])
            
        # Load Weight
        w = tl.load(w_ptrs + k * stride_wk, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        
        # GEMM
        acc += tl.dot(res_int.to(tl.float16), w.to(tl.float16))
        
    # -----------------------------------------------------------
    # Epilogue
    # -----------------------------------------------------------
    # Load scales
    scale_w = tl.load(scale_w_ptr) # Assume scalar for now
    
    # Finalize
    out = acc * scale_a[:, None] * scale_w
    
    # Add previous output
    offs_om = offs_m
    offs_on = offs_n
    op_ptrs = O_prev_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    out_ptrs = Out_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    
    mask_out = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    
    op = tl.load(op_ptrs, mask=mask_out, other=0.0)
    out = out + op
    
    tl.store(out_ptrs, out, mask=mask_out)

def fused_modiff_gemm(x, x_prev, weight, o_prev, scale_w):
    M, K = x.shape
    _, N = weight.shape
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _fused_modiff_kernel[grid](
        x, x_prev, weight, o_prev, out, scale_w,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0), # stride_wn, stride_wk
        out.stride(0), out.stride(1),
        GROUP_M=8
    )
    return out
