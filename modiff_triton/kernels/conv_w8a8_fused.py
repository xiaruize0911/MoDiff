"""
Fused W8A8 Conv2d Kernel with MoDiff Error-Compensated Modulation

This implements direct INT8 convolution without im2col, specifically optimized
for the MoDiff framework (Equations ec5-ec6).

Key optimization: Eliminate im2col intermediate buffer by computing convolution
directly in shared memory using a sliding window approach.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_w8a8_3x3_s1_kernel(
    # Input (FP32)
    X_ptr, X_prev_ptr,
    # Weight (INT8, pre-quantized)
    W_ptr,
    # Previous output (FP32) - for accumulation
    O_prev_ptr,
    # Outputs
    O_ptr,
    A_hat_new_ptr,  # NEW: Output updated a_hat cache
    # Scales
    scale_w_ptr,
    bias_ptr,
    static_scale,  # NEW: pre-computed activation scale (0.0 = dynamic)
    # Shapes
    batch, in_channels, height, width,
    out_channels,
    # Strides
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_xpb, stride_xpc, stride_xph, stride_xpw,
    stride_wout, stride_win, stride_wkh, stride_wkw,
    stride_opb, stride_opc, stride_oph, stride_opw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    stride_ahnewb, stride_ahnewc, stride_ahnewh, stride_ahneww, # NEW
    # Config
    HAS_PREV: tl.constexpr,  # True for MoDiff modulated path
    HAS_BIAS: tl.constexpr,
    UPDATE_A_HAT: tl.constexpr, # NEW
    SCALE_W_IS_VECTOR: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """
    Direct 3x3 conv2d kernel for W8A8 with optional MoDiff modulation.
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Compute spatial position
    h_out = pid_spatial // width
    w_out = pid_spatial % width
    
    # With padding=1, stride=1, output size = input size
    h_in = h_out  # Center of 3x3 kernel
    w_in = w_out
    
    # Offsets for input channels
    offs_in = tl.arange(0, BLOCK_IN)
    
    # ========================================================================
    # Step 1: Compute activation quantization scale
    # ========================================================================
    # If static_scale provided, use it directly (eliminates find_max overhead!)
    if static_scale > 0.0:
        scale_a = static_scale
    else:
        # Dynamic quantization: find max over all activations
        # We need to find max(|x - x_prev|) over ALL input channels and 3x3 window
        max_val = 0.0
        
        for c_start in range(0, in_channels, BLOCK_IN):
            # Load 3x3 window for this block of channels
            for kh in range(3):
                for kw in range(3):
                    h_idx = h_in + kh - 1
                    w_idx = w_in + kw - 1
                    
                    valid = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)
                    mask = (c_start + offs_in < in_channels) & valid
                    
                    # Load current input
                    x_ptrs = (
                        X_ptr + 
                        pid_batch * stride_xb + 
                        (c_start + offs_in) * stride_xc + 
                        h_idx * stride_xh + 
                        w_idx * stride_xw
                    )
                    x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                    
                    if HAS_PREV:
                        xp_ptrs = (
                            X_prev_ptr + 
                            pid_batch * stride_xpb + 
                            (c_start + offs_in) * stride_xpc + 
                            h_idx * stride_xph + 
                            w_idx * stride_xpw
                        )
                        xp_val = tl.load(xp_ptrs, mask=mask, other=0.0)
                        x_val = x_val - xp_val
                    
                    # Update max
                    max_val = tl.maximum(max_val, tl.max(tl.abs(x_val)))
        
        # Compute scale from dynamic max
        scale_a = max_val / 127.0
        scale_a = tl.where(scale_a < 1e-8, 1e-8, scale_a)
    
    # ========================================================================
    # Step 2: Compute Convolution
    # ========================================================================
    
    # Accumulator
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.int32)
    
    # Offsets for output channels
    offs_out = pid_out_ch * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_out = offs_out < out_channels
    
    for c_start in range(0, in_channels, BLOCK_IN):
        c_mask = (c_start + offs_in) < in_channels
        
        # Load 3x3 window and quantize
        # We store quantized values in registers: [BLOCK_IN, 3, 3]
        # Flattened to [BLOCK_IN, 9] for easier access? Or just loop.
        # Since we need to multiply with weights [BLOCK_OUT, BLOCK_IN, 3, 3],
        # we can iterate 3x3 inside.
        
        for kh in range(3):
            for kw in range(3):
                h_idx = h_in + kh - 1
                w_idx = w_in + kw - 1
                
                valid = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)
                mask = c_mask & valid
                
                # Load input (re-load)
                x_ptrs = (
                    X_ptr + 
                    pid_batch * stride_xb + 
                    (c_start + offs_in) * stride_xc + 
                    h_idx * stride_xh + 
                    w_idx * stride_xw
                )
                x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                
                if HAS_PREV:
                    xp_ptrs = (
                        X_prev_ptr + 
                        pid_batch * stride_xpb + 
                        (c_start + offs_in) * stride_xpc + 
                        h_idx * stride_xph + 
                        w_idx * stride_xpw
                    )
                    xp_val = tl.load(xp_ptrs, mask=mask, other=0.0)
                    x_val = x_val - xp_val
                
                # Quantize
                x_q = tl.floor(x_val / scale_a + 0.5)
                x_q = tl.maximum(tl.minimum(x_q, 127.0), -128.0).to(tl.int8)
                
                # Load weights: [BLOCK_OUT, BLOCK_IN] for this kh, kw
                # Weight shape: [out, in, 3, 3]
                # Stride: out*s_wout + in*s_win + kh*s_wkh + kw*s_wkw
                
                w_base = W_ptr + kh * stride_wkh + kw * stride_wkw
                
                # We need to load [BLOCK_OUT, BLOCK_IN] matrix
                # But Triton load requires pointers.
                # w_ptrs: [BLOCK_OUT, BLOCK_IN]
                w_ptrs = (
                    w_base + 
                    offs_out[:, None] * stride_wout + 
                    (c_start + offs_in)[None, :] * stride_win
                )
                
                w_val = tl.load(w_ptrs, mask=mask_out[:, None] & c_mask[None, :], other=0).to(tl.int8)
                
                # Multiply accumulate: [BLOCK_OUT, BLOCK_IN] * [BLOCK_IN] (broadcast) -> [BLOCK_OUT]
                # x_q is [BLOCK_IN]
                # w_val is [BLOCK_OUT, BLOCK_IN]
                
                prod = w_val * x_q[None, :]
                acc += tl.sum(prod, axis=1)

    # ========================================================================
    # Step 3: Dequantize and Store
    # ========================================================================
    
    # Load weight scale
    if SCALE_W_IS_VECTOR:
        scale_w = tl.load(scale_w_ptr + offs_out, mask=mask_out, other=1.0)
    else:
        scale_w = tl.load(scale_w_ptr)
        
    out_fp = acc.to(tl.float32) * scale_a * scale_w
    
    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_out, mask=mask_out, other=0.0)
        out_fp += bias
        
    # Add previous output
    if HAS_PREV:
        oprev_ptrs = (
            O_prev_ptr + 
            pid_batch * stride_opb + 
            offs_out * stride_opc + 
            h_out * stride_oph + 
            w_out * stride_opw
        )
        o_prev = tl.load(oprev_ptrs, mask=mask_out, other=0.0)
        out_fp += o_prev
        
    # Store
    out_ptrs = (
        O_ptr + 
        pid_batch * stride_ob + 
        offs_out * stride_oc + 
        h_out * stride_oh + 
        w_out * stride_ow
    )
    tl.store(out_ptrs, out_fp, mask=mask_out)

    # ========================================================================
    # Step 4: Update a_hat Cache (NEW)
    # ========================================================================
    if UPDATE_A_HAT:
        # Only one output channel block needs to update the activation cache
        if pid_out_ch == 0:
            for c_start in range(0, in_channels, BLOCK_IN):
                c_mask = (c_start + offs_in) < in_channels
                
                # Load current input at (h_in, w_in)
                x_ptrs = (
                    X_ptr + 
                    pid_batch * stride_xb + 
                    (c_start + offs_in) * stride_xc + 
                    h_in * stride_xh + 
                    w_in * stride_xw
                )
                x_val = tl.load(x_ptrs, mask=c_mask, other=0.0)
                
                xp_val = 0.0
                if HAS_PREV:
                    xp_ptrs = (
                        X_prev_ptr + 
                        pid_batch * stride_xpb + 
                        (c_start + offs_in) * stride_xpc + 
                        h_in * stride_xph + 
                        w_in * stride_xpw
                    )
                    xp_val = tl.load(xp_ptrs, mask=c_mask, other=0.0)
                
                # Compute Q(a_t - a_hat_prev) + a_hat_prev
                diff = x_val - xp_val
                q_diff = tl.floor(diff / scale_a + 0.5)
                q_diff = tl.maximum(tl.minimum(q_diff, 127.0), -128.0)
                a_hat_new = q_diff * scale_a + xp_val
                
                # Store new cache
                ahnew_ptrs = (
                    A_hat_new_ptr + 
                    pid_batch * stride_ahnewb + 
                    (c_start + offs_in) * stride_ahnewc + 
                    h_in * stride_ahnewh + 
                    w_in * stride_ahneww
                )
                tl.store(ahnew_ptrs, a_hat_new, mask=c_mask)


def conv2d_w8a8_3x3_fused(
    x: torch.Tensor,
    x_prev: torch.Tensor,
    weight_int8: torch.Tensor,
    o_prev: torch.Tensor,
    scale_w: torch.Tensor,
    a_hat_new: torch.Tensor = None,
    bias: torch.Tensor = None,
    static_scale: float = 0.0,
    output: torch.Tensor = None, # NEW
) -> torch.Tensor:
    """
    Fused W8A8 3×3 Conv2d with MoDiff modulation.
    
    Args:
        static_scale: Pre-computed activation scale. If > 0, skips find_max.
                     If == 0.0, uses dynamic quantization (default).
        output: Optional output tensor to avoid allocation.
    """
    # Shapes
    N, C, H, W = x.shape
    OutC = weight_int8.shape[0]
    
    # Output
    if output is None:
        out = torch.empty((N, OutC, H, W), device=x.device, dtype=torch.float32)
    else:
        out = output
    
    # Config
    BLOCK_OUT = 32
    BLOCK_IN = 32
    
    grid = (N, triton.cdiv(OutC, BLOCK_OUT), H * W)
    
    _conv2d_w8a8_3x3_s1_kernel[grid](
        x, x_prev,
        weight_int8,
        o_prev,
        out,
        a_hat_new if a_hat_new is not None else x,
        scale_w,
        bias if bias is not None else x,
        static_scale,
        N, C, H, W, OutC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        x_prev.stride(0), x_prev.stride(1), x_prev.stride(2), x_prev.stride(3),
        weight_int8.stride(0), weight_int8.stride(1), weight_int8.stride(2), weight_int8.stride(3),
        o_prev.stride(0), o_prev.stride(1), o_prev.stride(2), o_prev.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        a_hat_new.stride(0), a_hat_new.stride(1), a_hat_new.stride(2), a_hat_new.stride(3),
        HAS_PREV=True,
        HAS_BIAS=(bias is not None),
        UPDATE_A_HAT=(a_hat_new is not None),
        SCALE_W_IS_VECTOR=(scale_w.numel() > 1),
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
    )
    
    return out


def conv2d_w8a8_3x3_standard(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale_w: torch.Tensor,
    bias: torch.Tensor = None,
    static_scale: float = 0.0,
    output: torch.Tensor = None, # NEW
) -> torch.Tensor:
    """
    Standard W8A8 3×3 Conv2d without MoDiff modulation.
    
    Args:
        static_scale: Pre-computed activation scale. If > 0, skips find_max.
        output: Optional output tensor to avoid allocation.
    """
    # Shapes
    N, C, H, W = x.shape
    OutC = weight_int8.shape[0]
    
    # Output
    if output is None:
        out = torch.empty((N, OutC, H, W), device=x.device, dtype=torch.float32)
    else:
        out = output
    
    # Config
    BLOCK_OUT = 32
    BLOCK_IN = 32
    
    grid = (N, triton.cdiv(OutC, BLOCK_OUT), H * W)
    
    # Use same kernel with HAS_PREV=False
    _conv2d_w8a8_3x3_s1_kernel[grid](
        x, x,
        weight_int8,
        out,
        out,
        x, # a_hat_new unused
        scale_w,
        bias if bias is not None else x,
        static_scale,
        N, C, H, W, OutC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight_int8.stride(0), weight_int8.stride(1), weight_int8.stride(2), weight_int8.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        HAS_PREV=False,
        HAS_BIAS=(bias is not None),
        UPDATE_A_HAT=False,
        SCALE_W_IS_VECTOR=(scale_w.numel() > 1),
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
    )
    
    return out
