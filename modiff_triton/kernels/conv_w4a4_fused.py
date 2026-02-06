"""
Fused W4A4 Conv2d Kernel with MoDiff Error-Compensated Modulation

Implementation of MoDiff Equations (ec5-ec6) for 4-bit precision.
Optimized for NVIDIA L4 (channels_last).

â_t = Q(a_t - â_{t+1}) + â_{t+1}
ô_t = Conv(Q(a_t - â_{t+1}), W) + ô_{t+1}
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_w4a4_3x3_s1_kernel(
    # Input (FP32)
    X_ptr, X_prev_ptr,
    # Weight (INT4 packed)
    W_packed_ptr,
    # Previous output (FP32)
    O_prev_ptr,
    # Outputs
    O_ptr,
    A_hat_new_ptr,  # NEW: Output updated a_hat cache
    # Scales
    scale_w_ptr,
    bias_ptr,
    static_scale,
    # Shapes
    batch, in_channels, height, width,
    out_channels,
    # Strides
    stride_xb, stride_xh, stride_xw, stride_xc,
    stride_xpb, stride_xph, stride_xpw, stride_xpc,
    stride_woc, stride_wk,
    stride_opb, stride_oph, stride_opw, stride_opc,
    stride_ob, stride_oh, stride_ow, stride_oc,
    stride_ahb, stride_ahh, stride_ahw, stride_ahc, # NEW
    # Config
    HAS_PREV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    UPDATE_A_HAT: tl.constexpr, # NEW
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """
    Fused kernel that:
    1. Computes residual (X - X_prev)
    2. Quantizes to INT4
    3. Convolves with INT4 Weights (unpacked on-the-fly)
    4. Adds to O_prev
    """
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    h_out = pid_spatial // width
    w_out = pid_spatial % width
    
    # 3x3 Conv, s1, p1 -> h_in = h_out
    h_in = h_out
    w_in = w_out
    
    offs_in = tl.arange(0, BLOCK_IN)
    offs_out = pid_out_ch * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_out = offs_out < out_channels

    # Calculate Activation Scale
    if static_scale > 0.0:
        scale_a = static_scale
    else:
        # Dynamic: Find max(|X - X_prev|)
        max_val = 0.0
        for c_start in range(0, in_channels, BLOCK_IN):
            for kh in range(3):
                for kw in range(3):
                    hi, wi = h_in + kh - 1, w_in + kw - 1
                    if hi >= 0 and hi < height and wi >= 0 and wi < width:
                        c_mask = (c_start + offs_in) < in_channels
                        x = tl.load(X_ptr + pid_batch * stride_xb + hi * stride_xh + wi * stride_xw + (c_start + offs_in) * stride_xc, mask=c_mask, other=0.0)
                        if HAS_PREV:
                            xp = tl.load(X_prev_ptr + pid_batch * stride_xpb + hi * stride_xph + wi * stride_xpw + (c_start + offs_in) * stride_xpc, mask=c_mask, other=0.0)
                            x = x - xp
                        max_val = tl.maximum(max_val, tl.max(tl.abs(x)))
        scale_a = tl.maximum(max_val / 7.0, 1e-8) # INT4 signed min-max is [-8, 7]

    # Accumulator (using int32 for precision during sum)
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.int32)
    
    # Convolution Loop
    for c_start in range(0, in_channels, BLOCK_IN):
        c_mask = (c_start + offs_in) < in_channels
        
        for kh in range(3):
            for kw in range(3):
                hi, wi = h_in + kh - 1, w_in + kw - 1
                if hi >= 0 and hi < height and wi >= 0 and wi < width:
                    # 1. Load and Quantize Activation to INT4
                    x = tl.load(X_ptr + pid_batch * stride_xb + hi * stride_xh + wi * stride_xw + (c_start + offs_in) * stride_xc, mask=c_mask, other=0.0)
                    if HAS_PREV:
                        xp = tl.load(X_prev_ptr + pid_batch * stride_xpb + hi * stride_xph + wi * stride_xpw + (c_start + offs_in) * stride_xpc, mask=c_mask, other=0.0)
                        x = x - xp
                    
                    x_q = tl.floor(x / scale_a + 0.5)
                    x_q = tl.maximum(tl.minimum(x_q, 7.0), -8.0).to(tl.int8)
                    
                    # 2. Load Packed Weights and Unpack
                    # Weight map: [OC, IC, 3, 3] -> Flattened to [OC, IC*9]
                    # We need IC block for specific kh, kw.
                    # Index in flattened K dimension: (c_start + offs_in) * 9 + kh * 3 + kw
                    k_idx = (c_start + offs_in) * 9 + kh * 3 + kw
                    
                    # Packed index: k_idx // 2
                    w_packed_ptrs = W_packed_ptr + offs_out[:, None] * stride_woc + (k_idx[None, :] // 2) * stride_wk
                    w_packed = tl.load(w_packed_ptrs, mask=mask_out[:, None] & c_mask[None, :], other=0)
                    
                    # Unpack: if k_idx is even, take lo bits; if odd, take hi bits
                    is_hi = (k_idx % 2) == 1
                    w_unpacked = tl.where(is_hi[None, :], (w_packed >> 4) & 0xF, w_packed & 0xF).to(tl.int8) - 8
                    
                    # 3. DOT
                    acc += tl.sum(w_unpacked * x_q[None, :], axis=1)

    # Dequantize
    scale_w = tl.load(scale_w_ptr + offs_out, mask=mask_out, other=1.0)
    out_fp32 = acc.to(tl.float32) * (scale_a * scale_w)
    
    if HAS_BIAS:
        out_fp32 += tl.load(bias_ptr + offs_out, mask=mask_out, other=0.0)
    
    if HAS_PREV:
        # MoDiff update: ô_t = conv_residual + ô_{t+1}
        o_prev = tl.load(O_prev_ptr + pid_batch * stride_opb + h_out * stride_oph + w_out * stride_opw + offs_out * stride_opc, mask=mask_out, other=0.0)
        out_fp32 += o_prev

    # Store Output
    tl.store(O_ptr + pid_batch * stride_ob + h_out * stride_oh + w_out * stride_ow + offs_out * stride_oc, out_fp32, mask=mask_out)

    # NEW: Update a_hat inside the kernel to save a whole pass over memory
    if UPDATE_A_HAT:
        # We need to compute a_hat_new = Q(a_t - a_hat_prev) + a_hat_prev
        # Since we are parallelized spatially (h_out, w_out), we can update all channels here
        # But wait, the kernel is parallelized along out_channels (BLOCK_OUT).
        # We only need to update a_hat ONCE per (batch, spatial).
        # So only one output block (e.g. pid_out_ch == 0) performs the a_hat update.
        if pid_out_ch == 0:
            for c_start in range(0, in_channels, BLOCK_IN):
                c_mask = (c_start + offs_in) < in_channels
                
                # Load input
                ax = tl.load(X_ptr + pid_batch * stride_xb + h_in * stride_xh + w_in * stride_xw + (c_start + offs_in) * stride_xc, mask=c_mask, other=0.0)
                if HAS_PREV:
                    axp = tl.load(X_prev_ptr + pid_batch * stride_xpb + h_in * stride_xph + w_in * stride_xpw + (c_start + offs_in) * stride_xpc, mask=c_mask, other=0.0)
                    adiff = ax - axp
                else:
                    adiff = ax
                    axp = 0.0
                
                # Quantize-Dequantize Error Compensation
                aq = tl.floor(adiff / scale_a + 0.5)
                aq = tl.maximum(tl.minimum(aq, 7.0), -8.0)
                ah_new = (aq * scale_a) + axp
                
                # Store back to cache
                tl.store(A_hat_new_ptr + pid_batch * stride_ahb + h_out * stride_ahh + w_out * stride_ahw + (c_start + offs_in) * stride_ahc, ah_new, mask=c_mask)


def conv2d_w4a4_modiff(x, x_prev, w_packed, scale_w, bias, o_prev, a_hat_new=None, output=None, static_scale=0.0):
    """Python wrapper for the fused W4A4 MoDiff kernel."""
    batch, height, width, in_channels = x.shape
    out_channels = scale_w.shape[0]
    
    if output is None:
        output = torch.empty((batch, height, width, out_channels), device=x.device, dtype=x.dtype)
    
    # Launch config: Larger blocks for high-batch occupancy
    BLOCK_OUT = 64
    BLOCK_IN = 64
    grid = (batch, triton.cdiv(out_channels, BLOCK_OUT), height * width)
    
    _conv2d_w4a4_3x3_s1_kernel[grid](
        x, x_prev, w_packed, o_prev, output, a_hat_new,
        scale_w, bias, static_scale,
        batch, in_channels, height, width, out_channels,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        x_prev.stride(0), x_prev.stride(1), x_prev.stride(2), x_prev.stride(3),
        w_packed.stride(0), w_packed.stride(1),
        o_prev.stride(0), o_prev.stride(1), o_prev.stride(2), o_prev.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        a_hat_new.stride(0) if a_hat_new is not None else 0,
        a_hat_new.stride(1) if a_hat_new is not None else 0,
        a_hat_new.stride(2) if a_hat_new is not None else 0,
        a_hat_new.stride(3) if a_hat_new is not None else 0,
        HAS_PREV=True,
        HAS_BIAS=(bias is not None),
        UPDATE_A_HAT=(a_hat_new is not None),
        BLOCK_OUT=BLOCK_OUT, BLOCK_IN=BLOCK_IN,
    )
    return output
