
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
    # Output
    O_ptr,
    # Scales
    scale_w_ptr,
    bias_ptr,
    # Shapes
    batch, in_channels, height, width,
    out_channels,
    # Strides
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_xpb, stride_xpc, stride_xph, stride_xpw,
    stride_wout, stride_win, stride_wkh, stride_wkw,
    stride_opb, stride_opc, stride_oph, stride_opw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    # Config
    HAS_PREV: tl.constexpr,  # True for MoDiff modulated path
    HAS_BIAS: tl.constexpr,
    SCALE_W_IS_VECTOR: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
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
    # Step 1: Compute quantization scale (per-pixel)
    # ========================================================================
    max_val = 0.0
    
    for c_start in range(0, in_channels, BLOCK_IN):
        for kh in range(3):
            for kw in range(3):
                h_idx = h_in + kh - 1
                w_idx = w_in + kw - 1
                
                valid = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)
                mask = (c_start + offs_in < in_channels) & valid
                
                x_ptrs = X_ptr + pid_batch * stride_xb + (c_start + offs_in) * stride_xc + h_idx * stride_xh + w_idx * stride_xw
                x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                
                if HAS_PREV:
                    xp_ptrs = X_prev_ptr + pid_batch * stride_xpb + (c_start + offs_in) * stride_xpc + h_idx * stride_xph + w_idx * stride_xpw
                    xp_val = tl.load(xp_ptrs, mask=mask, other=0.0)
                    x_val = x_val - xp_val
                
                max_val = tl.maximum(max_val, tl.max(tl.abs(x_val)))
    
    scale_a = max_val / 127.0
    if scale_a < 1e-8: scale_a = 1e-8
    
    # ========================================================================
    # Step 2: Convolution
    # ========================================================================
    offs_out = pid_out_ch * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_out = offs_out < out_channels
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    
    for c_start in range(0, in_channels, BLOCK_IN):
        for kh in range(3):
            for kw in range(3):
                h_idx = h_in + kh - 1
                w_idx = w_in + kw - 1
                
                valid = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)
                mask = (c_start + offs_in < in_channels) & valid
                
                # Load Input
                x_ptrs = X_ptr + pid_batch * stride_xb + (c_start + offs_in) * stride_xc + h_idx * stride_xh + w_idx * stride_xw
                x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                
                if HAS_PREV:
                    xp_ptrs = X_prev_ptr + pid_batch * stride_xpb + (c_start + offs_in) * stride_xpc + h_idx * stride_xph + w_idx * stride_xpw
                    xp_val = tl.load(xp_ptrs, mask=mask, other=0.0)
                    x_val = x_val - xp_val
                
                # Quantize
                x_scaled = x_val / scale_a
                x_int = tl.floor(x_scaled + 0.5)
                x_int = tl.maximum(tl.minimum(x_int, 127.0), -128.0)
                
                # Load Weight
                # W: [Out, In, 3, 3]
                w_ptrs = W_ptr + offs_out[:, None] * stride_wout + (c_start + offs_in)[None, :] * stride_win + kh * stride_wkh + kw * stride_wkw
                w_val = tl.load(w_ptrs, mask=mask_out[:, None] & mask[None, :], other=0.0)
                
                # MAC
                acc += tl.sum(x_int[None, :] * w_val.to(tl.float32), 1)
                
    # ========================================================================
    # Epilogue
    # ========================================================================
    if SCALE_W_IS_VECTOR:
        scale_w = tl.load(scale_w_ptr + offs_out, mask=mask_out, other=0.0)
    else:
        scale_w = tl.load(scale_w_ptr)
        
    out_fp = acc * scale_a * scale_w
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_out, mask=mask_out, other=0.0)
        out_fp += bias
        
    if HAS_PREV:
        oprev_ptrs = O_prev_ptr + pid_batch * stride_opb + offs_out * stride_opc + h_out * stride_oph + w_out * stride_opw
        o_prev = tl.load(oprev_ptrs, mask=mask_out, other=0.0)
        out_fp += o_prev
        
    out_ptrs = O_ptr + pid_batch * stride_ob + offs_out * stride_oc + h_out * stride_oh + w_out * stride_ow
    tl.store(out_ptrs, out_fp, mask=mask_out)

def fused_modiff_conv2d(x, x_prev, weight, o_prev, scale_w, bias=None):
    N, C, H, W = x.shape
    OutC = weight.shape[0]
    out = torch.empty((N, OutC, H, W), device=x.device, dtype=torch.float32)
    
    BLOCK_OUT = 32
    BLOCK_IN = 32
    
    grid = (N, triton.cdiv(OutC, BLOCK_OUT), H * W)
    
    _conv2d_w8a8_3x3_s1_kernel[grid](
        x, x_prev, weight, o_prev, out, scale_w, bias,
        N, C, H, W, OutC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        x_prev.stride(0) if x_prev is not None else 0, x_prev.stride(1) if x_prev is not None else 0, x_prev.stride(2) if x_prev is not None else 0, x_prev.stride(3) if x_prev is not None else 0,
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        o_prev.stride(0) if o_prev is not None else 0, o_prev.stride(1) if o_prev is not None else 0, o_prev.stride(2) if o_prev is not None else 0, o_prev.stride(3) if o_prev is not None else 0,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        HAS_PREV=(x_prev is not None),
        HAS_BIAS=(bias is not None),
        SCALE_W_IS_VECTOR=(scale_w.numel() > 1),
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN
    )
    return out
