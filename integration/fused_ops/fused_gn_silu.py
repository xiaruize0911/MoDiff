"""
Fused GroupNorm + SiLU Triton kernel.

*** NOT USED IN PRODUCTION - measured slower than the plain-PyTorch alternative ***
Benchmarked at ~3.4ms/call vs ~1.9ms/call for `FusedGroupNormSiLU` in
integration/fused_ops/fused_resblock.py (which disables autocast locally and calls
native F.group_norm + F.silu instead) at this model's channel/resolution sizes - see
fused_resblock.py's module docstring for the comparison. `fuse_resblocks_in_module`
wires up fused_resblock.py's version, never this one. This module is kept only for the
one-off comparison in integration/benchmarks/benchmark_bottleneck.py; do not wire it
into any production conversion path without re-validating the timing claim above.

Fuses GroupNorm normalization with SiLU activation into a single kernel pass,
eliminating one full read+write of the feature tensor compared to separate ops.

Supports both contiguous (NCHW) and channels_last (NHWC) 4D tensors.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _group_norm_silu_fwd_kernel(
    X_ptr,          # input tensor [N, C, H, W] in either NCHW or NHWC layout
    Y_ptr,          # output tensor, same layout as input
    W_ptr,          # GN weight (affine gamma) [C]
    B_ptr,          # GN bias (affine beta)   [C]
    N: tl.constexpr,          # batch size
    C: tl.constexpr,          # channels
    HW: tl.constexpr,         # H * W (spatial size)
    G: tl.constexpr,          # num_groups
    eps: tl.constexpr,        # epsilon
    channels_last: tl.constexpr,  # whether input is NHWC
    BLOCK_SIZE: tl.constexpr,     # elements per group per sample
):
    """Each program handles one (sample, group) pair."""
    pid = tl.program_id(0)
    sample_id = pid // G
    group_id = pid % G

    CPG = C // G  # channels per group
    group_size = CPG * HW  # total elements in this group

    # Compute mean and variance for this group
    mean = tl.zeros([1], dtype=tl.float32)
    var = tl.zeros([1], dtype=tl.float32)

    # Two-pass: first compute mean, then variance
    for block_start in tl.range(0, group_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size

        # Convert linear offset within group to (channel_in_group, spatial) indices
        c_local = offsets // HW  # channel within group [0, CPG)
        hw_idx = offsets % HW     # spatial index [0, HW)
        c_global = group_id * CPG + c_local  # absolute channel index

        if channels_last:
            # NHWC: [N, H*W, C] (flattened spatial)
            idx = sample_id * (HW * C) + hw_idx * C + c_global
        else:
            # NCHW: [N, C, H*W]
            idx = sample_id * (C * HW) + c_global * HW + hw_idx

        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)

    mean = mean / group_size

    for block_start in tl.range(0, group_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size

        c_local = offsets // HW
        hw_idx = offsets % HW
        c_global = group_id * CPG + c_local

        if channels_last:
            idx = sample_id * (HW * C) + hw_idx * C + c_global
        else:
            idx = sample_id * (C * HW) + c_global * HW + hw_idx

        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        # Zero out masked elements to avoid inflating variance
        diff = tl.where(mask, diff, 0.0)
        var += tl.sum(diff * diff, axis=0)

    var = var / group_size
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize, apply affine, apply SiLU — fused in one write pass
    for block_start in tl.range(0, group_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size

        c_local = offsets // HW
        hw_idx = offsets % HW
        c_global = group_id * CPG + c_local

        if channels_last:
            idx = sample_id * (HW * C) + hw_idx * C + c_global
        else:
            idx = sample_id * (C * HW) + c_global * HW + hw_idx

        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        # GroupNorm
        w = tl.load(W_ptr + c_global, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        normed = (x - mean) * inv_std * w + b

        # SiLU: x * sigmoid(x)
        sigmoid_val = 1.0 / (1.0 + tl.exp(-normed))
        result = normed * sigmoid_val

        tl.store(Y_ptr + idx, result.to(x.dtype), mask=mask)


class TritonGroupNormSiLU(nn.Module):
    """Drop-in replacement for nn.Sequential(GroupNorm, SiLU) using a fused Triton kernel.
    
    Eliminates one full tensor read+write by fusing normalization and activation.
    """

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        # Share the same Parameter objects (no copy)
        self.weight = gn.weight
        self.bias = gn.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"Expected 4D input, got {x.ndim}D"
        N, C, H, W = x.shape
        assert C == self.num_channels

        is_cl = x.is_contiguous(memory_format=torch.channels_last)
        if not is_cl and not x.is_contiguous():
            x = x.contiguous()

        HW = H * W
        CPG = C // self.num_groups
        group_size = CPG * HW

        # Choose block size: must be power of 2
        BLOCK_SIZE = triton.next_power_of_2(min(group_size, 4096))

        y = torch.empty_like(x)

        grid = (N * self.num_groups,)
        _group_norm_silu_fwd_kernel[grid](
            x, y,
            self.weight, self.bias,
            N, C, HW, self.num_groups, self.eps,
            is_cl,
            BLOCK_SIZE,
        )
        return y


def replace_gn_silu_in_model(model: nn.Module) -> int:
    """Replace all GroupNorm+SiLU pairs in nn.Sequential with fused TritonGroupNormSiLU.
    
    Returns the number of replacements made. Works on the model in-place.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Look for GroupNorm followed by SiLU
            children = list(module.children())
            indices_to_fuse = []
            for i in range(len(children) - 1):
                if (isinstance(children[i], nn.GroupNorm) and
                    isinstance(children[i + 1], (nn.SiLU, nn.modules.activation.SiLU))):
                    indices_to_fuse.append(i)

            if not indices_to_fuse:
                continue

            # Rebuild Sequential with fused modules
            new_children = []
            skip_next = False
            for i, child in enumerate(children):
                if skip_next:
                    skip_next = False
                    continue
                if i in indices_to_fuse:
                    new_children.append(TritonGroupNormSiLU(child))
                    skip_next = True
                    count += 1
                else:
                    new_children.append(child)

            # Replace the Sequential's children in-place
            module._modules.clear()
            for i, child in enumerate(new_children):
                module._modules[str(i)] = child

    return count
