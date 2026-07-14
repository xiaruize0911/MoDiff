"""INT8-native conv->conv chaining for ResNet bottlenecks.

The MoDiff CUTLASS conv kernels normally dequantize to fp16 after every conv,
relying on the next GroupNorm to re-quantize cheaply. A BatchNorm-folded CNN has
no such layer, so each conv pays an unhidden quantize+dequant round-trip and int8
ends up slower than fp16 end-to-end.

Int8ChainedBottleneck keeps the activation INT8 across a bottleneck's
conv1->conv2->conv3 (ReLU folds into the requantize, since ReLU commutes with
positive scaling), so the per-conv quantize is paid ONCE at block entry instead of
three times. Only the residual skip-add returns to fp16 (handled by the existing
conv2d_int8_fprop_no_ohat_prealloc_bias_residual epilogue). This is the CNN
analogue of the diffusion GN->int8 fusion, and turns the raw-conv int8 win
(measured ~1.7x at 256ch, ~2x int4 at 512ch) into an end-to-end win.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
except ImportError:
    OptimizedInt8Conv2d = None


def _chainable_conv(m):
    return (OptimizedInt8Conv2d is not None
            and isinstance(m, OptimizedInt8Conv2d)
            and getattr(m, "is_calibrated", False)
            and getattr(m, "use_cutlass", False)
            and not getattr(m, "modiff_enabled", True))


def bottleneck_chainable(block):
    """A torchvision Bottleneck is chainable when conv1/conv2/conv3 are all
    calibrated CUTLASS int8 convs (conv3 additionally needs standard_output_fp16
    for the fp16 residual epilogue)."""
    return (hasattr(block, "conv1") and hasattr(block, "conv2") and hasattr(block, "conv3")
            and _chainable_conv(block.conv1) and _chainable_conv(block.conv2)
            and _chainable_conv(block.conv3)
            and getattr(block.conv3, "standard_output_fp16", False))


class Int8ChainedBottleneck(nn.Module):
    """Wraps a (BatchNorm-folded) torchvision Bottleneck whose conv1/2/3 are
    calibrated OptimizedInt8Conv2d, running the 3 convs int8-chained."""

    def __init__(self, block):
        super().__init__()
        assert bottleneck_chainable(block), "block is not int8-chainable"
        self.conv1, self.conv2, self.conv3 = block.conv1, block.conv2, block.conv3
        self.downsample = block.downsample          # fp16, stays unchained
        self.relu = block.relu if hasattr(block, "relu") else None
        # Wire each conv's output-requant scale to the NEXT conv's input scale, and
        # fold the intermediate ReLUs into the requantize.
        self.conv1.output_requant_scale = self.conv2.static_input_scale
        self.conv1.fuse_output_relu = True
        self.conv2.output_requant_scale = self.conv3.static_input_scale
        self.conv2.fuse_output_relu = True

    def _relu(self, x):
        return self.relu(x) if self.relu is not None else F.relu(x)

    def forward(self, x):
        # x: fp16 channels_last (post-ReLU from the previous block).
        identity = x if self.downsample is None else self.downsample(x)
        if not identity.is_contiguous(memory_format=torch.channels_last):
            identity = identity.contiguous(memory_format=torch.channels_last)
        x_int8 = self.conv1.quantize_input(x)               # block-entry K1 (once)
        h = self.conv1.forward_to_int8(x_int8, apply_relu=True)   # int8 @ conv2 scale
        h = self.conv2.forward_to_int8(h, apply_relu=True)        # int8 @ conv3 scale
        # conv3: dequant + bias + fp16 residual skip-add (existing fused epilogue)
        out = self.conv3.forward_from_int8(h, residual=identity.half())
        return self._relu(out)


def chain_int8_bottlenecks(model, verbose=False):
    """Replace every chainable Bottleneck in model.layer1..layer4 with
    Int8ChainedBottleneck (in place). Must run AFTER calibration +
    set_standard_output_fp16(True). Non-chainable blocks are left untouched.
    Returns the count chained."""
    n = 0
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, lname, None)
        if layer is None:
            continue
        for i, block in enumerate(layer):
            if bottleneck_chainable(block):
                layer[i] = Int8ChainedBottleneck(block)
                n += 1
            elif verbose:
                print(f"  skip {lname}[{i}] (not int8-chainable)")
    if verbose:
        print(f"chain_int8_bottlenecks: chained {n} bottlenecks")
    return n
