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

try:
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
except ImportError:
    OptimizedInt4Conv2d = None


def _chainable_conv_int4(m):
    return (OptimizedInt4Conv2d is not None
            and isinstance(m, OptimizedInt4Conv2d)
            and getattr(m, "is_calibrated", False)
            and getattr(m, "use_cutlass", False)
            and not getattr(m, "modiff_enabled", True))


def bottleneck_chainable_int4(block):
    """int4 twin of bottleneck_chainable: conv1/2/3 are calibrated CUTLASS int4 convs
    (conv3 needs standard_output_fp16 for the fp16 residual epilogue)."""
    return (hasattr(block, "conv1") and hasattr(block, "conv2") and hasattr(block, "conv3")
            and _chainable_conv_int4(block.conv1) and _chainable_conv_int4(block.conv2)
            and _chainable_conv_int4(block.conv3)
            and getattr(block.conv3, "standard_output_fp16", False))


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


class Int8FullyChainedResNet(nn.Module):
    """Whole-ResNet int8 threading (prototype for the block-entry-quantize fusion).

    Int8ChainedBottleneck keeps activations int8 WITHIN a bottleneck but pays a
    standalone quantize (`conv1.quantize_input`) at EVERY block entry -- nsys shows
    that quantize is the single largest kernel on ResNet-50 int8 (~58 ms, > the conv
    GEMM savings), because a BN-folded CNN has no norm to hide it in.

    This wrapper folds each block-entry quantize into the PREVIOUS block's conv3
    store: conv3's dual epilogue emits both the fp16 block output (the next block's
    identity/residual) AND that output requantized to int8 (the next block's conv1
    input), so the whole network quantizes to int8 exactly ONCE (after maxpool) and
    stays int8 through every conv until the final avgpool. Removes N-1 of the N
    per-block quantize kernels."""

    def __init__(self, model):
        super().__init__()
        self.stem_conv, self.stem_bn = model.conv1, model.bn1   # bn1 is Identity (BN folded)
        self.stem_relu, self.maxpool = model.relu, model.maxpool
        self.avgpool, self.fc = model.avgpool, model.fc
        blocks = []
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, lname, None)
            if layer is not None:
                blocks.extend(list(layer))
        assert all(bottleneck_chainable(b) for b in blocks), \
            "every bottleneck must be int8-chainable (calibrated CUTLASS int8, conv3 fp16-out)"
        self.blocks = nn.ModuleList(blocks)
        self.relus = [b.relu if hasattr(b, "relu") else None for b in blocks]
        self.downsamples = [b.downsample for b in blocks]
        # Intra-block chaining (conv1->conv2->conv3) + cross-block conv3->next-conv1.
        self._next_scale = []
        for i, b in enumerate(blocks):
            b.conv1.output_requant_scale = b.conv2.static_input_scale
            b.conv1.fuse_output_relu = True
            b.conv2.output_requant_scale = b.conv3.static_input_scale
            b.conv2.fuse_output_relu = True
            self._next_scale.append(blocks[i + 1].conv1.static_input_scale
                                    if i + 1 < len(blocks) else None)

    def forward(self, x):
        x = self.maxpool(self.stem_relu(self.stem_bn(self.stem_conv(x))))
        x_fp16 = x.contiguous(memory_format=torch.channels_last)
        x_int8 = self.blocks[0].conv1.quantize_input(x_fp16)   # the ONE entry quantize
        for i, b in enumerate(self.blocks):
            ds = self.downsamples[i]
            identity = (x_fp16 if ds is None else ds(x_fp16))
            identity = identity.contiguous(memory_format=torch.channels_last).half()
            h = b.conv1.forward_to_int8(x_int8, apply_relu=True)
            h = b.conv2.forward_to_int8(h, apply_relu=True)
            ns = self._next_scale[i]
            if ns is not None:
                # dual store: fp16 x_{i+1} (post-ReLU) + its int8 for the next conv1
                x_fp16, x_int8 = b.conv3.forward_from_int8_dual(h, identity, ns, apply_relu=True)
            else:
                out = b.conv3.forward_from_int8(h, residual=identity)   # last block: fp16 only
                x_fp16 = (self.relus[i](out) if self.relus[i] is not None else F.relu(out))
        x = self.avgpool(x_fp16)
        return self.fc(torch.flatten(x, 1))


def build_fully_chained(model, verbose=False):
    """Wrap a calibrated (set_standard_output_fp16 + enable_modiff(False)) ResNet whose
    bottlenecks are int8-chainable into an Int8FullyChainedResNet. Must run instead of
    chain_int8_bottlenecks (not after). Returns the wrapper module."""
    w = Int8FullyChainedResNet(model)
    if verbose:
        print(f"build_fully_chained: threaded {len(w.blocks)} bottlenecks, 1 entry quantize")
    return w


class Int4FullyChainedResNet(nn.Module):
    """int4 twin of Int8FullyChainedResNet: whole-net int4 threading. The per-block
    entry quantize+pack is folded into the previous block's conv3 dual store (fp16 +
    packed int4), so the net quantizes to int4 exactly once (after maxpool) and stays
    int4-packed through every conv until avgpool. int4 convs are unpacked-channel aware,
    so spatial dims (h,w) are threaded explicitly (conv2 may stride)."""

    def __init__(self, model):
        super().__init__()
        self.stem_conv, self.stem_bn = model.conv1, model.bn1
        self.stem_relu, self.maxpool = model.relu, model.maxpool
        self.avgpool, self.fc = model.avgpool, model.fc
        blocks = []
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, lname, None)
            if layer is not None:
                blocks.extend(list(layer))
        assert all(bottleneck_chainable_int4(b) for b in blocks), \
            "every bottleneck must be int4-chainable (calibrated CUTLASS int4, conv3 fp16-out)"
        self.blocks = nn.ModuleList(blocks)
        self.relus = [b.relu if hasattr(b, "relu") else None for b in blocks]
        self.downsamples = [b.downsample for b in blocks]
        self._next_scale = [blocks[i + 1].conv1.static_input_scale if i + 1 < len(blocks) else None
                            for i in range(len(blocks))]

    def forward(self, x):
        x = self.maxpool(self.stem_relu(self.stem_bn(self.stem_conv(x))))
        x_fp16 = x.contiguous(memory_format=torch.channels_last)
        h, w = x_fp16.shape[2], x_fp16.shape[3]
        x_packed = self.blocks[0].conv1.quantize_input(x_fp16)   # the ONE entry quantize+pack
        for i, b in enumerate(self.blocks):
            ds = self.downsamples[i]
            identity = (x_fp16 if ds is None else ds(x_fp16))
            identity = identity.contiguous(memory_format=torch.channels_last).half()
            p = b.conv1.forward_to_int4(x_packed, h, w, b.conv2.static_input_scale, apply_relu=True)
            h1, w1 = b.conv1._out_hw(h, w)
            p = b.conv2.forward_to_int4(p, h1, w1, b.conv3.static_input_scale, apply_relu=True)
            h2, w2 = b.conv2._out_hw(h1, w1)
            ns = self._next_scale[i]
            if ns is not None:
                x_fp16, x_packed = b.conv3.forward_from_int4_dual(p, h2, w2, identity, ns, apply_relu=True)
            else:
                out = b.conv3.forward_from_int4(p, h2, w2, residual=identity)
                x_fp16 = (self.relus[i](out) if self.relus[i] is not None else F.relu(out))
            h, w = b.conv3._out_hw(h2, w2)
        x = self.avgpool(x_fp16)
        return self.fc(torch.flatten(x, 1))


def build_fully_chained_int4(model, verbose=False):
    """Wrap a calibrated (standard_output_fp16 + modiff off) int4 ResNet into an
    Int4FullyChainedResNet. Returns the wrapper module."""
    w = Int4FullyChainedResNet(model)
    if verbose:
        print(f"build_fully_chained_int4: threaded {len(w.blocks)} bottlenecks, 1 entry quantize")
    return w


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
