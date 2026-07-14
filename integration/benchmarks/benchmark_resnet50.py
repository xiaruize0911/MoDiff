#!/usr/bin/env python3
"""ResNet-50 all-modes benchmark (fp16 / int8 / int4) through the MoDiff CUTLASS
conv kernels.

Motivation: the LDM UNet is attention/memory-bound (~34% quantizable-conv), so
INT8/INT4 only buys ~1.1-1.2x end-to-end. A ResNet-class CNN is conv-compute-bound
(BatchNorm folds into conv at inference; no attention; high channels), so the same
kernels should show the "textbook" ~1.5-2x INT8 win. This benchmarks exactly that
on the same kernels, as an apples-to-apples contrast.

  python integration/benchmarks/benchmark_resnet50.py --batch 64 --repeats 10
"""
import os, sys, time, argparse, statistics, copy
import torch
import torch.nn as nn
from torch.nn.utils import fuse_conv_bn_eval

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torchvision.models as tvm
from integration.kernels.int8_optimized import convert_model_to_optimized_int8, OptimizedInt8Conv2d
from integration.kernels.int4_optimized import convert_model_to_optimized_int4, OptimizedInt4Conv2d
from integration.fused_ops.chained_bottleneck import chain_int8_bottlenecks

QCONV = (OptimizedInt8Conv2d, OptimizedInt4Conv2d)


def fold_bn(model):
    """Fold every conv->BatchNorm pair into the conv (ResNet-50 structure), so the
    inference graph has no separate norm op -- what makes a CNN conv-bound."""
    def fuse(parent, cname, bname):
        conv, bn = getattr(parent, cname, None), getattr(parent, bname, None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            setattr(parent, cname, fuse_conv_bn_eval(conv, bn))
            setattr(parent, bname, nn.Identity())
    fuse(model, "conv1", "bn1")
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        for blk in getattr(model, lname):
            fuse(blk, "conv1", "bn1"); fuse(blk, "conv2", "bn2"); fuse(blk, "conv3", "bn3")
            ds = blk.downsample
            if isinstance(ds, nn.Sequential) and len(ds) == 2 \
                    and isinstance(ds[0], nn.Conv2d) and isinstance(ds[1], nn.BatchNorm2d):
                ds[0] = fuse_conv_bn_eval(ds[0], ds[1]); ds[1] = nn.Identity()
    return model


def build_fp16(weights=None):
    m = tvm.resnet50(weights=weights).eval().cuda()
    fold_bn(m)
    return m.to(memory_format=torch.channels_last).half()


def build_quantized(kind, x_calib, skip_pointwise, weights=None):
    """kind in {'int8','int4','int8_chained'}. Returns a calibrated static model.
    `weights` (a torchvision ResNet50_Weights enum) uses pretrained weights so the
    static PTQ scales are meaningful for real-accuracy checks (random weights only
    give valid *speed*, not accuracy)."""
    m = build_fp16(weights).float()  # convert works from fp32 weights
    convert = (convert_model_to_optimized_int4 if kind == "int4"
               else convert_model_to_optimized_int8)
    convert(m, skip_pointwise=skip_pointwise)
    # First conv has in_channels=3 -> too small for the 128-tile CUTLASS GEMM; keep fp16.
    # convert leaves it as OptimizedConv; restore a plain fp16 conv from a fresh model.
    fresh = build_fp16(weights)
    m.conv1 = fresh.conv1
    # NOTE: do NOT .half() -- that would cast the OptimizedConv's fp32 scale buffers
    # (static_input_scale / weight_scale_channel) to fp16, but the CUTLASS kernels
    # read them as float*. Keep params fp32; autocast makes the activations fp16.
    m = m.to(memory_format=torch.channels_last)
    # Calibrate each conv's input scale from the FP16 reference activations (standard
    # PTQ). The module-level begin/end_calibration runs convs in "calibrating mode"
    # whose output magnitude differs from the calibrated fast path, so downstream
    # scales come out ~10x wrong (saturating int8) -- fatal for chaining. Instead
    # hook the fp16 model, record each conv's input absmax, and set_static_scale.
    absmax = {}
    ref = build_fp16(weights)
    handles = []
    for name, mod in ref.named_modules():
        if isinstance(mod, nn.Conv2d):
            def hook(m_, inp, _n=name):
                a = inp[0].detach().abs().amax().item()
                absmax[_n] = max(absmax.get(_n, 0.0), a)
            handles.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            ref(x_calib)
    for h in handles:
        h.remove()
    n = 0
    for name, mod in m.named_modules():
        if isinstance(mod, QCONV):
            a = absmax.get(name, None)
            if a is None or a <= 0:
                continue
            mod.set_static_scale(127.0 / a)
            mod.set_standard_output_fp16(True)
            mod.enable_modiff(False)
            n += mod.is_calibrated
    del ref
    if kind == "int8_chained":
        # Keep activations int8 across conv1->conv2->conv3 within each bottleneck.
        chain_int8_bottlenecks(m)
    return m, n


def timed(model, x, repeats, warmups):
    """Median ms/iter over `repeats` timed runs (CUDA-event timed)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmups):
            model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); model(x); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--skip_pointwise", action="store_true",
                    help="quantize only 3x3 convs (skip 1x1). Default: quantize all convs.")
    a = ap.parse_args()

    x = torch.randn(a.batch, 3, 224, 224, device="cuda").contiguous(memory_format=torch.channels_last).half()
    print(f"ResNet-50  batch={a.batch}  input=224x224  (BatchNorm folded into conv)\n")

    models = {"fp16": build_fp16()}
    n_conv = sum(1 for _ in models["fp16"].modules() if isinstance(_, nn.Conv2d))
    print(f"fp16: {n_conv} conv layers")
    for kind in ("int8", "int4", "int8_chained", "int8_fullchain"):
        sp = a.skip_pointwise
        # int8_fullchain = whole-net int8 threading: build a plain calibrated int8
        # model, then wrap so the block-entry quantize is fused into the previous
        # block's conv3 store (one quantize for the whole net). See chained_bottleneck.
        build_kind = "int8" if kind == "int8_fullchain" else kind
        try:
            m, nq = build_quantized(build_kind, x, skip_pointwise=sp)
        except Exception as ex:
            print(f"{kind}: all-conv convert failed ({str(ex)[:60]}...), retrying 3x3-only")
            m, nq = build_quantized(build_kind, x, skip_pointwise=True)
        if kind == "int8_fullchain":
            from integration.fused_ops.chained_bottleneck import build_fully_chained
            m = build_fully_chained(m)
        models[kind] = m
        print(f"{kind}: {nq} calibrated quantized conv layers")

    # sanity: outputs finite
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for k, m in models.items():
            o = m(x)
            assert torch.isfinite(o).all(), f"{k} produced non-finite output"

    # warmup each model, then interleave the timed runs to spread any drift evenly
    order = ["fp16", "int8", "int4", "int8_chained", "int8_fullchain"]
    for k in order:
        timed(models[k], x, repeats=0, warmups=a.warmups)
    samples = {k: [] for k in order}
    for _ in range(a.repeats):
        for k in order:
            samples[k] += timed(models[k], x, repeats=1, warmups=0)

    med = {k: statistics.median(v) for k, v in samples.items()}
    std = {k: statistics.pstdev(v) for k, v in samples.items()}
    print(f"\n=== END-TO-END ===")
    print(f"{'mode':<14}{'median ms':>12}{'stdev':>9}{'vs fp16':>10}")
    for k in order:
        spd = med["fp16"] / med[k]
        print(f"{k:<14}{med[k]:>12.3f}{std[k]:>9.3f}{spd:>9.2f}x")
    print(f"int4 vs int8:              {med['int8']/med['int4']:.2f}x")
    print(f"int8_chained   vs int8:    {med['int8']/med['int8_chained']:.2f}x")
    print(f"int8_chained   vs fp16:    {med['fp16']/med['int8_chained']:.2f}x  "
          f"({'WIN' if med['int8_chained'] < med['fp16'] else 'no win'})")
    print(f"int8_fullchain vs chained: {med['int8_chained']/med['int8_fullchain']:.2f}x")
    print(f"int8_fullchain vs fp16:    {med['fp16']/med['int8_fullchain']:.2f}x  "
          f"({'WIN' if med['int8_fullchain'] < med['fp16'] else 'no win'})")

    microbench_convs(a.batch)


def microbench_convs(batch):
    """Isolate the COMPUTE win: time conv+dequant on an already-int8 activation
    (forward_from_intX, skipping the K1 quantize) vs fp16 F.conv2d, on the four
    representative ResNet-50 3x3 conv shapes. This is the speedup achievable IF the
    activation were already quantized (int8-native conv->conv chaining, TensorRT
    style) -- i.e. with the quantize overhead hidden, as diffusion hides it in GN."""
    import torch.nn.functional as F
    import modiff_cutlass as mc
    shapes = [(64, 56), (128, 28), (256, 14), (512, 7)]  # (channels, spatial), 3x3
    print(f"\n=== RAW 3x3 CONV KERNEL (quantize hidden; batch={batch}) ===")
    print(f"{'shape':<16}{'fp16 ms':>10}{'int8 ms':>10}{'int4 ms':>10}{'i8/fp16':>9}{'i4/fp16':>9}")

    def bench(fn, iters=50):
        for _ in range(10): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    for C, HW in shapes:
        x = torch.randn(batch, C, HW, HW, device="cuda").contiguous(memory_format=torch.channels_last)
        # fp16 reference conv (cuDNN)
        conv = nn.Conv2d(C, C, 3, padding=1, bias=False).cuda().to(memory_format=torch.channels_last).half()
        xh = x.half()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t_fp16 = bench(lambda: conv(xh))
            # int8 / int4: calibrate an OptimizedConv, pre-quantize input, time conv-only
            row = {}
            for kind, Opt, conv_mod in (("int8", OptimizedInt8Conv2d, None), ("int4", OptimizedInt4Conv2d, None)):
                base = nn.Conv2d(C, C, 3, padding=1, bias=False)
                q = Opt(base).cuda()
                q.begin_calibration()
                for _ in range(3): q(x.float())
                q.end_calibration(); q.set_standard_output_fp16(True); q.enable_modiff(False)
                q._ensure_conv_caches(x.device)
                if kind == "int8":
                    xq = mc.scale_quantize_int8(x.float(), q._cached_scale_tensor)
                    row[kind] = bench(lambda: q.forward_from_int8(xq))
                else:
                    xq = mc.scale_quantize_and_pack(x.float(), q._cached_scale_tensor)
                    row[kind] = bench(lambda: q.forward_from_int4(xq, HW, HW))
        print(f"{f'{C}ch@{HW}x{HW}':<16}{t_fp16:>10.3f}{row['int8']:>10.3f}{row['int4']:>10.3f}"
              f"{t_fp16/row['int8']:>8.2f}x{t_fp16/row['int4']:>8.2f}x")


if __name__ == "__main__":
    main()
