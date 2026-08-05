"""Attention speedup at LAYER scope: time the real converted AttentionBlock module, per shape, per mode.

This exists because two earlier attention numbers in this tree disagree, and both were measuring
something other than "the attention layer":

  4.54x  docs/report_2026-08-04 (first draft) -- a GN + quantize + flash-core microbenchmark whose
         fp16 reference was pinned to SDPBackend.MATH. MATH materializes the [BH,T,T] score matrix,
         so the reference was ~6x too slow and every ratio was inflated by that factor.
         token_major_attention.py:64-73 documents this exact trap.
  0.76x  the same microbenchmark with the fp16 reference on PyTorch's default (flash) backend --
         correct as a microbenchmark, but it charges int8 for a standalone Q/K/V quantize pass that
         the production forward does not run, and it excludes the qkv and output projections
         entirely.
  1.24x  MEASUREMENT_REPORT_2026-08-01, from NVTX ranges over the WHOLE attention layer in a real
         run. That is the number a report should quote, because it is the thing the model spends
         time on.

So this measures the whole layer: build the model in each mode, take the actual AttentionBlock
modules the mode produced (fused GN->qkv, quantized core, fused proj epilogue, all of it), and time
`block(x)` on its real input shape with CUDA events. Reports mean +- 95% CI, CV and spread over
rounds, matching the 08-01 report's statistics so the two are directly comparable.
"""

import csv
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks

D = "docs/report_2026-08-04"
BATCH = int(os.environ.get("AL_BATCH", "128"))
ITERS = int(os.environ.get("AL_ITERS", "60"))
WARM = int(os.environ.get("AL_WARM", "30"))
REPS = int(os.environ.get("AL_REPS", "8"))
MODES = ["fp16", "int8_baseline", "int8", "int4_baseline", "int4"]
CALIB = {"int8": "integration/calibration/int8_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}


def build(mode):
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    cal = None if mode == "fp16" else CALIB["int4" if "int4" in mode else "int8"]
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{D}/tmp_out", batch_size=BATCH, steps=50, shape=(4, 32, 32),
        calibration_path=cal)
    model, _ = runner._setup_model(mode)
    return model


def attn_blocks(model):
    """Every attention module in the UNet, with the (C, H) of its input.

    The channel count comes from the module's own `channels`/`norm` so it is whatever the conversion
    produced; H is derived from the UNet's downsampling schedule for that channel width.
    """
    out = []
    for name, m in model.model.diffusion_model.named_modules():
        c = getattr(m, "channels", None)
        if c is None or not hasattr(m, "norm"):
            continue
        if "Attention" not in type(m).__mro__[0].__name__ and "attn" not in type(m).__name__.lower():
            # accept anything whose class name mentions attention, at any wrapper depth
            if not any("attention" in t.__name__.lower() for t in type(m).__mro__):
                continue
        out.append((name, m, int(c)))
    return out


def spatial_for(c):
    """LSUN-churches LDM 32x32 latent, ch=192, mult (1,2,3,4) -> C 192/384/576/768 at H 32/16/8/4.
    Measured shapes in MEASUREMENT_REPORT_2026-08-01: C192/T1024, C384/T256, C384/T64, C768/T16,
    C768/T4 -- i.e. a channel width can appear at more than one resolution, so the map is by
    (C -> the resolutions that actually occur) and each module is timed at every one of them."""
    return {192: [32], 384: [16, 8], 576: [8], 768: [4, 2]}.get(c, [8])


def bench(fn):
    """CUDA-event median over REPS rounds; returns (mean, ci95, cv, spread, n) in us/call."""
    ts = []
    for _ in range(REPS):
        for _ in range(WARM):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(ITERS):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / ITERS * 1e3)
    mean = st.mean(ts)
    sd = st.stdev(ts) if len(ts) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(ts))
    return mean, ci, (sd / mean * 100 if mean else 0), ((max(ts) - min(ts)) / mean * 100 if mean else 0), len(ts)


def main():
    burn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(50):
        burn = burn @ burn * 1e-4 + 1.0
    torch.cuda.synchronize()
    del burn
    torch.cuda.empty_cache()

    per = {}          # (C,T) -> {mode: stats}
    counts = {}
    for mode in MODES:
        model = build(mode)
        blocks = attn_blocks(model)
        print(f"\n{mode}: {len(blocks)} attention modules", flush=True)
        seen = {}
        for name, m, c in blocks:
            for H in spatial_for(c):
                T = H * H
                key = (c, T)
                if key in seen:          # one timing per distinct shape, not per module
                    continue
                x = torch.randn(BATCH, c, H, H, device="cuda", dtype=torch.float16
                                ).contiguous(memory_format=torch.channels_last)
                try:
                    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                                    dtype=torch.float16):
                        m(x)
                        torch.cuda.synchronize()

                        def call(mm=m, xx=x):
                            mm(xx)
                        stats = bench(call)
                except Exception as ex:
                    print(f"   C{c}/T{T}: SKIP ({type(ex).__name__}: {str(ex)[:70]})", flush=True)
                    continue
                seen[key] = True
                per.setdefault(key, {})[mode] = stats
                counts[key] = counts.get(key, 0) + 1
                print(f"   C{c}/T{T:<5} {stats[0]:9.1f} +- {stats[1]:5.1f} us  "
                      f"CV {stats[2]:4.2f}%  spread {stats[3]:4.2f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    keys = sorted(per, key=lambda k: -k[1])
    rows = []
    print(f"\n{'=' * 92}\nAttention LAYER, whole module, batch {BATCH}, us/call\n{'=' * 92}")
    hdr = f"{'shape':>12} " + " ".join(f"{m:>16}" for m in MODES) + "   int8/fp16  int4/fp16"
    print(hdr)
    for k in keys:
        c, T = k
        r = {"shape": f"C{c}/T{T}", "C": c, "T": T}
        line = f"{r['shape']:>12} "
        for m in MODES:
            s = per[k].get(m)
            r[f"{m}_us"] = round(s[0], 1) if s else ""
            r[f"{m}_ci95"] = round(s[1], 1) if s else ""
            r[f"{m}_cv_pct"] = round(s[2], 2) if s else ""
            r[f"{m}_spread_pct"] = round(s[3], 2) if s else ""
            r[f"{m}_n"] = s[4] if s else ""
            line += f"{(s[0] if s else float('nan')):16.1f} "
        f16 = per[k].get("fp16", [None])[0]
        for m in ("int8_baseline", "int4_baseline", "int8", "int4"):
            s = per[k].get(m)
            r[f"{m}_vs_fp16"] = round(f16 / s[0], 3) if (s and f16) else ""
        print(line + f"   {r.get('int8_baseline_vs_fp16', ''):>9} {r.get('int4_baseline_vs_fp16', ''):>10}")
        rows.append(r)

    # Weighted total over the 21 real blocks, using the per-shape block counts from the model.
    with open(f"{D}/data/attn_layer_speed.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWROTE {D}/data/attn_layer_speed.csv")


if __name__ == "__main__":
    main()
