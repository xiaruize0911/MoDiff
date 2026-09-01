"""Store Freeze: per-kernel and per-layer skip vs commit.

Kernel:
  evt_nr     already-quantized int8 → EVT o_hat (no residual)
  evt_res    same + ResBlock skip (dual-store vs skip)
  step1      static delta quantize, write_ahat on/off
  gn         GN+SiLU+delta quantize, write_ahat on/off

Layer:
  fwd        OptimizedInt8Conv2d.forward (step1 + conv)
  gn_res     forward_gn_fused_modiff + residual (e2e hot path)

Run: source setup_cuda_env.sh && python docs/speedup_w8a8_2x_2026-08-31/scripts/skip_kernel_layer.py
"""
import json
import os
import statistics
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_WARMUP_STEPS"] = "1"

import torch
import torch.nn as nn
import modiff_cutlass as mc
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

DEV = "cuda"
CL = torch.channels_last
N = 128
WARMUP, REPS, TRIALS = 8, 40, 3
STRIDES = (1, 1, 1, 1, 1, 1)
GN_GROUPS = 32
OUT = "docs/speedup_w8a8_2x_2026-08-31/data/skip_kernel_layer.json"

UNET = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]


def cl(t):
    return t.contiguous(memory_format=CL)


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(REPS):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / REPS


def median_us(fn):
    return statistics.median(time_fn(fn) for _ in range(TRIALS)) * 1000.0


def pair(commit_fn, skip_fn):
    commit_fn(); skip_fn(); torch.cuda.synchronize()
    c, s = median_us(commit_fn), median_us(skip_fn)
    return {"commit_us": c, "skip_us": s, "skip_vs_commit": c / s if s else 0.0,
            "saved_us": c - s}


def empty_f32():
    return torch.empty(0, device=DEV, dtype=torch.float32)


def empty_f16():
    return torch.empty(0, device=DEV, dtype=torch.float16)


def time_kernels(Cin, Cout, H, W):
    xq = cl(torch.randint(-8, 8, (N, Cin, H, W), device=DEV, dtype=torch.int8))
    w = torch.randint(-8, 8, (Cout, 3, 3, Cin), device=DEV, dtype=torch.int8).contiguous()
    inv = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
    wscale = torch.full((Cout,), 0.02, device=DEV, dtype=torch.float32)
    out = cl(torch.empty(N, Cout, H, W, device=DEV, dtype=torch.float16))
    ohat = cl(0.1 * torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))
    res = cl(0.1 * torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))

    x = cl(torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
    ahat = cl(0.1 * torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
    scale = torch.tensor([16.0], device=DEV, dtype=torch.float32)
    smooth = empty_f32()
    ahat_s = empty_f32()

    rec = {}
    rec["evt_nr"] = pair(
        lambda: mc.conv2d_int8_evt_o_hat(xq, w, inv, wscale, ohat, *STRIDES),
        lambda: mc.conv2d_int8_evt_o_hat_skip(xq, w, inv, wscale, ohat, out, *STRIDES),
    )
    rec["evt_res"] = pair(
        lambda: mc.conv2d_int8_evt_o_hat_residual(xq, w, inv, wscale, ohat, res, out, *STRIDES),
        lambda: mc.conv2d_int8_evt_o_hat_residual_skip(xq, w, inv, wscale, ohat, res, out, *STRIDES),
    )
    rec["step1"] = pair(
        lambda: mc.step1_static_quantize_fprop(x, ahat, scale, smooth, False, True),
        lambda: mc.step1_static_quantize_fprop(x, ahat, scale, smooth, False, False),
    )

    # GN stats need C % groups == 0. Skip GN on odd channel counts.
    rec["gn"] = None
    if Cin % GN_GROUPS == 0:
        gw = torch.ones(Cin, device=DEV, dtype=torch.float16)
        gb = torch.zeros(Cin, device=DEV, dtype=torch.float16)
        e = empty_f32()
        em = empty_f16()
        rec["gn"] = pair(
            lambda: mc.group_norm_silu_delta_quantize_nhwc(
                x, gw, gb, ahat, GN_GROUPS, 1e-6, True, scale, smooth, em, em,
                e, e, e, e, 127.0, False, 1.0, False, True, ahat_s),
            lambda: mc.group_norm_silu_delta_quantize_nhwc(
                x, gw, gb, ahat, GN_GROUPS, 1e-6, True, scale, smooth, em, em,
                e, e, e, e, 127.0, False, 1.0, False, False, ahat_s),
        )
    return rec


def build_layer(Cin, Cout, H, W):
    raw = nn.Conv2d(Cin, Cout, 3, padding=1).to(DEV)
    layer = OptimizedInt8Conv2d(raw, layer_name="bench").to(DEV)
    layer.set_static_scale(16.0)
    layer.static_delta_scale.fill_(16.0)
    layer.static_delta_alpha.fill_(1.0 / 16.0)
    layer.is_delta_calibrated.fill_(True)
    layer._delta_cal = True
    layer.enable_modiff(True)
    layer.eval()
    return layer


def time_layers(Cin, Cout, H, W):
    xs = [cl(torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16)) for _ in range(3)]
    md = build_layer(Cin, Cout, H, W)
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    md.reset_state()
    with torch.inference_mode():
        md(xs[0])
    assert md.is_first_step is False

    def commit_fwd():
        os.environ["MODIFF_CACHE_SKIP_K"] = "1"
        with torch.inference_mode():
            md(xs[1])

    def skip_fwd():
        os.environ["MODIFF_CACHE_SKIP_K"] = "1000000"
        with torch.inference_mode():
            md(xs[1])

    rec = {"fwd": pair(commit_fwd, skip_fwd)}

    rec["gn_res"] = None
    if Cin % GN_GROUPS == 0:
        md.fuse_input_silu = True
        md.reset_state()
        with torch.inference_mode():
            md(xs[0])
        gw = torch.ones(Cin, device=DEV, dtype=torch.float16)
        gb = torch.zeros(Cin, device=DEV, dtype=torch.float16)
        em = empty_f16()
        residual = cl(torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))

        def commit_gn():
            os.environ["MODIFF_CACHE_SKIP_K"] = "1"
            with torch.inference_mode():
                md.forward_gn_fused_modiff(xs[1], gw, gb, GN_GROUPS, 1e-6, em, em, residual)

        def skip_gn():
            os.environ["MODIFF_CACHE_SKIP_K"] = "1000000"
            with torch.inference_mode():
                md.forward_gn_fused_modiff(xs[1], gw, gb, GN_GROUPS, 1e-6, em, em, residual)

        rec["gn_res"] = pair(commit_gn, skip_gn)

    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    return rec


def weighted(rows, nest, key):
    num = den = 0.0
    for r in rows:
        block = r[nest] if nest is None else r[nest]
        if block is None or key not in block:
            continue
        num += block[key] * r["freq"]
        den += r["freq"]
    return num / den if den else None


def fmt(block):
    if block is None:
        return "  n/a"
    return (f"  commit {block['commit_us']:8.1f}  skip {block['skip_us']:8.1f}  "
            f"{block['skip_vs_commit']:.3f}×  Δ {block['saved_us']:+7.1f} µs")


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  B={N}  "
          f"warmup={WARMUP} reps={REPS} trials={TRIALS}", flush=True)
    rows = []
    for Cin, Cout, H, W, freq in UNET:
        shape = f"{Cin}->{Cout} {H}x{W}"
        print(f"\n===== {shape}  f{freq} =====", flush=True)
        k = time_kernels(Cin, Cout, H, W)
        ly = time_layers(Cin, Cout, H, W)
        rec = {"Cin": Cin, "Cout": Cout, "H": H, "W": W, "freq": freq, "shape": shape,
               "kernel": k, "layer": ly}
        rows.append(rec)
        print(f"  kernel evt_nr  {fmt(k['evt_nr'])}", flush=True)
        print(f"  kernel evt_res {fmt(k['evt_res'])}", flush=True)
        print(f"  kernel step1   {fmt(k['step1'])}", flush=True)
        print(f"  kernel gn      {fmt(k['gn'])}", flush=True)
        print(f"  layer  fwd     {fmt(ly['fwd'])}", flush=True)
        print(f"  layer  gn_res  {fmt(ly['gn_res'])}", flush=True)

    def wkey(path, key):
        nest, name = path
        num = den = 0.0
        for r in rows:
            block = r[nest].get(name) if isinstance(r[nest], dict) else None
            if not block:
                continue
            num += block[key] * r["freq"]
            den += r["freq"]
        return num / den if den else None

    print("\n===== freq-weighted =====", flush=True)
    fw = {}
    for nest, name, label in [
        ("kernel", "evt_nr", "kernel evt_nr "),
        ("kernel", "evt_res", "kernel evt_res"),
        ("kernel", "step1", "kernel step1  "),
        ("kernel", "gn", "kernel gn     "),
        ("layer", "fwd", "layer  fwd    "),
        ("layer", "gn_res", "layer  gn_res "),
    ]:
        c = wkey((nest, name), "commit_us")
        s = wkey((nest, name), "skip_us")
        if c is None:
            print(f"  {label}  n/a", flush=True)
            continue
        fw[f"{nest}_{name}"] = {
            "commit_us": c, "skip_us": s, "skip_vs_commit": c / s, "saved_us": c - s,
        }
        print(f"  {label}  commit {c:8.1f}  skip {s:8.1f}  {c/s:.3f}×  Δ {c-s:+7.1f} µs",
              flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "batch": N,
        "method": f"CUDA events, median of {TRIALS} trials × {REPS} reps after {WARMUP} warmup",
        "gn_groups": GN_GROUPS,
        "rows": rows,
        "freq_weighted": fw,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
