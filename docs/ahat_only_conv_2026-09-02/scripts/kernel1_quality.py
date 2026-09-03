"""Kernel-1 speed AND quality in one table: frequency-weighted time+peak for every a_hat storage
format that has a measured eta_cum, joined against it. Extends kernel1_table.py to B=16 and B=64.
"""
OLDDOC = """

Same seven arms as the E2E table, but the only thing running is kernel 1 of one conv:
  fp16                       group_norm_silu_nhwc                    (GN+SiLU, fp16 out, no quant)
  W8A8 PTQ / W4A4 PTQ        group_norm_silu_quantize[_pack]_nhwc_fast   (shipped per-tensor)
  MoDiff, a_hat fp16         group_norm_silu_delta_quantize[_pack]_nhwc, fp16 a_hat
  MoDiff, a_hat i8 B=32      same, int8 a_hat + fp32 scales [N,H,W,C/32]

The `_fast` suffix on the PTQ arms is what _gnq() in fused_resblock.py resolves to at the
default MODIFF_GN_FAST=1 -- the generic entry point is 1.4-4.4x slower and is not what ships.

Time: CUDA events, median of 25 after 8 warmup.
Peak: max_memory_allocated over (allocate that arm's state + one launch), so it includes the
input, the a_hat cache, its block scales and the output -- the same accounting as the E2E table.
argv: shape label, or nothing for the whole 20-shape frequency-weighted sweep.
"""
import json, os, statistics, sys
ROOT = "/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0] = [ROOT]
import torch
import modiff_cutlass as mc

DEV, CL = "cuda", torch.channels_last
WARMUP, REPS, G, EPS, BLK = 8, 25, 32, 1e-6, 32
SHAPES = [(s["C"], s["H"], s["W"], s["B"], s["freq"])
          for s in json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))["unet"]]

# (precision, label, ahat format, block size) -- block size ignored when ahat is None/fp16
ARMS = [("int8", "W8A8 PTQ", None, 0),
        ("int8", "W8A8 MoDiff, a_hat fp16", "fp16", 0),
        ("int8", "W8A8 MoDiff, a_hat i8 B=16", "i8", 16),
        ("int8", "W8A8 MoDiff, a_hat i8 B=32", "i8", 32),
        ("int8", "W8A8 MoDiff, a_hat i8 B=64", "i8", 64),
        ("int8", "W8A8 MoDiff, a_hat i4 B=32", "i4", 32),
        ("int4", "W4A4 PTQ", None, 0),
        ("int4", "W4A4 MoDiff, a_hat fp16", "fp16", 0),
        ("int4", "W4A4 MoDiff, a_hat i8 B=16", "i8", 16),
        ("int4", "W4A4 MoDiff, a_hat i8 B=32", "i8", 32),
        ("int4", "W4A4 MoDiff, a_hat i8 B=64", "i8", 64),
        ("int4", "W4A4 MoDiff, a_hat i4 B=32", "i4", 32)]

# eta_cum measured through the real kernel over a captured 49-step trajectory,
# docs/ahat_accuracy_2026-09-02/data/validate_kernel.json (5-layer median).
ETA = {"fp16": 0.0015310330, "i8 B=16": 0.0411779419, "i8 B=32": 0.0507599212,
       "i8 B=64": 0.0607382938, "i4 B=32": 1.9247386456}

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

def one(B, C, H, W, prec, ahat, blk):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ef = torch.empty(0, device=DEV, dtype=torch.float32)
    eh = torch.empty(0, device=DEV, dtype=torch.float16)
    ei = torch.empty(0, device=DEV, dtype=torch.int32)
    x = torch.empty(B, C, H, W, device=DEV, dtype=torch.float16, memory_format=CL).normal_()
    gw = torch.randn(C, device=DEV, dtype=torch.float16)
    gb = torch.randn(C, device=DEV, dtype=torch.float16)
    sc = torch.full((1,), 16.0, device=DEV, dtype=torch.float32)
    ahat_MB = 0.0
    if prec == "fp16":
        fn = lambda: mc.group_norm_silu_nhwc(x, gw, gb, G, EPS, True, eh, eh)
    elif ahat is None:
        f = (mc.group_norm_silu_quantize_nhwc_fast if prec == "int8"
             else mc.group_norm_silu_quantize_pack_nhwc_fast)
        fn = ((lambda: f(x, gw, gb, G, EPS, True, sc, ef, eh, eh)) if prec == "int8"
              else (lambda: f(x, gw, gb, G, EPS, True, sc, ef, eh, eh, 0)))
    else:
        Q = 127.0 if prec == "int8" else 7.0
        if ahat == "fp16":
            A = torch.empty(B, C, H, W, device=DEV, dtype=torch.float16, memory_format=CL).zero_()
            As = ef; ahat_MB = A.numel() * 2 / 2**20
        else:
            # i4 packs two channels per byte, so the cache carries C/2 channels -- that int8
            # [N,C/2,H,W] channels_last tensor IS the nibble layout the kernel reads.
            chan = C // 2 if ahat == "i4" else C
            A = torch.empty(B, chan, H, W, device=DEV, dtype=torch.int8, memory_format=CL).zero_()
            As = torch.ones(B, H, W, C // blk, device=DEV, dtype=torch.float32)
            ahat_MB = (A.numel() + As.numel() * 4) / 2**20
        if prec == "int8":
            fn = lambda: mc.group_norm_silu_delta_quantize_nhwc(
                x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef, ei, Q, False, 1.0,
                False, True, As)
        else:
            fn = lambda: mc.group_norm_silu_delta_quantize_pack_nhwc(
                x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef, ei, Q, False, 1.0,
                True, As)
    fn(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    ms = bench(fn)
    del fn; torch.cuda.empty_cache()
    return ms, peak, ahat_MB


BPE = {"fp16": 2.0, "i8 B=16": 1.25, "i8 B=32": 1.125, "i8 B=64": 1.0625, "i4 B=32": 0.625}

agg = {lab: [0.0, 0.0, 0.0] for _, lab, _, _ in ARMS}
for C, H, W, B, f in SHAPES:
    for prec, lab, ah, blk in ARMS:
        ms, pk, am = one(B, C, H, W, prec, ah, blk)
        agg[lab][0] += ms * f
        agg[lab][1] = max(agg[lab][1], pk)
        agg[lab][2] += am
    print(f"  C{C} {H}x{W} done", flush=True)

out = []
for prec in ("int8", "int4"):
    base = agg[("W8A8" if prec == "int8" else "W4A4") + " PTQ"][0]
    for p, lab, ah, blk in ARMS:
        if p != prec: continue
        key = "fp16" if ah == "fp16" else (f"{ah} B={blk}" if ah else None)
        t, pk, am = agg[lab]
        out.append({"arm": lab, "prec": prec, "ms": t, "x_base": t / base,
                    "peak_MB": pk, "cache_MB": am, "bpe": BPE.get(key),
                    "eta_cum": ETA.get(key)})
json.dump(out, open("docs/ahat_only_conv_2026-09-02/data/kernel1_quality.json", "w"), indent=1)

print("\n| arm | ms | x PTQ | peak MB | cache MB | B/elem | eta_cum |")
print("|---|---|---|---|---|---|---|")
for r in out:
    e = f"{r['eta_cum']:.4f}" if r["eta_cum"] is not None else "—"
    b = f"{r['bpe']:.4f}" if r["bpe"] is not None else "—"
    print(f"| {r['arm']} | {r['ms']:.4f} | {r['x_base']:.3f}x | {r['peak_MB']:.0f} | "
          f"{r['cache_MB']:.0f} | {b} | {e} |")
