"""The E2E arm table, measured on ONE conv layer's KERNEL 1 (fused GroupNorm(+SiLU)->quantize).

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

ARMS = [("fp16", "fp16", None),
        ("int8", "W8A8 PTQ（无 MoDiff）", None),
        ("int8", "W8A8 MoDiff, a_hat fp16", "fp16"),
        ("int8", "W8A8 MoDiff, a_hat i8 B=32", "i8"),
        ("int8", "W8A8 MoDiff, a_hat i4 B=32", "i4"),
        ("int4", "W4A4 PTQ（无 MoDiff）", None),
        ("int4", "W4A4 MoDiff, a_hat fp16", "fp16"),
        ("int4", "W4A4 MoDiff, a_hat i8 B=32", "i8"),
        ("int4", "W4A4 MoDiff, a_hat i4 B=32", "i4")]

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

def one(B, C, H, W, prec, ahat):
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
            As = torch.ones(B, H, W, C // BLK, device=DEV, dtype=torch.float32)
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

def table(rows, title):
    base = rows[0]
    print(f"\n### {title}")
    print("| arm | ms | vs fp16 | peak alloc MB | vs fp16 | a_hat cache MB |")
    print("|---|---|---|---|---|---|")
    for lab, ms, pk, ah in rows:
        print(f"| {lab} | {ms:.4f} | {base[1]/ms:.3f}x | {pk:.1f} | {pk/base[2]:.2f}x | "
              f"{'—' if ah == 0 else f'{ah:.1f}'} |")

# ---- one representative layer: the most frequent shape in the UNet ----
C, H, W, B, f = max(SHAPES, key=lambda s: s[4])
rows = [(lab,) + one(B, C, H, W, prec, ah) for prec, lab, ah in ARMS]
table(rows, f"single conv layer, kernel 1 only — C={C} {H}x{W} batch={B} (most frequent, {f}x)")

# ---- a bigger, memory-bound layer, where the a_hat traffic actually shows ----
C2, H2, W2, B2, f2 = max(SHAPES, key=lambda s: s[0]*s[1]*s[2])
rows2 = [(lab,) + one(B2, C2, H2, W2, prec, ah) for prec, lab, ah in ARMS]
table(rows2, f"single conv layer, kernel 1 only — C={C2} {H2}x{W2} batch={B2} (largest, {f2}x)")

# ---- frequency-weighted over all 20 shapes, comparable to the E2E totals ----
agg = {lab: [0.0, 0.0, 0.0] for _, lab, _ in ARMS}
for C, H, W, B, f in SHAPES:
    for prec, lab, ah in ARMS:
        ms, pk, am = one(B, C, H, W, prec, ah)
        agg[lab][0] += ms * f
        agg[lab][1] = max(agg[lab][1], pk)
        agg[lab][2] += am
rows3 = [(lab, agg[lab][0], agg[lab][1], agg[lab][2]) for _, lab, _ in ARMS]
table(rows3, "all 20 UNet conv shapes, frequency-weighted total (peak = max over shapes, "
             "a_hat cache = sum over layers)")
json.dump({"single": rows, "largest": rows2, "weighted": rows3},
          open("docs/ahat_only_conv_2026-09-02/data/kernel1_table.json", "w"), indent=1)
