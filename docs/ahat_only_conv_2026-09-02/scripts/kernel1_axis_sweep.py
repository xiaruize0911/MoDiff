"""Kernel 1 (fused GN(+SiLU)->quantize) swept along ONE shape axis at a time.

The 20-shape model dump cannot answer "how does this scale with N/H/W/C": batch is fixed at
128 in it, and H==W in every UNet conv shape, so C and (H==W) are the only axes it separates.
This sweeps batch, C, H and W independently around a default, same style as
docs/conv_shape_sweep_2026-09-02/scripts/shape_sweep.py.

Arms: baseline (no MoDiff) / MoDiff a_hat fp16 / MoDiff a_hat int8 B=16 / B=32, at int8 and int4.
B=64 a_hat is not swept: it is rejected by bind_ahat_cache (block must be in [2,32]).
Time: CUDA events, median of 25 after 8 warmup. Peak: max_memory_allocated over
(allocate that arm's state + one launch), so it includes the a_hat cache and its block scales.
"""
import json, os, statistics, sys
ROOT = "/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0] = [ROOT]
import torch
import modiff_cutlass as mc

DEV, CL = "cuda", torch.channels_last
WARMUP, REPS = 8, 25
G, EPS = 32, 1e-6
DEFAULT = {"B": 128, "C": 384, "H": 16, "W": 16}
SWEEPS = {"B": [8, 16, 32, 64, 128, 256],
          "C": [128, 192, 256, 384, 512, 768, 1024, 1536],
          "H": [2, 4, 8, 16, 32, 64],
          "W": [2, 4, 8, 16, 32, 64]}
BLOCKS = [16, 32]

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)

def run_one(B, C, H, W, prec, arm, blk):
    """Fresh allocation per arm so the peak reflects exactly that arm's state."""
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        ef = torch.empty(0, device=DEV, dtype=torch.float32)
        eh = torch.empty(0, device=DEV, dtype=torch.float16)
        ei = torch.empty(0, device=DEV, dtype=torch.int32)
        # channels_last allocated DIRECTLY: zeros(...).contiguous(CL) holds the NCHW original
        # and its NHWC copy alive at once, and that 2x transient becomes the peak.
        x = torch.empty(B, C, H, W, device=DEV, dtype=torch.float16, memory_format=CL).normal_()
        gw = torch.randn(C, device=DEV, dtype=torch.float16)
        gb = torch.randn(C, device=DEV, dtype=torch.float16)
        sc = torch.full((1,), 16.0, device=DEV, dtype=torch.float32)
        Q = 127.0 if prec == "int8" else 7.0
        if arm in ("baseline", "baseline_generic"):
            sfx = "" if arm == "baseline_generic" else "_fast"
            if prec == "int8":
                f = getattr(mc, "group_norm_silu_quantize_nhwc" + sfx)
                fn = lambda: f(x, gw, gb, G, EPS, True, sc, ef, eh, eh)
            else:
                f = getattr(mc, "group_norm_silu_quantize_pack_nhwc" + sfx)
                fn = lambda: f(x, gw, gb, G, EPS, True, sc, ef, eh, eh, 0)
        else:
            if arm == "fp16":
                A = torch.empty(B, C, H, W, device=DEV, dtype=torch.float16, memory_format=CL).zero_()
                As = ef
            else:
                A = torch.empty(B, C, H, W, device=DEV, dtype=torch.int8, memory_format=CL).zero_()
                As = torch.ones(B, H, W, C // blk, device=DEV, dtype=torch.float32)
            if prec == "int8":
                fn = lambda: mc.group_norm_silu_delta_quantize_nhwc(
                    x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef, ei,
                    Q, False, 1.0, False, True, As)
            else:
                fn = lambda: mc.group_norm_silu_delta_quantize_pack_nhwc(
                    x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef, ei,
                    Q, False, 1.0, True, As)
        fn(); torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        ms = bench(fn)
        return {"ms": ms, "peak": peak}
    except Exception as e:
        return {"error": str(e).split("\n")[0][:110]}
    finally:
        for v in ("x", "gw", "gb", "sc", "A", "As", "fn"):
            if v in dir(): pass
        torch.cuda.empty_cache()

ARMS = [("baseline", 0), ("baseline_generic", 0), ("fp16", 0)] + [("i8", b) for b in BLOCKS]
out = {"gpu": torch.cuda.get_device_name(0), "default": DEFAULT,
       "method": f"CUDA events, median of {REPS} after {WARMUP} warmup; peak = "
                 "max_memory_allocated over arm-state allocation + one launch; num_groups=32",
       "sweeps": {}}
for axis, values in SWEEPS.items():
    rows = []
    for v in values:
        cfg = dict(DEFAULT); cfg[axis] = v
        row = {"axis": axis, "value": v, **cfg, "arms": {}}
        for prec in ("int8", "int4"):
            for arm, blk in ARMS:
                name = arm if arm.startswith("baseline") else (
                    "a_hat fp16" if arm == "fp16" else f"a_hat i8 B={blk}")
                if arm == "i8" and cfg["C"] % blk:
                    continue
                row["arms"][f"{prec}/{name}"] = run_one(
                    cfg["B"], cfg["C"], cfg["H"], cfg["W"], prec, arm, blk)
        rows.append(row)
        ok = {k: r for k, r in row["arms"].items() if "ms" in r and k.startswith("int8")}
        print(f"  {axis}={v:<5} " + "  ".join(
            f"{k.split('/')[1]}={r['ms']:.3f}/{r['peak']:.0f}MB" for k, r in ok.items()), flush=True)
        for k, r in row["arms"].items():
            if "error" in r: print(f"      SKIP {k}: {r['error']}", flush=True)
    out["sweeps"][axis] = rows
json.dump(out, open("docs/ahat_only_conv_2026-09-02/data/kernel1_axis_sweep.json", "w"), indent=1)
print("\nwrote docs/ahat_only_conv_2026-09-02/data/kernel1_axis_sweep.json")
