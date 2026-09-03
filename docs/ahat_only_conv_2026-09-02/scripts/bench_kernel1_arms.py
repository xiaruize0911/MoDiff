"""Kernel 1 of the conv block (the fused GN(+SiLU)->quantize stage) across three a_hat arms,
at W8A8 and W4A4, on every conv input shape the churches UNet actually has.

Arms (all at identical GN work; they differ only in what a_hat costs to read/write):
  baseline            -- no MoDiff at all: group_norm_silu_quantize[_pack]_nhwc
  MoDiff              -- delta + in-place a_hat update, a_hat stored fp16 (2 B/elem)
  MoDiff a_hat i8 B=x -- same, a_hat stored int8 + fp32 blockwise scales [N,H,W,C/B]

Time: CUDA events, median of REPS after WARMUP.
Peak memory: torch.cuda.max_memory_allocated() over (allocate arm state + run once), so it
includes the persistent a_hat cache and its scales, which is where the arms differ.
Shapes come from docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json's `unet` dump.
"""
import json, os, statistics, sys
ROOT = "/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0] = [ROOT]
import torch
import modiff_cutlass as mc

DEV, CL = "cuda", torch.channels_last
WARMUP, REPS = 8, 25
G, EPS = 32, 1e-6
BLOCKS = [int(b) for b in os.environ.get("AHAT_BLOCKS", "16,32,64").split(",")]

SHAPES = [(s["C"], s["H"], s["W"], s["B"], s["freq"])
          for s in json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))["unet"]]

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)

def measure(build, tag):
    """build() -> callable running the kernel once. Returns (ms, peak_MB) or (None, err)."""
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn = build()
        fn(); torch.cuda.synchronize()          # correctness/support probe + peak capture
    except Exception as e:
        torch.cuda.empty_cache()
        return None, str(e).split("\n")[0][:120]
    peak = torch.cuda.max_memory_allocated() / 2**20
    ms = bench(fn)
    del fn; torch.cuda.empty_cache()
    return ms, peak

out = {"gpu": torch.cuda.get_device_name(0), "shapes": [], "method":
       f"CUDA events, median of {REPS} after {WARMUP} warmup; peak = max_memory_allocated over "
       "arm-state allocation + one launch"}

for C, H, W, BATCH, freq in SHAPES:
    row = {"C": C, "H": H, "W": W, "B": BATCH, "freq": freq, "arms": {}}
    ef = torch.empty(0, device=DEV, dtype=torch.float32)
    ei = torch.empty(0, device=DEV, dtype=torch.int32)
    eh = torch.empty(0, device=DEV, dtype=torch.float16)

    def state(ahat=None, blk=0):
        """Allocate this arm's tensors; returns the closure's captured pieces."""
        x = torch.empty(BATCH, C, H, W, device=DEV, dtype=torch.float16,
                        memory_format=CL).normal_()
        gw = torch.randn(C, device=DEV, dtype=torch.float16)
        gb = torch.randn(C, device=DEV, dtype=torch.float16)
        sc = torch.full((1,), 16.0, device=DEV, dtype=torch.float32)
        A = As = None
        if ahat == "fp16":
            A = torch.empty(BATCH, C, H, W, device=DEV, dtype=torch.float16,
                            memory_format=CL).zero_()
            As = ef
        elif ahat == "int8":
            A = torch.empty(BATCH, C, H, W, device=DEV, dtype=torch.int8,
                            memory_format=CL).zero_()
            As = torch.ones(BATCH, H, W, C // blk, device=DEV, dtype=torch.float32)
        return x, gw, gb, sc, A, As

    for prec in ("int8", "int4"):
        Q = 127.0 if prec == "int8" else 7.0
        # ---- baseline: no a_hat, no delta ----
        # _gnq() in fused_resblock.py appends "_fast" unless MODIFF_GN_FAST=0, and it defaults
        # to 1 -- so the SHIPPED baseline is the _fast twin. The generic one is kept as a second
        # arm because it is 1.4-4.4x slower and skews every ratio taken against it.
        for sfx, label in (("_fast", "baseline"), ("", "baseline_generic")):
            def b_base(prec=prec, sfx=sfx):
                x, gw, gb, sc, _, _ = state()
                if prec == "int8":
                    f = getattr(mc, "group_norm_silu_quantize_nhwc" + sfx)
                    return lambda: f(x, gw, gb, G, EPS, True, sc, ef, eh, eh)
                f = getattr(mc, "group_norm_silu_quantize_pack_nhwc" + sfx)
                return lambda: f(x, gw, gb, G, EPS, True, sc, ef, eh, eh, 0)
            row["arms"][f"{prec}/{label}"] = measure(b_base, label)

        # ---- MoDiff, a_hat fp16 ----
        def b_modiff(prec=prec, Q=Q):
            x, gw, gb, sc, A, As = state("fp16")
            if prec == "int8":
                return lambda: mc.group_norm_silu_delta_quantize_nhwc(
                    x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef,
                    ei, Q, False, 1.0, False, True, ef)
            return lambda: mc.group_norm_silu_delta_quantize_pack_nhwc(
                x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef,
                ei, Q, False, 1.0, True, ef)
        row["arms"][f"{prec}/MoDiff a_hat fp16"] = measure(b_modiff, "modiff")

        # ---- MoDiff, a_hat int8 blockwise ----
        for blk in BLOCKS:
            if C % blk:
                continue
            def b_blk(prec=prec, Q=Q, blk=blk):
                x, gw, gb, sc, A, As = state("int8", blk)
                if prec == "int8":
                    return lambda: mc.group_norm_silu_delta_quantize_nhwc(
                        x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef,
                        ei, Q, False, 1.0, False, True, As)
                return lambda: mc.group_norm_silu_delta_quantize_pack_nhwc(
                    x, gw, gb, A, G, EPS, True, sc, ef, eh, eh, ef, ef, ef,
                    ei, Q, False, 1.0, True, As)
            row["arms"][f"{prec}/MoDiff a_hat i8 B={blk}"] = measure(b_blk, f"b{blk}")

    out["shapes"].append(row)
    ok = {k: v for k, v in row["arms"].items() if v[0] is not None}
    print(f"C{C:5d} {H:2d}x{W:<2d} B{BATCH} f{freq:<3d} " +
          " ".join(f"{k.split('/')[1][:12]}={v[0]:.3f}" for k, v in ok.items() if k.startswith("int8")),
          flush=True)
    for k, v in row["arms"].items():
        if v[0] is None:
            print(f"    UNSUPPORTED {k}: {v[1]}", flush=True)

json.dump(out, open("docs/ahat_only_conv_2026-09-02/data/kernel1_arms.json", "w"), indent=1)
print("\nwrote docs/ahat_only_conv_2026-09-02/data/kernel1_arms.json")
