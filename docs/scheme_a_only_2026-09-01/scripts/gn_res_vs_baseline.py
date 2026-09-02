"""PTQ vs MoDiff-commit vs skip-store on the e2e GN+residual conv path.

Same shapes/freq as skip_kernel_layer.py. PTQ = fused GN+SiLU+quantize +
forward_from_int8(+residual). MoDiff = forward_gn_fused_modiff write/skip.
"""
import json, os, statistics, sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]
os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_GN_FAST"] = "1"
os.environ["MODIFF_WARMUP_STEPS"] = "1"

import torch, torch.nn as nn
import modiff_cutlass as mc
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

DEV, CL, N = "cuda", torch.channels_last, 128
WARMUP, REPS, TRIALS = 8, 40, 3
GN_GROUPS = 32
OUT = "docs/scheme_a_only_2026-09-01/data/gn_res_vs_baseline.json"
UNET = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]


def cl(t):
    return t.contiguous(memory_format=CL)


def gnq():
    if os.environ.get("MODIFF_GN_FAST", "1") == "1":
        f = getattr(mc, "group_norm_silu_quantize_nhwc_fast", None)
        if f is not None:
            return f
    return mc.group_norm_silu_quantize_nhwc


def median_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(TRIALS):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(REPS):
            fn()
        e1.record()
        torch.cuda.synchronize()
        xs.append(e0.elapsed_time(e1) / REPS)
    return statistics.median(xs) * 1000.0


def build(Cin, Cout, modiff):
    raw = nn.Conv2d(Cin, Cout, 3, padding=1).to(DEV)
    layer = OptimizedInt8Conv2d(raw, layer_name="bench").to(DEV)
    layer.set_static_scale(16.0)
    layer.static_delta_scale.fill_(16.0)
    layer.static_delta_alpha.fill_(1.0 / 16.0)
    layer.is_delta_calibrated.fill_(True)
    layer._delta_cal = True
    layer.is_calibrated = True
    if not modiff:
        layer.set_standard_output_fp16(True)
    layer.enable_modiff(modiff)
    layer.fuse_input_silu = True
    layer.eval()
    return layer


def main():
    gn_quant = gnq()
    empty_f32 = torch.empty(0, device=DEV, dtype=torch.float32)
    empty_f16 = torch.empty(0, device=DEV, dtype=torch.float16)
    rows = []
    print(f"GPU {torch.cuda.get_device_name(0)}  gnq={gn_quant.__name__}", flush=True)
    for Cin, Cout, H, W, freq in UNET:
        if Cin % GN_GROUPS:
            continue
        x = cl(torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
        res = cl(torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))
        gw = torch.ones(Cin, device=DEV, dtype=torch.float16)
        gb = torch.zeros(Cin, device=DEV, dtype=torch.float16)
        ptq = build(Cin, Cout, False)
        md = build(Cin, Cout, True)
        md.reset_state()
        with torch.inference_mode():
            md(x)
        assert md.is_first_step is False
        scale = ptq.static_input_scale.view(1)
        smooth = getattr(ptq, "_empty_smooth", empty_f32)
        if smooth is None:
            smooth = empty_f32

        def baseline():
            with torch.inference_mode():
                q = gn_quant(x, gw, gb, GN_GROUPS, 1e-6, True, scale, smooth,
                             empty_f16, empty_f16)
                ptq.forward_from_int8(q, residual=res)

        def commit():
            with torch.inference_mode():
                md.forward_gn_fused_modiff(x, gw, gb, GN_GROUPS, 1e-6, empty_f16, empty_f16, res)

        def skip():
            with torch.inference_mode():
                md.forward_gn_fused_modiff(x, gw, gb, GN_GROUPS, 1e-6, empty_f16, empty_f16, res)

        baseline(); torch.cuda.synchronize()
        b = median_us(baseline)
        os.environ["MODIFF_CACHE_SKIP_K"] = "1"
        commit(); torch.cuda.synchronize()
        c = median_us(commit)
        os.environ["MODIFF_CACHE_SKIP_K"] = "1000000"
        skip(); torch.cuda.synchronize()
        s = median_us(skip)
        os.environ["MODIFF_CACHE_SKIP_K"] = "1"
        rec = {
            "shape": f"{Cin}->{Cout} {H}x{W}", "freq": freq,
            "Cin": Cin, "Cout": Cout, "H": H, "W": W,
            "baseline_us": b, "commit_us": c, "skip_us": s,
        }
        rows.append(rec)
        print(f"{rec['shape']:22s} f{freq:<3}  PTQ {b:7.1f}  MoDiff {c:7.1f}  "
              f"skip {s:7.1f}  skip/PTQ {b/s:.3f}×  MoDiff/PTQ {b/c:.3f}×", flush=True)

    den = sum(r["freq"] for r in rows)
    fw = {k: sum(r[k] * r["freq"] for r in rows) / den
          for k in ("baseline_us", "commit_us", "skip_us")}
    print(f"\nfreq-weighted            PTQ {fw['baseline_us']:7.1f}  "
          f"MoDiff {fw['commit_us']:7.1f}  skip {fw['skip_us']:7.1f}", flush=True)
    payload = {"gpu": torch.cuda.get_device_name(0), "batch": N, "gn_groups": GN_GROUPS,
               "method": f"median {TRIALS}×{REPS} after {WARMUP} warmup",
               "path": "PTQ=GN+quantize+forward_from_int8(+res); MoDiff=forward_gn_fused_modiff",
               "rows": rows, "freq_weighted": fw}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
