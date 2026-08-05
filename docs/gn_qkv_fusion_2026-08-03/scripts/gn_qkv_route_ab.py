"""Same-process A/B of the four INT8 GN->QKV routes, at every eligible attention shape.

Why this exists: the fused GN->qkv route is gated behind MODIFF_FUSE_GN_QKV_INT8 (and its
int8-emitting sibling behind MODIFF_ROUTE1), and the 1.37x/1.15x in the source comment was
measured on 2026-07-23 (7b5b431) against the THEN baseline -- separate GroupNorm+quantize
followed by a plain W8A8 qkv GEMM. The W8A8 QKV int8-epilogue route landed a week later and
returns from forward() BEFORE either opt-in branch is reached, so both flags are unreachable in
steady state and the recorded speedups no longer describe production. This measures all four
against each other in one process.

Routes (selected by instance attribute, exactly as forward() dispatches on them):
  P   production      GN+SiLU+quantize -> gemm_w8a8_awq_qkv_i8_layouts   (_int8_qkv_epilogue)
  R1  route 1         fused_gn_qkv_i8evt -> quantize_attn_kv_from_i8     (_route1)
  A   fp16 fused GN   fused_gn_qkv (fp16 out) -> normal flash quantize   (_fuse_gn_qkv_i8)
  N   pre-07-30 base  _qkv_from_gn: GN+quantize -> plain W8A8 qkv GEMM   (neither)

R1 and A rebuild an fp16 qkv weight from qweight*w_scale, so the INT8 WEIGHT quantization is
preserved bit-for-bit; what they drop is the qkv projection's ACTIVATION quantization (A8->A16).
Output deltas below are therefore expected to be nonzero and to move toward the fp32 reference,
not away from it.

Timing is round-alternating over the routes so thermal drift and clock ramp hit every route
equally; the reported number is the median of the round medians, matching ck_bench_stats.
"""

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/taming-transformers"))
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))

import torch

import layer_pipeline_bench as layer_bench

WARMUPS = int(os.environ.get("AB_WARMUPS", "12"))
ROUNDS = int(os.environ.get("AB_ROUNDS", "8"))
ITERS = int(os.environ.get("AB_ITERS", "60"))
CALIB = int(os.environ.get("AB_CALIB", "16"))

# (label, description, attribute overrides)
ROUTES = [
    ("P", "production W8A8 QKV int8-epilogue",
     dict(_int8_qkv_epilogue=True, _route1=False, _fuse_gn_qkv_i8=False)),
    ("R1", "int8-emitting fused GN->qkv (MODIFF_ROUTE1)",
     dict(_int8_qkv_epilogue=False, _route1=True, _fuse_gn_qkv_i8=False)),
    ("A", "fp16 fused GN->qkv (MODIFF_FUSE_GN_QKV_INT8)",
     dict(_int8_qkv_epilogue=False, _route1=False, _fuse_gn_qkv_i8=True)),
    ("N", "pre-07-30 baseline: GN+quantize -> plain W8A8 qkv",
     dict(_int8_qkv_epilogue=False, _route1=False, _fuse_gn_qkv_i8=False)),
]


def time_calls(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    return out


def rotating_bench(fns, warmups, rounds, iterations):
    """fns: {label: callable}. Rotates route order every round so no route always runs first."""
    labels = list(fns)
    for i in range(warmups):
        for label in labels[i % len(labels):] + labels[:i % len(labels)]:
            fns[label]()
    torch.cuda.synchronize()
    round_medians = {label: [] for label in labels}
    for r in range(rounds):
        order = labels[r % len(labels):] + labels[:r % len(labels)]
        for label in order:
            round_medians[label].append(
                statistics.median(time_calls(fns[label], iterations)))
    result = {}
    for label in labels:
        rm = round_medians[label]
        mean = statistics.mean(rm)
        sd = statistics.stdev(rm) if len(rm) > 1 else 0.0
        result[label] = {
            "median_us": statistics.median(rm),
            "mean_us": mean,
            "cv_pct": round(100.0 * sd / mean, 3) if mean else None,
            "round_medians_us": [round(v, 1) for v in rm],
        }
    return result


def apply_route(module, overrides):
    for key, value in overrides.items():
        setattr(module, key, value)


def main():
    model, sampler, layers = layer_bench.collect_layers("int8")
    del sampler
    groups = {}
    for row in layers:
        if row["kind"] == "attention":
            groups.setdefault(tuple(row["x_shape"]), []).append(row)

    from integration.fused_ops.token_major_attention import _FUSE_TILE_M

    out = {
        "gpu": torch.cuda.get_device_name(),
        "warmups": WARMUPS, "rounds": ROUNDS, "iters": ITERS,
        "routes": {label: desc for label, desc, _ in ROUTES},
        "shapes": [],
    }

    with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.float16):
        for shape in sorted(groups, key=lambda s: -s[2] * s[3]):
            rows = groups[shape]
            module = rows[0]["module"]
            b, c, H, W = shape
            T = H * W
            x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous(
                memory_format=torch.channels_last)
            eligible = (T % _FUSE_TILE_M) == 0 and (c % 8) == 0

            # Freeze the flash/qkv static scales on the production route first: every route
            # here requires _fq_frozen2, and calibration must not run inside a timed region.
            apply_route(module, ROUTES[0][2])
            for _ in range(CALIB):
                module(x)
            torch.cuda.synchronize()
            frozen = bool(getattr(module, "_fq_frozen2", False))

            entry = {
                "x_shape": list(shape), "T": T, "channels": c,
                "n_instances": len(rows), "example": rows[0]["name"],
                "fuse_eligible": eligible, "scales_frozen": frozen,
            }

            active = [r for r in ROUTES if eligible or r[0] in ("P", "N")]

            def make(overrides):
                def run():
                    apply_route(module, overrides)
                    return module(x)
                return run

            fns = {label: make(ov) for label, _, ov in active}

            # Reference output + kernel composition per route, before timing.
            refs, kernels = {}, {}
            for label, _, ov in active:
                apply_route(module, ov)
                refs[label] = module(x).float().clone()
                seq, roles, gpu_us = layer_bench.kernel_sequence(fns[label])
                kernels[label] = {
                    "gpu_us_sum": gpu_us,
                    "launches": len(seq),
                    "roles": {r: round(v["us"], 1) for r, v in roles.items()},
                    "kernels": [(k["kernel"], k["us_per_layer_call"]) for k in seq],
                }
            torch.cuda.synchronize()

            base = refs["P"]
            entry["vs_P"] = {
                label: {
                    "max_abs_diff": (r - base).abs().max().item(),
                    "rel_l2": ((r - base).norm() / base.norm()).item(),
                } for label, r in refs.items() if label != "P"
            }

            timing = rotating_bench(fns, WARMUPS, ROUNDS, ITERS)
            p_us = timing["P"]["median_us"]
            for label in timing:
                timing[label]["speedup_vs_P"] = round(p_us / timing[label]["median_us"], 4)
                timing[label].update(kernels[label])
            entry["timing"] = timing
            out["shapes"].append(entry)

            print(f"\n=== {list(shape)}  T={T} C={c}  x{len(rows)}  "
                  f"fuse_eligible={eligible} frozen={frozen}")
            for label, _, _ in active:
                t = timing[label]
                print(f"  {label:2s} {t['median_us']:8.1f} us  CV {t['cv_pct']:5.2f}%  "
                      f"{t['speedup_vs_P']:6.3f}x vs P  "
                      f"{t['launches']} launches  gpu_sum {t['gpu_us_sum']:7.1f}")
            for label, d in entry["vs_P"].items():
                print(f"     {label} vs P: max|d| {d['max_abs_diff']:.4g}  "
                      f"relL2 {d['rel_l2']:.4g}")

            del x
            torch.cuda.empty_cache()

    path = os.environ.get(
        "AB_OUT", "docs/gn_qkv_fusion_2026-08-03/data/gn_qkv_route_ab.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {path}")


if __name__ == "__main__":
    main()
