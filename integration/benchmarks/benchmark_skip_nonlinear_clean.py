"""
Clean Skip-ResBlock (Non-Linear Layers) Benchmark.

Complements the skip-attention ablation (Section 4 of BENCHMARK_REPORT.md) by
measuring the cost of ResBlock non-linear conv layers versus attention layers.

Each configuration runs in a separate subprocess via benchmark_ldm.py,
avoiding buffer-pool / global-state contamination across INT8/INT4 modes.

Layer taxonomy in the LSUN Churches UNet:
  - ResBlock   : GroupNorm + SiLU + Conv2d(3x3) x2  ← "non-linear" spatial ops
  - AttentionBlock : GroupNorm + QKV Conv1d + softmax + Proj Conv1d

Experiments:
  full            : all layers active          (reference)
  skip_attention  : ResBlocks only, no attention
  skip_resblock   : AttentionBlocks only, no ResBlock conv
  skip_both       : only time embedding + skip connections remain

Usage:
    python integration/benchmarks/benchmark_skip_nonlinear_clean.py \\
        --steps 200 --num_samples 168 --batch_size 42

    # Quick smoke-test
    python integration/benchmarks/benchmark_skip_nonlinear_clean.py \\
        --steps 10 --num_samples 8 --batch_size 8 --modes fp16 int8_baseline
"""
import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT = os.path.join(os.path.dirname(__file__), "benchmark_ldm.py")


def run_one(mode: str, no_attention: bool, no_resblock: bool,
            steps: int, num_samples: int, batch_size: int,
            calib: str, output_dir: str,
            linear_backend: str, linear_int_gemm_min_m: int) -> dict:
    """Run benchmark_ldm.py in a subprocess and parse the JSON result."""
    suffix = []
    if no_attention:
        suffix.append("skip_attn")
    if no_resblock:
        suffix.append("skip_res")
    if not suffix:
        suffix.append("full")
    label = f"{mode}_{'_'.join(suffix)}"
    out_dir = os.path.join(output_dir, label)
    result_json = os.path.join(out_dir, "results.json")

    cmd = [
        sys.executable, SCRIPT,
        "--mode", mode,
        "--steps", str(steps),
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--calibration", calib,
        "--output_dir", out_dir,
        "--skip_calibration",
        "--linear_backend", linear_backend,
        "--linear_int_gemm_min_m", str(linear_int_gemm_min_m),
    ]
    if no_attention:
        cmd.append("--no_attention")
    if no_resblock:
        cmd.append("--no_resblock")

    tag_parts = []
    if no_attention:
        tag_parts.append("SKIP_ATTN")
    if no_resblock:
        tag_parts.append("SKIP_RES")
    if not tag_parts:
        tag_parts.append("FULL")
    tag = "+".join(tag_parts)

    print(f"\n[{tag:18s}] Running {mode.upper()} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  ERROR (exit {proc.returncode}):")
        print(proc.stderr[-3000:])
        return {"label": label, "mode": mode,
                "no_attention": no_attention, "no_resblock": no_resblock,
                "error": True}

    try:
        with open(result_json) as f:
            data = json.load(f)
        r = data[mode]
        return {
            "label": label,
            "mode": mode,
            "no_attention": no_attention,
            "no_resblock": no_resblock,
            "time_per_sample_s": r["time_per_sample"],
            "time_per_step_ms":  r["time_per_step_ms"],
            "speedup_vs_fp32":   r.get("speedup", None),
            "wall_time_s":       round(elapsed, 1),
        }
    except Exception as e:
        print(f"  Could not parse results.json: {e}")
        return {"label": label, "mode": mode,
                "no_attention": no_attention, "no_resblock": no_resblock,
                "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Clean skip-nonlinear-layer benchmark (subprocess-isolated)"
    )
    parser.add_argument("--steps",       type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=168)
    parser.add_argument("--batch_size",  type=int, default=42)
    parser.add_argument("--int8_calib",
                        default="integration/calibration/int8_calibration.pt")
    parser.add_argument("--int4_calib",
                        default="integration/calibration/int4_calibration.pt")
    parser.add_argument("--output_dir",
                        default="integration/results/skip_nonlinear_clean")
    parser.add_argument("--modes", nargs="+",
                        default=["fp16", "int8_baseline", "int4_baseline",
                                 "int8", "int4"])
    parser.add_argument("--output_json",
                        default="integration/results/skip_nonlinear_clean.json")
    parser.add_argument("--linear_backend", choices=["fp16", "int_gemm"],
                        default="fp16")
    parser.add_argument("--linear_int_gemm_min_m", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Four conditions per mode: full / skip_attn / skip_resblock / skip_both
    conditions = [
        (False, False),   # full
        (True,  False),   # skip attention only
        (False, True),    # skip resblock only
        (True,  True),    # skip both
    ]

    results = []
    for mode in args.modes:
        calib = args.int4_calib if "int4" in mode else args.int8_calib
        for (no_attn, no_res) in conditions:
            r = run_one(
                mode=mode,
                no_attention=no_attn,
                no_resblock=no_res,
                steps=args.steps,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                calib=calib,
                output_dir=args.output_dir,
                linear_backend=args.linear_backend,
                linear_int_gemm_min_m=args.linear_int_gemm_min_m,
            )
            results.append(r)
            if "error" not in r:
                tag = []
                if no_attn: tag.append("skip_attn")
                if no_res:  tag.append("skip_res")
                tag_s = "+".join(tag) if tag else "full"
                print(f"  {mode:<18} {tag_s:<20}  "
                      f"{r['time_per_sample_s']*1000:8.1f} ms/sample  "
                      f"{r['time_per_step_ms']:7.2f} ms/step", flush=True)

    # ── Organise by mode ─────────────────────────────────────────────
    by_mode = {}
    for r in results:
        if "error" in r:
            continue
        mode = r["mode"]
        key  = ("attn" if r["no_attention"] else "", "res" if r["no_resblock"] else "")
        # Use a simple tuple key
        cond = (r["no_attention"], r["no_resblock"])
        by_mode.setdefault(mode, {})[cond] = r

    # ── Print summary ────────────────────────────────────────────────
    sep = "=" * 90
    print(f"\n{sep}")
    print("SKIP NON-LINEAR (RESBLOCK) BENCHMARK — CLEAN (subprocess-isolated)")
    print(sep)
    print(f"GPU: NVIDIA A40  |  steps={args.steps}  batch={args.batch_size}  "
          f"samples={args.num_samples}")
    print()

    cond_labels = {
        (False, False): "full",
        (True,  False): "skip_attn",
        (False, True):  "skip_resblock",
        (True,  True):  "skip_both",
    }

    hdr = f"  {'Mode':<18} {'Condition':<16} {'ms/sample':>10} {'ms/step':>10} " \
          f"{'vs FP32':>9} {'vs full':>10}"
    print(hdr)
    print("-" * 80)

    for mode in args.modes:
        p = by_mode.get(mode, {})
        full_r = p.get((False, False))
        for cond, clabel in cond_labels.items():
            r = p.get(cond)
            if r is None:
                print(f"  {mode:<18} {clabel:<16} {'ERROR':>10}")
                continue
            t    = r["time_per_sample_s"]
            ms_s = r["time_per_step_ms"]
            fp32 = r.get("speedup_vs_fp32")
            fp32_s = f"{fp32:.2f}x" if fp32 else "—"
            if full_r and cond != (False, False):
                delta = (1 - t / full_r["time_per_sample_s"]) * 100
                vs_full = f"{delta:+.1f}%"
            else:
                vs_full = "(ref)"
            print(f"  {mode:<18} {clabel:<16} {t*1000:>10.1f} {ms_s:>10.2f}"
                  f" {fp32_s:>9} {vs_full:>10}")
        print()

    # ── Cost decomposition table ─────────────────────────────────────
    print(sep)
    print("LAYER COST DECOMPOSITION (ms/sample at batch_size=%d)" % args.batch_size)
    print(sep)
    print(f"  {'Mode':<18} {'ResBlock (ms)':>14} {'Attention (ms)':>16} "
          f"{'Other (ms)':>12} {'Total (ms)':>12}")
    print("-" * 76)

    for mode in args.modes:
        p = by_mode.get(mode, {})
        full_r   = p.get((False, False))
        skip_a   = p.get((True,  False))  # full - attn
        skip_r   = p.get((False, True))   # full - resblock
        skip_b   = p.get((True,  True))   # only other
        if not all([full_r, skip_a, skip_r, skip_b]):
            print(f"  {mode:<18} {'data missing':>44}")
            continue
        total_ms  = full_r["time_per_sample_s"] * 1000
        # time when only resblocks are skipped → only attention + other runs
        attn_other_ms = skip_r["time_per_sample_s"] * 1000
        # time when only attention is skipped → only resblocks + other runs
        res_other_ms  = skip_a["time_per_sample_s"] * 1000
        # time when both are skipped → only other (embedding, norms outside blocks, etc.)
        other_ms      = skip_b["time_per_sample_s"] * 1000
        # isolate each component
        res_ms  = res_other_ms  - other_ms
        attn_ms = attn_other_ms - other_ms
        print(f"  {mode:<18} {res_ms:>12.1f}   {attn_ms:>14.1f}   "
              f"{other_ms:>10.1f}   {total_ms:>10.1f}")
    print()
    print("  Note: Other = time embedding, GroupNorm outside blocks, decoder input/output, etc.")
    print("        ResBlock + Attention + Other may not sum exactly to Total due to interaction")
    print("        effects (memory bandwidth, cache, kernel overlap).")

    # Save
    with open(args.output_json, "w") as f:
        json.dump({
            "config": vars(args),
            "results": results,
            "by_mode": {
                mode: {str(k): v for k, v in p.items()}
                for mode, p in by_mode.items()
            },
        }, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
