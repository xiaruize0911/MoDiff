"""
Clean Skip-Attention Benchmark.

Each configuration runs in a separate subprocess via benchmark_ldm.py,
which avoids buffer-pool / global-state contamination that occurs when
multiple INT8/INT4 modes share the same Python process.

Modes tested (with and without --no_attention):
  fp16, int8_baseline, int4_baseline

Usage:
    python integration/benchmarks/benchmark_skip_attn_clean.py
    python integration/benchmarks/benchmark_skip_attn_clean.py --steps 200 --num_samples 32
"""
import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT = os.path.join(os.path.dirname(__file__), "benchmark_ldm.py")


def run_one(mode: str, no_attention: bool, steps: int, num_samples: int,
            batch_size: int, calib: str, output_dir: str) -> dict:
    """Run benchmark_ldm.py in a subprocess and parse the JSON result."""
    label = f"{mode}_{'skip' if no_attention else 'full'}_attn"
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
        "--skip_calibration",           # use existing calibration
    ]
    if no_attention:
        cmd.append("--no_attention")

    print(f"\n[{'SKIP' if no_attention else 'FULL'}] Running {mode.upper()} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  ERROR (exit {proc.returncode}):")
        print(proc.stderr[-2000:])
        return {"label": label, "error": True}

    # Parse time_per_sample from the saved results JSON
    try:
        with open(result_json) as f:
            data = json.load(f)
        r = data[mode]
        return {
            "label": label,
            "mode": mode,
            "no_attention": no_attention,
            "time_per_sample_s": r["time_per_sample"],
            "time_per_step_ms":  r["time_per_step_ms"],
            "speedup_vs_fp32":   r.get("speedup", None),
            "wall_time_s":       round(elapsed, 1),
        }
    except Exception as e:
        print(f"  Could not parse results.json: {e}")
        return {"label": label, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Clean skip-attention benchmark (subprocess-isolated)")
    parser.add_argument("--steps",       type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--calib", default=None,
                        help="Legacy override: use one calibration file for all modes")
    parser.add_argument("--int8_calib", default="integration/calibration/int8_calibration.pt")
    parser.add_argument("--int4_calib", default="integration/calibration/int4_calibration.pt")
    parser.add_argument("--output_dir",  default="integration/results/skip_attn_clean")
    parser.add_argument("--modes",       nargs="+",
                        default=["fp16", "int8_baseline", "int4_baseline"])
    parser.add_argument("--output_json", default="integration/results/skip_attn_clean.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for mode in args.modes:
        calib = args.calib
        if calib is None:
            calib = args.int4_calib if "int4" in mode else args.int8_calib
        for skip in [False, True]:
            r = run_one(
                mode=mode, no_attention=skip,
                steps=args.steps, num_samples=args.num_samples,
                batch_size=args.batch_size, calib=calib,
                output_dir=args.output_dir,
            )
            results.append(r)
            if "error" not in r:
                print(f"  {r['label']:35s}  {r['time_per_sample_s']*1000:8.1f} ms/sample"
                      f"  {r['time_per_step_ms']:7.2f} ms/step", flush=True)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SKIP ATTENTION BENCHMARK — CLEAN (subprocess-isolated)")
    print("=" * 80)
    print(f"GPU: NVIDIA A40  |  steps={args.steps}  batch={args.batch_size}"
          f"  samples={args.num_samples}")
    print()

    header = f"{'Mode':<20} {'Attn':<8} {'ms/sample':>10} {'ms/step':>10} "
    header += f"{'vs FP32':>10} {'vs full_attn':>14}"
    print(header)
    print("-" * 76)

    paired = {}       # mode -> {full, skip}
    for r in results:
        if "error" in r:
            continue
        mode = r["mode"]
        key  = "skip" if r["no_attention"] else "full"
        paired.setdefault(mode, {})[key] = r

    for mode in args.modes:
        p = paired.get(mode, {})
        for key, label in [("full", "full"), ("skip", "SKIP")]:
            r = p.get(key)
            if r is None:
                print(f"  {mode:<18} {label:<8} {'ERROR':>10}")
                continue
            t    = r["time_per_sample_s"]
            ms_s = r["time_per_step_ms"]
            fp32 = r.get("speedup_vs_fp32")
            fp32_s = f"{fp32:.2f}x" if fp32 else "—"

            full_t = p.get("full", {}).get("time_per_sample_s")
            if full_t and key == "skip":
                savings = (1 - t / full_t) * 100
                vs_full = f"{savings:+.1f}%"
            else:
                vs_full = "(ref)"

            print(f"  {mode:<18} {label:<8} {t*1000:>10.1f} {ms_s:>10.2f}"
                  f" {fp32_s:>10} {vs_full:>14}")
        print()

    # ── Breakdown: how much time is attention? ────────────────────────
    print("=" * 80)
    print("ATTENTION TIME ESTIMATE (from skip vs full delta)")
    print("=" * 80)
    print(f"  {'Mode':<20} {'Attn time (ms)':>16} {'% of pipeline':>16}")
    print("-" * 56)
    for mode in args.modes:
        p = paired.get(mode, {})
        full_r = p.get("full")
        skip_r = p.get("skip")
        if full_r and skip_r:
            attn_ms = (full_r["time_per_sample_s"] - skip_r["time_per_sample_s"]) * 1000
            pct     = attn_ms / (full_r["time_per_sample_s"] * 1000) * 100
            sign    = "+" if attn_ms > 0 else ""
            print(f"  {mode:<20} {sign}{attn_ms:>14.1f} ms {pct:>14.1f}%")

    # Save
    with open(args.output_json, "w") as f:
        json.dump({"config": vars(args), "results": results, "paired": {
            mode: {k: v for k, v in p.items()} for mode, p in paired.items()
        }}, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
