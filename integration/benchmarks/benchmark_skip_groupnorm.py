"""
Benchmark GroupNorm cost within ResBlocks.

Runs only the `skip_gnorm` condition (FusedGroupNormSiLU → identity),
then combines with existing skip_nonlinear_fixed data to decompose T_R:

    T_GN   = T_full      - T_skip_gnorm    GroupNorm+SiLU inside each FusedResBlock
    T_Conv = T_skip_gnorm - T_skip_res     Conv2d 3×3 (×2) inside each FusedResBlock
    Verify: T_GN + T_Conv = T_R            (from Section 11)

Usage:
    python integration/benchmarks/benchmark_skip_groupnorm.py \\
        --steps 200 --num_samples 168 --batch_size 42

    # Quick smoke-test
    python integration/benchmarks/benchmark_skip_groupnorm.py \\
        --steps 10 --num_samples 8 --batch_size 8 --modes fp16 int4_baseline
"""
import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT = os.path.join(os.path.dirname(__file__), "benchmark_ldm.py")
_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_skip_gnorm(mode: str, steps: int, num_samples: int, batch_size: int,
                   calib: str, output_dir: str,
                   linear_backend: str, linear_int_gemm_min_m: int) -> dict | None:
    """Run benchmark_ldm.py --no_groupnorm in a subprocess."""
    out_dir     = os.path.join(output_dir, f"{mode}_skip_gnorm")
    result_json = os.path.join(out_dir, "results.json")

    cmd = [
        sys.executable, SCRIPT,
        "--mode", mode,
        "--steps", str(steps),
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--output_dir", out_dir,
        "--skip_calibration",
        "--no_groupnorm",
        "--linear_backend", linear_backend,
        "--linear_int_gemm_min_m", str(linear_int_gemm_min_m),
    ]
    # Only pass --calibration when one was asked for. Unset, benchmark_ldm resolves through
    # CALIBRATION_PREFERENCE (run_mode falls back at benchmark_ldm.py:1325); the previous
    # argparse default was the stub-checkpoint file. `None` in argv would also raise TypeError
    # in subprocess.run, so the flag is conditional rather than empty.
    if calib:
        cmd += ["--calibration", calib]

    print(f"\n[SKIP_GNORM] Running {mode.upper()} ...", flush=True)
    t0   = time.perf_counter()
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  ERROR (exit {proc.returncode}):")
        print(proc.stderr[-3000:])
        return None

    try:
        with open(result_json) as f:
            data = json.load(f)
        r  = data[mode]
        ms = r["time_per_sample"] * 1000
        print(f"  {mode:<18} skip_gnorm  {ms:8.1f} ms/sample  "
              f"{r['time_per_step_ms']:7.2f} ms/step  (wall {elapsed:.0f}s)")
        return {
            "mode":              mode,
            "time_per_sample_s": r["time_per_sample"],
            "time_per_step_ms":  r["time_per_step_ms"],
            "wall_time_s":       round(elapsed, 1),
        }
    except Exception as e:
        print(f"  Could not parse results.json: {e}")
        return None


def load_existing(json_path: str) -> dict:
    """Load by_mode data from a previous skip_nonlinear_fixed run."""
    if not os.path.exists(json_path):
        print(f"  [WARN] existing JSON not found: {json_path}")
        return {}
    with open(json_path) as f:
        data = json.load(f)
    by_mode = data.get("by_mode", {})
    result  = {}
    for mode, conds in by_mode.items():
        result[mode] = {
            "full":      conds.get("(False, False)"),
            "skip_attn": conds.get("(True, False)"),
            "skip_res":  conds.get("(False, True)"),
            "skip_both": conds.get("(True, True)"),
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GroupNorm cost benchmark (skip_gnorm = FusedGroupNormSiLU → identity)"
    )
    parser.add_argument("--steps",       type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=168)
    parser.add_argument("--batch_size",  type=int, default=42)
    parser.add_argument("--modes", nargs="+",
                        default=["fp32", "fp16", "int8_baseline",
                                 "int4_baseline", "int8", "int4"])
    parser.add_argument("--int8_calib",
                        default=None)
    parser.add_argument("--int4_calib",
                        default=None)
    parser.add_argument("--output_dir",
                        default="integration/results/skip_groupnorm_bs42_n168_s200")
    parser.add_argument("--existing_json",
                        default="integration/results/"
                                "skip_nonlinear_fixed_bs42_n168_s200.json",
                        help="JSON from previous skip_nonlinear_fixed run "
                             "(provides full/skip_res/skip_both data)")
    parser.add_argument("--output_json",
                        default="integration/results/"
                                "skip_groupnorm_bs42_n168_s200.json")
    parser.add_argument("--linear_backend", choices=["fp16", "int_gemm"],
                        default="fp16")
    parser.add_argument("--linear_int_gemm_min_m", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Run skip_gnorm for each mode ─────────────────────────────────
    skip_gnorm_results: dict[str, dict] = {}
    for mode in args.modes:
        calib = args.int4_calib if "int4" in mode else args.int8_calib
        r = run_skip_gnorm(mode, args.steps, args.num_samples, args.batch_size,
                           calib, args.output_dir,
                           args.linear_backend, args.linear_int_gemm_min_m)
        if r:
            skip_gnorm_results[mode] = r

    # ── Load existing full/skip_res/skip_both data ────────────────────
    existing = load_existing(args.existing_json)

    # ── Compute decomposition ─────────────────────────────────────────
    sep = "=" * 100
    print(f"\n{sep}")
    print("RESBLOCK INTERNAL DECOMPOSITION: T_GN  vs  T_Conv")
    print(sep)
    print(f"GPU: NVIDIA A40 | steps={args.steps} | batch={args.batch_size} | "
          f"samples={args.num_samples}")
    print()
    print("  T_GN   = T_full      - T_skip_gnorm   (GroupNorm+SiLU in each FusedResBlock)")
    print("  T_Conv = T_skip_gnorm - T_skip_res    (Conv2d 3×3 ×2 per ResBlock)")
    print("  T_O    = T_skip_both                  (Other: time_emb, decoder I/O, etc.)")
    print()

    hdr = (f"  {'Mode':<18} {'T_full':>9} {'T_sg':>9} {'T_sr':>9} "
           f"{'T_GN':>8} {'T_Conv':>8} "
           f"{'T_GN/T_R':>10} {'T_Conv/T_R':>12}")
    print(hdr)
    print("-" * 90)

    decomp: dict[str, dict] = {}
    for mode in args.modes:
        sg_r  = skip_gnorm_results.get(mode)
        ex    = existing.get(mode, {})
        full_r = ex.get("full")
        sr_r   = ex.get("skip_res")

        if sg_r and full_r and sr_r:
            t_full = full_r["time_per_sample_s"] * 1000
            t_sg   = sg_r["time_per_sample_s"]   * 1000
            t_sr   = sr_r["time_per_sample_s"]   * 1000
            t_r    = t_full - t_sr          # total ResBlock cost (Section 11)
            t_gn   = t_full - t_sg          # GroupNorm+SiLU
            t_conv = t_sg   - t_sr          # Conv2d only
            pct_gn   = t_gn   / t_r * 100 if t_r > 0 else float("nan")
            pct_conv = t_conv / t_r * 100 if t_r > 0 else float("nan")

            print(f"  {mode:<18} {t_full:>9.1f} {t_sg:>9.1f} {t_sr:>9.1f} "
                  f"{t_gn:>8.1f} {t_conv:>8.1f} "
                  f"{pct_gn:>9.1f}% {pct_conv:>10.1f}%")
            decomp[mode] = {
                "T_full_ms":      round(t_full, 1),
                "T_skip_gnorm_ms": round(t_sg,  1),
                "T_skip_res_ms":  round(t_sr,   1),
                "T_R_ms":         round(t_r,    1),
                "T_GN_ms":        round(t_gn,   1),
                "T_Conv_ms":      round(t_conv,  1),
                "T_GN_pct":       round(pct_gn,  1),
                "T_Conv_pct":     round(pct_conv, 1),
            }
        else:
            missing = []
            if not sg_r:   missing.append("skip_gnorm")
            if not full_r: missing.append("full")
            if not sr_r:   missing.append("skip_res")
            print(f"  {mode:<18} {'missing: ' + ', '.join(missing):>60}")

    # ── Per-component speedup table ───────────────────────────────────
    fp32 = decomp.get("fp32")
    if fp32:
        print(f"\n{sep}")
        print("PER-COMPONENT SPEEDUP vs FP32")
        print(sep)
        hdr2 = (f"  {'Mode':<18} {'T_GN':>8} {'T_Conv':>8} "
                f"{'SpeedGN':>9} {'SpeedConv':>11}")
        print(hdr2)
        print("-" * 60)
        for mode in args.modes:
            d = decomp.get(mode)
            if not d: continue
            sg  = fp32["T_GN_ms"]   / d["T_GN_ms"]   if d["T_GN_ms"]   > 0 else float("nan")
            sc  = fp32["T_Conv_ms"] / d["T_Conv_ms"]  if d["T_Conv_ms"] > 0 else float("nan")
            print(f"  {mode:<18} {d['T_GN_ms']:>8.1f} {d['T_Conv_ms']:>8.1f} "
                  f"{sg:>8.2f}x {sc:>10.2f}x")

    # ── Save ──────────────────────────────────────────────────────────
    with open(args.output_json, "w") as f:
        json.dump({
            "config":             vars(args),
            "skip_gnorm_results": skip_gnorm_results,
            "decomposition":      decomp,
        }, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
