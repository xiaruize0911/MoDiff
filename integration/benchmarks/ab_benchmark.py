#!/usr/bin/env python3
"""Noise-robust A/B speed benchmark for the LDM pipeline.

Run-to-run noise on this pipeline is ~5%, as large as the gaps between modes,
so single-shot numbers are meaningless. This harness:
  - builds each requested mode ONCE (model load + quant conversion + calibration),
  - warms up each (cuDNN autotune, clock spin-up),
  - takes N INTERLEAVED timed measurements (round-robin across modes each round,
    so thermal/clock drift is shared across modes rather than biasing one),
  - reports median / mean / stdev / IQR of per-step ms,
  - saves raw per-repeat samples to JSON, and can diff against a saved baseline,
    reporting a mode's before->after delta as significant only when
    |median Δ| > max(stdev_before, stdev_after).

Usage:
  # capture baseline (before a change):
  python integration/benchmarks/ab_benchmark.py --modes int8_baseline int8 \
      --repeats 6 --label before --out integration/results/ab/before.json

  # after the change, compare:
  python integration/benchmarks/ab_benchmark.py --modes int8_baseline int8 \
      --repeats 6 --label after --out integration/results/ab/after.json \
      --compare integration/results/ab/before.json
"""
import os, sys, json, time, argparse, statistics
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import importlib.util
spec = importlib.util.spec_from_file_location("bldm", os.path.join(HERE, "benchmark_ldm.py"))
bldm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bldm)


def build(mode, args):
    # A mode may carry a per-run linear backend as "base:backend", e.g.
    # "int8:int_gemm" -> base mode int8 with the fused W8A8 linear kernel.
    base, _, lb = mode.partition(":")
    lb = lb or args.linear_backend
    # CALIBRATION_PREFERENCE, not a hardcoded path -- see the note in
    # integration/benchmarks/report/kernel_suites_bench.py. The old literal was the
    # stub-checkpoint file, demoted to last resort on 2026-08-12.
    cal = args.calibration or bldm._default_calibration_path(base)
    runner = bldm.BenchmarkRunner(
        config_path=args.config, ckpt_path=args.ckpt, output_dir="/tmp/ab_out",
        batch_size=args.batch_size, steps=args.steps, calibration_path=cal, linear_backend=lb)
    model, sampler = runner._setup_model(base)
    # Calibration comes from the --calibration file (loaded by _setup_model), so
    # int8 and int8_baseline are calibrated identically. Pre-generate it once.
    return runner, model, sampler


def timed_step_ms(runner, model, sampler, mode, args):
    """One timed sample() call -> per-step ms."""
    use_ac = mode != "fp32"
    cond = runner._cond_kwargs(model, args.batch_size)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_ac):
        sampler.sample(S=args.steps, batch_size=args.batch_size,
                       shape=runner.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / args.steps * 1000.0


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    q1 = xs[n // 4]; q3 = xs[(3 * n) // 4]
    return {"median": statistics.median(xs), "mean": statistics.fmean(xs),
            "stdev": statistics.pstdev(xs) if n > 1 else 0.0,
            "min": xs[0], "max": xs[-1], "iqr": q3 - q1, "n": n, "samples": xs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["int8_baseline", "int8"])
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--config", default="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml")
    ap.add_argument("--ckpt", default="models/ldm/lsun_churches256/model.ckpt")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--linear_backend", default="fp16", choices=["fp16", "int_gemm"],
                    help="default linear backend; override per-mode with 'int8:int_gemm'")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="baseline json to diff against")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()} | modes={args.modes} "
          f"| repeats={args.repeats} steps={args.steps} batch={args.batch_size}")

    built = {}
    for m in args.modes:
        print(f"[build] {m} ...")
        runner, model, sampler = build(m, args)
        built[m] = (runner, model, sampler)
        for _ in range(args.warmups):
            timed_step_ms(runner, model, sampler, m, args)  # warmup (untimed)
    torch.cuda.synchronize()

    samples = {m: [] for m in args.modes}
    for r in range(args.repeats):              # interleave: round-robin over modes
        for m in args.modes:
            runner, model, sampler = built[m]
            samples[m].append(timed_step_ms(runner, model, sampler, m, args))
        print(f"[round {r+1}/{args.repeats}] " +
              " | ".join(f"{m}={samples[m][-1]:.3f}ms" for m in args.modes))

    stats = {m: summarize(samples[m]) for m in args.modes}
    print("\n=== per-step ms (label={}) ===".format(args.label))
    print(f"{'mode':<18}{'median':>9}{'stdev':>8}{'iqr':>8}{'min':>8}{'max':>8}")
    for m in args.modes:
        s = stats[m]
        print(f"{m:<18}{s['median']:>9.3f}{s['stdev']:>8.3f}{s['iqr']:>8.3f}{s['min']:>8.3f}{s['max']:>8.3f}")

    payload = {"label": args.label, "steps": args.steps, "batch_size": args.batch_size,
               "gpu": torch.cuda.get_device_name(), "stats": stats}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(payload, open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")

    if args.compare and os.path.exists(args.compare):
        base = json.load(open(args.compare))
        print(f"\n=== A/B vs {base.get('label','?')} ({args.compare}) ===")
        print(f"{'mode':<18}{'before':>9}{'after':>9}{'delta':>9}{'delta%':>8}  verdict")
        for m in args.modes:
            if m not in base["stats"]:
                continue
            b, a = base["stats"][m], stats[m]
            d = a["median"] - b["median"]
            noise = max(b["stdev"], a["stdev"])
            verdict = ("faster" if d < 0 else "slower") if abs(d) > noise else "within-noise"
            print(f"{m:<18}{b['median']:>9.3f}{a['median']:>9.3f}{d:>+9.3f}"
                  f"{100*d/b['median']:>+7.1f}%  {verdict} (noise={noise:.3f})")


if __name__ == "__main__":
    main()
