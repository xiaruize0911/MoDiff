"""Timing with a reported distribution, shared by every suite in the final report.

The previous harnesses each collapsed their measurement to a single median and threw the
distribution away (`layer_pipeline_bench.cuda_bench` returned `median(round_medians)`), so a
number could not be told apart from a number that happened to land there once. Everything here
keeps the per-round samples and reports spread alongside the central value.

Design notes that matter for reading the output:

* Two levels of aggregation. Inside a round we time `iters` back-to-back launches with CUDA
  events and take the MEDIAN of those, which rejects the occasional scheduler hiccup within a
  round. Across rounds we report mean/median/stdev of the round medians. So `cv_pct` is
  round-to-round reproducibility, NOT within-round jitter -- it is the quantity that tells you
  whether re-running the benchmark would give the same answer.
* The CI is a Student-t interval on the mean of the round medians, not +-1.96 sigma: with
  5-10 rounds the normal approximation understates the interval by 15-25%.
* Ratios (speedups) get their own interval by the delta method rather than dividing two
  central values and quoting it bare, because a 1.05x speedup whose interval straddles 1.0 is
  not a speedup.
"""
import math
import statistics

# Two-sided 95% t critical values by degrees of freedom (n-1). Falls back to the normal
# limit for large n. Hard-coded to avoid a scipy dependency in this container.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
        39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984}


def t95(df):
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.960


def summarize(samples):
    """Full descriptive stats for a list of per-round measurements (same unit in, same out)."""
    s = [float(x) for x in samples if x is not None and math.isfinite(float(x))]
    n = len(s)
    if n == 0:
        return None
    mean = statistics.fmean(s)
    med = statistics.median(s)
    sd = statistics.stdev(s) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    half = t95(n - 1) * sem if n > 1 else 0.0
    lo, hi = min(s), max(s)
    return {
        "n": n,
        "mean": mean,
        "median": med,
        "stdev": sd,
        "sem": sem,
        "cv_pct": (sd / mean * 100.0) if mean else 0.0,
        "min": lo,
        "max": hi,
        "spread_pct": ((hi - lo) / med * 100.0) if med else 0.0,
        "ci95_half": half,
        "ci95_lo": mean - half,
        "ci95_hi": mean + half,
        "iqr": (statistics.quantiles(s, n=4)[2] - statistics.quantiles(s, n=4)[0]) if n >= 4 else 0.0,
        "samples": [round(x, 4) for x in s],
    }


def ratio_ci(num, den):
    """95% CI for num/den (both summarize() dicts), delta method on independent means.

    Var(a/b) ~= (a/b)^2 * [ (sa/a)^2 + (sb/b)^2 ]  using the standard errors of the means.
    Returns None if either side is missing or degenerate.
    """
    if not num or not den or not num["mean"] or not den["mean"]:
        return None
    r = num["mean"] / den["mean"]
    rel = math.sqrt((num["sem"] / num["mean"]) ** 2 + (den["sem"] / den["mean"]) ** 2)
    df = max(1, min(num["n"], den["n"]) - 1)
    half = t95(df) * r * rel
    return {"ratio": r, "ci95_half": half, "ci95_lo": r - half, "ci95_hi": r + half,
            "rel_se_pct": rel * 100.0}


def speedup(fp16_stats, mode_stats):
    """Speedup = fp16_time / mode_time, with a CI. Ordered so >1 means the mode is faster."""
    return ratio_ci(fp16_stats, mode_stats)


def cuda_bench_stats(fn, warm=20, iters=60, rounds=8, sync=True):
    """Time `fn` with CUDA events: `warm` warmup calls, then `rounds` x `iters` timed calls.

    Returns (stats_dict_in_microseconds, error_or_None). The per-round value is the median of
    that round's `iters` samples; stats are computed over the round medians. Also records
    `within_round_cv_pct`, the median within-round CV, which separates "the kernel is jittery"
    from "the machine drifted between rounds".
    """
    import torch
    try:
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
    except Exception as e:                                  # noqa: BLE001
        return None, repr(e)[:160]

    round_meds, within_cv = [], []
    try:
        for _ in range(rounds):
            starts = [torch.cuda.Event(True) for _ in range(iters)]
            ends = [torch.cuda.Event(True) for _ in range(iters)]
            for i in range(iters):
                starts[i].record()
                fn()
                ends[i].record()
            if sync:
                torch.cuda.synchronize()
            ts = sorted(starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters))  # us
            round_meds.append(ts[len(ts) // 2])
            m = statistics.fmean(ts)
            within_cv.append((statistics.stdev(ts) / m * 100.0) if len(ts) > 1 and m else 0.0)
    except Exception as e:                                  # noqa: BLE001
        return None, repr(e)[:160]

    st = summarize(round_meds)
    if st is not None:
        st["iters_per_round"] = iters
        st["warmup"] = warm
        st["within_round_cv_pct"] = statistics.median(within_cv) if within_cv else 0.0
    return st, None


def fmt(st, unit="µs", prec=1):
    """Compact 'mean ± CI (CV%)' for tables."""
    if not st:
        return "—"
    return (f"{st['mean']:.{prec}f} ± {st['ci95_half']:.{prec}f} {unit} "
            f"(CV {st['cv_pct']:.2f}%)")


def fmt_speedup(sp):
    if not sp:
        return "—"
    return f"{sp['ratio']:.3f}× ± {sp['ci95_half']:.3f}"


def stability_verdict(st, cv_good=1.0, cv_ok=3.0):
    """One-word reproducibility label, so a table can be scanned for the untrustworthy rows."""
    if not st:
        return "n/a"
    cv = st["cv_pct"]
    return "tight" if cv <= cv_good else ("ok" if cv <= cv_ok else "NOISY")
