"""Audit the measured suites for results that are unexpected, self-inconsistent, or unexplained.

Purpose: catch the things a table of numbers hides. Each check is purely data-driven and prints
what it found plus the quantity that makes it suspicious; the judgement of whether a finding is
explained belongs in the report's prose, not here.

Checks:
  R1  quantized slower than FP16 (cross-mode ratio < 1)
  R2  INT4 slower than INT8 (the mode that should be strictly cheaper arithmetically)
  S1  NOISY rows (CV > 3%), and whether any is the dominant entry at its key
  S2  mean/median divergence -> skewed distribution, i.e. an outlier round
  S3  within-round CV far above between-round CV -> the kernel itself is jittery
  C1  gpu_busy_frac far from 1 -> launch-bound layer, summed kernel time != wall
  C2  stage sums vs measured wall (e2e)
  C3  layer kernel self-time sum vs pipeline_us
  I1  values identical across modes -> a shared, unquantized code path
  I2  signature/key count mismatches between modes
  D1  monotonic trend inside the e2e repeat sequence -> drift, samples not i.i.d.
  B1  implied bandwidth above the device peak -> a timing or accounting error
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ck_final_numbers import (MODES, attn_key, conv_key, lin_key, entry_priority)  # noqa: E402
from ck_bench_stats import ratio_ci  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
A40_PEAK_GBS = 696.0
FINDINGS = []


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def add(code, sev, what, detail):
    FINDINGS.append((code, sev, what, detail))


def cross_index(ks, suite, alignfn):
    idx = collections.defaultdict(dict)
    for m, _ in MODES:
        for r in ks["modes"].get(m, {}).get(suite, []):
            if not r.get("stats"):
                continue
            k = alignfn(r)
            if k is None:
                continue
            cur = idx[k].get(m)
            if cur is None or entry_priority(r) > entry_priority(cur):
                idx[k][m] = r
    return idx


def audit_kernels(ks):
    suites = {"attention": lambda r: "T=%s" % attn_key(r)[0],
              "conv": lambda r: conv_key(r), "linear": lambda r: lin_key(r)}
    for suite, alignfn in suites.items():
        idx = cross_index(ks, suite, alignfn)
        for k, v in idx.items():
            if len(v) != len(MODES):
                continue
            f = v["fp16"]["stats"]
            all_torch = all(v[m]["entry"].startswith("torch_") for m in v)
            for m, lbl in MODES[1:]:
                st = v[m]["stats"]
                r = ratio_ci(f, st)
                if r and r["ci95_hi"] < 1.0 and not all_torch:
                    add("R1", "high", "%s %s at %s slower than FP16" % (suite, lbl, k),
                        "%.3f× (CI %.3f-%.3f), %s vs FP16 %.1f µs, entry `%s`"
                        % (r["ratio"], r["ci95_lo"], r["ci95_hi"], "%.1f µs" % st["mean"],
                           f["mean"], v[m]["entry"]))
            e8, e4 = v["int8_baseline"]["stats"], v["int4_baseline"]["stats"]
            rr = ratio_ci(e8, e4)
            if rr and rr["ci95_hi"] < 1.0 and not all_torch:
                add("R2", "med", "%s INT4 slower than INT8 at %s" % (suite, k),
                    "INT4 %.1f vs INT8 %.1f µs (%.3f×, CI %.3f-%.3f)"
                    % (e4["mean"], e8["mean"], rr["ratio"], rr["ci95_lo"], rr["ci95_hi"]))
        # identical across modes
        for k, v in idx.items():
            if len(v) < 2:
                continue
            vals = {m: round(v[m]["stats"]["mean"], 1) for m in v}
            if len(set(vals.values())) == 1 and len(vals) > 1:
                add("I1", "low", "%s identical in %s at %s" % (suite, "/".join(vals), k),
                    "all %.1f µs -> the same kernel runs in each, entries %s"
                    % (list(vals.values())[0],
                       sorted({v[m]["entry"] for m in v})))
    # signature counts
    for suite in suites:
        counts = {m: len(ks["modes"].get(m, {}).get(suite, [])) for m, _ in MODES}
        if len(set(counts.values())) > 1:
            add("I2", "low", "%s signature count differs by mode" % suite,
                ", ".join("%s=%d" % (m, c) for m, c in counts.items()))
    # stability
    for m, lbl in MODES:
        for suite in list(suites) + ["norm_quantize", "other"]:
            for r in ks["modes"].get(m, {}).get(suite, []):
                st = r.get("stats")
                if not st:
                    continue
                if st["cv_pct"] > 3.0:
                    add("S1", "med", "NOISY %s %s `%s`" % (lbl, suite, r["entry"][:38]),
                        "CV %.2f%%, spread %.2f%%, mean %.1f µs, shape %s"
                        % (st["cv_pct"], st["spread_pct"], st["mean"],
                           r["arg_shapes"][0] if r["arg_shapes"] else "-"))
                if st["median"] and abs(st["mean"] / st["median"] - 1) > 0.03:
                    add("S2", "med", "skewed %s %s `%s`" % (lbl, suite, r["entry"][:38]),
                        "mean %.1f vs median %.1f (%+.1f%%), samples %s"
                        % (st["mean"], st["median"], 100 * (st["mean"] / st["median"] - 1),
                           st["samples"]))
                if st.get("within_round_cv_pct", 0) > 5 * max(st["cv_pct"], 0.01):
                    add("S3", "low", "jittery-in-round %s %s `%s`"
                        % (lbl, suite, r["entry"][:38]),
                        "within-round CV %.2f%% vs between-round %.2f%%"
                        % (st["within_round_cv_pct"], st["cv_pct"]))


def audit_layers(lay):
    for m, lbl in MODES:
        for e in lay["modes"][m]:
            gbf = e.get("gpu_busy_frac")
            if gbf is not None and (gbf < 0.95 or gbf > 1.05):
                add("C1", "med", "%s %s C%d/%d² busy-frac %.3f"
                    % (lbl, e["kind"], e["x_shape"][1], e["x_shape"][2], gbf),
                    "summed kernel time %.1f vs pipeline %.1f µs -> %s"
                    % (e["gpu_us_sum"], e["pipeline_us"],
                       "launch-bound gap" if gbf < 1 else "kernel sum exceeds wall"))
            st = e.get("stats")
            if st and st["cv_pct"] > 3.0:
                add("S1", "med", "NOISY layer %s %s C%d/%d²"
                    % (lbl, e["kind"], e["x_shape"][1], e["x_shape"][2]),
                    "CV %.2f%%, spread %.2f%%, samples %s"
                    % (st["cv_pct"], st["spread_pct"], st["samples"]))
    # INT4 vs INT8 per layer
    idx = {}
    for m, _ in MODES:
        for e in lay["modes"][m]:
            idx.setdefault((e["kind"], tuple(e["x_shape"])), {})[m] = e
    for k, v in idx.items():
        if len(v) != len(MODES):
            continue
        r = ratio_ci(v["int8_baseline"]["stats"], v["int4_baseline"]["stats"])
        if r and r["ci95_hi"] < 1.0:
            add("R2", "med", "layer INT4 slower than INT8: %s C%d/%d²"
                % (k[0], k[1][1], k[1][2]),
                "INT4 %.1f vs INT8 %.1f µs (%.3f×, CI %.3f-%.3f)"
                % (v["int4_baseline"]["stats"]["mean"], v["int8_baseline"]["stats"]["mean"],
                   r["ratio"], r["ci95_lo"], r["ci95_hi"]))
        f = v["fp16"]["stats"]
        for m, lbl in MODES[1:]:
            rr = ratio_ci(f, v[m]["stats"])
            if rr and rr["ci95_hi"] < 1.0:
                add("R1", "high", "layer %s slower than FP16: %s C%d/%d²"
                    % (lbl, k[0], k[1][1], k[1][2]),
                    "%.3f× (CI %.3f-%.3f)" % (rr["ratio"], rr["ci95_lo"], rr["ci95_hi"]))


def audit_e2e(e2e):
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ck_stages import STAGES, split
    for m, lbl in MODES:
        d = e2e["modes"][m]
        st = d.get("stats")
        if st:
            s = st["samples"]
            inc = sum(1 for i in range(len(s) - 1) if s[i + 1] > s[i])
            if len(s) > 3 and (inc >= len(s) - 2 or inc <= 1):
                add("D1", "high", "%s e2e repeats trend monotonically" % lbl,
                    "%d of %d consecutive pairs increase; %s -> the t-interval "
                    "(±%.1f ms) assumes independence the samples do not have"
                    % (inc, len(s) - 1, [round(x / 1e3) for x in s], st["ci95_half"] / 1e3))
        S = split(d["kernels"])
        tot = sum(S.values())
        wall = d["wall_us_per_batch"]
        if abs(tot / wall - 1) > 0.005:
            add("C2", "high", "%s stage sum != wall" % lbl,
                "stages %.1f vs wall %.1f ms (%+.2f%%)"
                % (tot / 1e3, wall / 1e3, 100 * (tot / wall - 1)))
        misc = S.get("misc", 0.0) / wall * 100
        if misc > 12:
            add("I1", "low", "%s unclassified 'other' stage is large" % lbl,
                "%.1f%% of wall lands in the catch-all bucket" % misc)


def audit_gn_bandwidth(lay):
    """The GN+quantize kernel's implied bandwidth per call site; a wide spread means the slow
    sites are not at a hardware limit, and anything above peak means the accounting is wrong."""
    rows = []
    for e in lay["modes"]["int4_baseline"]:
        n, c, h, w = e["x_shape"]
        for k in e.get("kernels", []):
            if "group_norm_silu_quantize_pack" not in k["kernel"]:
                continue
            per = k["us_per_layer_call"] / max(k["calls"], 1e-9)
            gbs = n * c * h * w * 4.5 / 1e9 / (per / 1e6)
            rows.append((gbs, e["kind"], c, h, per))
    if not rows:
        return
    rows.sort()
    lo, hi = rows[0], rows[-1]
    add("B1", "high" if hi[0] > A40_PEAK_GBS else "med",
        "GN+quantize implied bandwidth spans %.0f-%.0f GB/s (%.0f%%-%.0f%% of peak)"
        % (lo[0], hi[0], lo[0] / A40_PEAK_GBS * 100, hi[0] / A40_PEAK_GBS * 100),
        "slowest %s C%d/%d² at %.1f µs/firing; fastest %s C%d/%d² at %.1f. "
        "Same kernel, so the slow sites are not at a hardware limit%s"
        % (lo[1], lo[2], lo[3], lo[4], hi[1], hi[2], hi[3], hi[4],
           " -- and >peak means the 4.5 B/elem model is wrong" if hi[0] > A40_PEAK_GBS else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--e2e", required=True)
    a = ap.parse_args()
    ks = json.load(open(rel(a.kernels)))
    lay = json.load(open(rel(a.layers)))
    e2e = json.load(open(rel(a.e2e)))

    audit_kernels(ks)
    audit_layers(lay)
    audit_e2e(e2e)
    audit_gn_bandwidth(lay)

    order = {"high": 0, "med": 1, "low": 2}
    FINDINGS.sort(key=lambda f: (order[f[1]], f[0]))
    by_code = collections.Counter(f[0] for f in FINDINGS)
    print("%d findings: %s\n" % (len(FINDINGS),
                                 ", ".join("%s=%d" % kv for kv in sorted(by_code.items()))))
    seen = collections.Counter()
    for code, sev, what, detail in FINDINGS:
        seen[code] += 1
        if seen[code] > 8:
            continue
        print("[%s/%-4s] %s\n          %s" % (code, sev, what, detail))
    for code, n in sorted(by_code.items()):
        if n > 8:
            print("... %s: %d more suppressed" % (code, n - 8))


if __name__ == "__main__":
    main()
