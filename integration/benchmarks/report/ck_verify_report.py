"""Assert that the figures quoted in a checkpoint report match the measured JSON.

The 07-31 report accumulated five stale figures because its tables were transcribed by hand and
nothing re-checked them after the data changed. This greps the report text for the numbers that
matter and fails if any of them is not present, so a data refresh that is not carried into the
prose is a hard error rather than a silent inconsistency.

Usage:
  python3 ck_verify_report.py ../../CHECKPOINT_REPORT_2026-08-01.md \
      --e2e data/e2e_three_mode_2026-08-01.json --layers data/layers_2026-08-01.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ck_stages import STAGES, split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODES = ["fp16", "int8_baseline", "int4_baseline"]
KINDS = ["attention", "resblock_plain", "resblock_updown"]


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--layers", required=True)
    a = ap.parse_args()

    text = open(rel(a.report)).read()
    e2e = json.load(open(rel(a.e2e)))
    lay = json.load(open(rel(a.layers)))

    # (label, [acceptable strings]) -- a report may quote either central statistic, since both
    # the mean of the round medians and the median are in the JSON and they differ by <0.4%.
    # An earlier version demanded the median only and failed a report that consistently used
    # means, which is the statistic a confidence interval actually attaches to.
    checks = []

    # ---- e2e headline
    fp = e2e["modes"]["fp16"]["wall_us_per_batch"]
    fp_mean = (e2e["modes"]["fp16"].get("stats") or {}).get("mean", fp)
    for m, lbl in zip(MODES, ("FP16", "INT8", "INT4")):
        v = e2e["modes"][m]
        st = v.get("stats") or {}
        mean = st.get("mean", v["wall_us_per_batch"])
        checks.append((f"e2e {lbl} ms/batch",
                       [f"{v['wall_us_per_batch']/1e3:.1f}", f"{mean/1e3:.1f}"]))
        checks.append((f"e2e {lbl} ms/sample",
                       [f"{v['per_sample_ms']:.3f}", f"{mean/1e3/e2e['batch']:.3f}"]))
        checks.append((f"e2e {lbl} ms/step",
                       [f"{v['per_step_ms']:.2f}", f"{mean/1e3/e2e['steps']:.2f}"]))
        checks.append((f"e2e {lbl} speedup",
                       [f"{fp/v['wall_us_per_batch']:.3f}×", f"{fp_mean/mean:.3f}×"]))
        checks.append((f"e2e {lbl} CV",
                       [f"{v['wall_cv_pct']:.2f}%", f"{st.get('cv_pct', -1):.2f}%"]))

    # ---- stage table
    S = {m: split(e2e["modes"][m]["kernels"]) for m in MODES}
    for key, lbl, _, _ in STAGES:
        if max(S[m][key] for m in MODES) < 1000:      # sub-1ms rows print as 0.0
            continue
        for m in MODES:
            checks.append((f"stage {lbl} {m}", [f"{S[m][key]/1e3:.1f}"]))

    # ---- layer table
    for m, lbl in zip(MODES, ("FP16", "INT8", "INT4")):
        agg = {k: 0.0 for k in KINDS}
        for e in lay["modes"][m]:
            agg[e["kind"]] += e["pipeline_us"] * e["n_instances"] / 1e3
        tot = sum(agg.values())
        agg_mean = {k: 0.0 for k in KINDS}
        for e in lay["modes"][m]:
            if e.get("stats"):
                agg_mean[e["kind"]] += e["stats"]["mean"] * e["n_instances"] / 1e3
        for k in KINDS:
            checks.append((f"layer {lbl} {k}",
                           [f"{agg[k]:.2f} ms", f"{agg_mean[k]:.2f} ms"]))
        checks.append((f"layer {lbl} total",
                       [f"{tot:.2f} ms", f"{sum(agg_mean.values()):.2f} ms"]))

    # ---- attention per shape
    for m in MODES:
        for e in lay["modes"][m]:
            if e["kind"] == "attention":
                alts = [f"{e['pipeline_us']:.1f}"]
                if e.get("stats"):
                    alts.append(f"{e['stats']['mean']:.1f}")
                checks.append((f"attn {m} C{e['x_shape'][1]}/T"
                               f"{e['x_shape'][2]*e['x_shape'][3]}", alts))

    # ---- nothing below 1.0x must be claimed only if true
    below = [(m, e["kind"], e["x_shape"])
             for m in MODES[1:] for e in lay["modes"][m]
             if e["fp16_us"] / e["pipeline_us"] < 1.0]

    missing = [(lbl, alts) for lbl, alts in checks
               if not any(a in text for a in (alts if isinstance(alts, list) else [alts]))]
    print(f"{len(checks)} figures checked against {os.path.basename(a.e2e)} + "
          f"{os.path.basename(a.layers)}")
    if missing:
        print(f"\nFAIL: {len(missing)} figure(s) quoted nowhere in the report:")
        for lbl, alts in missing:
            print(f"  {lbl:44s} expected one of {alts!r}")
    if below:
        print(f"\nNOTE: {len(below)} layer(s) genuinely below 1.0x vs FP16: {below}")
        if "no red cells" in text or "No layer is slower" in text:
            print("  FAIL: the report claims no layer is slower than FP16.")
            missing.append(("sub-FP16 claim", "consistency"))
    else:
        print("0 layers below 1.0x vs FP16 -- the report's 'no red cells' claim holds.")

    print("\nOK -- every quoted figure matches the data." if not missing else "\nMISMATCHES ABOVE.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
