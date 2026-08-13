"""Emit docs/bench_report_2026-08-13/REPORT.md -- one file, all four measurements, data + figures.

GENERATED, not hand-written: every number comes from the JSON the measurement steps wrote, so the
report cannot drift from the data and no value is transcribed by hand. Re-run after any re-measurement.

DATA ONLY. The request was explicit that no analysis is needed, so this emits tables and figures with
provenance and nothing else -- no interpretation, no recommendations, no "therefore".

Run: python docs/bench_report_2026-08-13/scripts/make_report.py    # no GPU
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

# TWO MODULES NAMED make_plots.py -- this report's and docs/state_report_2026-08-12's, which owns the
# palette and bucket_kernel. Putting this directory on sys.path made `import make_plots` resolve to the
# local one, whose own `from make_plots import ...` then resolved to ITSELF: a circular import. Both are
# loaded by explicit file path under distinct module names so neither can shadow the other.
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_state_plots = _load("modiff_state_plots", "docs/state_report_2026-08-12/scripts/make_plots.py")
bucket_kernel, BUCKETS = _state_plots.bucket_kernel, _state_plots.BUCKETS
_bench_plots = _load("modiff_bench_plots", "docs/bench_report_2026-08-13/scripts/make_plots.py")
suite_totals = _bench_plots.suite_totals

D = "docs/bench_report_2026-08-13"
MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]
SUITES = [("attention", "2. Attention"), ("conv", "3. Conv"), ("linear", "4. Linear")]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, n=2):
    return "—" if v is None else f"{v:.{n}f}"


def sec_e2e(o, e2e):
    o.append("## 1. End-to-end latency\n")
    if not e2e:
        o.append("_data/e2e.json missing._\n")
        return
    o.append(f"`{e2e.get('gpu','?')}`, batch {e2e.get('batch')}, DDIM {e2e.get('steps')} steps, "
             f"{e2e.get('repeats')} timed repeats after 2 warm-up samples.\n")
    o.append("| mode | ms/batch | ms/sample | ms/step | vs fp16 | CV | spread |")
    o.append("|---|--:|--:|--:|--:|--:|--:|")
    for m, lab in MODES:
        d = (e2e.get("modes") or {}).get(m)
        if not d:
            continue
        sp = d.get("speedup_vs_fp16")
        o.append(f"| {lab} | {f(d['wall_us_per_batch']/1e3,1)} | {f(d['per_sample_ms'],3)} | "
                 f"**{f(d['per_step_ms'])}** | {('1.000×' if m=='fp16' else f(sp,3)+'×') if sp else '—'} | "
                 f"{f(d.get('wall_cv_pct'))}% | {f(d.get('wall_spread_pct'))}% |")
    o.append("")
    o.append("![e2e](plots/01_e2e.png)\n")

    # kernel-bucket composition, from the same profiled window
    agg = {}
    for m, lab in MODES:
        d = (e2e.get("modes") or {}).get(m)
        if not d or not d.get("kernels"):
            continue
        a = {}
        for r in d["kernels"]:
            a[bucket_kernel(r["kernel"])] = a.get(bucket_kernel(r["kernel"]), 0.0) + r["us"] / 1e3
        agg[lab] = a
    if agg:
        labs = [l for _, l in MODES if l in agg]
        o.append("### 1a. GPU time by kernel bucket (ms of the profiled window)\n")
        o.append("| bucket | " + " | ".join(labs) + " |")
        o.append("|---" + "|--:" * len(labs) + "|")
        for b in BUCKETS:
            if any(b in agg[l] for l in labs):
                o.append(f"| {b} | " + " | ".join(f(agg[l].get(b, 0.0), 0) for l in labs) + " |")
        o.append("| **total** | " + " | ".join(f"**{f(sum(agg[l].values()),0)}**" for l in labs) + " |")
        o.append("")
        o.append("### 1b. Top kernels per mode\n")
        for m, lab in MODES:
            d = (e2e.get("modes") or {}).get(m)
            if not d or not d.get("kernels"):
                continue
            o.append(f"**{lab}**\n")
            o.append("| ms | % | calls | kernel |")
            o.append("|--:|--:|--:|---|")
            for r in d["kernels"][:8]:
                o.append(f"| {f(r['us']/1e3,0)} | {f(r['pct'],1)} | {r['calls']} | "
                         f"`{r['kernel'][:72]}` |")
            o.append("")


def sec_blocks(o, rows):
    o.append("## 1c. Per-block attribution\n")
    if not rows:
        o.append("_data/profile_layers.json missing._\n")
        return
    o.append("Per-configuration wall time, and the share attributed to quantized layers grouped by "
             "block kind. Same batch and step count as section 1.\n")
    o.append("These are `profile_layers_and_model.py`'s OWN eight configurations, not the five modes of "
             "section 1: it sweeps what is quantized (conv only, conv+proj, the projection refresh "
             "period K, route B) rather than sweeping precision alone. `wall ms/step` is therefore "
             "comparable within this table but only the `fp16` row is directly comparable to "
             "section 1.\n")
    kinds = []
    for r in rows:
        for k in (r.get("kinds") or {}):
            if k not in kinds:
                kinds.append(k)
    o.append("| config | wall ms/step | " + " | ".join(kinds) + " | attributed |")
    o.append("|---|--:|" + "--:|" * len(kinds) + "--:|")
    for r in rows:
        ks = r.get("kinds") or {}
        tot = sum(ks.values())
        o.append(f"| {r.get('config')} | {f(r.get('wall_ms_per_step'))} | "
                 + " | ".join(f(ks.get(k, 0.0)) for k in kinds) + f" | {f(tot)} |")
    o.append("")
    o.append("![blocks](plots/02_blocks.png)\n")
    # per-layer detail, the heaviest few per config
    o.append("### 1d. Heaviest quantized layers (ms/step)\n")
    o.append("Entries whose name matches a block KIND (e.g. `updown`) are aggregates the harness "
             "reports as one row, not single layers; they are marked.\n")
    for r in rows:
        L = r.get("layers") or {}
        if not L:
            continue
        kindnames = set(r.get("kinds") or {})
        top = sorted(L.items(), key=lambda kv: -kv[1])[:8]
        o.append(f"**{r.get('config')}** — {len(L)} entries\n")
        o.append("| ms/step | layer |")
        o.append("|--:|---|")
        for n, v in top:
            tag = " _(aggregate)_" if n in kindnames else ""
            o.append(f"| {f(v,3)} | `{n}`{tag} |")
        o.append("")


def sec_suite(o, ks, suite, title):
    o.append(f"## {title} kernels\n")
    if not ks:
        o.append("_data/kernel_suites.json missing._\n")
        return
    o.append("Real call arguments captured at the C++ entry point during a live sample, then replayed "
             "in isolation. `ms/sample` is the median replay time × `calls_per_sample`, summed over "
             "call signatures.\n")
    tot, tops = suite_totals(ks, suite)
    labs = [l for _, l in MODES if l in tot]
    o.append("| mode | ms/sample | signatures |")
    o.append("|---|--:|--:|")
    for m, lab in MODES:
        if lab not in tot:
            continue
        n = len(((ks.get("modes") or {}).get(m) or {}).get(suite) or [])
        o.append(f"| {lab} | **{f(tot[lab],3)}** | {n} |")
    o.append("")
    o.append(f"![{suite}](plots/0{3 + [s for s, _ in SUITES].index(suite)}_{suite}.png)\n")
    o.append(f"### Entry points by cost\n")
    o.append("| mode | ms/sample | entry point |")
    o.append("|---|--:|---|")
    for m, lab in MODES:
        if lab not in tops:
            continue
        for e, us in tops[lab][:5]:
            o.append(f"| {lab} | {f(us/1e3,3)} | `{e}` |")
    o.append("")
    # per-signature detail
    o.append("### Per-signature detail\n")
    for m, lab in MODES:
        rows = ((ks.get("modes") or {}).get(m) or {}).get(suite) or []
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: -(float((r.get("stats") or {}).get("median") or 0)
                                            * float(r.get("calls_per_sample") or 0)))[:6]
        o.append(f"**{lab}**\n")
        o.append("| ms/sample | median µs | CV | calls | entry | shapes |")
        o.append("|--:|--:|--:|--:|---|---|")
        for r in rows:
            st = r.get("stats") or {}
            med = float(st.get("median") or 0.0)
            n = float(r.get("calls_per_sample") or 0.0)
            sh = str(r.get("arg_shapes"))[:60]
            o.append(f"| {f(med*n/1e3,3)} | {f(med,1)} | {f(st.get('cv_pct'))}% | {n:.0f} | "
                     f"`{r.get('entry')}` | `{sh}` |")
        o.append("")


def main():
    e2e = load(f"{D}/data/e2e.json")
    blocks = load(f"{D}/data/profile_layers.json")
    ks = load(f"{D}/data/kernel_suites.json")

    o = []
    o.append("# MoDiff benchmark and profile — 2026-08-13\n")
    o.append("End-to-end latency, per-block attribution, and per-kernel benchmarks for the attention, "
             "conv and linear suites. **Data only — no analysis.**\n")
    o.append("## Configuration measured\n")
    o.append("| | |")
    o.append("|---|---|")
    o.append(f"| GPU | {(e2e or {}).get('gpu', '?')} |")
    o.append(f"| batch / steps | {(e2e or {}).get('batch','?')} / {(e2e or {}).get('steps','?')} "
             f"(DDIM) |")
    o.append("| checkpoint | `models/ldm/lsun_churches256/model.ckpt` (real, 2.7 GB) |")
    o.append("| calibration | resolved through `CALIBRATION_PREFERENCE` / "
             "`DELTA_CALIBRATION_PREFERENCE` — no hardcoded paths |")
    o.append("| `MODIFF_CAT2_FOLD` | 1 (decoder skip-concat folded into the GN prologue) |")
    o.append("| `MODIFF_LINEAR` | 0 (attention projections quantized, not modulated) |")
    o.append("| `MODIFF_DELTA_MODE` | static (per-step delta table) |")
    o.append("| activation zero point | 0 everywhere (`MODIFF_ZP_STRICT=1`) |")
    o.append("")
    o.append("Modes: `fp16`, `W8A8 PTQ` (`int8_baseline`), `W8A8 MoDiff` (`int8`), "
             "`W4A4 PTQ` (`int4_baseline`), `W4A4 MoDiff` (`int4`).\n")

    sec_e2e(o, e2e)
    sec_blocks(o, blocks)
    for suite, title in SUITES:
        sec_suite(o, ks, suite, title)

    o.append("## Reproducing\n")
    o.append("```bash")
    o.append("bash docs/bench_report_2026-08-13/scripts/run_all.sh   # all four measurements + plots")
    o.append("python docs/bench_report_2026-08-13/scripts/make_report.py   # regenerate this file")
    o.append("```")
    out = f"{D}/REPORT.md"
    open(out, "w").write("\n".join(o) + "\n")
    print(f"wrote {out} ({len(o)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
