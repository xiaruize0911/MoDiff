"""Assemble MEASUREMENT_REPORT_MODIFF_2026-08-04.md -- the MoDiff counterpart of the 08-01 report.

Same suites, same statistics, same table shapes and the same "no analysis" rule as
docs/MEASUREMENT_REPORT_2026-08-01.md. Two deliberate differences:

  * FIVE modes instead of three. 08-01 carried fp16 + the two *_baseline modes, i.e. MoDiff
    DISABLED, so a MoDiff number there had nothing to be divided by. Here every MoDiff row has two
    references: fp16 and the same bit width with MoDiff off.
  * The three nsys-derived sections (in-context attention per shape, GPU idle on the timeline,
    share of end-to-end time by layer type) are ABSENT, not empty: Nsight Systems is not installed
    in this environment (`which nsys` finds nothing; only nsight-compute is present, and its
    counters are blocked by RmProfilingAdminOnly=1). Stated rather than silently dropped.

Why this is a separate generator instead of a flag on make_measurement_report.py: that script and
the two table scripts it shells out to (ck_final_numbers, ck_report_numbers) hardcode a three-mode
structure with pairwise deltas -- `S["int8_baseline"]`, `tots['int4_baseline'] - tots['int8_baseline']`,
a two-entry quant_ratios dict. Generalising them to five modes would rewrite their formatting and put
the 08-01 report's reproducibility at risk for no gain. The statistics themselves are shared: this
script reads the same JSON the same benches write, and buckets kernels through the same ck_stages.

No interpretation: nothing here ranks, recommends or concludes.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))

from ck_stages import STAGES, split as stage_split                       # noqa: E402
from profile_tree import short_kernel_name                               # noqa: E402

D = "docs/measurement_modiff_2026-08-04"
OUT = "docs/MEASUREMENT_REPORT_MODIFF_2026-08-04.md"

#: (json key, label, reference key). The reference is what the "vs own" column divides by.
MODES = [("fp16", "FP16", None),
         ("int8_baseline", "INT8 (MoDiff off)", None),
         ("int8", "INT8 + MoDiff", "int8_baseline"),
         ("int4_baseline", "INT4 (MoDiff off)", None),
         ("int4", "INT4 + MoDiff", "int4_baseline")]
#: layer_pipeline_bench writes its MoDiff modes under these keys instead.
LAYER_KEY = {"int8": "int8_modiff", "int4": "int4_modiff"}
MK = [m for m, _, _ in MODES]
LBL = {m: l for m, l, _ in MODES}
REF = {m: r for m, _, r in MODES}


#: baseline entry point -> the MoDiff entry point that does the same job at the same call site.
#: Declared, not inferred -- see cross_mode.pair() for what inferring it produced. Entries absent
#: from this map are only ever compared against themselves (torch_conv2d_fp16, the flash and
#: attention-quantize entries, which MoDiff does not modify).
COUNTERPART = {
    "conv2d_int8_evt_bias_residual_fp16": "conv2d_int8_evt_o_hat_residual",
    "conv2d_int4_evt_bias_residual_fp16": "conv2d_int4_evt_o_hat_residual",
    "conv2d_int8_fprop": "conv2d_int8_evt_o_hat",
    "conv2d_int4_fprop": "conv2d_int4_evt_o_hat",
    "group_norm_silu_quantize_nhwc": "group_norm_silu_delta_quantize_nhwc",
    "group_norm_silu_quantize_pack_nhwc": "group_norm_silu_delta_quantize_pack_nhwc",
    "group_norm_silu_quantize_resize_nhwc": "group_norm_silu_delta_quantize_resize_nhwc",
    "step1_static_quantize_fprop": "step1_static_quantize_fprop",
    "gemm_w8a8_awq_bias_res": "gemm_w8a8_awq_bias_res",
    "gemm_w4a4_awq_bias_res": "gemm_w4a4_awq_bias_res",
}


def jload(f):
    p = f if os.path.isabs(f) else os.path.join(ROOT, f)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"


# ---------------------------------------------------------------- end to end
def e2e_table(runs):
    """Three independent invocations, one row per mode. Mirrors 08-01's table plus a 'vs own' column."""
    out = ["| mode | " + " | ".join(f"run {i + 1}" for i in range(len(runs)))
           + " | mean ms/batch | cross-run CV | within-run CV | vs FP16 | vs own baseline |",
           "|---|" + "---:|" * (len(runs) + 5)]
    mean = {}
    for m in MK:
        v = [d["modes"][m]["stats"]["mean"] / 1e3 for d in runs if m in d["modes"]]
        if not v:
            continue
        mean[m] = sum(v) / len(v)
    for m in MK:
        if m not in mean:
            continue
        v = [d["modes"][m]["stats"]["mean"] / 1e3 for d in runs if m in d["modes"]]
        cvi = sum(d["modes"][m]["stats"]["cv_pct"] for d in runs if m in d["modes"]) / len(v)
        mu = mean[m]
        sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        own = "—" if not REF[m] or REF[m] not in mean else f"{mean[REF[m]] / mu:.3f}×"
        out.append("| %s | %s | %.1f | %.2f%% | %.2f%% | %s | %s |"
                   % (LBL[m], " | ".join(f"{x:.1f}" for x in v), mu, sd / mu * 100, cvi,
                      "1.000×" if m == "fp16" else f"{mean['fp16'] / mu:.3f}×", own))
    return "\n".join(out), mean


def stage_table(e2e):
    """Whole-model time by stage: profiler self-time scaled to the measured wall clock, ms/batch."""
    S, wall = {}, {}
    for m in MK:
        d = e2e["modes"].get(m)
        if not d:
            continue
        S[m] = stage_split(d["kernels"])
        wall[m] = d["wall_us_per_batch"] / 1e3
    present = [m for m in MK if m in S]
    out = ["| stage | " + " | ".join(LBL[m] for m in present) + " | i8 MoDiff − i8 base | i4 MoDiff − i4 base |",
           "|---|" + "---:|" * (len(present) + 2)]
    tot = {m: 0.0 for m in present}
    for key, lbl, _, _ in STAGES:
        row, vals = [], {}
        for m in present:
            # profiler_scale: the profiler total differs from the measured wall clock, so every
            # stage is scaled by wall / profiler-total. Same convention as the 08-01 report.
            sc = wall[m] / (sum(S[m].values()) / 1e3)
            vals[m] = S[m].get(key, 0.0) / 1e3 * sc
            tot[m] += vals[m]
            row.append(f"{vals[m]:.1f}")
        d8 = (vals.get("int8", 0.0) - vals.get("int8_baseline", 0.0)) if "int8" in vals else None
        d4 = (vals.get("int4", 0.0) - vals.get("int4_baseline", 0.0)) if "int4" in vals else None
        out.append("| %s | %s | %s | %s |" % (lbl, " | ".join(row),
                                              f"{d8:+.1f}" if d8 is not None else "—",
                                              f"{d4:+.1f}" if d4 is not None else "—"))
    # Same absent-mode guard as the per-stage rows. Without it a missing MoDiff mode makes the
    # delta read as minus the whole baseline total (-14133.2) instead of an em dash.
    t8 = (tot["int8"] - tot["int8_baseline"]) if {"int8", "int8_baseline"} <= set(tot) else None
    t4 = (tot["int4"] - tot["int4_baseline"]) if {"int4", "int4_baseline"} <= set(tot) else None
    out.append("| **total** | " + " | ".join(f"**{tot[m]:.1f}**" for m in present)
               + " | %s | %s |" % (f"**{t8:+.1f}**" if t8 is not None else "—",
                                   f"**{t4:+.1f}**" if t4 is not None else "—"))
    out.append("")
    for m in present:
        out.append("  check %s: stage sum %.1f vs measured wall %.1f (%+.2f%%)"
                   % (LBL[m], tot[m], wall[m], (tot[m] - wall[m]) / wall[m] * 100))
    return "\n".join(out)


def modiff_kernel_table(e2e):
    """The kernels that exist only on the MoDiff path, and the baseline kernel they replace.

    08-01 has no counterpart for this table because it measured MoDiff-off modes only.
    """
    tok = ("delta_quantize", "gn_apply_delta", "gn_stats_partials", "gn_stats_reduce",
           "delta_absmax", "evt_o_hat", "_o_hat")
    out = ["| kernel | INT8 base | INT8 + MoDiff | INT4 base | INT4 + MoDiff |",
           "|---|---:|---:|---:|---:|"]
    per = {}
    for m in ("int8_baseline", "int8", "int4_baseline", "int4"):
        d = e2e["modes"].get(m)
        if not d:
            continue
        for k in d["kernels"]:
            # ACCUMULATE: several template instantiations collapse to one shortened name
            # (group_norm_silu_quantize_nhwc_vec2_kernel<__half,false> and <__half,true> both do),
            # and assigning made the baseline read 411.1 ms where the bucket sums 3323.7.
            nm = short(k["kernel"])
            per.setdefault(nm, {})
            per[nm][m] = per[nm].get(m, 0.0) + k["us"] / 1e3
    rows = [(nm, v) for nm, v in per.items()
            if any(t in nm.lower() for t in tok)
            or any(t in nm.lower() for t in ("group_norm_silu_quantize", "implicitgemmconvolution"))]
    for nm, v in sorted(rows, key=lambda kv: -max(kv[1].values())):
        out.append("| `%s` | %s | %s | %s | %s |"
                   % (nm, fmt(v.get("int8_baseline")), fmt(v.get("int8")),
                      fmt(v.get("int4_baseline")), fmt(v.get("int4"))))
    return "\n".join(out)


def misc_audit(e2e, topn=8):
    """What actually landed in the 'elementwise / copies / other' catch-all, per mode.

    The stage table is only as good as ck_stages' name matching, and a MoDiff kernel that matches
    no token lands here silently -- which is exactly what happened to the whole MoDiff GN chain
    before the 2026-08-04 token additions. Printing the bucket's contents makes the omission visible
    instead of leaving it as an unexplained rise in 'other'.
    """
    from ck_stages import stage_of
    out = ["| mode | kernel | ms/batch | % of mode |", "|---|---|---:|---:|"]
    for m in MK:
        d = e2e["modes"].get(m)
        if not d:
            continue
        wall = d["wall_us_per_batch"] / 1e3
        agg = collections.defaultdict(float)
        for k in d["kernels"]:
            if stage_of(k["kernel"]) == "misc":
                agg[short(k["kernel"])] += k["us"] / 1e3
        for i, (nm, u) in enumerate(sorted(agg.items(), key=lambda kv: -kv[1])[:topn]):
            out.append("| %s | `%s` | %.1f | %.1f%% |"
                       % (LBL[m] if i == 0 else "", nm, u, u / wall * 100))
            # Build-time warning, not a doc row: anything this large in a catch-all is either a
            # real elementwise op or a name the bucketing does not know. Three MoDiff kernels
            # reached 0.7-1.0% here before their tokens were added, so the threshold is deliberately
            # low enough to surface the next one.
            if u / wall * 100 >= 0.5 and not any(
                    t in nm.lower() for t in ("elementwise", "copy", "cat2", "upsample",
                                              "avg_pool", "transpose", "permute")):
                print(f"  [misc-audit] {LBL[m]}: {nm} = {u:.1f} ms/batch "
                      f"({u / wall * 100:.1f}%) is in the catch-all and does not look elementwise")
    return "\n".join(out)


def short(name):
    """Kernel symbol -> readable name. Delegates to the shared shortener: a hand-rolled
    split-on-'(' leaves Itanium-mangled CUTLASS symbols (_ZN7cutlass6KernelIN6modiff26Implicit...)
    unreadable, and those are exactly the conv kernels this report is about."""
    return short_kernel_name(name)


# ---------------------------------------------------------------- kernel suites
def suite_tables(ks, suite):
    """Per-entry tables, one block per mode, then a cross-mode block. 08-01's layout exactly."""
    out = []
    for m in MK:
        d = (ks["modes"].get(m) or {}).get(suite)
        if not d:
            continue
        rows = sorted(d, key=lambda e: -(e["stats"]["mean"] if e.get("stats") else 0))
        out.append(f"**{LBL[m]}** — {len(rows)} signatures")
        out.append("| entry | signature | µs/call (mean ± 95% CI) | CV | spread | n | stability |")
        out.append("|---|---|---:|---:|---:|---:|---|")
        for e in rows:
            s = e.get("stats")
            if not s:
                continue
            out.append("| `%s` | %s | %.1f ± %.1f | %.2f%% | %.2f%% | %d | %s |"
                       % (e["entry"], sig_of(e), s["mean"], s["ci95_half"], s["cv_pct"],
                          s["spread_pct"], s["n"], e.get("stability", "")))
        out.append("")
    return "\n".join(out)


def sig_of(e):
    """A compact call signature, and the key the cross-mode join uses.

    The INPUT shape alone is not a key. At (128, 192, 32, 32) the int8 MoDiff conv suite holds
    conv2d_int8_fprop, conv2d_int8_evt_o_hat and conv2d_int8_evt_o_hat_residual, and the same entry
    appears twice with different weight shapes -- so an input-shape key silently compares a 3x3
    against a 1x1, or a residual variant against one without. Conv and linear are therefore keyed on
    (input, weight); attention on T/hd, where the padding of hd 24 -> 32 makes the raw shape a worse
    key than the logical one.
    """
    sh = e.get("arg_shapes") or []
    if not sh:
        return "—"
    a = sh[0]
    if len(a) == 4 and e["entry"].startswith("flash"):
        return f"T={a[2]} hd={a[3]}"
    if len(sh) > 1 and isinstance(sh[1], list) and len(sh[1]) == 4:
        return "(%s) w(%s)" % (", ".join(str(x) for x in a), ", ".join(str(x) for x in sh[1]))
    return "(" + ", ".join(str(x) for x in a) + ")"


def cross_mode(ks, suite):
    """MoDiff vs its own baseline, per call signature -- the comparison 08-01 could not make.

    Keyed on (entry, arg shapes). A MoDiff entry name often DIFFERS from its baseline counterpart
    (conv2d_int8_evt_o_hat vs conv2d_int8_evt_bias_residual_fp16), so an entry-name join would drop
    exactly the rows of interest. Keyed on the shape alone, aggregating every entry at that shape.
    """
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in MK:
        for e in (ks["modes"].get(m) or {}).get(suite) or []:
            if e.get("stats"):
                idx[sig_of(e)][m].append(e)

    def pair(v, base_m, mod_m):
        """The (baseline, MoDiff) entries that do the SAME job at this signature, or None.

        Picking each mode's busiest entry does not work: at one conv signature the MoDiff suite
        holds conv2d_int8_fprop, _evt_o_hat and _evt_o_hat_residual, and the busiest happened to be
        the plain fprop -- so the row compared a bias+residual EVT conv against a conv with neither
        and called the result 0.951x. In norm_quantize it paired group_norm_silu_quantize_nhwc
        against scale_quantize_int8, which are different operations entirely. So correspondence is
        declared, never inferred, and a signature with no declared counterpart gets no ratio.
        """
        for b in v.get(base_m, []):
            for d in v.get(mod_m, []):
                if d["entry"] == b["entry"] or COUNTERPART.get(b["entry"]) == d["entry"]:
                    return b, d
        return None

    out = ["| signature | baseline entry | µs | MoDiff entry | µs | W8A8 ratio | W4A4 ratio |",
           "|---|---|---:|---|---:|---:|---:|"]
    rows, unmatched, dtype_rows = [], collections.Counter(), []
    for k, v in idx.items():
        p8 = pair(v, "int8_baseline", "int8")
        p4 = pair(v, "int4_baseline", "int4")
        if not p8 and not p4:
            for m in ("int8", "int4"):
                for e in v.get(m, []):
                    unmatched[e["entry"]] += 1
            continue
        b, d = p8 or p4
        r8 = (p8[0]["stats"]["mean"] / p8[1]["stats"]["mean"]) if p8 else None
        r4 = (p4[0]["stats"]["mean"] / p4[1]["stats"]["mean"]) if p4 else None
        # A dtype that differs between the two modes means the two captures came from different
        # points in the run -- verified for the decoder skip convs, whose fp32 input occurs on
        # DDIM step 1 only. Such a row is a first-step comparison, not a steady-state one.
        dt = (b.get("arg_dtypes") or [None])[0] != (d.get("arg_dtypes") or [None])[0]
        rows.append((max(b["stats"]["mean"], d["stats"]["mean"]),
                     "| %s%s | `%s` | %.1f | `%s` | %.1f | %s | %s |"
                     % (k, " †" if dt else "", b["entry"], b["stats"]["mean"], d["entry"],
                        d["stats"]["mean"], f"{r8:.3f}×" if r8 else "—",
                        f"{r4:.3f}×" if r4 else "—")))
        if dt:
            dtype_rows.append((suite, k, (b.get("arg_dtypes") or [None])[0],
                               (d.get("arg_dtypes") or [None])[0]))
    out += [r for _, r in sorted(rows, key=lambda x: -x[0])]
    if dtype_rows:
        out += ["", "† the two modes' captures record different INPUT DTYPES at this signature, so "
                    "the row compares calls taken from different points in the run and its ratio is "
                    "not a steady-state figure: "
                + "; ".join(f"{k} ({a} vs {b})" for _, k, a, b in dtype_rows)]
    if unmatched:
        out += ["", "MoDiff entries at signatures with no declared baseline counterpart, so no ratio "
                    "is given: " + ", ".join(f"`{k}` (x{v})" for k, v in unmatched.most_common())]
    return "\n".join(out) if rows else \
        "_no signature has a declared baseline/MoDiff counterpart in this suite._"


# ---------------------------------------------------------------- layers
def layer_table(lay):
    """Per (kind, shape) x mode, with the MoDiff/own-baseline ratio."""
    idx = collections.defaultdict(dict)
    for m in MK:
        for e in lay["modes"].get(LAYER_KEY.get(m, m)) or []:
            xs = e["x_shape"]
            idx[(e["kind"], f"C{xs[1]}/{xs[2]}²")][m] = e
    out = ["| kind | shape | n | " + " | ".join(f"{LBL[m]} µs" for m in MK)
           + " | i8 MoD / i8 base | i4 MoD / i4 base |",
           "|---|---|---:|" + "---:|" * (len(MK) + 2)]
    for (kind, shape), v in sorted(idx.items(),
                                   key=lambda kv: -max(e["pipeline_us"] for e in kv[1].values())):
        cells = [fmt(v[m]["pipeline_us"]) if m in v else "—" for m in MK]
        r8 = (v["int8_baseline"]["pipeline_us"] / v["int8"]["pipeline_us"]
              if {"int8_baseline", "int8"} <= set(v) else None)
        r4 = (v["int4_baseline"]["pipeline_us"] / v["int4"]["pipeline_us"]
              if {"int4_baseline", "int4"} <= set(v) else None)
        n = next(iter(v.values())).get("n_instances", "")
        out.append("| %s | %s | %s | %s | %s | %s |"
                   % (kind, shape, n, " | ".join(cells),
                      f"{r8:.3f}×" if r8 else "—", f"{r4:.3f}×" if r4 else "—"))
    return "\n".join(out)


def layer_stability(lay):
    out = ["| mode | layers | tight | ok | NOISY | median CV | worst CV |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MK:
        rows = lay["modes"].get(LAYER_KEY.get(m, m)) or []
        cv = sorted(e["stats"]["cv_pct"] for e in rows if e.get("stats"))
        if not cv:
            continue
        c = collections.Counter(e.get("stability") for e in rows)
        out.append("| %s | %d | %d | %d | %d | %.2f%% | %.2f%% |"
                   % (LBL[m], len(rows), c.get("tight", 0), c.get("ok", 0), c.get("NOISY", 0),
                      cv[len(cv) // 2], cv[-1]))
    return "\n".join(out)


def attention_profile(lay):
    """Per-kernel profile inside each attention shape, one block per shape. 08-01 §'Attention layer'."""
    by = collections.defaultdict(dict)
    for m in MK:
        for e in lay["modes"].get(LAYER_KEY.get(m, m)) or []:
            if e["kind"] != "attention":
                continue
            xs = e["x_shape"]
            by[(xs[1], xs[2] * xs[3], e.get("n_instances", 1))][m] = e
    out = []
    for (c, t, n), v in sorted(by.items(), key=lambda kv: -kv[0][1]):
        out.append(f"#### C{c} / T={t}   x{n} instances\n")
        out.append("| kernel | " + " | ".join(f"{LBL[m]} µs" for m in MK if m in v) + " |")
        out.append("|---|" + "---:|" * len([m for m in MK if m in v]))
        names = []
        for m in MK:
            for k in (v[m]["kernels"] if m in v else []):
                if short(k["kernel"]) not in names:
                    names.append(short(k["kernel"]))
        tot = {m: 0.0 for m in v}
        for nm in names:
            cells = []
            for m in MK:
                if m not in v:
                    continue
                u = sum(k["us_per_layer_call"] for k in v[m]["kernels"] if short(k["kernel"]) == nm)
                tot[m] += u
                cells.append(fmt(u) if u else "—")
            out.append("| `%s` | %s |" % (nm, " | ".join(cells)))
        out.append("| **kernel total** | %s |"
                   % " | ".join(f"**{tot[m]:.1f}**" for m in MK if m in v))
        out.append("| **measured latency** | %s |"
                   % " | ".join(f"**{v[m]['pipeline_us']:.1f}**" for m in MK if m in v))
        out.append("| gpu_busy_frac | %s |"
                   % " | ".join(f"{v[m].get('gpu_busy_frac', 0):.3f}" for m in MK if m in v))
        out.append("")
    return "\n".join(out)


GLOSSARY = """## Glossary

Only the terms this report's headings use. Identical in meaning to the 08-01 report's glossary;
repeated here so the document stands alone.

**mean of the round medians.** Every timing does `warmup`, then `rounds` x `iters` timed calls.
Inside a round the median of the `iters` samples is taken, which drops a single scheduler hiccup;
the reported value is the mean of those round medians.

**CV (coefficient of variation).** stdev / mean as a percent, over the round medians — it measures
whether re-running gives the same number, not how much individual calls jitter.

**95% CI.** A Student-t interval on the mean of the round medians.

**stability.** A label on CV: `tight` ≤ 1%, `ok` ≤ 3%, `NOISY` > 3%.

**vs own baseline.** A MoDiff row divided by the SAME bit width with MoDiff off. This is the column
the 08-01 report could not have: it measured the MoDiff-off modes only, so it had no MoDiff row to
divide. Above 1 the MoDiff configuration is faster than its own baseline; below 1 it is slower.

**profiler self-time, and scaling.** The profiler reports each kernel's own GPU time. Its total
differs from the measured wall clock, so stage tables are scaled by `profiler_scale` = wall /
profiler total. Every timing here is measured once WITHOUT a profiler and attributed in a separate
profiled run.

**gpu_busy_frac.** Summed kernel self-time divided by the measured wall time of the same layer.
Below 1 the GPU was idle inside the measured interval.

**entry point vs kernel.** An entry point is the C++ function Python calls, e.g.
`conv2d_int8_evt_o_hat`. A kernel is what runs on the GPU, e.g. `ImplicitGemmConvolutionEVT`.

**â / ô.** MoDiff's two cache tensors. `â_t = Q(a_t − â_{t+1}) + â_{t+1}` is the reconstructed
activation; `ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}` is the accumulated linear-operator output
(Eqs 8–10 of arXiv 2506.22463). Both are full-precision and both are read and written every step.

**delta scale.** The quantizer step used on `a_t − â_{t+1}`. `MODIFF_DELTA_MODE=dynamic` recomputes
it on device as `Q / max|delta|`; `MODIFF_DELTA_REFRESH=4` runs that reduction every 4th step.

**T, C, hd.** T is tokens per attention block (H×W), C is channels, hd is head dimension. The
quantized kernels pad hd from 24 to 32 at T=1024.

**packed int4.** Two 4-bit values share one int8 byte, so a packed channel dimension is half the
logical one.
"""


def main():
    runs = [jload(f"{D}/data/e2e_modiff_b128_run{i}.json") for i in (1, 2, 3)]
    runs = [r for r in runs if r]
    e2e = jload(f"{D}/data/e2e_suite_b128.json") or (runs[0] if runs else None)
    ks = jload(f"{D}/data/kernel_suites_b128.json")
    lay = jload(f"{D}/data/layers_b128.json")
    missing = [n for n, v in (("e2e runs", runs), ("kernel suites", ks), ("layers", lay)) if not v]
    if missing:
        raise SystemExit("missing input data: " + ", ".join(missing))

    e2e_tab, means = e2e_table(runs)
    cfg = e2e["modes"]["int8"].get("route_check", {}) if "int8" in e2e["modes"] else {}

    doc = f"""# Measurement report — MoDiff (W8A8 / W4A4), against fp16 and against its own baseline

**Hardware** {ks.get('gpu', 'NVIDIA A40')} · **Model** LSUN-churches LDM, 21 AttentionBlocks + 35 ResBlocks
**Batch** {ks.get('batch', 128)} throughout · **End to end** {e2e.get('steps', 200)} DDIM steps
**MoDiff configuration** `MODIFF_DELTA_MODE=dynamic` · `MODIFF_DELTA_REFRESH=4` ·
`MODIFF_DELTA_CLIP=1.0` · `MODIFF_DELTA_REPORT=0` · `MODIFF_LINEAR=0`

Data and figures, with the vocabulary explained. No analysis: nothing here ranks, recommends or
concludes. Generated by `{D}/scripts/make_report.py`.

**Relation to `MEASUREMENT_REPORT_2026-08-01.md`.** Same suites, same statistics, same protocol,
same table shapes. That report's headline modes are `int8_baseline` / `int4_baseline` — MoDiff
temporal caching **disabled** — so it contains no MoDiff measurement at all; its own preamble says
so. This report adds the two MoDiff modes and keeps the two baselines alongside them, so every
MoDiff figure is divided by fp16 *and* by the same bit width with MoDiff off.

**Measurement protocol.** Every timing is taken twice over: once WITHOUT a profiler, which is the
number reported, and once WITH one, which supplies the per-kernel attribution. Kernel and layer
suites do {ks.get('warmup', 30)} warmup calls then {ks.get('rounds', 8)} rounds x
{ks.get('iters_per_round', 60)} timed calls; the end-to-end suite does 3 warmup samples then
{e2e.get('repeats', 5)} profiler-free repeats, and that whole procedure is run three times as three
independent invocations. Discarding the warmup samples is load-bearing for the MoDiff modes: the
quantized attention blocks self-calibrate over their first forwards, and sample 1 of a run is
several times worse than sample 2.

**What is NOT in this report, and why.** The 08-01 report's three nsys-derived sections — in-context
attention per shape, GPU idle on the timeline, share of end-to-end time by layer type — are absent.
Nsight Systems is not installed in this environment (`which nsys` finds nothing; only
nsight-compute is present, and its hardware counters are blocked by `RmProfilingAdminOnly=1` with no
`CAP_SYS_ADMIN` in the container). Those three sections need NVTX range attribution from a trace and
cannot be reconstructed from torch.profiler output. Reprovision with
`integration/benchmarks/report/setup_nsys.sh` and re-run `nsys_layer_trace.py` to add them.

{GLOSSARY}

## Source data

| file | contents |
|---|---|
| `{D}/data/e2e_modiff_b128_run{{1,2,3}}.json` | end-to-end suite, three independent invocations, 5 modes |
| `{D}/data/kernel_suites_b128.json` | attention / conv / linear / norm+quantize kernel suites |
| `{D}/data/layers_b128.json` | per-layer suite: every (kind, shape) x mode |
| `{D}/logs/` | stdout of every run, including the route check printed per mode |

| script | role |
|---|---|
| `{D}/scripts/run_all.sh` | runs all three suites, in order, with the MoDiff configuration pinned |
| `integration/benchmarks/report/e2e_three_mode_bench.py` | end-to-end suite (`E2EBENCH_MODES`) |
| `integration/benchmarks/report/kernel_suites_bench.py` | kernel suites (`KBENCH_MODES`) |
| `integration/benchmarks/report/layer_pipeline_bench.py` | layer suite (`LBENCH_MODES`) |
| `integration/benchmarks/report/ck_bench_stats.py` | the statistics shared by all suites |
| `integration/benchmarks/report/ck_stages.py` | kernel name → op class, used by the stage table |
| `{D}/scripts/make_report.py` | this document |

**Route check** (printed by the harness for the MoDiff INT8 mode, so the measured configuration is
on the record rather than assumed): `{cfg if cfg else 'not recorded'}`

## End to end, three independent invocations

{e2e_tab}

### Whole-model time by stage

Profiler self-time scaled to the measured median wall time, ms per batch. Op classes are
`ck_stages.STAGES`; the MoDiff GN chain (`gn_apply_delta_quantize`, `gn_stats_partials_chanmajor`,
`gn_stats_reduce_partials`, `delta_absmax_fp16`) is attributed to **GroupNorm + quantize**, which
required adding those four name tokens — before 2026-08-04 they matched no bucket and fell through
to elementwise/other. A second token was added in the same pass: `s161616gemm`
(`cutlass_80_wmma_tensorop_f16_s161616gemm`, the fp16 timestep-embedding Linear) matched neither
`s16816gemm` nor `s1688gemm` — the digit runs differ — so it was in elementwise/other in every mode.
**The 08-01 report was generated before both fixes**, so its stage rows and these are not identical
to the last decimal; the totals are unaffected because both fixes move time between stages.

{stage_table(e2e)}

### What is inside "elementwise / copies / other"

The stage table is only as good as `ck_stages`' name matching: a kernel matching no token lands in
the catch-all silently. Its contents are listed so the bucket is auditable, top 8 per mode.

{misc_audit(e2e)}

### Kernels that differ between a baseline and its MoDiff mode

ms per batch, from the same profiled run as the stage table. A blank means the kernel does not run
in that mode.

{modiff_kernel_table(e2e)}

## Suites: full tables

> **How the call signatures were captured, and what that means for the MoDiff modes.**
> `kernel_suites_bench.capture()` runs one real sample of 5 DDIM steps and records the FIRST
> occurrence of each (entry, argument-shape) signature; the suite then benchmarks that recorded call
> {ks.get("rounds", 8)} rounds x {ks.get("iters_per_round", 60)} times in isolation. For the baseline modes every step is alike, so the captured
> call is representative. **For the MoDiff modes it need not be**: step 1 is the unmodulated
> `â_T = Q(a_T)` step, so a signature first seen there is timed as though it ran on every step.
>
> This was verified, not assumed. Hooking `F.conv2d` for 13 steps at batch 8 shows the four decoder
> skip 1x1 convs (Cin 576/768/1152/1536) receiving **fp32 input on step 1 only** — 4 of 11 calls —
> and fp16 on every step from 2 onward. Their suite rows therefore carry an fp32→fp16 cast
> (809 µs on the largest) that a steady-state step does not pay. Extrapolating those rows across a
> 200-step run would claim 1.50 ms/step of cast overhead; the end-to-end profile puts the real
> figure at **+0.40 ms/step** of total copy time, and that is the number to use. Rows whose two
> captures disagree on input dtype are marked † and their ratios are not steady-state.
>
> The end-to-end stage table above is profiled over a full {e2e.get("steps", 200)}-step run and does not have this
> problem. Where the two disagree, the stage table is authoritative for attribution and the suite
> is authoritative for per-call cost of a given signature.

### ATTENTION KERNELS — per call, batch {ks.get('batch', 128)}, {ks.get('rounds', 8)} rounds x {ks.get('iters_per_round', 60)} iters, warmup {ks.get('warmup', 30)}

{suite_tables(ks, 'attention')}
**Cross-mode: MoDiff against its own baseline, on the call signature**

{cross_mode(ks, 'attention')}

### CONV KERNELS — per call, batch {ks.get('batch', 128)}, {ks.get('rounds', 8)} rounds x {ks.get('iters_per_round', 60)} iters, warmup {ks.get('warmup', 30)}

{suite_tables(ks, 'conv')}
**Cross-mode: MoDiff against its own baseline, on the call signature**

{cross_mode(ks, 'conv')}

### NORM + QUANTIZE KERNELS — per call, batch {ks.get('batch', 128)}

{suite_tables(ks, 'norm_quantize')}
**Cross-mode: MoDiff against its own baseline, on the call signature**

{cross_mode(ks, 'norm_quantize')}

### LINEAR KERNELS — per call, batch {ks.get('batch', 128)}

{suite_tables(ks, 'linear')}
**Cross-mode: MoDiff against its own baseline, on the call signature**

{cross_mode(ks, 'linear')}

## PER LAYER — batch {lay.get('batch', 128)}

`n` is how many layers of that (kind, shape) the UNet contains. `µs` is the measured module latency,
not a kernel sum.

{layer_table(lay)}

### Layer-suite stability

{layer_stability(lay)}

## Attention layer: per-kernel profile

Per-kernel self-time inside one attention forward, per shape, all five modes. `measured latency` is
the profiler-free module timing; where it exceeds the kernel total the layer was CPU-dispatch bound.

{attention_profile(lay)}

## Addendum 2026-08-05: the GroupNorm C>1024 fallback, fixed after these runs

Every number above predates one kernel change and is stale by 0.4-0.6% on the two MoDiff rows only.
Recorded here rather than silently regenerated, because re-running all three suites is ~2 hours and
the tables are otherwise a consistent snapshot of one commit.

`gn_stats_partials_chanmajor_kernel` required `C <= 1024` (one thread per channel) and
`gn_launch_group_stats` fell back to the historical group-major tree otherwise. That condition was
believed to cover the whole UNet; it covers the ENCODER only, because a decoder ResBlock normalises
`cat([h, hs.pop()])` and so sees 1152 or 1536 channels. In the profile above,
`gn_group_stats_kernel` is still running at 142.3 (W8A8) / 140.0 (W4A4) ms/batch for exactly those
layers. The kernel now takes `K = ceil(C/1024)` channels per thread; `K=1` is the original code.

Re-measured, same protocol, baselines included as a drift control:

| mode | before | after | change |
|---|---:|---:|---:|
| INT8 (MoDiff off) | 14221.1 | 14223.5 | +0.02% (control) |
| INT4 (MoDiff off) | 11450.3 | 11474.1 | +0.21% (control) |
| INT8 + MoDiff | 15114.9 | 15030.7 | **-0.56%** |
| INT4 + MoDiff | 12312.5 | 12253.6 | **-0.48%** |

Deficit against its own baseline, within-run so drift cancels: W8A8 **4.469 -> 4.036 ms/step**
(ratio 0.941 -> **0.946x**), W4A4 **4.311 -> 3.898 ms/step** (0.930 -> **0.936x**).
`gn_group_stats_kernel` is absent from both profiles afterwards, and the GroupNorm+quantize stage
falls 93.0 (W8A8) / 79.5 (W4A4) ms/batch while both baselines move by <= 8 ms.

Data: `data/e2e_after_gnfix.json`. A/B and fp64/determinism checks:
`scripts/gn_chanmajor_c_gt_1024.py` (run it twice, second time with `--alt0`; the variant selector
is a function-local static so an in-process A/B is silently ineffective).

## Measurement conditions

- `models/ldm/lsun_churches256/model.ckpt` is the **real trained checkpoint** — 2.69 GB, 1307
  `state_dict` entries, installed 2026-08-04. An earlier draft of this section described it as the
  856-byte stub with 0 entries and randomly-initialised weights; that was inherited verbatim from
  the 08-01 report, whose runs predate the checkpoint being put in place, and it is wrong for every
  measurement in this document.
- **No figure here is still an image-quality measurement**, but for a different reason than that
  draft gave. The calibration artifacts used are the un-suffixed ones, which were fitted against the
  stub and give latent relL2 0.88 (W8A8) / 3.02 (W4A4) when paired with real weights. They are used
  deliberately, to keep the configuration identical to 08-01's: scale VALUES do not affect kernel
  selection or duration, so every timing here is unaffected. Any accuracy figure needs
  `integration/calibration/int{{8,4}}_calibration_realckpt.pt` instead.
- Nsight Compute counters are unavailable: `ncu` returns `ERR_NVGPUCTRPERM`, the driver has
  `RmProfilingAdminOnly=1` and the container has no `CAP_SYS_ADMIN`. Nsight Systems is not installed
  at all in this environment, which is why the three trace-derived sections are absent.
- MoDiff on the Linear layers (`MODIFF_LINEAR=1`) is implemented but **off** in every measurement
  here. It is not a default and including it would measure a configuration nothing ships.
- The delta-absmax reporting path (`MODIFF_DELTA_REPORT=1`) is **off**. It is a wall-clock
  optimisation that publishes a scale used up to `2 x MODIFF_DELTA_REFRESH` steps later.
- Within one process, `cv_pct` does not predict cross-session reproducibility for launch-bound rows.
  The three end-to-end invocations are the cross-session measurement; their spread is the
  `cross-run CV` column.
"""
    with open(os.path.join(ROOT, OUT), "w") as f:
        f.write(doc)
    print("wrote %s (%d lines, %d table rows)"
          % (OUT, doc.count("\n") + 1, sum(1 for l in doc.splitlines() if l.startswith("|"))))


if __name__ == "__main__":
    main()
