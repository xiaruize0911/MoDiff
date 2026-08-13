"""Per-KERNEL breakdown inside the attention, conv and linear suites, from kernel_suites.json.

REPORT.md's sections 2-4 give each suite's total and its entry points. This drills one level further, to
the number that tells you where to optimise: for every kernel, the per-call median AND the calls per
sample, because a suite total conflates the two. A kernel at 300 us x 5 calls and one at 3 us x 500 calls
both cost 1.5 ms/sample and want completely different fixes.

WHAT THE NUMBERS ARE. The bench captures the real call arguments at the C++ entry point during a live
sample, then replays each call signature in isolation (8 rounds x 60 iters, median of round medians). So
`us/call` is that replay median at the shape the model actually runs, and

    ms/sample = us/call x calls_per_sample / 1000

summed over a kernel's signatures. This is a REPLAY total, not a profiler total: it excludes launch gaps
and any overlap between kernels, which is why a suite's ms/sample does not have to match its share of the
end-to-end wall time. Section 1a of REPORT.md is the profiler view; this is the kernel view.

SELF-CHECK. Every suite total recomputed here is asserted against the value REPORT.md already publishes,
so a units or aggregation mistake fails loudly instead of producing a plausible table.

Run: python docs/bench_report_2026-08-13_postzp/scripts/make_kernel_breakdown.py    # no GPU
Writes docs/bench_report_2026-08-13_postzp/KERNEL_BREAKDOWN.md
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

D = "docs/bench_report_2026-08-13_postzp"
SUITES = ("attention", "conv", "linear")
MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]
#: a signature is broken out individually when it costs at least this much of its suite
SIG_SHARE = 0.04


def ms_per_sample(rec):
    return rec["stats"]["median"] * rec["calls_per_sample"] / 1000.0


def shape_str(rec):
    """The first two argument shapes, which for every suite here are the ones that identify the work."""
    sh = [s for s in (rec.get("arg_shapes") or []) if s]
    return " x ".join("[" + ",".join(str(d) for d in s) + "]" for s in sh[:2]) or "-"


def published_totals():
    """The suite totals REPORT.md already states, parsed back out so this file can be checked against
    them rather than trusted alongside them."""
    txt = open(f"{D}/REPORT.md").read()
    out = {}
    for suite, header in (("attention", "## 2. Attention kernels"),
                          ("conv", "## 3. Conv kernels"),
                          ("linear", "## 4. Linear kernels")):
        seg = txt[txt.index(header):]
        seg = seg[:seg.index("### ")]
        for m in re.finditer(r"^\| (fp16|W8A8 PTQ|W8A8 MoDiff|W4A4 PTQ|W4A4 MoDiff) \| \*\*([\d.]+)\*\*",
                             seg, re.M):
            out[(suite, m.group(1))] = float(m.group(2))
    return out


def main():
    d = json.load(open(f"{D}/data/kernel_suites.json"))
    pub = published_totals()
    o = []
    o.append("# Per-kernel breakdown: attention, conv, linear")
    o.append("")
    o.append(f"`{d['gpu']}`, batch {d['batch']}, replayed at the shapes captured from a live sample "
             f"({d['rounds']} rounds x {d['iters_per_round']} iters, median of round medians). "
             f"Generated from `data/kernel_suites.json` -- see the script header for what `ms/sample` "
             f"is and is not.")
    o.append("")
    o.append("`ms/sample = us/call x calls/sample / 1000`. Both factors are shown because they point at "
             "different fixes: a fat kernel wants a better tile, a frequent one wants fusion.")
    o.append("")

    mismatches = []
    for suite in SUITES:
        o.append(f"## {suite}")
        o.append("")
        for key, label in MODES:
            recs = d["modes"].get(key, {}).get(suite) or []
            if not recs:
                continue
            by_entry = collections.defaultdict(lambda: [0.0, 0, 0.0, []])
            for r in recs:
                e = by_entry[r["entry"]]
                e[0] += ms_per_sample(r)
                e[1] += r["calls_per_sample"]
                e[2] = max(e[2], r["stats"]["cv_pct"])
                e[3].append(r)
            total = sum(v[0] for v in by_entry.values())
            want = pub.get((suite, label))
            if want is not None and abs(total - want) > max(0.05, 0.005 * want):
                mismatches.append(f"{suite}/{label}: recomputed {total:.3f} vs REPORT.md {want:.3f}")
            o.append(f"### {label} — {total:.2f} ms/sample total"
                     + (f"  (REPORT.md: {want:.2f} ✓)" if want is not None else ""))
            o.append("")
            o.append("| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |")
            o.append("|---|--:|--:|--:|--:|--:|--:|")
            for name, (ms, calls, cv, rs) in sorted(by_entry.items(), key=lambda kv: -kv[1][0]):
                us = ms * 1000.0 / calls if calls else float("nan")
                o.append(f"| `{name}` | **{ms:.2f}** | {100 * ms / total:.1f}% | {calls} | "
                         f"{us:.1f} | {len(rs)} | {cv:.2f}% |")
            o.append("")
            #: per-signature detail for the entries that carry the suite, so a big number can be traced
            #: to a shape rather than just to a name
            big = [r for r in recs if ms_per_sample(r) >= SIG_SHARE * total]
            if big:
                o.append(f"<details><summary>signatures ≥ {SIG_SHARE:.0%} of the suite "
                         f"({len(big)} of {len(recs)})</summary>")
                o.append("")
                o.append("| ms/sample | calls | µs/call | shapes | kernel |")
                o.append("|--:|--:|--:|---|---|")
                for r in sorted(big, key=lambda r_: -ms_per_sample(r_)):
                    o.append(f"| {ms_per_sample(r):.2f} | {r['calls_per_sample']} | "
                             f"{r['stats']['median']:.1f} | `{shape_str(r)}` | `{r['entry']}` |")
                o.append("")
                o.append("</details>")
                o.append("")

    if mismatches:
        raise SystemExit("SELF-CHECK FAILED, totals disagree with REPORT.md:\n  " +
                         "\n  ".join(mismatches))
    open(f"{D}/KERNEL_BREAKDOWN.md", "w").write("\n".join(o) + "\n")
    print(f"wrote {D}/KERNEL_BREAKDOWN.md ({len(o)} lines)")
    print("self-check: every suite total matches REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
