"""Plot 3: where the end-to-end time goes, drilled down three levels.

  level 1  layer type          attention / resblock_plain / resblock_updown / outside any layer
  level 2  op class            attention core / conv / GEMM / norm+quantize / elementwise
  level 3  individual kernel

Attribution comes from nsys NVTX ranges (one per layer instance) via
`cuda_gpu_kern_sum:nvtx-name`, which reports each kernel's GPU time split by the range it belongs
to. That correlation has to come from nsys: a CPU-side NVTX range closes when the CPU has finished
ISSUING the layer, and by then the GPU is still executing earlier work, so slicing the timeline by
range timestamps attributes kernels to the wrong layer entirely.

Totals are scaled to the end-to-end wall clock measured WITHOUT a profiler, so the bars sum to the
latency actually reported rather than to the traced total. Traces are averaged over the runs
available for each mode.

Writes plots/fig_e2e_hierarchy.png.
"""
import argparse
import collections
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from ck_stages import stage_of, STAGE_LABEL  # noqa: E402

MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
L1_COL = {"attention": "#c0392b", "resblock_plain": "#2f6fb2",
          "resblock_updown": "#7d5ba6", "outside layers": "#b9c2cc"}
L2_COL = {"attention core": "#e05c4a", "convolution": "#1b7f79",
          "QKV / output projection": "#3f86c9", "GroupNorm + quantize": "#e6a020",
          "elementwise / copies / other": "#aab3bd",
          "K/V gather + transpose": "#9b7fc0", "attention output quantize": "#c98bdb"}


def find_nsys():
    p = subprocess.run(["bash", os.path.join(HERE, "setup_nsys.sh")],
                       capture_output=True, text=True)
    return p.stdout.strip().splitlines()[-1]


def kern_by_range(nsys, rep):
    """[(range_or_None, kernel, total_ns)] from cuda_gpu_kern_sum:nvtx-name, cached beside the rep."""
    cache = rep.replace(".nsys-rep", ".kernrange.csv")
    if os.path.exists(cache):
        rows = list(csv.reader(open(cache)))
        return [(r[0] or None, r[1], float(r[2])) for r in rows]
    p = subprocess.run([nsys, "stats", "--force-export=true",
                        "--report", "cuda_gpu_kern_sum:nvtx-name", "--format", "csv", rep],
                       capture_output=True, text=True, cwd=os.path.dirname(rep) or ".")
    lines = p.stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("Time (%)")), None)
    out = []
    if start is not None:
        for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
            name = r.get("Name") or ""
            tot = float((r.get("Total Time (ns)") or "0").replace(",", ""))
            # The Name field is "<range>/<kernel>" when the kernel ran inside an NVTX range and
            # just "<kernel>" otherwise. Some reports prefix the range with ':' and some do not,
            # so keying on the leading colon (as a first version did) put every kernel in the
            # no-range bucket and level 1 read "outside layers 100%".
            nm = name.lstrip(":")
            if "/" in nm and nm.startswith("L|"):
                rng, kern = nm.split("/", 1)
            else:
                rng, kern = None, nm
            out.append((rng, kern, tot))
    with open(cache, "w", newline="") as f:
        csv.writer(f).writerows([[a or "", b, c] for a, b, c in out])
    return out


def short_kernel(name):
    n = re.sub(r"^void\s+", "", name).split("(")[0]
    n = n.split("<")[0]
    return n.split("::")[-1][:44] or name[:44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys-dir", default="docs/final_report_2026-07-28/data/nsys")
    ap.add_argument("--e2e", required=True, action="append",
                    help="e2e json for the no-profiler wall time; repeatable, means averaged")
    ap.add_argument("--out", default="docs/final_report_2026-07-28/plots/fig_e2e_hierarchy.png")
    ap.add_argument("--topk", type=int, default=6)
    a = ap.parse_args()
    nsys = find_nsys()
    e2es = [json.load(open(f if os.path.isabs(f) else os.path.join(ROOT, f))) for f in a.e2e]
    nd = a.nsys_dir if os.path.isabs(a.nsys_dir) else os.path.join(ROOT, a.nsys_dir)

    fig, axes = plt.subplots(3, 1, figsize=(17, 11.5))
    for ax, (mode, lbl) in zip(axes, MODES):
        reps = sorted(glob.glob(os.path.join(nd, "nsys_all_%s_b128_run*.nsys-rep" % mode)))
        if not reps:
            ax.text(.5, .5, "no traces for %s" % mode, ha="center"); ax.axis("off"); continue
        acc = collections.defaultdict(float)
        for rep in reps:
            for rng, kern, tot in kern_by_range(nsys, rep):
                l1 = "outside layers"
                if rng and rng.startswith("L|"):
                    l1 = rng.split("|")[1]
                acc[(l1, stage_of(kern), short_kernel(kern))] += tot / len(reps)
        traced = sum(acc.values())
        # Mean over the independent e2e invocations, each of which timed the model WITHOUT a
        # profiler before profiling it once. Cross-run CV here is 0.00-0.14%.
        walls = [d["modes"][mode]["stats"]["mean"] / 1e3 for d in e2es]
        wall_ms = sum(walls) / len(walls)
        scale = wall_ms / (traced / 1e6) if traced else 1.0

        l1t = collections.Counter()
        l2t = collections.Counter()
        for (l1, st, kern), v in acc.items():
            l1t[l1] += v * scale / 1e6
            l2t[(l1, st)] += v * scale / 1e6

        order1 = [k for k, _ in l1t.most_common()]
        y = {"L1": 2.4, "L2": 1.2, "L3": 0.0}
        x = 0.0
        for l1 in order1:
            w1 = l1t[l1]
            ax.barh(y["L1"], w1, left=x, height=0.9, color=L1_COL.get(l1, "#999"),
                    edgecolor="w")
            if w1 / wall_ms > 0.04:
                ax.text(x + w1 / 2, y["L1"], "%s\n%.0f ms  %.0f%%"
                        % (l1, w1, w1 / wall_ms * 100), ha="center", va="center",
                        fontsize=8.5, color="w", fontweight="bold")
            x2 = x
            for (ll1, st), w2 in sorted(l2t.items(), key=lambda kv: -kv[1]):
                if ll1 != l1:
                    continue
                lab = STAGE_LABEL.get(st, st)
                ax.barh(y["L2"], w2, left=x2, height=0.9,
                        color=L2_COL.get(lab, "#ccc"), edgecolor="w")
                if w2 / wall_ms > 0.035:
                    ax.text(x2 + w2 / 2, y["L2"], "%s\n%.0f ms" % (lab.split(" /")[0], w2),
                            ha="center", va="center", fontsize=7.5, color="w")
                ks = sorted(((v * scale / 1e6, kern) for (a1, s1, kern), v in acc.items()
                             if a1 == l1 and s1 == st), reverse=True)
                x3 = x2
                for i, (w3, kern) in enumerate(ks):
                    shade = 0.55 + 0.35 * (i % 3) / 2
                    ax.barh(y["L3"], w3, left=x3, height=0.9,
                            color=L2_COL.get(lab, "#ccc"), alpha=shade, edgecolor="w")
                    if w3 / wall_ms > 0.028:
                        ax.text(x3 + w3 / 2, y["L3"], "%s\n%.0f ms" % (kern[:26], w3),
                                ha="center", va="center", fontsize=6.6, color="black")
                    x3 += w3
                x2 += w2
            x += w1
        ax.set_xlim(0, wall_ms * 1.005)
        ax.set_ylim(-0.6, 3.1)
        ax.set_yticks([y["L3"], y["L2"], y["L1"]])
        ax.set_yticklabels(["kernel", "op class", "layer type"], fontsize=9)
        ax.set_xlabel("ms per batch of 128 (scaled to the profiler-free wall time)", fontsize=9)
        ax.set_title("%s  —  %.0f ms/batch, mean of %d profiler-free runs; "
                     "attribution from %d nsys trace(s)"
                     % (lbl, wall_ms, len(walls), len(reps)),
                     loc="left", fontsize=11, fontweight="bold")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Where the end-to-end time goes: layer type -> op class -> kernel",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    fig.savefig(out, dpi=150, facecolor="w")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
