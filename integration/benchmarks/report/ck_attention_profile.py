"""Per-kernel profile INSIDE the attention layer, for every real shape and all three modes.

Answers "where does an AttentionBlock's time actually go", which the speedup tables cannot: a
shape can win at the layer level while its attention-core kernel loses (T=16 does exactly that),
and only the kernel breakdown shows which stage paid for it.

Source is the layer suite (layers_*.json), whose per-layer profile records each CUDA kernel's
self time inside one layer call, its launch count, and profile_tree's role label. Times are
per layer call at the measured batch; the stack sums to the independently measured pipeline
latency (gpu_busy_frac is printed so a gap between summed kernel time and wall time -- launch
overhead -- is visible rather than hidden).

Writes plots/fig_attn_kernel_profile.png and prints the tables.
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]

# Role -> short bucket + colour, so the same kernel family gets the same colour in every panel.
BUCKETS = [
    ("attention core", "#c0392b",
     ("flash kernel", "flash", "sdpa", "softmax", "bmm", "attention")),
    ("QKV / out projection", "#2f6fb2", ("gemm", "linear", "projection", "matmul")),
    # FP16's GroupNorm is a two-pass gn_accum + gn_finalize pair whose names contain neither
    # "group_norm" nor "norm"; without the "gn_" token they fell into the elementwise bucket and
    # FP16's GN cost vanished from the comparison.
    ("GroupNorm + quantize", "#e6a020",
     ("gn+silu", "group_norm", "groupnorm", "gn_accum", "gn_finalize", "gn_", "norm")),
    ("quantize / pack", "#7d5ba6", ("quantize", "pack", "quant")),
    ("elementwise / copy", "#b9c2cc", ()),
]


def bucket_of(role, kernel):
    hay = (role + " " + kernel).lower()
    for name, colour, frags in BUCKETS:
        if frags and any(f in hay for f in frags):
            return name
    return BUCKETS[-1][0]


def shape_label(xs):
    return "C%d/%d² (T=%d)" % (xs[1], xs[2], xs[2] * xs[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="docs/final_report_2026-07-28/plots/fig_attn_kernel_profile.png")
    a = ap.parse_args()
    lay = json.load(open(a.layers if os.path.isabs(a.layers) else os.path.join(ROOT, a.layers)))

    idx = {}
    for m, _ in MODES:
        for e in lay["modes"][m]:
            if e["kind"] == "attention":
                idx.setdefault(tuple(e["x_shape"]), {})[m] = e
    keys = sorted(idx, key=lambda k: -(k[2] * k[3]))

    print("Per-kernel profile inside the AttentionBlock — batch %d, µs per layer call\n"
          % lay["batch"])
    for xs in keys:
        n = idx[xs]["fp16"]["n_instances"]
        print("=" * 96)
        print("%s   x%d instances" % (shape_label(xs), n))
        print("=" * 96)
        for m, lbl in MODES:
            e = idx[xs][m]
            print("\n%s — pipeline %.1f µs, kernels sum %.1f µs, GPU busy %.3f"
                  % (lbl, e["pipeline_us"], e["gpu_us_sum"], e["gpu_busy_frac"]))
            print("| us | % of layer | launches | kernel | bucket |")
            print("|---:|---:|---:|---|---|")
            for k in e["kernels"]:
                print("| %.1f | %.1f%% | %s | `%s` | %s |"
                      % (k["us_per_layer_call"], k["pct_of_layer"], k["calls"],
                         k["kernel"][:52], bucket_of(k["role"], k["kernel"])))
        print()

    # bucket rollup, the cross-mode view
    print("\n" + "=" * 96)
    print("BY BUCKET — µs per layer call")
    print("=" * 96)
    roll = collections.defaultdict(lambda: collections.defaultdict(float))
    for xs in keys:
        for m, _ in MODES:
            for k in idx[xs][m]["kernels"]:
                roll[(xs, m)][bucket_of(k["role"], k["kernel"])] += k["us_per_layer_call"]
    names = [b[0] for b in BUCKETS]
    for xs in keys:
        print("\n**%s**" % shape_label(xs))
        print("| bucket | FP16 | INT8 | INT4 | INT4 vs FP16 |")
        print("|---|---:|---:|---:|---:|")
        for b in names:
            v = [roll[(xs, m)].get(b, 0.0) for m, _ in MODES]
            if max(v) < 0.05:
                continue
            d = ("%.2f×" % (v[0] / v[2])) if v[2] > 0 else ("gone" if v[0] > 0 else "—")
            print("| %s | %.1f | %.1f | %.1f | %s |" % (b, v[0], v[1], v[2], d))
        tot = [idx[xs][m]["pipeline_us"] for m, _ in MODES]
        print("| **total (measured)** | **%.1f** | **%.1f** | **%.1f** | **%.2f×** |"
              % (tot[0], tot[1], tot[2], tot[0] / tot[2]))

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 5.4))
    for ax, xs in zip(np.atleast_1d(axes), keys):
        bot = np.zeros(len(MODES))
        for bname, colour, _ in BUCKETS:
            v = np.array([roll[(xs, m)].get(bname, 0.0) for m, _ in MODES])
            if v.max() < 0.05:
                continue
            ax.bar([l for _, l in MODES], v, .6, bottom=bot, color=colour,
                   label=bname if xs == keys[0] else None)
            bot += v
        # Mark the MEASURED pipeline latency. Without it the stack reads as the layer's time,
        # which is badly wrong where the layer is launch-bound: at T=4 gpu_busy_frac is
        # 0.45-0.57, i.e. over 40% of the wall clock is gaps between kernels and the stack
        # accounts for barely half of it. An earlier caption claimed the gap was 0-1%.
        wall = np.array([idx[xs][m]["pipeline_us"] for m, _ in MODES])
        ax.plot(np.arange(len(MODES)), wall, "_", color="black", markersize=26,
                markeredgewidth=2.2,
                label="measured pipeline" if xs == keys[0] else None)
        for i, (t, w) in enumerate(zip(bot, wall)):
            frac = t / w if w else 0
            ax.text(i, w * 1.03, "%.0f" % w, ha="center", fontsize=8.5, fontweight="bold")
            if frac < 0.95:
                ax.text(i, t * 0.5, "kernels\n%.0f%%" % (frac * 100), ha="center",
                        fontsize=7.5, color="black")
        ax.set_title(shape_label(xs), fontsize=10)
        ax.set_ylim(0, wall.max() * 1.22)
        ax.grid(axis="y", alpha=.25)
    np.atleast_1d(axes)[0].set_ylabel("µs per layer call")
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=9)
    # The stack height is the SUM OF KERNEL SELF-TIMES, which is slightly below the measured
    # pipeline latency (the gap is launch overhead -- gpu_busy_frac in the tables). Labelling it
    # "measured pipeline" would overstate what the bars show by that gap.
    fig.suptitle("Kernels inside the AttentionBlock, batch %d — stack = summed kernel "
                 "self-time, black tick + label = measured pipeline latency.\n"
                 "Where they differ the layer is launch-bound: kernels cover 98%% of the wall "
                 "at T>=64 but only 45-57%% at T=4."
                 % lay["batch"], y=.94, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, .86])
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    fig.savefig(out, dpi=150, facecolor="w")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
