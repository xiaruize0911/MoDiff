"""Plots for the FP16 / INT8 / INT4 attention-layer comparison across all 21 blocks.

Reads data/attn_three_mode_final.json (all three modes measured in ONE process, so the
columns are directly comparable) and writes two figures:

  fig_attn_three_mode.png       per-shape latency + speedup + weighted total
  fig_attn_kernel_breakdown.png where the time goes inside each layer, per mode

Kernel -> stage attribution is by name; anything unmatched lands in "other" and is printed
so it can never be silently absorbed into a neighbouring stage.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "docs/final_report_2026-07-28/data/attn_three_mode_final.json")
OUT = os.path.join(ROOT, "docs/final_report_2026-07-28/plots")
os.makedirs(OUT, exist_ok=True)

MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
COL = {"fp16": "#8d99ae", "int8_baseline": "#2f6fb2", "int4_baseline": "#27924f"}
COUNT = {1024: 5, 256: 5, 64: 5, 16: 5, 4: 1}
ORDER = [1024, 256, 64, 16, 4]

# stage -> (label, colour, name fragments)
STAGES = [
    ("attn", "attention core", "#c0392b",
     ("flash_attn_int8_mma", "flash_attn_int4_mma", "pytorch_flash", "flash_fwd",
      "qi8packed_small", "bmm", "softmax")),
    ("gemm", "QKV + output projection", "#2f6fb2",
     ("gemm_w8a8_kernel", "gemm_w4a4_kernel", "ImplicitGemmConvolutionFusion",
      "ampere_fp16_s1688gemm", "ampere_fp16_s16816gemm", "sm80_xmma_gemm",
      "gemm_w4a4_awq", "cutlass")),
    ("norm", "GroupNorm + quantize", "#e6a020",
     ("group_norm", "gn_accum", "gn_finalize")),
    ("prep", "K/V prep + out quantize", "#7d5ba6",
     ("aq_kv_packed", "quant_attn_out", "quantize_attn", "qkv_i4codes",
      "from_i8_kv_tiled")),
    ("misc", "residual / copies", "#b9c2cc",
     ("elementwise", "FillFunctor", "direct_copy", "reduce_kernel")),
]


def classify(name):
    for key, _, _, frags in STAGES:
        if any(f in name for f in frags):
            return key
    return None


def load():
    d = json.load(open(DATA))
    out = {}
    for mode, _ in MODES:
        if mode not in d["modes"]:
            sys.exit(f"mode {mode} missing from {DATA}")
        for e in d["modes"][mode]:
            if e.get("kind") != "attention":
                continue
            xs = e["x_shape"]
            out.setdefault(mode, {})[xs[2] * xs[3]] = {
                "C": xs[1], "us": e["pipeline_us"], "kernels": e["kernels"]}
    return out


def stage_split(entry, unmatched):
    tot = {k: 0.0 for k, _, _, _ in STAGES}
    for k in entry["kernels"]:
        c = classify(k["kernel"])
        if c is None:
            unmatched.append(k["kernel"])
            c = "misc"
        tot[c] += k["us_per_layer_call"]
    # profiler sum vs the independently measured pipeline time: keep them consistent by
    # scaling, so the stack heights add up to the latency actually reported.
    s = sum(tot.values())
    if s > 0:
        f = entry["us"] / s
        tot = {k: v * f for k, v in tot.items()}
    return tot


D = load()
unmatched = []
weighted = {m: sum(D[m][T]["us"] * COUNT[T] for T in ORDER) / 1000 for m, _ in MODES}

# ---------------------------------------------------------------- figure 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6),
                               gridspec_kw={"width_ratios": [2.3, 1]})
plt.rcParams.update({"font.size": 10})
w = 0.26
xs = range(len(ORDER))
for i, (m, lbl) in enumerate(MODES):
    vals = [D[m][T]["us"] for T in ORDER]
    pos = [x + (i - 1) * w for x in xs]
    ax1.bar(pos, vals, w, label=lbl, color=COL[m])
    for p, v, T in zip(pos, vals, ORDER):
        if m != "fp16":
            sp = D["fp16"][T]["us"] / v
            ax1.text(p, v * 1.03, f"{sp:.2f}x", ha="center", fontsize=7.5,
                     color=COL[m], fontweight="bold")
ax1.set_xticks(list(xs))
ax1.set_xticklabels([f"C{D['fp16'][T]['C']}/T{T}\n×{COUNT[T]} blocks" for T in ORDER])
ax1.set_yscale("log")
ax1.set_ylabel("µs per layer (log scale)")
ax1.set_title("A.  Attention-layer latency at every shape in the model\n"
              "A40, batch 128; labels are speedup vs FP16", loc="left")
ax1.legend(frameon=False, ncol=3)
ax1.grid(axis="y", alpha=.25, which="both")

names = [l for _, l in MODES]
vals = [weighted[m] for m, _ in MODES]
bars = ax2.bar(names, vals, .55, color=[COL[m] for m, _ in MODES])
for b, v, (m, _) in zip(bars, vals, MODES):
    sp = weighted["fp16"] / v
    ax2.text(b.get_x() + b.get_width() / 2, v + .25,
             f"{v:.3f} ms\n{sp:.3f}x", ha="center", fontsize=9.5, fontweight="bold")
ax2.set_ylim(0, max(vals) * 1.22)
ax2.set_ylabel("ms, weighted over all 21 blocks")
ax2.set_title("B.  Whole attention layer, model-weighted\n"
              "(5×T1024, 5×T256, 5×T64, 5×T16, 1×T4)", loc="left")
ax2.grid(axis="y", alpha=.25)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_attn_three_mode.png", dpi=150, facecolor="w")

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 5, figsize=(17, 5.2))
for ax, T in zip(axes, ORDER):
    bot = [0.0] * len(MODES)
    for key, lbl, colr, _ in STAGES:
        vals = [stage_split(D[m][T], unmatched)[key] for m, _ in MODES]
        ax.bar([l for _, l in MODES], vals, .6, bottom=bot, color=colr,
               label=lbl if T == ORDER[0] else None)
        bot = [b + v for b, v in zip(bot, vals)]
    for i, (m, _) in enumerate(MODES):
        ax.text(i, bot[i] * 1.02, f"{D[m][T]['us']:.0f}", ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_title(f"C{D['fp16'][T]['C']} / T{T}  ×{COUNT[T]}", fontsize=10)
    ax.set_ylim(0, max(bot) * 1.16)
    ax.grid(axis="y", alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("µs per layer")
fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.0))
fig.suptitle("Where each attention layer's time goes, by stage        "
             "(A40, batch 128; bar labels are total µs)",
             y=.86, x=.5, fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .84])
fig.savefig(f"{OUT}/fig_attn_kernel_breakdown.png", dpi=150, facecolor="w")

print("weighted (ms):", {l: round(weighted[m], 3) for m, l in MODES})
if unmatched:
    print("UNMATCHED kernels folded into 'other':", sorted(set(unmatched)))
else:
    print("every kernel matched a stage")
print(f"wrote {OUT}/fig_attn_three_mode.png")
print(f"wrote {OUT}/fig_attn_kernel_breakdown.png")
