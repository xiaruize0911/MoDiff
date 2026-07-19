"""Plots for the per-layer Linear profile (decompose each quantized attention Linear into
quantize-pass us vs int-GEMM us, vs the fp16 cuBLAS GEMM). Reads
data/linear_layer_profile_b{16,64,128}.csv (whichever exist). Emits:
  16_linear_layer_profile.png  -- per-shape decomposition at batch 64 (per-step contribution)
  17_linear_batch_sweep.png    -- aggregate per-step totals + net vs fp16 across batches
"""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
P = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16"
FP16, I8, I4 = "#888888", "#4C78A8", "#E45756"
I8Q, I4Q = "#AEC7E8", "#F2A9A6"   # lighter = quantize pass (the tax the GEMM must pay for)


def rd(b):
    p = f"{D}/linear_layer_profile_b{b}.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("M", "K", "N", "count"):
            r[k] = int(r[k])
        for k in ("fp16_us", "i8_quant_us", "i8_gemm_us", "i8_total_us",
                  "i4_quant_us", "i4_gemm_us", "i4_total_us"):
            r[k] = float(r[k])
    return rows


def label(r, batch):
    tok = r["M"] // batch
    side = int(round(tok ** 0.5))
    kind = "qkv" if r["N"] == 3 * r["K"] else ("proj" if r["N"] == r["K"] else f"N{r['N']}")
    return f"{side}² C{r['K']} {kind}"


def save(fig, name):
    fig.savefig(f"{P}/{name}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ============ FIGURE 16: per-shape decomposition at batch 64 ============
B = 64
rows = rd(B)
if rows:
    # per-step contribution = single-instance us * count (how much this shape adds to a step)
    rows = sorted(rows, key=lambda r: -r["fp16_us"] * r["count"])
    labs = [f"{label(r, B)}  ×{r['count']}" for r in rows]
    y = np.arange(len(rows))[::-1]                     # top = biggest contributor
    h = 0.26
    fp = np.array([r["fp16_us"] * r["count"] for r in rows]) / 1000       # -> us? keep us; /1000 = ms? no
    # keep in microseconds-per-step (sum of instances). Convert to ms for readability:
    fp = np.array([r["fp16_us"] * r["count"] for r in rows]) / 1000.0     # ms/step
    i8g = np.array([r["i8_gemm_us"] * r["count"] for r in rows]) / 1000.0
    i8q = np.array([r["i8_quant_us"] * r["count"] for r in rows]) / 1000.0
    i4g = np.array([r["i4_gemm_us"] * r["count"] for r in rows]) / 1000.0
    i4q = np.array([r["i4_quant_us"] * r["count"] for r in rows]) / 1000.0

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + h, fp, h, color=FP16, label="fp16 GEMM (one op)")
    ax.barh(y, i8g, h, color=I8, label="int8 int-GEMM")
    ax.barh(y, i8q, h, left=i8g, color=I8Q, hatch="///", edgecolor="white", label="int8 quantize pass")
    ax.barh(y - h, i4g, h, color=I4, label="int4 int-GEMM")
    ax.barh(y - h, i4q, h, left=i4g, color=I4Q, hatch="///", edgecolor="white", label="int4 quantize pass")

    for i in range(len(rows)):
        # net vs fp16 annotations at bar ends
        ax.text(fp[i] + 0.004, y[i] + h, f"{fp[i]:.2f}", va="center", fontsize=7, color="#333")
        r8 = fp[i] / (i8g[i] + i8q[i]); c8 = "#1a7a1a" if r8 >= 1 else "#b00"
        ax.text(i8g[i] + i8q[i] + 0.004, y[i], f"{i8g[i]+i8q[i]:.2f}  {r8:.2f}×", va="center", fontsize=7, color=c8)
        r4 = fp[i] / (i4g[i] + i4q[i]); c4 = "#1a7a1a" if r4 >= 1 else "#b00"
        ax.text(i4g[i] + i4q[i] + 0.004, y[i] - h, f"{i4g[i]+i4q[i]:.2f}  {r4:.2f}×", va="center", fontsize=7, color=c4)

    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("per-step time (ms) = single-call × count   —  lower is better")
    ax.set_title("Per-layer Linear profile @ batch 64: every quantized Linear is attention qkv/proj\n"
                 "int-GEMM (dark) beats fp16 on mid layers, but the quantize pass (hatched) is added work fp16 never pays.\n"
                 "green × = quantized total faster than fp16 · red × = slower", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_xlim(0, max(fp.max(), (i8g + i8q).max()) * 1.18)
    ax.margins(y=0.02)
    save(fig, "16_linear_layer_profile.png")


# ============ FIGURE 17: aggregate totals + batch sweep ============
batches = [b for b in (16, 64, 128) if rd(b)]
data = {b: rd(b) for b in batches}


def agg(rows, key):
    return sum(r[key] * r["count"] for r in rows) / 1000.0   # ms/step


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

# left: stacked totals per batch (fp16 vs int8 vs int4, quant/gemm split)
ax = axes[0]
x = np.arange(len(batches))
w = 0.26
for i, b in enumerate(batches):
    r = data[b]
    fp = agg(r, "fp16_us"); i8g = agg(r, "i8_gemm_us"); i8q = agg(r, "i8_quant_us")
    i4g = agg(r, "i4_gemm_us"); i4q = agg(r, "i4_quant_us")
    ax.bar(i - w, fp, w, color=FP16)
    ax.bar(i, i8g, w, color=I8); ax.bar(i, i8q, w, bottom=i8g, color=I8Q, hatch="///", edgecolor="white")
    ax.bar(i + w, i4g, w, color=I4); ax.bar(i + w, i4q, w, bottom=i4g, color=I4Q, hatch="///", edgecolor="white")
    ax.text(i - w, fp, f"{fp:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    n8 = fp / (i8g + i8q); ax.text(i, i8g + i8q, f"{i8g+i8q:.1f}\n{n8:.2f}×", ha="center", va="bottom",
                                   fontsize=7.5, color="#1a7a1a" if n8 >= 1 else "#b00")
    n4 = fp / (i4g + i4q); ax.text(i + w, i4g + i4q, f"{i4g+i4q:.1f}\n{n4:.2f}×", ha="center", va="bottom",
                                   fontsize=7.5, color="#1a7a1a" if n4 >= 1 else "#b00")
ax.set_xticks(x); ax.set_xticklabels([f"batch {b}" for b in batches])
ax.set_ylabel("total Linear time (ms/step)")
ax.set_title("Aggregate: fp16 | int8 | int4 (dark=GEMM, hatched=quantize)\n"
             "× = speed vs fp16 (green faster / red slower)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=FP16, label="fp16 GEMM"), Patch(color=I8, label="int8 GEMM"),
                   Patch(facecolor=I8Q, hatch="///", label="int8 quantize"), Patch(color=I4, label="int4 GEMM"),
                   Patch(facecolor=I4Q, hatch="///", label="int4 quantize")], fontsize=7.5, loc="upper right")
ax.set_ylim(0, max(agg(data[b], "fp16_us") for b in batches) * 1.28)

# right: the quantize tax as % of fp16, and GEMM win, per batch -> why net flips
ax = axes[1]
for i, b in enumerate(batches):
    r = data[b]
    fp = agg(r, "fp16_us")
    win8 = (fp - agg(r, "i8_gemm_us")) / fp * 100          # GEMM saving (+ = int GEMM faster)
    tax8 = agg(r, "i8_quant_us") / fp * 100                # quantize tax
    win4 = (fp - agg(r, "i4_gemm_us")) / fp * 100
    tax4 = agg(r, "i4_quant_us") / fp * 100
    ax.bar(i - 0.19, win8, 0.36, color=I8, label="int8 GEMM saving" if i == 0 else None)
    ax.bar(i - 0.19, -tax8, 0.36, color=I8Q, hatch="///", edgecolor="white", label="int8 quantize tax" if i == 0 else None)
    ax.bar(i + 0.19, win4, 0.36, color=I4, label="int4 GEMM saving" if i == 0 else None)
    ax.bar(i + 0.19, -tax4, 0.36, color=I4Q, hatch="///", edgecolor="white", label="int4 quantize tax" if i == 0 else None)
    ax.text(i - 0.19, win8 - tax8, f"net {win8-tax8:+.0f}%", ha="center",
            va="bottom" if win8 - tax8 >= 0 else "top", fontsize=7.5, fontweight="bold")
    ax.text(i + 0.19, win4 - tax4, f"{win4-tax4:+.0f}%", ha="center",
            va="bottom" if win4 - tax4 >= 0 else "top", fontsize=7.5, fontweight="bold")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"batch {b}" for b in batches])
ax.set_ylabel("% of fp16 Linear time")
ax.set_title("Why net flips: GEMM saving (up) vs quantize tax (down)\n"
             "net = saving − tax; the quantize pass is data-scaled and never amortizes", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left", ncol=2)
save(fig, "17_linear_batch_sweep.png")
print("done")
