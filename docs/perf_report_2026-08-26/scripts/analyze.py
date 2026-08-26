"""Derive the numbers and plots in REPORT.md from the source data.

NOT everything in REPORT.md comes from here. Figures quoted from other reports (the FID block, the
gn_stats bandwidth range), source-code readings (the byte-count table), and a handful of
hand-computed values are enumerated in REPORT.md's preamble. Revisions 1 and 2 both claimed
"nothing is hand-transcribed"; that was false both times.

Sources (all committed under this report's data/, or under the two prior report folders):
  warm-up          docs/warmup1_speedup_2026-08-25/data/warmup_cost_w{1,5}.json
  a_hat/o_hat      docs/conv_block_ablation_2026-08-26/data/combined_w8a8_w4a4.csv
  pipelining       data/pipeline_result2.json
  wave occupancy   data/conv_bw_trace_{cuda_gpu_trace,nvtx_pushpop_trace}.csv  (nsys, real grids)

Writes: data/derived.json, data/*.csv, plots/*.png
"""
import csv
import json
import math
import os
from statistics import mean as st_mean, stdev as st_stdev, median as st_median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/workspace/MoDiff"
HERE = os.path.join(ROOT, "docs/perf_report_2026-08-26")
os.chdir(ROOT)

SMS = 84                    # A40 = GA102, 84 SMs
A40_PEAK_GBS = 696.0        # GDDR6 384-bit
A40_PEAK_INT8_TOPS = 299.0  # dense int8 tensor core

C_BASE = "#3B82C4"   # blue   -- baseline / W8A8
C_MOD = "#D97642"    # orange -- MoDiff  / W4A4
C_GREY = "#C7C7C7"
GRID = "#DDDDDD"
INK = "#333333"
plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.edgecolor": "#999999",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)


derived = {}

# =====================================================================
# 1. WARM-UP
# =====================================================================
w = {}
for tag in ("w1", "w5"):
    with open(f"docs/warmup1_speedup_2026-08-25/data/warmup_cost_{tag}.json") as f:
        w[tag] = json.load(f)

MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]

warm_rows = []
for key, label in MODES:
    m5, m1 = w["w5"]["modes"][key], w["w1"]["modes"][key]
    # Cross-run drift control: total minus the warm-up excess should be ~identical between the two
    # runs, since only MODIFF_WARMUP_STEPS changed. Any residual is run-to-run drift, and it bounds
    # how much of the raw total-time ratio we are allowed to attribute to the warm-up change.
    nw5 = m5["total_ms"] - m5["modiff_warmup_ms"]
    nw1 = m1["total_ms"] - m1["modiff_warmup_ms"]
    drift = 100 * (nw1 - nw5) / nw5
    # speedup predicted from the warm-up share alone (drift-free), vs the raw measured ratio
    pred = 1.0 / (1.0 - (m5["modiff_warmup_pct"] - m1["modiff_warmup_pct"]) / 100.0)
    warm_rows.append(dict(
        mode=label, step0_w5=m5["step0_ms"], step0_w1=m1["step0_ms"],
        steady_w5=m5["steady_median_ms"], steady_w1=m1["steady_median_ms"],
        warm_ms_w5=m5["modiff_warmup_ms"], warm_ms_w1=m1["modiff_warmup_ms"],
        warm_pct_w5=m5["modiff_warmup_pct"], warm_pct_w1=m1["modiff_warmup_pct"],
        total_w5=m5["total_ms"], total_w1=m1["total_ms"],
        speedup_measured=m5["total_ms"] / m1["total_ms"],
        speedup_predicted=pred, drift_pct=drift,
    ))
warm = pd.DataFrame(warm_rows)
warm.to_csv(f"{HERE}/data/warmup_summary.csv", index=False)
derived["warmup"] = warm_rows
derived["warmup_meta"] = dict(steps=w["w5"]["steps"], batch=w["w5"]["batch"], gpu=w["w5"]["gpu"])

# plot 1: warm-up share of a 200-step sample, w5 vs w1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(warm))
bw = 0.36
ax1.bar(x - bw / 2, warm.warm_pct_w5, bw, color=C_MOD, label="warm-up = 5 (shipped)", zorder=3)
ax1.bar(x + bw / 2, warm.warm_pct_w1, bw, color=C_BASE, label="warm-up = 1", zorder=3)
for i, (a, b) in enumerate(zip(warm.warm_pct_w5, warm.warm_pct_w1)):
    if a > 0.3:
        ax1.text(i - bw / 2, a + 0.06, f"{a:.2f}%", ha="center", fontsize=8.5)
        ax1.text(i + bw / 2, b + 0.06, f"{b:.2f}%", ha="center", fontsize=8.5)
ax1.set_xticks(x)
ax1.set_xticklabels(warm["mode"], rotation=20, ha="right", fontsize=8.5)
ax1.set_ylabel("warm-up % of a 200-step COLD sample")
ax1.set_title("MoDiff warm-up cost", fontsize=11, loc="left")
ax1.legend(frameon=False, fontsize=8.5)
despine(ax1)

mo = warm[warm["mode"].str.contains("MoDiff")].reset_index(drop=True)
x2 = np.arange(len(mo))
ax2.bar(x2 - bw / 2, mo.speedup_measured, bw, color=C_MOD, label="measured (total-time ratio)", zorder=3)
ax2.bar(x2 + bw / 2, mo.speedup_predicted, bw, color=C_BASE, label="predicted from warm-up share", zorder=3)
ax2.axhline(1.0, color="#888888", lw=1, ls="--", zorder=2)
for i, (a, b) in enumerate(zip(mo.speedup_measured, mo.speedup_predicted)):
    ax2.text(i - bw / 2, a + 0.002, f"{a:.3f}x", ha="center", fontsize=8.5)
    ax2.text(i + bw / 2, b + 0.002, f"{b:.3f}x", ha="center", fontsize=8.5)
ax2.set_xticks(x2)
ax2.set_xticklabels(mo["mode"], fontsize=9)
ax2.set_ylim(0.99, 1.07)
ax2.set_ylabel("e2e speedup, warm-up 5 -> 1")
ax2.set_title("W8A8 agrees (0.07pp); W4A4 does not (1.15pp = its drift)", fontsize=10.5, loc="left")
ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
despine(ax2)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/1_warmup.png", dpi=150)
plt.close(fig)

# =====================================================================
# 2. a_hat / o_hat SHARE (existing ablation data)
# =====================================================================
ab = pd.read_csv("docs/conv_block_ablation_2026-08-26/data/combined_w8a8_w4a4.csv")
order = ab[ab.precision == "W8A8"].sort_values("freq", ascending=False)["shape"].tolist()

wsum = {}
for prec in ("W8A8", "W4A4"):
    g = ab[ab.precision == prec]
    fw = g.freq.sum()
    # call-weighted by frequency AND by that shape's own time -- the honest weighting for
    # "share of the model's conv-block time", since a slow shape called once matters more than a
    # fast shape called once. Both weightings are reported; they answer different questions.
    tw = (g.modiff_total_ms * g.freq).sum()
    wsum[prec] = dict(
        a_hat_pct_freq=float((g.a_hat_pct * g.freq).sum() / fw),
        o_hat_pct_freq=float((g.o_hat_pct * g.freq).sum() / fw),
        block_ratio_freq=float((g.block_ratio * g.freq).sum() / fw),
        a_hat_pct_time=float((g.a_hat_ms * g.freq).sum() / tw * 100),
        o_hat_pct_time=float((g.o_hat_ms * g.freq).sum() / tw * 100),
        block_ratio_time=float(tw / (g.base_total_ms * g.freq).sum()),
        total_modiff_ms_per_step=float(tw),
        total_base_ms_per_step=float((g.base_total_ms * g.freq).sum()),
    )
derived["ablation_weighted"] = wsum

# plot 2: a_hat/o_hat share per shape, faceted by precision
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
xx = np.arange(len(order))
for ax, prec, ca, co in [(axes[0], "W8A8", C_BASE, "#A8C8E4"), (axes[1], "W4A4", C_MOD, "#F0BE9B")]:
    s = ab[ab.precision == prec].set_index("shape").loc[order]
    a, o = s.a_hat_pct.values, s.o_hat_pct.values
    ax.bar(xx, a, 0.62, color=ca, label="a_hat (GN+delta-quantize step)", zorder=3)
    ax.bar(xx, o, 0.62, bottom=a, color=co, label="o_hat (conv epilogue)", zorder=3)
    wa = wsum[prec]["a_hat_pct_time"]
    wo = wsum[prec]["o_hat_pct_time"]
    ax.axhline(wa, color=ca, lw=1.2, ls="--", zorder=4)
    ax.text(len(order) - 0.4, wa, f" time-weighted a_hat {wa:.1f}%", va="center", fontsize=8.5, color=ca)
    ax.set_ylabel(f"{prec}\n% of MoDiff conv-block time")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    despine(ax)
axes[0].set_title("a_hat and o_hat share of MoDiff's own conv-block time, per real shape "
                  "(shapes ordered by call frequency)", fontsize=11.5, loc="left")
axes[1].set_xticks(xx)
axes[1].set_xticklabels(order, rotation=45, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/2_ahat_ohat.png", dpi=150)
plt.close(fig)

# =====================================================================
# 3. PIPELINING (a_hat overlap attempt) -- with the baseline control
# =====================================================================
with open(f"{HERE}/data/pipeline_result2.json") as f:
    pipe = json.load(f)

PIPE_ORDER = ["768_2x2", "384_8x8", "192_32x32", "384_16x16", "768_4x4"]
NICE = {"768_2x2": "768, 2x2", "384_8x8": "384, 8x8", "192_32x32": "192, 32x32",
        "384_16x16": "384, 16x16", "768_4x4": "768, 4x4"}

prow = []
for k in PIPE_ORDER:
    d = pipe[k]
    m = d["med"]
    prow.append(dict(
        shape=NICE[k], freq=d["freq"],
        base_full=m["base_full"], base_split=m["base_split"], base_pipe=m["base_pipe"],
        modiff_full=m["modiff_full"], modiff_split=m["modiff_split"], modiff_pipe=m["modiff_pipe"],
        sd_max_pct=100 * max(d["sd"][c] / m[c] for c in m),
        ovh_today=100 * (m["modiff_full"] - m["base_full"]) / m["base_full"],
        ovh_piped=100 * (m["modiff_pipe"] - m["base_pipe"]) / m["base_pipe"],
        base_gain=100 * (m["base_full"] - m["base_pipe"]) / m["base_full"],
        modiff_gain=100 * (m["modiff_full"] - m["modiff_pipe"]) / m["modiff_full"],
        split_penalty=100 * (m["modiff_split"] - m["modiff_full"]) / m["modiff_full"],
    ))
pdf = pd.DataFrame(prow)
pdf.to_csv(f"{HERE}/data/pipeline_summary.csv", index=False)

# freq x time weighted bottom line for each arm
for arm in ("base", "modiff"):
    tot = (pdf[f"{arm}_full"] * pdf.freq).sum()
    sav = ((pdf[f"{arm}_full"] - pdf[f"{arm}_pipe"]) * pdf.freq).sum()
    derived.setdefault("pipeline_weighted", {})[arm] = dict(
        total_ms=float(tot), saved_ms=float(sav), saved_pct=float(100 * sav / tot))
derived["pipeline"] = prow

# plot 3: the refutation. Panel 3 carries the audit's correction -- the RATIO metric can rise while
# the ABSOLUTE MoDiff-specific overhead falls (768,4x4), so both are shown.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.6))
x = np.arange(len(pdf))
pdf["abs_ovh_today"] = pdf.modiff_full - pdf.base_full
pdf["abs_ovh_piped"] = pdf.modiff_pipe - pdf.base_pipe
ax1.bar(x - bw / 2, pdf.base_gain, bw, color=C_BASE, label="baseline arm", zorder=3)
ax1.bar(x + bw / 2, pdf.modiff_gain, bw, color=C_MOD, label="MoDiff arm", zorder=3)
ax1.axhline(0, color="#888888", lw=1, zorder=2)
for i, (a, b) in enumerate(zip(pdf.base_gain, pdf.modiff_gain)):
    ax1.text(i - bw / 2, a + 0.8, f"{a:.1f}", ha="center", fontsize=8)
    ax1.text(i + bw / 2, b + 0.8, f"{b:.1f}", ha="center", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(pdf["shape"], rotation=25, ha="right", fontsize=8.5)
ax1.set_ylabel("% faster from batch-split pipelining")
ax1.set_title("% gain: baseline >= MoDiff on 4/5 (in absolute ms, only 3/5)", fontsize=11, loc="left")
ax1.legend(frameon=False, fontsize=9)
despine(ax1)

ax2.bar(x - bw / 2, pdf.ovh_today, bw, color=C_GREY, label="MoDiff overhead, today", zorder=3)
ax2.bar(x + bw / 2, pdf.ovh_piped, bw, color=C_MOD, label="MoDiff overhead, pipelined", zorder=3)
for i, (a, b) in enumerate(zip(pdf.ovh_today, pdf.ovh_piped)):
    ax2.text(i - bw / 2, a + 1.0, f"{a:.1f}", ha="center", fontsize=8)
    ax2.text(i + bw / 2, b + 1.0, f"{b:.1f}", ha="center", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(pdf["shape"], rotation=25, ha="right", fontsize=8.5)
ax2.set_ylabel("MoDiff overhead vs its own baseline (%)")
_shr = pdf.loc[pdf["shape"] == "768, 2x2", "ovh_today"].iloc[0] - pdf.loc[pdf["shape"] == "768, 2x2", "ovh_piped"].iloc[0]
ax2.set_title(f"overhead ratio rises on 4/5; 768,2x2 shrinks {_shr:.2f}pp", fontsize=11, loc="left")
ax2.legend(frameon=False, fontsize=9, loc="upper left")
despine(ax2)

ax3.bar(x - bw / 2, pdf.abs_ovh_today, bw, color=C_GREY, label="absolute overhead, today", zorder=3)
ax3.bar(x + bw / 2, pdf.abs_ovh_piped, bw, color=C_MOD, label="absolute overhead, pipelined", zorder=3)
for i, (a, b) in enumerate(zip(pdf.abs_ovh_today, pdf.abs_ovh_piped)):
    ax3.text(i, max(a, b) + 0.03, f"{100*(b-a)/a:+.0f}%", ha="center", fontsize=8,
             color=("#2E7D32" if b < a else "#B23A3A"))
ax3.set_xticks(x)
ax3.set_xticklabels(pdf["shape"], rotation=25, ha="right", fontsize=8.5)
ax3.set_ylabel("MoDiff overhead, ms per 6-stage chain")
ax3.set_title("but in ABSOLUTE ms it falls on 2/5 (green)", fontsize=11, loc="left")
ax3.legend(frameon=False, fontsize=9, loc="upper left")
despine(ax3)

fig.tight_layout()
fig.savefig(f"{HERE}/plots/3_pipeline.png", dpi=150)
plt.close(fig)

# =====================================================================
# 4. WHY THE BASELINE GOT FASTER -- wave quantization, from real nsys grids
# =====================================================================
rows = list(csv.DictReader(open(f"{HERE}/data/conv_bw_trace_cuda_gpu_trace.csv")))
ranges = list(csv.DictReader(open(f"{HERE}/data/conv_bw_trace_nvtx_pushpop_trace.csv")))
cut = [r for r in rows if r["Name"].startswith("void cutlass::Kernel<modiff::ImplicitGemmConvolutionEVT")]

# nsys_conv_bandwidth.py emits one NVTX range per shape, in this order
NSYS_SHAPES = [(128, 768, 2, 2, 768), (128, 384, 8, 8, 384), (128, 192, 32, 32, 192),
               (128, 384, 16, 16, 384), (128, 768, 4, 4, 768)]
gain_by = {r["shape"]: r["base_gain"] for r in prow}

wrow = []
for (N, Cin, H, W, Cout), rng in zip(NSYS_SHAPES, ranges):
    lo, hi = int(rng["Start (ns)"]), int(rng["End (ns)"])
    ks = [r for r in cut if lo <= int(r["Start (ns)"]) <= hi]
    durs = sorted(int(r["Duration (ns)"]) for r in ks)
    med_ns = durs[len(durs) // 2]
    k = ks[0]
    gx, gy, gz = int(k["GrdX"]), int(k["GrdY"]), int(k["GrdZ"])
    blocks = gx * gy * gz
    waves = blocks / SMS
    nw = math.ceil(waves)
    eff = blocks / (SMS * nw)

    sec = med_ns * 1e-9
    in_b = N * Cin * H * W
    w_b = Cout * Cin * 9
    oh_b = N * Cout * H * W * 4           # o_hat fp16 read + write
    gbs = (in_b + w_b + oh_b) / sec / 1e9
    tops = (2 * N * Cout * Cin * 9 * H * W) / sec / 1e12

    name = f"{Cin}, {H}x{W}"
    wrow.append(dict(shape=name, grid=f"({gx},{gy},{gz})", blocks=blocks, waves=waves,
                     sm_efficiency=100 * eff, sm_waste=100 * (1 - eff),
                     dur_us=med_ns / 1e3, gbs=gbs, pct_peak_bw=100 * gbs / A40_PEAK_GBS,
                     tops=tops, pct_peak_compute=100 * tops / A40_PEAK_INT8_TOPS,
                     base_gain=gain_by[name]))
wdf = pd.DataFrame(wrow)
wdf.to_csv(f"{HERE}/data/wave_occupancy.csv", index=False)
derived["wave"] = wrow
# rank correlation between SM waste and the baseline's pipelining gain, excluding the shape whose
# gain is blocked by its own split penalty (768,2x2 -- see REPORT.md section 4)
unblocked = wdf[wdf["shape"] != "768, 2x2"]
derived["wave_spearman_excl_768_2x2"] = float(
    unblocked["sm_waste"].corr(unblocked["base_gain"], method="spearman"))
derived["wave_spearman_all"] = float(wdf["sm_waste"].corr(wdf["base_gain"], method="spearman"))

# plot 4: SM waste vs realized gain
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
srt = wdf.sort_values("sm_waste", ascending=False).reset_index(drop=True)
x = np.arange(len(srt))
ax1.bar(x - bw / 2, srt.sm_waste, bw, color=C_GREY, label="SM waste (idle SM-slots across all waves)", zorder=3)
ax1.bar(x + bw / 2, srt.base_gain, bw, color=C_BASE, label="baseline gain from pipelining", zorder=3)
for i, (a, b) in enumerate(zip(srt.sm_waste, srt.base_gain)):
    ax1.text(i - bw / 2, a + 1.2, f"{a:.0f}%", ha="center", fontsize=8)
    ax1.text(i + bw / 2, b + 1.2, f"{b:.1f}%", ha="center", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{s}\n{int(b)} blk\n{w:.2f} wave" for s, b, w in
                     zip(srt["shape"], srt.blocks, srt.waves)], fontsize=7.5, linespacing=1.5)
ax1.set_ylabel("%")
ax1.set_ylim(0, 84)
ax1.set_title("baseline arm: gain rises with wasted SMs on 4/5 (768,2x2 is the exception)",
              fontsize=10.5, loc="left")
ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
despine(ax1)
_bp = 100 * (pdf.loc[pdf["shape"] == "768, 2x2", "base_split"].iloc[0]
             - pdf.loc[pdf["shape"] == "768, 2x2", "base_full"].iloc[0]) \
      / pdf.loc[pdf["shape"] == "768, 2x2", "base_full"].iloc[0]
ax1.annotate(f"split penalty ({_bp:.1f}% this arm), and\nthe deletion is not applied to 384,8x8",
             xy=(0.16, 4), xytext=(0.55, 58), fontsize=7.8, color="#777777",
             ha="center", arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.9,
                                          connectionstyle="arc3,rad=0.25"))

ax2.scatter(wdf.sm_waste, wdf.base_gain, s=70, color=C_BASE, zorder=3)
for _, r in wdf.iterrows():
    ax2.annotate(r["shape"], (r.sm_waste, r.base_gain), textcoords="offset points",
                 xytext=(7, -3), fontsize=8, color=INK)
ax2.set_xlabel("conv SM waste (%)")
ax2.set_ylabel("baseline gain from pipelining (%)")
ax2.set_title("rho=0.00 over all 5; rho=1.00 over 4 (post-hoc, p_eff=0.14)", fontsize=10.5, loc="left")
despine(ax2)
ax2.xaxis.grid(True, color=GRID, zorder=0)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/4_wave_occupancy.png", dpi=150)
plt.close(fig)

with open(f"{HERE}/data/derived.json", "w") as f:
    json.dump(derived, f, indent=2)

# ---------------- console dump, for pasting into REPORT.md ----------------
pd.set_option("display.width", 200, "display.max_columns", 50)
print("=== 1. WARM-UP ===")
print(warm[["mode", "step0_w5", "step0_w1", "steady_w5", "warm_pct_w5", "warm_pct_w1",
            "speedup_measured", "speedup_predicted", "drift_pct"]].to_string(index=False))
print("\n=== 2. a_hat / o_hat WEIGHTED ===")
print(json.dumps(wsum, indent=2))
print("\n=== 3. PIPELINE ===")
print(pdf[["shape", "freq", "base_full", "base_pipe", "modiff_full", "modiff_pipe",
           "ovh_today", "ovh_piped", "base_gain", "modiff_gain", "split_penalty",
           "sd_max_pct"]].to_string(index=False))
print("\nweighted:", json.dumps(derived["pipeline_weighted"], indent=2))
print("\n=== 4. WAVE OCCUPANCY ===")
print(wdf[["shape", "grid", "blocks", "waves", "sm_waste", "base_gain",
           "pct_peak_bw", "pct_peak_compute"]].to_string(index=False))
print("\nspearman(sm_waste, base_gain) all =", derived["wave_spearman_all"],
      " excl 768,2x2 =", derived["wave_spearman_excl_768_2x2"])
print("\nwrote data/derived.json + 4 plots")

# =====================================================================
# 5. PLAUSIBILITY CHECKS -- computed, not asserted
# =====================================================================
checks = {}

# (a) byte-count model for a_hat. Baseline GN moves read x (2B) + re-read x (2B) + write int8 (1B)
#     = 5 B/elem. MoDiff GN adds read a_hat (2B) + write a_hat (2B) = 9 B/elem. A purely
#     bandwidth-bound kernel should therefore slow down by 9/5 = 1.80x. int4 writes a packed 4-bit
#     code (0.5 B) instead of 1 B, so its ratio is 8.5/4.5 = 1.89x.
for prec, pred in (("W8A8", 9 / 5), ("W4A4", 8.5 / 4.5)):
    g = ab[ab.precision == prec]
    meas = float((g.gn_modiff * g.freq).sum() / (g.gn_base * g.freq).sum())
    checks[f"gn_ratio_{prec}"] = dict(predicted_from_bytes=pred, measured_time_weighted=meas,
                                      rel_err_pct=100 * (meas - pred) / pred)

# (b) achieved compute can never exceed the fraction of SMs that hold work. For a shape whose N
#     extent is not a multiple of the 128-wide tile there is a second, intra-tile waste on top.
for r in wrow:
    nm = r["shape"]
    # NOTE: these 5 shapes are Cin == Cout by construction, so parsing the leading field is
    # unambiguous. Named Cin_eq_Cout to avoid implying it generalises (audit finding 25).
    Cin_eq_Cout = int(nm.split(",")[0])
    n_tiles = math.ceil(Cin_eq_Cout / 128)
    tile_fill = Cin_eq_Cout / (128 * n_tiles)
    sm_occ = r["sm_efficiency"] / 100
    ceiling = 100 * sm_occ * tile_fill
    checks.setdefault("roofline", {})[nm] = dict(
        sm_occupancy_pct=100 * sm_occ, n_tile_fill_pct=100 * tile_fill,
        compute_ceiling_pct=ceiling, achieved_compute_pct=r["pct_peak_compute"],
        within_ceiling=bool(r["pct_peak_compute"] <= ceiling + 1e-9),
        efficiency_on_busy_smem_pct=100 * r["pct_peak_compute"] / ceiling)

# (c) does the pipeline result contradict the ablation result? The ablation says MoDiff's conv block
#     costs 1.215x baseline (freq-weighted, W8A8). The pipeline run measures the same quantity as
#     modiff_full / base_full over its 5-shape subset. They use different harnesses, so agreement is
#     a real cross-check.
sub5 = ab[(ab.precision == "W8A8") & (ab["shape"].isin(
    ["768->768,2x2", "384->384,8x8", "192->192,32x32", "384->384,16x16", "768->768,4x4"]))]
abl_ratio = float((sub5.modiff_total_ms * sub5.freq).sum() / (sub5.base_total_ms * sub5.freq).sum())
pipe_ratio = float((pdf.modiff_full * pdf.freq).sum() / (pdf.base_full * pdf.freq).sum())
checks["cross_harness_block_ratio"] = dict(ablation_harness=abl_ratio, pipeline_harness=pipe_ratio,
                                           rel_diff_pct=100 * (pipe_ratio - abl_ratio) / abl_ratio)

derived["checks"] = checks
with open(f"{HERE}/data/derived.json", "w") as f:
    json.dump(derived, f, indent=2)

print("\n=== 5. PLAUSIBILITY CHECKS ===")
print(json.dumps(checks, indent=2))

# =====================================================================
# 6. AUDIT-DRIVEN CORRECTIONS (2026-08-26 independent audit)
# =====================================================================
audit = {}

# (A) o_hat's INCREMENTAL bytes are +2, not +4. Verified in source: conv2d_int8_evt_o_hat
#     (csrc/modiff/conv/conv2d_evt.cu:231) does an in-place RMW on o_hat and RETURNS o_hat -- there is
#     no separate output tensor. The baseline conv2d_int8_evt_bias_residual_fp16
#     (csrc/baseline/conv/conv2d_evt.cu:153) writes a separate fp16 `output`. So MoDiff's o_hat store
#     REPLACES a store the baseline was already paying, and only the o_hat READ is new.
audit["byte_model"] = {
    "conv_baseline_B_per_elem": 1 + 2,          # read x_int8, write fp16 out
    "conv_modiff_B_per_elem": 1 + 2 + 2,        # read x_int8, read o_hat, write o_hat
    "o_hat_incremental_B": 2,
    "gn_baseline_B_per_elem": 2 + 2 + 1,        # read x (stats), re-read x (apply), write int8
    "gn_modiff_B_per_elem": 2 + 2 + 2 + 1 + 2,  # + read a_hat, + write a_hat
    "a_hat_incremental_B": 4,
}
for prec in ("W8A8", "W4A4"):
    a_pct, o_pct = wsum[prec]["a_hat_pct_freq"], wsum[prec]["o_hat_pct_freq"]
    audit["byte_model"][f"price_per_byte_ratio_{prec}"] = (a_pct / 4) / (o_pct / 2)

# (B) The 9/5 figure is NOT a ceiling: per-shape GN ratios exceed it on most calls.
for prec, pred in (("W8A8", 9 / 5), ("W4A4", 8.5 / 4.5)):
    g = ab[ab.precision == prec].copy()
    g["r"] = g.gn_modiff / g.gn_base
    over = g[g.r > pred]
    audit[f"gn_ratio_spread_{prec}"] = dict(
        model=pred, aggregate_time_weighted=float((g.gn_modiff * g.freq).sum() / (g.gn_base * g.freq).sum()),
        per_shape_min=float(g.r.min()), per_shape_max=float(g.r.max()),
        shapes_over_model=f"{len(over)}/{len(g)}", calls_over_model=f"{int(over.freq.sum())}/{int(g.freq.sum())}")
audit["total_calls_in_ablation_freq_column"] = int(ab[ab.precision == "W8A8"].freq.sum())

# (C) ABSOLUTE MoDiff-specific overhead (ms), not the scale-free ratio. On 768,4x4 the ratio rises
#     while the absolute overhead FALLS, because the denominator shrank 31%.
abs_rows, ta, tb = [], 0.0, 0.0
for k in PIPE_ORDER:
    m, f = pipe[k]["med"], pipe[k]["freq"]
    a = m["modiff_full"] - m["base_full"]
    b = m["modiff_pipe"] - m["base_pipe"]
    ta += a * f
    tb += b * f
    abs_rows.append(dict(shape=NICE[k], freq=f, abs_ovh_today=a, abs_ovh_piped=b,
                         change_pct=100 * (b - a) / a))
audit["absolute_overhead"] = abs_rows
audit["absolute_overhead_weighted_change_pct"] = 100 * (tb - ta) / ta

# (D) effect / sigma per cell. ROUND-3 AUDIT: revisions 1-3 used the trial-to-trial standard
#     DEVIATION as the uncertainty of a median, which is a dispersion, not a standard error -- it made
#     every sigma ~1.8x too small and produced spurious "unresolved" marks. Now computed from the raw
#     per-trial timings as mean +- SE(mean), with the effect's SE by quadrature.
def _mse(v):
    return st_mean(v), st_stdev(v) / math.sqrt(len(v))

eff = {}
for k in PIPE_ORDER:
    raw = pipe[k]["raw"]
    for arm in ("base", "modiff"):
        fm, fe = _mse(raw[f"{arm}_full"])
        pm, pe = _mse(raw[f"{arm}_pipe"])
        d = fm - pm
        se = math.sqrt(fe ** 2 + pe ** 2)
        eff[f"{NICE[k]}|{arm}"] = dict(effect_ms=d, se_ms=se, t=abs(d) / se,
                                       resolved=bool(abs(d) / se >= 3))
audit["effect_over_sigma"] = eff
audit["n_cells_resolved"] = f"{sum(v['resolved'] for v in eff.values())}/{len(eff)}"

# (E) Spearman honestly: all 5 points, the n=4 subset, and the post-hoc-selection p-value.
#     Enumerating all 120 rank permutations: what fraction admit SOME single-point deletion that
#     leaves a perfectly increasing sequence? That is the real null for "delete one, get rho=1".
from itertools import permutations
n_admit = sum(
    1 for perm in permutations(range(5))
    if any(all(sub[i] < sub[i + 1] for i in range(3))
           for sub in [tuple(v for j, v in enumerate(perm) if j != drop) for drop in range(5)]))
audit["spearman"] = dict(
    rho_all_5=derived["wave_spearman_all"],
    rho_excl_768_2x2_n4=derived["wave_spearman_excl_768_2x2"],
    p_exact_if_n4_were_prespecified=1 / 24,
    p_effective_post_hoc_deletion=n_admit / 120,
    n_permutations_admitting_a_deletion=f"{n_admit}/120")
# collinearity: SM waste is a deterministic decreasing function of block count over this set, so
# "tracks wasted SMs" is not separable from "tracks how few blocks launch" at n=4.
_u = wdf[wdf["shape"] != "768, 2x2"]
audit["spearman"]["rho_neg_blocks_vs_gain_n4"] = float(
    (-_u["blocks"]).corr(_u["base_gain"], method="spearman"))
# sensitivity to the 1-block/SM premise
for bps in (2, 3):
    waste = [100 * (1 - b / (SMS * bps * math.ceil(b / (SMS * bps)))) for b in wdf["blocks"]]
    t = wdf.assign(w2=waste)
    t = t[t["shape"] != "768, 2x2"]
    audit["spearman"][f"rho_n4_if_{bps}_blocks_per_SM"] = float(
        t["w2"].corr(t["base_gain"], method="spearman"))
# the trace's own columns settle the premise
k0 = [r for r in cut][0]
audit["one_block_per_sm_evidence"] = dict(
    threads_per_block=int(k0["BlkX"]), regs_per_thread=int(k0["Reg/Trd"]),
    regs_per_block=int(k0["BlkX"]) * int(k0["Reg/Trd"]), regs_available_per_sm=65536,
    dyn_smem_MB=float(k0["DymSMem (MB)"]), smem_available_per_sm_KB=100)

# (F) end-to-end scope of the pipelining gain. The 5.16%/2.26% are shares of a 5-shape conv-block
#     subset measured in 6-stage-chain units, NOT per-step time.
steady = {"base": w["w5"]["modes"]["int8_baseline"]["steady_median_ms"],
          "modiff": w["w5"]["modes"]["int8"]["steady_median_ms"]}
for arm in ("base", "modiff"):
    saved_per_step = sum((pipe[k]["med"][f"{arm}_full"] - pipe[k]["med"][f"{arm}_pipe"]) * pipe[k]["freq"]
                         for k in PIPE_ORDER) / N_STAGES_CHAIN if (N_STAGES_CHAIN := 6) else 0
    audit.setdefault("pipeline_e2e", {})[arm] = dict(
        saved_ms_per_step=saved_per_step, steady_step_ms=steady[arm],
        pct_of_step=100 * saved_per_step / steady[arm],
        pct_of_5shape_convblock=derived["pipeline_weighted"][arm]["saved_pct"])

# (G) how much of 768,4x4's 31.1pp gain could GN-behind-conv possibly explain?
r44 = ab[(ab.precision == "W8A8") & (ab["shape"] == "768->768,4x4")].iloc[0]
audit["gn_share_of_768_4x4_pair"] = dict(
    gn_base_ms=float(r44.gn_base), pair_ms=float(r44.base_total_ms),
    gn_pct_of_pair=float(100 * r44.gn_base / r44.base_total_ms),
    measured_base_gain_pct=float(gain_by["768, 4x4"]))

# (H) a_hat/o_hat as a share of a FULL step, not just the conv block.
for prec, key in (("W8A8", "int8"), ("W4A4", "int4")):
    step = w["w5"]["modes"][key]["steady_median_ms"]
    blk = wsum[prec]["total_modiff_ms_per_step"]
    g = ab[ab.precision == prec]
    audit.setdefault("share_of_full_step", {})[prec] = dict(
        conv_block_ms=blk, step_ms=step, conv_block_pct_of_step=100 * blk / step,
        a_hat_ms=float((g.a_hat_ms * g.freq).sum()), o_hat_ms=float((g.o_hat_ms * g.freq).sum()),
        a_hat_pct_of_step=float(100 * (g.a_hat_ms * g.freq).sum() / step),
        o_hat_pct_of_step=float(100 * (g.o_hat_ms * g.freq).sum() / step))

derived["audit"] = audit
with open(f"{HERE}/data/derived.json", "w") as f:
    json.dump(derived, f, indent=2)
print("\n=== 6. AUDIT-DRIVEN CORRECTIONS ===")
print(json.dumps(audit, indent=2, default=str))

# =====================================================================
# 7. ROUND-2 AUDIT CORRECTIONS
# =====================================================================
a2 = {}
FIVE_AB = ["768->768,2x2", "384->384,8x8", "192->192,32x32", "384->384,16x16", "768->768,4x4"]
g5 = ab[(ab.precision == "W8A8") & (ab["shape"].isin(FIVE_AB))].copy()

# (A) GN's share of its own GN+conv stage, PER ARM. Revision 2 quoted 12.2% -- which is the
#     BASELINE arm's share on the single shape 768,4x4, i.e. the minimum of the set -- and used it
#     both as "the offset is 8-12% of a stage" and as a cap on what GN hiding could be worth. The
#     quantity a cap needs is the MoDiff GN's share, which is 1.4x larger.
a2["gn_share_of_stage"] = {
    r["shape"]: dict(baseline_pct=100 * r.gn_base / r.base_total_ms,
                     modiff_pct=100 * r.gn_modiff / r.modiff_total_ms)
    for _, r in g5.iterrows()}
a2["gn_share_of_stage"]["freq_weighted"] = dict(
    baseline_pct=float(100 * (g5.gn_base * g5.freq).sum() / (g5.base_total_ms * g5.freq).sum()),
    modiff_pct=float(100 * (g5.gn_modiff * g5.freq).sum() / (g5.modiff_total_ms * g5.freq).sum()))

# (B) The verdict's aggregate, with a real standard error (see (D) above on why the old one was wrong).
ta = tb = va = vb = 0.0
for k in PIPE_ORDER:
    f = pipe[k]["freq"]
    r = pipe[k]["raw"]
    am, ae = _mse(r["modiff_full"]); bm, be = _mse(r["base_full"])
    cm, ce = _mse(r["modiff_pipe"]); dm, de = _mse(r["base_pipe"])
    ta += f * (am - bm); tb += f * (cm - dm)
    va += f * f * (ae ** 2 + be ** 2); vb += f * f * (ce ** 2 + de ** 2)
a2["aggregate_abs_overhead_rise"] = dict(
    today_ms=ta, piped_ms=tb, rise_ms=tb - ta, se_ms=math.sqrt(va + vb),
    t=abs(tb - ta) / math.sqrt(va + vb), pct=100 * (tb - ta) / ta)

# (B2) ROUND-3: the correct cap on how much GN-hiding could move the verdict. The verdict metric is
#      a DIFFERENCE between arms, so the cap is the MoDiff-SPECIFIC GN increment (= a_hat), not the
#      MoDiff GN's whole share -- which includes GN work the baseline pays too. Revision 3 used 34.7%
#      (the whole share) and claimed it "exceeds MoDiff's entire overhead on any shape"; it does not.
gg = ab[(ab.precision == "W8A8") & (ab["shape"].isin(
    ["768->768,2x2", "384->384,8x8", "192->192,32x32", "384->384,16x16", "768->768,4x4"]))].copy()
a2["gn_hiding_cap"] = dict(
    per_shape={r["shape"]: dict(cap_pp=100 * r.a_hat_ms / r.base_total_ms,
                                own_overhead_pct=100 * (r.modiff_total_ms - r.base_total_ms) / r.base_total_ms)
               for _, r in gg.iterrows()},
    cap_pp_freq_wtd=float(100 * (gg.a_hat_ms * gg.freq).sum() / (gg.base_total_ms * gg.freq).sum()),
    overhead_pct_freq_wtd=float(100 * ((gg.modiff_total_ms - gg.base_total_ms) * gg.freq).sum()
                                / (gg.base_total_ms * gg.freq).sum()))
a2["gn_hiding_cap"]["pct_of_overhead_removable"] = (
    100 * a2["gn_hiding_cap"]["cap_pp_freq_wtd"] / a2["gn_hiding_cap"]["overhead_pct_freq_wtd"])

# (B3) ROUND-3: in-phase fraction, MoDiff arm, these five shapes. Revision 3 published "~70-92%",
#      which is 100 - the BASELINE GN share over the 20-shape set -- the wrong arm AND the wrong set.
_gs = a2["gn_share_of_stage"]
_v = [v["modiff_pct"] for k2, v in _gs.items() if k2 != "freq_weighted"]
a2["in_phase_pct"] = dict(low=100 - max(_v), high=100 - min(_v),
                          freq_wtd=100 - _gs["freq_weighted"]["modiff_pct"])

# (B4) ROUND-3: is 768,4x4's arm asymmetry real? Revision 3 called it 2.56 sigma and hedged. With a
#      proper SE it is resolved, so the MoDiff-specific share of that shape's gain IS established.
_r = pipe["768_4x4"]["raw"]
_b = [f - q for f, q in zip(_r["base_full"], _r["base_pipe"])]
_m = [f - q for f, q in zip(_r["modiff_full"], _r["modiff_pipe"])]
_d = st_mean(_m) - st_mean(_b)
_se = math.sqrt((st_stdev(_m) / math.sqrt(len(_m))) ** 2 + (st_stdev(_b) / math.sqrt(len(_b))) ** 2)
a2["arm_asymmetry_768_4x4"] = dict(modiff_saves_ms=st_mean(_m), base_saves_ms=st_mean(_b),
                                   diff_ms=_d, se_ms=_se, t=abs(_d) / _se,
                                   modiff_specific_pct_of_gain=100 * _d / st_mean(_m))

# (C) Which arm saves more, in ABSOLUTE ms rather than percent. The ratio metric says 4/5; absolute
#     ms -- the column the report itself calls the honest one -- says 3/5.
cnt = 0
det = {}
for k in PIPE_ORDER:
    m = pipe[k]["med"]
    b = m["base_full"] - m["base_pipe"]
    d = m["modiff_full"] - m["modiff_pipe"]
    cnt += b > d
    det[NICE[k]] = dict(base_saves_ms=b, modiff_saves_ms=d, baseline_saves_more=bool(b > d))
a2["baseline_saves_more_abs"] = dict(count=f"{cnt}/5", detail=det)

# (D) Per-byte price ratio: element counts are NOT equal (a_hat lives on Cin, o_hat on Cout), so
#     dividing percentages by per-element byte counts is biased. Report the range, not a point.
for prec in ("W8A8", "W4A4"):
    q = ab[ab.precision == prec]
    e_in = float((q.Cin * q.H * q.W * q.freq).sum())
    e_out = float((q.Cout * q.H * q.W * q.freq).sum())
    a_tot = float((q.a_hat_ms * q.freq).sum())
    o_tot = float((q.o_hat_ms * q.freq).sum())
    a2.setdefault("per_byte_price", {})[prec] = dict(
        ahat_over_ohat_elements=e_in / e_out,
        incremental_byte_ratio=(4 * e_in) / (2 * e_out),
        from_freq_weighting=(wsum[prec]["a_hat_pct_freq"] / 4) / (wsum[prec]["o_hat_pct_freq"] / 2),
        from_time_weighting=(wsum[prec]["a_hat_pct_time"] / 4) / (wsum[prec]["o_hat_pct_time"] / 2),
        from_totals_element_corrected=(a_tot / (4 * e_in)) / (o_tot / (2 * e_out)))

# (E) Warm-up share at 50 steps, on THIS report's own definition (share of the sample).
for key, tag in (("int8", "W8A8"), ("int4", "W4A4")):
    m5 = w["w5"]["modes"][key]
    a2.setdefault("warmup_share_at_50_steps", {})[tag] = \
        100 * m5["modiff_warmup_ms"] / (50 * m5["steady_median_ms"] + m5["modiff_warmup_ms"])

# (F) Subset criterion, stated correctly: Cin == Cout is 9 of 20 shapes; the 5 used also need freq>=7.
q8 = ab[ab.precision == "W8A8"]
eq = q8[q8.Cin == q8.Cout]
a2["subset"] = dict(cin_eq_cout_shapes=len(eq), cin_eq_cout_calls=int(eq.freq.sum()),
                    used_shapes=5, used_calls=int(g5.freq.sum()),
                    dropped_freq1=sorted(set(eq["shape"]) - set(FIVE_AB)))

# (G) The low-ratio shapes that carry the aggregate: there are FOUR 32x32 shapes, not three, and
#     they are the four lowest ratios. Weight is by the denominator (baseline GN time).
q8 = q8.copy()
q8["r"] = q8.gn_modiff / q8.gn_base
h = q8[q8.H == 32]
a2["low_ratio_shapes"] = dict(
    n_32x32=len(h), ratios=sorted(h.r.round(3).tolist()),
    are_the_n_lowest=bool(set(h.r.nsmallest(4)) == set(q8.r.nsmallest(4))),
    share_of_base_gn_time=float(100 * (h.gn_base * h.freq).sum() / (q8.gn_base * q8.freq).sum()),
    share_of_modiff_gn_time=float(100 * (h.gn_modiff * h.freq).sum() / (q8.gn_modiff * q8.freq).sum()))

# (H) Negative-control spread, labelled correctly (revision 2 said "+-2.6 ms", which is the largest
#     single magnitude, not the spread).
exc = [w["w5"]["modes"][k]["excess_ms"] for k in ("fp16", "int8_baseline", "int4_baseline")]
a2["control_excess"] = dict(values_ms=exc, spread_ms=max(exc) - min(exc),
                            pct_of_sample=100 * (max(exc) - min(exc)) / w["w5"]["modes"]["int8"]["total_ms"])

# (I) Two-sided p for the post-hoc deletion (revision 2's 0.142 is one-sided).
from itertools import permutations as _pm
def _lis_ge4(perm, inc=True):
    return any(all((sub[i] < sub[i + 1]) if inc else (sub[i] > sub[i + 1]) for i in range(3))
               for sub in [tuple(v for j, v in enumerate(perm) if j != d) for d in range(5)])
_all = list(_pm(range(5)))
a2["p_post_hoc"] = dict(one_sided=sum(_lis_ge4(p_) for p_ in _all) / 120,
                        two_sided=sum(_lis_ge4(p_) or _lis_ge4(p_, False) for p_ in _all) / 120)

# (J) Cross-harness agreement on ABSOLUTE ms/step for the 5 shapes (revision 2 published only the
#     ratio-level check). This is the missing validity argument for the e2e figures.
for arm, col in (("base", "base_total_ms"), ("modiff", "modiff_total_ms")):
    pipe_ms = sum(pipe[k]["med"][f"{arm}_full"] * pipe[k]["freq"] for k in PIPE_ORDER) / 6
    abl_ms = float((g5[col] * g5.freq).sum())
    a2.setdefault("cross_harness_absolute", {})[arm] = dict(
        pipeline_ms_per_step=pipe_ms, ablation_ms_per_step=abl_ms,
        rel_diff_pct=100 * (pipe_ms - abl_ms) / abl_ms)

# (K) Share-of-step under a CONSISTENT weighting (revision 2 mixed freq and time weightings).
for prec, key in (("W8A8", "int8"), ("W4A4", "int4")):
    step = w["w5"]["modes"][key]["steady_median_ms"]
    q = ab[ab.precision == prec]
    blk = float((q.modiff_total_ms * q.freq).sum())
    a2.setdefault("share_of_step_freq_weighted", {})[prec] = dict(
        a_hat_pct=wsum[prec]["a_hat_pct_freq"] * blk / step,
        o_hat_pct=wsum[prec]["o_hat_pct_freq"] * blk / step)

# (L) per-shape absolute cross-harness spread, not just the aggregate (round-3 minor 18)
a2["cross_harness_absolute_per_shape"] = {
    NICE[k]: 100 * ((pipe[k]["med"]["modiff_full"] / 6)
                    - float(gg[gg["shape"] == sh].modiff_total_ms.iloc[0]))
             / float(gg[gg["shape"] == sh].modiff_total_ms.iloc[0])
    for k, sh in zip(PIPE_ORDER, ["768->768,2x2", "384->384,8x8", "192->192,32x32",
                                  "384->384,16x16", "768->768,4x4"])}

derived["audit_round2"] = a2
with open(f"{HERE}/data/derived.json", "w") as f:
    json.dump(derived, f, indent=2)
print("\n=== 7. ROUND-2 CORRECTIONS ===")
print(json.dumps(a2, indent=2, default=str))

# =====================================================================
# 8. GENERATE THE VOLATILE TABLES DIRECTLY INTO REPORT.md
#
# Rounds 1-4 each shipped numbers that were correct when typed and stale after the next re-run;
# round 4 found ~15 of them in section 4 alone. Hand-typing volatile figures is the recurring bug,
# so the tables below are written into REPORT.md between markers and can no longer drift.
# =====================================================================
import re
from scipy import stats as _sps

RAW = {k: pipe[k]["raw"] for k in PIPE_ORDER}
PRETTY = {"768, 2x2": "768, 2×2", "384, 8x8": "384, 8×8", "192, 32x32": "192, 32×32",
          "384, 16x16": "384, 16×16", "768, 4x4": "768, 4×4"}


def paired_t(a, b):
    """Paired t on 5 trials. bench_pipeline2.py appends one timing per config per trial, so the
    trials ARE paired by index; revisions 1-4 combined independent SEs by quadrature instead."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    t = st_mean(d) / (st_stdev(d) / math.sqrt(n))
    pv = 2 * (1 - _sps.t.cdf(abs(t), n - 1))
    sig = abs(_sps.norm.ppf(pv / 2)) if pv > 0 else float("inf")
    return st_mean(d), t, pv, sig


blocks = {}

# --- §3 main table ---
L = ["| shape | freq | base gain | MoDiff gain | ovh today | ovh piped | abs ovh (ms) | split pen. |",
     "|---|---|---|---|---|---|---|---|"]
ta = tb = 0.0
for k in PIPE_ORDER:
    m, f = pipe[k]["med"], pipe[k]["freq"]
    bg = 100 * (m["base_full"] - m["base_pipe"]) / m["base_full"]
    mg = 100 * (m["modiff_full"] - m["modiff_pipe"]) / m["modiff_full"]
    o1 = 100 * (m["modiff_full"] - m["base_full"]) / m["base_full"]
    o2 = 100 * (m["modiff_pipe"] - m["base_pipe"]) / m["base_pipe"]
    a = m["modiff_full"] - m["base_full"]
    b = m["modiff_pipe"] - m["base_pipe"]
    sp = 100 * (m["modiff_split"] - m["modiff_full"]) / m["modiff_full"]
    ta += a * f
    tb += b * f
    L.append(f"| {PRETTY[NICE[k]]} | {f} | {bg:.1f}% | {mg:.1f}% | {o1:.1f}% | {o2:.1f}% "
             f"{'↓' if o2 < o1 else '↑'} | {a:.3f} → {b:.3f} (**{100*(b-a)/a:+.1f}%**) | {sp:.1f}% |")
aggm = [sum(pipe[k]["freq"] * (RAW[k]["modiff_full"][i] - RAW[k]["base_full"][i]) for k in PIPE_ORDER)
        for i in range(5)]
aggp = [sum(pipe[k]["freq"] * (RAW[k]["modiff_pipe"][i] - RAW[k]["base_pipe"][i]) for k in PIPE_ORDER)
        for i in range(5)]
ad, at, ap, asig = paired_t(aggp, aggm)
L.append(f"| **freq-wtd** | | | | | | **{st_mean(aggm):.2f} → {st_mean(aggp):.2f} "
         f"({100*ad/st_mean(aggm):+.1f}%)** | |")
blocks["S3_MAIN"] = "\n".join(L)
a2["verdict_aggregate"] = dict(rise_ms=ad, t=at, p=ap, sigma_equiv=asig,
                               pct=100 * ad / st_mean(aggm), df=4)

# --- §3 significance table: BOTH the per-arm effect and the arm DIFFERENCE (what the verdict needs) ---
L = ["| shape | base arm *t* | MoDiff arm *t* | **arm difference** (MoDiff − base saving) | *t* | *p* |",
     "|---|---|---|---|---|---|"]
diffs = {}
for k in PIPE_ORDER:
    r = RAW[k]
    _, tb_, pb_, _ = paired_t(r["base_full"], r["base_pipe"])
    _, tm_, pm_, _ = paired_t(r["modiff_full"], r["modiff_pipe"])
    bs = [f - q for f, q in zip(r["base_full"], r["base_pipe"])]
    msv = [f - q for f, q in zip(r["modiff_full"], r["modiff_pipe"])]
    dd, dt, dp, dsig = paired_t(msv, bs)
    diffs[NICE[k]] = dict(diff_ms=dd, t=dt, p=dp, sigma_equiv=dsig)
    L.append(f"| {PRETTY[NICE[k]]} | {tb_:.1f} | {tm_:.1f} | {dd:+.4f} ms | {dt:.1f} | "
             f"{dp:.4f}{'' if dp < 0.05 else ' **n.s.**'} |")
aggd = sum(pipe[k]["freq"] * diffs[NICE[k]]["diff_ms"] for k in PIPE_ORDER)
L.append(f"| **freq-wtd aggregate** | | | **{aggd:+.3f} ms** | **{at:.1f}** | **{ap:.5f}** |")
blocks["S3_STATS"] = "\n".join(L)
a2["arm_difference_per_shape"] = diffs

# --- §3 phase / cap table, derived from run_pipe's actual schedule ---
L = ["| | " + " | ".join(PRETTY[NICE[k]] for k in PIPE_ORDER) + " | freq-wtd |", "|---|" + "---|" * 6]
gg2 = {r["shape"]: r for _, r in gg.iterrows()}
SH = {"768_2x2": "768->768,2x2", "384_8x8": "384->384,8x8", "192_32x32": "192->192,32x32",
      "384_16x16": "384->384,16x16", "768_4x4": "768->768,4x4"}
sw = float((gg.gn_modiff * gg.freq).sum() / (gg.modiff_total_ms * gg.freq).sum())
L.append("| MoDiff GN, % of its stage | " + " | ".join(
    f"{100*gg2[SH[k]].gn_modiff/gg2[SH[k]].modiff_total_ms:.1f}" for k in PIPE_ORDER)
    + f" | {100*sw:.1f} |")
L.append("| conv / GN | " + " | ".join(
    f"{gg2[SH[k]].conv_modiff/gg2[SH[k]].gn_modiff:.2f}" for k in PIPE_ORDER) + " | — |")
# ROUND-5 AUDIT: revision 5 hard-coded "100%" and used 2*s (the N->infinity limit) for a 6-stage
# chain. Correct: stream 2 waits on ev_offset after stream 1's FIRST GN, so that GN runs solo; of the
# 2N GNs per chain, 2N-1 are inside a conv. Paired wall time = (2N-1)*g against a wall of g+N(g+c),
# i.e. (2N-1)s/(N+s) with s = g/(g+c) -- 11s/(6+s) at N=6.
NST = 6
_BLK = {'768_2x2': 24, '384_8x8': 192, '192_32x32': 2048, '384_16x16': 768, '768_4x4': 96}
L.append(f"| GN launches inside a conv (of {2*NST}) | " + " | ".join(
    f"{2*NST-1}/{2*NST}" for _ in PIPE_ORDER) + f" | **{100*(2*NST-1)/(2*NST):.1f}%** |")
_ps = {}
for k in PIPE_ORDER:
    _s = gg2[SH[k]].gn_modiff / gg2[SH[k]].modiff_total_ms
    _ps[k] = 100 * (2 * NST - 1) * _s / (NST + _s)
# Wall-time weighted -- Sum(f*11g) / Sum(f*(g+6(g+c))) -- which is algebraically 11s/(6+s) on the
# time-weighted s printed in the row above. Revision 6 published a call-weighted MEAN of the
# per-shape percentages (53.3%), which corresponds to a different s than the one shown.
_pw = 100 * sum(pipe[k]["freq"] * 11 * gg2[SH[k]].gn_modiff for k in PIPE_ORDER) / sum(
    pipe[k]["freq"] * (gg2[SH[k]].gn_modiff + 6 * gg2[SH[k]].modiff_total_ms) for k in PIPE_ORDER)
L.append("| paired share of wall time, N=6 | " + " | ".join(f"{_ps[k]:.1f}%" for k in PIPE_ORDER)
         + f" | {_pw:.1f}% |")
# MEASURED counterpart: if co-execution happened as modelled, pipe/split ~ (N+s)/2N ~ 0.53.
L.append("| **measured** pipe / split | " + " | ".join(
    f"{pipe[k]['med']['modiff_pipe']/pipe[k]['med']['modiff_split']:.3f}" for k in PIPE_ORDER)
    + " | — |")
L.append("| measured pipe / split, **baseline** arm | " + " | ".join(
    f"{pipe[k]['med']['base_pipe']/pipe[k]['med']['base_split']:.3f}" for k in PIPE_ORDER) + " | — |")
# Work-conservation floor: pipelining can at best match the un-split full-batch run, so full/split --
# not the capacity-free model -- is the honest comparison. Below the floor = real concurrency gain.
L.append("| work-conservation floor (full / split) | " + " | ".join(
    f"{pipe[k]['med']['modiff_full']/pipe[k]['med']['modiff_split']:.3f}" for k in PIPE_ORDER) + " | — |")
L.append("| conv blocks (from §4) | " + " | ".join(
    str(_BLK[k]) for k in PIPE_ORDER) + " | — |")
blocks["S3_PHASE"] = "\n".join(L)
a2["phase_model"] = dict(
    c_gt_g_all_shapes=bool((gg.conv_modiff > gg.gn_modiff).all()),
    gn_inside_conv_fraction=(2 * 6 - 1) / (2 * 6),
    paired_wall_pct_per_shape={NICE[k]: _ps[k] for k in PIPE_ORDER},
    paired_wall_pct_call_wtd=_pw,
    measured_pipe_over_split={NICE[k]: pipe[k]["med"]["modiff_pipe"] / pipe[k]["med"]["modiff_split"]
                              for k in PIPE_ORDER},
    # c > g must survive halving the batch, since g and c above are batch-128 ablation durations.
    # Conv scales by wave count, GN linearly.
    half_batch_conv_over_gn={
        NICE[k]: float((gg2[SH[k]].conv_modiff
                        * (math.ceil((b / 2) / SMS) / math.ceil(b / SMS)))
                       / (gg2[SH[k]].gn_modiff * 0.5))
        for k, b in zip(PIPE_ORDER, [24, 192, 2048, 768, 96])},
    # the cap's second endpoint, previously hand-computed and undisclosed
    cap_pct_of_pipeline_harness_overhead=float(
        100 * a2["gn_hiding_cap"]["cap_pp_freq_wtd"]
        / (100 * (sum(pipe[k]["med"]["modiff_full"] * pipe[k]["freq"] for k in PIPE_ORDER)
                  - sum(pipe[k]["med"]["base_full"] * pipe[k]["freq"] for k in PIPE_ORDER))
           / sum(pipe[k]["med"]["base_full"] * pipe[k]["freq"] for k in PIPE_ORDER))),
    spearman_blocks_vs_pipe_split_modiff=float(_sps.spearmanr(
        [_BLK[k] for k in PIPE_ORDER],
        [pipe[k]["med"]["modiff_pipe"] / pipe[k]["med"]["modiff_split"] for k in PIPE_ORDER]).statistic),
    spearman_blocks_vs_pipe_split_base=float(_sps.spearmanr(
        [_BLK[k] for k in PIPE_ORDER],
        [pipe[k]["med"]["base_pipe"] / pipe[k]["med"]["base_split"] for k in PIPE_ORDER]).statistic),
    max_arm_gap_in_pipe_split=float(max(
        abs(pipe[k]["med"]["modiff_pipe"] / pipe[k]["med"]["modiff_split"]
            - pipe[k]["med"]["base_pipe"] / pipe[k]["med"]["base_split"]) for k in PIPE_ORDER)),
    work_conservation_floor={NICE[k]: pipe[k]["med"]["modiff_full"] / pipe[k]["med"]["modiff_split"]
                             for k in PIPE_ORDER},
    identity_residual=float((gg.a_hat_ms + gg.o_hat_ms
                             - (gg.modiff_total_ms - gg.base_total_ms)).abs().max()))
# retired statistics: keep them out of derived.json so they cannot be re-quoted
audit.pop("n_cells_resolved", None)
a2.pop("in_phase_pct", None)
audit.pop("effect_over_sigma", None)      # retired unpaired-quadrature t values
a2.get("arm_asymmetry_768_4x4", {}).pop("t", None)   # superseded by the paired t in S3_STATS

# --- §4 occupancy table (base_gain must track the CURRENT run) ---
L = ["| shape | real grid | blocks | waves | last wave full | SM waste | base gain |",
     "|---|---|---|---|---|---|---|"]
for r in sorted(wrow, key=lambda z: -z["sm_waste"]):
    star = " ✳" if r["shape"] == "768, 2x2" else ""
    L.append(f"| {PRETTY[r['shape']]} | {r['grid']} | {r['blocks']} | {r['waves']:.2f} | "
             f"{100*(r['blocks']-SMS*(math.ceil(r['waves'])-1))/SMS:.0f}% | "
             f"**{r['sm_waste']:.1f}%** | {r['base_gain']:.1f}%{star} |")
blocks["S4_WAVE"] = "\n".join(L)

# --- §4 bottom line ---
L = ["| arm | saved | of the 5-shape conv block | of a full step (e2e) |", "|---|---|---|---|"]
for arm, lab in (("base", "baseline"), ("modiff", "MoDiff")):
    e = audit["pipeline_e2e"][arm]
    L.append(f"| {lab} | {e['saved_ms_per_step']:.3f} ms/step | "
             f"{e['pct_of_5shape_convblock']:.2f}% | **{e['pct_of_step']:.2f}%** |")
blocks["S4_BOTTOM"] = "\n".join(L)

rp = f"{HERE}/REPORT.md"
txt = open(rp).read()
for tag, body in blocks.items():
    pat = re.compile(rf"(<!-- GEN:{tag} -->\n).*?(\n<!-- /GEN:{tag} -->)", re.S)
    if pat.search(txt):
        txt = pat.sub(lambda m: m.group(1) + body + m.group(2), txt)
    else:
        print(f"  WARNING: marker GEN:{tag} not found in REPORT.md")
open(rp, "w").write(txt)

# (M) position confound, quantified. The rotation gives PARTIAL protection: each config occupies 5 of
#     6 positions, leaving a net tilt on the arm-difference metric. Fit the drift and sign the bias.
_NM = ["base_full", "base_split", "base_pipe", "modiff_full", "modiff_split", "modiff_pipe"]
_xs, _ys = [], []
for k in PIPE_ORDER:
    for t in range(5):
        rot = _NM[t % 6:] + _NM[:t % 6]
        for pos, nm in enumerate(rot):
            med = st_median(pipe[k]["raw"][nm])
            _xs.append(pos)
            _ys.append(100 * (pipe[k]["raw"][nm][t] - med) / med)
_fit = _sps.linregress(_xs, _ys)
_mp = {nm: st_mean([(_NM[t % 6:] + _NM[:t % 6]).index(nm) for t in range(5)]) for nm in _NM}
_tilt = (_mp["modiff_pipe"] - _mp["modiff_full"]) - (_mp["base_pipe"] - _mp["base_full"])
a2["position_confound"] = dict(
    drift_pct_per_position=_fit.slope, p=_fit.pvalue, mean_positions=_mp,
    net_tilt_positions=_tilt,
    bias_on_arm_difference_ms=-_tilt * (_fit.slope / 100) * st_mean(aggm),
    effect_ms=ad,
    bias_pct_of_effect=abs(_tilt * (_fit.slope / 100) * st_mean(aggm) / ad) * 100)

derived["audit_round2"] = a2
with open(f"{HERE}/data/derived.json", "w") as f:
    json.dump(derived, f, indent=2)
print("\n=== 8. GENERATED TABLES ===")
print("injected:", ", ".join(blocks))
print(f"verdict aggregate: {ad:+.3f} ms, t={at:.2f} (df=4), p={ap:.5f}, sigma-equiv={asig:.2f}")
print(f"phase: c>g on all 5; {2*NST-1}/{2*NST} GNs inside a conv; paired wall {_pw:.1f}% (call-wtd); "
      f"measured pipe/split {min(a2['phase_model']['measured_pipe_over_split'].values()):.3f}"
      f"-{max(a2['phase_model']['measured_pipe_over_split'].values()):.3f} vs modelled ~0.53")
