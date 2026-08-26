"""Derive every number and plot in REPORT.md from the source data. Nothing is hand-transcribed.

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
ax1.set_ylabel("warm-up % of a 200-step sample")
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
ax2.set_title("consistency check: measured vs predicted", fontsize=11, loc="left")
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
ax1.set_title("baseline gains >= MoDiff on 4/5 (768,2x2 both within noise)", fontsize=11, loc="left")
ax1.legend(frameon=False, fontsize=9)
despine(ax1)

ax2.bar(x - bw / 2, pdf.ovh_today, bw, color=C_GREY, label="MoDiff overhead, today", zorder=3)
ax2.bar(x + bw / 2, pdf.ovh_piped, bw, color=C_MOD, label="MoDiff overhead, pipelined", zorder=3)
for i, (a, b) in enumerate(zip(pdf.ovh_today, pdf.ovh_piped)):
    ax2.text(i - bw / 2, a + 1.0, f"{a:.0f}", ha="center", fontsize=8)
    ax2.text(i + bw / 2, b + 1.0, f"{b:.0f}", ha="center", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(pdf["shape"], rotation=25, ha="right", fontsize=8.5)
ax2.set_ylabel("MoDiff overhead vs its own baseline (%)")
ax2.set_title("the overhead ratio does not shrink (it rises on 4/5)", fontsize=11, loc="left")
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
ax1.bar(x - bw / 2, srt.sm_waste, bw, color=C_GREY, label="SM waste (idle share of last wave)", zorder=3)
ax1.bar(x + bw / 2, srt.base_gain, bw, color=C_BASE, label="baseline gain from pipelining", zorder=3)
for i, (a, b) in enumerate(zip(srt.sm_waste, srt.base_gain)):
    ax1.text(i - bw / 2, a + 1.2, f"{a:.0f}%", ha="center", fontsize=8)
    ax1.text(i + bw / 2, b + 1.2, f"{b:.1f}%", ha="center", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{s}\n{int(b)} blk\n{w:.2f} wave" for s, b, w in
                     zip(srt["shape"], srt.blocks, srt.waves)], fontsize=7.5, linespacing=1.5)
ax1.set_ylabel("%")
ax1.set_ylim(0, 84)
ax1.set_title("the gain tracks wasted SMs, not bytes moved", fontsize=11, loc="left")
ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
despine(ax1)
ax1.annotate("split penalty (87.5%)\neats this one's gain",
             xy=(0.16, 4), xytext=(0.55, 58), fontsize=7.8, color="#777777",
             ha="center", arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.9,
                                          connectionstyle="arc3,rad=0.25"))

ax2.scatter(wdf.sm_waste, wdf.base_gain, s=70, color=C_BASE, zorder=3)
for _, r in wdf.iterrows():
    ax2.annotate(r["shape"], (r.sm_waste, r.base_gain), textcoords="offset points",
                 xytext=(7, -3), fontsize=8, color=INK)
ax2.set_xlabel("conv SM waste (%)")
ax2.set_ylabel("baseline gain from pipelining (%)")
ax2.set_title("monotonic, once the split-penalty-blocked shape is set aside", fontsize=11, loc="left")
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

# (D) effect / sigma per cell -- "sd <= 1.7%" was the sd of a timing, not of the effect.
eff = {}
for k in PIPE_ORDER:
    m, s = pipe[k]["med"], pipe[k]["sd"]
    for arm in ("base", "modiff"):
        d = m[f"{arm}_full"] - m[f"{arm}_pipe"]
        # sd of a difference of two independent medians, conservatively via quadrature
        sd_d = math.sqrt(s[f"{arm}_full"] ** 2 + s[f"{arm}_pipe"] ** 2)
        eff[f"{NICE[k]}|{arm}"] = dict(effect_ms=d, sd_ms=sd_d,
                                       effect_over_sigma=(abs(d) / sd_d if sd_d else float("inf")),
                                       resolved=bool(sd_d and abs(d) / sd_d >= 3))
audit["effect_over_sigma"] = eff

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
