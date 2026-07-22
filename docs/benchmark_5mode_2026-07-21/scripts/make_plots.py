"""Generate every report figure from data/*.csv -> figs/*.png. Skips a figure if its CSV is missing.
matplotlib Agg, 150 dpi. CVD-safe categorical palette (Wong): fp16 blue / int8 orange / int4 green;
modiff = same hue + hatch (texture = secondary encoding so identity is never colour-alone). Legends
always present for >=2 series; selective direct value labels; single y-axis; log-y where ranges span
orders of magnitude."""
import os, csv
os.chdir("/workspace/MoDiff")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = "docs/benchmark_5mode_2026-07-21"
D, Fg = f"{HERE}/data", f"{HERE}/figs"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
LBL = {"fp16": "fp16", "int8_baseline": "int8 base", "int4_baseline": "int4 base",
       "int8_modiff": "int8 modiff", "int4_modiff": "int4 modiff"}
COL = {"fp16": "#0072B2", "int8_baseline": "#E69F00", "int4_baseline": "#009E73",
       "int8_modiff": "#E69F00", "int4_modiff": "#009E73"}
HATCH = {"int8_modiff": "///", "int4_modiff": "///"}
C3 = {"fp16": "#0072B2", "int8": "#E69F00", "int4": "#009E73"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.facecolor": "white"})


def load(name):
    p = f"{D}/{name}"
    if not os.path.exists(p):
        print(f"skip (missing {name})"); return None
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return float("nan")


def bars_labels(ax, bars, vals, fmt="{:.0f}", fs=8, rot=0):
    for b, v in zip(bars, vals):
        if v == v:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=fs, rotation=rot)


# ---------- 1. e2e speed ----------
r = load("e2e_speed.csv")
if r:
    by = {row["mode"]: row for row in r}; modes = [m for m in MODES if m in by]
    ms = [fnum(by[m]["ms_step"]) for m in modes]; sp = [fnum(by[m]["speedup_vs_fp16"]) for m in modes]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    b0 = ax[0].bar(range(len(modes)), ms, color=[COL[m] for m in modes],
                   hatch=[HATCH.get(m, "") for m in modes], edgecolor="white", lw=1.2)
    ax[0].set_ylabel("ms / step"); ax[0].set_title("E2E DDIM step time (b128, 200 steps)")
    bars_labels(ax[0], b0, ms, "{:.1f}", 9)
    b1 = ax[1].bar(range(len(modes)), sp, color=[COL[m] for m in modes],
                   hatch=[HATCH.get(m, "") for m in modes], edgecolor="white", lw=1.2)
    ax[1].axhline(1.0, color="#888", ls="--", lw=1); ax[1].set_ylabel("speedup vs fp16")
    ax[1].set_title("E2E speedup vs true fp16"); bars_labels(ax[1], b1, sp, "{:.2f}x", 9)
    for a in ax:
        a.set_xticks(range(len(modes))); a.set_xticklabels([LBL[m] for m in modes], rotation=15)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_e2e_speed.png", dpi=150); plt.close(fig)
    print("wrote fig_e2e_speed.png")

# ---------- 2. e2e timing profile (stacked) ----------
r = load("e2e_timing_profile.csv")
if r:
    by = {row["mode"]: row for row in r}; modes = [m for m in MODES if m in by]
    buckets = [k for k in r[0].keys() if k not in ("mode", "gpu_busy", "wall")]
    palette = plt.cm.tab20(np.linspace(0, 1, len(buckets)))
    fig, ax = plt.subplots(figsize=(11.5, 6))
    bottom = np.zeros(len(modes))
    for bi, bk in enumerate(buckets):
        vals = np.array([fnum(by[m][bk]) for m in modes])
        ax.bar(range(len(modes)), vals, bottom=bottom, label=bk, color=palette[bi], edgecolor="white", lw=0.5)
        bottom += np.nan_to_num(vals)
    walls = [fnum(by[m]["wall"]) for m in modes]
    ax.plot(range(len(modes)), walls, "kD", ms=8, label="wall (indep.)")
    for i, w in enumerate(walls):
        ax.text(i, w, f" {w:.0f}", ha="left", va="center", fontsize=8)
    ax.set_ylabel("ms / step (GPU self time)"); ax.set_title("E2E per-component timing profile (b128, measured)")
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([LBL[m] for m in modes], rotation=15)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_e2e_timing_profile.png", dpi=150); plt.close(fig)
    print("wrote fig_e2e_timing_profile.png")

# ---------- 3. conv kernel: per-call (quant-eligible) + per-step total ----------
r = load("conv_kernel_speed.csv")
if r:
    body = [row for row in r if row["Cin"] != "TOTAL_PER_STEP"]
    tot = next((row for row in r if row["Cin"] == "TOTAL_PER_STEP"), None)
    qe = [row for row in body if row["quant_eligible"] == "1"]
    qe = sorted(qe, key=lambda x: -fnum(x["fp16_us"]))
    labels = [f"{row['Cin']}>{row['Cout']}\n{row['H']}²×{row['count_per_step']}" for row in qe]
    x = np.arange(len(qe)); w = 0.16
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [3, 1]})
    for i, m in enumerate(MODES):
        vals = [fnum(row[f"{m}_us"]) for row in qe]
        ax[0].bar(x + (i - 2) * w, vals, w, label=LBL[m], color=COL[m],
                  hatch=HATCH.get(m, ""), edgecolor="white", lw=0.3)
    ax[0].set_yscale("log"); ax[0].set_ylabel("µs / call (log)")
    ax[0].set_title("Conv kernel time per quant-eligible geometry (b128)  ·  label = Cin>Cout, HW², ×count/step")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=0, fontsize=6.5); ax[0].legend(fontsize=8, ncol=2)
    if tot:
        pv = [fnum(tot[f"{m}_us_per_step"]) / 1000 for m in MODES]
        b = ax[1].bar(range(len(MODES)), pv, color=[COL[m] for m in MODES],
                      hatch=[HATCH.get(m, "") for m in MODES], edgecolor="white", lw=1.2)
        bars_labels(ax[1], b, pv, "{:.1f}", 9)
        ax[1].set_ylabel("ms / step"); ax[1].set_title("Conv total per step\n(all 89 convs)")
        ax[1].set_xticks(range(len(MODES))); ax[1].set_xticklabels([LBL[m] for m in MODES], rotation=25, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_conv_kernel.png", dpi=150); plt.close(fig)
    print("wrote fig_conv_kernel.png")

# ---------- 4. linear kernel: per-shape (qkv/proj) + per-step total ----------
r = load("linear_kernel_speed.csv")
if r:
    body = [row for row in r if row["role"] in ("qkv", "proj")]
    tot = next((row for row in r if row["role"] == "TOTAL_PER_STEP"), None)
    labels = [f"{row['role']}\n{row['K']}>{row['N']}\nM{int(fnum(row['M']))//1000}k×{row['count_per_step']}" for row in body]
    x = np.arange(len(body)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [3, 1]})
    series = [("fp16_us", "fp16", C3["fp16"]), ("int8_full_us", "int8 (quant+GEMM)", C3["int8"]),
              ("int4_full_us", "int4 (quant+GEMM)", C3["int4"])]
    for i, (key, lab, c) in enumerate(series):
        ax[0].bar(x + (i - 1) * w, [fnum(row[key]) for row in body], w, label=lab, color=c, edgecolor="white", lw=0.3)
    ax[0].set_yscale("log"); ax[0].set_ylabel("µs / call (log)")
    ax[0].set_title("Linear qkv/proj full-forward time per shape (b128)  ·  quantize NOT fused into GN (MODIFF_FUSE_GN_QKV=0)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=6.5); ax[0].legend(fontsize=8)
    if tot:
        pv = [fnum(tot["fp16_us"]) / 1000, fnum(tot["int8_full_us"]) / 1000, fnum(tot["int8_gemmonly_us"]) / 1000,
              fnum(tot["int4_full_us"]) / 1000, fnum(tot["int4_gemmonly_us"]) / 1000]
        cc = [C3["fp16"], C3["int8"], C3["int8"], C3["int4"], C3["int4"]]
        hh = ["", "", "///", "", "///"]
        b = ax[1].bar(range(5), pv, color=cc, hatch=hh, edgecolor="white", lw=1.2)
        bars_labels(ax[1], b, pv, "{:.1f}", 8)
        ax[1].set_ylabel("ms / step")
        ax[1].set_title("Linear total per step (79)\nfull=quant+GEMM · hatch=GEMM-only (if fused)")
        ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["fp16", "i8\nfull", "i8\ngemm", "i4\nfull", "i4\ngemm"], fontsize=8)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_linear_kernel.png", dpi=150); plt.close(fig)
    print("wrote fig_linear_kernel.png")

# ---------- 5. attention kernel: stacked GN+quant+attn per shape + per-step ----------
r = load("attn_kernel_speed.csv")
if r:
    body = [row for row in r if row["C"] != "TOTAL_PER_STEP"]
    tot = next((row for row in r if row["C"] == "TOTAL_PER_STEP"), None)
    groups = [f"{row['C']}/hd{row['hd']}\nT{row['T']}×{row['count_per_step']}" for row in body]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [3, 1]})
    sub = [("fp16", "attn16_us", None, C3["fp16"]), ("int8", "attn8_us", "q8_us", C3["int8"]),
           ("int4", "attn4_us", "q4_us", C3["int4"])]
    for i, (lab, akey, qkey, col) in enumerate(sub):
        gn = np.array([fnum(row["gn_us"]) for row in body])
        q = np.array([fnum(row[qkey]) if (qkey and row[qkey]) else 0.0 for row in body])
        a = np.array([fnum(row[akey]) if row[akey] else fnum(row["attn16_us"]) for row in body])
        pos = x + (i - 1) * w
        ax[0].bar(pos, gn, w, color="#999999", edgecolor="white", lw=0.3, label="GroupNorm" if i == 0 else None)
        ax[0].bar(pos, q, w, bottom=gn, color="#CC79A7", edgecolor="white", lw=0.3, label="quantize" if i == 1 else None)
        ax[0].bar(pos, a, w, bottom=gn + q, color=col, edgecolor="white", lw=0.3, label=f"{lab} attn")
    ax[0].set_yscale("log"); ax[0].set_ylabel("µs / call (log)")
    ax[0].set_title("Attention block WITH norm: GroupNorm + quantize + attention (b128)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, fontsize=7); ax[0].legend(fontsize=8, ncol=2)
    if tot:
        pv = [fnum(tot["fp16_us_per_step"]) / 1000, fnum(tot["int8_us_per_step"]) / 1000, fnum(tot["int4_us_per_step"]) / 1000]
        cc = [C3["fp16"], C3["int8"], C3["int4"]]
        b = ax[1].bar(range(3), pv, color=cc, edgecolor="white", lw=1.2)
        bars_labels(ax[1], b, pv, "{:.1f}", 9)
        ax[1].set_ylabel("ms / step"); ax[1].set_title("Attention total per step\n(21 blocks, incl GN+quant)")
        ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["fp16", "int8", "int4"], fontsize=9)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_attn_kernel.png", dpi=150); plt.close(fig)
    print("wrote fig_attn_kernel.png")

# ---------- 6. per-step summary (grouped: family x mode) ----------
r = load("perstep_summary.csv")
if r:
    fams = [row for row in r if row["family"] != "SUM of standalone kernels"]
    x = np.arange(len(fams)); w = 0.16
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, m in enumerate(MODES):
        vals = [fnum(row[m]) for row in fams]
        bb = ax.bar(x + (i - 2) * w, vals, w, label=LBL[m], color=COL[m],
                    hatch=HATCH.get(m, ""), edgecolor="white", lw=0.4)
        bars_labels(ax, bb, vals, "{:.1f}", 7)
    ax.set_ylabel("ms / step (standalone kernel time)")
    ax.set_title("Per-kernel-family time in one DDIM step, by mode (b128)  ·  count/step × µs/call")
    ax.set_xticks(x); ax.set_xticklabels([row["family"] for row in fams], fontsize=8)
    ymax = max(fnum(row[m]) for row in fams for m in MODES)
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_perstep_summary.png", dpi=150); plt.close(fig)
    print("wrote fig_perstep_summary.png")

# ---------- 7. workload: conv per-step time contribution by shape (where the step goes) ----------
r = load("conv_kernel_speed.csv")
if r:
    body = [row for row in r if row["Cin"] != "TOTAL_PER_STEP"]
    body = sorted(body, key=lambda x: -fnum(x["fp16_us_per_step"]))[:15]
    labels = [f"{row['Cin']}>{row['Cout']} {row['H']}²  ×{row['count_per_step']}{'' if row['quant_eligible']=='1' else '  (fp16-only)'}" for row in body]
    y = np.arange(len(body))[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    modes_show = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]; w = 0.16
    for i, m in enumerate(modes_show):
        vals = [fnum(row[f"{m}_us_per_step"]) / 1000 for row in body]
        ax.barh(y + (i - 2) * w, vals, w, label=LBL[m], color=COL[m], hatch=HATCH.get(m, ""), edgecolor="white", lw=0.3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("ms / step contribution (count × µs/call)")
    ax.set_title("Top-15 conv geometries by per-step time contribution (b128)")
    ax.legend(fontsize=8, ncol=5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_conv_perstep.png", dpi=150); plt.close(fig)
    print("wrote fig_conv_perstep.png")

# ---------- 8. GN->qkv quantize fusion (§7) ----------
r = load("fuse_gn_qkv_quant.csv")
e2e = load("fuse_gn_qkv_e2e.csv")
if r:
    body = [row for row in r if row["C"] != "TOTAL_PER_STEP"]
    tot = next((row for row in r if row["C"] == "TOTAL_PER_STEP"), None)
    groups = [f"{row['C']}/T{row['T']}\n×{row['count_per_step']}" for row in body]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [3, 1.2]})
    series = [("fp16_fe_us", "fp16", C3["fp16"], ""), ("nonfused_fe_us", "int8 non-fused (today)", C3["int8"], ""),
              ("fused_fe_us", "int8 GN→quant fused", C3["int4"], "///")]
    for i, (key, lab, col, ht) in enumerate(series):
        ax[0].bar(x + (i - 1) * w, [fnum(row[key]) for row in body], w, label=lab, color=col, hatch=ht, edgecolor="white", lw=0.3)
    ax[0].set_yscale("log"); ax[0].set_ylabel("µs / call (log)")
    ax[0].set_title("qkv front-end (GroupNorm + quantize + qkv GEMM) per block (b128)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, fontsize=8); ax[0].legend(fontsize=8)
    if tot:
        pv = [fnum(tot["fp16_fe_us"]) / 1000, fnum(tot["nonfused_fe_us"]) / 1000, fnum(tot["fused_fe_us"]) / 1000]
        cc = [C3["fp16"], C3["int8"], C3["int4"]]; hh = ["", "", "///"]
        b = ax[1].bar(range(3), pv, color=cc, hatch=hh, edgecolor="white", lw=1.2)
        bars_labels(ax[1], b, pv, "{:.2f}", 9)
        sub = f"fused {fnum(tot['fused_vs_nonfused'])}× vs non-fused"
        if e2e:
            g = e2e[0]
            sub += f"\ne2e wall {g['ms_step_off']}→{g['ms_step_on']} ms (flat)\nrel-L2 {g['output_relL2_on_vs_off']} · gpu_busy −1ms"
        ax[1].set_ylabel("ms / step"); ax[1].set_title("qkv front-end total / step\n" + sub, fontsize=9)
        ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["fp16", "non-\nfused", "fused"], fontsize=8)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_fuse_gn_qkv.png", dpi=150); plt.close(fig)
    print("wrote fig_fuse_gn_qkv.png")

print("plots done")
