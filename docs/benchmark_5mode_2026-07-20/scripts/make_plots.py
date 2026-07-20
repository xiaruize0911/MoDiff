"""Generate all report figures from data/*.csv -> figs/*.png. Skips any figure whose CSV is missing.
matplotlib Agg, 150 dpi. Palette: fp16 #2563eb, int8 #f59e0b, int4 #dc2626 (modiff = lighter + hatch)."""
import os, csv
os.chdir("/workspace/MoDiff")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = "docs/benchmark_5mode_2026-07-20"
D, Fg = f"{HERE}/data", f"{HERE}/figs"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
COL = {"fp16": "#2563eb", "int8_baseline": "#f59e0b", "int4_baseline": "#dc2626",
       "int8_modiff": "#fbbf24", "int4_modiff": "#f87171"}
HATCH = {"int8_modiff": "//", "int4_modiff": "//"}


def load(name):
    p = f"{D}/{name}"
    if not os.path.exists(p):
        print(f"skip (missing {name})"); return None
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


# ---------- e2e speed ----------
r = load("e2e_speed.csv")
if r:
    by = {row["mode"]: row for row in r}
    modes = [m for m in MODES if m in by]
    ms = [fnum(by[m]["ms_step"]) for m in modes]; sp = [fnum(by[m]["speedup_vs_fp16"]) for m in modes]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    b0 = ax[0].bar(modes, ms, color=[COL[m] for m in modes], hatch=[HATCH.get(m, "") for m in modes], edgecolor="white")
    ax[0].set_ylabel("ms / step"); ax[0].set_title("E2E DDIM step time (b128, 200 steps)")
    for b, v in zip(b0, ms): ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    b1 = ax[1].bar(modes, sp, color=[COL[m] for m in modes], hatch=[HATCH.get(m, "") for m in modes], edgecolor="white")
    ax[1].axhline(1.0, color="#888", ls="--", lw=1); ax[1].set_ylabel("speedup vs fp16"); ax[1].set_title("E2E speedup vs true fp16")
    for b, v in zip(b1, sp): ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
    for a in ax: a.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_e2e_speed.png", dpi=150); plt.close(fig); print("wrote fig_e2e_speed.png")

# ---------- e2e timing profile (stacked) ----------
r = load("e2e_timing_profile.csv")
if r:
    by = {row["mode"]: row for row in r}
    modes = [m for m in MODES if m in by]
    buckets = [k for k in r[0].keys() if k not in ("mode", "gpu_busy", "wall")]
    palette = plt.cm.tab20(np.linspace(0, 1, len(buckets)))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(modes))
    for bi, bk in enumerate(buckets):
        vals = np.array([fnum(by[m][bk]) for m in modes])
        ax.bar(modes, vals, bottom=bottom, label=bk, color=palette[bi], edgecolor="white", lw=0.4)
        bottom += np.nan_to_num(vals)
    walls = [fnum(by[m]["wall"]) for m in modes]
    ax.plot(range(len(modes)), walls, "kD", ms=7, label="wall (indep.)")
    ax.set_ylabel("ms / step (GPU self time)"); ax.set_title("E2E per-component timing profile (b128, measured)")
    ax.tick_params(axis="x", rotation=20); ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_e2e_timing_profile.png", dpi=150); plt.close(fig); print("wrote fig_e2e_timing_profile.png")

# ---------- e2e memcpy (stacked H2D/D2H/D2D) ----------
r = load("e2e_memcpy_total.csv")
if r:
    by = {row["mode"]: row for row in r}
    modes = [m for m in MODES if m in by]
    kinds = [("H2D_MiB", "#60a5fa"), ("D2H_MiB", "#34d399"), ("D2D_MiB", "#f59e0b")]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(modes))
    for key, c in kinds:
        vals = np.array([fnum(by[m][key]) for m in modes])
        ax.bar(modes, vals, bottom=bottom, label=key.replace("_MiB", ""), color=c, edgecolor="white")
        bottom += np.nan_to_num(vals)
    for i, m in enumerate(modes):
        ax.text(i, bottom[i], f"{fnum(by[m]['total_MiB']):.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MiB / step (measured memcpy)"); ax.set_title("E2E memcpy traffic (nsys, copy-only; excludes in-kernel DRAM)")
    ax.tick_params(axis="x", rotation=20); ax.legend()
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_e2e_memcpy.png", dpi=150); plt.close(fig); print("wrote fig_e2e_memcpy.png")

# ---------- conv kernel (grouped, log-y) ----------
r = load("conv_kernel_speed.csv")
if r:
    shapes = [row["shape"] for row in r]
    x = np.arange(len(shapes)); w = 0.16
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(MODES):
        vals = [fnum(row.get(f"{m}_us", "nan")) for row in r]
        ax.bar(x + (i - 2) * w, vals, w, label=m, color=COL[m], hatch=HATCH.get(m, ""), edgecolor="white", lw=0.3)
    ax.set_yscale("log"); ax.set_ylabel("us (log)"); ax.set_title("Conv kernel time per churches shape (b128)")
    ax.set_xticks(x); ax.set_xticklabels(shapes, rotation=30, ha="right", fontsize=8); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_conv_kernel.png", dpi=150); plt.close(fig); print("wrote fig_conv_kernel.png")

# ---------- linear kernel (grouped per shape: fp16 / i8 gemm / i4 gemm) ----------
r = load("linear_kernel_speed.csv")
if r:
    body = [row for row in r if row["kind"] in ("qkv", "proj")]
    labels = [f"{row['kind']}\n{row['K']}->{row['N']}" for row in body]
    x = np.arange(len(labels)); w = 0.26
    series = [("fp16_us", "fp16", "#2563eb"), ("i8_gemm_us", "int8 GEMM", "#f59e0b"), ("i4_gemm_us", "int4 GEMM", "#dc2626")]
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (key, lab, c) in enumerate(series):
        ax.bar(x + (i - 1) * w, [fnum(row[key]) for row in body], w, label=lab, color=c, edgecolor="white", lw=0.3)
    ax.set_yscale("log"); ax.set_ylabel("us (log)"); ax.set_title("Linear qkv/proj GEMM time per shape (b128, GEMM-only)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7); ax.legend()
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_linear_kernel.png", dpi=150); plt.close(fig); print("wrote fig_linear_kernel.png")

# ---------- attn fair (stacked GN+quant+attn per shape x mode) ----------
r = load("attn_kernel_fair_speed.csv")
if r:
    body = [row for row in r if row["C"] != "WEIGHTED_TOTAL"]
    groups = [f"{row['C']}/{row['hd']}/{row['T']}" for row in body]
    sub = [("fp16", "attn16_us", None), ("int8", "attn8_us", "q8_us"), ("int4", "attn4_us", "q4_us")]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (lab, akey, qkey) in enumerate(sub):
        gn = np.array([fnum(row["gn_us"]) for row in body])
        q = np.array([fnum(row[qkey]) if qkey else 0.0 for row in body]) if qkey else np.zeros(len(body))
        a = np.array([fnum(row[akey]) if row[akey] not in ("", None) else fnum(row["attn16_us"]) for row in body])
        col = {"fp16": "#2563eb", "int8": "#f59e0b", "int4": "#dc2626"}[lab]
        pos = x + (i - 1) * w
        ax.bar(pos, gn, w, color="#94a3b8", edgecolor="white", lw=0.3, label="GroupNorm" if i == 0 else None)
        ax.bar(pos, np.nan_to_num(q), w, bottom=gn, color="#c084fc", edgecolor="white", lw=0.3, label="quantize" if i == 1 else None)
        ax.bar(pos, np.nan_to_num(a), w, bottom=gn + np.nan_to_num(q), color=col, edgecolor="white", lw=0.3, label=f"{lab} attn")
    ax.set_yscale("log"); ax.set_ylabel("us (log)"); ax.set_title("Attention kernel WITH norm (fair): GroupNorm + quantize + attention (b128)")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, fontsize=8); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{Fg}/fig_attn_fair.png", dpi=150); plt.close(fig); print("wrote fig_attn_fair.png")

print("plots done")
