"""Plots for the int8/int4-vs-fp16 speedup report. Reuses the measured CSVs from the
static_vs_dynamic study (same runs) + the balanced-config quality here. fp16 baseline = materialized
(precision-isolated: same attention algorithm, only precision differs)."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
SD = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"
D = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
P = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16"
FP16, INT8, INT4, BAL = "#888888", "#4C78A8", "#E45756", "#F58518"

def rd(d, name):
    p = f"{d}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None
def save(fig, name): fig.tight_layout(); fig.savefig(f"{P}/{name}", dpi=120, bbox_inches="tight"); plt.close(fig); print(" ", name)

cs = {r["config"]: r for r in (rd(SD, "clean_speed.csv") or [])}
def g(cfg): return float(cs[cfg]["gpu_busy_ms"]) if cfg in cs else np.nan

# ---- 1. e2e speedup vs fp16 (materialized) ----
fp16 = g("fp16 dyn")
bars = [("fp16", fp16, FP16), ("int8\nstatic", g("int8 full-static"), INT8),
        ("int8\nbalanced", g("int8 attn-dyn"), BAL), ("int4\nstatic", g("int4 full-static"), INT4),
        ("int4\nbalanced", g("int4 attn-dyn"), BAL)]
fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(len(bars))
ax.bar(x, [b[1] for b in bars], 0.62, color=[b[2] for b in bars])
for i, (lab, v, _) in enumerate(bars):
    tag = f"{v:.0f}" + (f"\n{fp16/v:.2f}×" if lab != "fp16" else "")
    ax.text(i, v, tag, ha="center", va="bottom", fontsize=8)
ax.axhline(fp16, ls="--", c=FP16, lw=1)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars]); ax.set_ylabel("GPU-busy ms/step")
ax.set_title("E2E speedup vs fp16 (materialized attention, batch 32) — lower is better")
ax.set_ylim(0, fp16 * 1.15); save(fig, "01_speedup_vs_fp16.png")

# ---- 2. matmul breakdown: conv vs qkv/proj+attn, fp16/int8/int4 ----
pf = rd(SD, "kernel_profile.csv")
if pf:
    def bucket(mode, b): return next((float(r["ms_step"]) for r in pf if r["mode"] == mode and r["bucket"] == b), 0.0)
    modes = [("dynamic_fp16", "fp16", FP16), ("static_int8", "int8", INT8), ("static_int4", "int4", INT4)]
    conv = [bucket(m, "conv (GEMM)") for m, _, _ in modes]
    gemm = [bucket(m, "GEMM (qkv/proj + attn QK·AV)") for m, _, _ in modes]
    x = np.arange(len(modes)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - w/2, conv, w, label="conv (GEMM)", color="#54A24B")
    b2 = ax.bar(x + w/2, gemm, w, label="qkv/proj + attn QKᵀ/AV", color="#B279A2")
    for i in range(len(modes)):
        ax.text(i - w/2, conv[i], f"{conv[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, gemm[i], f"{gemm[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n, _ in modes]); ax.set_ylabel("GPU-busy ms/step")
    ax.set_title("Matmul by type: conv quantization delivers; qkv/proj+attn does NOT (short-K wall)")
    ax.legend(); save(fig, "02_matmul_breakdown.png")

# ---- 3. Amdahl: where the time goes (stacked), fp16 vs int8-static vs int4-static ----
if pf:
    modes = [("dynamic_fp16", "fp16"), ("static_int8", "int8 static"), ("static_int4", "int4 static")]
    order = ["conv (GEMM)", "GEMM (qkv/proj + attn QK·AV)", "attention (softmax)", "GroupNorm",
             "elementwise / copy", "quantize / absmax", "conv store epilogue", "upsample / concat", "other"]
    M = {m: {b: 0.0 for b in order} for m, _ in modes}
    for r in pf:
        if r["mode"] in M and r["bucket"] in M[r["mode"]]: M[r["mode"]][r["bucket"]] = float(r["ms_step"])
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(modes)); bot = np.zeros(len(modes)); cmap = plt.get_cmap("tab10")
    for i, b in enumerate(order):
        vals = [M[m][b] for m, _ in modes]
        lab = b + (" ★matmul" if "GEMM" in b else "")
        ax.bar(x, vals, 0.6, bottom=bot, label=lab, color=cmap(i % 10)); bot += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in modes]); ax.set_ylabel("GPU-busy ms/step")
    ax.set_title("Where the time goes — matmul (★) is only ~40%, so quant e2e speedup is Amdahl-capped")
    ax.legend(fontsize=7, ncol=2); save(fig, "03_amdahl.png")

# ---- 4. attention kernel micro: int8/int4 vs fp16 (static path) ----
ak = rd(SD, "attn_kernel_speed.csv")
if ak:
    Ts = sorted({int(r["T"]) for r in ak}, reverse=True)
    def us(prec, T): return next((float(r["static_us"]) for r in ak if r["precision"] == prec and int(r["T"]) == T), np.nan)
    x = np.arange(len(Ts)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for j, (prec, col) in enumerate([("fp16", FP16), ("int8", INT8), ("int4", INT4)]):
        v = [us(prec, T) for T in Ts]; ax.bar(x + (j-1)*w, v, w, label=prec, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"T={T}" for T in Ts]); ax.set_ylabel("attention kernel µs")
    ax.set_title("Attention QKᵀ/AV+softmax kernel: int8/int4 ARE faster than fp16 (isolated)")
    ax.legend(); save(fig, "04_attn_micro.png")

# ---- 5. analytical IO + peak mem ----
an = rd(SD, "pipeline_io_analytic.csv"); io = rd(SD, "pipeline_io.csv")
if an and io:
    precs = ["fp16", "int8", "int4"]
    ioT = [next((float(r["total_MiB"]) for r in an if r["precision"] == p and r["variant"] == "static"), np.nan) for p in precs]
    peakmap = {"fp16": "static_fp16", "int8": "static_int8", "int4": "static_int4"}
    peak = [next((float(r["peak_mem_MiB"]) for r in io if r["mode"] == peakmap[p]), np.nan) for p in precs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4)); x = np.arange(3); cols = [FP16, INT8, INT4]
    axes[0].bar(x, ioT, color=cols); axes[0].set_xticks(x); axes[0].set_xticklabels(precs)
    axes[0].set_ylabel("analytical DRAM MiB/step"); axes[0].set_title("Total IO (static)")
    for i, v in enumerate(ioT): axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(x, peak, color=cols); axes[1].set_xticks(x); axes[1].set_xticklabels(precs)
    axes[1].set_ylabel("peak MiB"); axes[1].set_title("Peak memory (static)")
    for i, v in enumerate(peak): axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    save(fig, "05_io_mem.png")

print("plots done")
