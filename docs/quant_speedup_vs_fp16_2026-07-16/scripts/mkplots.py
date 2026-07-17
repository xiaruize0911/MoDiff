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
    ax.set_title("Matmul by type (e2e profiler): conv shrinks with int; qkv/proj+attn MERGED bucket flat\n(merged hides the attn gain behind the slow int linear — see 02b for the attention split)")
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
    ax.set_title("Where the fp16->int8 time goes: elementwise (fp16 S*scale, folded by int8) + softmax\ndominate the saving; matmul (★) contributes little net")
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

# ---- 6. int8-output qkv->attn fusion prototype (component breakdown, C192 T1024) ----
fz = rd(D, "fusion_qkv_attn.csv")
if fz:
    r = next((r for r in fz if r["block"] == "C192"), None)
    if r:
        # measured components (us): A = gemm(fp16) + reshape + quantize; B = gemm(int8) + from_i8
        A = [("gemm (fp16 out)", 269.5), ("reshape copy", 186.6), ("quantize_attn_qkv", 570.5)]
        B = [("gemm (int8 out)", 286.1), ("quantize_from_i8", 616.9)]
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for i, parts in enumerate([A, B]):
            bot = 0
            for j, (lab, v) in enumerate(parts):
                ax.bar(i, v, 0.55, bottom=bot, color=plt.get_cmap("tab20")(j * 2 + i))
                ax.text(i, bot + v / 2, f"{lab}\n{v:.0f}", ha="center", va="center", fontsize=7)
                bot += v
            ax.text(i, bot, f"{bot:.0f}µs", ha="center", va="bottom", fontsize=9, weight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["A: fp16 round-trip", "B: int8 fused"])
        ax.set_ylabel("µs  (qkv-linear + quantize, C192 T1024)")
        ax.set_title(f"int8-output qkv→attn fusion: {float(r['fusion_speedup']):.2f}× (eliminates reshape copy)")
        save(fig, "06_fusion_qkv_attn.png")
# ---- 7. softmax/score memory-traffic roofline (T=1024) ----
mm = rd(D, "softmax_mem.csv")
if mm:
    r10 = [r for r in mm if int(r["T"]) == 1024]
    names = [r["kernel"] for r in r10]; gbps = [float(r["GBps"]) for r in r10]; pct = [float(r["pct_peak"]) for r in r10]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    cols = [INT8 if "int8" in n else (INT4 if "AV" in n or "QK" in n else FP16) for n in names]
    ax.bar(x, gbps, 0.6, color=cols)
    ax.axhline(696, ls="--", c="#333", lw=1.2, label="A40 peak 696 GB/s")
    for i, (g, p) in enumerate(zip(gbps, pct)):
        ax.text(i, g, f"{g:.0f}\n{p:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    ax.set_ylabel("achieved DRAM GB/s"); ax.set_ylim(0, 760); ax.legend()
    ax.set_title("Softmax / score-matrix kernels are DRAM-bandwidth-bound (T=1024, A40)")
    save(fig, "07_softmax_mem_roofline.png")
# ---- 2b. attention op-by-op speedup vs fp16 (QKᵀ / softmax / AV), T=1024 ----
ao = rd(D, "attn_ops.csv")
if ao:
    r10 = {r["op"]: r for r in ao if int(r["T"]) == 1024}
    ops = ["QKT", "softmax", "AV"]; x = np.arange(len(ops)); w = 0.38
    i8 = [float(r10[o]["int8_speedup"]) for o in ops]; i4 = [float(r10[o]["int4_speedup"]) for o in ops]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w/2, i8, w, label="int8", color=INT8); ax.bar(x + w/2, i4, w, label="int4", color=INT4)
    ax.axhline(1.0, ls="--", c=FP16, lw=1, label="fp16 (=1.0)")
    for i in range(len(ops)):
        ax.text(i - w/2, i8[i], f"{i8[i]:.2f}×", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, i4[i], f"{i4[i]:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["QKᵀ", "softmax", "AV"]); ax.set_ylabel("speedup vs fp16")
    ax.set_title("Attention op-by-op: matmuls (QKᵀ, AV) DO speed up; softmax is memory-bound (T=1024)")
    ax.legend(); save(fig, "02b_attn_ops.png")
# ---- 8. full int8-score attention: QKᵀ / softmax / full, fp16-S vs int8-S (T=1024) ----
i8s = rd(D, "int8_score.csv")
if i8s:
    r = next((x for x in i8s if int(x["T"]) == 1024), i8s[0])
    ops = [("QKᵀ", "qkT_fp16_us", "qkT_int8_us"), ("softmax", "softmax_fp16_us", "softmax_int8_us"),
           ("full attn", "full_fp16_us", "full_int8_us")]
    x = np.arange(len(ops)); w = 0.38
    fp = [float(r[a]) for _, a, _ in ops]; i8 = [float(r[b]) for _, _, b in ops]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w/2, fp, w, label="fp16 scores", color=FP16); ax.bar(x + w/2, i8, w, label="int8 scores", color=INT8)
    for i, (a, b) in enumerate(zip(fp, i8)):
        ax.text(i, max(a, b), f"{a/b:.2f}×", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([o[0] for o in ops]); ax.set_ylabel("µs (T=1024)")
    ax.set_title(f"Full int8-score attention (T=1024): QKᵀ write + softmax read both halved\n"
                 f"full {fp[2]/i8[2]:.2f}×, quality-free (rel {r['rel_int8S']} vs fp16-S {r['rel_fp16S']})")
    ax.legend(); save(fig, "08_int8_score.png")
# ---- 9/10. AWQ vs ours vs fp16 kernel benchmark ----
aw = rd(D, "awq_vs_ours.csv")
if aw:
    labels = [r["shape"] for r in aw]; x = np.arange(len(labels)); w = 0.26
    o8 = [float(r["ours8_vs_fp16"]) for r in aw]; aq = [float(r["awq_vs_fp16"]) for r in aw]; o4 = [float(r["ours4_vs_fp16"]) for r in aw]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w, o8, w, label="ours w8a8", color=INT8)
    ax.bar(x, aq, w, label="AWQ w8a8", color="#72B7B2")
    ax.bar(x + w, o4, w, label="ours w4a4", color=INT4)
    ax.axhline(1.0, ls="--", c=FP16, lw=1.2, label="fp16 cuBLAS (=1.0)")
    for i in range(len(labels)):
        for off, v in [(-w, o8[i]), (0, aq[i]), (w, o4[i])]:
            ax.text(i + off, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("kernel speedup vs fp16")
    ax.set_title("Kernel-only GEMM: AWQ w8a8 ≥ ours on every shape; both beat fp16 for K≥384 (A40)")
    ax.legend(fontsize=8); save(fig, "09_awq_vs_ours_speedup.png")

    fig, ax = plt.subplots(figsize=(11, 4.8)); w2 = 0.2
    fp = [float(r["fp16_TFLOPS"]) for r in aw]; t8 = [float(r["ours8_TFLOPS"]) for r in aw]
    ta = [float(r["awq_TFLOPS"]) for r in aw]; t4 = [float(r["ours4_TFLOPS"]) for r in aw]
    ax.bar(x - 1.5*w2, fp, w2, label="fp16", color=FP16); ax.bar(x - 0.5*w2, t8, w2, label="ours w8a8", color=INT8)
    ax.bar(x + 0.5*w2, ta, w2, label="AWQ w8a8", color="#72B7B2"); ax.bar(x + 1.5*w2, t4, w2, label="ours w4a4", color=INT4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("effective TFLOPS (2·M·K·N)")
    ax.set_title("Kernel throughput (TFLOPS): AWQ w8a8 up to 129 TFLOPS on large-K qkv")
    ax.legend(fontsize=8); save(fig, "10_awq_vs_ours_tflops.png")
print("plots done")
