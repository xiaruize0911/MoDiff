"""Plot attention + linear kernel benchmark across all churches shapes @ b128."""
import os, csv
os.chdir("/workspace/MoDiff")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = 128; D = "docs/flash_attention_2026-07-19/data"
attn = list(csv.DictReader(open(f"{D}/kernel_attn_b{B}.csv")))
lin = list(csv.DictReader(open(f"{D}/linear_gemm_only_b{B}.csv")))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# --- attention: kernel us per shape (log) ---
ash, acnt = [], {}
for r in attn:
    if r["hd_or_K"] not in ash: ash.append(r["hd_or_K"]); acnt[r["hd_or_K"]] = r["count"]
apaths = ["fp16 MATH (real)", "int8 materialized attn", "int4 materialized attn"]
acol = {"fp16 MATH (real)": "#2563eb", "int8 materialized attn": "#f59e0b", "int4 materialized attn": "#dc2626"}
av = {p: {s: np.nan for s in ash} for p in apaths}
for r in attn:
    if r["path"] in av: av[r["path"]][r["hd_or_K"]] = float(r["us"])
x = np.arange(len(ash)); w = 0.26
for i, p in enumerate(apaths):
    ax1.bar(x + (i - 1) * w, [av[p][s] for s in ash], w, label=p.replace(" materialized attn", " mat"), color=acol[p])
ax1.set_yscale("log"); ax1.set_xticks(x); ax1.set_xticklabels([f"hd{s}\n(x{acnt[s]})" for s in ash], fontsize=9)
ax1.set_ylabel("kernel us (log)")
ax1.set_title("ATTENTION per shape @ b128 — fp16 MATH stays best\n(int8 ~neutral/regress; int4 fast but rel-L2~0.4 broken)")
ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.3)

# --- linear: speedup vs fp16 per shape (gemm-only vs full) ---
lsh = [f"{r['K']}->{r['N']}\nM{r['M']}(x{r['count']})" for r in lin]
i8g = [float(r["fp16_us"]) / float(r["i8_gemm"]) for r in lin]
i4g = [float(r["fp16_us"]) / float(r["i4_gemm"]) for r in lin]
i8f = [float(r["fp16_us"]) / float(r["i8_full"]) for r in lin]
i4f = [float(r["fp16_us"]) / float(r["i4_full"]) for r in lin]
xl = np.arange(len(lsh)); w2 = 0.2
ax2.bar(xl - 1.5 * w2, i8g, w2, label="int8 GEMM-only (fused quant)", color="#f59e0b")
ax2.bar(xl - 0.5 * w2, i4g, w2, label="int4 GEMM-only (fused quant)", color="#dc2626")
ax2.bar(xl + 0.5 * w2, i8f, w2, label="int8 +standalone quant", color="#fcd34d")
ax2.bar(xl + 1.5 * w2, i4f, w2, label="int4 +standalone quant", color="#fca5a5")
ax2.axhline(1.0, color="#2563eb", ls="--", lw=1.2, label="fp16 (real)")
ax2.set_xticks(xl); ax2.set_xticklabels(lsh, fontsize=7)
ax2.set_ylabel("speedup vs fp16")
ax2.set_title("LINEAR qkv/proj per shape @ b128 — int quant wins at K>=384,\nfused-quant int8 1.46x / int4 1.83x weighted; standalone quant erases it")
ax2.legend(fontsize=7, loc="upper left"); ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out = f"docs/flash_attention_2026-07-19/fig_kernel_all_b{B}.png"
plt.savefig(out, dpi=130, bbox_inches="tight"); print("WROTE", out)
