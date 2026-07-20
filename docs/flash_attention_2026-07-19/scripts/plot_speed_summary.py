"""Visualize the MoDiff speed results (churches LDM-8, b128, DDIM, A40) vs a TRUE fp16 baseline.
Left: e2e speedup ladder across configs. Right: the int8-attention fusion arc."""
import os
os.chdir("/workspace/MoDiff")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Panel A: e2e config ladder (ms/step, speedup vs true fp16) ----
cfg = [
    ("fp16 (true baseline)", 189.6, "#9ca3af"),
    ("int8  ·  materialized attn", 179.1, "#93c5fd"),
    ("int8  ·  fp16 attn", 175.2, "#93c5fd"),
    ("int4  ·  fp16 attn", 173.1, "#fdba74"),
    ("int4  ·  materialized attn", 162.9, "#fdba74"),
    ("int4  ·  fused-flash attn", 152.1, "#f59e0b"),
    ("int8  ·  fused-flash attn  (DEFAULT)", 133.3, "#16a34a"),
]
cfg = sorted(cfg, key=lambda x: -x[1])   # slowest at top
labels = [c[0] for c in cfg]; ms = [c[1] for c in cfg]; cols = [c[2] for c in cfg]
sp = [189.6 / m for m in ms]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6))
y = np.arange(len(labels))
axA.barh(y, sp, color=cols, edgecolor="#374151", linewidth=0.6)
axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=9)
axA.axvline(1.0, color="#6b7280", ls="--", lw=1)
for i, (s, m) in enumerate(zip(sp, ms)):
    axA.text(s + 0.006, i, f"{s:.2f}×  ({m:.0f} ms)", va="center", fontsize=9,
             fontweight="bold" if "DEFAULT" in labels[i] else "normal")
axA.set_xlim(0.9, 1.52); axA.set_xlabel("e2e speedup vs TRUE fp16")
axA.set_title("End-to-end speedup by config\n(LSUN-churches LDM-8, b128, DDIM, A40)")
axA.grid(True, axis="x", alpha=0.3)

# ---- Panel B: int8 attention fusion arc ----
steps = ["fp16\nMATH attn", "materialized\nint8 attn", "+ flash\nfusion", "+ kernel\nquantize",
         "+ V pre-\ntransposed", "+ static\nquantize\n(DEFAULT)"]
arc = [1.083, 1.058, 1.142, 1.315, 1.361, 1.418]
x = np.arange(len(steps))
barcols = ["#93c5fd", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#16a34a"]
axB.bar(x, arc, color=barcols, edgecolor="#374151", linewidth=0.6)
axB.plot(x, arc, "-o", color="#111827", lw=1.4, ms=4)
for i, a in enumerate(arc):
    axB.text(i, a + 0.008, f"{a:.3f}×", ha="center", fontsize=9,
             fontweight="bold" if i == len(arc) - 1 else "normal")
axB.axhline(1.083, color="#94a3b8", ls=":", lw=1)
axB.text(0.05, 1.083 + 0.004, "fp16-attn start", fontsize=8, color="#64748b")
axB.set_xticks(x); axB.set_xticklabels(steps, fontsize=8)
axB.set_ylim(1.0, 1.47); axB.set_ylabel("int8 e2e speedup vs fp16")
axB.set_title("The int8-attention fusion arc\n(each step's e2e gain; materialized loses, fusion wins)")
axB.grid(True, axis="y", alpha=0.3)

fig.suptitle("MoDiff quantization speed — vs a TRUE fp16 baseline "
             "(the old \"2×\" was a fp32/tf32-baseline artifact)", fontsize=12, y=1.02)
plt.tight_layout()
out = "docs/flash_attention_2026-07-19/fig_speed_summary_b128.png"
plt.savefig(out, dpi=135, bbox_inches="tight"); print("WROTE", out)
