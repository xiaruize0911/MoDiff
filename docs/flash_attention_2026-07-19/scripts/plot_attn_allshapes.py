"""Plot the per-shape attention kernel benchmark (from attn_allshapes_bench.py).
Left: per-shape kernel-only us for each path (log y), block count annotated.
Right: expected total-attention us/forward per policy, speedup vs real annotated.
Writes docs/flash_attention_2026-07-19/fig_attn_allshapes_b128.png
"""
import os, csv
os.chdir("/workspace/MoDiff")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = 128
D = "docs/flash_attention_2026-07-19/data"
rows = list(csv.DictReader(open(f"{D}/attn_allshapes_kernel_b{B}.csv")))
pol = list(csv.DictReader(open(f"{D}/attn_policy_b{B}.csv")))

# organize per shape
shapes, counts = [], {}
for r in rows:
    key = f"hd{r['hd']}/T{r['T']}"
    if key not in shapes:
        shapes.append(key); counts[key] = int(r["count"])
paths = ["fp16 flash (real)", "fp16 MATH (old)", "int8 flash (ours)", "int4 flash (ours)"]
colors = {"fp16 flash (real)": "#2563eb", "fp16 MATH (old)": "#9ca3af",
          "int8 flash (ours)": "#f59e0b", "int4 flash (ours)": "#dc2626"}
kern = {p: {} for p in paths}
for r in rows:
    key = f"hd{r['hd']}/T{r['T']}"
    if r["path"] in kern:
        kern[r["path"]][key] = float(r["kernel_us"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(shapes)); w = 0.2
for i, p in enumerate(paths):
    ys = [kern[p].get(s, np.nan) for s in shapes]
    ax1.bar(x + (i - 1.5) * w, ys, w, label=p, color=colors[p])
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{s}\n(x{counts[s]})" for s in shapes], fontsize=9)
ax1.set_ylabel("kernel-only time (us, log)")
ax1.set_title(f"Attention kernel per shape @ b{B} (BH=1024)\nfp16 flash wins at every shape; (xN)=blocks/forward")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, axis="y", alpha=0.3)

# policy panel
pnames = [p["policy"] for p in pol]
pus = [float(p["us_per_forward"]) / 1000 for p in pol]  # ms
psp = [float(p["speedup_vs_real"]) for p in pol]
pc = ["#9ca3af", "#2563eb", "#f59e0b", "#dc2626", "#fbbf24", "#f87171"]
yb = np.arange(len(pnames))
ax2.barh(yb, pus, color=pc[:len(pnames)])
ax2.set_yticks(yb); ax2.set_yticklabels([n.replace(" where eligible", "\n(elig.)") for n in pnames], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("expected total attention (ms / forward)")
ax2.set_title(f"Expected total attention per forward (21 blocks) @ b{B}\nlabel = speedup vs real (fp16 flash)")
for i, (u, s) in enumerate(zip(pus, psp)):
    ax2.text(u, i, f"  {u:.1f}ms  {s:.2f}x", va="center", fontsize=8)
ax2.grid(True, axis="x", alpha=0.3)
ax2.set_xlim(0, max(pus) * 1.25)

plt.tight_layout()
out = f"docs/flash_attention_2026-07-19/fig_attn_allshapes_b{B}.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("WROTE", out)
