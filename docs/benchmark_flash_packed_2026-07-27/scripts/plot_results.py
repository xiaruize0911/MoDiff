"""Plots for the MODIFF_FLASH_PACKED default-flip A/B benchmark.

Reads:
  - docs/benchmark_flash_packed_2026-07-27/data/ab_speed.csv (new: int8_baseline/int8_modiff x packed 0/1)
  - docs/benchmark_flash_packed_2026-07-27/data/block_choices.json (per-block frozen flash/packed choice)
  - docs/benchmark_5mode_2026-07-25/data/e2e_speed.csv (reused: fp16/int4_baseline/int4_modiff, unaffected
    by this flag -- fp16 doesn't use the flash quantized-attention class, int4's _flash_packed gate is
    hardcoded bits==8)
Writes PNGs into docs/benchmark_flash_packed_2026-07-27/plots/.
"""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "docs/benchmark_flash_packed_2026-07-27"
OLD = "docs/benchmark_5mode_2026-07-25"
os.makedirs(f"{HERE}/plots", exist_ok=True)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                      "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})

C_OLD = "#888780"   # gray -- old default (packed=0)
C_NEW = "#1D9E75"   # teal -- new default (packed=1)
C_REF = "#B4B2A9"   # light gray -- unaffected reference modes

# ---- load new A/B data ----
ab = {}
with open(f"{HERE}/data/ab_speed.csv") as f:
    for row in csv.DictReader(f):
        ab[(row["mode"], int(row["flash_packed"]))] = float(row["ms_step"])

# ---- load old 5-mode reference (unaffected modes) ----
ref = {}
with open(f"{OLD}/data/e2e_speed.csv") as f:
    for row in csv.DictReader(f):
        ref[row["mode"]] = float(row["ms_step"])

# =========================================================================
# Plot 1: e2e ms/step across all 5 modes, int8 modes split packed 0 vs 1
# =========================================================================
labels = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
fig, ax = plt.subplots(figsize=(9, 5))
x = list(range(len(labels)))
bar_w = 0.38

for i, lab in enumerate(labels):
    if lab in ("int8_baseline", "int8_modiff"):
        v0, v1 = ab[(lab, 0)], ab[(lab, 1)]
        ax.bar(i - bar_w / 2, v0, bar_w, color=C_OLD, edgecolor="none",
               label="packed=0 (old default)" if i == 1 else None)
        ax.bar(i + bar_w / 2, v1, bar_w, color=C_NEW, edgecolor="none",
               label="packed=1 (new default)" if i == 1 else None)
        for xpos, v in ((i - bar_w / 2, v0), (i + bar_w / 2, v1)):
            ax.text(xpos, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, color="#2C2C2A")
    else:
        v = ref[lab]
        ax.bar(i, v, bar_w * 2, color=C_REF, edgecolor="none",
               label="unaffected (reused 07-25 run)" if i == 0 else None)
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, color="#2C2C2A")

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("ms / step (b128, A40)")
ax.set_title("E2E DDIM step time: MODIFF_FLASH_PACKED default 0 -> 1")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.02,
         "fp16/int4 bars are reused from the 07-25 run (unaffected by this flag); gray/teal int8 bars\n"
         "are both from today's session, back-to-back, so only that pair is a same-environment A/B.",
         ha="center", fontsize=9, color="#5F5E5A")
fig.tight_layout()
fig.savefig(f"{HERE}/plots/01_e2e_ms_per_step.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =========================================================================
# Plot 2: int8-only zoom with % delta annotated
# =========================================================================
fig, ax = plt.subplots(figsize=(6.5, 5.5))
int8_labels = ["int8_baseline", "int8_modiff"]
x = list(range(len(int8_labels)))
for i, lab in enumerate(int8_labels):
    v0, v1 = ab[(lab, 0)], ab[(lab, 1)]
    ax.bar(i - bar_w / 2, v0, bar_w, color=C_OLD, label="packed=0 (old default)" if i == 0 else None)
    ax.bar(i + bar_w / 2, v1, bar_w, color=C_NEW, label="packed=1 (new default)" if i == 0 else None)
    ax.text(i - bar_w / 2, v0 + 1, f"{v0:.1f}", ha="center", fontsize=10)
    ax.text(i + bar_w / 2, v1 + 1, f"{v1:.1f}", ha="center", fontsize=10)
    pct = (v0 - v1) / v0 * 100
    ymax = max(v0, v1)
    ax.text(i, ymax + 9, f"{pct:+.1f}%", ha="center", fontsize=11, fontweight="bold",
            color="#04342C" if pct > 0 else "#501313")
ax.set_ylim(0, max(ab[l, p] for l in int8_labels for p in (0, 1)) * 1.15)
ax.set_xticks(x); ax.set_xticklabels(int8_labels)
ax.set_ylabel("ms / step (b128, A40)")
ax.set_title("int8 modes only: effect of\nMODIFF_FLASH_PACKED default flip")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/02_int8_zoom.png", dpi=150)
plt.close(fig)

# =========================================================================
# Plot 3: per-block frozen choice (flash vs SDPA, packed vs non-packed) by head_dim,
#         for int8_modiff, comparing packed=0 vs packed=1
# =========================================================================
with open(f"{HERE}/data/block_choices.json") as f:
    blocks = json.load(f)

def level_of(name):
    """Map a QuantizedStandardAttentionBlock module path to its UNet level label,
    using the known input_blocks/output_blocks index layout for this config
    (num_res_blocks=2, channel_mult [1,2,2,4,4], attention_resolutions [1,2,4,8])."""
    if "middle_block" in name:
        return "Middle"
    idx = int(name.split(".")[-2])
    if name.split(".")[-3] == "input_blocks":
        return {1: "L0", 2: "L0", 4: "L1", 5: "L1", 7: "L2", 8: "L2", 10: "L3", 11: "L3"}[idx]
    return {12: "L0", 13: "L0", 14: "L0", 9: "L1", 10: "L1", 11: "L1",
            6: "L2", 7: "L2", 8: "L2", 3: "L3", 4: "L3", 5: "L3"}[idx]

LEVELS = ["L0", "L1", "L2", "L3", "Middle"]

def summarize(tag):
    """-> {level: (n_blocks, n_flash, n_packed)}"""
    by_lvl = {}
    for b in blocks[tag]:
        lvl = level_of(b["name"])
        n, nf, npk = by_lvl.get(lvl, (0, 0, 0))
        n += 1
        nf += 1 if b["flash_choice"] else 0
        npk += 1 if b["packed_choice"] else 0
        by_lvl[lvl] = (n, nf, npk)
    return by_lvl

s0 = summarize("int8_modiff_packed0")
s1 = summarize("int8_modiff_packed1")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, s, title, color in ((axes[0], s0, "packed=0 (old default)", C_OLD),
                            (axes[1], s1, "packed=1 (new default)", C_NEW)):
    xs = list(range(len(LEVELS)))
    flash_frac = [(s.get(l, (1, 0, 0))[1] / s.get(l, (1, 0, 0))[0]) for l in LEVELS]
    packed_frac = [(s.get(l, (1, 0, 0))[2] / s.get(l, (1, 0, 0))[0]) for l in LEVELS]
    w = 0.35
    ax.bar([i - w/2 for i in xs], flash_frac, w, color=color, label="uses flash (vs fp16 SDPA)")
    ax.bar([i + w/2 for i in xs], packed_frac, w, color=color, alpha=0.5, label="uses packed-fused kernel")
    ax.set_xticks(xs); ax.set_xticklabels(LEVELS)
    ax.set_title(title)
    ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("fraction of blocks")
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("int8_modiff: per-block score-path choice, by level (L3/Middle: hd=96, always ineligible)")
fig.tight_layout()
fig.savefig(f"{HERE}/plots/03_per_block_choice.png", dpi=150)
plt.close(fig)

print("wrote:")
for p in ("01_e2e_ms_per_step.png", "02_int8_zoom.png", "03_per_block_choice.png"):
    print(f"  {HERE}/plots/{p}")
