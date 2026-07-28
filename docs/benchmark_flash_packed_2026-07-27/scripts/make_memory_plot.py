"""Memory figures for REPORT.md: peak allocated vs peak reserved VRAM, all 5 modes,
default allocator vs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "docs/benchmark_flash_packed_2026-07-27"
mem = json.load(open(f"{HERE}/data/memory_profile.json"))
mem_exp = json.load(open(f"{HERE}/data/memory_profile_expandable.json"))

modes5 = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
labels = ["fp16", "int8\nbaseline", "int4\nbaseline", "int8\nmodiff", "int4\nmodiff"]
alloc = [mem[m]["peak_alloc_mb"] / 1024 for m in modes5]
reserved = [mem[m]["peak_reserved_mb"] / 1024 for m in modes5]

x = range(len(modes5))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar([i - w/2 for i in x], alloc, w, color="#378ADD", label="peak allocated (tensors)")
b2 = ax.bar([i + w/2 for i in x], reserved, w, color="#888780", label="peak reserved (VRAM footprint)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.1f}",
                ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("GB")
ax.set_title("Peak GPU memory, b128 (same-session measurement)")
ax.set_ylim(0, max(reserved) * 1.2)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/fig_memory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {HERE}/plots/fig_memory.png")

# Second figure: default allocator vs expandable_segments, peak reserved only.
reserved_exp = [mem_exp[m]["peak_reserved_mb"] / 1024 for m in modes5]
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar([i - w/2 for i in x], reserved, w, color="#888780", label="default allocator")
b2 = ax.bar([i + w/2 for i in x], reserved_exp, w, color="#1D9E75", label="expandable_segments:True")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.1f}",
                ha="center", fontsize=9)
for i, (r0, r1) in enumerate(zip(reserved, reserved_exp)):
    pct = (r1 - r0) / r0 * 100
    ax.text(i, max(r0, r1) + 0.7, f"{pct:+.1f}%", ha="center", fontsize=9, fontweight="bold", color="#0F6E56")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("GB")
ax.set_title("Peak reserved VRAM: default allocator vs expandable_segments (b128)")
ax.set_ylim(0, max(reserved) * 1.25)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/fig_memory_expandable.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {HERE}/plots/fig_memory_expandable.png")
