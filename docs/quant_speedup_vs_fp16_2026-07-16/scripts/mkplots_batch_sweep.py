"""Three-batch (16/64/128) comparison of the pipeline bucket profile. Reads
data/pipeline_buckets_b{B}.csv + pipeline_speed_b{B}.csv. Emits 20_pipeline_batch_sweep.png."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
P = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16"
BATCHES = [16, 64, 128]
COL = {"attention (softmax)": "#E45756", "elementwise / copy": "#F58518", "GroupNorm": "#EECA3B",
       "conv (GEMM)": "#54A24B", "qkv/proj GEMM (+ fp16 attn bmm)": "#4C78A8",
       "quantize / absmax": "#B279A2", "conv store epilogue": "#72B7B2",
       "upsample / concat": "#9D755D", "other": "#BAB0AC"}


def rd(name):
    p = f"{D}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def buckets(mode, b):
    return {r["bucket"]: float(r["ms_step"]) for r in rd(f"pipeline_buckets_b{b}.csv") if r["mode"] == mode}


def speed(mode, b):
    r = [x for x in rd(f"pipeline_speed_b{b}.csv") if x["mode"] == mode]
    return float(r[0]["wall_ms_step"]) if r else float("nan")


fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

# --- panel 1: int8 bucket ms/step, grouped by bucket across 3 batches ---
ax = axes[0]
order = ["qkv/proj GEMM (+ fp16 attn bmm)", "attention (softmax)", "conv (GEMM)",
         "elementwise / copy", "GroupNorm", "quantize / absmax", "other"]
x = np.arange(len(order)); w = 0.26
for j, b in enumerate(BATCHES):
    bk = buckets("int8", b)
    vals = [bk.get(o, 0) for o in order]
    bars = ax.bar(x + (j - 1) * w, vals, w, color=[COL[o] for o in order],
                  alpha=[0.45, 0.72, 1.0][j], edgecolor="#333", linewidth=0.3)
    for xi, v in zip(x + (j - 1) * w, vals):
        if v > 2: ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([o.replace(" (+ fp16 attn bmm)", "\n+attn bmm").replace(" / ", "/\n").replace(" (GEMM)", "") for o in order], fontsize=7.5)
ax.set_ylabel("int8 device self-time (ms/step)")
ax.set_title("int8 baseline buckets across batch (light→dark = b16/b64/b128)\nall scale ~linearly; attention is the largest, conv shrinks under int8", fontsize=10)

# --- panel 2: attention (softmax) SHARE of the step, int8, across batch ---
ax = axes[1]
sm_share = [buckets("int8", b)["attention (softmax)"] / sum(buckets("int8", b).values()) * 100 for b in BATCHES]
attn_tot = []  # softmax + qkv/proj bucket (~80% of which is fp16 attn bmm) as an upper proxy
for b in BATCHES:
    bk = buckets("int8", b); attn_tot.append((bk["attention (softmax)"] + bk["qkv/proj GEMM (+ fp16 attn bmm)"]) / sum(bk.values()) * 100)
xb = np.arange(len(BATCHES))
ax.plot(xb, sm_share, "-o", color="#E45756", lw=2, label="softmax only")
ax.plot(xb, attn_tot, "--s", color="#B23", lw=1.6, label="softmax + (qkv/proj+attn bmm) bucket")
for i, v in enumerate(sm_share): ax.text(xb[i], v + 0.4, f"{v:.0f}%", ha="center", color="#E45756", fontsize=9)
for i, v in enumerate(attn_tot): ax.text(xb[i], v + 0.6, f"{v:.0f}%", ha="center", color="#B23", fontsize=8)
ax.set_xticks(xb); ax.set_xticklabels([f"b{b}" for b in BATCHES]); ax.set_ylabel("% of int8 step")
ax.set_ylim(0, 55); ax.set_title("Attention share grows with batch\n(softmax is memory-bound ∝ B·T²)", fontsize=10)
ax.legend(fontsize=8, loc="center right")

# --- panel 3: wall ms/step fp16 vs int8 + ratio, across batch ---
ax = axes[2]
fp = [speed("fp16", b) for b in BATCHES]; q = [speed("int8", b) for b in BATCHES]
ax.plot(xb, fp, "-o", color="#888", lw=2, label="fp16")
ax.plot(xb, q, "-o", color="#4C78A8", lw=2, label="int8")
for i in range(len(BATCHES)):
    ax.text(xb[i], q[i] + 4, f"{q[i]/fp[i]:.2f}×", ha="center", fontsize=8.5, color="#4C78A8")
ax.set_xticks(xb); ax.set_xticklabels([f"b{b}" for b in BATCHES]); ax.set_ylabel("wall ms/step")
ax.set_title("int8 baseline is 0.95–0.98× fp16 e2e (FASTER)\n(conv quant win; no MoDiff cache overhead)", fontsize=10)
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Pipeline profile across batch 16 / 64 / 128 (int8 baseline, no MoDiff caching) — churches UNet, A40", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{P}/20_pipeline_batch_sweep.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("wrote 20_pipeline_batch_sweep.png")
