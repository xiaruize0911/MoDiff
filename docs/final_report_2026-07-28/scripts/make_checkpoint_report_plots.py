"""Figures for the FP16 / INT8 / INT4 checkpoint report.

Inputs (all measured with the three modes in ONE process, so columns are comparable):
  data/e2e_three_mode.json        end-to-end DDIM + per-kernel profile
  data/attn_three_mode_final.json per-layer-type benchmark + per-kernel profile

Outputs, in plots/:
  fig_ck_e2e.png            e2e latency, speedup, and where the whole model's time goes
  fig_ck_layers.png         every layer type x shape, three modes side by side
  fig_ck_attn_stages.png    attention layers broken into stages
  fig_ck_speedup_matrix.png one heatmap: speedup vs FP16 for every (layer, shape, mode)
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(ROOT, "docs/final_report_2026-07-28/data")
OUT = os.path.join(ROOT, "docs/final_report_2026-07-28/plots")
os.makedirs(OUT, exist_ok=True)

MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
COL = {"fp16": "#8d99ae", "int8_baseline": "#2f6fb2", "int4_baseline": "#27924f"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---- kernel -> stage, shared by the e2e and layer views ------------------------------------
STAGES = [
    ("attn", "attention core", "#c0392b",
     ("flash_attn_int8_mma", "flash_attn_int4_mma", "pytorch_flash", "flash_fwd",
      "qi8packed_small", "bmm", "softmax")),
    ("conv", "convolution", "#1b7f79",
     ("implicit_gemm", "ImplicitGemmConvolution", "conv2d", "cudnn", "Kernel_conv",
      "implicit_convolve", "wgrad", "dgrad", "xmma_conv")),
    ("gemm", "GEMM / projection", "#2f6fb2",
     ("gemm_w8a8_kernel", "gemm_w4a4_kernel", "ampere_fp16_s1688gemm",
      "ampere_fp16_s16816gemm", "sm80_xmma_gemm", "gemm_w4a4_awq", "cutlass",
      "ampere_s16816gemm", "gemv")),
    ("norm", "GroupNorm + quantize", "#e6a020",
     ("group_norm", "gn_accum", "gn_finalize", "quantize_act", "quant_act")),
    ("prep", "K/V prep + out quantize", "#7d5ba6",
     ("aq_kv_packed", "quant_attn_out", "quantize_attn", "qkv_i4codes", "from_i8_kv_tiled",
      "layout_transform", "modiff_delta")),
    ("misc", "elementwise / copies / other", "#b9c2cc", ()),
]


def stage_of(name):
    for key, _, _, frags in STAGES:
        if frags and any(f.lower() in name.lower() for f in frags):
            return key
    return "misc"


def split(kernels, keyname="kernel", usname="us"):
    t = defaultdict(float)
    for k in kernels:
        t[stage_of(k[keyname])] += k[usname]
    return t


# =============================================================== figure 1: end to end
e2e = json.load(open(f"{D}/e2e_three_mode.json"))
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.4),
                                 gridspec_kw={"width_ratios": [1, 1, 1.5]})

vals = [e2e["modes"][m]["wall_us_per_batch"] / 1e3 for m, _ in MODES]
bars = a1.bar([l for _, l in MODES], vals, .55, color=[COL[m] for m, _ in MODES])
for b, v, (m, _) in zip(bars, vals, MODES):
    a1.text(b.get_x() + b.get_width() / 2, v + max(vals) * .015,
            f"{v:.0f} ms\n{e2e['modes'][m]['speedup_vs_fp16']:.3f}x",
            ha="center", fontweight="bold", fontsize=9.5)
a1.set_ylim(0, max(vals) * 1.2)
a1.set_ylabel(f"ms for one batch of {e2e['batch']}")
a1.set_title(f"A.  End-to-end DDIM\n{e2e['steps']} steps, batch {e2e['batch']}, A40", loc="left")
a1.grid(axis="y", alpha=.25)

ps = [e2e["modes"][m]["per_step_ms"] for m, _ in MODES]
bars = a2.bar([l for _, l in MODES], ps, .55, color=[COL[m] for m, _ in MODES])
for b, v in zip(bars, ps):
    a2.text(b.get_x() + b.get_width() / 2, v + max(ps) * .015, f"{v:.2f}", ha="center",
            fontweight="bold", fontsize=9.5)
a2.set_ylim(0, max(ps) * 1.2)
a2.set_ylabel("ms per denoising step")
a2.set_title("B.  Per DDIM step\n(the number that scales with step count)", loc="left")
a2.grid(axis="y", alpha=.25)

bot = np.zeros(len(MODES))
_tot = max(sum(split(e2e["modes"][m]["kernels"]).values()) for m, _ in MODES) / 1e3
_hidden = []
for key, lbl, colr, _ in STAGES:
    vv = np.array([split(e2e["modes"][m]["kernels"])[key] / 1e3 for m, _ in MODES])
    # Every segment is still drawn; only the LEGEND entry is dropped for stages too small to
    # see (<1% of the tallest bar). A key for an invisible colour is pure clutter -- but the
    # data stays in the stack and in the report's table, so nothing is hidden.
    visible = vv.max() >= 0.01 * _tot
    if not visible:
        _hidden.append(f"{lbl} (max {vv.max():.0f} ms)")
    a3.bar([l for _, l in MODES], vv, .55, bottom=bot, color=colr,
           label=lbl if visible else None)
    bot += vv
if _hidden:
    print("legend omits (still drawn, <1% of tallest bar):", "; ".join(_hidden))
for i, v in enumerate(bot):
    a3.text(i, v * 1.01, f"{v:.0f} ms", ha="center", fontweight="bold", fontsize=9)
a3.set_ylim(0, bot.max() * 1.18)
a3.set_ylabel(f"ms per batch of {e2e['batch']}")
a3.set_title("C.  Whole-model time by stage\n"
             "profiler self-time scaled to measured wall time; stages <1% omitted from the key",
             loc="left")
a3.legend(frameon=False, fontsize=8.5, loc="upper right")
a3.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ck_e2e.png", dpi=150, facecolor="w")

# =============================================================== layer-level data
# attn_three_mode_final.json holds all three modes, but its INT4 column predates the fusion
# round. attn_int4_m4.json is the re-measured INT4. Overlay it where present, and say which
# entries came from where -- reading the older file alone silently reproduces stale INT4 numbers,
# which is exactly the mistake this overlay exists to prevent.
lay = json.load(open(f"{D}/attn_three_mode_final.json"))
L = {}
for m, _ in MODES:
    for e in lay["modes"][m]:
        key = (e["kind"], tuple(e["x_shape"]))
        L.setdefault(key, {})[m] = e
_i4_path = f"{D}/attn_int4_m4.json"
_overlaid = 0
if os.path.exists(_i4_path):
    for e in json.load(open(_i4_path))["modes"]["int4_baseline"]:
        key = (e["kind"], tuple(e["x_shape"]))
        if key in L:
            L[key]["int4_baseline"] = e
            _overlaid += 1
    print(f"INT4 layer column: {_overlaid}/{len(L)} entries overlaid from attn_int4_m4.json")
# order: attention by descending T, then resblocks by descending pixels
keys = sorted(L.keys(), key=lambda k: (k[0], -(k[1][2] * k[1][3]), -k[1][1]))


def short(k):
    kind, xs = k
    tag = {"attention": "attn", "resblock_plain": "resblk", "resblock_updown": "resblk↕"}[kind]
    return f"{tag}\nC{xs[1]}/{xs[2]}²"


# =============================================================== figure 2: every layer
fig, ax = plt.subplots(figsize=(max(14, len(keys) * .62), 6.0))
w, xs_ = .26, np.arange(len(keys))
for i, (m, lbl) in enumerate(MODES):
    v = [L[k][m]["pipeline_us"] for k in keys]
    ax.bar(xs_ + (i - 1) * w, v, w, label=lbl, color=COL[m])
for j, k in enumerate(keys):
    f = L[k]["fp16"]["pipeline_us"]
    best = min(("int8_baseline", "int4_baseline"), key=lambda m: L[k][m]["pipeline_us"])
    ax.text(j, max(L[k][m]["pipeline_us"] for m, _ in MODES) * 1.08,
            f"{f / L[k][best]['pipeline_us']:.2f}x", ha="center", fontsize=7,
            color=COL[best], fontweight="bold")
ax.set_xticks(xs_)
ax.set_xticklabels([short(k) for k in keys], fontsize=7.5)
ax.set_yscale("log")
ax.set_ylabel("µs per layer call (log)")
ax.set_title("Every layer type and shape in the UNet, three modes side by side\n"
             "A40 batch 128; label = best quantized speedup vs FP16, coloured by which mode won",
             loc="left")
ax.legend(frameon=False, ncol=3)
ax.grid(axis="y", alpha=.25, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ck_layers.png", dpi=150, facecolor="w")

# =============================================================== figure 3: attention stages
akeys = [k for k in keys if k[0] == "attention"]
fig, axes = plt.subplots(1, len(akeys), figsize=(3.4 * len(akeys), 5.2))
for ax, k in zip(np.atleast_1d(axes), akeys):
    bot = np.zeros(len(MODES))
    for key, lbl, colr, _ in STAGES:
        vv = np.array([split(L[k][m]["kernels"], usname="us_per_layer_call")[key]
                       for m, _ in MODES])
        # rescale so the stack equals the measured pipeline latency
        vv = vv * np.array([L[k][m]["pipeline_us"] /
                            max(sum(split(L[k][m]["kernels"],
                                          usname="us_per_layer_call").values()), 1e-9)
                            for m, _ in MODES])
        ax.bar([l for _, l in MODES], vv, .6, bottom=bot, color=colr,
               label=lbl if k == akeys[0] else None)
        bot += vv
    for i, v in enumerate(bot):
        ax.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_title(f"C{k[1][1]} / T{k[1][2]*k[1][3]}", fontsize=10)
    ax.set_ylim(0, bot.max() * 1.16)
    ax.grid(axis="y", alpha=.25)
np.atleast_1d(axes)[0].set_ylabel("µs per layer")
fig.legend(loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(.5, 1.0))
fig.suptitle("Attention layers, broken into stages   (bar label = total µs)", y=.9, fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .85])
fig.savefig(f"{OUT}/fig_ck_attn_stages.png", dpi=150, facecolor="w")

# =============================================================== figure 4: speedup heatmap
qm = [m for m, _ in MODES if m != "fp16"]
M = np.array([[L[k]["fp16"]["pipeline_us"] / L[k][m]["pipeline_us"] for k in keys]
              for m in qm])
fig, ax = plt.subplots(figsize=(max(13, len(keys) * .6), 3.4))
vmax = max(2.0, float(np.nanmax(M)))
im = ax.imshow(M, cmap="RdYlGn", vmin=2 - vmax, vmax=vmax, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                fontweight="bold" if M[i, j] == M[:, j].max() else "normal",
                color="black")
ax.set_yticks(range(len(qm)))
ax.set_yticklabels(["INT8", "INT4"])
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([short(k).replace("\n", " ") for k in keys], rotation=55,
                   ha="right", fontsize=7.5)
ax.set_title("Speedup vs FP16 per layer — green is faster, red is SLOWER than FP16.  "
             "Bold = winning mode for that layer", loc="left")
fig.colorbar(im, ax=ax, shrink=.85, label="x vs FP16")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ck_speedup_matrix.png", dpi=150, facecolor="w")

print("e2e ms/batch:", {l: round(e2e["modes"][m]["wall_us_per_batch"] / 1e3, 1)
                        for m, l in MODES})
for f in ("fig_ck_e2e", "fig_ck_layers", "fig_ck_attn_stages", "fig_ck_speedup_matrix"):
    print(f"wrote {OUT}/{f}.png")
