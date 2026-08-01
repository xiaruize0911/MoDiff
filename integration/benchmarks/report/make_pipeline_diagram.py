"""Plot 2: what each kernel in the attention block does, and the tensors flowing between them.

The timing plots say how long each kernel takes but not what it is or what it hands on, which is
what makes the quantized chains hard to read: the tensor between two kernels is an int8 buffer
holding two int4 nibbles per byte, so its channel dimension is half the logical one and looks
wrong unless it is labelled.

Source of truth is the LAYER profile (layers_*.json) for the attention block at C192/32², i.e.
the kernels that actually run inside that block, with the times measured by the same protocol as
everywhere else (timed without the profiler, attributed with it). It is not the kernel-suite
capture: at that shape the GroupNorm entry cannot be told apart from a ResBlock's by its arguments,
and picking by name alone put a 500 us ResBlock GN into the attention chain.

The three modes do NOT share a kernel structure, so the diagram does not force them into one
template. FP16 has no standalone GroupNorm kernel at all -- its normalisation is folded into
ImplicitGemmConvolutionFusionPerSample together with the QKV projection -- while the quantized
modes run a separate fused GN+SiLU+quantize. Showing that difference is the point.

Boxes are ordered by dataflow role, which is the call order in
quantized_std_attention.py::_int4_layout_epilogue_forward (GN+quantize -> QKV -> attention core ->
output projection + bias + residual).

Writes plots/fig_attn_pipeline_diagram.png.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
COL = {"fp16": "#8d99ae", "int8_baseline": "#2f6fb2", "int4_baseline": "#27924f"}
SHAPE = (128, 192, 32, 32)          # the C192/32² attention block, T = 32*32 = 1024

# dataflow slot -> (kernel-name fragments, title, what it does)
SLOTS = [
    (("gn_accum", "gn_finalize", "group_norm"), "GroupNorm (+SiLU, +quantize)",
     "per-group mean/var, SiLU, and for the\nquantized modes quantize+pack, fused"),
    (("FusionPerSample", "gemm_w8a8_kernel_awq", "gemm_w4a4_kernel_awq"), "QKV projection",
     "1x1 projection to 3 x head_dim per head"),
    (("flash", "sdpa"), "attention core",
     "QK^T, softmax and AV in one kernel;\nthe quantized epilogue writes packed output"),
    (("out_i8", "bias_res", "s1688gemm", "s16816gemm"), "output projection",
     "1x1 projection back to C with bias and\nthe residual add folded into the epilogue"),
    (("elementwise", "Fill", "copy"), "elementwise / residual",
     "whatever the epilogues did not absorb"),
]

# Matching precedence, most specific first. Display order is SLOTS; this is only for assignment.
# Without it `gemm_w8a8_kernel_awq_out_i8` matches the QKV fragment `gemm_w8a8_kernel_awq` and
# the output projection is reported as fused away, while the real QKV GEMM falls through to the
# elementwise catch-all -- which is exactly what a first version of this figure showed.
MATCH_ORDER = [2, 3, 1, 0, 4]

# Logical tensors between the slots. Derived from the block's shape, not captured per-kernel:
# the packed forms are stated so a halved channel count is not read as an error.
FLOW = {
    "fp16": ["x (128,192,32,32) fp16",
             "normalised x, fp16",
             "q/k/v (128,8,1024,24) fp16",
             "attn out (128,8,1024,24) fp16",
             "y (128,1024,192) fp16"],
    "int8_baseline": ["x (128,192,32,32) fp16",
                      "xq (128,192,32,32) int8",
                      "qkv codes (128,1024,8,32) int8",
                      "attn out packed int8",
                      "y (128,1024,192) fp16"],
    "int4_baseline": ["x (128,192,32,32) fp16",
                      "xq (128,32,32,96) int8  = 192 int4 nibbles",
                      "qkv codes (128,1024,8,32) int8",
                      "attn out packed int4",
                      "y (128,1024,192) fp16"],
}


def slot_of(kernel):
    k = kernel.lower()
    for i in MATCH_ORDER:
        if any(f.lower() in k for f in SLOTS[i][0]):
            return i
    return len(SLOTS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out",
                    default="docs/final_report_2026-07-28/plots/fig_attn_pipeline_diagram.png")
    a = ap.parse_args()
    lay = json.load(open(a.layers if os.path.isabs(a.layers)
                         else os.path.join(ROOT, a.layers)))

    per_mode = {}
    for m, _ in MODES:
        e = next(x for x in lay["modes"][m]
                 if x["kind"] == "attention" and tuple(x["x_shape"]) == SHAPE)
        slots = [[] for _ in SLOTS]
        for k in e["kernels"]:
            slots[slot_of(k["kernel"])].append(k)
        per_mode[m] = (e, slots)

    n = len(SLOTS)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 10.4))
    for ax, (mode, lbl) in zip(axes, MODES):
        e, slots = per_mode[mode]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n * 2.05 + 1.5)
        ax.axis("off")
        ax.set_title("%s   —   layer %.0f µs, %d kernels"
                     % (lbl, e["pipeline_us"], len(e["kernels"])),
                     fontsize=12, fontweight="bold", color=COL[mode], pad=14)
        y = n * 2.05 - 0.55
        flow = FLOW[mode]
        ax.text(5.0, y + 1.62, flow[0], fontsize=7.6, ha="center", family="monospace",
                color="#444")
        for i, (frags, title, what) in enumerate(SLOTS):
            ks = sorted(slots[i], key=lambda k: -k["us_per_layer_call"])
            tot = sum(k["us_per_layer_call"] for k in ks)
            h = 1.28
            empty = not ks
            ax.add_patch(FancyBboxPatch(
                (0.4, y), 9.2, h, boxstyle="round,pad=0.07", linewidth=1.5,
                edgecolor=("#c9ced4" if empty else COL[mode]),
                facecolor=("#fbfbfc" if empty else "#f6f9fb"),
                linestyle=("dotted" if empty else "solid")))
            ax.text(0.65, y + h - 0.24, title, fontsize=10,
                    fontweight="bold", color=("#9aa1a8" if empty else "black"))
            if empty:
                ax.text(0.65, y + 0.42, "no separate kernel — folded into the box above",
                        fontsize=7.8, color="#9aa1a8", style="italic")
            else:
                ax.text(9.35, y + h - 0.24, "%.0f µs  %.0f%%"
                        % (tot, tot / e["gpu_us_sum"] * 100), fontsize=9.5, ha="right",
                        color=COL[mode], fontweight="bold")
                yy = y + h - 0.52
                for k in ks[:3]:
                    ax.text(0.65, yy, "%-46s %6.1f µs"
                            % (k["kernel"][:46], k["us_per_layer_call"]),
                            fontsize=7.0, family="monospace", color="#333")
                    yy -= 0.235
                if len(ks) > 3:
                    ax.text(0.65, yy, "+ %d more" % (len(ks) - 3), fontsize=7,
                            family="monospace", color="#777")
                note = what
                if i == 1 and mode == "fp16":
                    # FP16's GroupNorm is two-pass: gn_accum/gn_finalize compute the statistics
                    # (the box above) and the NORMALISATION is folded into this projection --
                    # hence the kernel's name. Saying "FP16 folds GroupNorm in here" would be
                    # wrong: the stats pass is still a separate 111 us of its own.
                    note += ";\nthe normalisation (not the stats pass) is fused in"
                elif i == 1:
                    note += ";\nthe epilogue writes Q/K/V already quantized"
                ax.text(0.65, y + 0.06, note, fontsize=7.4, color="#555", va="bottom")
            y -= 2.05
            if i < n - 1:
                ax.add_patch(FancyArrowPatch((5.0, y + 2.05), (5.0, y + 1.30),
                                             arrowstyle="-|>", mutation_scale=12,
                                             linewidth=1.2, color="#888"))
                ax.text(5.12, y + 1.68, flow[min(i + 1, len(flow) - 1)], fontsize=7.4,
                        family="monospace", color="#555", va="center")

    fig.suptitle("Inside one AttentionBlock (C192/32², T=1024, batch 128): what each kernel does "
                 "and the tensors between them", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.011,
             "Kernels and times are from the layer profile; tensor shapes are the logical "
             "dataflow. INT4 packs two 4-bit values per int8 byte, so a packed channel "
             "dimension is half the logical one.", ha="center", fontsize=8.4, color="#555")
    fig.tight_layout(rect=[0, 0.024, 1, 0.955])
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    fig.savefig(out, dpi=150, facecolor="w")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
