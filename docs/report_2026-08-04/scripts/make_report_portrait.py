"""Portrait measurement report — the single consolidated report for this work.

Supersedes the landscape deck (make_pdf.py, deleted): every figure it carried is here, plus the
attention deep profile and the delta-quantizer ablations that previously lived only in
docs/attn_modiff_profile_2026-08-04/FINDINGS.md and docs/modiff_correctness_2026-08-03/data/.

Content rule for this document: measurement scope, data, figures, and the explanation needed to
read them. No recommendations, no priorities, no verdicts on what to do next.

Data sources, all read at build time or cited inline:
  docs/report_2026-08-04/data/                     e2e wall clock, conv/attn layer + kernel, buckets
  docs/attn_modiff_profile_2026-08-04/data/        attention per-kernel stages, roofline
  docs/modiff_correctness_2026-08-03/data/         delta-quantizer ablations, calibration, samples
  docs/MEASUREMENT_REPORT_2026-08-01.md            whole-model stage table, layer share, nsys idle
"""

import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
D = "docs/report_2026-08-04"
A = "docs/attn_modiff_profile_2026-08-04"
K = "docs/modiff_correctness_2026-08-03"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FS = (8.27, 11.69)              # A4 portrait
INK, MUTE = "#1a1a1a", "#6b6b6b"
BASE_C, MOD_C, FP_C = "#4C72B0", "#C44E52", "#8C8C8C"
WARN = "#B8860B"
OK = "#55A868"
I4_C = "#6f9bd1"
_n = [0]


def _load(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".json"):
        return json.load(open(path))
    return list(csv.DictReader(open(path)))


def jload(f):
    return _load(f"{D}/data/{f}")


cload = jload
aload = lambda f: _load(f"{A}/data/{f}")
kload = lambda f: _load(f"{K}/data/{f}")


def save(pdf, fig):
    pdf.savefig(fig)
    _n[0] += 1
    os.makedirs(f"{D}/plots_portrait", exist_ok=True)
    fig.savefig(f"{D}/plots_portrait/p{_n[0]:02d}.png", dpi=92)
    plt.close(fig)


#: Bottom edge of the current page's scope block, so tbl() can tell whether it would overprint it.
_SCOPE_BOT = [0.912]


def page(sec, title, scope=None):
    fig = plt.figure(figsize=FS)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.965, sec, fontsize=9, color=MUTE, va="top")
    fig.text(0.08, 0.945, title, fontsize=15.5, color=INK, weight="bold", va="top")
    _SCOPE_BOT[0] = 0.912
    if scope:
        fig.text(0.08, 0.912, scope, fontsize=8.3, color=MUTE, va="top", linespacing=1.55)
        _SCOPE_BOT[0] = 0.912 - (scope.count("\n") + 1) * 8.3 * 1.55 / 841.7
    return fig


_TOPCLASH = []


def tbl(fig, rect, header, rows, colw=None, fs=7.6, hi=None, warn=None):
    #: A table is drawn from the TOP of its rect downward. A rect top above ~0.872 runs into the
    #: scope block under the page title, which renders as overprinted text rather than an error.
    if rect[1] + rect[3] > _SCOPE_BOT[0] + 0.014:
        _TOPCLASH.append((_n[0] + 1, str(header[0])[:34], round(rect[1] + rect[3], 3)))
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                 colWidths=colw)
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.42)
    for j in range(len(header)):
        t[0, j].set_facecolor("#eceff4")
        t[0, j].set_text_props(weight="bold", color=INK)
    for i in range(1, len(rows) + 1):
        for j in range(len(header)):
            t[i, j].set_edgecolor("#dcdcdc")
        if hi and (i - 1) in hi:
            for j in range(len(header)):
                t[i, j].set_facecolor("#eef6ee")
        if warn and (i - 1) in warn:
            for j in range(len(header)):
                t[i, j].set_facecolor("#fdf6e3")
    return ax


#: Overflow guard. va="top" text grows DOWNWARD, so a block that is a few lines too long silently
#: runs off the page bottom -- it renders without error and is only visible in the PNG. One A4 page
#: is 841.7 pt, so a line at font size fs costs fs*1.68/841.7 in figure fraction.
_LINE = lambda fs: fs * 1.68 / 841.7
_OVERFLOW = []


def body(fig, y, text, fs=8.4, color=INK):
    end = y - _LINE(fs) * (text.count("\n") + 1)
    if end < 0.035:
        _OVERFLOW.append((_n[0] + 1, f"{text.splitlines()[0][:46]}...", round(end, 3)))
    fig.text(0.08, y, text, fontsize=fs, color=color, va="top", linespacing=1.68)


def head(fig, y, text, color=INK, fs=10.5):
    fig.text(0.08, y, text, fontsize=fs, color=color, weight="bold", va="top")


def img(fig, path, top, width=0.88, left=0.06, max_h=0.55):
    """Place a PNG with its aspect preserved, top edge at figure-fraction `top`."""
    if not os.path.exists(path):
        body(fig, top, f"MISSING: {path}", color=MOD_C)
        return top - 0.02
    im = plt.imread(path)
    h, w = im.shape[:2]
    ah = min(max_h, width * (h / w) * (FS[0] / FS[1]))
    ax = fig.add_axes([left, top - ah, width, ah])
    ax.imshow(im)
    ax.axis("off")
    return top - ah


def bare(ax, axis="y"):
    ax.grid(axis=axis, alpha=0.3, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =====================================================================================
# Front matter
# =====================================================================================
def p_cover(pdf):
    fig = plt.figure(figsize=FS)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.92, "Quantized LDM inference on A40", fontsize=21, color=INK,
             weight="bold", va="top")
    fig.text(0.08, 0.877, "Baseline fusion, MoDiff, and the attention profile — measurement report",
             fontsize=12.5, color=MUTE, va="top")
    body(fig, 0.825,
         "Model      LSUN-churches LDM, real checkpoint. 21 AttentionBlocks, 35 ResBlocks.\n"
         "Hardware   NVIDIA A40 (GA102). 696 GB/s, 299.4 TOPS int8 dense, 149.7 TFLOPS fp16.\n"
         "Modes      fp16 · W8A8 (int8) · W4A4 (int4), each with MoDiff off and on.\n"
         "Sampler    DDIM, eta 0. End-to-end figures at 200 steps, batch 128.\n"
         "Date       2026-08-04", fs=9)
    head(fig, 0.705, "Contents", fs=11.5)
    body(fig, 0.678,
         "   1     Scopes, references, measurement discipline\n"
         "   2     Corrections register — every figure withdrawn in this work\n"
         "   3-8   Baseline: fusion scheme, end-to-end, conv, attention, whole-model profile\n"
         "   9     Attention deep profile: per-kernel stages, fusion accounting, GEMM roofline\n"
         "   10-13 MoDiff: fusion scheme, end-to-end, GPU-time profile\n"
         "   14-19 Delta-quantizer ablations, GroupNorm variants, calibration health, invariants\n"
         "   20-23 Accuracy vs step count, equal-accuracy speedup, decoded samples\n"
         "   24-25 Kernel surface, and what is not established", fs=8.8)
    head(fig, 0.475, "Statistics convention", fs=11.5)
    body(fig, 0.448,
         "   mean +- 95% CI     over independent timing rounds, CUDA-event timed\n"
         "   CV                 stdev / mean across rounds\n"
         "   spread             (max - min) / mean across rounds\n"
         "   ratios             always name the reference in the column header\n\n"
         "Differences smaller than the reported spread are not claims.\n\n"
         "Two batch sizes appear and are labelled on every table. End-to-end headline figures are\n"
         "batch 128, DDIM 200. The ablation sweeps in sections 14-20 are batch 8, DDIM 50 — chosen\n"
         "so a sweep fits in one session. ms/step is NOT comparable between the two.", fs=8.8)
    fig.text(0.08, 0.085, "Data and scripts: docs/report_2026-08-04/{data,scripts}/", fontsize=7.6,
             color=MUTE, va="top")
    save(pdf, fig)


def p_method(pdf):
    fig = page("Section 1", "Scopes, reference paths, measurement discipline")
    body(fig, 0.895,
         "Three scopes appear in this report. They are not interchangeable.", fs=8.6)
    tbl(fig, [0.08, 0.685, 0.86, 0.15],
        ["scope", "what is timed", "used for"],
        [["end-to-end", "full DDIM sample loop, wall clock", "headline speedup"],
         ["layer", "one nn.Module forward, CUDA events", "conv / attention speedup"],
         ["kernel", "one CUDA kernel or fused group", "attribution, roofline"]],
        colw=[0.16, 0.44, 0.26], fs=8)
    body(fig, 0.645,
         "Reference paths — what each quantized number is divided by\n\n"
         "   fp16 attention    F.scaled_dot_product_attention with NO pinned backend, i.e. PyTorch\n"
         "                     picks flash. This is what the production forward does\n"
         "                     (token_major_attention._SDPA_CTX returns nullcontext by default).\n"
         "                     SDPBackend.MATH is NOT the reference; it is 6.04x slower here and\n"
         "                     appears only as a labelled control (section 7).\n"
         "   fp16 conv         cuDNN fp16 convolution, channels_last.\n"
         "   own baseline      for MoDiff rows: the same bit-width with MoDiff disabled, so the\n"
         "                     comparison isolates MoDiff and not quantization.\n"
         "   roofline          for section 9c: max(bytes/696 GB/s, ops/peak) for that same kernel,\n"
         "                     so no cross-precision attribution is involved.\n\n"
         "Warm-up discipline\n\n"
         "   end-to-end        1 full sampling run discarded, then median of 3. Required: the\n"
         "                     quantized attention blocks self-calibrate over their first forwards,\n"
         "                     and run 1 is several x worse than run 2.\n"
         "   layer / kernel    30-60 warm-up calls per round, then 60-200 timed calls, 5-8 rounds.\n\n"
         "Calibration artifacts\n\n"
         "   integration/calibration/int{8,4}_calibration_realckpt.pt. The un-suffixed files were\n"
         "   fitted against an 856-byte stub checkpoint and give latent relL2 0.88 / 3.02 with real\n"
         "   weights; no accuracy figure in this report uses them. The section 9 timing run does use\n"
         "   them, deliberately: scale VALUES do not affect kernel selection or duration, and it\n"
         "   reports no accuracy number.\n\n"
         "Thermal note\n\n"
         "   One benchmark run was discarded after nvidia-smi showed SW power cap at 304 W and fp16\n"
         "   at 223 ms/step instead of 102. fp16 exercises none of the changed code. All figures here\n"
         "   come from runs whose spread is <= 0.9%.", fs=8.4)
    save(pdf, fig)


def p_corrections(pdf):
    fig = page("Section 2", "Corrections register",
               "Every figure produced during this work and later withdrawn, with the defect and the\n"
               "replacement. Listed so that no superseded number is quoted from an earlier draft.")
    tbl(fig, [0.04, 0.665, 0.93, 0.22],
        ["withdrawn figure", "defect", "replacement"],
        [["attention 4.54x\n(kernel scope)",
          "fp16 reference pinned to SDPBackend.MATH,\nwhich materializes [BH,T,T]: 6.04x too slow",
          "0.76x microbench,\n1.15-1.84x at layer scope"],
         ["out projection\n0.55x (int8 1.8x slower)",
          "int8 kernel has bias+residual in its epilogue;\nfp16's has neither",
          "paired: fp16 459.8 vs\nint8 345.7 us = 1.33x"],
         ["projection GEMM at\n9.3% vs 33.7% of peak",
          "peak TOPS is the wrong denominator: at K=C\nthese GEMMs are roofline bound",
          "% of own roofline:\n24-52% vs fp16 47-75%"],
         ["all per-stage\nfp16/int8 ratios",
          "the fp16 column does not balance — its work\nwould need 939 GB/s against a 696 peak",
          "TOTALs only; plus the\nint8 roofline (section 9c)"],
         ["dynamic delta loses\nat W8A8 (0.178->0.231)",
          "measured on run 1; quantized attention\nself-calibrates over its first forwards",
          "steady state 0.0399,\n5.3x better than run 1"],
         ["static delta table\nloses (0.178->0.214)",
          "same first-run artifact",
          "steady state: table ON\n0.0385 vs OFF 0.1864"],
         ["free delta-absmax\nreporting, -1.1% time",
          "publishes a scale used up to 2x refresh steps\nlater; W4A4 diverges 0.4746 -> 11.6553",
          "MODIFF_DELTA_REPORT=0\nby default"]],
        colw=[0.20, 0.42, 0.24], fs=6.5, warn=[0, 1, 2, 3, 4, 5, 6])
    body(fig, 0.705,
         "Common cause of rows 2-4: comparing a FUSED int8 kernel against an UNFUSED fp16 one, stage\n"
         "by stage. When one side folds bias / residual / normalization into a producer kernel and the\n"
         "other pays them separately, per-stage attribution is not a measurement. The check that\n"
         "catches it is a byte count against peak bandwidth (section 9b).\n\n"
         "Common cause of rows 5-6: the first sampling run of a quantized mode is not steady state.\n"
         "Every end-to-end figure in this report discards run 1.\n\n"
         "Row 7 was found by an end-to-end latent check after 16/16 kernel unit tests passed. The\n"
         "hazard is cross-kernel ordering WITHIN a step — the reporting kernel quantizes with the\n"
         "current scale but publishes the next one into the buffer the following conv reads as its\n"
         "dequant alpha. Single-launch tests cannot observe it.", fs=8.4)
    head(fig, 0.415, "Figures that were checked and stand")
    body(fig, 0.388,
         "   whole-model stage TOTALs        every kernel is inside them regardless of bucketing\n"
         "   end-to-end wall clock           profiler-free, spread <= 0.9%, run 1 discarded\n"
         "   conv layer 1.65x / 3.00x        single kernel each side, no fusion asymmetry\n"
         "   MoDiff inside attention 0.99-1.03x   int8-vs-int8, unaffected by any of the above\n"
         "   GPU busy 97.5-98.5%             two independent sources agree (section 8c)", fs=8.4)
    save(pdf, fig)


# =====================================================================================
# Baseline
# =====================================================================================
def p_base_scheme(pdf):
    fig = page("Section 3", "Baseline fusion scheme (MoDiff off)",
               "Each row is one CUDA kernel in the production quantized forward.")
    tbl(fig, [0.06, 0.615, 0.90, 0.26],
        ["site", "fused operations", "kernel symbol"],
        [["ResBlock entry", "GroupNorm + SiLU + mod + SmoothQuant + quantize",
          "group_norm_silu_quantize[_pack]_nhwc"],
         ["updown ResBlock", "GN + SiLU + 2x resize + quantize", "group_norm_silu_quantize_resize_nhwc"],
         ["Conv", "implicit-GEMM int8/int4 + dequant epilogue", "ImplicitGemmConvolution (CUTLASS EVT)"],
         ["Conv + skip", "conv + bias + residual in the epilogue", "..._deepfuse_bias_residual_fp16"],
         ["Attention qkv", "GroupNorm folded into the qkv 1x1 conv", "fused_gn_qkv[_i8evt]"],
         ["Attention core", "QK^T + softmax + AV, int8/int4 flash", "flash_attn_int{8,4}_vt[_static][_qout]"],
         ["Attention proj", "GEMM + dequant + bias + residual", "gemm_w{8a8,4a4}_awq_bias_res"],
         ["Up/Downsample", "resize + quantize (no cache)", "{upsample2x,avgpool2x}_quantize_noahat"]],
        colw=[0.15, 0.40, 0.35], fs=7.4)
    body(fig, 0.575,
         "Design rule: a quantize is never a standalone pass. It is folded into whichever kernel\n"
         "produces the tensor, so the fp16 intermediate is never written to memory. GroupNorm\n"
         "statistics are computed inside the fused kernel and never materialized.\n\n"
         "Consequence for every profile in this report: there is no 'quantize' line item in the\n"
         "baseline breakdown. Its cost is inside the GN+quantize and conv-epilogue lines. This is also\n"
         "why the int8 column of section 9a has no elementwise kernel at all, and why the fp16 and\n"
         "int8 columns of a per-stage table are not a like-for-like pair.", fs=8.4)
    save(pdf, fig)


def p_e2e(pdf, e2e, which):
    base = which == "baseline"
    fig = page("Section 4" if base else "Section 11",
               "Baseline end-to-end" if base else "MoDiff end-to-end, identical step count",
               "Scope: end-to-end. DDIM 200 steps, batch 128, median of 3 after 1 warm-up run.\n"
               "Ratios vs fp16 and, for MoDiff rows, vs the same bit-width with MoDiff off.")
    if not e2e:
        body(fig, 0.85, "NOT MEASURED: data/e2e_wallclock.json absent", color=MOD_C)
        save(pdf, fig); return
    keys = (["fp16", "int8_baseline", "int4_baseline"] if base else
            ["fp16", "int8_baseline", "int8 modiff static", "int8 dynamic K=1", "int8 dynamic K=4",
             "int4_baseline", "int4 modiff static", "int4 dynamic K=1", "int4 dynamic K=4"])
    keys = [k for k in keys if k in e2e]
    fp = e2e["fp16"]["ms_per_step"]
    rows = []
    for k in keys:
        v = e2e[k]
        own = (e2e["int8_baseline"]["ms_per_step"] if "int8" in k else
               e2e["int4_baseline"]["ms_per_step"] if "int4" in k else fp)
        rows.append([k, f"{v['ms_per_step']:.2f}", f"{v['ms_per_step'] * 200 / 1000:.2f}",
                     f"{fp / v['ms_per_step']:.3f}x",
                     "—" if k == "fp16" else f"{own / v['ms_per_step']:.3f}x",
                     f"{v.get('spread_pct', '?')}%"])
    hi = [i for i, k in enumerate(keys) if "K=4" in k]
    # The table is drawn from the TOP of its rect downward and its cell height scales with the rect,
    # so the rect must be sized per row count and anchored by its top edge -- a fixed y0 with a
    # row-dependent height pushes a 9-row table off the top of the page.
    th = 0.0245 * len(rows) + 0.022
    tbl(fig, [0.06, 0.855 - th, 0.90, th],
        ["configuration", "ms/step", "s/sample", "vs fp16", "vs own baseline", "spread"],
        rows, colw=[0.24, 0.13, 0.13, 0.13, 0.17, 0.11], fs=8, hi=hi)
    ytop = 0.855 - th - 0.03
    ax = fig.add_axes([0.14, ytop - 0.20, 0.74, 0.185])
    lab = [k.replace(" dynamic ", "\nMoD dyn ").replace(" modiff static", "\nMoD static")
            .replace("_baseline", " base") for k in keys]
    sp = [fp / e2e[k]["ms_per_step"] for k in keys]
    col = [FP_C if k == "fp16" else (MOD_C if ("dynamic" in k or "modiff" in k) else BASE_C)
           for k in keys]
    b = ax.bar(range(len(sp)), sp, color=col, width=0.62, zorder=3)
    for r, v in zip(b, sp):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}x", ha="center", va="bottom",
                fontsize=7.6, weight="bold")
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, fontsize=6.4 if not base else 8)
    ax.set_ylabel("speedup vs fp16", fontsize=8.5)
    ax.axhline(1.0, ls="--", lw=1, color=MUTE)
    ax.set_ylim(0, max(sp) * 1.22)
    bare(ax)
    if base:
        body(fig, ytop - 0.245,
             "Quantization alone buys 1.44x at W8A8 and 1.77x at W4A4 end to end. The conv is 37% of\n"
             "the fp16 model (section 8a); attention and the fp16 tails do not scale with weight\n"
             "bit-width, so the end-to-end ratio is well below the arithmetic ratio.", fs=8.4)
    else:
        body(fig, ytop - 0.245,
             "Reading the rows. 'modiff static' uses one delta scale per layer per step index from a\n"
             "calibrated table; 'dynamic K' recomputes scale = Q/max|delta| on device, refreshing the\n"
             "reduction every K-th step. K=1 refreshes every step and costs 6.4 ms/step more than K=4\n"
             "at int8; K=4 is the shipped default.\n\n"
             "At an identical step count MoDiff costs 8-9% per step at both bit-widths. That is the\n"
             "a_hat / o_hat state traffic required by Eqs 9-10: per big ResBlock conv (C=192, 32^2,\n"
             "batch 128) the baseline moves 151 MB per step and MoDiff moves 352 MB (2.33x). About\n"
             "2.2 ms/step of it is a floor no fusion removes, because a_hat and o_hat are\n"
             "full-precision state tensors that must be read and written every step.\n\n"
             "Section 21 holds the step count free, which is the axis the paper's claim lives on.",
             fs=8.4)
    save(pdf, fig)


def p_conv(pdf, conv):
    fig = page("Section 5", "Conv layer speedup",
               "Scope: layer. Real churches UNet conv shapes, batch 128, CUDA-event median,\n"
               "200 timed calls x 5 rounds after 50 warm-up. Reference: cuDNN fp16, channels_last.")
    if not conv:
        body(fig, 0.85, "NOT MEASURED", color=MOD_C); save(pdf, fig); return
    rows = []
    for r in conv:
        rows.append([r["shape"], r["fp16_us"], r["int8_baseline_us"], r["int4_baseline_us"],
                     r["int8_modiff_us"], r["int4_modiff_us"],
                     r["int8_baseline_vs_fp16"], r["int4_baseline_vs_fp16"],
                     r["int8_modiff_vs_fp16"], r["int4_modiff_vs_fp16"]])

    def mean(k):
        v = [float(r[k]) for r in conv if r.get(k)]
        return sum(v) / len(v)
    means = {k: mean(k) for k in ("int8_baseline_vs_fp16", "int4_baseline_vs_fp16",
                                  "int8_modiff_vs_fp16", "int4_modiff_vs_fp16")}
    rows.append(["MEAN", "", "", "", "", ""] + [f"{means[k]:.3f}" for k in
                 ("int8_baseline_vs_fp16", "int4_baseline_vs_fp16",
                  "int8_modiff_vs_fp16", "int4_modiff_vs_fp16")])
    tbl(fig, [0.04, 0.465, 0.93, 0.38],
        ["shape", "fp16 us", "i8 base", "i4 base", "i8 MoD", "i4 MoD",
         "i8b/fp16", "i4b/fp16", "i8M/fp16", "i4M/fp16"],
        rows, colw=[0.155] + [0.083] * 9, fs=6.8, hi=[len(rows) - 1])

    ax = fig.add_axes([0.16, 0.215, 0.70, 0.19])
    lab = ["int8\nbaseline", "int4\nbaseline", "int8\n+MoDiff", "int4\n+MoDiff"]
    val = [means["int8_baseline_vs_fp16"], means["int4_baseline_vs_fp16"],
           means["int8_modiff_vs_fp16"], means["int4_modiff_vs_fp16"]]
    b = ax.bar(range(4), val, color=[BASE_C, I4_C, MOD_C, "#d98b8e"], width=0.6, zorder=3)
    for r, v in zip(b, val):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}x", ha="center", va="bottom",
                fontsize=8.5, weight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("mean speedup vs cuDNN fp16", fontsize=8.5)
    ax.axhline(1.0, ls="--", lw=1, color=MUTE)
    ax.set_ylim(0, max(val) * 1.22)
    bare(ax)
    body(fig, 0.178,
         "MoDiff's conv is slower than the baseline's (1.30x vs 1.65x at int8). It is the same\n"
         "implicit-GEMM with an in-place o_hat read-modify-write added to the epilogue, so it moves\n"
         "one extra full-precision tensor per call. MoDiff Eq 10 requires that accumulation.", fs=8.4)
    save(pdf, fig)


def p_attn_layer(pdf, lay):
    fig = page("Section 6", "Attention layer speedup",
               "Scope: layer. The whole AttentionBlock module as each mode produced it (fused\n"
               "GN->qkv, quantized core, fused proj epilogue), batch 128, CUDA events,\n"
               "60 timed calls x 8 rounds after 30 warm-up.")
    if not lay:
        body(fig, 0.85, "NOT MEASURED: data/attn_layer_speed.csv absent", color=MOD_C)
        save(pdf, fig); return
    REF = {"C192/T1024": (2978.5, 2518.9, 2473.6), "C384/T256": (1073.5, 841.9, 773.3),
           "C384/T64": (429.5, 228.7, 210.5), "C768/T16": (220.9, 184.8, 155.4),
           "C768/T4": (97.5, 75.2, 56.6)}
    rows, warn = [], []
    mine_r, ref_r, shapes = [], [], []
    for i, r in enumerate(lay):
        sh = r["shape"]
        ref = REF.get(sh)
        mine = float(r["int8_baseline_vs_fp16"]) if r.get("int8_baseline_vs_fp16") else None
        refr = (ref[0] / ref[1]) if ref else None
        dev = abs(mine - refr) / refr * 100 if (mine and refr) else None
        if dev is not None and dev > 15:
            warn.append(i)
        shapes.append(sh); mine_r.append(mine or 0); ref_r.append(refr or 0)
        rows.append([sh, r["fp16_us"], f"{float(r['fp16_cv_pct']):.2f}", r["int8_baseline_us"],
                     r["int4_baseline_us"], r["int8_us"],
                     r.get("int8_baseline_vs_fp16", ""), r.get("int4_baseline_vs_fp16", ""),
                     f"{refr:.3f}" if refr else "—", f"{dev:.0f}%" if dev is not None else "—"])
    tbl(fig, [0.04, 0.710, 0.93, 0.135],
        ["shape", "fp16 us", "CV%", "i8 base", "i4 base", "i8 MoD",
         "i8b/fp16", "i4b/fp16", "08-01 i8b/fp16", "dev"],
        rows, colw=[0.135, 0.085, 0.062, 0.085, 0.085, 0.085, 0.09, 0.09, 0.115, 0.06],
        fs=6.7, warn=warn)

    x = range(len(shapes))
    ax = fig.add_axes([0.11, 0.535, 0.36, 0.16])
    w = 0.26
    for i, (k, c, lb) in enumerate([("fp16_us", FP_C, "fp16"), ("int8_baseline_us", BASE_C, "int8"),
                                    ("int4_baseline_us", I4_C, "int4")]):
        ax.bar([j + (i - 1) * w for j in x], [float(r[k]) for r in lay], width=w, color=c,
               label=lb, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels(shapes, rotation=38, ha="right", fontsize=6.4)
    ax.set_ylabel("us / call (log)", fontsize=7.6)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.3, zorder=0, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax2 = fig.add_axes([0.60, 0.535, 0.34, 0.16])
    ax2.bar([j - 0.19 for j in x], mine_r, width=0.38, color=BASE_C, label="this report", zorder=3)
    ax2.bar([j + 0.19 for j in x], ref_r, width=0.38, color="#B0B0B0", label="08-01 report", zorder=3)
    ax2.axhline(1.0, ls="--", lw=1, color=MUTE)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(shapes, rotation=38, ha="right", fontsize=6.4)
    ax2.set_ylabel("int8 base / fp16", fontsize=7.6)
    ax2.legend(fontsize=6.8, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    bare(ax2)
    ax2.annotate("only\ndisagreement", (4, mine_r[4]), (3.1, mine_r[4] * 0.72), fontsize=6.2,
                 color=WARN, arrowprops=dict(arrowstyle="->", color=WARN, lw=0.9))

    body(fig, 0.485,
         "Right panel: 4 of 5 shapes agree with MEASUREMENT_REPORT_2026-08-01 within 3%. C768/T4\n"
         "disagrees by 84%; section 8d shows that shape is dispatch-bound (GPU/issue 0.35-0.47), so\n"
         "what it measures depends on whether the CPU or the GPU is the constraint at the moment of\n"
         "measurement — in isolation versus surrounded by other work in a real run. It is 1 block\n"
         "out of 21 and it is not reconciled.\n\n"
         "The 'i8 MoD' column matches 'i8 base' to within noise. Attention has no MoDiff variant by\n"
         "construction: the qkv epilogue exists to emit quantized codes, and under MoDiff the GEMM\n"
         "produces an accumulator increment, so there are none to emit. Section 9d measures this\n"
         "directly rather than asserting it.", fs=8.4)
    save(pdf, fig)


def p_attn_micro(pdf, micro):
    fig = page("Section 7", "Attention kernel microbenchmark, and the retracted 4.54x",
               "Scope: kernel. GroupNorm + quantize + attention core only — NOT the projections.\n"
               "Both fp16 references were run in the same script so the inflation is measured, not\n"
               "inferred.")
    tbl(fig, [0.06, 0.735, 0.90, 0.115],
        ["figure", "scope", "fp16 reference", "status"],
        [["4.54x", "GN + quantize + core", "SDPBackend.MATH", "WITHDRAWN — reference 6.04x too slow"],
         ["0.76x", "GN + quantize + core", "default (flash)", "correct at this scope"],
         ["1.15-1.84x", "whole attention layer", "default (flash)", "the layer-scope figure (sec 6)"]],
        colw=[0.13, 0.24, 0.20, 0.33], fs=7.3, hi=[2], warn=[0])
    if micro:
        rows = [[r.get("shape", ""), r.get("fp16_us", ""), r.get("int8_us", ""),
                 r.get("int8_vs_fp16", "")] for r in micro]
        tbl(fig, [0.14, 0.545, 0.72, 0.135],
            ["shape", "fp16 us (flash)", "int8 us", "int8 / fp16"],
            rows, colw=[0.20, 0.20, 0.18, 0.18], fs=7.4)
    body(fig, 0.475,
         "Why the two scopes differ by 1.5-2.4x. The microbenchmark charges int8 for a standalone\n"
         "Q/K/V quantize that the production forward never runs — in production the qkv epilogue\n"
         "emits int8 codes directly — and it excludes the qkv and output projections entirely.\n\n"
         "The control, from the same script: with the reference pinned to MATH the int8 ratio reads\n"
         "4.60x instead of 0.76x. 6.04x of inflation from the choice of reference alone. MATH\n"
         "materializes the [BH,T,T] score matrix; flash does not. token_major_attention.py:64-73\n"
         "documents this trap in-source.\n\n"
         "Section 9 replaces this microbenchmark for attribution purposes: it profiles the real\n"
         "attention module per kernel, in all five modes, rather than timing a subset of it.", fs=8.4)
    save(pdf, fig)


# ---- whole-model profile (data from the 08-01 report) --------------------------------
STAGE = {                     # ms per batch, 200 steps, profiler self-time scaled to measured wall
    "attention core":            (2252.3, 1765.6, 1688.3),
    "QKV / output projection":   (1773.5, 1850.5, 1772.9),
    "convolution":               (7480.1, 5359.0, 2811.0),
    "GroupNorm + quantize":      (3975.6, 3454.8, 3563.1),
    "elementwise / copies / other": (4826.2, 1703.4, 1711.9),
}
STAGE_TOTAL = (20307.7, 14133.2, 11547.3)
LAYER_SHARE = {
    "resblock_plain":   ((10245, 50), (6990, 49), (5126, 45)),
    "attention":        ((4830, 24), (3907, 28), (3745, 33)),
    "resblock_updown":  ((4128, 20), (2299, 16), (1676, 15)),
    "outside layers":   ((1118, 5), (969, 7), (970, 8)),
}
IDLE = [("FP16", 98.5, 1.5, 0.80, 135, 0.6, 101.37, 101.12),
        ("INT8 (MoDiff off)", 98.1, 1.9, 0.80, 149, 1.0, 70.76, 70.28),
        ("INT4 (MoDiff off)", 97.5, 2.5, 0.80, 161, 1.3, 57.46, 56.99)]
ATTN_RUN = [
    ("C192/T1024", 2978.5, 272.7, 11.01, 2518.9, 178.9, 14.11, 2473.6, 156.1, 15.89),
    ("C384/T256", 1073.5, 267.0, 4.04, 841.9, 173.4, 4.87, 773.3, 153.0, 5.07),
    ("C384/T64", 429.5, 278.7, 1.55, 228.7, 170.9, 1.34, 210.5, 149.0, 1.42),
    ("C768/T16", 220.9, 285.5, 0.78, 184.8, 157.1, 1.18, 155.4, 139.0, 1.12),
    ("C768/T4", 97.5, 283.2, 0.35, 75.2, 160.9, 0.47, 56.6, 139.6, 0.41),
]


def p_stage(pdf):
    fig = page("Section 8a", "Whole-model time by stage",
               "Scope: kernel, aggregated to op classes. Profiler self-time SCALED to the measured\n"
               "profiler-free wall clock, ms per batch of 200 steps. Source: 08-01 report /\n"
               "final_report_2026-07-28/data/e2e_plots_b128_run*.json. Not re-measured.")
    rows = [[k, f"{a:.1f}", f"{b:.1f}", f"{c:.1f}", f"{c - b:+.1f}"]
            for k, (a, b, c) in STAGE.items()]
    rows.append(["total", f"{STAGE_TOTAL[0]:.1f}", f"{STAGE_TOTAL[1]:.1f}",
                 f"{STAGE_TOTAL[2]:.1f}", f"{STAGE_TOTAL[2] - STAGE_TOTAL[1]:+.1f}"])
    tbl(fig, [0.06, 0.715, 0.90, 0.145],
        ["stage", "FP16", "INT8 base", "INT4 base", "INT4 - INT8"],
        rows, colw=[0.30, 0.15, 0.15, 0.15, 0.15], fs=7.4, hi=[len(rows) - 1])

    ax = fig.add_axes([0.30, 0.450, 0.64, 0.235])
    ks_ = list(STAGE)[::-1]
    y = range(len(ks_))
    h = 0.26
    for i, (mi, c, lb) in enumerate([(0, FP_C, "FP16"), (1, BASE_C, "INT8 base"),
                                     (2, I4_C, "INT4 base")]):
        ax.barh([j + (i - 1) * h for j in y], [STAGE[k][mi] for k in ks_], height=h,
                color=c, label=lb, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([k.replace(" / ", "/\n") for k in ks_], fontsize=7)
    ax.set_xlabel("ms per batch (200 steps)", fontsize=8)
    ax.legend(fontsize=7.2, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    bare(ax, "x")

    body(fig, 0.415,
         "Convolution carries the change: 7480 -> 5359 -> 2811 ms, -2121 at int8 and a further -2548\n"
         "at int4. The other four stages are close to flat.\n\n"
         "   QKV / output projection   1773 -> 1850 ms.  Not a like-for-like pair. int8's out-proj\n"
         "                             GEMM adds bias AND the residual in its epilogue; fp16 pays\n"
         "                             that residual in the ELEMENTWISE row. Section 9b locates\n"
         "                             >=453 ms/batch of fp16 attention elementwise work that int8\n"
         "                             folds into this row, against a +77 ms rise here.\n"
         "   GroupNorm + quantize      3976 -> 3455 -> 3563 ms.  int4 is worse than int8: the int4\n"
         "                             nibble pack costs more than the narrower store saves.\n"
         "   elementwise / copies      4826 -> 1703 ms.  This is the fusion work, not the arithmetic:\n"
         "                             fused epilogues delete the fp16 intermediates these copies\n"
         "                             moved. It is the second-largest single change in the table.\n\n"
         "Consistency check, as the 08-01 report states it: the stage sum equals the measured wall\n"
         "clock to +0.00% in all three modes. Essentially none of the end-to-end time is unattributed.",
         fs=8.2)
    save(pdf, fig)


def p_layer_share(pdf):
    fig = page("Section 8b", "Share of end-to-end time by layer type",
               "Scope: layer, attributed by NVTX range from the nsys traces and scaled to the mean\n"
               "profiler-free wall time. ms per batch (share of end-to-end). Source: 08-01 report.")
    rows = [[k, f"{a} ({ap}%)", f"{b} ({bp}%)", f"{c} ({cp}%)"]
            for k, ((a, ap), (b, bp), (c, cp)) in LAYER_SHARE.items()]
    tbl(fig, [0.10, 0.70, 0.80, 0.15],
        ["layer type", "FP16", "INT8 base", "INT4 base"], rows,
        colw=[0.24, 0.19, 0.19, 0.19], fs=8)
    ax = fig.add_axes([0.14, 0.40, 0.74, 0.24])
    modes = ["FP16", "INT8", "INT4"]
    bottom = [0, 0, 0]
    cols = [BASE_C, MOD_C, OK, WARN]
    for j, k in enumerate(LAYER_SHARE):
        v = [LAYER_SHARE[k][i][0] for i in range(3)]
        ax.bar(modes, v, bottom=bottom, color=cols[j], label=k, width=0.55, zorder=3)
        bottom = [bottom[i] + v[i] for i in range(3)]
    ax.set_ylabel("ms / batch (200 steps)", fontsize=8.5)
    ax.legend(fontsize=7.4, frameon=False, ncol=2, loc="upper right")
    bare(ax)
    body(fig, 0.34,
         "ResBlocks are half the model and are where quantization pays (10245 -> 6990 -> 5126 ms).\n\n"
         "Attention's SHARE grows as the rest gets faster: 24% -> 28% -> 33%. In absolute terms it\n"
         "falls (4830 -> 3907 -> 3745 ms); it is the part quantization helps least, so it dominates\n"
         "more as conv shrinks. Even eliminating attention entirely at int4 would leave 67% of the\n"
         "time.\n\n"
         "'outside layers' is 5-8%: the input/output convs, timestep embedding, and the DDIM update\n"
         "arithmetic itself. Its share grows for the same reason.", fs=8.3)
    save(pdf, fig)


def p_idle(pdf):
    fig = page("Section 8c", "GPU occupancy and launch gaps",
               "Scope: timeline. From the nsys traces, 3 per mode. This is the direct measurement\n"
               "behind every launch/gap figure in this report. Source: 08-01 report.")
    rows = [[m, f"{b:.1f}%", f"{i:.1f}%", f"{g:.2f} us", f"{n}", f"{sp:.1f}%",
             f"{w:.2f}", f"{k:.2f}", f"{(w - k):.2f}"]
            for (m, b, i, g, n, sp, w, k) in IDLE]
    tbl(fig, [0.04, 0.740, 0.93, 0.105],
        ["mode", "busy", "idle", "med gap", "gaps>50us", "their span",
         "wall ms/st", "gpu ms/st", "diff"],
        rows, colw=[0.20, 0.075, 0.075, 0.085, 0.095, 0.10, 0.095, 0.095, 0.07], fs=7.1)

    md = [m.split(" ")[0] for (m, *_) in IDLE]
    ax = fig.add_axes([0.11, 0.535, 0.36, 0.19])
    busy = [b for (_, b, *_) in IDLE]
    idle = [i for (_, _, i, *_) in IDLE]
    ax.bar(md, busy, color=BASE_C, label="busy", width=0.55, zorder=3)
    ax.bar(md, idle, bottom=busy, color=MOD_C, label="idle", width=0.55, zorder=3)
    for i, d in enumerate(idle):
        ax.text(i, 100.6, f"{d:.1f}% idle", ha="center", fontsize=7, color=MOD_C, weight="bold")
    ax.set_ylim(90, 103)
    ax.set_ylabel("% of timeline span", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    bare(ax)
    ax.set_title("y axis starts at 90%", fontsize=6.8, color=MUTE, loc="left")

    ax2 = fig.add_axes([0.60, 0.535, 0.34, 0.19])
    lp = [677.5, 517.6, 1086.6]
    b2 = ax2.bar(md, lp, color=[FP_C, BASE_C, I4_C], width=0.55, zorder=3)
    for r, v in zip(b2, lp):
        ax2.text(r.get_x() + r.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=7.5, weight="bold")
    ax2.set_ylabel("kernel launches / step", fontsize=8)
    ax2.set_ylim(0, max(lp) * 1.22)
    bare(ax2)

    body(fig, 0.49,
         "The GPU is saturated: 97.5-98.5% busy, median inter-kernel gap 0.80 us. This pipeline is\n"
         "not dispatch-bound, so CUDA-graph capture or launch batching has at most 1.1-1.4 ms/step\n"
         "available — the wall-minus-gpu column.\n\n"
         "Derivation of the ~1.5 ms/step launch/gap figure: 1.9% idle x 70.76 ms/step = 1.34 ms/step\n"
         "at INT8. It had previously been inferred by subtracting kernel time from wall time; the\n"
         "nsys traces measure it directly.\n\n"
         "Right panel: int4 issues 1087 launches/step against int8's 518, because its kernels are\n"
         "smaller and more numerous. That is why its idle share is largest (2.5%) despite it being\n"
         "the fastest mode — per-launch cost is fixed while the work per launch shrank.", fs=8.2)
    head(fig, 0.245, "A conflicting source, and why it is not used", color=WARN, fs=10)
    body(fig, 0.222,
         "final_report_2026-07-28/data/gpu_busy_fraction.json reports 48.6% busy for FP16 and 68.6%\n"
         "for INT8. Its own numbers locate the defect: its 'unprofiled_ms_step' is 210.25 for FP16\n"
         "against the actual 101.37, i.e. 2.07x inflated. It was written from a PROFILED run, so its\n"
         "wall clock carries profiler overhead while its GPU total does not.\n\n"
         "Cross-check: e2e_plots_b128_run1.json holds both a profiler-free wall time and the profiler\n"
         "GPU total per mode; their ratio gives 99.8 / 99.3 / 99.2% busy, agreeing with the nsys\n"
         "table to about a point. Two independent sources agree; the third has a locatable defect.",
         fs=8.2)
    save(pdf, fig)


def p_attn_run(pdf):
    fig = page("Section 8d", "Attention layers during a real run",
               "Scope: layer, in situ. NVTX ranges over three traces per mode. GPU us/call, the CPU\n"
               "time to issue that call, and their ratio. Source: 08-01 report.")
    rows = [[sh, f"{f:.1f}", f"{fc:.1f}", f"{fr:.2f}", f"{i8:.1f}", f"{i8c:.1f}", f"{i8r:.2f}",
             f"{i4:.1f}", f"{i4r:.2f}"]
            for (sh, f, fc, fr, i8, i8c, i8r, i4, i4c, i4r) in ATTN_RUN]
    tbl(fig, [0.04, 0.740, 0.93, 0.105],
        ["shape", "fp16 gpu", "fp16 issue", "g/i", "i8 gpu", "i8 issue", "g/i", "i4 gpu", "g/i"],
        rows, colw=[0.13, 0.10, 0.105, 0.07, 0.10, 0.10, 0.07, 0.10, 0.07], fs=7.1)

    sh = [r[0] for r in ATTN_RUN]
    x = range(len(sh))
    ax = fig.add_axes([0.12, 0.525, 0.36, 0.20])
    w = 0.26
    for i, (idx, c, lb) in enumerate([(1, FP_C, "fp16"), (4, BASE_C, "int8"), (7, I4_C, "int4")]):
        ax.bar([j + (i - 1) * w for j in x], [r[idx] for r in ATTN_RUN], width=w, color=c,
               label=lb, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels(sh, rotation=38, ha="right", fontsize=6.6)
    ax.set_ylabel("GPU us / call (log)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.3, zorder=0, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax2 = fig.add_axes([0.60, 0.525, 0.34, 0.20])
    for idx, c, lb, mk in [(3, FP_C, "fp16", "o"), (6, BASE_C, "int8", "s"), (9, I4_C, "int4", "^")]:
        ax2.plot(list(x), [r[idx] for r in ATTN_RUN], mk + "-", color=c, label=lb, lw=1.6, ms=4)
    ax2.axhline(1.0, ls="--", lw=1.2, color=MOD_C)
    ax2.text(len(sh) - 0.1, 1.15, "GPU = issue", fontsize=6.8, color=MOD_C, ha="right")
    ax2.fill_between([-0.4, len(sh) - 0.6], 0, 1.0, color=MOD_C, alpha=0.07)
    ax2.text(0.1, 0.42, "dispatch-bound", fontsize=7, color=MOD_C)
    ax2.set_yscale("log")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(sh, rotation=38, ha="right", fontsize=6.6)
    ax2.set_ylabel("GPU / CPU-issue (log)", fontsize=8)
    ax2.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax2.grid(alpha=0.3, which="both")
    ax2.set_xlim(-0.4, len(sh) - 0.6)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    body(fig, 0.475,
         "Above the dashed line the GPU runs longer than the CPU needs to issue the work, so the\n"
         "layer is GPU-bound. Below it the GPU finishes before the CPU can hand it the next call.\n\n"
         "   C192/T1024   GPU/issue 11-16x.  GPU-bound.\n"
         "   C384/T256    4-5x.              GPU-bound.\n"
         "   C384/T64     1.3-1.6x.          Marginal.\n"
         "   C768/T16     0.78-1.18x.        At the boundary.\n"
         "   C768/T4      0.35-0.47x.        Dispatch-bound: ~100 us of GPU behind ~283 us of issue.\n\n"
         "This is the C768/T4 disagreement of section 6: what that shape measures depends on whether\n"
         "the CPU or the GPU is the constraint at the moment of measurement — in isolation versus\n"
         "surrounded by other work in a real run.\n\n"
         "The CPU issue cost also drops with quantization: 273 -> 179 us at C192/T1024. The fused\n"
         "path issues fewer, larger kernels.", fs=8.2)
    save(pdf, fig)


# =====================================================================================
# Section 9 — attention deep profile
# =====================================================================================
ATTN_MODES = ["FP16", "INT8", "INT8+MoDiff", "INT4", "INT4+MoDiff"]
ATTN_STAGES = ["GroupNorm (+quantize)", "qkv projection", "attention core", "out projection",
               "elementwise / copy"]


def p_attn_stages(pdf, reb):
    fig = page("Section 9a", "Attention per-kernel profile, five modes",
               "Scope: kernel, inside one AttentionBlock forward, batch 128, us per forward.\n"
               "Five modes time-aligned per shape in one Perfetto trace. Buckets are keyed on the\n"
               "kernel names actually observed. Source: docs/attn_modiff_profile_2026-08-04.")
    if not reb:
        body(fig, 0.85, "MISSING attn_stages_rebucketed.json", color=MOD_C); save(pdf, fig); return
    rows = []
    for st in ATTN_STAGES:
        v = [reb.get(f"{m}|C192/T1024", {}).get(st, 0.0) for m in ATTN_MODES]
        if sum(v) < 0.05:
            continue
        rows.append([st] + [f"{x:.1f}" for x in v])
    tot = [sum(reb.get(f"{m}|C192/T1024", {}).values()) for m in ATTN_MODES]
    rows.append(["TOTAL"] + [f"{x:.1f}" for x in tot])
    tbl(fig, [0.04, 0.735, 0.93, 0.13],
        ["C192/T1024 — stage"] + ATTN_MODES, rows,
        colw=[0.24] + [0.135] * 5, fs=7.0, hi=[len(rows) - 1])

    shapes = ["C192/T1024", "C384/T256", "C384/T64", "C768/T16", "C768/T4"]
    ninst = {"C192/T1024": 5, "C384/T256": 5, "C384/T64": 5, "C768/T16": 5, "C768/T4": 1}
    rows2 = []
    for s in shapes:
        t = [sum(reb.get(f"{m}|{s}", {}).values()) for m in ATTN_MODES]
        rows2.append([s, f"x{ninst[s]}"] + [f"{x:.1f}" for x in t] +
                     [f"{t[0] / t[1]:.2f}x" if t[1] else "—"])
    tbl(fig, [0.04, 0.545, 0.93, 0.135],
        ["shape", "n", "FP16", "INT8", "INT8+MoD", "INT4", "INT4+MoD", "fp16/int8"],
        rows2, colw=[0.145, 0.055] + [0.115] * 5 + [0.115], fs=7.0)

    img(fig, f"{A}/plots/attn_stage_breakdown.png", 0.50, width=0.92, left=0.04, max_h=0.20)
    body(fig, 0.275,
         "How to read the two tables. The first is the dominant shape broken down by stage; the\n"
         "second is every shape's TOTAL, with n = how many blocks of that shape the UNet contains\n"
         "(multiply a per-forward figure by n for a per-step figure).\n\n"
         "Only the TOTAL column is comparable between fp16 and int8. The per-stage rows are not:\n"
         "int8 has no elementwise kernel at all — bias, residual and the GroupNorm apply are inside\n"
         "fused kernels — while fp16 pays them in one aggregated elementwise entry that the profiler\n"
         "cannot split. Section 9b quantifies this.\n\n"
         "fp16's qkv row is 0 at three shapes because both fp16 projections can be 1x1 convs sharing\n"
         "the same cutlass ImplicitGemmConvolution name; at those shapes they cannot be separated by\n"
         "name and both land in 'out projection'. C192/T1024 does separate them.", fs=8.2)
    save(pdf, fig)


def p_attn_fusion(pdf):
    fig = page("Section 9b", "Fusion accounting: the fp16 column does not balance",
               "Scope: kernel, C192/T1024, us per forward. A byte count against the A40's 696 GB/s\n"
               "peak, applied to the fp16 column.")
    tbl(fig, [0.06, 0.735, 0.88, 0.115],
        ["work the fp16 elementwise entry must contain", "bytes moved", "at 696 GB/s"],
        [["GroupNorm apply: read x + write normalized", "100.7 MB", "145 us"],
         ["residual add: read proj_out + read x + write", "151.0 MB", "217 us"],
         ["total", "251.7 MB", "362 us"]],
        colw=[0.50, 0.17, 0.17], fs=7.6, hi=[2])
    body(fig, 0.685,
         "The measured entry is 268.2 us. 251.7 MB in 268.2 us is 939 GB/s, against a 696 GB/s peak —\n"
         "impossible. So at least one fp16 stage is fused somewhere the attribution does not see, or\n"
         "is absent from it. Consistent with that, fp16's gn_accum reads 50.3 MB in 108.4 us =\n"
         "464 GB/s, a read-only reduction: fp16's GroupNorm rows are statistics only, and there is no\n"
         "GN apply kernel anywhere in the fp16 column.", fs=8.4)
    tbl(fig, [0.10, 0.475, 0.80, 0.105],
        ["", "fp16", "int8", ""],
        [["out-proj GEMM alone", "191.6", "345.7", "0.55x — mis-paired"],
         ["residual add", "268.2", "0 (fused)", ""],
         ["out proj + the residual it adds", "459.8", "345.7", "1.33x — int8 faster"]],
        colw=[0.30, 0.13, 0.13, 0.24], fs=7.6, hi=[2], warn=[0])
    img(fig, f"{A}/plots/attn_fusion_accounting.png", 0.462, width=0.94, left=0.03, max_h=0.185)
    body(fig, 0.262,
         "Bounding the split. The residual add is the largest single thing the 268.2 us can be: 151 MB\n"
         "at 563 GB/s is 81% of peak and fits, whereas adding the GN apply does not. So 459.8 us is\n"
         "an upper bound on fp16's out-projection-plus-residual. At the other extreme — all 268.2 us\n"
         "is GN apply and the residual add is absent from the trace — the fp16 column is short one\n"
         "whole stage. Under either reading the per-stage ratio is not a measurement.\n\n"
         "Scaled to the whole model: 268.2 x5 + 137.9 x5 + 32.1 x5 + 13.5 x5 + 4.9 x1 = 2263 us/step\n"
         "= 453 ms/batch of fp16 attention elementwise work that int8 folds into its projection and\n"
         "GroupNorm kernels. That is the quantity referenced from section 8a's projection row, which\n"
         "rises by 77 ms/batch.", fs=8.2)
    save(pdf, fig)


def p_attn_roofline(pdf, roof):
    fig = page("Section 9c", "Projection GEMMs against their own roofline",
               "Scope: kernel, int8 only — no cross-precision attribution. A40: 696 GB/s, 299.4\n"
               "TOPS int8. roofline = max(bytes/BW, ops/TOPS) for that same kernel.")
    if not roof:
        body(fig, 0.85, "MISSING attn_roofline.json", color=MOD_C); save(pdf, fig); return
    order = [(s, k) for s in ["C192/T1024", "C384/T256", "C384/T64", "C768/T16", "C768/T4"]
             for k in ("qkv", "proj")]
    rows = []
    for s, k in order:
        v = roof.get(f"{s}|{k}")
        if not v:
            continue
        rows.append([s, k, f"{v['M']}", f"{v['K']}", f"{v['N']}", f"{v['bytes'] / 2**20:.1f}",
                     f"{v['mem_us']:.1f}", f"{v['cmp_us']:.1f}", f"{v['roof_us']:.1f}",
                     f"{v['measured_us']:.1f}", f"{v['pct_roofline']:.1f}%",
                     "mem" if v["bound"] == "mem" else "cmp"])
    tbl(fig, [0.03, 0.652, 0.95, 0.205],
        ["shape", "kernel", "M", "K", "N", "MB", "mem us", "cmp us", "roof", "meas",
         "% roof", "bound"],
        rows, colw=[0.125, 0.065, 0.075, 0.05, 0.055, 0.06, 0.07, 0.07, 0.06, 0.065, 0.07, 0.06],
        fs=6.6)
    tbl(fig, [0.16, 0.545, 0.68, 0.070],
        ["fp16 reference, C192/T1024", "roofline", "measured", "% of its own roofline"],
        [["qkv (1x1 conv, fp16 -> 3x fp16)", "289.3", "617.5", "46.8%"],
         ["out proj (cuBLAS, no residual)", "144.6", "191.6", "75.5%"]],
        colw=[0.28, 0.13, 0.13, 0.20], fs=7.3)
    img(fig, f"{A}/plots/attn_gemm_roofline.png", 0.533, width=0.94, left=0.03, max_h=0.168)
    body(fig, 0.352,
         "Byte counts. qkv (gemm_w8a8_kernel_awq_out_i8) reads A int8 (M*C) and writes int8 Q/K/V^T\n"
         "(M*3C); weights are negligible (K*3C = 442 KB at C=768). out proj (gemm_w8a8_kernel_awq)\n"
         "reads A int8 (M*C), reads the residual fp16 (2M*C) and writes fp16 (2M*C) — the total\n"
         "absence of int8 elementwise kernels in section 9a is the evidence that the residual really\n"
         "is in that epilogue.\n\n"
         "The AWQ GEMM reaches 10-52% of its own roofline; cuBLAS fp16 reaches 75% of its on the same\n"
         "shape. GWQ_CTA_K = 64, so K=192 is three K-tiles against a three-stage software pipeline:\n"
         "there is no steady state over which to amortize the shared-memory staging, the swizzle\n"
         "setup and the epilogue. The kernel has one fixed tile config (GWQ_CTA_M/N/K = 128/128/64,\n"
         "csrc/kernels/linear/gemm_wxax.cu:104-106) ported from large-K LLM shapes.\n\n"
         "C768/T4 is a different cause: M=512 at CTA_M=128 is 4 x 6 = 24 CTAs on 84 SMs, so the\n"
         "kernel cannot fill the GPU regardless of tile shape.\n\n"
         "Distance to fp16's achieved fraction, at the two shapes that hold the time: qkv 604.9 ->\n"
         "309.0 and 302.7 -> 206.9 us; out proj 345.7 -> 239.5 and 185.2 -> 119.7 us. x5 instances\n"
         "each: 1.96 ms/step for qkv, 0.86 for out proj, 2.82 total = 4.0% of the 71 ms/step int8\n"
         "baseline. The three small shapes add 0.55 ms/step at the same target but are\n"
         "occupancy-bound.", fs=7.6)
    save(pdf, fig)


def p_attn_modiff(pdf, reb):
    fig = page("Section 9d", "MoDiff inside the attention layer",
               "Scope: kernel, int8-vs-int8 and int4-vs-int4, so none of section 9b's fusion\n"
               "asymmetry applies. us per forward, batch 128.")
    shapes = ["C192/T1024", "C384/T256", "C384/T64", "C768/T16", "C768/T4"]
    rows = []
    for s in shapes:
        t = {m: sum(reb.get(f"{m}|{s}", {}).values()) for m in ATTN_MODES}
        rows.append([s, f"{t['INT8']:.1f}", f"{t['INT8+MoDiff']:.1f}",
                     f"{t['INT8+MoDiff'] / t['INT8']:.3f}x" if t["INT8"] else "—",
                     f"{t['INT4']:.1f}", f"{t['INT4+MoDiff']:.1f}",
                     f"{t['INT4+MoDiff'] / t['INT4']:.3f}x" if t["INT4"] else "—"])
    tbl(fig, [0.08, 0.735, 0.85, 0.135],
        ["shape", "INT8", "INT8+MoDiff", "ratio", "INT4", "INT4+MoDiff", "ratio"],
        rows, colw=[0.16, 0.11, 0.14, 0.11, 0.11, 0.14, 0.11], fs=7.4)
    img(fig, f"{A}/plots/attn_modiff_delta.png", 0.695, width=0.86, left=0.07, max_h=0.24)
    body(fig, 0.415,
         "0.99-1.03x at every shape and both bit-widths. The residual 1-2% is the shared GroupNorm\n"
         "kernel, which under MoDiff also serves the conv path's delta work.\n\n"
         "The exclusion is structural, not an omission. The qkv epilogue exists to emit quantized\n"
         "Q/K/V codes in the layouts the flash kernel consumes. Under MoDiff the GEMM produces an\n"
         "accumulator increment rather than an activation, so there are no codes to emit; and\n"
         "int8-quantizing an increment with a scale calibrated on the accumulated output destroys it,\n"
         "with no error compensation available (MoDiff compensates the input side only).\n\n"
         "Consequence for the rest of section 9: any attention work is orthogonal to MoDiff. It helps\n"
         "the baseline and MoDiff by the same amount.", fs=8.4)
    save(pdf, fig)


# =====================================================================================
# MoDiff
# =====================================================================================
def p_mod_scheme(pdf):
    fig = page("Section 10", "MoDiff fusion scheme — what changes",
               "MoDiff adds three elementwise operations: subtract a_hat, quantize the delta,\n"
               "advance a_hat. Kernel COUNT is unchanged at every site but one.")
    body(fig, 0.875,
         "   a_hat_T = Q(a_T)                        o_hat_T = A(a_hat_T)\n"
         "   a_hat_t = Q(a_t - a_hat_{t+1}) + a_hat_{t+1}\n"
         "   o_hat_t = A(Q(a_t - a_hat_{t+1})) + o_hat_{t+1}          (Eqs 8-10)\n\n"
         "A(.) is any linear operator. Bias belongs to o_hat_T only: a modulated step's increment\n"
         "carries no bias.", fs=8.4)
    tbl(fig, [0.05, 0.605, 0.92, 0.22],
        ["site", "what MoDiff adds", "kernels", "symbol"],
        [["ResBlock entry", "subtract + advance a_hat", "1 (same)", "..._delta_quantize[_pack]_nhwc"],
         ["GN statistics", "materialized, two consumers share", "+1", "gn_stats_partials_chanmajor  NEW"],
         ["updown ResBlock", "delta ops, a_hat at post-resize res", "1 (same)", "..._delta_quantize_resize  NEW"],
         ["Conv", "in-place o_hat read-modify-write", "1 (same)", "conv2d_int{8,4}_evt_o_hat"],
         ["Delta scale", "dynamic Q/max|delta| per call", "+1 per 4 steps", "delta_absmax_fp16  NEW"],
         ["Up/Downsample", "optional a_hat arg (null = baseline)", "1 (same)", "..._quantize_noahat"],
         ["Attention", "nothing — structurally excluded", "1 (same)", "flash_attn_int{8,4}_vt..."]],
        colw=[0.15, 0.34, 0.14, 0.29], fs=7.2)
    body(fig, 0.565,
         "Fusion parity. Every baseline fusion has a MoDiff counterpart, so MoDiff never falls back to\n"
         "an unfused path. Confirmed by the profile in section 13: MoDiff's GN+quantize chain is net\n"
         "cheaper than the baseline's fused kernel, which could not happen if a fusion were missing.\n\n"
         "No cloned kernels are needed. The delta ops are expressible as a nullable a_hat pointer on\n"
         "the existing kernel — with a_hat = nullptr the kernel is bit-identical to the baseline\n"
         "(verified on upsample2x / avgpool2x, 4/4 shapes, int8 and int4). `if (ptr != nullptr)` is a\n"
         "predicated register op.\n\n"
         "One genuine exception: the o_hat accumulate is a separate CUTLASS instantiation, because\n"
         "epilogues are compile-time template parameters and there is no runtime-nullable form of\n"
         "'also read-modify-write this tensor'.\n\n"
         "Shipped configuration: MODIFF_DELTA_MODE=dynamic, MODIFF_DELTA_REFRESH=4,\n"
         "MODIFF_DELTA_CLIP=1.0, MODIFF_DELTA_REPORT=0. Sections 14-16 are the ablations behind each.",
         fs=8.4)
    save(pdf, fig)


def p_profile(pdf, prof, which):
    base = which == "baseline"
    fig = page("Section 12" if base else "Section 13",
               f"{'Baseline' if base else 'MoDiff'} GPU-time profile",
               "Scope: kernel. torch.profiler self-time bucketed by role, batch 128, ms per DDIM step.")
    if not prof:
        body(fig, 0.85, "NOT MEASURED", color=MOD_C); save(pdf, fig); return
    key = "int8_baseline" if base else next((k for k in prof if "dynamic" in k), None)
    if not key:
        body(fig, 0.85, "case absent", color=MOD_C); save(pdf, fig); return
    roles = sorted(prof[key]["roles"].items(), key=lambda kv: -kv[1])
    tot = prof[key]["total_ms_per_step"]
    other = prof["int8_baseline"]["roles"] if (not base and "int8_baseline" in prof) else None
    fig.text(0.08, 0.878, f"{key} — total GPU kernel time {tot:.2f} ms/step",
             fontsize=10, color=INK, weight="bold", va="top")

    top = roles[:11]
    lab = [(r[0][:40] + "..." if len(r[0]) > 43 else r[0]) for r in top][::-1]
    val = [r[1] for r in top][::-1]
    ax = fig.add_axes([0.34, 0.585, 0.62, 0.275])
    ax.barh(range(len(val)), val, color=BASE_C if base else MOD_C, zorder=3)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=6.6)
    ax.set_xlabel("ms / step", fontsize=8)
    bare(ax, "x")
    for i, v in enumerate(val):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=6.6, color=INK)

    if other:
        d = sorted(((r, v - other.get(r, 0.0)) for r, v in prof[key]["roles"].items()),
                   key=lambda kv: kv[1])
        d = [x for x in d if abs(x[1]) > 0.05]
        ax2 = fig.add_axes([0.34, 0.375, 0.62, 0.155])
        nm = [(k[:34] + "..." if len(k) > 37 else k) for k, _ in d]
        vv = [v for _, v in d]
        ax2.barh(range(len(vv)), vv, color=[OK if v < 0 else MOD_C for v in vv], zorder=3)
        ax2.set_yticks(range(len(nm))); ax2.set_yticklabels(nm, fontsize=6.2)
        ax2.axvline(0, color=INK, lw=0.9)
        ax2.set_xlabel("ms/step vs int8_baseline  (left = cheaper)", fontsize=7.4)
        bare(ax2, "x")
        for i, v in enumerate(vv):
            ax2.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v > 0 else "right",
                     fontsize=6.2, color=INK)

    tp = sorted(prof[key]["kernels"].items(), key=lambda kv: -kv[1])[:8]
    fig.text(0.08, 0.330, "top kernels (ms/step)", fontsize=8.6, color=INK, weight="bold", va="top")
    fig.text(0.08, 0.307, "\n".join(f"{v:6.2f}  {k[:44]}" for k, v in tp),
             fontsize=6.8, color=MUTE, va="top", family="monospace", linespacing=1.55)
    if base:
        body(fig, 0.180,
             "The conv dominates at 24.2 ms/step, then the fused GN+quantize at 16.2, then attention\n"
             "at 8.6 (core) + 8.6 (projections). There is no standalone quantize line item — every\n"
             "quantize is inside one of these, by the design rule in section 3.", fs=8.2)
    else:
        body(fig, 0.198,
             "The lower chart is the signed difference against int8_baseline, i.e. what MoDiff trades:\n\n"
             "   -14.1  the baseline's fused GN+quantize kernel disappears\n"
             "   +10.8  the delta-quantize kernel replaces it\n"
             "   +4.4   GN statistics now materialized (two consumers share them)\n"
             "   +1.6   the dynamic scale's reduction pass (1 step in 4)\n"
             "   +1.3   the conv's in-place o_hat read-modify-write\n\n"
             "Net on the GN chain is about +1.0 ms/step; the rest is a_hat traffic and the o_hat\n"
             "accumulate, both required by Eqs 9-10.", fs=7.8)
    save(pdf, fig)


# =====================================================================================
# Ablations
# =====================================================================================
def p_delta_mode(pdf, ab, st, i4t):
    fig = page("Section 14", "Delta quantizer: scale grid and mode",
               "Scope: end-to-end, DDIM 50, batch 8, seed 1234. Latent relative L2 against the fp16\n"
               "model. ms/step at batch 8 is NOT comparable with the batch-128 figures in section 11.")
    if ab:
        rows = [[k, f"{v['rel_l2_vs_fp16']:.4f}", f"{v['ms_per_step']:.2f}",
                 v.get("delta_mode", "—"), str(v.get("table_layers", "—")),
                 f"{v.get('latent_absmax', float('nan')):.2f}" if "latent_absmax" in v else "—"]
                for k, v in ab["results"].items()]
        tbl(fig, [0.05, 0.613, 0.90, 0.242],
            ["configuration", "latent relL2", "ms/step (b8)", "delta mode", "table layers",
             "latent absmax"],
            rows, colw=[0.28, 0.13, 0.14, 0.12, 0.12, 0.13], fs=7.0,
            warn=[i for i, (k, v) in enumerate(ab["results"].items())
                  if v["rel_l2_vs_fp16"] > 1.0])
    body(fig, 0.645,
         "Three grids for the same delta, at W8A8:\n\n"
         "   activation grid    0.1864   scale calibrated on the full activation. Per Theorem 4.3 the\n"
         "                               error is proportional to s^2, so an unchanged s means an\n"
         "                               unchanged step size: MoDiff buys only error feedback.\n"
         "   static per-step    0.0415   one calibrated scale per layer per step index (70 layers).\n"
         "   dynamic            0.0395   scale = Q/max|delta| recomputed on device.\n\n"
         "The highlighted row is the first dynamic implementation, which diverged (relL2 10.32,\n"
         "latent absmax 47.8). Cause: on a reporting step the kernel quantized with the current scale\n"
         "but published the next one into the buffer the following conv reads as its dequant alpha,\n"
         "so o_hat accumulated on a scale that was never used to quantize. Fixed by double-buffering\n"
         "the published pair; the reporting path is off by default regardless (section 2, row 7).\n\n"
         "At W4A4 the activation grid gives 0.7742 against a 0.7837 baseline — no measurable gain —\n"
         "while dynamic gives 0.4746. The scale grid is the whole difference between MoDiff working\n"
         "and not working at 4-bit activations.", fs=8.4)
    if i4t:
        rows = [[k, f"{v['rel_l2_vs_fp16']:.4f}", f"{v['ms_per_step']:.2f}",
                 str(v.get("table_layers", "—"))] for k, v in i4t["results"].items()]
        tbl(fig, [0.14, 0.200, 0.72, 0.145],
            ["W4A4 static delta table", "relL2", "ms/step (b8)", "table layers"],
            rows, colw=[0.34, 0.13, 0.14, 0.13], fs=7.2)
        body(fig, 0.180,
             "The static table helps at W8A8 (0.1864 -> 0.0415) but not at W4A4 (0.7763 -> 0.7555):\n"
             f"its median per-step gain there is {i4t.get('median_step_gain', 0):.2f}, i.e. the "
             "calibrated delta range is LARGER than\n"
             "the activation range it replaces. Only the dynamic scale reaches 0.4746 at 4 bits.\n\n"
             "Theorem 4.3's bound assumes dynamic quantizers, which is consistent with this.", fs=8.4)
    save(pdf, fig)


def p_sweeps(pdf, clip, refr):
    fig = page("Section 15", "Delta clip and refresh-interval sweeps",
               "Scope: end-to-end, DDIM 50, batch 8. clip = the fraction of max|delta| the dynamic\n"
               "scale is set from; refresh K = the reduction runs every K-th step.")
    if clip:
        rows = []
        for r in clip["ratios"]:
            k = str(r)
            a, b = clip["results"]["int8"][k], clip["results"]["int4"][k]
            rows.append([f"{r:.2f}", f"{a['rel_l2_vs_fp16']:.4f}", f"{a['ms_per_step']:.2f}",
                         f"{b['rel_l2_vs_fp16']:.4f}", f"{b['ms_per_step']:.2f}"])
        tbl(fig, [0.14, 0.662, 0.72, 0.194],
            ["clip", "W8A8 relL2", "ms/step", "W4A4 relL2", "ms/step"],
            rows, colw=[0.13, 0.17, 0.14, 0.17, 0.14], fs=7.2, hi=[0])
        ax = fig.add_axes([0.13, 0.470, 0.34, 0.145])
        rr = clip["ratios"]
        ax.plot(rr, [clip["results"]["int8"][str(r)]["rel_l2_vs_fp16"] for r in rr], "o-",
                color=BASE_C, lw=1.8, label="W8A8")
        ax.axhline(clip["static_reference"]["int8"]["baseline"], ls="--", lw=1, color=MUTE)
        ax.text(0.11, clip["static_reference"]["int8"]["baseline"] * 1.03, "int8 baseline",
                fontsize=6.6, color=MUTE)
        ax.set_xlabel("clip ratio", fontsize=8); ax.set_ylabel("latent relL2", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        bare(ax)
        ax2 = fig.add_axes([0.61, 0.470, 0.33, 0.145])
        ax2.plot(rr, [clip["results"]["int4"][str(r)]["rel_l2_vs_fp16"] for r in rr], "s-",
                 color=MOD_C, lw=1.8, label="W4A4")
        ax2.set_xlabel("clip ratio", fontsize=8); ax2.set_ylabel("latent relL2", fontsize=8)
        ax2.legend(fontsize=7, frameon=False)
        bare(ax2)
    if refr:
        rows = []
        for kk, v in refr["results"].items():
            rows.append([kk, str(v["refresh"]), f"{v['rel_l2_vs_fp16']:.4f}",
                         f"{v['ms_per_step']:.2f}"])
        tbl(fig, [0.20, 0.252, 0.60, 0.169],
            ["configuration", "K", "relL2", "ms/step (b8)"],
            rows, colw=[0.26, 0.09, 0.14, 0.16], fs=7.2)
    body(fig, 0.228,
         "Clip. At W8A8 the error rises monotonically as the scale is clipped below max|delta|\n"
         "(0.0393 at 1.0 to 0.1973 at 0.1), so no clipping is applied. W4A4 is non-monotone across\n"
         "the sweep with a spread of about 0.11 — within the run-to-run variation seen elsewhere at\n"
         "batch 8 — so 1.0 is used at both bit-widths.\n\n"
         "Refresh. K=8 measures slightly better than K=4 at W8A8 (0.0413 vs 0.0462) and slightly\n"
         "worse at W4A4 (0.4851 vs 0.4633); the differences are of the same order as the spread. The\n"
         "wall-clock difference at batch 128 is 6.4 ms/step between K=1 and K=4 and under 1 ms/step\n"
         "between K=4 and K=8 (section 11), so K=4 is the default.", fs=8.4)
    save(pdf, fig)


def p_gn_ab(pdf, gn):
    fig = page("Section 16", "GroupNorm statistics kernel variants",
               "Scope: end-to-end, DDIM 50, batch 8. Under MoDiff the GN statistics are materialized\n"
               "and shared by two consumers, so this kernel is on the critical path. 'deterministic'\n"
               "= identical output over a replayed launch; max_abs_diff_replay quantifies it.")
    if not gn:
        body(fig, 0.85, "MISSING gn_stats_ab.json", color=MOD_C); save(pdf, fig); return
    rows = [[k.split("|")[0], k.split("|")[1], f"{v['rel_l2_vs_fp16']:.4f}",
             f"{v['ms_per_step']:.2f}", "yes" if v["deterministic"] else "NO",
             f"{v['max_abs_diff_replay']:.4f}"]
            for k, v in gn["results"].items()]
    hi = [i for i, k in enumerate(gn["results"]) if "default" in k]
    warn = [i for i, (k, v) in enumerate(gn["results"].items()) if not v["deterministic"]]
    tbl(fig, [0.06, 0.695, 0.88, 0.155],
        ["mode", "variant", "relL2", "ms/step (b8)", "deterministic", "replay diff"],
        rows, colw=[0.09, 0.32, 0.12, 0.14, 0.14, 0.13], fs=7.0, hi=hi, warn=warn)
    ax = fig.add_axes([0.16, 0.475, 0.70, 0.185])
    ks = [k for k in gn["results"] if k.startswith("int8")]
    lb = [k.split("|")[1].replace(" (", "\n(") for k in ks]
    vv = [gn["results"][k]["ms_per_step"] for k in ks]
    cc = [OK if "default" in k else (MOD_C if not gn["results"][k]["deterministic"] else BASE_C)
          for k in ks]
    b = ax.bar(range(len(vv)), vv, color=cc, width=0.6, zorder=3)
    for r, v in zip(b, vv):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, weight="bold")
    ax.set_xticks(range(len(lb))); ax.set_xticklabels(lb, fontsize=6.6)
    ax.set_ylabel("ms/step at batch 8, W8A8", fontsize=8)
    bare(ax)
    body(fig, 0.425,
         "The default is a two-stage chan-major reduction: blockDim.x == C, so thread t owns channel\n"
         "t and its group is invariant. Reads are fully coalesced, there are no atomics, and the\n"
         "reduction order is fixed, so it is deterministic. It replaced the group-major tree (ALT=0),\n"
         "whose group-major reads waste 4-8x of every 128 B sector at 4 or 8 channels per group.\n\n"
         "At batch 128 the same change is 9.61 -> 3.35 ms/step.\n\n"
         "Both non-deterministic variants use atomics and are also slower here: ALT=1 at 45.3 and\n"
         "ALT=2 at 31.3 ms/step against the default's 16.8. An in-source comment previously described\n"
         "ALT=2 as a candidate replacement worth about 9.4 ms/step; measured, it is 12.7 ms/step\n"
         "slower and non-deterministic. The comment was wrong and has been corrected.\n\n"
         "Acceptance criterion. Once the delta scale changed, bit-exactness against the old kernel is\n"
         "gone by design, so the criterion is agreement with an fp64 reference plus determinism over\n"
         "replayed launches — not max_code_diff == 0 against ALT=0.", fs=8.4)
    save(pdf, fig)


def p_calibration(pdf, util, gain):
    fig = page("Section 17", "Quantizer calibration health",
               "Scope: layer statistics over the 35 in_conv / 35 out_conv sites, real checkpoint.\n"
               "'step gain' = activation range / delta range, i.e. how many times finer the delta\n"
               "grid is than the activation grid it replaces.")
    if util:
        rows = []
        for cfg, v in util.items():
            for site in ("in_conv", "out_conv"):
                s = v[site]
                sq = v["_smoothquant"][site]
                rows.append([cfg.replace("int8 | REAL ckpt: ", ""), site, str(s["n"]),
                             f"{s['median']:.0f}", f"{s['max']:.0f}", f"{s['clipping']}",
                             f"{sq['smooth_max_median']:.2f}"])
        tbl(fig, [0.03, 0.725, 0.95, 0.120],
            ["calibration configuration", "site", "n", "median gain", "max gain",
             "layers clipping", "SmoothQuant max"],
            rows, colw=[0.32, 0.10, 0.05, 0.12, 0.11, 0.13, 0.14], fs=6.6)
    if gain:
        rows = [[s, f"{v['median_gain']:.2f}", f"{v['max_gain']:.2f}", f"{v['min_gain']:.2f}",
                 str(v["clipped_layers"])] for s, v in gain.items()]
        tbl(fig, [0.18, 0.565, 0.64, 0.120],
            ["DDIM steps", "median step gain", "max", "min", "clipped layers"],
            rows, colw=[0.14, 0.18, 0.11, 0.11, 0.16], fs=7.2)
        ax = fig.add_axes([0.20, 0.415, 0.60, 0.145])
        xs = [int(s) for s in gain]
        ax.plot(xs, [gain[str(s)]["median_gain"] for s in xs], "o-", color=BASE_C, lw=2)
        ax.axhline(1.0, ls="--", lw=1, color=MUTE)
        ax.set_xlabel("DDIM steps", fontsize=8)
        ax.set_ylabel("median step gain", fontsize=8)
        bare(ax)
    body(fig, 0.393,
         "The step gain grows with the step count: 1.04 at 20 steps, 2.39 at 200. Consecutive\n"
         "activations are closer together the finer the timestep grid is, which is the mechanism\n"
         "MoDiff exploits. At 20 steps the median gain is 1.04, i.e. the delta is barely narrower\n"
         "than the activation and the method has little to work with.\n\n"
         "Calibration configuration. Matching the calibration horizon and batch to the run and adding\n"
         "one refinement round moves the in_conv median gain from 1151 to 713 and the number of\n"
         "layers whose observed range clips from 35/35 to 28/35 — the first configuration was\n"
         "observing ranges from a 5-step, batch-2 pass that does not resemble the real trajectory,\n"
         "so its ranges were far too wide.\n\n"
         "SmoothQuant is active at every site (identity count 0) with a median channel scale of\n"
         "5.9-9.2, and every weight tensor uses the full int8 grid (wint8_absmax_median = 127).",
         fs=8.4)
    save(pdf, fig)


def p_int4_attr(pdf, attr, lin):
    fig = page("Section 18", "W4A4 error attribution, and MoDiff on the Linear layers",
               "Scope: end-to-end, DDIM 50, batch 8. Each row disables quantization for one part of\n"
               "the model, so the remaining error is attributable to the parts still quantized.")
    if attr:
        bar = attr["bar_int8_baseline"]
        rows = [[k, f"{v['rel_l2_vs_fp16']:.4f}", f"{v['ms_per_step']:.2f}",
                 ", ".join(f"{a.replace('MODIFF_QUANT_', '').replace('MODIFF_', '')}={b}"
                           for a, b in v.get("env", {}).items()) or "—",
                 f"{v['rel_l2_vs_fp16'] / bar:.2f}x"]
                for k, v in attr["results"].items()]
        tbl(fig, [0.04, 0.700, 0.93, 0.145],
            ["configuration", "relL2", "ms/step (b8)", "env", "vs int8 baseline"],
            rows, colw=[0.28, 0.11, 0.13, 0.24, 0.15], fs=7.0)
    body(fig, 0.615,
         "Reading the attribution. Putting attention in fp16 changes nothing (0.4746 -> 0.4766).\n"
         "Putting the Linear layers in fp16 recovers 0.0389 (0.4746 -> 0.4358). Putting both in fp16\n"
         "leaves 0.4358, identical to Linear alone. So about 92% of W4A4's error is the 4-bit conv\n"
         "path — which MoDiff already covers — and with attention AND the Linear layers in fp16 W4A4\n"
         "is still 1.83x the int8 baseline's 0.2378.\n\n"
         "This is why W4A4 + MoDiff does not reach the W8A8 baseline's accuracy: the residual error\n"
         "is in the part MoDiff is already applied to, at 4-bit weights as well as 4-bit activations.",
         fs=8.4)
    if lin:
        rows = [[k, f"{v['rel_l2_vs_fp16']:.4f}", f"{v['ms_per_step']:.2f}",
                 f"{v['latent_absmax']:.2f}"] for k, v in lin["results"].items()]
        tbl(fig, [0.16, 0.360, 0.68, 0.125],
            ["MODIFF_LINEAR A/B", "relL2", "ms/step (b8)", "latent absmax"],
            rows, colw=[0.28, 0.14, 0.16, 0.16], fs=7.2)
    body(fig, 0.325,
         "MoDiff on the Linear layers (qkv and output projections) is implemented and measurable:\n"
         "W8A8 0.0413 -> 0.0396 (+4%), W4A4 0.4804 -> 0.4513 (+6%). It is off by default because at\n"
         "batch 128 it costs +25.5 ms/step. An earlier batch-8 measurement put that at +5.0 ms/step\n"
         "and understated it 5x.\n\n"
         "The reason it is expensive rather than cheap: the Linear path's MoDiff overhead is 296% of\n"
         "its own GEMM time, against 5% for the conv path. The a_hat/o_hat state tensors are the same\n"
         "size as the activation either way, but a projection GEMM at K=N=C does far less arithmetic\n"
         "per byte than a 3x3 conv, so the same extra traffic is a much larger fraction of it.\n\n"
         "The path is complete: three kernels, no eager ops and no host synchronization\n"
         "(delta_absmax_fp16 -> step1_static_quantize_fprop -> gemm_w{8a8,4a4}_awq_o_hat, the last\n"
         "of which accumulates o_hat and adds bias and residual only to the returned output).",
         fs=8.4)
    save(pdf, fig)


def p_simulator(pdf, sim):
    fig = page("Section 19", "MoDiff invariants over a 200-step trajectory",
               "Scope: kernel, synthetic slowly-varying trajectory at real attention shapes, 200\n"
               "steps, seed 1234. No UNet involved. e_inf = max|a_t - a_hat_t|; i2_rel = relative L2\n"
               "of o_hat against an fp64 reference; both must stay bounded for Eqs 9-10 to hold.")
    if not sim:
        body(fig, 0.85, "MISSING step_simulator.json", color=MOD_C); save(pdf, fig); return
    rows = []
    for k, v in sim["results"].items():
        first, last = v["rows"][0], v["rows"][-1]
        # i2_growth_over_run in the JSON is the ABSOLUTE increase, not a ratio -- formatting it
        # as "{:.2f}x" printed 0.00x for every row. The ratio is what the column claims to show.
        rows.append([k, f"{first['e_inf']:.2e}", f"{last['e_inf']:.2e}",
                     f"{first['i2_rel']:.2e}", f"{last['i2_rel']:.2e}",
                     f"{last['i2_rel'] / first['i2_rel']:.2f}x", f"{last['out_absmax']:.2f}",
                     "yes" if last["finite"] else "NO"])
    tbl(fig, [0.03, 0.690, 0.95, 0.155],
        ["case", "e_inf @0", "e_inf @199", "i2_rel @0", "i2_rel @199", "growth",
         "out absmax", "finite"],
        rows, colw=[0.22, 0.11, 0.11, 0.11, 0.11, 0.09, 0.11, 0.08], fs=6.7)
    ax = fig.add_axes([0.14, 0.475, 0.72, 0.19])
    for k, v in sim["results"].items():
        xs = [r["step"] for r in v["rows"]]
        ys = [r["i2_rel"] for r in v["rows"]]
        ax.plot(xs, ys, "-", lw=1.5, label=k, color=BASE_C if "W8A8" in k else MOD_C,
                alpha=0.75)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("o_hat relative L2 vs fp64", fontsize=8)
    ax.set_yscale("log")
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    body(fig, 0.425,
         "Both invariants hold for the full run at both bit-widths. e_inf is flat at one quantization\n"
         "step (4.88e-4 at W8A8, 7.8e-3 to 9.8e-3 at W4A4), which is the bound Theorem 4.3 requires:\n"
         "the error fed forward never accumulates, because Q(a_t - a_hat_{t+1}) + a_hat_{t+1} keeps\n"
         "a_hat within one step of a_t by construction.\n\n"
         "o_hat's drift against fp64 grows by 1.02-1.47x over 200 steps and ends at 5.0e-3 (W8A8)\n"
         "to 1.4e-2 (W4A4). This is the fp16 accumulator's rounding over the run. It is\n"
         "below the per-step delta-quantization error at both bit-widths, which is why o_hat is kept\n"
         "in fp16 rather than promoted to fp32 — the conv path already accumulates in fp16 and does\n"
         "not diverge.\n\n"
         "The 'finite' column is the divergence check. It caught the reporting hazard of section 2\n"
         "row 7 when the end-to-end latent check did.", fs=8.4)
    save(pdf, fig)


def p_quality(pdf, eq):
    fig = page("Section 20", "Accuracy versus step count",
               "Scope: end-to-end. Latent relative L2 against the fp16 model at 200 steps, real\n"
               "checkpoint, batch 8, steady state (run 1 discarded).")
    tbl(fig, [0.10, 0.745, 0.80, 0.135],
        ["configuration", "latent relL2", "vs own baseline", "decoded appearance"],
        [["int8_baseline", "0.2376", "—", "invents / deletes architecture"],
         ["int8 + MoDiff", "0.0385 – 0.0421", "5.6 – 6.2x better", "visually ~ fp16"],
         ["int4_baseline", "0.7810", "—", "collapses: flat brown, no structure"],
         ["int4 + MoDiff", "0.4570 – 0.4979", "1.57 – 1.71x better", "structure restored, hazy"]],
        colw=[0.19, 0.17, 0.19, 0.29], fs=7.8, hi=[1, 3])
    body(fig, 0.705,
         "Ranges are across separate processes; the spread is real and is stated rather than\n"
         "averaged away. Both MoDiff rows are far outside it.", fs=8.4)
    if eq:
        for i, (bits, c) in enumerate((("int8", BASE_C), ("int4", I4_C))):
            r = eq["results"][bits]
            ax = fig.add_axes([0.11 + i * 0.47, 0.435, 0.36, 0.19])
            stp = sorted(int(s) for s in r["baseline_curve"])
            ax.plot(stp, [r["baseline_curve"][str(s)] for s in stp], "o-", color=c, lw=1.8,
                    label=f"{bits}_baseline")
            ax.plot(stp, [r["modiff"][str(s)]["dist"] for s in stp], "s-", color=MOD_C, lw=1.8,
                    label=f"{bits} + MoDiff")
            ax.axhline(r["bar_dist"], ls="--", lw=1, color=MUTE)
            ax.set_xlabel("DDIM steps", fontsize=8)
            ax.set_ylabel("distance to fp16 @ 200", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            ax.set_ylim(0, max(r["baseline_curve"].values()) * 1.15)
            bare(ax)
    body(fig, 0.385,
         "Both baseline curves are nearly flat — int8 0.2789 at 16 steps against 0.2505 at 50, int4\n"
         "0.8109 against 0.7852. Their error is quantization, so extra steps buy them almost nothing.\n"
         "MoDiff is better at every step count at both bit-widths.\n\n"
         "The MoDiff curves are NON-MONOTONE (int8 scores 0.1292 at 32 steps and 0.1465 at 50). DDIM\n"
         "picks a different timestep subset for every S, which contributes roughly +-0.05 independent\n"
         "of quality. So the crossover with the 50-step baseline is a range, not a single point.\n\n"
         "Latent relative L2 is not FID. It orders the modes consistently with the decoded samples in\n"
         "section 22, but it is not a perceptual metric.", fs=8.4)
    save(pdf, fig)


def p_equal(pdf, eq):
    fig = page("Section 21", "Equal-accuracy speedup",
               "Scope: end-to-end, per SAMPLE rather than per step. ms/sample = steps x ms/step at\n"
               "batch 128. 'matches bar' = the configuration's distance to fp16@200 is at or below\n"
               "the baseline's at 50 steps.")
    if not eq:
        body(fig, 0.85, "MISSING steps_equal_quality.json", color=MOD_C); save(pdf, fig); return
    mspk = eq["ms_per_step_batch128"]
    rows = []
    for bits in ("int8", "int4"):
        r = eq["results"][bits]
        rows.append([f"{bits}_baseline", "50", f"{r['bar_ms']:.0f}", f"{r['bar_dist']:.4f}",
                     "1.00x", "reference"])
        for s in ("50", "32", "25", "16"):
            m = r["modiff"][s]
            rows.append([f"{bits} + MoDiff", s, f"{m['ms']:.0f}", f"{m['dist']:.4f}",
                         f"{m['speedup_vs_bar']:.2f}x", "yes" if m["matches_bar"] else "no"])
    hi = [i for i, r in enumerate(rows) if r[1] == "16"]
    tbl(fig, [0.06, 0.635, 0.90, 0.225],
        ["configuration", "steps", "ms/sample", "distance to fp16@200", "vs own baseline@50",
         "matches bar"],
        rows, colw=[0.20, 0.09, 0.13, 0.20, 0.18, 0.13], fs=7.0, hi=hi)
    body(fig, 0.598,
         f"ms/step used: int8_baseline {mspk['int8_baseline']:.2f}, int8 {mspk['int8']:.2f}, "
         f"int4_baseline {mspk['int4_baseline']:.2f}, int4 {mspk['int4']:.2f} (batch 128).", fs=8.0)
    ax = fig.add_axes([0.14, 0.395, 0.72, 0.175])
    for bits, c, mk in (("int8", BASE_C, "o"), ("int4", MOD_C, "s")):
        r = eq["results"][bits]
        st = sorted((int(s) for s in r["modiff"]), reverse=True)
        ax.plot([r["modiff"][str(s)]["ms"] for s in st],
                [r["modiff"][str(s)]["dist"] for s in st], mk + "-", color=c, lw=1.8,
                label=f"{bits} + MoDiff (label = steps)")
        for s in st:
            ax.annotate(str(s), (r["modiff"][str(s)]["ms"], r["modiff"][str(s)]["dist"]),
                        textcoords="offset points", xytext=(0, 6), fontsize=6.4, ha="center",
                        color=c)
        ax.scatter([r["bar_ms"]], [r["bar_dist"]], marker="X", s=70, color=c, zorder=5)
        ax.annotate(f"{bits} base @50", (r["bar_ms"], r["bar_dist"]), textcoords="offset points",
                    xytext=(6, -10), fontsize=6.8, color=c)
    ax.set_xlabel("ms per sample (batch 128)", fontsize=8.5)
    ax.set_ylabel("distance to fp16 @ 200", fontsize=8.5)
    ax.legend(fontsize=7.4, frameon=False)
    bare(ax)
    body(fig, 0.355,
         "Two axes, which are not interchangeable:\n\n"
         "   per STEP, fixed step count      MoDiff = 0.92x the baseline (8% slower, section 11)\n"
         "   per SAMPLE, equal accuracy      MoDiff = 2.86 - 2.91x the baseline at 16 steps\n\n"
         "MoDiff changes accuracy per function evaluation; the step count is what a diffusion sampler\n"
         "gets to reduce. The paper reports no wall-clock at all (Remark 5.1) and this is consistent\n"
         "with why: MoDiff spends a fixed bit budget better, it does not make a step cheaper.\n\n"
         "Three routes to a cheaper step were measured, and all three are closed:\n\n"
         "   1  faster per step               a_hat and o_hat are full-precision STATE tensors that\n"
         "                                    Eqs 9-10 require reading and writing every step:\n"
         "                                    ~2.2 ms/step floor.\n"
         "   2  W4A4 reaching the W8A8 bar    92% of W4A4's error is the 4-bit conv path, which\n"
         "                                    MoDiff already covers (section 18).\n"
         "   3  W8A4, the paper's own config  int4 tensor cores need BOTH operands 4-bit, so W8A4\n"
         "                                    runs on the int8 datapath at W8A8 GEMM speed. No speed\n"
         "                                    advantage on this hardware.", fs=8.0)
    save(pdf, fig)


def p_samples(pdf):
    for p, sec, title, sub in [
        (f"{K}/samples/comparison_grid.png", "Section 22",
         "Decoded samples, matched step count",
         "Same seed. Rows: fp16 / W8A8 base / W8A8+MoDiff / W4A4 base / W4A4+MoDiff. DDIM 50."),
        (f"{K}/samples_steps/steps_comparison.png", "Section 23",
         "Decoded samples, matched wall-clock budget",
         "Rows: fp16@50 / baseline@50 (3547 ms) / MoDiff@50 / MoDiff@25 (1906) / MoDiff@16 (1220).")]:
        fig = page(sec, title, sub)
        if not os.path.exists(p):
            body(fig, 0.85, f"NOT MEASURED: {p}", color=MOD_C); save(pdf, fig); continue
        img(fig, p, 0.885, width=0.88, left=0.06, max_h=0.80)
        save(pdf, fig)


def p_surface(pdf, reach, dele):
    fig = page("Section 24", "Kernel surface",
               "Scope: runtime reachability. Every modiff_cutlass export wrapped in a counting shim,\n"
               "then each mode run for 20 DDIM steps at batch 4. Fired-set diffed against the export\n"
               "list, so the classification is evidence rather than inference.")
    if reach:
        n = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in reach.items()}
        tbl(fig, [0.18, 0.705, 0.64, 0.14],
            ["measure", "count"],
            [["exports in modiff_cutlass", str(n["n_exports"])],
             [f"fired in any of {n['modes']} modes", str(n["fired_any"])],
             ["fired in steady state (any mode)", str(n["fired_steady_any_mode"])],
             ["fired only during calibration / setup", str(n["setup_only"])],
             ["never fired", str(n["never_fired"])]],
            colw=[0.40, 0.14], fs=7.8, hi=[4])
        ax = fig.add_axes([0.24, 0.545, 0.52, 0.16])
        seg = [("steady state", n["fired_steady_any_mode"], OK),
               ("calibration only", n["setup_only"], WARN),
               ("never fired", n["never_fired"], MOD_C)]
        left = 0
        for lb, v, c in seg:
            ax.barh([0], [v], left=left, color=c, height=0.5, label=f"{lb} ({v})", zorder=3)
            ax.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=8,
                    color="white", weight="bold")
            left += v
        ax.set_yticks([]); ax.set_xlim(0, n["n_exports"])
        ax.set_xlabel("exported kernel entry points", fontsize=8)
        ax.legend(fontsize=7.4, frameon=False, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.02))
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    body(fig, 0.485,
         "The three categories are not interchangeable and the distinction is what makes the count\n"
         "actionable:\n\n"
         "   steady state        on the hot path of at least one shipped mode.\n"
         "   calibration only    fires during the 8-forward attention calibration window or during\n"
         "                       static-scale export, then never again. Removing one breaks a mode\n"
         "                       on its first run only, which no steady-state benchmark detects.\n"
         "   never fired         reachable only through an env-gated rollback route, or dominated by\n"
         "                       another entry point, or genuinely dead.\n\n"
         "A kernel being unreferenced is not by itself grounds for deletion in this tree. Three\n"
         "categories are kept deliberately: executable evidence (a kernel retained because an\n"
         "in-source comment cites its measured behaviour), a rollback for a named live kill-switch,\n"
         "and a shape-generality fallback with its condition stated. Every unreferenced block is\n"
         "required to declare which it is (csrc/modiff_kernels_api.h).\n\n"
         "Sentinel hazard recorded during this pass: three fusion gates probe for a kernel with\n"
         "hasattr as a build-capability sentinel rather than calling it — fused_resblock.py:109,\n"
         "int4_optimized.py:21, int8_optimized.py:686. Deleting a sentinel silently disables the\n"
         "fusion it gates instead of raising. They were repointed to live entry points before any\n"
         "deletion.", fs=8.4)
    save(pdf, fig)


def p_summary(pdf):
    fig = page("Section 25", "Summary, and what is not established")
    body(fig, 0.895,
         "Baseline\n"
         "   Quantization alone: 1.44x (W8A8), 1.77x (W4A4) end to end against fp16.\n"
         "   Conv at layer scope: 1.65x / 3.00x. Attention at layer scope: 1.15-1.84x per shape.\n"
         "   Every quantize is folded into a producer kernel; GN statistics never materialized.\n"
         "   Accuracy cost: W8A8 0.238 latent relL2, W4A4 0.781.\n\n"
         "Attention profile\n"
         "   MoDiff does nothing inside attention: 0.99-1.03x, measured in all five modes.\n"
         "   The projection GEMMs reach 24-52% of their own roofline; cuBLAS fp16 reaches 75% of its\n"
         "   on the same shape. Distance to fp16's achieved fraction: 2.82 ms/step, 4.0% of the\n"
         "   71 ms/step int8 baseline, of which 1.96 is qkv and 0.86 the out projection.\n\n"
         "MoDiff\n"
         "   Fusion parity with the baseline; no unfused fallback anywhere.\n"
         "   Accuracy at matched steps: 5.6-6.2x better at W8A8, 1.57-1.71x at W4A4, and W4A4 goes\n"
         "   from collapse to structured output.\n"
         "   Cost: 1.08-1.09x the baseline per step, of which ~2.2 ms/step is a floor.\n"
         "   Equal-accuracy cost per sample: 2.86-2.91x faster than the baseline at 16 steps vs 50.\n"
         "   Both MoDiff invariants hold over a 200-step trajectory at both bit-widths.\n\n"
         "This session\n"
         "   MoDiff's per-step overhead against its own baseline: 23% -> 8%.\n"
         "   Four kernels added, each by parameterizing or copying a baseline kernel.\n"
         "   Delta quantizer switched from the activation grid to a dynamic per-call delta scale.\n"
         "   16/16 kernel correctness tests pass in both delta modes.", fs=8.4)
    head(fig, 0.415, "Not established", color=WARN, fs=11)
    body(fig, 0.388,
         "   FID. Every accuracy figure here is latent relative L2. The equal-accuracy step count in\n"
         "   particular needs FID per step count before it is publication-grade.\n\n"
         "   W8A4. The paper's actual configuration is untested in this tree — only W8A8 and W4A4\n"
         "   exist. The paper's headline FID table is therefore not reproduced here, and W4A4 losing\n"
         "   to the W8A8 baseline does not contradict it: this W4A4 also has 4-bit weights.\n\n"
         "   The C768/T4 attention shape disagrees with the 08-01 report by 84%. One block, T=4,\n"
         "   ~100 us/call, dispatch-bound. Not reconciled.\n\n"
         "   The fp16 GroupNorm apply is not located in the section 9 trace. fp16's GN rows are\n"
         "   statistics only and the elementwise entry cannot hold both it and the residual add.\n\n"
         "   Unit-test blind spot. 16/16 kernel tests passed while the Linear epilogue dropped its\n"
         "   bias (int8 0.039 -> 0.300) and while the delta-report hazard diverged W4A4. Both are\n"
         "   cross-kernel, within-step ordering faults; single-launch tests cannot observe them.\n"
         "   There is still no end-to-end latent regression gate in the test suite.", fs=8.4)
    fig.text(0.08, 0.085,
             "Data and scripts: docs/report_2026-08-04/{data,scripts}/  ·  "
             "docs/attn_modiff_profile_2026-08-04/  ·  docs/modiff_correctness_2026-08-03/",
             fontsize=7.4, color=MUTE, va="top")
    save(pdf, fig)


def main():
    e2e, prof = jload("e2e_wallclock.json"), jload("bucket_breakdown.json")
    conv = cload("conv_kernel_speed.csv")
    lay, micro = cload("attn_layer_speed.csv"), cload("attn_kernel_fair_speed.csv")
    reb, roof = aload("attn_stages_rebucketed.json"), aload("attn_roofline.json")
    ab, st = kload("dynamic_delta_ab.json"), kload("static_delta_q.json")
    i4t, clip = kload("int4_delta_table.json"), kload("delta_clip_sweep.json")
    refr, gn = kload("delta_refresh_sweep.json"), kload("gn_stats_ab.json")
    util, gain = kload("utilisation.json"), kload("gain_vs_steps.json")
    attr, lin = kload("int4_error_attribution.json"), kload("linear_modiff_ab.json")
    sim, eq = kload("step_simulator.json"), kload("steps_equal_quality.json")
    reach, dele = kload("kernel_reachability.json"), kload("deletion_classification.json")

    out = f"{D}/MoDiff_measurement_report_2026-08-04.pdf"
    with PdfPages(out) as pdf:
        p_cover(pdf)
        p_method(pdf)
        p_corrections(pdf)
        p_base_scheme(pdf)
        p_e2e(pdf, e2e, "baseline")
        p_conv(pdf, conv)
        p_attn_layer(pdf, lay)
        p_attn_micro(pdf, micro)
        p_stage(pdf)
        p_layer_share(pdf)
        p_idle(pdf)
        p_attn_run(pdf)
        p_attn_stages(pdf, reb)
        p_attn_fusion(pdf)
        p_attn_roofline(pdf, roof)
        p_attn_modiff(pdf, reb)
        p_mod_scheme(pdf)
        p_e2e(pdf, e2e, "modiff")
        p_profile(pdf, prof, "baseline")
        p_profile(pdf, prof, "modiff")
        p_delta_mode(pdf, ab, st, i4t)
        p_sweeps(pdf, clip, refr)
        p_gn_ab(pdf, gn)
        p_calibration(pdf, util, gain)
        p_int4_attr(pdf, attr, lin)
        p_simulator(pdf, sim)
        p_quality(pdf, eq)
        p_equal(pdf, eq)
        p_samples(pdf)
        p_surface(pdf, reach, dele)
        p_summary(pdf)
    print(f"WROTE {out}  ({_n[0]} pages)")
    if _OVERFLOW:
        print(f"\n  {len(_OVERFLOW)} text block(s) run off the page bottom:")
        for pg, snippet, end in _OVERFLOW:
            print(f"    p{pg:02d}  ends at y={end:+.3f}   {snippet}")
    else:
        print("  no text block overflows the page")
    if _TOPCLASH:
        print(f"  {len(_TOPCLASH)} table(s) reach into the scope block:")
        for pg, h0, top in _TOPCLASH:
            print(f"    p{pg:02d}  top={top:.3f}   first column '{h0}'")
    else:
        print("  no table collides with a page header")
    for n, v in [("e2e", e2e), ("buckets", prof), ("conv", conv), ("attn layer", lay),
                 ("attn micro", micro), ("attn stages", reb), ("attn roofline", roof),
                 ("delta A/B", ab), ("clip sweep", clip), ("refresh sweep", refr),
                 ("gn A/B", gn), ("utilisation", util), ("gain vs steps", gain),
                 ("int4 attribution", attr), ("linear A/B", lin), ("simulator", sim),
                 ("equal quality", eq), ("reachability", reach)]:
        print(f"  {n:18s} {'ok' if v else 'MISSING'}")


if __name__ == "__main__":
    main()
