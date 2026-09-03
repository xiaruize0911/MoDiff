"""Speedup and peak-memory response of kernel 1 to each shape axis.

Palette slots 1-3 of the dataviz reference instance (#2a78d6 / #eb6834 / #1baf7a), used
unchanged, same as docs/conv_shape_sweep_2026-09-02/scripts/make_plots.py.
"""
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/workspace/MoDiff"; os.chdir(ROOT)
D = json.load(open("docs/ahat_only_conv_2026-09-02/data/kernel1_axis_sweep.json"))
M = json.load(open("docs/ahat_only_conv_2026-09-02/data/kernel1_arms.json"))
OUT = "docs/ahat_only_conv_2026-09-02/plots"; os.makedirs(OUT, exist_ok=True)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white",
                     "axes.facecolor": "white"})
S4 = "#9b59b6"
ARMS = [("baseline_generic", "per-tensor GENERIC (not shipped)", S4),
        ("a_hat fp16", "MoDiff, a_hat fp16", S1),
        ("a_hat i8 B=16", "MoDiff, a_hat i8 B=16", S2),
        ("a_hat i8 B=32", "MoDiff, a_hat i8 B=32", S3)]
AXES = [("B", "batch N"), ("C", "channels C"), ("H", "height H"), ("W", "width W")]
PRECS = [("int8", "W8A8"), ("int4", "W4A4")]
def rows(ax): return [r for r in D["sweeps"][ax]]
def get(r, prec, arm, field):
    v = r["arms"].get(f"{prec}/{arm}")
    return v[field] if v and field in v else None

def grid(fname, title, ylab, valfn, ref=None, logy=False, extra=None):
    fig, axs = plt.subplots(2, 4, figsize=(15.5, 6.6), sharey="row")
    for pi, (prec, ptitle) in enumerate(PRECS):
        for ai, (ax_name, ax_lab) in enumerate(AXES):
            ax = axs[pi][ai]
            rr = rows(ax_name)
            xs = [r["value"] for r in rr]
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1.2, ls="--", zorder=1)
            if extra:
                extra(ax, rr, prec)
            for arm, lab, col in ARMS:
                ys = [valfn(r, prec, arm) for r in rr]
                pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if not pts: continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, lw=1.6,
                        ms=4.2, label=lab, zorder=3)
            ax.set_xscale("log", base=2)
            if logy: ax.set_yscale("log", base=2)
            ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
            ax.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            if pi == 1: ax.set_xlabel(ax_lab)
            if ai == 0: ax.set_ylabel(f"{ptitle}\n{ylab}")
            if pi == 0 and ai == 0: ax.legend(frameon=False, fontsize=7.6, loc="best")
    d = D["default"]
    fig.suptitle(f"{title}\nkernel 1 = fused GroupNorm(+SiLU) -> quantize, one axis at a time; "
                 f"default N={d['B']} C={d['C']} H={d['H']} W={d['W']}, num_groups=32, "
                 f"{D['gpu']}", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/{fname}", dpi=150); plt.close(fig)
    print(f"  {OUT}/{fname}")

# ---- Figure 1: speedup vs the no-MoDiff baseline -------------------------------------------
def speedup(r, prec, arm):
    b = get(r, prec, "baseline", "ms"); v = get(r, prec, arm, "ms")
    return b / v if (b and v) else None
grid("kernel1_speedup_axes.png",
     "Kernel-1 speedup over the SHIPPED per-tensor baseline (group_norm_silu_quantize*_fast,\n"
     "which is what _gnq() resolves to at the default MODIFF_GN_FAST=1). Dashed = parity.",
     "speedup vs baseline", speedup, ref=1.0)

# ---- Figure 2: peak memory, absolute ---------------------------------------------------------
def peak(r, prec, arm): return get(r, prec, arm, "peak")
def draw_base(ax, rr, prec):
    ys = [get(r, prec, "baseline", "peak") for r in rr]
    xs = [r["value"] for r in rr]
    ax.plot(xs, ys, "-s", color=INK, lw=1.6, ms=4.0, label="baseline (no MoDiff)", zorder=2)
grid("kernel1_peak_axes.png",
     "Kernel-1 peak allocated memory (input + a_hat cache + block scales + output)",
     "peak alloc (MiB)", peak, logy=True, extra=draw_base)

# ---- Figure 3: peak memory relative to baseline ----------------------------------------------
def peak_ratio(r, prec, arm):
    b = get(r, prec, "baseline", "peak"); v = get(r, prec, arm, "peak")
    return v / b if (b and v) else None
grid("kernel1_peak_ratio_axes.png",
     "Kernel-1 peak memory RELATIVE to the no-MoDiff baseline (lower is better; dashed = parity)",
     "peak / baseline peak", peak_ratio, ref=1.0)

# ---- Figure 4: the 20 real UNet conv shapes -------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(15.5, 6.6), sharey="col")
for pi, (prec, ptitle) in enumerate(PRECS):
    for ci, (xkey, xlab, field) in enumerate(
            [("C", "channels C", "sp"), ("HW", "spatial H*W", "sp"),
             ("C", "channels C", "pk"), ("HW", "spatial H*W", "pk")]):
        ax = axs[pi][ci]
        ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
        for arm, lab, col in ARMS:
            X, Y, Sz = [], [], []
            for r in M["shapes"]:
                b = r["arms"].get(f"{prec}/baseline")
                v = r["arms"].get(f"{prec}/MoDiff {arm}") or r["arms"].get(f"{prec}/{arm}")
                if not (b and v and v[0] is not None): continue
                X.append(r["C"] if xkey == "C" else r["H"] * r["W"])
                Y.append(b[0] / v[0] if field == "sp" else v[1] / b[1])
                Sz.append(12 + 9 * r["freq"])
            ax.scatter(X, Y, s=Sz, c=col, alpha=0.72, edgecolors="white",
                       linewidths=0.6, label=lab, zorder=3)
        ax.set_xscale("log", base=2)
        ax.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if pi == 1: ax.set_xlabel(xlab)
        ylab = "speedup vs baseline" if field == "sp" else "peak / baseline peak"
        if ci in (0, 2): ax.set_ylabel(f"{ptitle}\n{ylab}")
        if pi == 0 and ci == 0: ax.legend(frameon=False, fontsize=7.6, loc="best")
fig.suptitle("The 20 real churches-UNet conv input shapes (batch 128). Marker area is how often "
             "that shape occurs.\nLeft two columns: speedup. Right two: peak memory vs baseline. "
             f"H==W in every real shape, so the spatial axis is one degree of freedom. {M['gpu']}",
             fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(f"{OUT}/kernel1_model_shapes.png", dpi=150); plt.close(fig)
print(f"  {OUT}/kernel1_model_shapes.png")

# ---- Figure 5: the CLEAN a_hat-only comparison ----------------------------------------------
# "vs baseline" conflates two things: the baseline runs the group-major single-pass kernel
# (group_norm_silu_quantize_nhwc) while every MoDiff arm runs gn_group_stats + the flat
# element-major vec2 apply. Comparing the int8-a_hat arms against the fp16-a_hat arm holds the
# kernel structure fixed, so the ratio is purely the cost of a_hat's bytes.
ARMS2 = [a for a in ARMS if a[0] not in ("a_hat fp16", "baseline_generic")]
def vs_fp16(r, prec, arm):
    b = get(r, prec, "a_hat fp16", "ms"); v = get(r, prec, arm, "ms")
    return b / v if (b and v) else None
_ARMS_SAVE = list(ARMS)
ARMS[:] = ARMS2
grid("kernel1_ahat_only_axes.png",
     "int8 a_hat vs fp16 a_hat at IDENTICAL kernel structure (isolates a_hat's byte cost)",
     "speedup vs a_hat fp16", vs_fp16, ref=1.0)
ARMS[:] = _ARMS_SAVE
