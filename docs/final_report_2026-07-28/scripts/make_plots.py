"""All figures for the final report, from data/*.json.

  fig_e2e_speedup.png       e2e speedup vs fp16, 5 modes
  fig_icicle_<mode>.png     hierarchical (icicle) profile tree: layer type -> role -> kernel
  fig_layer_pipeline_speedup.png   per-layer-type pipeline speedup, every real shape
  fig_intra_layer_*.png     inside each layer, by role (absolute us and % variants)
  fig_profile_tree.png      hierarchical time cost: layer type -> role, per mode
  fig_layer_stack.png       absolute ms/step stacked by layer type, all modes side by side
"""
import json, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = "docs/final_report_2026-07-28"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
LBL = ["fp16", "int8\nbaseline", "int4\nbaseline", "int8\nmodiff", "int4\nmodiff"]
# Consistent layer-type colors across every figure in the report.
LAYER_COLORS = {
    "Attention": "#E24B4A", "Conv": "#378ADD", "Linear-GEMM": "#BA7517",
    "Normalization": "#1D9E75", "Quantize": "#7F77DD", "Resize": "#888780",
    "Elementwise-Cast": "#D3D1C7", "Other / unclassified": "#000000",
}


def load(name):
    try:
        return json.load(open(f"{HERE}/data/{name}"))
    except FileNotFoundError:
        return None


def load_tree():
    """Prefer the caller-attributed profile when present.

    profile_tree.json classifies kernels by NAME, which cannot tell apart the same kernel
    serving different callers -- in fp16 mode cuBLAS dispatches SDPA's QK^T/AV (aten::bmm)
    to a `cutlass_..._gemm` kernel indistinguishable from a plain Linear GEMM, so 44 ms/step
    of attention work was billed to Linear-GEMM. profile_tree_by_caller.json joins kernels
    to the ATen op that launched them and is therefore the correct source; keep the
    name-based file only as a fallback.
    """
    t = load("profile_tree_by_caller.json")
    if t:
        return t, "by-caller"
    return load("profile_tree.json"), "by-name"


def fig_e2e_speedup(tree):
    if not tree:
        return
    base = tree["fp16"]["ms_step"]
    modes = MODES[1:]
    sp = [base / tree[m]["ms_step"] for m in modes]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(range(len(modes)), sp, color="#1D9E75", width=0.6)
    for i, (b, m) in enumerate(zip(bars, modes)):
        ax.text(i, b.get_height() + 0.02, f"{b.get_height():.2f}x",
                ha="center", fontweight="bold")
        ax.text(i, 0.08, f"{tree[m]['ms_step']:.1f} ms", ha="center",
                color="white", fontsize=9)
    ax.axhline(1.0, ls="--", color="#888780", lw=1)
    ax.text(len(modes) - 0.5, 1.02, f"fp16 = {base:.1f} ms/step", ha="right",
            color="#888780", fontsize=9)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([l for l in LBL[1:]])
    ax.set_ylabel("speedup vs fp16")
    ax.set_title(f"End-to-end speedup vs fp16 (b128, A40, same session)")
    ax.set_ylim(0, max(sp) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_e2e_speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_e2e_speedup.png")


def fig_layer_stack(tree):
    if not tree:
        return
    layers = ["Attention", "Conv", "Linear-GEMM", "Normalization",
              "Quantize", "Resize", "Elementwise-Cast", "Other / unclassified"]
    layers = [l for l in layers
              if any(l in tree[m]["tree"] for m in MODES)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(MODES))
    for l in layers:
        vals = np.array([tree[m]["tree"].get(l, {}).get("ms_step", 0.0) for m in MODES])
        ax.bar(range(len(MODES)), vals, bottom=bottom, label=l,
               color=LAYER_COLORS.get(l, "#999999"), width=0.62)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(i, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom += vals
    for i, m in enumerate(MODES):
        ax.text(i, bottom[i] + 2, f"{tree[m]['ms_step']:.1f} ms",
                ha="center", fontweight="bold", fontsize=9.5)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(LBL)
    ax.set_ylabel("ms / step (absolute)")
    ax.set_title("Time cost by layer type — absolute ms/step, b128")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)
    ax.set_ylim(0, max(bottom) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_layer_stack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_layer_stack.png")


def fig_profile_tree(tree, mode="int4_baseline"):
    """Horizontal 2-level tree for one mode: layer type, then its roles."""
    if not tree or mode not in tree:
        return
    t = tree[mode]["tree"]
    rows, colors, is_layer = [], [], []
    for layer, node in t.items():
        rows.append((f"{layer}", node["ms_step"], node["pct_of_total"]))
        colors.append(LAYER_COLORS.get(layer, "#999999"))
        is_layer.append(True)
        for role, rnode in node["roles"].items():
            if rnode["ms_step"] < 0.05:
                continue
            rows.append((f"      {role}", rnode["ms_step"], rnode["pct_of_total"]))
            colors.append(LAYER_COLORS.get(layer, "#999999"))
            is_layer.append(False)
    fig, ax = plt.subplots(figsize=(13, 0.34 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    # matplotlib's alpha must be scalar, so fold the layer/role distinction into RGBA.
    import matplotlib.colors as mcolors
    rgba = [mcolors.to_rgba(c, 1.0 if L else 0.45) for c, L in zip(colors, is_layer)]
    ax.barh(y, vals, color=rgba, height=0.72)
    for yy, (label, ms, pct), L in zip(y, rows, is_layer):
        ax.text(ms + max(vals) * 0.012, yy, f"{ms:.2f} ms  ({pct:.1f}%)",
                va="center", fontsize=8.5, fontweight="bold" if L else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.shorten(r[0], 78, placeholder="...") for r in rows],
                       fontsize=8.5,
                       fontfamily="monospace")
    for tick, L in zip(ax.get_yticklabels(), is_layer):
        tick.set_fontweight("bold" if L else "normal")
    ax.set_xlabel("ms / step")
    ax.set_xlim(0, max(vals) * 1.26)
    nk = tree[mode].get('n_distinct_kernels')
    nk_txt = f", {nk} distinct CUDA kernels" if nk else ""
    ax.set_title(f"Hierarchical profile — {mode} ({tree[mode]['ms_step']:.1f} ms/step{nk_txt})\n"
                 f"bold = layer type, indented = role within it")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_profile_tree_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_profile_tree_{mode}.png")


def main_e2e():
    tree, src = load_tree()
    print(f"[tree source] {src}")
    fig_e2e_speedup(tree)
    fig_layer_stack(tree)
    if tree:
        for m in ("int4_baseline", "int8_baseline", "int4_modiff", "fp16"):
            fig_profile_tree(tree, m)


# ---------------------------------------------------------------------------
# Layer-pipeline figures (from layer_pipeline_bench.py): per-layer-type pipeline
# speedup, and the INTRA-layer time split (which roles eat a layer's own GPU time).
# ---------------------------------------------------------------------------
ROLE_COLORS = {}


# Elementwise-Cast's layer color (#D3D1C7) is already near-white, so shading it produces
# several roles that are indistinguishable in a legend. Give that family a darker base and
# spread its variants over hue as well as lightness.
_ROLE_BASE_OVERRIDE = {"Elementwise-Cast": "#8C8880", "Memory-op": "#5F5E5A",
                       "Sampler-side": "#6B6560"}


def _role_color(role, layer_type):
    """Stable color per role: same hue family as its layer type, distinguishable within it."""
    if role not in ROLE_COLORS:
        base = _ROLE_BASE_OVERRIDE.get(layer_type, LAYER_COLORS.get(layer_type, "#999999"))
        n = sum(1 for r in ROLE_COLORS if ROLE_COLORS[r][1] == layer_type)
        import matplotlib.colors as mc
        import colorsys
        h, l, sat = colorsys.rgb_to_hls(*mc.to_rgb(base))
        # walk lightness up and hue slightly, so 5+ roles in one family stay separable
        l = min(0.88, l + 0.13 * (n % 4))
        h = (h + 0.035 * (n // 4)) % 1.0
        ROLE_COLORS[role] = (colorsys.hls_to_rgb(h, l, sat), layer_type)
    return ROLE_COLORS[role][0]


def fig_layer_pipeline_speedup(lp):
    """Per-layer-type pipeline speedup vs the fp16 pipeline, at every real shape."""
    if not lp:
        return
    modes = [m for m in ["int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
             if m in lp["modes"]]
    kinds = ["resblock_plain", "resblock_updown", "attention"]
    kinds = [k for k in kinds if any(r["kind"] == k for r in lp["modes"]["fp16"])]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(13, 4.2 * len(kinds)), squeeze=False)
    colors = {"int8_baseline": "#378ADD", "int4_baseline": "#1D9E75",
              "int8_modiff": "#BA7517", "int4_modiff": "#E24B4A"}
    for ax, kind in zip(axes[:, 0], kinds):
        ref = [r for r in lp["modes"]["fp16"] if r["kind"] == kind]
        ref.sort(key=lambda r: -(r["x_shape"][1] * r["x_shape"][2] * r["x_shape"][3]))
        keys = [tuple(r["x_shape"]) for r in ref]
        labels = [f"C{k[1]}\n{k[2]}x{k[3]}" for k in keys]
        x = np.arange(len(keys))
        w = 0.8 / max(1, len(modes))
        for i, m in enumerate(modes):
            byshape = {tuple(r["x_shape"]): r for r in lp["modes"][m] if r["kind"] == kind}
            sp = [byshape.get(k, {}).get("speedup_vs_fp16") or np.nan for k in keys]
            ax.bar(x + (i - (len(modes) - 1) / 2) * w, sp, w, label=m, color=colors.get(m))
        ax.axhline(1.0, ls="--", color="#555", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("pipeline speedup vs fp16")
        n_inst = {tuple(r["x_shape"]): r["n_instances"] for r in ref}
        ax.set_title(f"{kind}  —  whole-layer kernel-pipeline speedup vs the fp16 pipeline "
                     f"(b{lp['batch']}); x-labels are input C and HxW", fontsize=10.5)
        ax.legend(frameon=False, ncol=len(modes), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_layer_pipeline_speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_layer_pipeline_speedup.png")


def fig_intra_layer_split(lp, mode="int4_baseline", kind="resblock_plain", topn=10,
                          yaxis="abs"):
    """Inside each layer instance, which roles eat its GPU time.

    Both views are produced and both are kept, because they answer different questions:

      yaxis="abs"  Y = real us per layer call. Bar HEIGHT is comparable across shapes, so
                   this is the view that shows WHERE THE TIME IS (a 3369 us layer towers
                   over a 567 us one).
      yaxis="pct"  Y = % of that layer's own GPU time, every bar normalized to 100%. This
                   is the view that shows COMPOSITION, i.e. how the mix shifts with shape
                   independent of the layer's absolute cost -- which the absolute view
                   compresses out of visibility on the small shapes.

    Filenames: fig_intra_layer_<kind>_<mode>.png (abs) and ..._pct.png (percentage).
    """
    if not lp or mode not in lp["modes"]:
        return
    rows = [r for r in lp["modes"][mode] if r["kind"] == kind and r.get("roles")]
    if not rows:
        return
    rows.sort(key=lambda r: -(r["x_shape"][1] * r["x_shape"][2] * r["x_shape"][3]))
    rows = rows[:topn]
    all_roles = {}
    for r in rows:
        for role, a in r["roles"].items():
            all_roles[role] = all_roles.get(role, 0.0) + a["us"]
    order = [r for r, _ in sorted(all_roles.items(), key=lambda kv: -kv[1])]
    key = "us" if yaxis == "abs" else "pct_of_layer"

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for role in order:
        vals = np.array([r["roles"].get(role, {}).get(key, 0.0) for r in rows])
        lt = next((k["layer_type"] for r in rows for k in r["kernels"] if k["role"] == role),
                  "Other / unclassified")
        ax.bar(x, vals, bottom=bottom, width=0.68,
               label=textwrap.shorten(role, 58, placeholder="..."),
               color=_role_color(role, lt))
        bottom += vals

    # Label segments, thresholding relative to the tallest bar so labels stay legible at
    # any scale (a fixed threshold drops every label on the short bars in the abs view).
    ymax = float(bottom.max()) if len(bottom) else 1.0
    bottom2 = np.zeros(len(rows))
    fmt = (lambda v: f"{v:.0f}") if yaxis == "abs" else (lambda v: f"{v:.0f}")
    for role in order:
        vals = np.array([r["roles"].get(role, {}).get(key, 0.0) for r in rows])
        for i, (v, b) in enumerate(zip(vals, bottom2)):
            if v >= ymax * 0.035:
                ax.text(i, b + v / 2, fmt(v), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom2 += vals

    ax.set_xticks(x)
    if yaxis == "abs":
        for i, tot in enumerate(bottom):
            ax.text(i, tot + ymax * 0.012, f"{tot:.0f} us", ha="center",
                    fontsize=8.5, fontweight="bold", color="#2C2C2A")
        ax.set_xticklabels([f"C{r['x_shape'][1]}\n{r['x_shape'][2]}x{r['x_shape'][3]}"
                            for r in rows], fontsize=8.5)
        ax.set_ylabel("us per layer call (absolute GPU time)")
        ax.set_ylim(0, ymax * 1.13)
        sub = "bar height is real GPU time, so heights are comparable across shapes"
    else:
        # keep the absolute total on the x label so the normalized view stays interpretable
        ax.set_xticklabels([f"C{r['x_shape'][1]}\n{r['x_shape'][2]}x{r['x_shape'][3]}\n"
                            f"{r['pipeline_us']:.0f}us" for r in rows], fontsize=8)
        ax.set_ylabel("% of this layer's own GPU time")
        ax.set_ylim(0, 105)
        sub = ("every bar normalized to 100% to show COMPOSITION; the layer's absolute "
               "cost is printed under each label")
    ax.set_title(f"Inside each layer — {'absolute' if yaxis == 'abs' else 'relative'} time "
                 f"split by role  ({kind}, {mode}, b{lp['batch']})\n{sub}", fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    suffix = "" if yaxis == "abs" else "_pct"
    fig.savefig(f"{HERE}/plots/fig_intra_layer_{kind}_{mode}{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_intra_layer_{kind}_{mode}{suffix}.png")


def main_layers():
    lp = load("layer_pipeline_bench.json")
    fig_layer_pipeline_speedup(lp)
    if lp:
        for mode in ("fp16", "int8_baseline", "int4_baseline", "int4_modiff"):
            for kind in ("resblock_plain", "resblock_updown", "attention"):
                fig_intra_layer_split(lp, mode, kind, yaxis="abs")
                fig_intra_layer_split(lp, mode, kind, yaxis="pct")





# ---------------------------------------------------------------------------
# Icicle (hierarchical) plot of the profile tree: 3 stacked levels, each level
# partitioning the SAME x range, width proportional to ms/step. This shows the
# grouping (which kernels roll up into which role, which roles into which layer
# type) and the time cost in one picture -- a plain bar chart can only do one.
# ---------------------------------------------------------------------------
def fig_profile_icicle(tree, mode="int4_baseline", min_ms=0.05, kernel_level=True):
    if not tree or mode not in tree:
        return
    import matplotlib.colors as mcolors
    t = tree[mode]["tree"]
    total = sum(n["ms_step"] for n in t.values())
    fig_h = 5.2 if kernel_level else 3.4
    fig, ax = plt.subplots(figsize=(16, fig_h))

    ROW = {0: (2.10, 0.92), 1: (1.08, 0.92), 2: (0.06, 0.92)}   # level -> (y, height)

    def draw(x, w, level, label, color, alpha, ms):
        y, h = ROW[level]
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=mcolors.to_rgba(color, alpha),
                                   edgecolor="white", linewidth=1.1))
        # only label if the box is wide enough to hold readable text
        frac = w / total
        if frac >= 0.042:
            fs = 9.0 if level == 0 else (8.0 if level == 1 else 7.0)
            txt = label if len(label) <= int(frac * 165) else label[:max(3, int(frac * 165))] + "…"
            # Pick text color from the ACTUAL blended background luminance -- a fixed
            # "white on level 0" breaks on light layer colors (Elementwise-Cast is #D3D1C7).
            bg = np.array(mcolors.to_rgb(color)) * alpha + (1 - alpha)   # over white
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            fg = "#FFFFFF" if lum < 0.55 else "#1A1A1A"
            fg2 = "#F0F0F0" if lum < 0.55 else "#3A3A3A"
            ax.text(x + w / 2, y + h * 0.60, txt, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if level == 0 else "normal", color=fg)
            ax.text(x + w / 2, y + h * 0.22, f"{ms:.1f} ms · {frac*100:.1f}%",
                    ha="center", va="center", fontsize=fs - 1.2, color=fg2)

    x0 = 0.0
    for layer, node in t.items():
        base = LAYER_COLORS.get(layer, "#999999")
        draw(x0, node["ms_step"], 0, layer, base, 1.0, node["ms_step"])
        xr = x0
        for role, rnode in node["roles"].items():
            draw(xr, rnode["ms_step"], 1, role, base, 0.55, rnode["ms_step"])
            if kernel_level:
                xk = xr
                # group the sub-min_ms tail into one "(+N small)" box so the row stays readable
                ks = [k for k in rnode["kernels"] if k["ms_step"] >= min_ms]
                tail = [k for k in rnode["kernels"] if k["ms_step"] < min_ms]
                for k in ks:
                    draw(xk, k["ms_step"], 2, k["kernel"], base, 0.27, k["ms_step"])
                    xk += k["ms_step"]
                if tail:
                    tms = sum(k["ms_step"] for k in tail)
                    draw(xk, tms, 2, f"(+{len(tail)} small)", base, 0.18, tms)
            xr += rnode["ms_step"]
        x0 += node["ms_step"]

    ax.set_xlim(0, total)
    ax.set_ylim(0, 3.15)
    ax.set_yticks([ROW[0][0] + 0.46, ROW[1][0] + 0.46, ROW[2][0] + 0.46][:3 if kernel_level else 2])
    ax.set_yticklabels(["layer type", "role", "kernel"][:3 if kernel_level else 2],
                       fontsize=10, fontweight="bold")
    ax.set_xlabel("ms / step  (box width is proportional to time; each row partitions the same total)")
    nk = tree[mode].get('n_distinct_kernels')
    nk_txt = f", {nk} distinct CUDA kernels" if nk else ""
    ax.set_title(f"Profile tree — {mode}: {tree[mode]['ms_step']:.2f} ms/step{nk_txt}\n"
                 f"layer type → role → kernel; boxes narrower than ~5.5% are unlabelled, "
                 f"kernels under {min_ms} ms are pooled as \"(+N small)\"", fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_icicle_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_icicle_{mode}.png")


def fig_icicle_all(tree):
    for m in ("fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"):
        fig_profile_icicle(tree, m)


# Single entry point, kept at the very END of the file: every helper above must already be
# defined when it runs (an earlier __main__ block here raised NameError on fig_icicle_all).
def fig_fp16_baselines(bv, tree):
    """fp16 baseline variants + dual-baseline speedups.

    The repo's fp16 "baseline" is not vanilla PyTorch and is measurably slower than it, so a
    single speedup number is ambiguous. Left panel: what each fp16 variant costs. Right
    panel: every quantized mode's speedup against BOTH baselines, so the reader cannot
    accidentally read one for the other.
    """
    if not bv or not tree:
        return
    order = ["vanilla_nchw", "vanilla", "repo_fp16", "vanilla_math"]
    order = [v for v in order if v in bv and "ms_step" in bv[v]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    ax = axes[0]
    vals = [bv[v]["ms_step"] for v in order]
    ours = [bv[v]["repo_kernel_ms_step"] for v in order]
    x = np.arange(len(order))
    ax.bar(x, vals, 0.6, color=["#1D9E75", "#5FB49C", "#E24B4A", "#888780"][:len(order)])
    ax.bar(x, ours, 0.6, color="#000000", alpha=0.30,
           label="of which: this repo's own kernels")
    for i, (t_, o) in enumerate(zip(vals, ours)):
        ax.text(i, t_ + 3, f"{t_:.1f} ms", ha="center", fontweight="bold", fontsize=10)
        if o > 0.5:
            ax.text(i, o / 2, f"{o:.0f}", ha="center", va="center", color="white", fontsize=8)
    labels = {"vanilla_nchw": "vanilla_nchw\n(pure PyTorch)", "vanilla": "vanilla\n(+channels_last)",
              "repo_fp16": "repo_fp16\n(repo baseline)", "vanilla_math": "vanilla_math\n(forced MATH)"}
    ax.set_xticks(x); ax.set_xticklabels([labels.get(v, v) for v in order], fontsize=9)
    ax.set_ylabel("ms / step")
    ax.set_title("fp16 baseline variants -- one axis at a time (b128)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    pure = bv["vanilla_nchw"]["ms_step"]
    repo = tree["fp16"]["ms_step"]
    modes = [m for m in ["int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
             if m in tree]
    x = np.arange(len(modes)); w = 0.38
    sp_pure = [pure / tree[m]["ms_step"] for m in modes]
    sp_repo = [repo / tree[m]["ms_step"] for m in modes]
    ax.bar(x - w / 2, sp_pure, w, label=f"vs PURE PyTorch fp16 ({pure:.0f} ms)", color="#1D9E75")
    ax.bar(x + w / 2, sp_repo, w, label=f"vs repo fp16 baseline ({repo:.0f} ms)", color="#E24B4A")
    for i, (a, b) in enumerate(zip(sp_pure, sp_repo)):
        ax.text(i - w / 2, a + 0.03, f"{a:.2f}×", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, b + 0.03, f"{b:.2f}×", ha="center", fontsize=9)
    ax.axhline(1.0, ls="--", color="#555", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
    ax.set_ylabel("speedup")
    ax.set_title("Same measurements, two baselines -- speedups differ by ~39%", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(0, max(sp_repo) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_fp16_baselines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_fp16_baselines.png")


def fig_attn_stages(sp):
    """Per-AttentionBlock kernel composition in absolute us, all shapes x 3 modes.

    Absolute (not %) so bar heights compare across shapes -- the C192/32x32 block is ~20x
    the cost of C768/2x2 and a normalized view would hide that.
    """
    if not sp:
        return
    import matplotlib.colors as mcolors
    modes = [m for m in ("fp16", "int8_baseline", "int4_baseline") if m in sp["modes"]]
    # stable color per kernel family across all panels
    FAM = [("pytorch_flash", "PyTorch flash SDPA (S4)", "#E24B4A"),
           ("ImplicitGemmConvolutionFusionPerSample", "fused GN->qkv (S1+S2)", "#1D9E75"),
           ("group_norm_silu_quantize", "GN+quantize fused (S1+S2q)", "#1D9E75"),
           ("group_norm_silu_nhwc", "GN only (S1, unfused)", "#7FCBB0"),
           ("gemm_w8a8", "int GEMM qkv+proj (S2/S5)", "#378ADD"),
           ("gemm_w4a4", "int GEMM qkv+proj (S2/S5)", "#378ADD"),
           ("ampere_fp16", "fp16 GEMM (S2/S5)", "#7FB3E8"),
           ("cutlass::Kernel2", "fp16 GEMM (S2/S5)", "#7FB3E8"),
           ("sm80_xmma_gemm", "fp16 GEMM (S2/S5)", "#7FB3E8"),
           ("quant_attn_out", "transpose+quantize (S5 prep)", "#BA7517"),
           ("quant_act_int4_pack", "standalone int4 pack (unfused)", "#E8A33D"),
           ("gn_accum", "GN stats helper", "#A8DCC8"),
           ("gn_finalize", "GN stats helper", "#A8DCC8"),
           ("elementwise", "elementwise / residual (S6)", "#B0ADA5"),
    ]

    def fam(name):
        for pat, label, col in FAM:
            if pat in name:
                return label, col
        return "other", "#D3D1C7"

    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 6.0), squeeze=False)
    seen = {}
    for ax, mode in zip(axes[0], modes):
        rows = sp["modes"][mode]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        # accumulate per family so one family = one stacked segment
        per_row = []
        for r in rows:
            acc = {}
            for k in r["kernels"]:
                lab, col = fam(k["kernel"])
                a = acc.setdefault(lab, [0.0, col])
                a[0] += k["us"]
            per_row.append(acc)
        labels_order = []
        for _, lab, _c in FAM:
            if lab not in labels_order and any(lab in pr for pr in per_row):
                labels_order.append(lab)
        for lab in labels_order:
            vals = np.array([pr.get(lab, [0.0, "#000"])[0] for pr in per_row])
            col = next(pr[lab][1] for pr in per_row if lab in pr)
            h = ax.bar(x, vals, bottom=bottom, width=0.66, color=col,
                       label=lab if lab not in seen else None)
            seen[lab] = True
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= max(1.0, 0.045 * float((bottom + vals).max())):
                    ax.text(i, b + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals
        for i, (r, tot) in enumerate(zip(rows, bottom)):
            ax.text(i, tot + tot.max() * 0.01 if hasattr(tot, "max") else tot + 20,
                    f"{r['full_us']:.0f}us", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{r['C']}\n{r['H']}x{r['W']}\nT={r['T']}\nx{r['n_instances']}"
                            for r in rows], fontsize=7.5)
        ax.set_ylabel("us per block call" if mode == modes[0] else "")
        ax.set_title(f"{mode}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"AttentionBlock: per-kernel time inside one block, every real shape "
                 f"(b{sp['batch']}, absolute us)", fontsize=12)
    handles, labs = [], []
    for a in axes[0]:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labs:
                handles.append(h); labs.append(l)
    fig.legend(handles, labs, frameon=False, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(f"{HERE}/plots/fig_attn_stages.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_attn_stages.png")


if __name__ == "__main__":
    main_e2e()
    main_layers()
    _t = load_tree()[0]
    fig_icicle_all(_t)
    fig_fp16_baselines(load("fp16_baseline_variants.json"), _t)
    fig_attn_stages(load("attn_stage_profile.json"))
