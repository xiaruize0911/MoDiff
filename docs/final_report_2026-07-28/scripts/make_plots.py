"""All figures for the final report, from data/*.json.

  fig_e2e_speedup.png       e2e speedup vs fp16, 5 modes
  fig_profile_tree.png      hierarchical time cost: layer type -> role, per mode
  fig_layer_stack.png       absolute ms/step stacked by layer type, all modes side by side
  fig_kernel_conv.png       conv kernel speedup vs cuDNN fp16, every real shape
  fig_kernel_gn.png         GN(+SiLU[+quantize]) fused vs reference, every real shape
  fig_kernel_fusions.png    this session's resize+quantize / skip-concat fusions
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
    ax.set_title(f"Hierarchical profile — {mode} ({tree[mode]['ms_step']:.1f} ms/step, "
                 f"{tree[mode]['n_distinct_kernels']} distinct CUDA kernels)\n"
                 f"bold = layer type, indented = role within it")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_profile_tree_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_profile_tree_{mode}.png")


def fig_kernel_conv(kd):
    if not kd:
        return
    rows = [r for r in kd["conv"] if r.get("fp16_us")]
    if not rows:
        return
    rows.sort(key=lambda r: (r["C"], r["H"], r["K"]))
    labels = [f"{r['C']}->{r['K']}\n{r['H']}x{r['W']}" + (f" s{r['stride']}" if r['stride'] != 1 else "")
              for r in rows]
    series = [("int8_baseline_us", "int8 baseline", "#378ADD"),
              ("int4_baseline_us", "int4 baseline", "#1D9E75"),
              ("int8_modiff_us", "int8 modiff", "#BA7517"),
              ("int4_modiff_us", "int4 modiff", "#E24B4A")]
    x = np.arange(len(rows))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(11, 0.85 * len(rows)), 5.4))
    for i, (key, lab, col) in enumerate(series):
        sp = [(r["fp16_us"] / r[key]) if r.get(key) else np.nan for r in rows]
        ax.bar(x + (i - 1.5) * w, sp, w, label=lab, color=col)
    ax.axhline(1.0, ls="--", color="#555", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("speedup vs cuDNN fp16 conv")
    ax.set_title(f"Conv kernel speedup vs fp16, every real UNet conv shape (b{kd['batch']})\n"
                 "above 1.0 = the quantized kernel is faster at that shape")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_kernel_conv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_kernel_conv.png")


def fig_kernel_gn(kd):
    if not kd:
        return
    rows = [r for r in kd["groupnorm"] if r.get("fp16_gn_silu_us")]
    if not rows:
        return
    rows.sort(key=lambda r: (r["C"], r["H"]))
    labels = [f"C{r['C']}\n{r['H']}x{r['W']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(rows)), 5.2))
    a = [r["fp16_gn_silu_us"] / r["fused_gn_silu_us"] if r.get("fused_gn_silu_us") else np.nan
         for r in rows]
    b = [r["twostep_gn_then_quant_int8_us"] / r["fused_gn_silu_quant_int8_us"]
         if r.get("fused_gn_silu_quant_int8_us") and r.get("twostep_gn_then_quant_int8_us") else np.nan
         for r in rows]
    c = [r["twostep_gn_then_quant_int4_us"] / r["fused_gn_silu_quant_int4_us"]
         if r.get("fused_gn_silu_quant_int4_us") and r.get("twostep_gn_then_quant_int4_us") else np.nan
         for r in rows]
    ax.bar(x - w, a, w, label="GN+SiLU kernel vs F.group_norm+F.silu", color="#1D9E75")
    ax.bar(x, b, w, label="GN+SiLU+int8-quant FUSED vs 2-kernel", color="#378ADD")
    ax.bar(x + w, c, w, label="GN+SiLU+int4-quant FUSED vs 2-kernel", color="#BA7517")
    ax.axhline(1.0, ls="--", color="#555", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("speedup vs reference")
    ax.set_title(f"GroupNorm(+SiLU[+quantize]) kernel speedup, every real GN shape (b{kd['batch']})")
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_kernel_gn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_kernel_gn.png")


def fig_kernel_fusions(kd):
    """This session's fusions: resize+quantize (both directions) and skip-concat."""
    if not kd:
        return
    up = [r for r in kd["resize"] if r["family"] == "resize_upsample" and r.get("twostep_int8_us")]
    dn = [r for r in kd["resize"] if r["family"] == "resize_avgpool" and r.get("twostep_int8_us")]
    cc = [r for r in kd["concat"] if r.get("torch_cat_us")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, rows, title in ((axes[0], up, "upsample(nearest,2x)+quantize"),
                            (axes[1], dn, "avg_pool(2x2)+quantize")):
        rows = sorted(rows, key=lambda r: (r["C"], r["H"]))
        x = np.arange(len(rows))
        s8 = [r["twostep_int8_us"] / r["fused_int8_us"] if r.get("fused_int8_us") else np.nan for r in rows]
        s4 = [r["twostep_int4_us"] / r["fused_int4_us"]
              if r.get("fused_int4_us") and r.get("twostep_int4_us") else np.nan for r in rows]
        ax.bar(x - 0.2, s8, 0.4, label="int8", color="#378ADD")
        ax.bar(x + 0.2, s4, 0.4, label="int4", color="#1D9E75")
        ax.axhline(1.0, ls="--", color="#555", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{r['C']}\n{r['H']}x{r['W']}" for r in rows], fontsize=7.5)
        ax.set_title(f"{title}\nFUSED vs resize-then-quantize", fontsize=10)
        ax.set_ylabel("speedup")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    cc = sorted(cc, key=lambda r: -(r["C1"] + r["C2"]))
    x = np.arange(len(cc))
    sp = [r["torch_cat_us"] / r["fused_us"] if r.get("fused_us") else np.nan for r in cc]
    ax.bar(x, sp, 0.55, color="#BA7517")
    ax.axhline(1.0, ls="--", color="#555", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['C1']}+{r['C2']}\n{r['H']}x{r['W']}" for r in cc], fontsize=7.5)
    ax.set_title("decoder skip-concat\ncat2_channels_last vs torch.cat", fontsize=10)
    ax.set_ylabel("speedup")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Fusion/specialization kernels added this session — speedup at every real shape (b{kd['batch']})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_kernel_fusions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_kernel_fusions.png")


def main_e2e():
    tree = load("profile_tree.json")
    kd = load("kernel_speedup_all_shapes.json")
    fig_e2e_speedup(tree)
    fig_layer_stack(tree)
    if tree:
        for m in ("int4_baseline", "int8_baseline", "int4_modiff", "fp16"):
            fig_profile_tree(tree, m)
    fig_kernel_conv(kd)
    fig_kernel_gn(kd)
    fig_kernel_fusions(kd)


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


def fig_intra_layer_split(lp, mode="int4_baseline", kind="resblock_plain", topn=10):
    """Inside each layer instance, which roles eat its GPU time -- in ABSOLUTE us.

    Y is real time (us per layer call), not a percentage, so bar HEIGHT is comparable
    across shapes: a 100%-normalized stack makes a 569 us layer look as tall as a 3377 us
    one, which hides where the time actually is.
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
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for role in order:
        vals = np.array([r["roles"].get(role, {}).get("us", 0.0) for r in rows])
        lt = next((k["layer_type"] for r in rows for k in r["kernels"] if k["role"] == role),
                  "Other / unclassified")
        ax.bar(x, vals, bottom=bottom, width=0.68,
               label=textwrap.shorten(role, 58, placeholder="..."),
               color=_role_color(role, lt))
        bottom += vals
    # Label each segment with its absolute us, but only where the segment is tall enough
    # to hold text (relative to the tallest bar, so labels stay legible at every scale).
    ymax = float(bottom.max()) if len(bottom) else 1.0
    bottom2 = np.zeros(len(rows))
    for role in order:
        vals = np.array([r["roles"].get(role, {}).get("us", 0.0) for r in rows])
        for i, (v, b) in enumerate(zip(vals, bottom2)):
            if v >= ymax * 0.035:
                ax.text(i, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom2 += vals
    for i, tot in enumerate(bottom):
        ax.text(i, tot + ymax * 0.012, f"{tot:.0f} us", ha="center",
                fontsize=8.5, fontweight="bold", color="#2C2C2A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{r['x_shape'][1]}\n{r['x_shape'][2]}x{r['x_shape'][3]}" for r in rows],
                       fontsize=8.5)
    ax.set_ylabel("us per layer call (absolute GPU time)")
    ax.set_ylim(0, ymax * 1.13)
    ax.set_title(f"Inside each layer — absolute time split by role  ({kind}, {mode}, b{lp['batch']})\n"
                 "bar height is real GPU time, so heights are comparable across shapes",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/fig_intra_layer_{kind}_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_intra_layer_{kind}_{mode}.png")


def main_layers():
    lp = load("layer_pipeline_bench.json")
    fig_layer_pipeline_speedup(lp)
    if lp:
        for mode in ("fp16", "int8_baseline", "int4_baseline", "int4_modiff"):
            for kind in ("resblock_plain", "resblock_updown", "attention"):
                fig_intra_layer_split(lp, mode, kind)


if __name__ == "__main__":
    main_e2e()
    main_layers()
