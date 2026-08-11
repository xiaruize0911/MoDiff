"""Per-layer and whole-model GPU profile, plotted.

METHOD, and why it is not either of the two that were rejected. `docs/component_attribution_2026-08-07`
threw out two per-component profiles:

  * `register_forward_pre_hook` MISSED 62 of the 70 MoDiff convs, because the ResBlock calls
    `forward_gn_fused_modiff` directly and never goes through `__call__`. Fixed here by timing the
    REAL dispatch targets -- the exact methods `fusion_audit.py` counts and has shown to be live --
    rather than the module boundary. The updown ResBlocks are timed at the module-level
    `_prequant_gn_resize_conv*` dispatchers for the same reason: they are not conv methods at all.
  * `ProfilerActivity.CPU` + summing `self_device_time_total` over every entry DOUBLE COUNTED (the
    record_function scopes carry the device time of the kernels inside them) and inflated the total
    by 2.2x. Fixed here by using CUDA events, not the torch profiler, and by timing only LEAF
    dispatch targets -- `forward` is deliberately not timed, because it wraps the others.

The instrument reports its own error rather than asking to be trusted: every mode is also timed
WITHOUT instrumentation, in the same process on the same model, and the ratio
sum(per-layer) / wall is printed. The 2026-08-07 report's trace-based equivalent ran 0.948-0.977; a
number far from that band means the event timing is perturbing what it measures, and the per-layer
split should be read as shares rather than as absolute ms.

Outputs plots/{layers,kinds,model}.png and data/profile_layers.json.

Run: python integration/tests/profile_layers_and_model.py [--batch 128] [--steps 20]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

#: (label, mode, MODIFF_LINEAR, act_bits, K). fp16 has no quantized layers to attribute, so it
#: appears in the whole-model plot only.
CONFIGS = [("W8A8 PTQ", "int8_baseline", "0", 8, 4),
           ("W8A8 conv-only", "int8", "0", 8, 4),
           ("W8A8 conv+proj", "int8", "1", 8, 4),
           ("W8A4 conv+proj", "int8", "1", 4, 4),
           ("W4A4 conv+proj", "int4", "1", 8, 4)]
WALL_ONLY = [("fp16", "fp16", "0", 8, 4)]

#: Leaf dispatch targets, per layer kind. `forward` is absent ON PURPOSE: it wraps the others, and
#: timing both is the double count that invalidated the earlier profile.
CONV_METHODS = ["forward_gn_fused_modiff", "_forward_modulated",
                "_forward_modulated_static_fused_silu", "forward_modiff_fused_silu_residual",
                "_forward_first_step", "_forward_standard",
                "forward_from_int8", "forward_from_int8_dual", "forward_to_int8",
                "forward_from_int4", "forward_from_int4_dual", "forward_to_int4"]


class EventTimer:
    """Accumulates GPU time per key with CUDA events, no synchronisation in the hot path.

    Events are recorded around each call and read back once at the end. Recording is ~microseconds
    and does not serialise the stream, which is what keeps the perturbation small enough for the
    wall-clock ratio below to stay near 1.
    """

    def __init__(self):
        self.pairs = defaultdict(list)

    def wrap(self, obj, name, key):
        fn = getattr(obj, name, None)
        if fn is None:
            return False
        pairs = self.pairs[key]

        def inner(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            pairs.append((s, e))
            return out
        setattr(obj, name, inner)
        return True

    def totals(self):
        torch.cuda.synchronize()
        return {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.pairs.items() if v}

    def counts(self):
        return {k: len(v) for k, v in self.pairs.items() if v}


def profile(label, mode, lin, act_bits, k, steps):
    os.environ["MODIFF_LINEAR"] = lin
    os.environ["MODIFF_ACT_BITS"] = str(act_bits)
    os.environ["MODIFF_DELTA_REFRESH"] = str(k)
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ.pop("MODIFF_DELTA_CLIP", None)
    H.STEPS = steps
    r, m, s = H.build(mode, None if mode == "fp16" else H.CALIB["int4" if "int4" in mode else "int8"],
                      "dynamic" if mode not in ("fp16", "int8_baseline", "int4_baseline") else "static")
    unet = m.model.diffusion_model

    # --- uninstrumented wall clock FIRST, on the same model: the denominator of the error check ---
    H.SEED = 1234
    H.latent(r, m, s)                                    # warm-up, discarded
    _, wall_ms = H.latent(r, m, s)

    if mode == "fp16":
        del r, m, s
        torch.cuda.empty_cache()
        return {"config": label, "wall_ms_per_step": wall_ms, "layers": {}, "kinds": {}}

    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    import integration.fused_ops.fused_resblock as FR

    t = EventTimer()
    convs = list({id(c): c for c in unet.modules()
                  if isinstance(c, (OptimizedInt8Conv2d, OptimizedInt4Conv2d))}.values())
    # Depth order: unet.modules() walks in definition order, which for this UNet is input blocks ->
    # middle -> output blocks. That is the order the x-axis of the per-layer plot means.
    shapes = {}
    for i, c in enumerate(convs):
        # in_channels does not exist on these wrappers -- the first attempt used it and every bar
        # came out the default grey. Read the width off the quantized weight instead.
        w = getattr(c, "weight_int8", None)
        if w is None:
            w = getattr(c, "weight_packed", None)
        shapes[i] = int(w.shape[1]) if w is not None and w.dim() >= 2 else None
        for meth in CONV_METHODS:
            t.wrap(c, meth, f"conv{i:03d}")

    # updown dispatchers are module globals, resolved at call time -- see fusion_audit.py
    for name in ("_prequant_gn_resize_conv_modiff", "_prequant_gn_resize_conv"):
        t.wrap(FR, name, "updown")

    attn = [b for b in unet.modules() if type(b).__name__ == "QuantizedStandardAttentionBlock"]
    for i, b in enumerate(attn):
        t.wrap(b, "_forward_routes", f"attn{i:02d}")
    try:
        from integration.kernels.wxax_linear import QuantLinearWxAx
        wx = list({id(x): x for x in unet.modules() if isinstance(x, QuantLinearWxAx)}.values())
    except Exception:
        wx = []
    for i, x in enumerate(wx):
        t.wrap(x, "forward", f"proj{i:03d}")

    H.SEED = 1234
    _, instr_ms = H.latent(r, m, s)
    tot = t.totals()
    cnt = t.counts()
    del r, m, s
    torch.cuda.empty_cache()

    per_step = {kk: v / steps for kk, v in tot.items()}
    # attn keys include their two projections (the projection forward runs inside _forward_routes),
    # so summing everything would double count. Report attn NET of its projections.
    proj_total = sum(v for kk, v in per_step.items() if kk.startswith("proj"))
    attn_gross = sum(v for kk, v in per_step.items() if kk.startswith("attn"))
    kinds = {"conv": sum(v for kk, v in per_step.items() if kk.startswith("conv")),
             "updown": per_step.get("updown", 0.0),
             "attn (score path)": max(attn_gross - proj_total, 0.0),
             "proj (42 linears)": proj_total}
    return {"config": label, "mode": mode, "modiff_linear": lin, "act_bits": act_bits, "K": k,
            "wall_ms_per_step": wall_ms, "instrumented_ms_per_step": instr_ms,
            "layers": per_step, "calls": cnt, "conv_in_channels": shapes, "kinds": kinds,
            "sum_over_wall": sum(kinds.values()) / wall_ms if wall_ms else 0.0,
            "n_convs": len(convs), "n_attn": len(attn), "n_proj": len(wx)}


# ----------------------------------------------------------------------------- plots
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cc7"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
    })
    return plt


def plot_layers(rows, out):
    """Per conv layer, in UNet depth order, one panel per mode."""
    plt = setup_mpl()
    rows = [r for r in rows if r.get("layers")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11.5, 2.35 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        keys = sorted(k for k in r["layers"] if k.startswith("conv"))
        vals = [r["layers"][k] for k in keys]
        ch = [r["conv_in_channels"].get(k[4:].lstrip("0") or "0") for k in keys]
        # Colour by input width: the UNet's cost concentrates in the wide/high-resolution blocks,
        # and that is the thing a reader wants to see without cross-referencing a table.
        uniq = sorted({c for c in ch if c})
        cmap = {c: [BLUE, AQUA, ORANGE, PLUM, INK3][i % 5] for i, c in enumerate(uniq)}
        ax.bar(range(len(vals)), vals, color=[cmap.get(c, INK3) for c in ch], width=0.86)
        ax.set_title(f"{r['config']}   conv layers, {sum(vals):.1f} ms/step total", loc="left")
        ax.set_ylabel("ms/step")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("quantized conv layer, in UNet depth order (input blocks -> middle -> output)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_kinds(rows, out):
    """Stacked ms/step by layer kind, one bar per mode."""
    plt = setup_mpl()
    rows = [r for r in rows if r.get("kinds")]
    order = ["conv", "updown", "attn (score path)", "proj (42 linears)"]
    colors = {"conv": BLUE, "updown": AQUA, "attn (score path)": ORANGE,
              "proj (42 linears)": PLUM}
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    labels = [r["config"] for r in rows]
    bottom = [0.0] * len(rows)
    for kind in order:
        vals = [r["kinds"].get(kind, 0.0) for r in rows]
        ax.bar(labels, vals, bottom=bottom, label=kind, color=colors[kind], width=0.62)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, r in enumerate(rows):
        ax.text(i, bottom[i] * 1.01, f"{bottom[i]:.0f}", ha="center", va="bottom",
                color=INK2, fontsize=9)
    ax.set_ylabel("ms/step, summed over layers of that kind")
    ax.set_title("Where the step goes, by layer kind  (CUDA events on the live dispatch targets)",
                 loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


CANON = "docs/updown_refresh_fusion_2026-08-10/data/differential_timing_canonical.json"
#: profile label -> differential arm. The differential harness is the authority for ms/step: it is
#: profiler-free, 200 steps, 5 repeats, and it does not time the reset walk or the scheduler.
CANON_ARM = {"W8A8 PTQ": "int8_ptq", "W8A8 conv-only": "modiff_conv_k4",
             "W8A8 conv+proj": "modiff_full_k4"}


def canonical():
    try:
        with open(CANON) as f:
            d = json.load(f)
        return {k: a["stats"]["median"] / 1e3 / d["steps"] for k, a in d["arms"].items()}
    except Exception:
        return {}


def plot_model(rows, out):
    """Whole-model ms/step. Uses the canonical differential arms where one exists, this harness's
    own uninstrumented wall clock otherwise, and says which is which."""
    plt = setup_mpl()
    canon = canonical()
    for r in rows:
        arm = CANON_ARM.get(r["config"])
        r["ms_source"] = "differential (200 steps)" if arm in canon else "this harness"
        r["ms_per_step"] = canon.get(arm, r["wall_ms_per_step"])
    fp16 = next((r["ms_per_step"] for r in rows if r["config"] == "fp16"), None)
    rows = [r for r in rows if r["config"] != "fp16"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    vals = [r["ms_per_step"] for r in rows]
    labels = [r["config"] for r in rows]
    ax.bar(labels, vals, color=BLUE, width=0.58)
    for i, v in enumerate(vals):
        tag = f"{v:.1f}" + (f"\n{fp16 / v:.2f}x fp16" if fp16 else "")
        ax.text(i, v * 1.01, tag, ha="center", va="bottom", color=INK2, fontsize=9)
    if fp16:
        ax.axhline(fp16, color=ORANGE, linewidth=1.6, linestyle="--")
        ax.text(len(vals) - 0.4, fp16, f"  fp16 {fp16:.1f}", color=ORANGE, va="bottom", fontsize=9)
    ax.set_ylabel("ms/step (wall clock, NO instrumentation)")
    ax.set_title("Whole model, profiler-free", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals + ([fp16] if fp16 else [])) * 1.22)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--outdir", default="docs/updown_refresh_fusion_2026-08-10")
    args = ap.parse_args()
    H.BATCH = args.batch

    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "data"), exist_ok=True)

    print(f"batch {args.batch}, DDIM {args.steps}, {torch.cuda.get_device_name(0)}\n", flush=True)
    rows = []
    for label, mode, lin, ab, k in WALL_ONLY + CONFIGS:
        print(f"=== {label}", flush=True)
        r = profile(label, mode, lin, ab, k, args.steps)
        rows.append(r)
        if r.get("kinds"):
            print(f"    wall {r['wall_ms_per_step']:.1f} ms/step, instrumented "
                  f"{r['instrumented_ms_per_step']:.1f}, sum(per-layer)/wall "
                  f"{r['sum_over_wall']:.3f}", flush=True)
            for kk, v in r["kinds"].items():
                print(f"      {kk:<20} {v:8.2f} ms/step", flush=True)
        else:
            print(f"    wall {r['wall_ms_per_step']:.1f} ms/step", flush=True)
        with open(os.path.join(args.outdir, "data/profile_layers.json"), "w") as f:
            json.dump(rows, f, indent=2)

    p = os.path.join(args.outdir, "plots")
    made = [plot_layers(rows, os.path.join(p, "profile_layers.png")),
            plot_kinds(rows, os.path.join(p, "profile_kinds.png")),
            plot_model(rows, os.path.join(p, "profile_model.png"))]
    print()
    for m in made:
        print(f"  {m}", flush=True)
    ratios = [r["sum_over_wall"] for r in rows if r.get("sum_over_wall")]
    if ratios:
        print(f"\nsum(per-layer)/wall across modes: {min(ratios):.3f}-{max(ratios):.3f}. "
              f"Far from 1.0 means the event timing perturbs what it measures; read the split as "
              f"shares, not absolutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
