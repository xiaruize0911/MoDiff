"""Figures for the blockwise report. Every figure is regenerated from data/*.json.

Run: python docs/blockwise_2026-08-31/scripts/make_plots.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from integration.utils.preflight import preflight  # noqa: E402

preflight("matplotlib", what="make_plots.py")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = "docs/blockwise_2026-08-31/data"
P = "docs/blockwise_2026-08-31/plots"
INK = "#0b0b0b"
C8, C4 = "#1f6feb", "#d1242f"
CG = "#8250df"


def _load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _cost():
    """Full 20-shape UNet cost data if present, else the 5-shape subset."""
    return _load("blockwise_cost_unet20.json") or _load("blockwise_cost.json")


def _style(ax, xlab, ylab, title):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_weights(d):
    """Weight-only reconstruction error vs block size."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, bits, col in ((axes[0], 8, C8), (axes[1], 4, C4)):
        arms = [a for a in d["arms"] if a["bits"] == bits]
        blocks = d["blocks"]
        for rule, ls, mk in (("absmax", "-", "o"), ("mse", "--", "s")):
            ys = [next(a["median"] for a in arms if a["block"] == g and a["rule"] == rule)
                  for g in blocks]
            ws = [next(a["worst"] for a in arms if a["block"] == g and a["rule"] == rule)
                  for g in blocks]
            ax.plot(blocks, ys, ls, marker=mk, color=col, label=f"block {rule} (median)")
            ax.plot(blocks, ws, ls, marker=mk, color=col, alpha=0.35,
                    label=f"block {rule} (worst)")
        for rule, c in (("absmax", "#57606a"), ("mse", "#0b0b0b")):
            v = next(a["median"] for a in arms if a["block"] is None and a["rule"] == rule)
            ax.axhline(v, color=c, linewidth=1.0, linestyle=":",
                       label=f"per-channel {rule} (median)")
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        _style(ax, "block size G (elements, flat K axis)", "rel Frobenius error",
               f"W{bits} weight reconstruction")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "fig1_weight_granularity.png"), dpi=150)
    plt.close(fig)


def _series(agg, bits, refresh, groups):
    """(G, mean, halfspread) for the all-blockwise arms at one (bits, refresh)."""
    rt = "" if refresh == 4 else f" r={refresh}"
    out = []
    for g in groups:
        k = f"W{bits}A{bits} bw G={g}{rt}"
        if k in agg:
            r = agg[k]
            out.append((g, r["relL2_mean"], (r["relL2_max"] - r["relL2_min"]) / 2))
    return out


def fig_e2e(d):
    """End-to-end relL2 vs G, both bit-widths, both refresh cadences, with seed spread."""
    agg, groups = d["agg"], d["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, bits in ((axes[0], 8), (axes[1], 4)):
        for refresh, col, mk in ((4, CG, "o"), (1, C8 if bits == 8 else C4, "s")):
            s = _series(agg, bits, refresh, groups)
            if not s:
                continue
            xs = [a for a, _, _ in s]
            ys = [b for _, b, _ in s]
            es = [c for _, _, c in s]
            ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, capsize=3, linewidth=1.4,
                        label=f"blockwise, refresh={refresh}")
            rt = "" if refresh == 4 else f" r={refresh}"
            sk = f"W{bits}A{bits} shipped{rt}"
            if sk in agg:
                ax.axhline(agg[sk]["relL2_mean"], color=col, linestyle=":", linewidth=1.2,
                           label=f"shipped (per-tensor), refresh={refresh}")
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
        _style(ax, "block size G (input channels)", "relL2 vs fp16 latent",
               f"W{bits}A{bits} end-to-end")
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "fig2_e2e_relL2.png"), dpi=150)
    plt.close(fig)


def fig_clip(d):
    """Clip fraction vs G -- the mechanism behind the refresh=4 penalty."""
    agg, groups = d["agg"], d["groups"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for bits, col in ((8, C8), (4, C4)):
        ys = []
        for g in groups:
            k = f"W{bits}A{bits} bw G={g}"
            ys.append(agg[k]["clip_frac"] * 100 if k in agg else None)
        ax.plot(groups, ys, marker="o", color=col, label=f"W{bits}A{bits}, refresh=4")
        sk = f"W{bits}A{bits} shipped"
        if sk in agg:
            ax.axhline(agg[sk]["clip_frac"] * 100, color=col, linestyle=":", linewidth=1.2,
                       label=f"W{bits}A{bits} shipped (per-tensor)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.invert_xaxis()
    _style(ax, "block size G (input channels)", "clipped delta codes (%)",
           "Held scale: finer blocks clip more")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "fig3_clip_frac.png"), dpi=150)
    plt.close(fig)


def fig_cost(d):
    """Relative speedup on conv, per shape and freq-weighted, plus where it goes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # left: every shape's speedup vs the fused per-tensor call, over G. The point is
    # that the curves cluster by G rather than spreading by block count.
    ax = axes[0]
    for r in d["shapes"]:
        gs = sorted((int(g) for g in r["splits"]), reverse=True)
        if not gs:
            continue
        ax.plot(gs, [r["splits"][str(g)]["vs_fused"] for g in gs], marker="o", ms=3,
                color="#57606a", alpha=0.45, linewidth=0.9)
    w = d["freq_weighted_ms"]
    ks = sorted((k for k in w if k != "fused"), key=lambda k: -int(k.split("=")[1]))
    gs = [int(k.split("=")[1]) for k in ks]
    ax.plot(gs, [w["fused"] / w[k] for k in ks], marker="o", color=CG, linewidth=2.2,
            label="freq-weighted, all 20 shapes")
    ax.plot([], [], color="#57606a", alpha=0.45, linewidth=0.9, label="individual shapes")
    ax.axhline(1.0, color=INK, linewidth=1.0, linestyle=":", label="fused per-tensor = 1.0x")
    for g, k in zip(gs, ks):
        ax.annotate(f"{w['fused'] / w[k]:.3f}x", (g, w["fused"] / w[k]), fontsize=7.5,
                    color=CG, textcoords="offset points", xytext=(5, 5))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.invert_xaxis()
    _style(ax, "block size G (input channels)", "speedup vs fused per-tensor",
           "Conv speedup with everything blockwise")
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # right: where a single split call's time goes
    a = d.get("attribution")
    if a:
        ax = axes[1]
        bars = [("1/nb of fused\n(free split)", a["ideal_per_call_us"], "#8c959f"),
                ("epilogue only\n(o_hat RMW, no GEMM)", a["epilogue_only_us"], "#bf8700"),
                ("actual per call\n(Cin=G conv)", a["standalone_cin_g_us"], C4)]
        ax.bar([b[0] for b in bars], [b[1] for b in bars],
               color=[b[2] for b in bars], width=0.6)
        for i, b in enumerate(bars):
            ax.text(i, b[1] + 2, f"{b[1]:.1f} us", ha="center", fontsize=8.5, color=INK)
        ax.set_ylim(0, a["standalone_cin_g_us"] * 1.22)
        _style(ax, "", "µs per split call",
               f"{a['shape']}, G={a['G']}, {a['n_blocks']} blocks: "
               f"{a['epilogue_share_of_per_call'] * 100:.0f}% epilogue, rest is the thin GEMM")
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "fig4_cost.png"), dpi=150)
    plt.close(fig)


def fig_attrib(_ignored):
    """Attribution: weight granularity matters, activation granularity does not.

    Reads the two dedicated sweeps (blockwise_wonly / blockwise_actonly) plus the
    combined arm, so each curve varies exactly one tensor's granularity.
    """
    w = _load("blockwise_wonly.json")
    ac = _load("blockwise_actonly.json")
    both = _load("blockwise_e2e_mse.json") or _load("blockwise_e2e.json")
    if not (w and ac):
        print("skip fig_attrib: attribution data missing")
        return
    groups = w["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, bits in ((axes[0], 8), (axes[1], 4)):
        def draw(d, key, lab, col, mk, ls="-"):
            if not d:
                return
            ys, es = [], []
            for g in groups:
                k = f"W{bits}A{bits} {key} G={g}"
                if k not in d["agg"]:
                    return
                r = d["agg"][k]
                ys.append(r["relL2_mean"])
                es.append((r["relL2_max"] - r["relL2_min"]) / 2)
            ax.errorbar(groups, ys, yerr=es, marker=mk, linestyle=ls, color=col,
                        capsize=3, linewidth=1.4, label=lab)

        draw(w, "W-only", "weights blockwise only", "#1a7f37", "^")
        draw(ac, "A-only", "activations blockwise only", "#bf3989", "v")
        draw(both, "bw", "both blockwise", CG, "o")
        sk = f"W{bits}A{bits} shipped"
        if sk in w["agg"]:
            ax.axhline(w["agg"][sk]["relL2_mean"], color=INK, linestyle=":", linewidth=1.2,
                       label="shipped (per-channel W, per-tensor A)")
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
        _style(ax, "block size G (input channels)", "relL2 vs fp16 latent",
               f"W{bits}A{bits}, refresh=4: which tensor pays?")
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "fig5_attribution.png"), dpi=150)
    plt.close(fig)


def fig_tradeoff(_ignored):
    """Quality against measured cost, so the block-size choice is visible in one place."""
    e = _load("blockwise_e2e_mse.json") or _load("blockwise_e2e.json")
    c = _cost()
    if not (e and c):
        print("skip fig_tradeoff: data missing")
        return
    fw = c["freq_weighted_ms"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for bits, refresh, col, mk in ((8, 1, C8, "s"), (4, 4, C4, "o")):
        xs, ys, labs = [], [], []
        rt = "" if refresh == 4 else f" r={refresh}"
        sk = f"W{bits}A{bits} shipped{rt}"
        if sk in e["agg"]:
            xs.append(fw["fused"])
            ys.append(e["agg"][sk]["relL2_mean"])
            labs.append("per-tensor")
        for g in e["groups"]:
            k = f"W{bits}A{bits} bw G={g}{rt}"
            ck = f"G={g}"
            if k in e["agg"] and ck in fw:
                xs.append(fw[ck])
                ys.append(e["agg"][k]["relL2_mean"])
                labs.append(f"G={g}")
        ax.plot(xs, ys, marker=mk, color=col, linewidth=1.4,
                label=f"W{bits}A{bits}, refresh={refresh}")
        for x, y, l in zip(xs, ys, labs):
            ax.annotate(l, (x, y), fontsize=6.5, color=col,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "conv-path cost, freq-weighted (ms/step, log)", "relL2 vs fp16 (log)",
           "Blockwise buys quality at a steep price")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(P, "fig6_tradeoff.png"), dpi=150)
    plt.close(fig)


def main() -> int:
    os.makedirs(P, exist_ok=True)
    made = []
    for name, fn in (("weight_granularity.json", fig_weights),
                     ("blockwise_e2e.json", fig_e2e),
                     ("blockwise_e2e.json", fig_clip),
                     ("blockwise_cost.json", fig_cost),
                     ("blockwise_wonly.json", fig_attrib),
                     ("blockwise_cost.json", fig_tradeoff)):
        d = _cost() if fn in (fig_cost,) else _load(name)
        if d is None:
            print(f"skip {fn.__name__}: {name} not present")
            continue
        fn(d)
        made.append(fn.__name__)
    print("made: " + ", ".join(made))
    for f in sorted(os.listdir(P)):
        print(f"  {P}/{f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
