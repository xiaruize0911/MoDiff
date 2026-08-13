"""Speed against fidelity for all six arms, with the dominated one identified.

THE POINT OF THE CHART. int4_linmodiff (MoDiff extended to the 42 attention projections) is better on
fidelity than the shipped int4 and worse on latency, which sounds like an ordinary trade-off until it
is plotted against every other arm: W8A8 MoDiff is BOTH faster and more accurate, so int4_linmodiff is
strictly dominated and no objective selects it. That is a statement about geometry, so it gets a chart
rather than a sentence -- and the frontier is COMPUTED here, not drawn by hand, so the claim is
checked by the same code that renders it.

PALETTE. Two slots from the data-viz reference palette in its documented fixed order, imported from
docs/state_report_2026-08-12/scripts/make_plots.py rather than re-typed, so the report has one source
of truth for them. node is not installed in this container, so scripts/validate_palette.js cannot be
run; as in make_plots.py the response is to stay strictly inside what the reference already states as
validated rather than to eyeball anything. A scatter is on the ALL-PAIRS pairlist, where at most three
slots validate; this uses two (slots 0 and 1), and every point carries a direct label so identity is
never colour-alone.

DATA PROVENANCE, mixed on purpose and stated rather than hidden:
  latency   all six arms from ONE run of e2e_three_mode_bench on the shipped calibration
            (data/e2e_linmodiff.json, batch 128, DDIM 200, 3 repeats, CV <= 0.23%).
  fidelity  int4 / int4_linmodiff / int4_baseline from data/linear_modiff_w4a4_ab.json (pinned fp16
            references, self-check passed at +0.1% vs committed). int8 / int8_baseline from
            docs/static_qdiff_2026-08-12, which was re-run unmodified on 2026-08-13 and reproduced
            per-seed exactly, so the two sets are comparable.

Run: python docs/attn_modiff_2026-08-13/scripts/plot_frontier.py    # no GPU
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs/state_report_2026-08-12/scripts"))

import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import Rectangle                                    # noqa: E402
from make_plots import SERIES, SURFACE, INK, INK2, GRID                     # noqa: E402

D = "docs/attn_modiff_2026-08-13"
OUT = f"{D}/plots/01_frontier.png"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def dominated_by(pt, others):
    """Arms that beat `pt` on BOTH axes (lower ms and lower relL2). Ties do not dominate."""
    return [o for o in others
            if o["ms"] <= pt["ms"] and o["rel"] <= pt["rel"] and o is not pt
            and (o["ms"] < pt["ms"] or o["rel"] < pt["rel"])]


def main():
    e2e = json.load(open(f"{D}/data/e2e_linmodiff.json"))["modes"]
    ab = json.load(open(f"{D}/data/linear_modiff_w4a4_ab.json"))["results"]

    def ab_mean(frag):
        return next(v["mean"] for k, v in ab.items() if k.startswith(frag))

    #: (label, e2e key, relL2). fp16 is the reference, so its relL2 is 0 by definition.
    ARMS = [("fp16", "fp16", 0.0),
            ("W8A8 PTQ", "int8_baseline", 0.1138),
            ("W8A8 MoDiff", "int8", 0.0605),
            ("W4A4 PTQ", "int4_baseline", ab_mean("int4_baseline")),
            ("W4A4 MoDiff\n(shipped)", "int4", ab_mean("int4  ")),
            ("W4A4 MoDiff\n+ attn proj", "int4_linmodiff", ab_mean("int4_linmodiff"))]
    pts = [{"lab": l, "ms": e2e[k]["per_step_ms"], "rel": r} for l, k, r in ARMS]

    dom = {p["lab"]: dominated_by(p, pts) for p in pts}
    bad = [p for p in pts if dom[p["lab"]]]
    print("frontier (dominated by nothing):")
    for p in pts:
        if not dom[p["lab"]]:
            print(f"  {p['lab'][:24]:26s} {p['ms']:7.2f} ms  {p['rel']:.4f}")
    print("dominated:")
    for p in bad:
        print(f"  {p['lab'][:24]:26s} {p['ms']:7.2f} ms  {p['rel']:.4f}  <- beaten on BOTH axes by "
              f"{', '.join(d['lab'].replace(chr(10), ' ') for d in dom[p['lab']])}")
    if len(bad) != 1 or "attn proj" not in bad[0]["lab"]:
        print("\nUNEXPECTED: the chart's premise is that exactly one arm (+ attn proj) is dominated. "
              "It is not, so the annotation below would be wrong -- refusing to draw it.")
        return 1

    victim = bad[0]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    # The dominating region: everything faster AND more accurate than the dominated arm. Drawn as a
    # recessive neutral fill because it is context, not data -- the message is which points fall in it.
    ax.add_patch(Rectangle((0, 0), victim["ms"], victim["rel"], facecolor=INK2, alpha=0.055,
                           edgecolor="none", zorder=0))
    ax.text(victim["ms"] - 2.5, victim["rel"] - 0.022,
            "faster AND more accurate\nthan +attn proj", ha="right", va="top", fontsize=8.5,
            color=INK2, style="italic")

    front = sorted((p for p in pts if not dom[p["lab"]]), key=lambda p: p["ms"])
    ax.plot([p["ms"] for p in front], [p["rel"] for p in front], linewidth=2.0,
            color=SERIES[0], alpha=0.35, zorder=1, label="Pareto frontier")
    ax.scatter([p["ms"] for p in front], [p["rel"] for p in front], s=90, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2.0, zorder=3, label="on the frontier")
    ax.scatter([victim["ms"]], [victim["rel"]], s=130, color=SERIES[1], marker="X",
               edgecolor=SURFACE, linewidth=2.0, zorder=4, label="dominated")

    for p in pts:
        off = (10, -16) if p is victim else (10, 6)
        ax.annotate(f"{p['lab']}\n{p['rel']:.3f} @ {p['ms']:.0f} ms", (p["ms"], p["rel"]),
                    textcoords="offset points", xytext=off, fontsize=8.5, color=INK2)

    ax.set_xlabel("ms per denoising step   (batch 128, DDIM 200, A40 — lower is faster)")
    ax.set_ylabel("latent relL2 vs fp16   (lower is closer)")
    ax.set_xlim(45, 128)
    ax.set_ylim(-0.035, 0.56)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Extending MoDiff to the attention projections lands off the frontier",
                 loc="left", pad=10)
    os.makedirs(f"{D}/plots", exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
