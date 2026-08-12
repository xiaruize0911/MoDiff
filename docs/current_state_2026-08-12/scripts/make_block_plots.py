"""Per-block-type figures. Offline: reads data/profile_blocks.json, no GPU.

One figure per block type, as two panels:

  LEFT   every instance of that type, in UNet execution order, for the current configuration
  RIGHT  that type's total in each of the eight configurations

The right panel is what makes the figure worth having per type rather than one chart of everything:
it says how that specific block type responds to the configuration space, which differs sharply
between types (the hd=48 attention tiers move with the qkv fusion, the ResBlocks do not).

Palette copied from make_plots.py in this directory.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "docs/current_state_2026-08-12/data/profile_blocks.json")
PLOTS = os.path.join(ROOT, "docs/current_state_2026-08-12/plots")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, PLUM, ROSE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cc7", "#d64570"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})

CURRENT = "W8A8 conv+proj +projK4 +routeB"
#: short labels for the right-hand per-configuration panel, in ladder order
ORDER = [("W8A8 PTQ", "W8A8 PTQ"), ("W8A8 conv-only", "conv only"),
         ("W8A8 conv+proj", "conv+proj"), ("W8A8 conv+proj +projK4", "+ refresh K=4"),
         (CURRENT, "+ int8 qkv"), ("W8A4 conv+proj", "W8A4"), ("W4A4 conv+proj", "W4A4"),
         ("fp16", "fp16")]

#: (filename stem, title, colour, predicate on a block's meta dict)
GROUPS = [
    ("10_resblock", "ResBlock -- ordinary (27)", BLUE,
     lambda m: m.get("type") == "resblock"),
    ("11_resblock_down", "ResBlock -- downsampling (4)", AQUA,
     lambda m: m.get("type") == "resblock_down"),
    ("12_resblock_up", "ResBlock -- upsampling (4)", PLUM,
     lambda m: m.get("type") == "resblock_up"),
    ("13_attn_hd24_T1024", "Attention -- hd 24, T 1024 (5)", ROSE,
     lambda m: m.get("tier") == "hd24 T1024"),
    ("14_attn_hd48_T256", "Attention -- hd 48, T 256 (5)", ORANGE,
     lambda m: m.get("tier") == "hd48 T256"),
    ("15_attn_hd48_T64", "Attention -- hd 48, T 64 (5)", "#c08a2e",
     lambda m: m.get("tier") == "hd48 T64"),
    ("16_attn_hd96", "Attention -- hd 96 (6: five at T 16, the middle block at T 4)", INK3,
     lambda m: str(m.get("tier", "")).startswith("hd96")),
]


def load():
    if not os.path.exists(SRC):
        return {}
    return {r["config"]: r for r in json.load(open(SRC))}


def members(row, pred):
    """Keys of this row matching the predicate, in UNet execution order (the key encodes it)."""
    return sorted((k for k, m in row["meta"].items() if pred(m)), key=lambda k: int(k[2:]))


def label_of(row, k):
    m = row["meta"][k]
    w, c = m.get("where", "?"), m.get("container", -1)
    return f"{w}{c}"


def one_group(stem, title, color, pred, rows):
    cur = rows.get(CURRENT)
    if cur is None:
        return False
    keys = members(cur, pred)
    if not keys:
        return False
    vals = [cur["blocks"][k] for k in keys]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.8, 4.4),
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    axl.bar(range(len(keys)), vals, color=color, width=0.62)
    for i, v in enumerate(vals):
        axl.text(i, v * 1.012, f"{v:.2f}", ha="center", va="bottom", color=INK2,
                 fontsize=7.5 if len(keys) > 14 else 8.5)
    axl.set_xticks(range(len(keys)))
    axl.set_xticklabels([label_of(cur, k) for k in keys],
                        fontsize=6.5 if len(keys) > 14 else 8.5,
                        rotation=90 if len(keys) > 14 else 0)
    axl.set_xlabel("UNet position (in = input block, mid = middle, out = output block)")
    axl.set_ylabel("ms/step")
    axl.set_title(f"{title} -- {sum(vals):.2f} ms/step", loc="left")
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    axl.set_ylim(0, max(vals) * 1.20)

    labs, tots, cols = [], [], []
    for cfg, short in ORDER:
        r = rows.get(cfg)
        if r is None or not r.get("meta"):
            continue
        ks = members(r, pred)
        if not ks:
            continue
        labs.append(short)
        tots.append(sum(r["blocks"][k] for k in ks))
        cols.append(color if cfg == CURRENT else INK3)
    axr.barh(range(len(labs)), tots, color=cols, height=0.6)
    for i, v in enumerate(tots):
        axr.text(v + max(tots) * 0.015, i, f"{v:.2f}", va="center", color=INK2, fontsize=8.5)
    axr.set_yticks(range(len(labs)))
    axr.set_yticklabels(labs, fontsize=8.5)
    axr.invert_yaxis()
    axr.set_xlabel("ms/step, summed over the type")
    axr.set_title("Same type, every configuration", loc="left")
    axr.grid(axis="x", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    axr.set_xlim(0, max(tots) * 1.22)
    fig.suptitle(f"Batch 128, A40 -- block-level CUDA events, coverage {cur['sum_over_wall']:.3f}. "
                 f"Left: {CURRENT}.", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(PLOTS, stem + ".png"), dpi=170)
    plt.close(fig)
    return True


def plot_summary(rows):
    """Every block type at once, per configuration -- the map the per-type figures zoom into."""
    types = [("resblock", "ResBlock x27", BLUE), ("resblock_down", "ResBlock down x4", AQUA),
             ("resblock_up", "ResBlock up x4", PLUM), ("attention", "Attention x21", ROSE),
             ("conv_in", "conv_in", "#c08a2e"), ("out_tail", "out tail", ORANGE),
             ("time_embed", "time_embed", INK3)]
    cfgs = [(c, s) for c, s in ORDER if c in rows and rows[c].get("meta")]
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    bottom = [0.0] * len(cfgs)
    for tname, tlabel, col in types:
        vals = []
        for c, _ in cfgs:
            r = rows[c]
            vals.append(sum(v for k, v in r["blocks"].items()
                            if r["meta"].get(k, {}).get("type") == tname))
        ax.bar(range(len(cfgs)), vals, bottom=bottom, color=col, width=0.6, label=tlabel)
        for i, v in enumerate(vals):
            if v > 3.0:
                ax.text(i, bottom[i] + v / 2, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=8.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, (c, _) in enumerate(cfgs):
        w = rows[c]["wall_ms_per_step"]
        ax.plot([i - 0.3, i + 0.3], [w, w], color=INK, linewidth=1.8)
        ax.text(i, w * 1.01, f"wall {w:.1f}", ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([s for _, s in cfgs], fontsize=9)
    ax.set_ylabel("ms/step")
    ax.set_title("Every UNet block type, stacked -- black rule is the uninstrumented wall clock, "
                 "so the gap above the stack is the instrument's residual", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, ncol=4, loc="upper left")
    ax.set_ylim(0, max(rows[c]["wall_ms_per_step"] for c, _ in cfgs) * 1.22)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "09_block_types.png"), dpi=170)
    plt.close(fig)
    return True


def plot_head_tail(rows):
    """The three singletons. One bar each is not a figure, so they share one."""
    cur = rows.get(CURRENT)
    if cur is None:
        return False
    keys = [k for k, m in cur["meta"].items()
            if m.get("type") in ("conv_in", "time_embed", "out_tail")]
    if not keys:
        return False
    cfgs = [(c, s) for c, s in ORDER if c in rows and rows[c].get("meta")]
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    w = 0.26
    for j, (k, col) in enumerate(zip(sorted(keys), ("#c08a2e", INK3, ORANGE))):
        vals = [rows[c]["blocks"].get(k, 0.0) for c, _ in cfgs]
        xs = [i + (j - 1) * w for i in range(len(cfgs))]
        ax.bar(xs, vals, width=w * 0.9, color=col, label=cur["meta"][k]["type"])
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.02, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([s for _, s in cfgs], fontsize=9)
    ax.set_ylabel("ms/step")
    ax.set_title("Head and tail: the three unquantized singletons -- 3.3 of ~100 ms/step together, "
                 "and flat across every configuration", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "17_head_and_tail.png"), dpi=170)
    plt.close(fig)
    return True


def main():
    rows = load()
    if not rows:
        print(f"NOTE: {SRC} absent -- run integration/tests/profile_blocks.py first")
        return 1
    os.makedirs(PLOTS, exist_ok=True)
    n = plot_summary(rows) + plot_head_tail(rows)
    for stem, title, color, pred in GROUPS:
        if one_group(stem, title, color, pred, rows):
            n += 1
        else:
            print(f"NOTE: skipped {stem}")
    print(f"wrote {n} block figures to {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
