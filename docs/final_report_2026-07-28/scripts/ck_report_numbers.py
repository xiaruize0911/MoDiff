"""Emit every table in the checkpoint report as markdown, straight from the measured JSON.

The previous checkpoint report's tables were transcribed by hand, which is how it ended up
carrying five stale figures and one self-contradiction across successive revisions. Anything
quoted in CHECKPOINT_REPORT_2026-08-01.md that is a number should come out of here.

Usage:
  python3 ck_report_numbers.py --e2e data/e2e_three_mode_2026-08-01.json \
                               --layers data/layers_2026-08-01.json \
                               [--cmp-layers data/attn_uniform.json] [--cmp-e2e ...]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ck_stages import STAGES, split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
KINDS = ["attention", "resblock_plain", "resblock_updown"]


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def h(title):
    print(f"\n\n{'='*90}\n{title}\n{'='*90}")


# ------------------------------------------------------------------ layer helpers
def layer_index(lay):
    L = {}
    for m, _ in MODES:
        for e in lay["modes"][m]:
            L.setdefault((e["kind"], tuple(e["x_shape"])), {})[m] = e
    return L


def by_kind(lay):
    """ms summed over every layer INSTANCE (pipeline_us * n_instances)."""
    out = {}
    for m, _ in MODES:
        agg = defaultdict(float)
        for e in lay["modes"][m]:
            agg[e["kind"]] += e["pipeline_us"] * e["n_instances"] / 1e3
        agg["total"] = sum(agg[k] for k in KINDS)
        out[m] = agg
    return out


def attn_weighted(lay):
    """Weighted attention us/layer: sum(us*n)/sum(n) over attention rows."""
    out = {}
    for m, _ in MODES:
        rows = [e for e in lay["modes"][m] if e["kind"] == "attention"]
        n = sum(e["n_instances"] for e in rows)
        out[m] = sum(e["pipeline_us"] * e["n_instances"] for e in rows) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--cmp-e2e", action="append", default=[])
    ap.add_argument("--cmp-layers", action="append", default=[])
    ap.add_argument("--base-e2e", help="pre-change e2e, for the layer->e2e transfer section")
    ap.add_argument("--base-layers", help="pre-change layers, for the same")
    a = ap.parse_args()

    e2e = json.load(open(rel(a.e2e)))
    lay = json.load(open(rel(a.layers)))

    # ============================================================ 1. end to end
    h(f"1. END TO END   ({a.e2e})")
    print(f"gpu={e2e['gpu']}  batch={e2e['batch']}  steps={e2e['steps']}  repeats={e2e['repeats']}")
    print("\n| mode | ms / batch | ms / sample | ms / step | vs FP16 | CV | spread |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    fp = e2e["modes"]["fp16"]["wall_us_per_batch"]
    for m, lbl in MODES:
        v = e2e["modes"][m]
        print(f"| {lbl} | {v['wall_us_per_batch']/1e3:.1f} | {v['per_sample_ms']:.3f} | "
              f"{v['per_step_ms']:.2f} | {fp/v['wall_us_per_batch']:.3f}x | "
              f"{v['wall_cv_pct']:.2f}% | {v['wall_spread_pct']:.2f}% |")
    i8 = e2e["modes"]["int8_baseline"]["wall_us_per_batch"]
    i4 = e2e["modes"]["int4_baseline"]["wall_us_per_batch"]
    print(f"\nINT4 vs INT8: {(i8/i4-1)*100:.1f}% faster ({(i8-i4)/1e3:.1f} ms/batch)")

    print("\nroute_check:")
    print("| mode | attention blocks | qout-eligible | expected | qkv / proj type |")
    print("|---|---:|---:|---:|---|")
    for m, lbl in MODES[1:]:
        r = e2e["modes"][m].get("route_check", {})
        print(f"| {lbl} | {r.get('attn_blocks','-')} | {r.get('qout_eligible','-')} | "
              f"{r.get('expected_eligible','-')} | `{r.get('qkv_type','-')}` / "
              f"`{r.get('proj_type','-')}` |")

    # ============================================================ 2. stage table
    h("2. WHOLE-MODEL TIME BY STAGE (profiler self-time, ms per batch)")
    S = {m: split(e2e["modes"][m]["kernels"]) for m, _ in MODES}
    print("| stage | FP16 | INT8 | INT4 | INT4 - INT8 |")
    print("|---|---:|---:|---:|---:|")
    for key, lbl, _, _ in STAGES:
        f_, e_, o_ = (S["fp16"][key]/1e3, S["int8_baseline"][key]/1e3,
                      S["int4_baseline"][key]/1e3)
        print(f"| {lbl} | {f_:.1f} | {e_:.1f} | {o_:.1f} | {o_-e_:+.1f} |")
    tots = {m: sum(S[m].values())/1e3 for m, _ in MODES}
    print(f"| **total** | **{tots['fp16']:.1f}** | **{tots['int8_baseline']:.1f}** | "
          f"**{tots['int4_baseline']:.1f}** | "
          f"**{tots['int4_baseline']-tots['int8_baseline']:+.1f}** |")
    for m, lbl in MODES:
        wall = e2e["modes"][m]["wall_us_per_batch"]/1e3
        print(f"  check {lbl}: stage sum {tots[m]:.1f} vs measured wall {wall:.1f} "
              f"({(tots[m]/wall-1)*100:+.2f}%)")
    fp_tot = tots["fp16"]
    print(f"\nattention core as % of FP16 whole model: "
          f"{S['fp16']['attn']/1e3/fp_tot*100:.1f}%")

    # ============================================================ 3. layer by kind
    h(f"3. LAYER LEVEL, BY KIND   ({a.layers})")
    print(f"batch={lay['batch']}")
    K = by_kind(lay)
    print("\n| mode | attention | resblock_plain | resblock_updown | total |")
    print("|---|---:|---:|---:|---:|")
    ft = K["fp16"]["total"]
    for m, lbl in MODES:
        sp = "" if m == "fp16" else f" ({ft/K[m]['total']:.3f}x)"
        print(f"| {lbl} | {K[m]['attention']:.2f} ms | {K[m]['resblock_plain']:.2f} ms | "
              f"{K[m]['resblock_updown']:.2f} ms | {K[m]['total']:.2f} ms{sp} |")
    print(f"\nattention as % of FP16 layer time: {K['fp16']['attention']/ft*100:.0f}%")
    d = K["int8_baseline"]["total"] - K["int4_baseline"]["total"]
    print(f"INT4 lead over INT8: {d:.2f} ms  "
          f"(attn {K['int8_baseline']['attention']-K['int4_baseline']['attention']:+.2f}, "
          f"plain {K['int8_baseline']['resblock_plain']-K['int4_baseline']['resblock_plain']:+.2f}, "
          f"updown {K['int8_baseline']['resblock_updown']-K['int4_baseline']['resblock_updown']:+.2f})")

    # ============================================================ 4. attention per shape
    h("4. ATTENTION, PER SHAPE (us per layer call)")
    L = layer_index(lay)
    akeys = sorted([k for k in L if k[0] == "attention"], key=lambda k: -(k[1][2]*k[1][3]))
    print("| shape | FP16 | INT8 | INT4 | INT8 x | INT4 x |")
    print("|---|---:|---:|---:|---:|---:|")
    for k in akeys:
        f_ = L[k]["fp16"]["pipeline_us"]
        e_ = L[k]["int8_baseline"]["pipeline_us"]
        o_ = L[k]["int4_baseline"]["pipeline_us"]
        n = L[k]["fp16"]["n_instances"]
        print(f"| C{k[1][1]}/T{k[1][2]*k[1][3]} x{n} | {f_:.1f} | {e_:.1f} | {o_:.1f} | "
              f"{f_/e_:.2f}x | {f_/o_:.2f}x |")
    # Two different quantities, both previously printed as "weighted" in a us column: the
    # instance-weighted mean us PER CALL, and the sum over all 21 instances in ms. The old
    # report put the ms total in the us table without relabelling it.
    W = attn_weighted(lay)
    n_all = sum(e["n_instances"] for e in lay["modes"]["fp16"] if e["kind"] == "attention")
    print(f"| **weighted mean, µs/call** | **{W['fp16']:.1f}** | **{W['int8_baseline']:.1f}** | "
          f"**{W['int4_baseline']:.1f}** | **{W['fp16']/W['int8_baseline']:.3f}x** | "
          f"**{W['fp16']/W['int4_baseline']:.3f}x** |")
    print(f"| **all {n_all} instances, ms** | **{W['fp16']*n_all/1e3:.3f}** | "
          f"**{W['int8_baseline']*n_all/1e3:.3f}** | **{W['int4_baseline']*n_all/1e3:.3f}** | "
          f"**{W['fp16']/W['int8_baseline']:.3f}x** | "
          f"**{W['fp16']/W['int4_baseline']:.3f}x** |")

    # ============================================================ 5. every layer speedup
    h("5. EVERY LAYER, SPEEDUP VS FP16  (flagging anything below 1.0x)")
    keys = sorted(L.keys(), key=lambda k: (k[0], -(k[1][2]*k[1][3]), -k[1][1]))
    print("| layer | shape | n | FP16 us | INT8 x | INT4 x | flag |")
    print("|---|---|---:|---:|---:|---:|---|")
    below = {"int8_baseline": [], "int4_baseline": []}
    for k in keys:
        f_ = L[k]["fp16"]["pipeline_us"]
        s8 = f_ / L[k]["int8_baseline"]["pipeline_us"]
        s4 = f_ / L[k]["int4_baseline"]["pipeline_us"]
        for m, s in (("int8_baseline", s8), ("int4_baseline", s4)):
            if s < 1.0:
                below[m].append((k, s))
        flag = "SLOWER THAN FP16" if min(s8, s4) < 1.0 else ""
        xs = k[1]
        print(f"| {k[0]} | C{xs[1]}/{xs[2]}^2 | {L[k]['fp16']['n_instances']} | {f_:.1f} | "
              f"{s8:.2f}x | {s4:.2f}x | {flag} |")
    for m, lbl in MODES[1:]:
        rows = sorted(below[m], key=lambda t: t[1])
        print(f"\n{lbl}: {len(rows)} layer(s) below 1.0x vs FP16"
              + ("" if not rows else ":"))
        for k, s in rows:
            print(f"   {k[0]:16s} C{k[1][1]}/{k[1][2]}^2  {s:.3f}x")
        s8s = [L[k]["fp16"]["pipeline_us"]/L[k][m]["pipeline_us"] for k in keys]
        print(f"   range {min(s8s):.2f}x - {max(s8s):.2f}x")

    # ============================================================ 6. attention stages
    h("6. ATTENTION STAGES PER SHAPE (us per layer, scaled to measured pipeline_us)")
    for k in akeys:
        print(f"\nC{k[1][1]} / T{k[1][2]*k[1][3]}")
        print("| stage | FP16 | INT8 | INT4 |")
        print("|---|---:|---:|---:|")
        sc = {}
        for m, _ in MODES:
            sp = split(L[k][m]["kernels"], usname="us_per_layer_call")
            tot = max(sum(sp.values()), 1e-9)
            f = L[k][m]["pipeline_us"] / tot
            sc[m] = {kk: vv * f for kk, vv in sp.items()}
        for key, lbl, _, _ in STAGES:
            vals = [sc[m].get(key, 0.0) for m, _ in MODES]
            if max(vals) < 0.05:
                continue
            print(f"| {lbl} | {vals[0]:.1f} | {vals[1]:.1f} | {vals[2]:.1f} |")
        print(f"| **total** | **{L[k]['fp16']['pipeline_us']:.1f}** | "
              f"**{L[k]['int8_baseline']['pipeline_us']:.1f}** | "
              f"**{L[k]['int4_baseline']['pipeline_us']:.1f}** |")

    # ============================================================ 7. cross-checks
    for p in a.cmp_layers:
        h(f"7. CROSS-CHECK, LAYERS: {p}")
        c = json.load(open(rel(p)))
        Kc = by_kind(c)
        print("| mode | total (this file) | total (primary) | delta |")
        print("|---|---:|---:|---:|")
        for m, lbl in MODES:
            t_c, t_p = Kc[m]["total"], K[m]["total"]
            print(f"| {lbl} | {t_c:.2f} ms | {t_p:.2f} ms | {t_p-t_c:+.2f} ms "
                  f"({(t_p/t_c-1)*100:+.1f}%) |")
        print(f"speedups: INT8 {Kc['fp16']['total']/Kc['int8_baseline']['total']:.3f}x  "
              f"INT4 {Kc['fp16']['total']/Kc['int4_baseline']['total']:.3f}x")
        sub = [(e["kind"], e["x_shape"], e["speedup_vs_fp16"])
               for e in c["modes"]["int4_baseline"] if e["speedup_vs_fp16"] < 1.0]
        print(f"INT4 rows below 1.0x in this file: {len(sub)}")
        for s in sorted(sub, key=lambda t: t[2]):
            print("  ", s)

    if a.base_e2e and a.base_layers:
        h(f"8. LAYER -> E2E TRANSFER  (base: {a.base_layers} + {a.base_e2e})")
        for m in ("int4_baseline", "int8_baseline"):
            transfer_gap(json.load(open(rel(a.base_layers))), json.load(open(rel(a.base_e2e))),
                         lay, e2e, mode=m)
            print()

    for p in a.cmp_e2e:
        h(f"7. CROSS-CHECK, E2E: {p}")
        c = json.load(open(rel(p)))
        print("| mode | ms/batch (this file) | ms/batch (primary) | delta | vs FP16 (this) |")
        print("|---|---:|---:|---:|---:|")
        fpc = c["modes"]["fp16"]["wall_us_per_batch"]
        for m, lbl in MODES:
            t_c = c["modes"][m]["wall_us_per_batch"]/1e3
            t_p = e2e["modes"][m]["wall_us_per_batch"]/1e3
            print(f"| {lbl} | {t_c:.1f} | {t_p:.1f} | {t_p-t_c:+.1f} "
                  f"({(t_p/t_c-1)*100:+.2f}%) | {fpc/c['modes'][m]['wall_us_per_batch']:.3f}x |")


def transfer_gap(base_lay, base_e2e, lay, e2e, mode="int4_baseline"):
    """How much of an isolated layer-level gain shows up end to end.

    Both halves are normalised by the FP16 column measured in the SAME session, because the
    absolute numbers drift ~1% between sessions and the raw difference would otherwise mix that
    drift into the effect. The old report's version of this quantity compared a layer delta in
    ms/step against an e2e delta in ms/batch without normalising either.
    """
    def lay_ratio(d):
        K = by_kind(d)
        return K[mode]["total"] / K["fp16"]["total"]

    def e2e_ratio(d):
        return (d["modes"][mode]["wall_us_per_batch"]
                / d["modes"]["fp16"]["wall_us_per_batch"])

    lr0, lr1 = lay_ratio(base_lay), lay_ratio(lay)
    er0, er1 = e2e_ratio(base_e2e), e2e_ratio(e2e)
    steps = e2e["steps"]
    # predicted e2e ms/batch if the layer-level fraction transferred in full
    fp_batch = e2e["modes"]["fp16"]["wall_us_per_batch"] / 1e3
    print(f"mode: {mode}   steps={steps}   batch={e2e['batch']}")
    print(f"  layer-level {mode}/FP16 ratio: {lr0:.4f} -> {lr1:.4f}  "
          f"({(lr1/lr0-1)*100:+.2f}%)")
    print(f"  e2e         {mode}/FP16 ratio: {er0:.4f} -> {er1:.4f}  "
          f"({(er1/er0-1)*100:+.2f}%)")
    gain_lay, gain_e2e = 1 - lr1 / lr0, 1 - er1 / er0
    if abs(gain_lay) > 1e-9:
        print(f"  transfer: {gain_e2e/gain_lay*100:.0f}% of the layer-level gain appears e2e")
    print(f"  in ms/batch: layer-level gain predicts {gain_lay*er0*fp_batch:.0f} ms, "
          f"measured {(er0-er1)*fp_batch:.0f} ms")
    b_ms = base_e2e["modes"][mode]["wall_us_per_batch"] / 1e3
    n_ms = e2e["modes"][mode]["wall_us_per_batch"] / 1e3
    print(f"  raw {mode} e2e: {b_ms:.1f} -> {n_ms:.1f} ms/batch ({n_ms-b_ms:+.1f}), "
          f"FP16 {base_e2e['modes']['fp16']['wall_us_per_batch']/1e3:.1f} -> {fp_batch:.1f}")


if __name__ == "__main__":
    main()
