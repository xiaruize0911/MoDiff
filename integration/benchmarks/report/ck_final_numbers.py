"""Emit every table of the five-suite final report as markdown, from the measured JSON.

Suites, and the file each comes from:
  1 attention kernels  kernel_suites_*.json   (suite "attention")
  2 conv kernels       kernel_suites_*.json   (suite "conv")
  3 linear kernels     kernel_suites_*.json   (suite "linear")
  4 per layer          layers_*.json
  5 end to end         e2e_*.json

Everything quoted in the report should come out of here rather than being transcribed; the
07-31 checkpoint report accumulated five stale figures precisely because its tables were
hand-copied.

Two things this script deliberately does NOT do:

* It does not weight kernel times by the call counts recorded during capture. The capture
  window is a handful of steps and the attention route is NOT the same in the first steps as
  in the steady state -- at T=1024 the 25 calls of a 5-step window split 10 / 5 / 10 across
  flash_attn_*_vt, _vt_static and the fused *_qout entry, and at T=16 they split 15 / 10
  between fp16 SDPA and the quantized small kernel. Over 200 steps the fused entry dominates
  (the e2e profile shows the T=1024 kernel firing exactly 1000 times = 5 blocks x 200 steps,
  i.e. one entry per block per step). So per-call times come from the kernel suite and the
  call MIX comes from the 200-step e2e profile.

* It does not force a per-shape fp16-vs-quantized alignment for conv and linear. The layouts
  differ per entry -- fp16 conv weight is (Cout,Cin,k,k), int8 is (Cout,k,k,Cin), int4 is
  (Cout,k,k,Cin/2) against an NHWC packed input, and the int4 GEMM pads K -- so a positional
  rule silently mismatches rows. Conv and linear are compared on the canonical key derived
  from the weight tensor where that is unambiguous, and rows that cannot be aligned are
  listed rather than dropped.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ck_bench_stats import ratio_ci, summarize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
KINDS = ["attention", "resblock_plain", "resblock_updown"]


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def h(t):
    print("\n\n" + "=" * 94 + "\n" + t + "\n" + "=" * 94)


def f_ci(st, prec=1):
    if not st:
        return "—"
    return "%.*f ± %.*f" % (prec, st["mean"], prec, st["ci95_half"])


# ------------------------------------------------------------------ conv / linear canonical key
def conv_key(row):
    """(Cin, H, W, Cout, k) from whichever layout this entry uses, or None."""
    e, sh = row["entry"], row["arg_shapes"]
    if len(sh) < 2:
        return None
    x, w = sh[0], sh[1]
    if e.startswith("torch_conv2d"):                       # x=(N,Cin,H,W) w=(Cout,Cin,kh,kw)
        return (w[1], x[2], x[3], w[0], w[2])
    if "conv2d_int8" in e:                                 # x=(N,Cin,H,W) w=(Cout,kh,kw,Cin)
        return (w[3], x[2], x[3], w[0], w[1])
    if "conv2d_int4" in e:                                 # x=(N,H,W,Cin/2) w=(Cout,kh,kw,Cin/2)
        return (w[3] * 2, x[1], x[2], w[0], w[1])
    return None


def lin_key(row):
    """(M, K, N) where derivable. int4 pads K, so it is reported but not used for matching."""
    e, sh = row["entry"], row["arg_shapes"]
    if len(sh) < 2:
        return None
    x, w = sh[0], sh[1]
    m = x[0] if len(x) == 2 else x[0] * x[1]
    if e.startswith("torch_linear"):
        return (m, w[1], w[0])
    if "gemm_" in e:
        return (m, w[1], w[0])
    return None


def attn_key(row):
    """(T, hd) for any attention entry: T is the dim in the known token set, hd the last."""
    sh = row["arg_shapes"][0]
    tokens = {1024, 256, 64, 16, 4}
    hd = sh[-1]
    cand = [i for i, v in enumerate(sh) if v in tokens and i != len(sh) - 1]
    t = sh[cand[-1]] if cand else None
    return (t, hd)


def entry_priority(row):
    """Which entry represents the STEADY-STATE production path at a given key.

    Picking the slowest entry (an earlier version's rule) is wrong, because several entries
    coexist at one key and the extra ones belong to the first few steps, not to a 200-step run:
    at T=1024 the INT8 mode shows flash_attn_int8_vt (1843.5 us), _vt_static (1597.8) and the
    fused qi8_kv_static_qout_hd24 (1597.3), and "slowest" picked the startup one, reporting
    1.10x where the steady-state kernel gives 1.27x. At T=4 it was worse: it picked the fp16
    SDPA fallback that fires in early steps (59.5 us) over the quantized small kernel that runs
    afterwards (19.1 us), turning a 2.55x win into a reported 0.82x loss.

    Priority: fused *_qout entry > any other native entry > torch's own op.
    """
    e = row["entry"]
    if e.startswith("torch_"):
        return 0
    if "_qout" in e:
        return 2
    return 1


def suite_table(ks, suite, keyfn, keyname, top=None, alignfn=None):
    """Per-mode table of per-call time, then a cross-mode view on the canonical key."""
    h("%s KERNELS — per call, batch %d, %d rounds x %d iters, warmup %d"
      % (suite.upper(), ks["batch"], ks["rounds"], ks["iters_per_round"], ks["warmup"]))
    for m, lbl in MODES:
        rows = [r for r in ks["modes"].get(m, {}).get(suite, []) if r.get("stats")]
        rows.sort(key=lambda r: -r["stats"]["mean"])
        shown = rows if top is None else rows[:top]
        print("\n**%s** — %d signatures%s" %
              (lbl, len(rows), "" if top is None else " (top %d by time)" % len(shown)))
        print("| entry | %s | µs/call (mean ± 95%% CI) | CV | spread | n | stability |"
              % keyname)
        print("|---|---|---:|---:|---:|---:|---|")
        for r in shown:
            st = r["stats"]
            k = keyfn(r)
            print("| `%s` | %s | %s | %.2f%% | %.2f%% | %d | %s |"
                  % (r["entry"], k if k else "—", f_ci(st), st["cv_pct"],
                     st["spread_pct"], st["n"], r["stability"]))

    # Cross-mode alignment. For attention this must NOT include the head dim: the quantized
    # kernels pad hd 24 -> 32 at T=1024, so a (T, hd) key never matches fp16 there even though
    # it is the same attention layer. Alignment is therefore on T alone, with hd shown per mode
    # in the tables above.
    align = alignfn or keyfn
    idx = collections.defaultdict(dict)
    for m, _ in MODES:
        for r in ks["modes"].get(m, {}).get(suite, []):
            if not r.get("stats"):
                continue
            k = align(r)
            if k is None:
                continue
            cur = idx[k].get(m)
            if cur is None or entry_priority(r) > entry_priority(cur):
                idx[k][m] = r
    full = [(k, v) for k, v in idx.items() if len(v) == len(MODES)]
    print("\n**Cross-mode, on %s** — %d of %d keys present in all three modes"
          % (keyname, len(full), len(idx)))
    if full:
        print("| %s | FP16 µs | INT8 µs | INT4 µs | INT8 × (95%% CI) | INT4 × (95%% CI) |"
              " quantized? |" % keyname)
        print("|---|---:|---:|---:|---:|---:|---|")
        quant_ratios = {"int8_baseline": [], "int4_baseline": []}
        for k, v in sorted(full, key=lambda kv: -kv[1]["fp16"]["stats"]["mean"]):
            f, e8, e4 = v["fp16"]["stats"], v["int8_baseline"]["stats"], v["int4_baseline"]["stats"]
            s8, s4 = ratio_ci(f, e8), ratio_ci(f, e4)
            # A key whose entry is torch's own op in ALL THREE modes was never quantized. Its
            # cross-mode ratio measures the input dtype the surrounding pipeline handed it
            # (fp32 in fp16 mode vs fp16 in the quantized modes), NOT a quantization gain --
            # for conv that is every 1x1 skip conv plus the in/out convs.
            is_q = not all(v[m]["entry"].startswith("torch_") for m in v)
            if is_q:
                quant_ratios["int8_baseline"].append(s8["ratio"])
                quant_ratios["int4_baseline"].append(s4["ratio"])
            print("| %s | %.1f | %.1f | %.1f | %.3f ± %.3f | %.3f ± %.3f | %s |"
                  % (k, f["mean"], e8["mean"], e4["mean"],
                     s8["ratio"], s8["ci95_half"], s4["ratio"], s4["ci95_half"],
                     "yes" if is_q else "**no — dtype only**"))
        for m, lbl in (("int8_baseline", "INT8"), ("int4_baseline", "INT4")):
            rs = sorted(quant_ratios[m])
            if rs:
                print("\n%s over the %d genuinely quantized keys: %.2f-%.2fx, median %.2fx"
                      % (lbl, len(rs), rs[0], rs[-1], rs[len(rs) // 2]))
    unaligned = {k: sorted(v) for k, v in idx.items() if len(v) != len(MODES)}
    if unaligned:
        print("\nkeys NOT present in all three modes (listed, not dropped): %d"
              % len(unaligned))
        for k, v in sorted(unaligned.items())[:12]:
            print("   %s -> %s" % (k, ", ".join(v)))


def layer_tables(lay):
    h("PER LAYER — batch %d, %d rounds x %d iters" %
      (lay["batch"], lay["modes"]["fp16"][0]["stats"]["n"],
       lay["modes"]["fp16"][0]["stats"]["iters_per_round"]))
    agg = {}
    for m, _ in MODES:
        a = collections.defaultdict(float)
        for e in lay["modes"][m]:
            a[e["kind"]] += e["pipeline_us"] * e["n_instances"] / 1e3
        a["total"] = sum(a[k] for k in KINDS)
        agg[m] = a
    print("\n| mode | attention | resblock_plain | resblock_updown | total | vs FP16 |")
    print("|---|---:|---:|---:|---:|---:|")
    ft = agg["fp16"]["total"]
    for m, lbl in MODES:
        print("| %s | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %s |"
              % (lbl, agg[m]["attention"], agg[m]["resblock_plain"],
                 agg[m]["resblock_updown"], agg[m]["total"],
                 "1.000×" if m == "fp16" else "%.3f×" % (ft / agg[m]["total"])))

    print("\n**Per (kind, shape), with the distribution**")
    print("| kind | shape | n | FP16 µs ± CI (CV) | INT8 µs ± CI (CV) | INT4 µs ± CI (CV) |"
          " INT8 × | INT4 × |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    L = {}
    for m, _ in MODES:
        for e in lay["modes"][m]:
            L.setdefault((e["kind"], tuple(e["x_shape"])), {})[m] = e
    for k in sorted(L, key=lambda k: (k[0], -(k[1][2] * k[1][3]), -k[1][1])):
        v = L[k]
        if len(v) != len(MODES):
            continue
        f, e8, e4 = (v["fp16"]["stats"], v["int8_baseline"]["stats"],
                     v["int4_baseline"]["stats"])
        s8, s4 = ratio_ci(f, e8), ratio_ci(f, e4)
        print("| %s | C%d/%d² | %d | %s (%.2f%%) | %s (%.2f%%) | %s (%.2f%%) | %.2f± %.2f | %.2f ± %.2f |"
              % (k[0], k[1][1], k[1][2], v["fp16"]["n_instances"],
                 f_ci(f), f["cv_pct"], f_ci(e8), e8["cv_pct"], f_ci(e4), e4["cv_pct"],
                 s8["ratio"], s8["ci95_half"], s4["ratio"], s4["ci95_half"]))


def e2e_table(e2e):
    h("END TO END — batch %d, %d steps, %d repeats" %
      (e2e["batch"], e2e["steps"], e2e["repeats"]))
    # Files written before the stats contract landed carry only the hand-rolled fields; rebuild
    # the summary from the retained per-repeat samples so older data still reads.
    for m, _ in MODES:
        d = e2e["modes"][m]
        if not d.get("stats") and d.get("wall_all_us"):
            d["stats"] = summarize(d["wall_all_us"])
            d.setdefault("stability", "—")
    fp = e2e["modes"]["fp16"]
    print("\n| mode | ms/batch (mean ± 95% CI) | ms/step | vs FP16 (95% CI) | CV | spread |"
          " n | stability |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for m, lbl in MODES:
        d = e2e["modes"][m]
        st = d.get("stats")
        sp = ratio_ci(fp.get("stats"), st) if m != "fp16" else None
        print("| %s | %.1f ± %.1f | %.2f | %s | %.2f%% | %.2f%% | %d | %s |"
              % (lbl, st["mean"] / 1e3, st["ci95_half"] / 1e3, d["per_step_ms"],
                 "1.000×" if sp is None else "%.3f ± %.3f" % (sp["ratio"], sp["ci95_half"]),
                 st["cv_pct"], st["spread_pct"], st["n"], d.get("stability", "—")))
    print("\nper-repeat samples (ms/batch):")
    for m, lbl in MODES:
        s = e2e["modes"][m]["stats"]["samples"]
        print("  %-5s %s" % (lbl, "  ".join("%.0f" % (x / 1e3) for x in s)))


def stability_summary(ks, lay, e2e):
    h("STABILITY ACROSS ALL FIVE SUITES")
    print("| suite | measurements | median CV | p90 CV | max CV | NOISY (CV>3%) |")
    print("|---|---:|---:|---:|---:|---:|")

    def line(name, cvs, noisy):
        cvs = sorted(cvs)
        if not cvs:
            return
        p90 = cvs[int(0.9 * (len(cvs) - 1))]
        print("| %s | %d | %.2f%% | %.2f%% | %.2f%% | %d |"
              % (name, len(cvs), cvs[len(cvs) // 2], p90, cvs[-1], noisy))

    for suite in ("attention", "conv", "linear"):
        cvs, noisy = [], 0
        for m, _ in MODES:
            for r in ks["modes"].get(m, {}).get(suite, []):
                if r.get("stats"):
                    cvs.append(r["stats"]["cv_pct"])
                    noisy += r.get("stability") == "NOISY"
        line("%s kernels" % suite, cvs, noisy)
    cvs = [e["stats"]["cv_pct"] for m, _ in MODES for e in lay["modes"][m] if e.get("stats")]
    noisy = sum(1 for m, _ in MODES for e in lay["modes"][m] if e.get("stability") == "NOISY")
    line("per layer", cvs, noisy)
    cvs = [e2e["modes"][m]["stats"]["cv_pct"] for m, _ in MODES]
    line("end to end", cvs, sum(1 for c in cvs if c > 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    ks = json.load(open(rel(a.kernels)))
    lay = json.load(open(rel(a.layers)))
    e2e = json.load(open(rel(a.e2e)))

    print("gpu=%s  batch=%d" % (ks["gpu"], ks["batch"]))
    suite_table(ks, "attention", lambda r: "T=%s hd=%s" % attn_key(r), "T / head dim",
                alignfn=lambda r: "T=%s" % attn_key(r)[0])
    suite_table(ks, "conv", lambda r: "%s" % (conv_key(r),), "(Cin,H,W,Cout,k)", top=a.top)
    suite_table(ks, "linear", lambda r: "%s" % (lin_key(r),), "(M,K,N)", top=a.top)
    layer_tables(lay)
    e2e_table(e2e)
    stability_summary(ks, lay, e2e)


if __name__ == "__main__":
    main()
