"""Does an asymmetric activation grid help W4A4? Export a zero-point calibration and measure it.

This is the measurement the plan's fix #2 said could only be made by implementing the thing -- the
fake-quant harness that would have priced it in advance failed its own self-check twice (predicted
0.1147 for fix #1 where the kernels delivered 0.3099; then put the symmetric optimum at 6.7 where the
real kernels put it at 4.5, i.e. wrong ORDERING, not just wrong magnitude). So the kernel was built
first and this measures it on the real datapath.

WHAT IS BEING TESTED. silu(gn(x)) is one-sided: measured |max|/|min| = 19.91x, with only 5 of 15 int4
codes carrying >0.1% of the mass -- an effective 2.32 bits of a nominal 3.91. A symmetric grid sized by
the positive tail spends 7 codes on a range 20x narrower than the one that matters. An asymmetric grid
spans [lo, hi] instead of [-absmax, +absmax]:

    a_q = clamp(round(a*s) + z, -7, 7)      s = 14 / (hi - lo)      z = -round(lo*s) - 7

and the dequantization's -z*sum(w_q) term is folded into the conv bias at calibration time, so the GEMM
never sees z (OptimizedInt4Conv2d._refold_zp_bias, gated bit-exactly in test_int4_zero_point.py).

THE EXPECTATION, STATED BEFORE THE RUN so the result cannot be reinterpreted afterwards: this should
help the PTQ axis and barely move MoDiff. MoDiff reads the static activation grid essentially only at
t=T and then refines a_hat with 5 warm-up rounds -- which is why the earlier real-kernel sweep found it
FLAT across a 10x clip-ratio range (1.09x) while PTQ moved 1.84x. zp_headroom.py's own docstring says
the decision rests on the PTQ axis. If MoDiff moves a lot here, something else is going on and the
number should be distrusted, not celebrated.

PINNED fp16 REFERENCES. fp16 sampling is nondeterministic across processes on this machine (relL2
~4-6e-3, cuDNN picking its convolution algorithm per process), which is enough to move a relL2 by ~1%
and is what made an earlier A/B unable to reproduce a committed value. References come from
fp16_refs.py so both arms are graded against literally the same tensors.

Run: python docs/zero_point_2026-08-13/scripts/export_and_measure_zp.py    # ~25 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                #: act_fake_quant.target_convs lives here -- it is the mapping from calibration-file
                #: names to conv modules that every earlier probe used, verified 70/70.
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

D = "docs/zero_point_2026-08-13"
SEEDS = [1234, 20260805, 777]
Q = 7.0
#: committed, same protocol (docs/static_qdiff_2026-08-12/data/static_vs_dynamic_ab.json)
SHIPPED = {"int4_baseline": 0.4695, "int4": 0.3090}


class Range:
    def __init__(self):
        self.lo, self.hi = float("inf"), float("-inf")

    def __call__(self, mod, args):
        v = args[0].detach()
        self.lo = min(self.lo, float(v.min()))
        self.hi = max(self.hi, float(v.max()))
        return None


def collect_ranges():
    """Range of silu(gn(x)) per conv -- the quantity the activation grid actually quantizes.

    COLLECTED ON THE fp16 MODEL, hooking the plain nn.Conv2d, which is how
    docs/state_report_2026-08-12/scripts/probe_int4_code_use.py measured |max|/|min| = 19.91x.

    The first version of this function built the INT4 model and hooked OptimizedInt4Conv2d. That looks
    more faithful and is wrong: on the fused path the conv is entered through
    `conv.forward_from_int4(packed, ...)`, a direct method call, so a forward_pre_hook never sees
    silu(gn(x)) at all -- it fired somewhere else and captured the PRE-GroupNorm input. The tell was
    right there and I did not look at it: the hook reported median |max|/|min| = 0.79x for a quantity
    that is bounded below by -0.2785 and unbounded above. Every derived (scale, zero_point) described
    the wrong tensor, and the run "concluded" that fix #2 does not help, by +86% on PTQ and +3435% on
    MoDiff -- magnitudes that are a bug signature, not a grid being suboptimal.

    In fp16 mode the ResBlock runs `self.fused_in_norm_silu(x)` and then calls `self.in_conv(h)`
    normally, so the hook receives exactly silu(gn(x)).
    """
    import act_fake_quant as A
    os.environ["MODIFF_LINEAR"] = "0"
    r, m, s = H.build("fp16", None, "static")
    convs = A.target_convs(m.model.diffusion_model)
    rng = {k: Range() for k in convs}
    hs = [convs[k].register_forward_pre_hook(rng[k]) for k in convs]
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
    for h in hs:
        h.remove()
    del r, m, s
    torch.cuda.empty_cache()
    return {k: (v.lo, v.hi) for k, v in rng.items()}, {}


def build_table(ranges, clip):
    """Asymmetric (scale, zero_point) per layer. `clip` shrinks the positive tail the way
    ACT_CLIP_RATIO does for the symmetric grid -- the one-sidedness is why clipping helps at all, so
    the asymmetric grid needs the same lever or the comparison is unfair to it."""
    out = {}
    for k, (lo, hi) in ranges.items():
        hi_c = hi / clip
        if not (hi_c > lo):
            continue
        s = (2.0 * Q) / (hi_c - lo)
        z = -round(lo * s) - Q
        out[k] = {"static_scale": float(s), "zero_point": float(z)}
    return out


def measure(mode, table, refs, label):
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    H.AUTO_DELTA_TABLE = True
    cal = table if table is not None else B._default_calibration_path(mode)
    r, m, s = H.build(mode, cal, "static")
    if table is not None:
        # Prove the zero points actually reached the modules. A table that silently matched nothing
        # would produce the symmetric numbers and look like "no effect" -- the failure mode that has
        # cost the most time in this session.
        from integration.kernels.int4_optimized import OptimizedInt4Conv2d
        nz = sum(1 for mo in m.model.diffusion_model.modules()
                 if isinstance(mo, OptimizedInt4Conv2d)
                 and float(mo.static_input_zp.item()) != 0.0)
        if nz == 0:
            print(f"  {label}: GATE FAILED -- 0 convs have a non-zero zero point")
            del r, m, s
            torch.cuda.empty_cache()
            return None
        print(f"  ({nz} convs carry a non-zero zero point)", flush=True)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                        # discard: attention self-calibration
    rels = []
    for sd in SEEDS:
        H.SEED = sd
        H.latent(r, m, s)
        lat, _ = H.latent(r, m, s)
        rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
    del r, m, s
    torch.cuda.empty_cache()
    mean = statistics.mean(rels)
    print(f"  {label:36s} {mean:.4f}   {[round(x, 4) for x in rels]}", flush=True)
    return mean


def main():
    H.STEPS, H.BATCH = 50, 8
    os.makedirs(f"{D}/data", exist_ok=True)

    print("collecting per-conv silu(gn(x)) ranges ...", flush=True)
    ranges, _ = collect_ranges()
    asym = statistics.median(abs(hi) / max(abs(lo), 1e-9) for lo, hi in ranges.values())
    print(f"  {len(ranges)} convs, median |max|/|min| = {asym:.2f}x")
    # SANITY GATE ON THE INSTRUMENT, not on the result. silu is bounded below by -0.2785 and unbounded
    # above; probe_int4_code_use.py measured 19.91x on this model. A value near 1 means the hook
    # captured a roughly symmetric tensor -- i.e. NOT silu(gn(x)) -- and every scale derived from it is
    # meaningless. The first run of this script reported 0.79x and went on to draw a confident
    # conclusion from it anyway.
    if asym < 5.0:
        print(f"REFUSING TO CONTINUE: |max|/|min| = {asym:.2f}x, but silu(gn(x)) is one-sided and this "
              f"model measures 19.91x. The hook is not seeing silu(gn(x)); fix the collection before "
              f"trusting any number below.")
        return 1

    import fp16_refs
    print("\nfp16 references (pinned) ...", flush=True)
    refs = fp16_refs.get(H.STEPS, H.BATCH, SEEDS)

    results = {}
    print("\nsymmetric baseline (the shipped files):", flush=True)
    for mode, lab in (("int4_baseline", "W4A4 PTQ  symmetric"), ("int4", "W4A4 MoDiff symmetric")):
        results[f"{mode}/sym"] = measure(mode, None, refs, lab)

    for clip in (1.0, 4.5):
        tbl = build_table(ranges, clip)
        path = f"{D}/data/int4_calibration_zp_clip{clip:g}.pt"
        torch.save(tbl, path)
        zs = [v["zero_point"] for v in tbl.values()]
        print(f"\nasymmetric, positive tail / {clip:g}  ({len(tbl)} layers, "
              f"z in [{min(zs):.0f}, {max(zs):.0f}], median {statistics.median(zs):.0f}):", flush=True)
        for mode, lab in (("int4_baseline", f"W4A4 PTQ  asym clip {clip:g}"),
                          ("int4", f"W4A4 MoDiff asym clip {clip:g}")):
            results[f"{mode}/asym{clip:g}"] = measure(mode, path, refs, lab)

    print(f"\n{'axis':14}{'symmetric':>11}{'asym r=1':>11}{'asym r=4.5':>12}{'best change':>13}")
    verdict = {}
    for mode, name in (("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")):
        sym = results[f"{mode}/sym"]
        a1, a45 = results.get(f"{mode}/asym1"), results.get(f"{mode}/asym4.5")
        cands = [v for v in (a1, a45) if v is not None]
        best = min(cands) if cands else None
        verdict[mode] = {"sym": sym, "asym_r1": a1, "asym_r4.5": a45, "best": best,
                         "change_pct": (best / sym - 1) * 100 if (best and sym) else None}
        print(f"{name:14}{sym:11.4f}{(a1 if a1 else float('nan')):11.4f}"
              f"{(a45 if a45 else float('nan')):12.4f}"
              f"{(best / sym - 1) * 100 if (best and sym) else float('nan'):12.1f}%")

    #: W4A4 run-to-run floor on this protocol (docs/paper_repro_2026-08-12/FINDINGS.md section 7)
    FLOOR = 0.006
    print()
    ptq = verdict["int4_baseline"]["change_pct"]
    mod = verdict["int4"]["change_pct"]
    #: A wrong GRID costs tens of percent. A wrong IMPLEMENTATION diverges. Distinguishing them is the
    #: difference between a finding and a bug, and the first run of this script called +3435% a
    #: finding. Anything past 3x is treated as a defect to hunt, not a conclusion to draw.
    if (ptq is not None and ptq > 200) or (mod is not None and mod > 200):
        print(f"THESE ARE BUG MAGNITUDES, NOT A RESULT (PTQ {ptq:+.0f}%, MoDiff {mod:+.0f}%). A merely "
              f"suboptimal 4-bit grid costs tens of percent; relL2 in the units seen here means the "
              f"datapath is inconsistent -- most likely a quantize kernel that does not apply z while "
              f"the bias carries a correction for it. Do not report this as fix #2's answer.")
    elif ptq is not None and ptq < -FLOOR * 100:
        print(f"THE ZERO POINT HELPS PTQ: {ptq:+.1f}%. That is the axis it was predicted to help, and "
              f"the prediction is on the record above.")
    else:
        print(f"THE ZERO POINT DOES NOT HELP PTQ ({ptq:+.1f}% vs a {FLOOR*100:.1f}% floor), and the "
              f"magnitude is small enough to be a real result rather than a defect.")
    if mod is not None:
        print(f"MoDiff: {mod:+.1f}% -- expected to be small; it reads this grid mainly at t=T.")
    json.dump({"seeds": SEEDS, "shipped_reference": SHIPPED, "median_asymmetry": asym,
               "floor": FLOOR, "results": results, "verdict": verdict},
              open(f"{D}/data/zp_measured.json", "w"), indent=1)
    print(f"wrote {D}/data/zp_measured.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
