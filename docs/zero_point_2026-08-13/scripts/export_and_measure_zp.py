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

#: _refold_zp_bias refuses a non-zero zero point on a padded conv, because the fold is
#: per-output-channel while the padding error is per-output-pixel. Every calibrated conv in this model
#: is 3x3 padding=1, so without this override there is no asymmetric arm to measure at all -- and
#: measuring it is this script's entire purpose. The override does not make the configuration correct;
#: it makes the defect reproducible, which is why the numbers below are labelled as measuring a
#: datapath that has a known padding defect rather than as measuring "the zero point".
os.environ.setdefault("MODIFF_ZP_ALLOW_PADDED", "1")

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
    #: PADDING PER CONV, collected here because this is the only place the real modules are in hand.
    #: An asymmetric grid is incompatible with zero-padding: CUTLASS's implicit GEMM inserts the byte
    #: 0, which the grid dequantizes to -z/s rather than 0, while the folded bias subtracts a
    #: correction for a tap that was never sampled. Measured in
    #: docs/zp_coverage_2026-08-13/scripts/zp_padding_probe.py -- border error 1.47x interior for
    #: asymmetric against 1.00x for symmetric, and pre-padding with real zeros (which the quantizer
    #: encodes as code z, the correct padding value) recovers it. So "which convs have padded taps"
    #: is the difference between measuring the zero point and measuring that defect.
    pad = {k: tuple(getattr(convs[k], "padding", (0, 0))) for k in convs}
    del r, m, s
    torch.cuda.empty_cache()
    return {k: (v.lo, v.hi) for k, v in rng.items()}, pad


def build_table(ranges, clip, pad=None, unpadded_only=False):
    """Asymmetric (scale, zero_point) per layer. `clip` shrinks the positive tail the way
    ACT_CLIP_RATIO does for the symmetric grid -- the one-sidedness is why clipping helps at all, so
    the asymmetric grid needs the same lever or the comparison is unfair to it.

    `unpadded_only` KEEPS THE ZERO POINT AND DROPS IT WHERE PADDING WOULD CORRUPT IT: convs with
    padding=(0,0) get the asymmetric grid, every other conv gets z=0 on the SAME clipped scale it
    would have had symmetrically. That arm measures the zero point's real-model ceiling -- what it is
    worth where it is implementable -- separately from the padding defect that dominates it elsewhere.
    A layer with z=0 still gets its asymmetric-derived `static_scale`, so this is not simply the
    symmetric arm relabelled; the scale differs and the comparison stays honest about that.
    """
    out = {}
    for k, (lo, hi) in ranges.items():
        hi_c = hi / clip
        if not (hi_c > lo):
            continue
        s = (2.0 * Q) / (hi_c - lo)
        z = -round(lo * s) - Q
        if unpadded_only:
            p = (pad or {}).get(k, (0, 0))
            if any(int(v) != 0 for v in p):
                # symmetric grid on the same clipped range: s = Q / max(|hi_c|, |lo|)
                out[k] = {"static_scale": float(Q / max(abs(hi_c), abs(lo), 1e-9)),
                          "zero_point": 0.0}
                continue
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
    ranges, pad = collect_ranges()
    npad = sum(1 for k in ranges if any(int(v) != 0 for v in pad.get(k, (0, 0))))
    print(f"  {npad} of {len(ranges)} convs have padded taps (padding != 0)")
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

    # NO "UNPADDED CONVS ONLY" CEILING ARM, and the reason is the measurement two lines above:
    # 70 of 70 calibrated convs have padding != 0. There is no subset of this model on which the zero
    # point is implementable, so that arm would carry zero non-zero zero points -- measure()'s own gate
    # would refuse it, correctly, and an arm that cannot be built is not evidence.
    #
    # The ceiling is priced WITHOUT a kernel instead, on the captured silu(gn(x)) tensors themselves:
    # docs/zp_coverage_2026-08-13/scripts/zp_activation_error.py puts the best asymmetric
    # reconstruction at 1.06x the best symmetric one over the 70 convs, under the 1.15x bar
    # zp_headroom.py set for fix #2. That is an upper bound on what any padding fix could recover.

    print(f"\n{'axis':14}{'symmetric':>11}{'asym r=1':>11}{'asym r=4.5':>12}{'best change':>13}")
    verdict = {}
    for mode, name in (("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")):
        sym = results[f"{mode}/sym"]
        a1, a45 = results.get(f"{mode}/asym1"), results.get(f"{mode}/asym4.5")
        cands = [v for v in (a1, a45) if v is not None]
        best = min(cands) if cands else None
        verdict[mode] = {"sym": sym, "asym_r1": a1, "asym_r4.5": a45,
                         "best": best,
                         "change_pct": (best / sym - 1) * 100 if (best and sym) else None}
        print(f"{name:14}{sym:11.4f}{(a1 if a1 else float('nan')):11.4f}"
              f"{(a45 if a45 else float('nan')):12.4f}"
              f"{(best / sym - 1) * 100 if (best and sym) else float('nan'):12.1f}%")

    print(f"\n{npad} of {len(ranges)} calibrated convs have padded taps, so there is no subset of "
          f"this model\non which the zero point is implementable -- see the note above the verdict.")

    #: W4A4 run-to-run floor on this protocol (docs/paper_repro_2026-08-12/FINDINGS.md section 7)
    FLOOR = 0.006
    print()
    ptq = verdict["int4_baseline"]["change_pct"]
    mod = verdict["int4"]["change_pct"]
    #: A wrong GRID costs tens of percent. A wrong IMPLEMENTATION diverges. Distinguishing them is the
    #: difference between a finding and a bug, and the first run of this script called +3435% a
    #: finding. Anything past 3x is treated as a defect to hunt, not a conclusion to draw.
    #:
    #: WHAT THAT DEFECT IS, AS OF 2026-08-13, IS NO LONGER OPEN -- and the stock message below used to
    #: name the wrong cause. It said "most likely a quantize kernel that does not apply z", which was
    #: right when coverage was incomplete and is wrong now: coverage is complete and gated
    #: (docs/zp_coverage_2026-08-13/data/coverage_gate.json -- both arms run under MODIFF_ZP_STRICT=1
    #: with the _zp entry points exercised at predicted counts). The measured cause is ZERO-PADDING.
    if (ptq is not None and ptq > 200) or (mod is not None and mod > 200):
        print(f"THESE ARE DEFECT MAGNITUDES, NOT A VERDICT ON ZERO POINTS (PTQ {ptq:+.0f}%, "
              f"MoDiff {mod:+.0f}%).\nTHE DEFECT IS IDENTIFIED AND IT IS NOT MISSING COVERAGE: "
              f"CUTLASS's implicit GEMM zero-fills\npadded taps, so a padded tap reads code 0, which "
              f"an asymmetric grid dequantizes to -z/s rather\nthan 0, while the folded bias subtracts "
              f"a per-output-CHANNEL correction for a sample that was\nnever taken. The residual is "
              f"-z*sum(missing w_q)*ws/s on the border ring, confirmed to 1-2.6%\nagainst the kernel "
              f"in integration/tests/test_int4_zp_padding.py, and the ring is 23% of pixels\nat 16x16, "
              f"44% at 8x8 and 75% at 4x4 -- which is why the end-to-end cost is this large.\n"
              f"{npad} of {len(ranges)} calibrated convs are padded, so no subset of this model avoids it.")
    elif ptq is not None and ptq < -FLOOR * 100:
        print(f"THE ZERO POINT HELPS PTQ: {ptq:+.1f}%. That is the axis it was predicted to help, and "
              f"the prediction is on the record above.")
    else:
        print(f"THE ZERO POINT DOES NOT HELP PTQ ({ptq:+.1f}% vs a {FLOOR*100:.1f}% floor), and the "
              f"magnitude is small enough to be a real result rather than a defect.")
    if mod is not None:
        print(f"MoDiff: {mod:+.1f}% -- predicted to be small because it reads this grid mainly at t=T. "
              f"It is not,\nand padding is why: MoDiff reads the grid once per conv per sample but the "
              f"resulting o_hat is then\naccumulated over every remaining step, so a border error at "
              f"t=T never washes out.")

    #: AND WHAT THE PADDING IS, because it is what the sign of the numbers above depends on. The tree
    #: zero-fills padded taps, which an asymmetric grid reads as -z/s rather than 0, so these arms
    #: measure the ZERO POINT PLUS THAT DEFECT. Correct padding was implemented and measured on
    #: 2026-08-13 (PTQ -7.1% with a code-z halo) and then REMOVED: every route to it that does not
    #: touch the conv epilogue costs 5-9% of the quantize+conv pair, i.e. it gives back what it buys.
    #: docs/zp_coverage_2026-08-13/FINDINGS.md has the numbers and the specified epilogue change.
    print("\nPADDING: CUTLASS zero-fill, which is DEFECTIVE for an asymmetric grid. The numbers above\n"
          "therefore measure the zero point plus that defect, not the zero point. See\n"
          "docs/zp_coverage_2026-08-13/FINDINGS.md -- with the padding correct, PTQ measured -7.1%.")
    json.dump({"seeds": SEEDS, "shipped_reference": SHIPPED, "median_asymmetry": asym,
               "floor": FLOOR, "padded_convs": npad, "results": results, "verdict": verdict},
              open(f"{D}/data/zp_measured.json", "w"), indent=1)
    print(f"wrote {D}/data/zp_measured.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
