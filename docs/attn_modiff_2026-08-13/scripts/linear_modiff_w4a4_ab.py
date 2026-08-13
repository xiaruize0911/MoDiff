"""What does extending MoDiff to the attention projections cost and buy at W4A4?

THE QUESTION. The shipped default modulates the 70 ResBlock convs and leaves the 42 attention
qkv/proj Linears as plain static PTQ (benchmark_ldm.py:927, MODIFF_LINEAR defaults to 0). The comment
justifying that default says the quality cost is acceptable because "the 'recognisable churches vs
fog' concern was always about W4A4, where BOTH arms (0.36 / 0.42) are bad and which is not a
recommended configuration."

That premise has expired. W4A4 MoDiff is now 0.3090 and produces legible cathedrals, because
DELTA_CLIP_RATIO and ACT_CLIP_RATIO landed on 2026-08-12. So the flag has never been measured in the
regime the tree now ships, and this measures it.

LABELLED AS ITS OWN ARM, not as a variant of "int4". `int4_linmodiff` = mode "int4" +
MODIFF_LINEAR=1. It is deliberately NOT called int4_attn_modiff: that name is already taken by a
DIFFERENT implementation (modiff_attention wraps the qkv/proj Conv1d and leaves QK^T/AV in fp16
SDPA), and conflating two routes under one label is how a measurement ends up describing the wrong
configuration.

A STRUCTURAL GATE RUNS FIRST, because a flag that silently does nothing is this session's most
repeated failure -- MODIFF_USE_EMA was committed swapping 0 parameters, the static delta path shipped
with 0 call sites, and MODIFF_DELTA_REFRESH's default is dead code in static mode. So before any
relL2 is believed, this counts the Linears that actually carry modiff=True and REFUSES to report if
the count is not 0 (off arm) and 42 (on arm). A null result from a flag that never engaged looks
exactly like a null result from a flag that does nothing useful.

ONE fp16 REFERENCE SET IS CORRECT HERE, unlike the EMA A/B. MODIFF_LINEAR changes only how the
quantized arms quantize; it does not touch the fp16 network, so both arms are graded against the same
references. (The EMA arm changed the weights themselves, which is why that experiment needed two.)

Protocol as every other W4A4 A/B in this tree: real LSUN-churches checkpoint, DDIM S=50, batch 8,
seeds {1234, 20260805, 777}, latent relL2 vs a per-seed fp16 reference, first run per arm discarded
(the quantized attention self-calibrates on it), all arms in one process, calibration and delta table
resolved through the shipped preference mechanism.

NOTE ON THE LINEAR DELTA GRID. The conv delta is a static per-step table; the wxax linear delta is
measured per call (delta_absmax_fp16, MODIFF_LINEAR_DELTA_REFRESH default 1 = every step). So this arm
is not "static Q-Diffusion everywhere" -- it is static on the convs and dynamic on the projections.
Left at the defaults on purpose: the question is what the flag does as shipped, not as tuned.

Run: python docs/attn_modiff_2026-08-13/scripts/linear_modiff_w4a4_ab.py    # ~25 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

OUT = "docs/attn_modiff_2026-08-13/data/linear_modiff_w4a4_ab.json"
SEEDS = [1234, 20260805, 777]
#: shipped, same protocol (docs/static_qdiff_2026-08-12/data/static_vs_dynamic_ab.json)
SHIPPED = {"int4_baseline": 0.4695, "int4": 0.3090}
#: 21 AttentionBlocks x (qkv, proj)
EXPECT_MODULATED = 42


def count_modulated(model):
    """Modules carrying MoDiff temporal state, counted on the MODULE not the environment.

    THERE ARE THREE FAMILIES, not two, and only one of them is what MODIFF_LINEAR gates:

      convs              OptimizedInt4Conv2d, 70. Modulated in mode int4, always.
      emb Linears        OptimizedInt4Linear, 37 (35 ResBlock emb_layers + 2 time_embed). ALSO
                         modulated in mode int4, always -- benchmark_ldm.py:668 calls
                         enable_modiff_mode_int4_linear(..., True) unconditionally for the mode.
                         Easy to miss because it is a different class and a different call site from
                         the attention projections, and because these Linears are still nn.Linear at
                         the time the count in "Converting UNet linear layers to INT4 (37)" is taken.
      attn qkv/proj      QuantLinearWxAx, 42 (21 blocks x 2). The ONLY family MODIFF_LINEAR gates.

    So the shipped default modulates 107 of 149 quantized modules, not 70 -- which is worth stating
    precisely, since the plan's fix #5 framed coverage as "the paper quantizes 168 modules, we
    calibrate 70" and that conflated calibration with modulation.

    Reads the same attribute the benchmark's own route check reads on the wxax family (`.modiff`), so
    a disagreement between this and that guard would mean the two look at different objects.
    """
    #: Substring, not startswith("_QuantLinear"). The first draft used the leading underscore that
    #: e2e_three_mode_bench's comments use, which matches NOTHING -- the class is QuantLinearWxAx
    #: (wxax_linear.py:48). That draft would have failed the gate for a reason unrelated to the flag,
    #: i.e. the instrument would have reported the exact failure it exists to detect.
    wxax = emb = conv = 0
    for m in model.model.diffusion_model.modules():
        t = type(m).__name__
        if "QuantLinear" in t and bool(getattr(m, "modiff", False)):
            wxax += 1
        elif t == "OptimizedInt4Linear" and bool(getattr(m, "modiff_enabled", False)):
            emb += 1
        elif t == "OptimizedInt4Conv2d" and bool(getattr(m, "modiff_enabled", False)):
            conv += 1
    return {"attn_proj_wxax": wxax, "emb_linear": emb, "conv": conv}


def measure(mode, linear, refs, label):
    """One arm. `linear` is MODIFF_LINEAR, which must be set BEFORE _setup_model: it is read at
    conversion time in convert_linears_to_wxax, not at sampling time."""
    os.environ["MODIFF_LINEAR"] = "1" if linear else "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    cal = B._default_calibration_path(mode)
    r, m, s = H.build(mode, cal, "static")

    cnt = count_modulated(m)
    want = EXPECT_MODULATED if (linear and mode == "int4") else 0
    if cnt["attn_proj_wxax"] != want:
        print(f"  GATE FAILED for {label}: {cnt['attn_proj_wxax']} modulated attention "
              f"projections, expected {want}   (full count {cnt})")
        del r, m, s
        torch.cuda.empty_cache()
        return None
    print(f"  gate ok: modulated {cnt}", flush=True)

    H.SEED = SEEDS[0]
    H.latent(r, m, s)                        # discard: attention self-calibration
    rels, mss = [], []
    for sd in SEEDS:
        H.SEED = sd
        H.latent(r, m, s)
        lat, ms = H.latent(r, m, s)
        rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
        mss.append(ms)
    del r, m, s
    torch.cuda.empty_cache()
    print(f"  {label:34s} {statistics.mean(rels):.4f}   {[round(x, 3) for x in rels]}   "
          f"{statistics.mean(mss):.1f} ms/step", flush=True)
    return {"mean": statistics.mean(rels), "relL2": rels, "modulated": cnt,
            "ms_per_step_smallbatch": statistics.mean(mss), "calibration": cal}


def main():
    H.STEPS, H.BATCH = 50, 8
    p = B._default_delta_path("int4")
    if not p or "qdiff" not in p:
        print(f"FAIL: int4 delta default resolves to {p}, not a qdiff table")
        return 1
    print(f"gate  int4 delta table -> {p}")

    # PINNED REFERENCES, not rebuilt here. This script rebuilt them itself twice and could not
    # reproduce the reference that three other harnesses agree on bit-for-bit: the first attempt was
    # 6.9% off (H.AUTO_DELTA_TABLE was assigned inside measure(), so the references saw the module
    # default False) and fixing that still left 1.1%, above the 0.6% floor. Four experiments
    # eliminated the arms, the delta table, the calibration file, the arm order and the measure()
    # path. So the reference is loaded from one cached file instead, which makes cross-harness
    # comparison exact by construction. See fp16_refs.py for the residual nondeterminism it confines.
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"
    print("\nfp16 references (pinned) ...", flush=True)
    import fp16_refs
    refs = fp16_refs.get(H.STEPS, H.BATCH, SEEDS)

    #: (label, mode, MODIFF_LINEAR). int4_baseline with the flag ON is a CONTROL, not an arm:
    #: benchmark_ldm's is_modiff whitelist excludes the baselines, so it must come out identical to
    #: the shipped 0.4695. If it moves, the flag is reaching somewhere it should not and every other
    #: number here is suspect.
    ARMS = [("int4          MoDiff conv only (shipped)", "int4", False),
            ("int4_linmodiff  + attn projections", "int4", True),
            ("int4_baseline   CONTROL, flag must not reach", "int4_baseline", True)]
    out = {}
    print("\narms:", flush=True)
    for label, mode, lin in ARMS:
        res = measure(mode, lin, refs, label)
        if res is None:
            return 1
        out[label] = res
    os.environ["MODIFF_LINEAR"] = "0"

    off = out["int4          MoDiff conv only (shipped)"]["mean"]
    on = out["int4_linmodiff  + attn projections"]["mean"]
    ctrl = out["int4_baseline   CONTROL, flag must not reach"]["mean"]

    #: REFERENCE SELF-CHECK, added after the first run of this script reported the shipped arm at
    #: 0.3303 against a committed 0.3090 and it took four separate experiments to establish that the
    #: arms were fine and the REFERENCES were not. A harness that grades against fp16 has to prove its
    #: fp16 is the same fp16 everything else used, and the cheapest proof is that an arm with a known
    #: value reproduces it. Without this the run still "succeeds" and quietly reports every arm with a
    #: shared offset -- which is invisible in a within-run comparison and wrong in a cross-run table.
    FLOOR = 0.006
    off_drift = abs(off / SHIPPED["int4"] - 1)
    if off_drift > FLOOR:
        print(f"\nREFERENCES SUSPECT: the shipped arm reads {off:.4f} where the committed value is "
              f"{SHIPPED['int4']:.4f} ({off_drift*100:+.1f}%, floor {FLOOR*100:.1f}%). Every arm in "
              f"this run shares those references, so the WITHIN-RUN comparison below is still "
              f"internally consistent, but none of these numbers may be quoted against a committed "
              f"one. Fix the references before reporting.")
        json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "shipped_reference": SHIPPED,
                   "noise_floor": FLOOR, "results": out, "references_trusted": False},
                  open(OUT, "w"), indent=1)
        return 1
    print(f"\nreference self-check ok: shipped arm {off:.4f} vs committed {SHIPPED['int4']:.4f} "
          f"({off_drift*100:+.1f}%)")

    print(f"\n{'arm':40s}{'relL2':>9}{'vs shipped int4':>18}")
    print(f"{'int4 (shipped, conv-only MoDiff)':40s}{off:9.4f}{'-':>18}")
    print(f"{'int4_linmodiff (+ 42 projections)':40s}{on:9.4f}{(on/off-1)*100:17.1f}%")
    print(f"{'int4_baseline (control)':40s}{ctrl:9.4f}"
          f"{'  drift ' + format((ctrl/SHIPPED['int4_baseline']-1)*100, '.1f') + '%':>18}")

    ctrl_drift = abs(ctrl / SHIPPED["int4_baseline"] - 1)
    print()
    if ctrl_drift > 0.03:
        print(f"CONTROL DRIFTED {ctrl_drift*100:.1f}% -- the baseline arm should be untouched by "
              f"MODIFF_LINEAR. Treat the comparison above as unverified until that is explained.")
    if abs(on / off - 1) <= FLOOR:
        print("WITHIN THE NOISE FLOOR: extending MoDiff to the projections changes fidelity by less "
              "than run-to-run scatter. Then it is a pure throughput question -- keep the default 0.")
    elif on < off:
        print(f"HELPS on fidelity ({off:.4f} -> {on:.4f}). Whether to flip the default is now a "
              f"price question: see the e2e arm int4_linmodiff for the ms/step it costs.")
    else:
        print(f"HURTS on fidelity ({off:.4f} -> {on:.4f}), so the default 0 is right on BOTH axes "
              f"and the expired justification in benchmark_ldm.py:927 should be replaced with this.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "shipped_reference": SHIPPED,
               "noise_floor": FLOOR, "results": out, "references_trusted": True},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
