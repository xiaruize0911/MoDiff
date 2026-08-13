"""Coverage gate for fix #2: with a real asymmetric table loaded, does every quantize honour z?

TWO ASSERTIONS, AND THE SECOND IS THE ONE THAT MATTERS.

  1. NOTHING RAISES under MODIFF_ZP_STRICT=1. Every site that would ignore a non-zero zero point
     while its conv adds the corrected bias now raises by default, so a clean run means coverage.

  2. THE _zp KERNELS WERE ACTUALLY CALLED, per arm, with the expected counts. Assertion 1 alone is
     worthless without this: "0 contaminated layers" is exactly what the previous census reported for
     the PTQ arm before its coverage was complete, and docs/zero_point_2026-08-13/FINDINGS.md's fourth
     lesson is that absence of evidence from an instrument with known gaps is not evidence. A table
     that silently matched nothing, or a router that fell through to the symmetric entry point, would
     pass assertion 1 and fail this one.

EXPECTED, WRITTEN DOWN BEFORE THE RUN (from docs/zp_coverage_2026-08-13/data/site_census.json):

    W4A4 PTQ    group_norm_silu_quantize_pack_nhwc_zp     > 0   (the 62 non-updown convs)
                group_norm_silu_quantize_resize_nhwc_zp   > 0   (the 8 updown convs)
                scale_quantize_and_pack_zp                = 0   (PTQ never runs a t=T warm-up)
    W4A4 MoDiff scale_quantize_and_pack_zp                > 0   (t=T, once per conv per sample)
                group_norm_silu_quantize_pack_nhwc_zp     = 0   (MoDiff convs use the delta kernels)
                group_norm_silu_quantize_resize_nhwc_zp   = 0

The MoDiff row is the interesting one and it is small ON PURPOSE: 62 of that arm's 70 quantize sites
are delta sites, z-free by construction, so its entire activation-grid exposure is the t=T seed.

Run: python docs/zp_coverage_2026-08-13/scripts/verify_zp_coverage.py    # ~4 min, needs the GPU
"""
import collections
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

#: STRICT. This is the gate: a site that ignores a live zero point must stop the run, not log.
os.environ["MODIFF_ZP_STRICT"] = "1"
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                              # noqa: E402
import modiff_cutlass as MC                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

D = "docs/zp_coverage_2026-08-13"
ZP_TABLE = "docs/zero_point_2026-08-13/data/int4_calibration_zp_clip4.5.pt"

ZP_KERNELS = ["scale_quantize_and_pack_zp",
              "group_norm_silu_quantize_pack_nhwc_zp",
              "group_norm_silu_quantize_pack_nhwc_fast_zp",
              "group_norm_silu_quantize_resize_nhwc_zp",
              "upsample2x_quantize_pack_noahat_fprop_zp"]
#: symmetric twins, counted so a fall-through is visible rather than inferred
SYM_KERNELS = ["scale_quantize_and_pack",
               "group_norm_silu_quantize_pack_nhwc",
               "group_norm_silu_quantize_resize_nhwc",
               "upsample2x_quantize_pack_noahat_fprop"]

EXPECT = {
    "int4_baseline": {"group_norm_silu_quantize_pack_nhwc_zp": "> 0",
                      "group_norm_silu_quantize_resize_nhwc_zp": "> 0",
                      "scale_quantize_and_pack_zp": "== 0"},
    "int4": {"scale_quantize_and_pack_zp": "> 0",
             "group_norm_silu_quantize_pack_nhwc_zp": "== 0",
             "group_norm_silu_quantize_resize_nhwc_zp": "== 0"},
}

#: STRUCTURAL INVARIANT, and the sharpest assertion in this file. On the MoDiff arm _forward_first_step
#: quantizes the activation ONCE with the zero point (with_bias=True) and then quantizes
#: warmup_steps-1 = 4 RESIDUALS without it (with_bias=False, bias-free conv, dynamic per-round scale).
#: So the symmetric call count must be exactly 4x the asymmetric one.
#:
#: This is here because the first version of the routing failed it 350 : 0 -- it applied z to all five,
#: shifting every warm-up residual against a conv that adds no bias to compensate. "> 0" passed that
#: bug happily; the ratio is what caught it.
WARMUP_RATIO = {"int4": ("scale_quantize_and_pack", "scale_quantize_and_pack_zp", 4)}

counts = collections.Counter()


def patch():
    for name in ZP_KERNELS + SYM_KERNELS:
        f = getattr(MC, name, None)
        if f is None:
            continue

        def w(*a, _f=f, _n=name, **k):
            counts[_n] += 1
            return _f(*a, **k)
        setattr(MC, name, w)


def run(mode):
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_DELTA_MODE"] = "static"
    r, m, s = H.build(mode, ZP_TABLE, "static")
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    nz = sum(1 for mo in m.model.diffusion_model.modules()
             if isinstance(mo, OptimizedInt4Conv2d) and float(mo.static_input_zp.item()) != 0.0)
    if nz == 0:
        raise RuntimeError("GATE FAILED: the asymmetric table reached 0 convs")
    print(f"  {nz} convs carry a non-zero zero point", flush=True)
    H.latent(r, m, s)
    del r, m, s
    torch.cuda.empty_cache()
    return nz


def main():
    H.STEPS, H.BATCH = 6, 2
    patch()
    out, failures = {}, []
    for mode in ("int4_baseline", "int4"):
        counts.clear()
        print(f"=== {mode}, asymmetric table, MODIFF_ZP_STRICT=1 ===", flush=True)
        try:
            nz = run(mode)
            raised = None
        except Exception as e:
            nz, raised = None, f"{type(e).__name__}: {e}"
            print(f"  RAISED: {raised}")
            traceback.print_exc(limit=3)
            failures.append(f"{mode}: strict run raised")
        seen = {k: counts[k] for k in ZP_KERNELS + SYM_KERNELS if counts[k]}
        print(f"  kernel calls: {seen}")
        for k, rule in EXPECT[mode].items():
            got = counts[k]
            ok = (got > 0) if rule == "> 0" else (got == 0)
            print(f"    {k:44s} {rule:5s} got {got:6d}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{mode}: {k} {rule} but got {got}")
        if mode in WARMUP_RATIO:
            sym_k, zp_k, ratio = WARMUP_RATIO[mode]
            sym, zp = counts[sym_k], counts[zp_k]
            ok = zp > 0 and sym == ratio * zp
            print(f"    warm-up invariant: {sym_k} == {ratio}x {zp_k}")
            print(f"      {sym} vs {ratio} x {zp} = {ratio * zp}   {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{mode}: warm-up ratio {sym} != {ratio}*{zp}")
        out[mode] = {"convs_with_zp": nz, "raised": raised, "counts": dict(counts)}
        print()

    os.makedirs(f"{D}/data", exist_ok=True)
    with open(f"{D}/data/coverage_gate.json", "w") as f:
        json.dump({"expectations": EXPECT, "results": out, "failures": failures}, f, indent=2)
    print(f"wrote {D}/data/coverage_gate.json")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nCOVERAGE COMPLETE: both W4A4 arms run to completion with a live asymmetric table under\n"
          "MODIFF_ZP_STRICT=1, and the _zp entry points were exercised with the counts predicted\n"
          "above. An end-to-end accuracy measurement is now valid on both arms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
