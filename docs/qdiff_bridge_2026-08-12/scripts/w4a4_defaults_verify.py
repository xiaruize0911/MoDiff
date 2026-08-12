"""Does the W4A4 landing actually arrive? Measure the DEFAULT, not a hand-passed path.

w4a4_ab.py graded four scale files by passing each one explicitly. That proves the files are good; it
does not prove the shipped default resolves to them. This script closes that gap: the calibration
path comes from benchmark_ldm._default_calibration_path(mode) -- the same call run_mode makes -- so
resolution, the mode split, and the apply path are all inside the measurement.

FOUR ARMS, so the win is re-measured in this container rather than only against committed numbers:

  int4_baseline  default (nosmooth)   vs  shipped     expect 0.4885  vs  0.7119   (-31%)
  int4           default (qdiff)      vs  shipped     expect 0.3398  vs  0.4200   (-19%)

TWO GATES BEFORE THE NUMBERS, because a silent no-op here looks exactly like a small regression:

  1. the resolved path is the intended file (and the two int4 modes resolve DIFFERENTLY -- that
     asymmetry is the whole landing).
  2. every one of the 70 convs carries the file's scale, and its SmoothQuant state matches the file
     format: identity for the bare-float files, folded for the shipped dict. A key the loader failed
     to match is invisible at runtime -- the layer simply keeps static_input_scale = 1.0 -- so this
     is asserted per layer, not counted.

Protocol identical to w4a4_ab.py: real LSUN-churches checkpoint, DDIM S=50, batch 8, seeds
{1234, 20260805, 777}, latent relL2 vs a per-seed fp16 reference, first run per arm discarded (the
attention quantizer self-calibrates on it). MODIFF_LINEAR=0 is set explicitly; it is also the default
since Stage D, so this pins the protocol rather than changing it.

Run: python docs/qdiff_bridge_2026-08-12/scripts/w4a4_defaults_verify.py    # ~15 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

SHIPPED = "integration/calibration/int4_calibration_realckpt.pt"
OUT = "docs/qdiff_bridge_2026-08-12/data/w4a4_defaults.json"
SEEDS = [1234, 20260805, 777]
#: data/w4a4_ab.json. The default arms should reproduce the file they resolve to.
EXPECT = {("int4_baseline", "default"): 0.4885, ("int4_baseline", "shipped"): 0.7119,
          ("int4", "default"): 0.3398, ("int4", "shipped"): 0.4200}
#: what _default_calibration_path must return, per mode. Asserted, not printed for eyeballing.
WANT_PATH = {"int4_baseline": "integration/calibration/int4_calibration_nosmooth.pt",
             "int4": "integration/calibration/int4_calibration_qdiff.pt"}


def check_applied(model, calib_path):
    """Every int4 conv carries this file's scale, with the SmoothQuant state the format implies."""
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    scales = torch.load(calib_path, map_location="cpu", weights_only=True)
    convs = [m for m in model.model.diffusion_model.modules() if isinstance(m, OptimizedInt4Conv2d)]
    dict_format = any(isinstance(v, dict) for v in scales.values())
    seen, bad = 0, []
    for m in convs:
        if m.layer_name not in scales:
            continue
        entry = scales[m.layer_name]
        want = float(entry["static_scale"]) if isinstance(entry, dict) else float(entry)
        got = float(m.static_input_scale.item())
        # A missed key leaves the buffer at 1.0, which is a plausible-looking scale, so compare it
        # against the file rather than against "not the default".
        if abs(got - want) > 1e-4 * max(1.0, abs(want)):
            bad.append(f"{m.layer_name}: static_input_scale {got:.4f} != file {want:.4f}")
        if bool(m._smooth_is_identity) == dict_format:
            bad.append(f"{m.layer_name}: smooth identity={bool(m._smooth_is_identity)} but the file "
                       f"{'carries' if dict_format else 'has no'} smooth_scale")
        seen += 1
    if seen != len(scales):
        bad.append(f"only {seen}/{len(scales)} of the file's layers were found in the model")
    return seen, len(convs), bad


def main():
    for mode, want in WANT_PATH.items():
        got = B._default_calibration_path(mode)
        if got != want:
            print(f"FAIL: _default_calibration_path({mode!r}) = {got}\n      expected {want}\n"
                  f"      run scripts/make_int4_defaults.py first")
            return 1
        print(f"gate 1  {mode:14s} -> {got}")
    if not os.path.exists(SHIPPED):
        print(f"FAIL: missing the control file {SHIPPED}")
        return 1

    H.STEPS, H.BATCH = 50, 8
    os.environ["MODIFF_LINEAR"] = "0"

    print("\nfp16 references ...", flush=True)
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for s in SEEDS:
        H.SEED = s
        H.latent(rf, mf, sf)                                # discard: warm-up
        refs[s] = H.latent(rf, mf, sf)[0].float()
    del rf, mf, sf
    torch.cuda.empty_cache()

    out, gates = {}, {}
    for mode, kind in (("int4_baseline", "PTQ"), ("int4", "MoDiff")):
        for arm in ("default", "shipped"):
            cal = B._default_calibration_path(mode) if arm == "default" else SHIPPED
            r, m, s = H.build(mode, cal, "static" if "baseline" in mode else "dynamic")
            seen, n_convs, bad = check_applied(m, cal)
            gates[f"{mode}/{arm}"] = {"path": cal, "layers_applied": seen, "convs": n_convs,
                                      "problems": bad}
            if bad:
                print(f"\nFAIL gate 2 on {mode}/{arm}: {len(bad)} problems")
                for b in bad[:6]:
                    print(f"       {b}")
                return 1
            print(f"gate 2  {mode:14s} {arm:8s} {seen}/{n_convs} convs carry {os.path.basename(cal)}",
                  flush=True)

            H.SEED = SEEDS[0]
            H.latent(r, m, s)                               # discard: attention self-calibration
            rels = []
            for sd in SEEDS:
                H.SEED = sd
                H.latent(r, m, s)
                lat, _ = H.latent(r, m, s)
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            out[f"{mode}/{arm}"] = {"path": cal, "relL2": rels, "mean": statistics.mean(rels)}
            exp = EXPECT[(mode, arm)]
            print(f"        {kind:6s} {arm:8s} {statistics.mean(rels):.4f}  "
                  f"{[round(x, 4) for x in rels]}   (w4a4_ab {exp:.4f})", flush=True)
            del r, m, s
            torch.cuda.empty_cache()

    print(f"\n{'arm':16s} {'shipped':>9} {'default':>9} {'change':>9}   per-seed default")
    verdict = {}
    for mode, kind in (("int4_baseline", "PTQ"), ("int4", "MoDiff")):
        sh, df = out[f"{mode}/shipped"]["mean"], out[f"{mode}/default"]["mean"]
        wins = sum(1 for a, b in zip(out[f"{mode}/shipped"]["relL2"], out[f"{mode}/default"]["relL2"])
                   if b < a)
        verdict[mode] = {"shipped": sh, "default": df, "ratio": df / sh,
                         "seeds_improved": wins, "of": len(SEEDS)}
        print(f"W4A4 {kind:11s} {sh:9.4f} {df:9.4f} {(df / sh - 1) * 100:8.1f}%   "
              f"{wins}/{len(SEEDS)} seeds improved")

    ok = all(v["ratio"] < 1.0 and v["seeds_improved"] == len(SEEDS) for v in verdict.values())
    print(f"\n{'PASS' if ok else 'FAIL'}: the default beats the shipped file on "
          f"{'both axes, all seeds' if ok else 'NOT every axis/seed -- see above'}")

    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "expect_w4a4_ab": {
        f"{m}/{a}": v for (m, a), v in EXPECT.items()}, "gates": gates, "results": out,
        "verdict": verdict, "pass": ok}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
