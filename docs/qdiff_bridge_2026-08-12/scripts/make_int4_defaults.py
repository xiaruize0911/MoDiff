"""Install the two W4A4 wins as integration/'s int4 defaults.

The W4A4 A/B (data/w4a4_ab.json, 3 seeds, DDIM S=50, batch 8) found the best file to be DIFFERENT on
the two axes, so landing both wins means shipping two files, not one:

                    shipped   no-smooth   qdiff sym     best
      W4A4 PTQ       0.7119    0.4885      1.1945       no-smooth  (-31%)
      W4A4 MoDiff    0.4200    0.3963      0.3398       qdiff sym  (-19%)

  int4_calibration_nosmooth.pt   -> default for int4_baseline   (PTQ)
  int4_calibration_qdiff.pt      -> default for int4, int4_attn_modiff  (MoDiff)

Wired up in integration/benchmarks/benchmark_ldm.py:CALIBRATION_PREFERENCE, which is where the
mode split and its cost are documented.

WHAT "NO-SMOOTH" IS. Not a recalibration -- the same 70 static scales as the shipped file, re-emitted
as bare floats instead of {"static_scale", "smooth_scale"} dicts. apply_int4_static_scales refolds
smooth_scale only for dict-valued entries, so dropping the key IS the switch that turns SmoothQuant
off; the float path (set_static_scale) leaves the weights unsmoothed and the activations unsmoothed.
That the shipped scale was derived from the SMOOTHED range and is therefore ~3-5x too large for
unsmoothed input is not a bug in this file -- it is the measured trade: at 4 bits, spreading the
per-output-channel weight range by s (2.96-5.39) costs more than the extra activation clipping does.
See FINDINGS.md 5a/5c -- the sample grid shows fog (shipped) turning into cathedral structure.

THE SHIPPED FILE IS NOT TOUCHED. int4_calibration_realckpt.pt stays byte-identical, so every
committed number that names it explicitly keeps meaning what it meant, and both new preference lists
fall back to it when these files are absent.

Run:
  python docs/qdiff_bridge_2026-08-12/scripts/make_int4_defaults.py            # ~1 s, no GPU
  python docs/qdiff_bridge_2026-08-12/scripts/make_int4_defaults.py --dry-run

The qdiff source is data/qdiff_w4a4_sym.pt, itself regenerable with:
  python docs/qdiff_bridge_2026-08-12/scripts/export_qdiff_scales.py \
      --run docs/qdiff_bridge_2026-08-12/qdiff_runs/w4a4_sym --kind static --target int4 \
      --out docs/qdiff_bridge_2026-08-12/data/qdiff_w4a4_sym.pt
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT]

import torch                                                                # noqa: E402

SHIPPED = "integration/calibration/int4_calibration_realckpt.pt"
QDIFF_SRC = "docs/qdiff_bridge_2026-08-12/data/qdiff_w4a4_sym.pt"
OUT_NOSMOOTH = "integration/calibration/int4_calibration_nosmooth.pt"
OUT_QDIFF = "integration/calibration/int4_calibration_qdiff.pt"
REPORT = "docs/qdiff_bridge_2026-08-12/data/int4_defaults.json"
#: data/w4a4_ab.json, for the record this script writes next to the files it makes
MEASURED = {"PTQ": {"shipped": 0.7119, "nosmooth": 0.4885, "qdiff_sym": 1.1945},
            "MoDiff": {"shipped": 0.4200, "nosmooth": 0.3963, "qdiff_sym": 0.3398}}


def static_scale(entry):
    """Both formats collapse to one float: the dict's static_scale, or the bare float itself."""
    return float(entry["static_scale"]) if isinstance(entry, dict) else float(entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    a = ap.parse_args()

    for p in (SHIPPED, QDIFF_SRC):
        if not os.path.exists(p):
            print(f"FAIL: missing {p}" + ("\n      regenerate with export_qdiff_scales.py, see the "
                                          "module docstring" if p == QDIFF_SRC else ""))
            return 1

    shipped = torch.load(SHIPPED, map_location="cpu", weights_only=True)
    qdiff = torch.load(QDIFF_SRC, map_location="cpu", weights_only=True)

    # A dropped key is INVISIBLE at runtime -- apply_int4_static_scales just skips the layer and it
    # keeps static_input_scale = 1.0. FINDINGS trap #3 lost 4 keys to a regex exactly this way, so
    # both files are gated on set equality with the shipped 70 rather than on a count.
    n_smoothed = sum(1 for v in shipped.values() if isinstance(v, dict))
    print(f"shipped   {SHIPPED}\n          {len(shipped)} layers, {n_smoothed} carrying smooth_scale")
    if set(qdiff) != set(shipped):
        only_q, only_s = set(qdiff) - set(shipped), set(shipped) - set(qdiff)
        print(f"FAIL: qdiff key set != shipped. qdiff-only {sorted(only_q)[:4]}, "
              f"shipped-only {sorted(only_s)[:4]}")
        return 1
    if any(isinstance(v, dict) for v in qdiff.values()):
        print("FAIL: the qdiff export carries smooth_scale entries. Q-Diffusion has no SmoothQuant, "
              "so a dict here means the wrong file was passed.")
        return 1

    nosmooth = {k: static_scale(v) for k, v in shipped.items()}
    assert all(static_scale(shipped[k]) == nosmooth[k] for k in shipped), "nosmooth must not requantize"

    bad = [k for k, v in list(nosmooth.items()) + list(qdiff.items())
           if not (v > 0.0 and v == v and v != float("inf"))]
    if bad:
        print(f"FAIL: non-positive or non-finite scale on {len(bad)} layers, e.g. {bad[:4]}")
        return 1

    ratio = [float(qdiff[k]) / nosmooth[k] for k in shipped]
    print(f"nosmooth  {len(nosmooth)} bare floats, scales bit-identical to shipped static_scale")
    print(f"qdiff     {len(qdiff)} bare floats, {statistics.median(ratio):.3f}x the shipped scale "
          f"(median; {min(ratio):.3f}-{max(ratio):.3f})")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    for path, obj in ((OUT_NOSMOOTH, nosmooth), (OUT_QDIFF, {k: float(v) for k, v in qdiff.items()})):
        assert os.path.abspath(path) != os.path.abspath(SHIPPED), "refusing to overwrite the shipped file"
        torch.save(obj, path)
        print(f"wrote {path}")

    json.dump({"shipped": SHIPPED, "qdiff_src": QDIFF_SRC,
               "out": {"int4_baseline": OUT_NOSMOOTH, "int4": OUT_QDIFF},
               "layers": len(shipped), "smoothed_in_shipped": n_smoothed,
               "qdiff_over_shipped_scale": {"median": statistics.median(ratio),
                                            "min": min(ratio), "max": max(ratio)},
               "measured_relL2": MEASURED},
              open(REPORT, "w"), indent=1)
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
