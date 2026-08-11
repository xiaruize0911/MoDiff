"""Fold the corrected fp16 arm into differential_timing.json, and say that it came from elsewhere.

The first run of `differential_timing.py` measured its fp16 arm with `MODIFF_QUANT_LINEAR=1` still
inherited from BASE_ENV, which converted 79 nn.Linear to W8A8 -- so that arm was not fp16 (the route
check recorded `wxax: 79`, and `_assert_route` now refuses it). Rather than spend another 40 minutes
re-measuring 11 correct quantized arms, the fp16 arm alone is re-run and merged here.

That makes fp16 the ONE row in the file measured in a different process from the others, which is
worth a flag rather than a footnote: this project's own control puts same-script cross-process drift
at up to ~10% per seed on relL2, and the two prior e2e runs put fp16 at 105.84 and 106.98 ms/step,
1.1% apart. Nothing structural depends on it -- every arm's `delta_from` is another quantized arm,
so fp16 enters only as the `speedup_vs_fp16` denominator -- but a 1% wobble in that column is
measurement, not model, and `fp16_separate_process` in the output says so.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAIN = os.path.join(ROOT, "docs/component_attribution_2026-08-07/data/differential_timing.json")


def main(rerun_path):
    main_d = json.load(open(MAIN))
    re_d = json.load(open(rerun_path))
    assert "fp16" in re_d["arms"], f"{rerun_path} has no fp16 arm"
    rc = re_d["arms"]["fp16"]["route_check"]
    assert not rc.get("wxax") and not rc.get("int8_convs"), f"re-run is still not fp16: {rc}"
    for k in ("batch", "steps", "repeats", "warmups"):
        assert main_d[k] == re_d[k], f"protocol differs on {k}"

    main_d["arms"]["fp16"] = re_d["arms"]["fp16"]
    main_d["fp16_separate_process"] = True
    main_d["fp16_note"] = (
        "The fp16 row was re-measured in its own process after the first run's fp16 arm was found "
        "to be partly quantized (MODIFF_QUANT_LINEAR=1 leaked from BASE_ENV; route check read "
        "wxax=79). Every other arm is from one process. fp16 is only the speedup_vs_fp16 "
        "denominator; no marginal in this file is taken against it.")

    for lab, d in main_d["arms"].items():
        b = d["delta_from"]
        if b and b in main_d["arms"]:
            d["delta_ms_per_step"] = d["ms_per_step"] - main_d["arms"][b]["ms_per_step"]
            d["ratio_vs_base"] = d["ms_per_step"] / main_d["arms"][b]["ms_per_step"]
    fp = main_d["arms"]["fp16"]["ms_per_step"]
    for d in main_d["arms"].values():
        d["speedup_vs_fp16"] = fp / d["ms_per_step"]

    with open(MAIN, "w") as f:
        json.dump(main_d, f, indent=1)

    print(f"{'arm':<24}{'ms/step':>10}{'vs fp16':>9}{'delta':>10}  {'from':<20}{'CV':>7}")
    for lab, d in main_d["arms"].items():
        dl = (f"{d['delta_ms_per_step']:+.2f}" if d.get("delta_ms_per_step") is not None else "—")
        print(f"{lab:<24}{d['ms_per_step']:>10.2f}{d['speedup_vs_fp16']:>8.3f}x{dl:>10}  "
              f"{str(d['delta_from'] or ''):<20}{d['wall_cv_pct']:>6.2f}%")
    print(f"\nUPDATED {MAIN}")


if __name__ == "__main__":
    main(sys.argv[1])
