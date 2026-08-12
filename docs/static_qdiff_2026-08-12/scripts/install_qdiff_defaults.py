"""Install the four artifacts static Q-Diffusion needs, and gate them before they ship.

The shipped configuration follows README:96 -- MoDiff reproduced with
`--modulate --quant_mode qdiff --cali_min_max`, i.e. calibrated STATIC scales rather than the
dynamic per-call ones. That takes two files per bit width, not one, and the delta half has never
shipped before:

  int8_calibration_qdiff.pt   activation scale, t=T and the PTQ baseline   <- from qdiff_runs/act_sym
  int8_delta_qdiff.pt         per-step delta table, t<T                    <- from qdiff_runs/delta
  int4_calibration_qdiff.pt   ditto at 4 bits                              <- from qdiff_runs/w4a4_sym
  int4_delta_qdiff.pt         ditto at 4 bits                              <- from qdiff_runs/w4a4_delta

WHY THE DELTA HALF IS NEW. apply_int8_delta_scales / apply_int4_delta_scales existed with ZERO call
sites in integration/, so no delta table was ever loaded and MODIFF_DELTA_MODE=static ran on an
uncalibrated grid -- at int4 it did not even fall back to a table, it quantized the temporal delta on
the full activation grid, which per Theorem 4.3 leaves the error unchanged from baseline. Both halves
are wired now (benchmark_ldm.py:_load_delta_table), which is what makes "static Q-Diffusion" a thing
this tree can actually run rather than a flag that appeared to work.

GATES, because a delta table is exactly the kind of artifact that fails silently. A file whose keys
do not match the model's layer names loads 0 layers and the run still samples -- on an uncalibrated
grid, looking like a mild regression. So each file is checked for: the shipped 70-layer key set,
finite positive scales, and (delta only) the Tensor[256] per-step shape the loader forward-fills from.

Run: python docs/static_qdiff_2026-08-12/scripts/install_qdiff_defaults.py [--dry-run]   # ~2 s, no GPU
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

import torch                                                                # noqa: E402

QD = "docs/qdiff_bridge_2026-08-12/data"
REPORT = "docs/static_qdiff_2026-08-12/data/installed.json"
#: (source, destination, kind). The reference key set comes from the shipped absmax file of the
#: same bit width, which is what integration/ has always keyed layers by.
PLAN = [(f"{QD}/qdiff_act_sym.pt", "integration/calibration/int8_calibration_qdiff.pt",
         "act", "integration/calibration/int8_calibration_realckpt.pt"),
        #: _flat, i.e. --delta-head 0, NOT the default --delta-head 2. The head policy clamps the
        #: first H modulated steps to min(qdiff_scale, act_scale/2) for a provable non-clipping
        #: guarantee, and qdiff FINDINGS §8 measured that guarantee as a LOSS: flat 0.0240 against
        #: H=2's 0.0317. min() picks the coarser grid and the coarseness costs more than the
        #: clipping it prevents. Installing qdiff_delta.pt here would ship the worse of the two.
        (f"{QD}/qdiff_delta_flat.pt", "integration/calibration/int8_delta_qdiff.pt",
         "delta", "integration/calibration/int8_delta_calibration.pt"),
        (f"{QD}/qdiff_w4a4_sym.pt", "integration/calibration/int4_calibration_qdiff.pt",
         "act", "integration/calibration/int4_calibration_realckpt.pt"),
        (f"{QD}/qdiff_w4a4_delta.pt", "integration/calibration/int4_delta_qdiff.pt",
         "delta", "integration/calibration/int4_delta_calibration.pt")]
#: integration/kernels/*_optimized.py MODIFF_MAX_STEPS
MAX_STEPS = 256


def scalar(v):
    return float(v["static_scale"]) if isinstance(v, dict) else float(v)


def check(src, kind, ref_path):
    d = torch.load(src, map_location="cpu", weights_only=True)
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    problems = []
    if set(d) != set(ref):
        problems.append(f"key set != {os.path.basename(ref_path)}: "
                        f"+{sorted(set(d) - set(ref))[:3]} -{sorted(set(ref) - set(d))[:3]}")
    if kind == "act":
        if any(isinstance(v, dict) for v in d.values()):
            # Q-Diffusion has no SmoothQuant, so a dict here means a non-qdiff file was passed --
            # and it would silently re-enable the fold on load.
            problems.append("carries smooth_scale entries; Q-Diffusion files must be bare floats")
        bad = [k for k, v in d.items() if not (scalar(v) > 0 and scalar(v) == scalar(v))]
        stat = f"scale median {statistics.median(scalar(v) for v in d.values()):.4f}"
    else:
        wrong = [k for k, v in d.items()
                 if not torch.is_tensor(v) or v.ndim != 1 or v.numel() > MAX_STEPS]
        if wrong:
            problems.append(f"{len(wrong)} entries are not a 1-D tensor of <= {MAX_STEPS} steps, "
                            f"e.g. {wrong[:3]}")
        bad = [k for k, v in d.items() if torch.is_tensor(v)
               and not bool((v > 0).all() and torch.isfinite(v).all())]
        n = {v.numel() for v in d.values() if torch.is_tensor(v)}
        stat = f"steps {sorted(n)}, scale median {statistics.median(float(v.median()) for v in d.values()):.4f}"
    if bad:
        problems.append(f"{len(bad)} non-positive or non-finite entries, e.g. {bad[:3]}")
    return d, len(d), stat, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    missing = [s for s, _, _, _ in PLAN if not os.path.exists(s)]
    if missing:
        print("FAIL: missing source exports:")
        for m in missing:
            print(f"  {m}")
        print("\n  run docs/qdiff_bridge_2026-08-12/scripts/run_calibration.sh, then export each run:")
        print("  python docs/qdiff_bridge_2026-08-12/scripts/export_qdiff_scales.py \\\n"
              "      --run <qdiff_runs/NAME> --kind static|delta --target int8|int4 --out <path>")
        return 1

    rows, failed = [], False
    for src, dst, kind, ref in PLAN:
        d, n, stat, problems = check(src, kind, ref)
        tag = "FAIL" if problems else "ok  "
        print(f"{tag} {os.path.basename(dst):32s} {kind:5s} {n:3d} layers   {stat}")
        for p in problems:
            print(f"       {p}")
            failed = True
        rows.append({"src": src, "dst": dst, "kind": kind, "layers": n, "stat": stat,
                     "problems": problems})
    if failed:
        print("\nnothing written")
        return 1
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    for src, dst, kind, _ in PLAN:
        torch.save(torch.load(src, map_location="cpu", weights_only=True), dst)
        print(f"wrote {dst}")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump({"plan": rows}, open(REPORT, "w"), indent=1)
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
