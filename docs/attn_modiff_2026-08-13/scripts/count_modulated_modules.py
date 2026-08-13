"""Where do 140 modulated convs come from when only 70 get static scales?

The gate in linear_modiff_w4a4_ab.py counted 140 OptimizedInt4Conv2d with modiff_enabled=True in
mode int4, and 0 in mode int4_baseline (so it is reading real state, not a constant). But the setup
log says "Converting UNet to INT4 (89 conv layers)" and "Loaded static scales for 70 INT4 conv
layers". Three different numbers for what sounds like one set, so this enumerates them instead of
reconciling them by argument -- an earlier coverage claim in this session ("35 emb linears are
unquantized") was inferred rather than counted and had to be withdrawn.

Prints every OptimizedInt4Conv2d by qualified name with the three properties that distinguish the
subsets: whether MoDiff is on, whether a static activation scale was loaded, and whether it is
reachable as a direct child of the UNet or nested inside a fused ResBlock.

Run: python docs/attn_modiff_2026-08-13/scripts/count_modulated_modules.py   # ~1 min, needs the GPU
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

OUT = "docs/attn_modiff_2026-08-13/data/module_census.json"


def main():
    H.STEPS, H.BATCH = 5, 2          # nothing is sampled; the model just has to be built
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "1"       # so the wxax family is present and countable too
    os.environ["MODIFF_DELTA_MODE"] = "static"
    r, m, s = H.build("int4", B._default_calibration_path("int4"), "static")
    unet = m.model.diffusion_model

    rows, kinds = [], Counter()
    for name, mod in unet.named_modules():
        t = type(mod).__name__
        if t not in ("OptimizedInt4Conv2d", "OptimizedInt4Linear", "QuantLinearWxAx"):
            continue
        kinds[t] += 1
        rows.append({
            "name": name,
            "cls": t,
            "modiff": bool(getattr(mod, "modiff_enabled", getattr(mod, "modiff", False))),
            "calibrated": bool(getattr(mod, "is_calibrated", False)),
            "delta_cal": bool(getattr(mod, "is_delta_calibrated", False)),
            # A conv nested under a fused ResBlock is reached through the wrapper, so its qualified
            # name carries the wrapper's attribute. This is the property that would explain a count
            # above the number of convs the UNet nominally has.
            "depth": name.count("."),
        })

    print(f"instances by class: {dict(kinds)}\n")
    for t in sorted(kinds):
        sub = [x for x in rows if x["cls"] == t]
        print(f"{t}  n={len(sub)}  modiff={sum(x['modiff'] for x in sub)}  "
              f"calibrated={sum(x['calibrated'] for x in sub)}  "
              f"delta_cal={sum(x['delta_cal'] for x in sub)}")

    convs = [x for x in rows if x["cls"] == "OptimizedInt4Conv2d"]
    print(f"\nconvs: {len(convs)} total, {sum(c['calibrated'] for c in convs)} calibrated")
    # DUPLICATE OBJECTS vs DUPLICATE NAMES. named_modules() dedupes by identity, so 140 distinct
    # names means 140 distinct objects unless one object is reachable under two names -- which
    # named_modules() would NOT show twice. Counting unique ids settles which it is.
    ids = {id(mod) for name, mod in unet.named_modules()
           if type(mod).__name__ == "OptimizedInt4Conv2d"}
    print(f"unique object ids among those convs: {len(ids)}")
    print("\nuncalibrated convs (these fall back to a per-call dynamic scale):")
    for c in convs:
        if not c["calibrated"]:
            print(f"  {c['name']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"by_class": dict(kinds), "unique_conv_ids": len(ids), "modules": rows},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    os.environ["MODIFF_LINEAR"] = "0"
    return 0


if __name__ == "__main__":
    sys.exit(main())
