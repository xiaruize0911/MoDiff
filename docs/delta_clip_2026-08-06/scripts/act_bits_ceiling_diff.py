"""Pre- vs post-ceiling A8..A2, per seed, both arms. What the flattered rows were actually worth.

`docs/act_bits_2026-08-05`'s sweep ran with a delta quantizer that clamped codes at a literal 127
regardless of the activation bit width, so on a MODIFF_DELTA_REFRESH>1 reuse step -- where the scale is
up to K-1 steps old and the delta may have outgrown it -- an "A4" layer could emit a code of 100. This
re-runs the same script with the ceiling in place and diffs it.

Two predictions to check, and they matter more than the numbers themselves:

  * the BASELINE arm must not move at any precision. It goes through scale_quantize_int8 /
    dynamic_quantize_int8_fprop, which this change never touched, and `_delta_code_ceiling` returns -1
    in static mode anyway. If a baseline row moves, the change is reaching calls it should not.
  * the MoDiff arm must not move at K=1 (every step measures its own absmax, so no code could exceed
    Q_b) and must get WORSE at K>1, by more the lower the precision -- A2 most of all, where the gap
    between Q_b=1 and 127 is largest.

Everything is read against the ~10% per-seed / ~3% mean cross-process floor measured in
data/a8_control_repeat.json, so only a consistent 3/3-seed shift counts.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAIRS = [
    ("K=4 (shipped default)", "docs/act_bits_2026-08-05/data/act_bit_sweep.json",
     "docs/delta_clip_2026-08-06/data/act_bit_sweep_ceiling_k4.json"),
    ("K=1 (control)", "docs/act_bits_2026-08-05/data/act_bit_sweep_refresh1.json",
     "docs/delta_clip_2026-08-06/data/act_bit_sweep_ceiling_k1.json"),
]


def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


def main():
    for label, old_p, new_p in PAIRS:
        if not os.path.exists(os.path.join(ROOT, new_p)):
            print(f"=== {label}: {new_p} not written yet, skipping ===\n")
            continue
        old, new = load(old_p), load(new_p)
        o = {r["act_bits"]: r for r in old["rows"]}
        print(f"=== {label} ===")
        print(f"  old {old_p}  (refresh={old.get('delta_refresh')}, "
              f"warmup={old.get('warmup_steps')}, anchor={old.get('anchor')})")
        print(f"  new {new_p}  (refresh={new.get('delta_refresh')}, "
              f"warmup={new.get('warmup_steps')}, anchor={new.get('anchor')})\n")
        for arm in ("baseline", "modiff"):
            print(f"  --- {arm} arm ---")
            print(f"  {'bits':>5} | {'old mean':>9} {'new mean':>9} {'ratio':>7} | "
                  f"{'per-seed old -> new':>44} | {'worse':>5}")
            print("  " + "-" * 88)
            for row in new["rows"]:
                b = row["act_bits"]
                if b not in o:
                    continue
                a, n = o[b][arm], row[arm]
                seeds = list(n["per_seed"])
                worse = sum(1 for s in seeds if n["per_seed"][s] > a["per_seed"][s])
                per = "  ".join(f"{a['per_seed'][s]:.4f}->{n['per_seed'][s]:.4f}" for s in seeds)
                print(f"  {('A%d' % b):>5} | {a['mean']:>9.4f} {n['mean']:>9.4f} "
                      f"{n['mean'] / a['mean']:>6.3f}x | {per:>44} | {worse}/{len(seeds)}")
            print()
        # The gain column is what the sweep exists to report, so show how the correction moved it.
        print(f"  --- baseline/modiff gain (what the sweep claims MoDiff buys) ---")
        print(f"  {'bits':>5} | {'old gain':>9} {'new gain':>9}")
        print("  " + "-" * 27)
        for row in new["rows"]:
            b = row["act_bits"]
            if b in o:
                print(f"  {('A%d' % b):>5} | {o[b]['gain_baseline_over_modiff']:>8.2f}x "
                      f"{row['gain_baseline_over_modiff']:>8.2f}x")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
