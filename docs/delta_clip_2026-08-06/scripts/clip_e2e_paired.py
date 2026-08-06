"""Paired per-seed read of clip_e2e_a8.json. Means alone cannot carry this comparison.

At A8/batch 8 the relL2 seed spread is enormous -- the r=1.0 row is 0.0596 +- 0.0316 over three
seeds, a CV above 50%, which is several times any effect the offline probe predicts. The sweep is
paired (every ratio sees the same three seeds and the same per-seed fp16 reference), so the question
to ask of it is "how often does ratio r beat r=1.0 on the SAME seed", not "is the mean lower".
This is the same discipline the 2026-08-06 refresh revert used ("K=1 wins 0/3").
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "docs/delta_clip_2026-08-06/data/clip_e2e_a8.json")

with open(IN) as f:
    d = json.load(f)

seeds = [str(s) for s in d["seeds"]]
by_k = {}
for row in d["rows"]:
    by_k.setdefault(row["delta_refresh"], {})[row["clip_ratio"]] = row

for k, rows in sorted(by_k.items()):
    base = rows.get(1.0)
    if base is None:
        continue
    print(f"=== MODIFF_DELTA_REFRESH={k}, A8, paired over seeds {seeds} ===")
    print(f"{'clip r':>7} | " + " ".join(f"{s:>9}" for s in seeds)
          + f" | {'mean':>8} | {'vs r=1':>7} | {'wins':>5} | {'worst seed':>10}")
    print("-" * 88)
    for r in sorted(rows, reverse=True):
        row = rows[r]
        per = [row["per_seed"][s] for s in seeds]
        b = [base["per_seed"][s] for s in seeds]
        wins = sum(1 for x, y in zip(per, b) if x < y)
        # The largest per-seed regression, as a fraction. A ratio that wins on average while
        # tripling one seed is not a default, and the mean hides exactly that.
        worst = max((x - y) / y for x, y in zip(per, b))
        print(f"{r:>7.2f} | " + " ".join(f"{v:>9.4f}" for v in per)
              + f" | {row['mean']:>8.4f} | {row['mean'] / base['mean']:>6.3f}x"
              + f" | {wins}/{len(seeds)} | {worst:>+9.1%}")
    print()
