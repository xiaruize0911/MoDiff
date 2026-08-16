"""Compute FID for every generated mode against the real LSUN-Churches reference set.

Reads the folders produced by export_lsun_reference.py and generate_fid_samples.py and runs
pytorch_fid's Inception-v3 pool3 features. The real set's statistics are computed ONCE and reused
for every mode -- recomputing them per mode would be five identical passes over 10k images, and,
more importantly, any nondeterminism in that pass would show up as a difference between modes.

Reported alongside FID:
  * n on each side. FID is biased by sample count -- it decreases monotonically with N and does not
    converge until well past 10k. Numbers here are NOT comparable to published LSUN-Churches FIDs
    computed at 50k. Comparisons BETWEEN modes at the same N are what this is for.
  * FID against fp16 as well as against real. The first answers "how close is this mode to the
    fp16 model it is approximating", which is the quantization question; the second answers "how
    good are the images", which also carries the base model's own error. The paper's tables are
    against real, so both are given.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
#: pytorch_fid was missing here on 2026-08-16 and it failed the LAST step of a 25-minute pipeline, so
#: the check runs before this script's own imports. It inserts ROOT itself because this script's
#: sys.path setup happens further down -- the preflight has to precede that to be worth having.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..")))
from integration.utils.preflight import preflight, FID
preflight(*FID, what="compute_fid.py")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/fid")
ap.add_argument("--real", default="real")
ap.add_argument("--modes", default="fp16,int8_baseline,int8_modiff,int4_baseline,int4_modiff")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--dims", type=int, default=2048)
ap.add_argument("--out", default="docs/fid_2026-08-05/data/fid.json")
a = ap.parse_args()

from pytorch_fid.fid_score import calculate_frechet_distance, compute_statistics_of_path  # noqa
from pytorch_fid.inception import InceptionV3                                             # noqa

LABEL = {"fp16": "FP16 (reference model)",
         "int8_baseline": "W8A8 baseline (MoDiff off)",
         "int8_modiff": "W8A8 + MoDiff",
         "int4_baseline": "W4A4 baseline (MoDiff off)",
         "int4_modiff": "W4A4 + MoDiff"}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = InceptionV3.BLOCK_INDEX_BY_DIM[a.dims]
    model = InceptionV3([block]).to(dev)

    def stats(path):
        n = len([f for f in os.listdir(path) if f.endswith(".png")])
        mu, sigma = compute_statistics_of_path(path, model, a.batch, a.dims, dev, num_workers=8)
        return mu, sigma, n

    real_dir = os.path.join(a.root, a.real)
    print(f"real: {real_dir}", flush=True)
    mu_r, s_r, n_r = stats(real_dir)
    print(f"  n={n_r}", flush=True)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    cache, out = {}, {}
    for m in modes:
        d = os.path.join(a.root, m)
        if not os.path.isdir(d):
            print(f"  {m}: MISSING {d}")
            continue
        mu, s, n = stats(d)
        cache[m] = (mu, s, n)
        out[m] = {"label": LABEL.get(m, m), "n": n,
                  "fid_vs_real": float(calculate_frechet_distance(mu_r, s_r, mu, s))}
        print(f"  {m}: n={n}  FID vs real = {out[m]['fid_vs_real']:.3f}", flush=True)

    if "fp16" in cache:
        mu_f, s_f, _ = cache["fp16"]
        for m in cache:
            out[m]["fid_vs_fp16"] = (0.0 if m == "fp16" else
                                     float(calculate_frechet_distance(mu_f, s_f, *cache[m][:2])))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"n_real": n_r, "dims": a.dims, "modes": out}, f, indent=2)

    print(f"\n{'mode':<30}{'n':>7}{'FID vs real':>14}{'FID vs fp16':>14}")
    for m in modes:
        if m not in out:
            continue
        v = out[m]
        vf = v.get("fid_vs_fp16")
        print(f"{v['label']:<30}{v['n']:>7}{v['fid_vs_real']:>14.3f}"
              f"{(f'{vf:.3f}' if vf is not None else '—'):>14}")
    print(f"\nwrote {a.out}")
    print(f"NOTE: n={n_r} reference / {min(v['n'] for v in out.values())} generated. FID is "
          f"biased upward at this N versus the standard 50k; cross-mode comparison is the valid use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
