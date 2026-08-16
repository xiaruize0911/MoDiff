"""Error bars for a paired 10k FID comparison, by bootstrap over the generated set.

WHY. B3 measured the MSE int4 weight scale at FID 180.593 against a paired absmax arm at 170.854, i.e.
+5.70% -- and the honest caveat on that was "one 10k FID per arm, so the pairing removes seed noise but
there are no error bars on FID itself". A 9.7-point gap on a 170 baseline is the same order as the latent
relL2 effect it contradicts, so the number cannot flip a default until it has an interval. This session
has already retracted four conclusions that were single measurements; that is the reason this exists.

WHAT IT DOES. Extracts Inception pool3 features ONCE per set (real, arm A, arm B), then resamples image
INDICES with replacement and recomputes FID from the resampled statistics. Feature extraction is the only
GPU work and it happens once; the bootstrap itself is linear algebra on 2048-dim statistics.

PAIRED BOOTSTRAP, not two independent ones. The two arms were generated from the SAME seed sequence, so
image i in arm A and image i in arm B share a noise draw. Resampling the two arms independently would
throw that pairing away and inflate the interval on the DIFFERENCE, which is the quantity in question. So
one index set per replicate is applied to both arms, and the statistic is FID(B) - FID(A) per replicate.

WHAT IT DOES NOT CAPTURE. Only sampling variance of the generated set. It does not capture variation from
a different noise draw (the pairing removes that from the difference by construction) nor from a different
real reference (held fixed on purpose -- see compute_fid.py's header).

Run: python docs/gn_fast_reduce_2026-08-16/scripts/fid_bootstrap.py \
         --real /workspace/fid/real --a /workspace/fid_b3_absmax/int4_modiff \
         --b /workspace/fid_b3/int4_modiff --label-a absmax --label-b mse
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from pytorch_fid.fid_score import calculate_frechet_distance          # noqa: E402
from pytorch_fid.inception import InceptionV3                         # noqa: E402
from PIL import Image                                                 # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"


def features(path, model, batch=64):
    """Inception pool3 features for every PNG in `path`, in sorted order so the pairing survives."""
    files = sorted(pathlib.Path(path).glob("*.png"))
    assert files, f"no PNGs in {path}"
    out = np.empty((len(files), 2048), dtype=np.float64)
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        arr = np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in chunk])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).cuda()
        with torch.inference_mode():
            f = model(t)[0].squeeze(-1).squeeze(-1)
        out[i:i + len(chunk)] = f.cpu().numpy().astype(np.float64)
        if (i // batch) % 40 == 0:
            print(f"  {path}: {i + len(chunk)}/{len(files)}", flush=True)
    return out


def fid_from(feat_gen, mu_r, sigma_r):
    mu = feat_gen.mean(axis=0)
    sigma = np.cov(feat_gen, rowvar=False)
    return calculate_frechet_distance(mu, sigma, mu_r, sigma_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = InceptionV3([3]).cuda().eval()
    print("extracting features (once per set) ...")
    fr = features(args.real, model)
    fa = features(args.a, model)
    fb = features(args.b, model)
    assert len(fa) == len(fb), f"arms must be paired: {len(fa)} vs {len(fb)}"
    mu_r, sigma_r = fr.mean(axis=0), np.cov(fr, rowvar=False)

    point_a = fid_from(fa, mu_r, sigma_r)
    point_b = fid_from(fb, mu_r, sigma_r)
    print(f"\npoint estimates: {args.label_a} {point_a:.3f}   {args.label_b} {point_b:.3f}   "
          f"diff {point_b - point_a:+.3f} ({100 * (point_b - point_a) / point_a:+.2f}%)")

    rng = np.random.default_rng(args.seed)
    n = len(fa)
    da, db, dd = [], [], []
    for r in range(args.reps):
        idx = rng.integers(0, n, size=n)          # ONE index set, applied to both arms
        va = fid_from(fa[idx], mu_r, sigma_r)
        vb = fid_from(fb[idx], mu_r, sigma_r)
        da.append(va); db.append(vb); dd.append(vb - va)
        if r % 10 == 0:
            print(f"  replicate {r + 1}/{args.reps}: diff {vb - va:+.3f}", flush=True)

    da, db, dd = np.array(da), np.array(db), np.array(dd)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print(f"\n{args.reps} paired bootstrap replicates, n={n} resampled with replacement")
    print(f"  {args.label_a:>8}: {da.mean():8.3f} +- {da.std(ddof=1):.3f}")
    print(f"  {args.label_b:>8}: {db.mean():8.3f} +- {db.std(ddof=1):.3f}")
    print(f"  difference ({args.label_b} - {args.label_a}): {dd.mean():+.3f} "
          f"+- {dd.std(ddof=1):.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    crosses = lo <= 0.0 <= hi
    print(f"  -> {'NOT RESOLVED: the interval contains 0' if crosses else 'RESOLVED: 0 is outside the interval'}")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"label_a": args.label_a, "label_b": args.label_b, "n": int(n), "reps": args.reps,
               "point_a": point_a, "point_b": point_b,
               "boot_a_mean": da.mean(), "boot_a_sd": da.std(ddof=1),
               "boot_b_mean": db.mean(), "boot_b_sd": db.std(ddof=1),
               "diff_mean": dd.mean(), "diff_sd": dd.std(ddof=1),
               "ci95": [float(lo), float(hi)], "resolved": bool(not crosses)},
              open(f"{D}/data/fid_bootstrap.json", "w"), indent=1)
    print(f"wrote {D}/data/fid_bootstrap.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
