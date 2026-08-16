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
from scipy import linalg                                              # noqa: E402
from pytorch_fid.inception import InceptionV3                         # noqa: E402
from PIL import Image                                                 # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"


def features(path, model, batch=64, cache_dir=None):
    """Inception pool3 features for every PNG in `path`, in sorted order so the pairing survives.

    CACHED TO DISK, because extraction -- not the Frechet distance -- turned out to be the second
    bottleneck and the one that survives re-runs. Decoding is PIL single-threaded here at ~46 img/s
    against pytorch_fid's ~256 (it uses a multi-worker DataLoader), so 3 x 10k sets cost ~10 min every
    time. The features of a fixed image set never change, so paying that once per set is the fix;
    speeding up the decode would help the first run only.
    Cache key is the directory path plus file count -- enough to catch "the folder was regenerated with a
    different N", not enough to catch "regenerated with the same N", so it is keyed off mtime too.
    """
    files = sorted(pathlib.Path(path).glob("*.png"))
    assert files, f"no PNGs in {path}"
    ck = None
    if cache_dir:
        import hashlib
        newest = max(f.stat().st_mtime for f in files)
        key = hashlib.md5(f"{os.path.abspath(path)}|{len(files)}|{newest:.0f}".encode()).hexdigest()[:16]
        ck = os.path.join(cache_dir, f"feat_{key}.npy")
        if os.path.exists(ck):
            print(f"  {path}: cached ({len(files)} imgs)")
            return np.load(ck)
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
    if ck:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(ck, out)
    return out


class FidAgainstFixedReference:
    """FID against a FIXED reference, with the reference's matrix square root factored out ONCE.

    WHY NOT pytorch_fid's calculate_frechet_distance: it evaluates tr(sqrtm(S_g @ S_r)) with
    scipy.linalg.sqrtm on the 2048x2048 product -- MEASURED at 124 s per call. A 40-replicate paired
    bootstrap is 80 calls, so ~50 minutes of CPU for a statistic whose replicate spread turned out to be
    0.35 FID. I ran it that way once; this exists so nobody does it twice.

    S_g and S_r are covariances, hence symmetric positive semidefinite. S_g @ S_r is similar to
    S_r^(1/2) @ S_g @ S_r^(1/2), which IS symmetric, so they share eigenvalues and

        tr(sqrtm(S_g @ S_r)) = sum_i sqrt(lambda_i(S_r^(1/2) S_g S_r^(1/2)))

    can be had from `eigh` -- 1-2 orders of magnitude cheaper than a general `sqrtm` -- with S_r^(1/2)
    computed a single time because the reference set is held fixed by design (compute_fid.py's header).
    Eigenvalues are clamped at 0 before the sqrt: they are non-negative in exact arithmetic and land at
    ~-1e-10 in practice.

    Verified against calculate_frechet_distance on the real data this file was written for; the check is
    in the __main__ path so it runs every time rather than being a claim in a comment.
    """

    def __init__(self, mu_r, sigma_r):
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        w, v = linalg.eigh(sigma_r)
        self.sqrt_r = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T

    def __call__(self, feat_gen):
        mu = feat_gen.mean(axis=0)
        sigma = np.cov(feat_gen, rowvar=False)
        m = self.sqrt_r @ sigma @ self.sqrt_r
        lam = linalg.eigh(m, eigvals_only=True)
        tr_covmean = np.sqrt(np.clip(lam, 0.0, None)).sum()
        d = mu - self.mu_r
        return float(d @ d + np.trace(sigma) + np.trace(self.sigma_r) - 2.0 * tr_covmean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="/tmp/claude-0/-workspace/52444f32-4dcb-4c6e-a32f-ebb7833c049f/scratchpad/fid_feat_cache")
    args = ap.parse_args()

    model = InceptionV3([3]).cuda().eval()
    print("extracting features (once per set) ...")
    fr = features(args.real, model, cache_dir=args.cache)
    fa = features(args.a, model, cache_dir=args.cache)
    fb = features(args.b, model, cache_dir=args.cache)
    assert len(fa) == len(fb), f"arms must be paired: {len(fa)} vs {len(fb)}"
    mu_r, sigma_r = fr.mean(axis=0), np.cov(fr, rowvar=False)
    fid = FidAgainstFixedReference(mu_r, sigma_r)

    # Self-check on every run: the fast path must agree with pytorch_fid's own function. Cheap here (two
    # sqrtm calls) and it is what lets the 80 bootstrap calls skip it.
    ref_a = calculate_frechet_distance(fa.mean(axis=0), np.cov(fa, rowvar=False), mu_r, sigma_r)
    point_a, point_b = fid(fa), fid(fb)
    assert abs(point_a - ref_a) < 1e-3, f"eigh path {point_a} vs scipy sqrtm {ref_a}"
    print(f"self-check: eigh path agrees with pytorch_fid to {abs(point_a - ref_a):.2e} FID")
    print(f"\npoint estimates: {args.label_a} {point_a:.3f}   {args.label_b} {point_b:.3f}   "
          f"diff {point_b - point_a:+.3f} ({100 * (point_b - point_a) / point_a:+.2f}%)")

    rng = np.random.default_rng(args.seed)
    n = len(fa)
    da, db, dd = [], [], []
    for r in range(args.reps):
        idx = rng.integers(0, n, size=n)          # ONE index set, applied to both arms
        va, vb = fid(fa[idx]), fid(fb[idx])
        da.append(va); db.append(vb); dd.append(vb - va)
        if True:   # every replicate: a 4-point progress trace in a long loop is unreadable
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
