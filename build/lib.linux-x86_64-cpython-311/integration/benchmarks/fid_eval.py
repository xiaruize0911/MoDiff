#!/usr/bin/env python3
"""
FID evaluation for MoDiff: fp32, int8, int4 vs. real LSUN-Churches-256.

Pipeline:
  1. Generate 2000 samples per mode (fp32 / int8 / int4) via benchmark_ldm.py.
  2. Build a directory of 2000 real LSUN-Churches-256 reference images (256×256 PNG).
     Real data is obtained by one of (in order of preference):
       a. --real_dir   : user-supplied directory already containing PNGs
       b. local lmdb   : LSUN lmdb at --lmdb_path, decoded on-the-fly
       c. download     : fetch LSUN val-set via the official Princeton downloader
  3. Compute FID (pytorch_fid) for each mode against the real reference.
  4. Print a summary table and save results to fid_results.json.

Usage examples:
    # Auto-generate + use existing real-image dir:
    python integration/fid_eval.py --real_dir /path/to/church_real_256

    # Auto-generate + decode from lmdb:
    python integration/fid_eval.py --lmdb_path /data/church_outdoor_train_lmdb

    # Skip generation (images already exist), just compute FID:
    python integration/fid_eval.py --real_dir /path/to/real --skip_generate

    # Only generate, skip FID:
    python integration/fid_eval.py --only_generate
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ─── pytorch_fid ─────────────────────────────────────────────────────────────
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import (
        get_activations,
        calculate_frechet_distance,
        compute_statistics_of_path,
    )
    HAS_FID = True
except ImportError:
    HAS_FID = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_images(path: str) -> int:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sum(
        1 for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts
    )


# ─── Step 1: Generate samples via benchmark_ldm.py ───────────────────────────

def run_benchmark_generation(mode: str, num_samples: int, output_dir: str,
                              steps: int, batch_size: int,
                              bench_script: str,
                              extra_args: list):
    """Call benchmark_ldm.py to generate `num_samples` images for `mode`."""
    mode_dir = os.path.join(output_dir, mode)
    already = count_images(mode_dir) if os.path.isdir(mode_dir) else 0

    if already >= num_samples:
        print(f"  [{mode}] {already} images already present in {mode_dir}, skipping generation.")
        return

    print(f"\n{'='*60}")
    print(f"  Generating {num_samples} samples — mode={mode}")
    print(f"{'='*60}")

    # bench_script is absolute; cwd must be the MoDiff root (two levels up)
    modiff_root = os.path.dirname(os.path.dirname(os.path.abspath(bench_script)))

    cmd = [
        sys.executable, bench_script,
        "--mode", mode,
        "--num_samples", str(num_samples),
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--output_dir", output_dir,
    ] + extra_args

    ret = subprocess.run(cmd, cwd=modiff_root)
    if ret.returncode != 0:
        raise RuntimeError(f"benchmark_ldm.py failed for mode={mode}")

    generated = count_images(mode_dir) if os.path.isdir(mode_dir) else 0
    print(f"  [{mode}] {generated} images saved to {mode_dir}")


# ─── Step 2a: Decode from LSUN lmdb ──────────────────────────────────────────

def extract_real_from_lmdb(lmdb_path: str, out_dir: str, num: int, size: int = 256):
    """Extract `num` centre-cropped, resized images from an LSUN lmdb."""
    try:
        import lmdb
    except ImportError:
        raise RuntimeError("lmdb not installed. Run: pip install lmdb")

    import io
    ensure_dir(out_dir)
    already = count_images(out_dir)
    if already >= num:
        print(f"  [real] {already} images already in {out_dir}, skipping extraction.")
        return

    print(f"  [real] Extracting {num} images from {lmdb_path} → {out_dir} ...")
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True,
                    lock=False, readahead=False, meminit=False)
    transform = _center_crop_resize(size)

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        idx = 0
        for _, imgbuf in tqdm(cursor, total=num, desc="Decoding lmdb"):
            if idx >= num:
                break
            img = Image.open(io.BytesIO(imgbuf)).convert("RGB")
            img = transform(img)
            img.save(os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1

    print(f"  [real] Saved {idx} real images.")


def _center_crop_resize(size: int):
    """Return a callable that centre-crops to square and resizes."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.center_crop(
            img, min(img.size))),  # square crop
        transforms.Resize((size, size), interpolation=Image.LANCZOS),
    ])


# ─── Step 2b: Download LSUN val set via official Princeton script ─────────────

# Train set has ~126k images (2.4 GB compressed); val set has only 300 images.
LSUN_URLS = {
    "church_train": (
        "http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip",
        "church_outdoor_train_lmdb",
    ),
    "church_val": (
        "http://dl.yf.io/lsun/scenes/church_outdoor_val_lmdb.zip",
        "church_outdoor_val_lmdb",
    ),
}


def download_lsun(out_root: str, split: str = "train") -> str:
    """
    Download the LSUN Church lmdb for `split` into `out_root` and return
    the path to the extracted lmdb directory.
    """
    key = f"church_{split}"
    url, folder_name = LSUN_URLS[key]
    zip_path = os.path.join(out_root, f"{folder_name}.zip")
    lmdb_path = os.path.join(out_root, folder_name)

    if os.path.isdir(lmdb_path) and os.path.exists(os.path.join(lmdb_path, "data.mdb")):
        print(f"  [real] LSUN {split} lmdb already at {lmdb_path}")
        return lmdb_path

    ensure_dir(out_root)
    print(f"  [real] Downloading LSUN Church {split} set from {url} ...")
    size_mb = {"train": 2335, "val": 6}.get(split, 0)
    if size_mb:
        print(f"         Size: ~{size_mb} MB")

    class _Progress:
        def __init__(self):
            self.last = 0
        def __call__(self, block, block_size, total):
            done = block * block_size
            if total > 0 and done - self.last > 10_000_000:  # report every 10 MB
                self.last = done
                print(f"    {done/1e6:.0f} / {total/1e6:.0f} MB ({done/total*100:.0f}%)",
                      flush=True)

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_Progress())
        print()
    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(
            f"Failed to download {url}: {e}\n"
            "Please manually supply --lmdb_path or --real_dir."
        )

    import zipfile
    print(f"  [real] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_root)
    os.remove(zip_path)
    print(f"  [real] Extracted to {lmdb_path}")
    return lmdb_path


# ─── Step 3: Compute FID ─────────────────────────────────────────────────────

def compute_fid(gen_dir: str, real_dir: str, batch_size: int = 64,
                device: str = "cuda", dims: int = 2048) -> float:
    """Compute FID between gen_dir and real_dir using pytorch_fid."""
    if not HAS_FID:
        raise RuntimeError("pytorch_fid not installed.")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    m1, s1 = compute_statistics_of_path(real_dir, model, batch_size, dims, device)
    m2, s2 = compute_statistics_of_path(gen_dir,  model, batch_size, dims, device)
    return calculate_frechet_distance(m1, s1, m2, s2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FID evaluation: fp32/int8/int4 vs. real LSUN-Churches")

    # Generation settings
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of images to generate per mode (default: 2000)")
    parser.add_argument("--steps",       type=int, default=200,
                        help="DDIM steps (default: 200, matches benchmark_ldm.py default)")
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Generation batch size")
    parser.add_argument("--modes",       nargs="+", default=["fp32", "int8", "int4"],
                        help="Modes to evaluate (default: fp32 int8 int4)")
    parser.add_argument("--gen_dir",     type=str,
                        default="integration/results_fid_eval",
                        help="Root output directory for generated images")

    # Real data
    real_group = parser.add_mutually_exclusive_group()
    real_group.add_argument("--real_dir",  type=str, default=None,
                             help="Directory of real reference images (256×256 PNG/JPEG)")
    real_group.add_argument("--lmdb_path", type=str, default=None,
                             help="Path to LSUN lmdb (e.g. church_outdoor_train_lmdb)")

    parser.add_argument("--real_size",   type=int, default=256,
                        help="Resize/crop real images to this size (default: 256)")
    parser.add_argument("--real_download_dir", type=str,
                        default="integration/lsun_val",
                        help="Where to download LSUN val if no real_dir/lmdb_path given")

    # Benchmark_ldm.py path
    parser.add_argument("--bench_script", type=str,
                        default="integration/benchmark_ldm.py",
                        help="Path to benchmark_ldm.py")
    parser.add_argument("--bench_extra",  nargs="*", default=[],
                        help="Extra args forwarded to benchmark_ldm.py (e.g. --skip_calibration)")

    # Control flow
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip generation, use existing images in --gen_dir")
    parser.add_argument("--only_generate", action="store_true",
                        help="Only generate images, skip FID computation")
    parser.add_argument("--fid_batch_size", type=int, default=64,
                        help="Batch size for Inception feature extraction")
    parser.add_argument("--output_json",    type=str,
                        default="integration/results_fid_eval/fid_results.json",
                        help="Where to save FID results JSON")

    args = parser.parse_args()

    # ── Resolve bench_script to absolute path ──────────────────────────────
    bench_abs = os.path.abspath(args.bench_script)
    if not os.path.isfile(bench_abs):
        sys.exit(f"ERROR: benchmark script not found: {bench_abs}")

    # Always run benchmark_ldm.py from the MoDiff project root
    modiff_root = os.path.dirname(os.path.dirname(bench_abs))  # …/MoDiff
    os.chdir(modiff_root)

    gen_root = os.path.abspath(args.gen_dir)
    ensure_dir(gen_root)

    # ── Step 1: Generate ───────────────────────────────────────────────────
    if not args.skip_generate:
        for mode in args.modes:
            run_benchmark_generation(
                mode=mode,
                num_samples=args.num_samples,
                output_dir=gen_root,
                steps=args.steps,
                batch_size=args.batch_size,
                bench_script=bench_abs,
                extra_args=args.bench_extra,
            )
    else:
        print("Skipping generation (--skip_generate).")

    if args.only_generate:
        print("Done generating. Exiting (--only_generate).")
        return

    # ── Step 2: Prepare real reference ────────────────────────────────────
    if not HAS_FID:
        sys.exit("ERROR: pytorch_fid not installed. Run: pip install pytorch-fid")

    real_ref_dir = None

    if args.real_dir:
        real_ref_dir = os.path.abspath(args.real_dir)
        n = count_images(real_ref_dir)
        if n < args.num_samples:
            print(f"WARNING: --real_dir has only {n} images; need {args.num_samples}. "
                  "FID may be less reliable.")
        else:
            print(f"Using {n} real images from {real_ref_dir}")

    elif args.lmdb_path:
        real_ref_dir = os.path.join(gen_root, "real_reference")
        extract_real_from_lmdb(
            lmdb_path=os.path.abspath(args.lmdb_path),
            out_dir=real_ref_dir,
            num=args.num_samples,
            size=args.real_size,
        )

    else:
        # Auto-download LSUN Church train lmdb (~2.4 GB, 126k images)
        print("\nNo --real_dir or --lmdb_path specified.")
        print("Auto-downloading LSUN Church train lmdb (~2.4 GB) ...")
        download_root = os.path.abspath(args.real_download_dir)
        try:
            lmdb_path = download_lsun(
                out_root=download_root, split="train"
            )
        except RuntimeError as e:
            print(f"\nWARNING: Train download failed: {e}")
            print("Falling back to val set (~6 MB, 300 images) ...")
            try:
                lmdb_path = download_lsun(
                    out_root=download_root, split="val"
                )
            except RuntimeError as e2:
                print(f"\nERROR: {e2}")
                print(
                    "\nPlease supply:\n"
                    "  --real_dir /path/to/real_churches_256\n"
                    "  --lmdb_path /path/to/church_outdoor_train_lmdb"
                )
                sys.exit(1)

        real_ref_dir = os.path.join(download_root, "real_reference_256")
        extract_real_from_lmdb(
            lmdb_path=lmdb_path,
            out_dir=real_ref_dir,
            num=args.num_samples,
            size=args.real_size,
        )

    # ── Step 3: Compute FID ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print(f"\n{'='*60}")
    print(f"Computing FID  (device={device}, dims=2048, fid_batch={args.fid_batch_size})")
    print(f"Real reference : {real_ref_dir}  ({count_images(real_ref_dir)} images)")
    print(f"{'='*60}")

    for mode in args.modes:
        mode_dir = os.path.join(gen_root, mode)
        n_gen = count_images(mode_dir) if os.path.isdir(mode_dir) else 0
        if n_gen == 0:
            print(f"  [{mode}] No generated images found in {mode_dir}, skipping.")
            continue

        print(f"\n  [{mode}] {n_gen} images — computing FID vs real ...", flush=True)
        t0 = time.time()
        fid = compute_fid(
            gen_dir=mode_dir,
            real_dir=real_ref_dir,
            batch_size=args.fid_batch_size,
            device=device,
        )
        elapsed = time.time() - t0
        results[mode] = {"fid": fid, "num_samples": n_gen, "elapsed_s": elapsed}
        print(f"  [{mode}] FID = {fid:.2f}  ({elapsed:.1f}s)")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Mode':<12} {'# Samples':>10} {'FID':>8}")
    print("-" * 34)
    for mode in args.modes:
        if mode in results:
            r = results[mode]
            print(f"{mode:<12} {r['num_samples']:>10} {r['fid']:>8.2f}")
        else:
            print(f"{mode:<12} {'N/A':>10} {'N/A':>8}")
    print(f"{'='*60}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_json = os.path.abspath(args.output_json)
    ensure_dir(os.path.dirname(out_json))
    with open(out_json, "w") as f:
        json.dump({
            "settings": {
                "num_samples": args.num_samples,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "modes": args.modes,
                "real_ref_dir": real_ref_dir,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
