"""FID on the real checkpoint: extract LSUN reference images, generate per mode, compute FID.

Requires the REAL checkpoint (see FINDINGS 2026-08-04) and the recalibrated
`integration/calibration/*_realckpt.pt` -- the shipped calibration files were produced against the
stub's random weights and give latent relL2 0.88 (int8) / 3.02 (int4) with real weights, so FID on
them would be a meaningless-but-plausible number.

Stages, each skippable so a long run can resume:
  --reference   decode N images out of the LSUN church_outdoor LMDB to PNG
  --generate    sample latents per mode, VAE-decode, save PNG
  --fid         pytorch_fid between the reference dir and each mode's dir

N is reported explicitly. FID is biased upward at small N; the bias is common to every mode, so
mode-to-mode comparison is valid while the absolute value is not comparable to a published FID-50k.
"""

import argparse
import io
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch
from PIL import Image

OUTDIR = "/workspace/fid_run"
LMDB_DIR = "/workspace/lsun_dl/church_outdoor_train_lmdb"
MODES = ["fp16", "int8_baseline", "int8", "int4_baseline", "int4"]
CALIB = {"int4_baseline": "integration/calibration/int4_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}
DEFAULT_CALIB = "integration/calibration/int8_calibration_realckpt.pt"


def extract_reference(n, size=256):
    import lmdb
    d = os.path.join(OUTDIR, "reference")
    os.makedirs(d, exist_ok=True)
    have = len([f for f in os.listdir(d) if f.endswith(".png")])
    if have >= n:
        print(f"  reference: {have} images already present")
        return d
    env = lmdb.open(LMDB_DIR, readonly=True, lock=False, readahead=False, meminit=False)
    i = 0
    with env.begin(write=False) as txn:
        for _, val in txn.cursor():
            if i >= n:
                break
            img = Image.open(io.BytesIO(val)).convert("RGB")
            # LSUN images vary in size; centre-crop the short side then resize, the standard
            # LSUN-256 preprocessing.
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            img.resize((size, size), Image.BICUBIC).save(os.path.join(d, f"{i:06d}.png"))
            i += 1
            if i % 1000 == 0:
                print(f"    reference {i}/{n}", flush=True)
    print(f"  reference: wrote {i} images -> {d}")
    return d


def generate(mode, n, steps, batch, seed=1234):
    import integration.benchmarks.benchmark_ldm as B
    import kernel_suites_bench as ks
    d = os.path.join(OUTDIR, mode)
    os.makedirs(d, exist_ok=True)
    have = len([f for f in os.listdir(d) if f.endswith(".png")])
    if have >= n:
        print(f"  {mode}: {have} images already present")
        return d

    ks.set_env(mode)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUTDIR, "tmp"), batch_size=batch, steps=steps,
        shape=(4, 32, 32),
        calibration_path=(None if mode == "fp16" else CALIB.get(mode, DEFAULT_CALIB)))
    model, sampler = runner._setup_model(mode)
    cond = runner._cond_kwargs(model, batch)

    from integration.kernels.int8_optimized import reset_modiff_state as r8
    from integration.kernels.int4_optimized import reset_modiff_state as r4

    idx = have
    torch.manual_seed(seed + 1); torch.cuda.manual_seed_all(seed + 1)
    while idx < n:
        for r in (r8, r4):
            try:
                r(model.model.diffusion_model)
            except Exception:
                pass
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            out = sampler.sample(S=steps, batch_size=batch, shape=runner.shape, eta=0.0,
                                verbose=False, **cond)
            lat = out[0] if isinstance(out, (tuple, list)) else out
            img = model.decode_first_stage(lat)
        img = torch.clamp((img.float() + 1.0) / 2.0, 0.0, 1.0)
        arr = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
        for k in range(arr.shape[0]):
            if idx >= n:
                break
            Image.fromarray(arr[k]).save(os.path.join(d, f"{idx:06d}.png"))
            idx += 1
        print(f"    {mode} {idx}/{n}", flush=True)
    del model, sampler, runner
    torch.cuda.empty_cache()
    return d


def fid(ref_dir, gen_dir, batch=50):
    r = subprocess.run([sys.executable, "-m", "pytorch_fid", ref_dir, gen_dir,
                        "--batch-size", str(batch), "--device", "cuda"],
                       capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if "FID" in line:
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    print(r.stdout[-500:], r.stderr[-500:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--fid", action="store_true")
    a = ap.parse_args()
    if not (a.reference or a.generate or a.fid):
        a.reference = a.generate = a.fid = True
    os.makedirs(OUTDIR, exist_ok=True)

    if a.reference:
        print("== reference ==", flush=True)
        extract_reference(a.n)
    if a.generate:
        print(f"== generate: {a.n} images/mode, {a.steps} steps, batch {a.batch} ==", flush=True)
        for m in MODES:
            generate(m, a.n, a.steps, a.batch)
    if a.fid:
        print(f"== FID (N={a.n} per side; biased upward at this N, consistent across modes) ==")
        res = {}
        ref = os.path.join(OUTDIR, "reference")
        for m in MODES:
            v = fid(ref, os.path.join(OUTDIR, m))
            res[m] = v
            print(f"  {m:16s} FID = {v}", flush=True)
        with open("docs/modiff_correctness_2026-08-03/data/fid.json", "w") as f:
            json.dump({"n": a.n, "steps": a.steps, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
