"""Generate N decoded 256x256 samples per mode for FID.

Every accuracy figure in this project so far is latent relative L2. It orders the modes consistently
with the decoded samples, but it is not perceptual and it cannot be compared to the paper, whose
claim is FID. This produces the images FID needs.

Four things are load-bearing and easy to get wrong:

  * REAL-checkpoint calibration (integration/calibration/*_realckpt.pt). The un-suffixed artifacts
    were fitted against the 856-byte stub's random weights and give latent relL2 0.88 (W8A8) /
    3.02 (W4A4) with real weights. Samples generated on them would look broken for a reason that has
    nothing to do with the mode under test.
  * one warm-up sampling run per mode, DISCARDED. The quantized attention blocks self-calibrate over
    their first forwards; run 1 is several x worse than run 2. Generating from a cold model would
    penalise every quantized mode.
  * a DIFFERENT seed per batch. FID measures a distribution. Reusing one seed across batches would
    produce N/batch copies of the same 128 images and a meaningless covariance.
  * the SAME seed sequence across modes, so mode-to-mode differences are the mode and not the noise.
    (This makes the FID comparison paired, which is strictly better here than independent draws.)

Preprocessing matches export_lsun_reference.py exactly -- 256x256, lossless PNG. FID compares two
Inception feature distributions, so any preprocessing asymmetry between real and generated is
measured as distance.
"""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                              # noqa: E402
from PIL import Image                                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--out", default="/workspace/fid")
ap.add_argument("--modes", default="fp16,int8_baseline,int8,int4_baseline,int4")
ap.add_argument("--seed0", type=int, default=20260805)
ap.add_argument("--decode-chunk", type=int, default=32,
                help="images per decode_first_stage call; bounds the VAE activation peak")
a = ap.parse_args()

CALIB = {"int8": "integration/calibration/int8_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}
#: mode key -> (folder name, delta mode). The MoDiff modes ship dynamic; baselines have no delta.
SPEC = {"fp16": ("fp16", "static"),
        "int8_baseline": ("int8_baseline", "static"),
        "int8": ("int8_modiff", "dynamic"),
        "int4_baseline": ("int4_baseline", "static"),
        "int4": ("int4_modiff", "dynamic")}


def build(mode, delta_mode):
    import integration.benchmarks.benchmark_ldm as B
    import kernel_suites_bench as ks
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    os.environ["MODIFF_DELTA_REFRESH"] = "4"
    os.environ["MODIFF_DELTA_CLIP"] = "1.0"
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_LINEAR"] = "0"
    calib = None if mode == "fp16" else CALIB["int4" if "int4" in mode else "int8"]
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{a.out}/tmp_out", batch_size=a.batch, steps=a.steps,
        shape=(4, 32, 32), calibration_path=calib)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def reset(model):
    from integration.kernels.int4_optimized import reset_modiff_state as r4
    from integration.kernels.int8_optimized import reset_modiff_state as r8
    for r in (r8, r4):
        try:
            r(model.model.diffusion_model)
        except Exception:
            pass


def sample_batch(runner, model, sampler, n, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=a.steps, batch_size=n, shape=runner.shape, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
        lat = lat.to("cuda", torch.float16)
        # Decode in chunks. The VAE upsamples 32x32 latents to 256x256 RGB, so one 128-image decode
        # asks for an 8 GiB activation in a single allocation -- it OOM'd against a second process
        # holding 1.8 GiB, having already produced 10k images in two other modes. Chunking bounds
        # the peak without changing the output: the decoder is per-sample, so chunk boundaries do
        # not affect any pixel.
        chunks = []
        for i in range(0, lat.shape[0], a.decode_chunk):
            d = model.decode_first_stage(lat[i:i + a.decode_chunk])
            chunks.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0)
                          .permute(0, 2, 3, 1).cpu())
            del d
        img = torch.cat(chunks, 0)
    return (img.numpy() * 255).round().astype("uint8")


def main():
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        folder, dm = SPEC[mode]
        d = os.path.join(a.out, folder)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".png")])
        if have >= a.n:
            print(f"[{folder}] already has {have} images, skipping", flush=True)
            continue
        print(f"=== {mode} -> {d}  (delta={dm}, {a.n} images, {a.steps} steps)", flush=True)
        runner, model, sampler = build(mode, dm)

        reset(model)
        sample_batch(runner, model, sampler, min(a.batch, 16), a.seed0 - 1)   # warm-up, discarded

        t0, written = time.time(), 0
        bi = 0
        while written < a.n:
            k = min(a.batch, a.n - written)
            reset(model)
            arr = sample_batch(runner, model, sampler, k, a.seed0 + bi)
            for i in range(arr.shape[0]):
                Image.fromarray(arr[i]).save(os.path.join(d, f"{written + i:06d}.png"), "PNG")
            written += arr.shape[0]
            bi += 1
            if bi % 10 == 0 or written >= a.n:
                el = time.time() - t0
                print(f"  {written}/{a.n}  {el:.0f}s elapsed, "
                      f"{el / max(written, 1) * (a.n - written):.0f}s left", flush=True)
        del model, sampler, runner
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
