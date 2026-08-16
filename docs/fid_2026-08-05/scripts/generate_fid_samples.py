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
from integration.utils.preflight import preflight, MODEL                  # noqa: E402
#: 10k x 50 DDIM steps is ~15 min per mode; a missing import found at minute 14 costs the whole run.
preflight(*MODEL, what="generate_fid_samples.py")
from PIL import Image                                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--out", default="/workspace/fid")
ap.add_argument("--modes", default="fp16,int8_baseline,int8,int4_baseline,int4")
#: the experiment surface: every precision as PTQ baseline, conv-only MoDiff (_l0) and
#: conv+projection MoDiff (_l1). Pass --modes all for it.
ap.add_argument("--seed0", type=int, default=20260805)
ap.add_argument("--linear", type=int, default=0, choices=(0, 1),
                help="MODIFF_LINEAR for the MoDiff arms: whether the 42 attention projections carry "
                     "a_hat/o_hat too. 0 (default) is conv-only MoDiff and reproduces the 2026-08-05 "
                     "run; 1 is the datapath that became the default on 2026-08-06. Baselines and "
                     "fp16 always get 0 -- temporal state in a PTQ arm would stop it being one.")
ap.add_argument("--decode-chunk", type=int, default=32,
                help="images per decode_first_stage call; bounds the VAE activation peak")
a = ap.parse_args()

#: The 2026-08-05 run used the absmax files, and every FID number committed from it is keyed to
#: them. FID_CALIB8 / FID_CALIB4 let a later run score a different calibration WITHOUT silently
#: changing what those committed numbers mean -- e.g. the Q-Diffusion export
#: (integration/calibration/int8_calibration_qdiff.pt), which improved baseline latent relL2 2.29x
#: and whose FID consequence is untested. Defaults unchanged on purpose.
CALIB = {"int8": os.environ.get("FID_CALIB8",
                                "integration/calibration/int8_calibration_realckpt.pt"),
         "int4": os.environ.get("FID_CALIB4",
                                "integration/calibration/int4_calibration_realckpt.pt")}
#: mode key -> (folder name, delta mode, activation bits). The MoDiff modes ship dynamic; baselines
#: have no delta. Activation bits added 2026-08-10: 8 is what every original entry ran at, so those
#: rows are unchanged, and w8a4* are the new W8A4 configuration -- the paper's own claim, which had
#: no FID row because MODIFF_ACT_Q was a sweep knob rather than a mode. W4A4's 4 bits come from its
#: int4 datapath, not from this field, so int4* stay 8 here (Int4Conv ignores it).
#: The 4th field is MODIFF_LINEAR: whether the 42 attention projections carry a_hat/o_hat too.
#: None means "take it from --linear", which is what the legacy keys did.
#:
#: The _l0 / _l1 pairs are FIRST-CLASS MODES, not a flag, because conv-only and conv+projection
#: MoDiff are two different methods rather than one method with a tuning knob -- and measured
#: 2026-08-10 they are not even close at W4A4, where L1 recovers structure L0 loses entirely
#: (cross-batch mean|delta| 16.7/255 against a 0.45 pipeline noise floor) while being visually
#: indistinguishable at W8A8 and W8A4. Separate folders so both can be reviewed side by side and
#: neither overwrites the other. The un-suffixed legacy keys are kept so the 2026-08-05 FID run and
#: the /workspace/fid folders it produced still reproduce byte for byte.
SPEC = {"fp16": ("fp16", "static", 8, 0),
        "int8_baseline": ("int8_baseline", "static", 8, 0),
        "int8_l0": ("int8_modiff_l0", "dynamic", 8, 0),
        "int8_l1": ("int8_modiff_l1", "dynamic", 8, 1),
        "w8a4_baseline": ("w8a4_baseline", "static", 4, 0),
        "w8a4_l0": ("w8a4_modiff_l0", "dynamic", 4, 0),
        "w8a4_l1": ("w8a4_modiff_l1", "dynamic", 4, 1),
        "int4_baseline": ("int4_baseline", "static", 8, 0),
        "int4_l0": ("int4_modiff_l0", "dynamic", 8, 0),
        "int4_l1": ("int4_modiff_l1", "dynamic", 8, 1),
        # legacy keys: folder and behaviour exactly as before, --linear still applies
        "int8": ("int8_modiff", "dynamic", 8, None),
        "w8a4": ("w8a4_modiff", "dynamic", 4, None),
        "int4": ("int4_modiff", "dynamic", 8, None)}
#: w8a4* run the int8 datapath; only the activation bit-width differs.
BASE_MODE = {"w8a4_baseline": "int8_baseline", "w8a4": "int8",
             "w8a4_l0": "int8", "w8a4_l1": "int8",
             "int8_l0": "int8", "int8_l1": "int8",
             "int4_l0": "int4", "int4_l1": "int4"}


def build(mode, delta_mode, act_bits=8, linear=None):
    import integration.benchmarks.benchmark_ldm as B
    import kernel_suites_bench as ks
    base = BASE_MODE.get(mode, mode)
    ks.set_env(base)
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    os.environ["MODIFF_DELTA_REFRESH"] = "4"
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    # MODIFF_LINEAR follows the arm, not the flag alone. --linear 1 asks for MoDiff on the 42
    # attention projections, which only means anything where MoDiff is on at all: giving a PTQ
    # baseline an a_hat/o_hat cache would stop it being a baseline, so the comparison would no
    # longer be MoDiff-vs-PTQ. Tied to delta_mode for that reason.
    want_lin = a.linear if linear is None else linear
    os.environ["MODIFF_LINEAR"] = "1" if (want_lin and delta_mode == "dynamic") else "0"
    # MODIFF_ACT_BITS replaced MODIFF_ACT_Q on 2026-08-10 and accepts only 8 or 4. MODIFF_DELTA_CLIP
    # was retired in the same pass and now RAISES on anything but 1.0, so it is no longer set here.
    os.environ["MODIFF_ACT_BITS"] = str(act_bits)
    os.environ.pop("MODIFF_DELTA_CLIP", None)
    calib = None if base == "fp16" else CALIB["int4" if "int4" in base else "int8"]
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{a.out}/tmp_out", batch_size=a.batch, steps=a.steps,
        shape=(4, 32, 32), calibration_path=calib)
    model, sampler = runner._setup_model(base)
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
    ALL = ("fp16,int8_baseline,int8_l0,int8_l1,w8a4_baseline,w8a4_l0,w8a4_l1,"
           "int4_baseline,int4_l0,int4_l1")
    modes = ALL if a.modes.strip() == "all" else a.modes
    for mode in [m.strip() for m in modes.split(",") if m.strip()]:
        folder, dm, ab, lin = SPEC[mode]
        d = os.path.join(a.out, folder)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".png")])
        if have >= a.n:
            print(f"[{folder}] already has {have} images, skipping", flush=True)
            continue
        print(f"=== {mode} -> {d}  (delta={dm}, A{ab}, LINEAR={lin if lin is not None else a.linear}, "
              f"{a.n} images, {a.steps} steps)",
              flush=True)
        runner, model, sampler = build(mode, dm, ab, lin)

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
