"""Attempt a generation-quality evaluation, and show exactly why this tree cannot produce one.

This runs the evaluation rather than asserting the conclusion, because "we can't measure quality
here" is the kind of claim that should come with evidence. Three checks:

  1. Is the UNet's epsilon prediction nonzero?  (if not, sampling ignores the whole network)
  2. Do the five modes produce different latents from the same seed?  (if not, no metric can
     separate them -- FID included)
  3. What quality evidence IS valid?  Kernel-level accuracy against fp32/fp64 references, which
     needs no checkpoint and is what the test suite already measures.

Background: docs/gn_qkv_fusion_2026-08-03/FINDINGS.md section 5.
"""

import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from integration.utils import attention_identity_guard as guard

BATCH = int(os.environ.get("Q_BATCH", "4"))
STEPS = int(os.environ.get("Q_STEPS", "20"))
SEED = 1234
MODES = ["fp16", "int8_baseline", "int8", "int4_baseline", "int4"]
CALIB = {"int4_baseline": "integration/calibration/int4_calibration.pt",
         "int4": "integration/calibration/int4_calibration.pt"}
DEFAULT_CALIB = "integration/calibration/int8_calibration.pt"


def sample_mode(mode):
    """(latent, eps_absmax, n_identity_attn) for `mode`, with the model seeded reproducibly."""
    ks.set_env(mode)
    guard.seed_model_construction()          # otherwise every process builds a different random net
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=(None if mode == "fp16" else CALIB.get(mode, DEFAULT_CALIB)))
    model, sampler = runner._setup_model(mode)
    unet = model.model.diffusion_model

    # Check 1: does the network produce any signal at all?
    x = torch.randn(2, 4, 32, 32, device="cuda", dtype=torch.float16)
    t = torch.full((2,), 500, device="cuda", dtype=torch.long)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        eps = unet(x, t)
    eps_absmax = float(eps.abs().max())
    n_identity = len(guard.find_identity_attention_blocks(model))

    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                            verbose=False, **cond)
    lat = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().cpu()
    del model, sampler, runner
    torch.cuda.empty_cache()
    return lat, eps_absmax, n_identity


def main():
    res = {}
    print("=" * 78)
    print("CHECK 1+2: does the network contribute anything, and do the modes differ?")
    print("=" * 78)
    lats = {}
    for m in MODES:
        lat, eps, n_id = sample_mode(m)
        lats[m] = lat
        res[m] = {"eps_absmax": eps, "identity_attn_blocks": n_id,
                  "latent_absmax": float(lat.abs().max())}
        print(f"  {m:16s} UNet |eps|max = {eps:.6g}   identity attn blocks = {n_id}/21   "
              f"latent |x|max = {float(lat.abs().max()):.4g}")

    print()
    print("=" * 78)
    print("Pairwise latent comparison, same seed (any metric can only see differences here)")
    print("=" * 78)
    pairs = {}
    for a, b in itertools.combinations(MODES, 2):
        same = bool(torch.equal(lats[a], lats[b]))
        rel = float((lats[a] - lats[b]).norm() / lats[b].norm().clamp_min(1e-12))
        pairs[f"{a}|{b}"] = {"bit_identical": same, "rel_l2": rel}
        print(f"  {a:16s} vs {b:16s} bit-identical={same}  relL2={rel:.3e}")

    allsame = all(v["bit_identical"] for v in pairs.values())
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if allsame:
        print("  Every mode produced a BIT-IDENTICAL latent. No image-quality metric -- FID, IS,")
        print("  sFID, LPIPS, anything -- can distinguish these modes, because the quantity they")
        print("  all consume is the same bytes. Any FID reported from this tree would be one")
        print("  number describing the initial noise, not the model.")
        print("  Cause: models/ldm/lsun_churches256/model.ckpt is an 856-byte stub with an empty")
        print("  state_dict, and UNetModel.out[-1] is a zero_module -- so eps == 0 above.")
        print("  To evaluate quality you need the real trained LSUN-churches LDM checkpoint.")
    else:
        print("  Modes differ, so a quality metric is meaningful in principle. Note it still would")
        print("  not be a QUALITY number unless the checkpoint is trained -- differences between")
        print("  random networks measure numeric agreement, not image fidelity.")

    with open("docs/modiff_correctness_2026-08-03/data/quality_evaluation.json", "w") as f:
        json.dump({"modes": res, "pairs": pairs, "all_bit_identical": allsame,
                   "batch": BATCH, "steps": STEPS, "seed": SEED}, f, indent=2)


if __name__ == "__main__":
    main()
