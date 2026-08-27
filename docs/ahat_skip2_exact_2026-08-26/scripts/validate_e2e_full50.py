"""End-to-end correctness gate for the K=2 deferred-write patch: run a SMALL real generation twice
with the SAME seed -- once with patch_skip2 installed, once without -- and compare the raw output
tensor (before any PNG encoding) bit-for-bit. Also runs the unpatched path TWICE to establish
whether the pipeline is deterministic at all independent of this patch (cuDNN algorithm selection
can be a source of run-to-run noise unrelated to the change under test).

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/validate_e2e.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.utils import attention_identity_guard as guard  # noqa: E402
import patch_skip2  # noqa: E402

# benchmark_ldm sets cudnn.benchmark=True at import time, which autotunes conv algorithms with a
# timing heuristic that is NOT run-to-run deterministic on its own -- confirmed by the first
# version of this script (baseline vs baseline differed on 60% of pixels with NO patch involved).
# Force it off here so this gate isolates the patch's effect from that unrelated noise source.
torch.backends.cudnn.benchmark = False

BATCH, STEPS, N_SAMPLES = 8, 50, 8
OUT_ROOT = "docs/ahat_skip2_exact_2026-08-26/e2e_check"


def run(tag):
    guard.seed_model_construction()
    torch.manual_seed(777)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=OUT_ROOT, batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
    )
    runner.run_mode("int8", num_samples=N_SAMPLES, calibrate=True, force_recalibrate=False)
    import shutil
    src = os.path.join(OUT_ROOT, "int8")
    dst = os.path.join(OUT_ROOT, tag)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)
    imgs = []
    from PIL import Image
    import numpy as np
    for i in range(N_SAMPLES):
        imgs.append(np.array(Image.open(os.path.join(dst, f"{i:05d}.png"))))
    return np.stack(imgs)


print("=== Run 1: baseline (unpatched) ===")
out_baseline1 = run("baseline1")

print("\n=== Run 2: baseline again (unpatched) -- checks pipeline determinism ===")
out_baseline2 = run("baseline2")

det_exact = (out_baseline1 == out_baseline2).all()
print(f"\nbaseline run1 == baseline run2 (pipeline determinism, no patch involved): {det_exact}")
if not det_exact:
    diff = (out_baseline1.astype(int) - out_baseline2.astype(int))
    print(f"  max abs pixel diff: {abs(diff).max()}, fraction of pixels differing: "
          f"{(diff != 0).mean():.6f}")

print("\n=== Run 3: with skip2 patch installed ===")
patch_skip2.install()
out_skip2 = run("skip2")
patch_skip2.report()
patch_skip2.uninstall()

exact = (out_baseline1 == out_skip2).all()
print(f"\nbaseline == skip2 (the actual thing being tested): {exact}")
if not exact:
    diff = (out_baseline1.astype(int) - out_skip2.astype(int))
    print(f"  max abs pixel diff: {abs(diff).max()}, fraction of pixels differing: "
          f"{(diff != 0).mean():.6f}")
    if det_exact:
        print("  (baseline was deterministic run-to-run, so this diff is real and caused by the patch)")
    else:
        print("  (baseline itself was NOT deterministic run-to-run -- this diff may not be "
              "attributable to the patch; investigate the nondeterminism source first)")
