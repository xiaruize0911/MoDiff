"""Real generation with the K=2 skip2 patch installed, alongside an fp16 reference and the
unpatched int8 baseline, for a genuine quality check (FID) and a real end-to-end timing number --
not just the isolated-kernel benchmark in verify_and_bench.py.

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/generate_samples.py
"""
import os
import sys
import time

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

N_SAMPLES = int(os.environ.get("FID_N", "200"))
STEPS = int(os.environ.get("FID_STEPS", "50"))
BATCH = 16

import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.utils import attention_identity_guard as guard  # noqa: E402
import patch_skip2  # noqa: E402

OUT_ROOT = "docs/ahat_skip2_exact_2026-08-26/fid_run"


def generate(mode, tag, patched):
    guard.seed_model_construction()
    torch.manual_seed(1234)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=OUT_ROOT, batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
    )
    if patched:
        patch_skip2.install()
    t0 = time.time()
    runner.run_mode(mode, num_samples=N_SAMPLES, calibrate=True, force_recalibrate=False)
    elapsed = time.time() - t0
    if patched:
        patch_skip2.report()
        patch_skip2.uninstall()
    src = os.path.join(OUT_ROOT, mode)
    dst = os.path.join(OUT_ROOT, tag)
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    os.rename(src, dst)
    return dst, elapsed


if __name__ == "__main__":
    ref_dir, t_ref = generate("fp16", "fp16_ref", patched=False)
    base_dir, t_base = generate("int8", "int8_baseline", patched=False)
    skip2_dir, t_skip2 = generate("int8", "int8_skip2", patched=True)

    from pytorch_fid import fid_score
    fid_base = fid_score.calculate_fid_given_paths([ref_dir, base_dir], batch_size=50,
                                                   device="cuda", dims=2048)
    fid_skip2 = fid_score.calculate_fid_given_paths([ref_dir, skip2_dir], batch_size=50,
                                                    device="cuda", dims=2048)
    fid_direct = fid_score.calculate_fid_given_paths([base_dir, skip2_dir], batch_size=50,
                                                      device="cuda", dims=2048)

    print(f"\n{'=' * 70}")
    print(f"N={N_SAMPLES} samples, {STEPS} DDIM steps, W8A8+MoDiff")
    print(f"{'=' * 70}")
    print(f"FID(fp16_ref, int8_baseline) = {fid_base:.3f}   generation time: {t_base:.2f}s "
          f"({1000*t_base/N_SAMPLES/STEPS:.4f} ms/sample-step)")
    print(f"FID(fp16_ref, int8_skip2)    = {fid_skip2:.3f}   generation time: {t_skip2:.2f}s "
          f"({1000*t_skip2/N_SAMPLES/STEPS:.4f} ms/sample-step)")
    print(f"FID(baseline, skip2) direct  = {fid_direct:.3f}")
    print(f"wall-clock delta: {t_base - t_skip2:+.3f}s over the whole run "
          f"({1000*(t_base-t_skip2)/N_SAMPLES/STEPS:+.4f} ms/sample-step)")
