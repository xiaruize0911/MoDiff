"""Actually try quantizing a_hat to int8 for W8A8, and measure the FID effect -- moving past proxy
math (C15) to a real measurement, at the project owner's request.

METHOD: fake quantization. Patches OptimizedInt8Conv2d's three methods that write
self.a_hat_cache (`forward`, `forward_gn_fused_modiff`, `forward_modiff_fused_silu_residual` --
found by grepping every method touching `self.a_hat_cache`) so that immediately after each real
call returns, a_hat_cache is rounded to an int8 grid and immediately dequantized back to fp16.
This simulates exactly the precision loss int8 storage would introduce, without needing to change
any CUDA kernel or its dtype -- the tensor stays fp16-shaped, but its VALUES are what an int8
buffer would actually hold. This measures the QUALITY question only; the bandwidth question
(2.024 ms/step ceiling) was already measured directly in ahat_overlap_2026-08-26.

Quantization range: a single fixed range across all layers for this first pass (not per-layer
calibrated) -- see AHAT_RANGE below. Generates three matched sample sets with the same seeds
(fp16 reference, int8+MoDiff baseline, int8+MoDiff with a_hat fake-quantized to int8) and reports
FID of each MoDiff arm against the fp16 reference.

Run: python docs/int8_ahat_cache_2026-08-26/scripts/fid_ahat_int8.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")

import torch  # noqa: E402

AHAT_RANGE = 8.0   # fixed +-range for the int8 grid; real captured a_hat max was 11.05, p99.9 2.48
N_SAMPLES = int(os.environ.get("FID_N", "200"))
STEPS = int(os.environ.get("FID_STEPS", "50"))
BATCH = 16

import integration.kernels.int8_optimized as i8opt  # noqa: E402

_patched = {"on": False}


def fake_quantize_(t, rng=AHAT_RANGE):
    lsb = rng / 127.0
    q = torch.clamp(torch.round(t.float() / lsb), -127, 127)
    t.copy_((q * lsb).to(t.dtype))


def _make_wrapper(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if _patched["on"] and getattr(self, "a_hat_cache", None) is not None \
                and self.a_hat_cache.numel() > 0:
            fake_quantize_(self.a_hat_cache)
        return out
    return wrapped


_Cls = i8opt.OptimizedInt8Conv2d
_ORIGINALS = {}
for _name in ("forward", "forward_gn_fused_modiff", "forward_modiff_fused_silu_residual"):
    _ORIGINALS[_name] = getattr(_Cls, _name)
    setattr(_Cls, _name, _make_wrapper(_ORIGINALS[_name]))


import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.utils import attention_identity_guard as guard  # noqa: E402

OUT_ROOT = "docs/int8_ahat_cache_2026-08-26/fid_run"


def generate(mode, tag, ahat_quant):
    guard.seed_model_construction()
    torch.manual_seed(1234)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=OUT_ROOT, batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
    )
    _patched["on"] = ahat_quant
    print(f"\n=== generating {tag} (mode={mode}, a_hat int8={ahat_quant}, n={N_SAMPLES}) ===")
    runner.run_mode(mode, num_samples=N_SAMPLES, calibrate=True, force_recalibrate=False)
    _patched["on"] = False
    src = os.path.join(OUT_ROOT, mode)
    dst = os.path.join(OUT_ROOT, tag)
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    os.rename(src, dst)
    return dst


if __name__ == "__main__":
    ref_dir = generate("fp16", "fp16_ref", ahat_quant=False)
    base_dir = generate("int8", "int8_modiff_baseline", ahat_quant=False)
    quant_dir = generate("int8", "int8_modiff_ahat_int8", ahat_quant=True)

    from pytorch_fid import fid_score
    fid_base = fid_score.calculate_fid_given_paths([ref_dir, base_dir], batch_size=50,
                                                   device="cuda", dims=2048)
    fid_quant = fid_score.calculate_fid_given_paths([ref_dir, quant_dir], batch_size=50,
                                                    device="cuda", dims=2048)
    fid_between = fid_score.calculate_fid_given_paths([base_dir, quant_dir], batch_size=50,
                                                      device="cuda", dims=2048)

    print(f"\n{'=' * 60}")
    print(f"N={N_SAMPLES} samples, {STEPS} DDIM steps, W8A8+MoDiff, a_hat range=+-{AHAT_RANGE}")
    print(f"{'=' * 60}")
    print(f"FID(fp16_ref, int8_modiff_baseline)   = {fid_base:.3f}")
    print(f"FID(fp16_ref, int8_modiff_ahat_int8)  = {fid_quant:.3f}")
    print(f"FID(baseline, ahat_int8)  (direct)     = {fid_between:.3f}")
    print(f"delta vs baseline: {fid_quant - fid_base:+.3f} ({100*(fid_quant-fid_base)/fid_base:+.1f}%)")
