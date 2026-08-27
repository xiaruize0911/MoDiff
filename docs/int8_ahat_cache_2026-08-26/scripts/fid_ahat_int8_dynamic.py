"""Dynamic (per-tensor, on-the-fly) int8 quantization of a_hat -- no separate calibration pass at
all, to remove any possibility that the calibration script in fid_ahat_int8_calibrated.py (a
separate run on a different seed, table lookup by (layer_name, step_count)) has a bug or a
distribution mismatch versus the eval run. This is the cleanest possible test of "can a_hat live
in 8 bits": at every single write, the quantization range is set to that EXACT tensor's own
abs().max() at that EXACT moment -- by construction this can never clip and always uses the full
8-bit code range for whatever the real distribution is, for every single call, with zero
calibration/lookup/margin logic to get wrong. If this still degrades FID sharply, the conclusion
is not an artifact of any calibration methodology -- static OR dynamic -- it is the fundamental
bit-budget shortfall from the C15 analysis (re-expressing a_hat's absolute value onto ANY grid,
however chosen, introduces a rounding term o_hat's accumulation cannot cancel).

Run: python docs/int8_ahat_cache_2026-08-26/scripts/fid_ahat_int8_dynamic.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")

import torch  # noqa: E402

N_SAMPLES = int(os.environ.get("FID_N", "200"))
STEPS = int(os.environ.get("FID_STEPS", "50"))
BATCH = 16

import integration.kernels.int8_optimized as i8opt  # noqa: E402

_patched = {"on": False}
_stats = {"clipped_calls": 0, "total_calls": 0}


def fake_quantize_dynamic_(t):
    m = float(t.detach().abs().max())
    _stats["total_calls"] += 1
    if m < 1e-8:
        return  # all-zero tensor (e.g. a_hat_cache before its first write) -- nothing to do
    lsb = m / 127.0
    q = torch.round(t.float() / lsb)
    # by construction |t| <= m so q is always in [-127, 127] -- clamp only guards fp rounding
    q = torch.clamp(q, -127, 127)
    t.copy_((q * lsb).to(t.dtype))


def _make_wrapper(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if _patched["on"] and getattr(self, "a_hat_cache", None) is not None \
                and self.a_hat_cache.numel() > 0:
            fake_quantize_dynamic_(self.a_hat_cache)
        return out
    return wrapped


_Cls = i8opt.OptimizedInt8Conv2d
_ORIGINALS = {}
for _name in ("forward", "forward_gn_fused_modiff", "forward_modiff_fused_silu_residual"):
    _ORIGINALS[_name] = getattr(_Cls, _name)
    setattr(_Cls, _name, _make_wrapper(_ORIGINALS[_name]))

import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.utils import attention_identity_guard as guard  # noqa: E402

OUT_ROOT = "docs/int8_ahat_cache_2026-08-26/fid_run_dynamic"


def generate(mode, tag, ahat_quant):
    guard.seed_model_construction()
    torch.manual_seed(1234)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=OUT_ROOT, batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
    )
    _patched["on"] = ahat_quant
    print(f"\n=== generating {tag} (mode={mode}, a_hat int8 dynamic={ahat_quant}, n={N_SAMPLES}) ===")
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
    quant_dir = generate("int8", "int8_modiff_ahat_int8_dynamic", ahat_quant=True)

    print(f"\nquantize calls: {_stats['total_calls']} (each is a fresh per-tensor abs().max() "
          f"range -- zero-clip by construction, so no clip-rate stat to report)")

    from pytorch_fid import fid_score
    fid_base = fid_score.calculate_fid_given_paths([ref_dir, base_dir], batch_size=50,
                                                   device="cuda", dims=2048)
    fid_quant = fid_score.calculate_fid_given_paths([ref_dir, quant_dir], batch_size=50,
                                                    device="cuda", dims=2048)
    fid_between = fid_score.calculate_fid_given_paths([base_dir, quant_dir], batch_size=50,
                                                      device="cuda", dims=2048)

    print(f"\n{'=' * 70}")
    print(f"N={N_SAMPLES} samples, {STEPS} DDIM steps, W8A8+MoDiff, "
          f"a_hat range = dynamic per-call abs().max() (zero calibration)")
    print(f"{'=' * 70}")
    print(f"FID(fp16_ref, int8_modiff_baseline)             = {fid_base:.3f}")
    print(f"FID(fp16_ref, int8_modiff_ahat_int8_dynamic)    = {fid_quant:.3f}")
    print(f"FID(baseline, ahat_int8_dynamic)  (direct)       = {fid_between:.3f}")
    print(f"delta vs baseline: {fid_quant - fid_base:+.3f} ({100*(fid_quant-fid_base)/fid_base:+.1f}%)")
