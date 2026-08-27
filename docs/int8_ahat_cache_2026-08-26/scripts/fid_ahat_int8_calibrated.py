"""Corrected version of fid_ahat_int8.py.

fid_ahat_int8.py used ONE fixed +-8.0 range for every layer at every one of the 50 DDIM steps,
chosen from a distribution POOLED across all ~70 layers and all steps. Re-checking the raw capture
data after the project owner pushed back on the FID=304.6 result showed why that is unfair: even a
SINGLE layer's own max|a_hat| swings from 1.68 to 16.48 across the 50-step schedule (matching the
step_gain_tail growth already measured in the C15 calibration). A static range simultaneously (a)
clips whenever a layer's true peak exceeds it and (b) wastes nearly the whole 8-bit budget at every
step where the true range is far smaller than the peak.

This version calibrates a PER-LAYER, PER-STEP range table first (mirroring how static_delta_scale
already works for the quantized codes), then fake-quantizes a_hat_cache against the step- and
layer-appropriate range instead of one global constant. This is the fair version of "does a_hat
fit in 8 bits" -- best-effort calibrated, not worst-case naive.

Run: python docs/int8_ahat_cache_2026-08-26/scripts/fid_ahat_int8_calibrated.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")

import torch  # noqa: E402

N_SAMPLES = int(os.environ.get("FID_N", "200"))
STEPS = int(os.environ.get("FID_STEPS", "50"))
CAL_SAMPLES = int(os.environ.get("FID_CAL_N", "32"))
BATCH = 16
MARGIN = 1.15   # headroom over the largest calibration-observed |a_hat| for that (layer, step)

import integration.kernels.int8_optimized as i8opt  # noqa: E402

_Cls = i8opt.OptimizedInt8Conv2d
_WRITE_METHODS = ("forward", "forward_gn_fused_modiff", "forward_modiff_fused_silu_residual")

# ---------------------------------------------------------------------------
# Phase 1: calibrate a per-(layer_name, step_count) |a_hat| range table
# ---------------------------------------------------------------------------
_cal_table = {}   # (layer_name, step_count) -> observed max|a_hat|
_cal_on = {"on": False}


def _make_cal_wrapper(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if _cal_on["on"] and getattr(self, "a_hat_cache", None) is not None \
                and self.a_hat_cache.numel() > 0:
            key = (self.layer_name, self.step_count)
            m = float(self.a_hat_cache.detach().abs().max())
            if m > _cal_table.get(key, 0.0):
                _cal_table[key] = m
        return out
    return wrapped


_CAL_ORIGINALS = {name: getattr(_Cls, name) for name in _WRITE_METHODS}
for name in _WRITE_METHODS:
    setattr(_Cls, name, _make_cal_wrapper(_CAL_ORIGINALS[name]))

import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.utils import attention_identity_guard as guard  # noqa: E402

OUT_ROOT = "docs/int8_ahat_cache_2026-08-26/fid_run_calibrated"
CAL_ROOT = "docs/int8_ahat_cache_2026-08-26/fid_run_calibrated/_cal_scratch"

print(f"=== calibrating per-layer per-step a_hat ranges, n={CAL_SAMPLES}, steps={STEPS} "
      f"(seed 999, disjoint from the seed-1234 eval set) ===")
guard.seed_model_construction()
torch.manual_seed(999)
cal_runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir=CAL_ROOT, batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
)
_cal_on["on"] = True
cal_runner.run_mode("int8", num_samples=CAL_SAMPLES, calibrate=True, force_recalibrate=False)
_cal_on["on"] = False

max_step_seen = max(s for _, s in _cal_table)
n_layers = len({name for name, _ in _cal_table})
print(f"calibrated {len(_cal_table)} (layer, step) entries, {n_layers} layers, "
      f"steps observed up to {max_step_seen}")

for name in _WRITE_METHODS:
    setattr(_Cls, name, _CAL_ORIGINALS[name])

# ---------------------------------------------------------------------------
# Phase 2: fake-quantize a_hat against the calibrated per-(layer, step) range
# ---------------------------------------------------------------------------
_patched = {"on": False}
_FLOOR = 1e-3


def _range_for(layer_name, step_count):
    key = (layer_name, step_count)
    if key in _cal_table:
        return max(_cal_table[key] * MARGIN, _FLOOR)
    # longer run than calibrated, or a step this layer's write path never hit during
    # calibration -- clamp to the last step actually observed for this layer, same
    # fallback behaviour static_delta_scale already uses for its own table.
    seen = [s for n, s in _cal_table if n == layer_name]
    if not seen:
        return 8.0  # never calibrated at all -- fall back to the old global constant
    nearest = min(seen, key=lambda s: abs(s - step_count))
    return max(_cal_table[(layer_name, nearest)] * MARGIN, _FLOOR)


def fake_quantize_calibrated_(t, rng):
    lsb = rng / 127.0
    q = torch.clamp(torch.round(t.float() / lsb), -127, 127)
    t.copy_((q * lsb).to(t.dtype))


def _make_quant_wrapper(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if _patched["on"] and getattr(self, "a_hat_cache", None) is not None \
                and self.a_hat_cache.numel() > 0:
            rng = _range_for(self.layer_name, self.step_count)
            fake_quantize_calibrated_(self.a_hat_cache, rng)
        return out
    return wrapped


_ORIGINALS = {name: getattr(_Cls, name) for name in _WRITE_METHODS}
for name in _WRITE_METHODS:
    setattr(_Cls, name, _make_quant_wrapper(_ORIGINALS[name]))


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
    quant_dir = generate("int8", "int8_modiff_ahat_int8_calibrated", ahat_quant=True)

    from pytorch_fid import fid_score
    fid_base = fid_score.calculate_fid_given_paths([ref_dir, base_dir], batch_size=50,
                                                   device="cuda", dims=2048)
    fid_quant = fid_score.calculate_fid_given_paths([ref_dir, quant_dir], batch_size=50,
                                                    device="cuda", dims=2048)
    fid_between = fid_score.calculate_fid_given_paths([base_dir, quant_dir], batch_size=50,
                                                      device="cuda", dims=2048)

    print(f"\n{'=' * 70}")
    print(f"N={N_SAMPLES} samples, {STEPS} DDIM steps, W8A8+MoDiff, "
          f"a_hat range = per-layer per-step calibrated (margin {MARGIN}x)")
    print(f"{'=' * 70}")
    print(f"FID(fp16_ref, int8_modiff_baseline)              = {fid_base:.3f}")
    print(f"FID(fp16_ref, int8_modiff_ahat_int8_calibrated)  = {fid_quant:.3f}")
    print(f"FID(baseline, ahat_int8_calibrated)  (direct)     = {fid_between:.3f}")
    print(f"delta vs baseline: {fid_quant - fid_base:+.3f} ({100*(fid_quant-fid_base)/fid_base:+.1f}%)")
