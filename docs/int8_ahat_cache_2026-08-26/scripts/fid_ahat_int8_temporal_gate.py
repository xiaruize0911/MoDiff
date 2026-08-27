"""Idea from the "any other ideas" menu: instead of quantizing a_hat to int8 for the WHOLE 50-step
schedule (already measured catastrophic at every range strategy tried -- naive/calibrated/dynamic,
all >20x FID degradation), gate it to only a SUBSET of steps and leave a_hat in fp16 for the rest.

Uses the DYNAMIC (per-call exact abs().max(), the best range strategy found so far -- 182.4 FID at
full schedule, docs/int8_ahat_cache_2026-08-26/FINDINGS.md follow-up (5)) quantizer, but only
applies it when self.step_count falls in a given [GATE_START, GATE_END] window; outside the window
a_hat is left untouched (real fp16).

Two windows tested: FIRST half of the schedule (steps 1-25) and SECOND half (steps 26-50). Reuses
the fp16_ref and int8_modiff_baseline sample sets already on disk from the full-schedule dynamic
run (same seed 1234, same 50 steps, same batch 16) instead of regenerating them.

Run: python docs/int8_ahat_cache_2026-08-26/scripts/fid_ahat_int8_temporal_gate.py
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
MID = STEPS // 2   # 25

import integration.kernels.int8_optimized as i8opt  # noqa: E402

_patched = {"on": False, "lo": 1, "hi": STEPS}


def fake_quantize_dynamic_(t):
    m = float(t.detach().abs().max())
    if m < 1e-8:
        return
    lsb = m / 127.0
    q = torch.clamp(torch.round(t.float() / lsb), -127, 127)
    t.copy_((q * lsb).to(t.dtype))


def _make_wrapper(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if _patched["on"] and getattr(self, "a_hat_cache", None) is not None \
                and self.a_hat_cache.numel() > 0 \
                and _patched["lo"] <= self.step_count <= _patched["hi"]:
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

OUT_ROOT = "docs/int8_ahat_cache_2026-08-26/fid_run_temporal"
EXISTING_ROOT = "docs/int8_ahat_cache_2026-08-26/fid_run_dynamic"


def generate(mode, tag, lo, hi):
    guard.seed_model_construction()
    torch.manual_seed(1234)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=OUT_ROOT, batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
    )
    _patched["on"] = True
    _patched["lo"], _patched["hi"] = lo, hi
    print(f"\n=== generating {tag} (mode={mode}, a_hat int8 window=[{lo},{hi}], n={N_SAMPLES}) ===")
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
    ref_dir = os.path.join(EXISTING_ROOT, "fp16_ref")
    base_dir = os.path.join(EXISTING_ROOT, "int8_modiff_baseline")
    assert os.path.isdir(ref_dir) and os.path.isdir(base_dir), \
        "expected existing fp16_ref / int8_modiff_baseline from the prior full-schedule dynamic run"

    first_half_dir = generate("int8", "int8_ahat_first_half", 1, MID)
    second_half_dir = generate("int8", "int8_ahat_second_half", MID + 1, STEPS)

    from pytorch_fid import fid_score
    fid_base = fid_score.calculate_fid_given_paths([ref_dir, base_dir], batch_size=50,
                                                   device="cuda", dims=2048)
    fid_first = fid_score.calculate_fid_given_paths([ref_dir, first_half_dir], batch_size=50,
                                                    device="cuda", dims=2048)
    fid_second = fid_score.calculate_fid_given_paths([ref_dir, second_half_dir], batch_size=50,
                                                     device="cuda", dims=2048)

    print(f"\n{'=' * 70}")
    print(f"N={N_SAMPLES} samples, {STEPS} DDIM steps, W8A8+MoDiff, a_hat int8 = dynamic range, "
          f"GATED to a window")
    print(f"{'=' * 70}")
    print(f"FID(fp16_ref, int8_modiff_baseline)                       = {fid_base:.3f}")
    print(f"FID(fp16_ref, a_hat int8 steps 1-{MID} only)                = {fid_first:.3f}")
    print(f"FID(fp16_ref, a_hat int8 steps {MID+1}-{STEPS} only)              = {fid_second:.3f}")
    print(f"(for reference: full-schedule a_hat int8, all {STEPS} steps  = 182.383, from the prior run)")
