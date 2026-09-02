"""yq meanabs over steps: held int8 B=32 vs fp16 a_hat."""
from __future__ import annotations
import os, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]
os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "32"

from integration.utils.preflight import preflight, MODEL
preflight(*MODEL, what="probe_yq_steps.py")

import torch
import modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B

runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_blockwise_2026-09-01/tmp_probe",
    batch_size=2, steps=50, shape=(4, 32, 32),
    calibration_path=B._default_calibration_path("int8"),
    auto_delta_table=True)
model, sampler = runner._setup_model("int8")
unet = model.model.diffusion_model

orig = mc.group_norm_silu_delta_quantize_nhwc
orig_rs = mc.group_norm_silu_delta_quantize_resize_nhwc
# 62 GN + 8 resize per modulated step
PER = 70
watch = {0, 1, 5, 10, 24, 48}
bucket = []  # list of (absmax, meanabs) this step
step_i = [0]


def flush_if_full():
    if len(bucket) >= PER:
        si = step_i[0]
        if si in watch:
            am = sum(a for a, _ in bucket) / len(bucket)
            mm = sum(m for _, m in bucket) / len(bucket)
            nsat = sum(1 for a, _ in bucket if a >= 120)
            print(f"  mod_step {si:2d}  yq_absmax_mean={am:6.1f}  yq_meanabs={mm:6.2f}  n_sat~127={nsat}/{len(bucket)}",
                  flush=True)
        bucket.clear()
        step_i[0] += 1


def wrap(*a, **k):
    yq = orig(*a, **k)
    bucket.append((float(yq.abs().max()), float(yq.abs().float().mean())))
    flush_if_full()
    return yq


def wrap_rs(*a, **k):
    yq = orig_rs(*a, **k)
    bucket.append((float(yq.abs().max()), float(yq.abs().float().mean())))
    flush_if_full()
    return yq

mc.group_norm_silu_delta_quantize_nhwc = wrap
mc.group_norm_silu_delta_quantize_resize_nhwc = wrap_rs


def run(tag):
    bucket.clear()
    step_i[0] = 0
    print(f"\n===== {tag} =====", flush=True)
    B.reset_modiff_state_int8(unet)
    B._reset_wxax_modiff_safe(model)
    torch.manual_seed(0)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=50, batch_size=2, shape=(4, 32, 32), eta=0.0, verbose=False)


run("B=32 int8 held")
os.environ["MODIFF_AHAT_BLOCK"] = "0"
# drop leftover 4D scales so fp16 is a clean arm
for m in unet.modules():
    if hasattr(m, "_ahat_qscale"):
        m._ahat_qscale = None
run("fp16 a_hat")
