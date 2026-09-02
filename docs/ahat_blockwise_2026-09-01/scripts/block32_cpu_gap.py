"""Where does the int8-B=32 a_hat gap live -- GPU kernels or CPU launch time?

block32_real.py measures wall time; block32_why_slower.py measures summed kernel
time. When those two deltas disagree the difference is host-side, so this runs
both over the same 20-step window and prints them side by side.
"""
from __future__ import annotations
import os, sys, statistics, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]
os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="block32_cpu_gap.py")
import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity, DeviceType  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE, BATCH, STEPS = (4, 32, 32), 128, 20


def set_block(b):
    os.environ["MODIFF_AHAT_BLOCK"] = str(b)


def sample(model, sampler):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)


def arm(model, sampler, label):
    sample(model, sampler)
    torch.cuda.synchronize()
    walls = []
    for _ in range(3):
        t0 = time.perf_counter()
        sample(model, sampler)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3 / STEPS)
    wall = statistics.median(walls)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        sample(model, sampler)
        torch.cuda.synchronize()
    gpu, launches = 0.0, 0
    for evt in prof.key_averages():
        if evt.device_type == DeviceType.CUDA and evt.self_device_time_total > 0:
            gpu += evt.self_device_time_total / 1e3
            launches += evt.count
    gpu /= STEPS
    print(f"{label:14s} wall {wall:7.3f}  gpu-kernels {gpu:7.3f}  "
          f"idle {wall-gpu:+7.3f}  launches/step {launches/STEPS:.0f}", flush=True)
    return wall, gpu, launches / STEPS


def main():
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_cpu_gap",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    set_block(0)
    model, sampler = runner._setup_model("int8")
    out = {}
    for b, label in [(0, "fp16 a_hat"), (32, "int8 B=32")]:
        set_block(b)
        out[label] = arm(model, sampler, label)
    w0, g0, l0 = out["fp16 a_hat"]
    w1, g1, l1 = out["int8 B=32"]
    print(f"\ndelta   wall {w1-w0:+7.3f}   gpu {g1-g0:+7.3f}   "
          f"host {(w1-g1)-(w0-g0):+7.3f}   launches {l1-l0:+.0f}", flush=True)


if __name__ == "__main__":
    main()
