"""One-layer model, 200 real steps. Skip/replay use production cadence, not mix().

Model: a single OptimizedInt8Conv2d (192->192, 3x3, batch 128, 32x32).
Untimed t=T first_step to seed a_hat/o_hat, then 200 modulated forwards.
Cadence is the module's own step_count % K (same as DDIM).

Run: source setup_cuda_env.sh && python docs/cache_schemes_report_2026-08-28/scripts/one_layer_200.py
"""
import json
import os
import statistics
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_WARMUP_STEPS"] = "1"

import torch
import torch.nn as nn
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

N, C, H, STEPS, TRIALS = 128, 192, 32, 200, 3
CL = torch.channels_last
DEV = "cuda"


def apply_knobs(skip_k, replay_k, bits, refresh):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(skip_k)
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_AHAT_REFRESH"] = str(refresh)


def build():
    raw = nn.Conv2d(C, C, 3, padding=1).to(DEV)
    layer = OptimizedInt8Conv2d(raw, layer_name="one").to(DEV)
    layer.enable_modiff(True)
    layer.set_static_scale(16.0)
    layer.static_delta_scale.fill_(16.0)
    layer.static_delta_alpha.fill_(1.0 / 16.0)
    layer.is_delta_calibrated.fill_(True)
    layer._delta_cal = True
    layer.eval()
    return layer


def make_xs(nbuf=8):
    return [torch.randn(N, C, H, H, device=DEV, dtype=torch.float16).contiguous(memory_format=CL)
            for _ in range(nbuf)]


@torch.inference_mode()
def first_step(layer, xs):
    layer.reset_state()
    layer(xs[0])
    assert layer.is_first_step is False


@torch.inference_mode()
def run_200(layer, xs):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for t in range(STEPS):
        layer(xs[t % len(xs)])
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / STEPS


ARMS = [
    ("W8A8 full fp16", 1, 1, 16, 0),
    ("W8A8 skip-K=2 fp16", 2, 1, 16, 0),
    ("W8A8 skip-K=4 fp16", 4, 1, 16, 0),
    ("W8A8 skip-K=8 fp16", 8, 1, 16, 0),
    ("W8A8 replay-K=2", 1, 2, 16, 0),
    ("W8A8 replay-K=4", 1, 4, 16, 0),
    ("W8A8 replay-K=8", 1, 8, 16, 0),
    ("W8A8 a_hat int8 held", 1, 1, 8, 0),
    ("W8A8 a_hat int8 refresh", 1, 1, 8, 1),
    ("W8A8 a_hat int4 held", 1, 1, 4, 0),
    ("W8A8 a_hat int4 refresh", 1, 1, 4, 1),
    ("W8A8 skip-K=4 int8 held", 4, 1, 8, 0),
    ("W8A8 skip-K=4 int4 held", 4, 1, 4, 0),
    ("W8A8 replay-K=4 int8 held", 1, 4, 8, 0),
    ("W8A8 replay-K=4 int4 held", 1, 4, 4, 0),
]


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  one conv {C}->{C} {H}x{H}  "
          f"N={N} steps={STEPS}", flush=True)
    layer = build()
    xs = make_xs()
    apply_knobs(1, 1, 16, 0)
    first_step(layer, xs)
    run_200(layer, xs)  # autotune / caches
    rows = []
    for label, sk, rk, bits, ref in ARMS:
        apply_knobs(sk, rk, bits, ref)
        samples = []
        for _ in range(TRIALS):
            first_step(layer, xs)
            samples.append(run_200(layer, xs))
        ms = statistics.median(samples)
        rows.append({"label": label, "skip_k": sk, "replay_k": rk,
                     "bits": bits, "refresh": ref, "ms_step": ms})
        print(f"  {label:32s} {ms:.4f} ms/step  trials={['%.3f'%s for s in samples]}",
              flush=True)

    ref_ms = rows[0]["ms_step"]
    for r in rows:
        r["vs_full"] = ref_ms / r["ms_step"]
        print(f"  {r['label']:32s} {r['ms_step']:7.3f}  {r['vs_full']:.3f}x", flush=True)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "model": "one OptimizedInt8Conv2d",
        "shape": {"N": N, "C": C, "H": H, "steps": STEPS},
        "method": "untimed t=T first_step, then 200 modulated forwards, CUDA event / 200",
        "arms": rows,
    }
    path = "docs/cache_schemes_report_2026-08-28/data/one_layer_200.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
