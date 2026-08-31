"""I-MoDiff speed: one-layer 200-step + e2e UNet vs W8A8 full (~93.4 ms).

Run: source setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/imode_bench.py
     python docs/cache_schemes_report_2026-08-28/scripts/imode_bench.py --e2e
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"

import torch
import torch.nn as nn
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

OUT = "docs/cache_schemes_report_2026-08-28/data/imode.json"
N, C, H, STEPS, TRIALS = 128, 192, 32, 200, 3
CL = torch.channels_last
DEV = "cuda"


def _knobs(imode, bits, replay_k=1):
    os.environ["MODIFF_IMODE"] = "1" if imode else "0"
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    os.environ["MODIFF_DELTA_FREEZE"] = "0"


def build_layer():
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


@torch.inference_mode()
def time_200(layer, xs):
    layer.reset_state()
    layer(xs[0])
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for t in range(STEPS):
        layer(xs[t % len(xs)])
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / STEPS


def one_layer():
    os.environ["MODIFF_WARMUP_STEPS"] = "1"
    print(f"GPU {torch.cuda.get_device_name(0)}  one conv {C}->{C} {H}x{H} N={N}",
          flush=True)
    layer = build_layer()
    xs = [torch.randn(N, C, H, H, device=DEV, dtype=torch.float16).contiguous(memory_format=CL)
          for _ in range(8)]
    _knobs(False, 16)
    time_200(layer, xs)
    arms = [
        ("full_fp16", False, 16, 1),
        ("imode16", True, 16, 1),
        ("imode8", True, 8, 1),
        ("imode4", True, 4, 1),
    ]
    rows = []
    for label, imode, bits, rk in arms:
        _knobs(imode, bits, rk)
        samples = [time_200(layer, xs) for _ in range(TRIALS)]
        ms = statistics.median(samples)
        rows.append({"label": label, "imode": imode, "bits": bits,
                     "replay_k": rk, "ms_step": ms})
        print(f"  {label:16s} {ms:.4f} ms/step  {['%.3f' % s for s in samples]}",
              flush=True)
    return rows


def e2e():
    import integration.benchmarks.benchmark_ldm as B
    STEPS_E, BATCH = 50, 128
    SHAPE = (4, 32, 32)
    SEED = 20260827
    print(f"GPU {torch.cuda.get_device_name(0)}  e2e batch={BATCH} steps={STEPS_E}"
          f"  warmup_steps={os.environ.get('MODIFF_WARMUP_STEPS', '5')}",
          flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/cache_schemes_report_2026-08-28/tmp_imode_e2e",
        batch_size=BATCH, steps=STEPS_E, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    _knobs(False, 16)
    model, sampler = runner._setup_model("int8")

    def time_once():
        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS_E, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / STEPS_E

    arms = [
        ("full_fp16", False, 16, 1),
        ("imode16", True, 16, 1),
        ("imode8", True, 8, 1),
        ("imode4", True, 4, 1),
    ]
    rows = []
    for label, imode, bits, rk in arms:
        _knobs(imode, bits, rk)
        print(f"warmup {label}", flush=True)
        time_once()
        ms = time_once()
        rows.append({"label": label, "imode": imode, "bits": bits,
                     "replay_k": rk, "ms_step": ms})
        print(f"  {label:16s} {ms:.3f} ms/step", flush=True)
    _knobs(False, 16)
    return rows


def _merge(key, rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev[key] = rows
    json.dump(prev, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} [{key}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--one-layer", action="store_true")
    a = ap.parse_args()
    if not a.e2e and not a.one_layer:
        a.one_layer = True
        a.e2e = True
    if a.one_layer:
        _merge("one_layer", one_layer())
    if a.e2e:
        _merge("e2e", e2e())
    _knobs(False, 16)
    return 0


if __name__ == "__main__":
    sys.exit(main())
