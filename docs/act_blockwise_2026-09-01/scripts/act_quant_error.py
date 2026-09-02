"""How much accuracy would a blockwise-along-C conv-input quantizer actually buy?

Measures the quantizer in isolation, on the real tensors it sees during a real
sampling run, instead of through the sampler. Final-latent relL2 is a bad instrument
for this question: a single conv's quantization error starts at ~3e-4, and both depth
(70 convs) and the 50-step trajectory amplify it, so the number you read out is mostly
trajectory chaos (this project already knows relL2 is only reproducible to +-0.03).

For every calibrated conv, at every step, this captures the exact tensor the int8
kernel quantizes -- x for the baseline arm, x - a_hat for MoDiff -- and reports

    relerr(Q) = ||dequant(Q(v)) - v|| / ||v||

under the granularities that are on the table:

    static   per-tensor, the calibrated scale / per-step delta table   (what ships)
    dyn      per-tensor, dynamic absmax
    B=32     dynamic, 32 consecutive channels per pixel  (the proposal)
    B=16     dynamic, 16 channels                        (bracket)
    perpix   dynamic, one scale per pixel over all C     (the granularity that,
             unlike B=32, IS expressible as an epilogue broadcast along M)

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/act_blockwise_2026-09-01/scripts/act_quant_error.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

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
os.environ["MODIFF_ACT_BLOCK"] = "0"
for _k in ("MODIFF_DISABLE_GN_MODIFF_FUSION", "MODIFF_DISABLE_GN_INT8_FUSION",
           "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION",
           "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION",
           "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION"):
    os.environ[_k] = "1"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="act_quant_error.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.kernels.int8_optimized import OptimizedInt8Conv2d  # noqa: E402

SHAPE = (4, 32, 32)
BATCH, STEPS, SEED = 4, 50, 20260805
OUT_JSON = "docs/act_blockwise_2026-09-01/data/act_quant_error.json"
QMAX = 127.0


def _err(v: torch.Tensor, deq: torch.Tensor) -> float:
    return float((deq - v).norm() / v.norm().clamp_min(1e-20))


def q_pertensor(v, scale=None):
    s = (v.abs().amax().clamp_min(1e-12) / QMAX) if scale is None else (1.0 / scale)
    return (v / s).round().clamp_(-QMAX, QMAX) * s


def q_block_c(v, block):
    """v is NCHW fp32. `block` consecutive channels per pixel share a scale."""
    t = v.permute(0, 2, 3, 1)
    n, h, w, c = t.shape
    bsz = min(block, c)
    pad = (bsz - c % bsz) % bsz
    tp = F.pad(t, (0, pad)) if pad else t
    blk = tp.reshape(n, h, w, tp.shape[-1] // bsz, bsz)
    s = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12) / QMAX
    r = ((blk / s).round().clamp_(-QMAX, QMAX) * s).reshape(n, h, w, -1)[..., :c]
    return r.permute(0, 3, 1, 2)


def q_perpixel(v):
    s = v.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / QMAX
    return (v / s).round().clamp_(-QMAX, QMAX) * s


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={BATCH} steps={STEPS}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/act_blockwise_2026-09-01/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    for arm, mode in (("baseline", "int8_baseline"), ("MoDiff", "int8")):
        model, sampler = runner._setup_model(mode)
        convs = [(n, m) for n, m in model.model.diffusion_model.named_modules()
                 if isinstance(m, OptimizedInt8Conv2d) and m.is_calibrated]
        acc = defaultdict(lambda: defaultdict(list))  # layer -> scheme -> [relerr]
        hooks = []

        def mk(name, mod):
            def pre(m, inp):
                x = inp[0]
                if m.fuse_input_silu:
                    x = F.silu(x)
                v = x.detach().float()
                # The tensor the int8 kernel actually quantizes.
                static = float(m.static_input_scale)
                if m.modiff_enabled and m.a_hat_cache is not None \
                        and m.a_hat_cache.shape == x.shape and not m.is_first_step:
                    v = v - m.a_hat_cache.detach().float()
                    i = min(max(m.step_count, 0), m.static_delta_scale.numel() - 1)
                    if bool(m.is_delta_calibrated):
                        static = float(m.static_delta_scale[i])
                if v.abs().amax() <= 0:
                    return
                a = acc[name]
                a["static"].append(_err(v, q_pertensor(v, static)))
                a["dyn"].append(_err(v, q_pertensor(v)))
                for b in (16, 32, 64, 128):
                    a[f"B={b}"].append(_err(v, q_block_c(v, b)))
                a["perpix"].append(_err(v, q_perpixel(v)))
            return pre

        for n, m in convs:
            hooks.append(m.register_forward_pre_hook(mk(n, m)))

        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                        dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
        for h in hooks:
            h.remove()

        schemes = ["static", "dyn", "perpix", "B=128", "B=64", "B=32", "B=16"]
        per_layer = {n: {s: sum(v[s]) / len(v[s]) for s in schemes} for n, v in acc.items()}
        overall = {s: sum(d[s] for d in per_layer.values()) / len(per_layer) for s in schemes}

        print(f"\n===== {arm}: mean relative quantization error of the conv-input "
              f"quantizer, {len(per_layer)} layers x {STEPS} steps =====", flush=True)
        for s in schemes:
            vs = overall["dyn"]
            print(f"  {s:8s} {overall[s]:.5f}"
                  + ("" if s == "dyn" else f"   {100.0 * (vs - overall[s]) / vs:+6.1f}% vs dyn"),
                  flush=True)

        worst = sorted(per_layer.items(), key=lambda kv: -kv[1]["dyn"])[:8]
        print(f"  -- 8 layers with the largest per-tensor-dynamic error --", flush=True)
        for n, d in worst:
            print(f"    {d['dyn']:.4f} -> {d['B=32']:.4f} (B=32)  {n}", flush=True)

        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        path = OUT_JSON.replace(".json", f"_{arm}.json")
        json.dump({"arm": arm, "steps": STEPS, "batch": BATCH, "overall": overall,
                   "per_layer": per_layer}, open(path, "w"), indent=1)
        print(f"  wrote {path}", flush=True)

        del model, sampler
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
