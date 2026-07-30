"""Three-seed model-level quality gate for the accepted INT4 attention optimizations.

Compares the pre-plan/reference switches against the optimized production switches with identical
DDIM noise. The acceptance threshold is latent relative L2 < 0.02 for every seed.
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path[:0] = ["/workspace/MoDiff", "/workspace/MoDiff/src/taming-transformers"]

import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 4
STEPS = 50
SEEDS = (1234, 5678, 9012)


def _latent(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _setup(optimized):
    os.environ.update({
        "MODIFF_QUANT_LINEAR": "1",
        "MODIFF_QUANT_ATTN": "1",
        "MODIFF_QUANT_ATTN_STATIC": "1",
        "MODIFF_QATTN_FLASH": "1",
        "MODIFF_FLASH_GATE": "on",
        "MODIFF_QUANT_ATTN_ALLT": "0",
        "MODIFF_LINEAR_OUT_I8": "0",
        "MODIFF_INT4_GN_FAST": "1" if optimized else "0",
        "MODIFF_INT4_KV_FUSED": "1" if optimized else "0",
        "MODIFF_INT4_Q_IN_FLASH": "on" if optimized else "off",
        "MODIFF_INT4_COMPACT_STATIC": "1" if optimized else "0",
        "MODIFF_INT4_QKV_EPILOGUE": "0",
    })
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="integration/results/quality",
        batch_size=BATCH,
        steps=STEPS,
        shape=(4, 32, 32),
        calibration_path="integration/calibration/int4_calibration.pt",
        linear_backend="int_gemm",
    )
    model, sampler = runner._setup_model("int4_baseline")
    cond = runner._cond_kwargs(model, BATCH)
    return runner, model, sampler, cond


def _all_latents(optimized):
    runner, model, sampler, cond = _setup(optimized)

    def sample(steps, seed):
        torch.manual_seed(seed)
        with torch.inference_mode(), torch.amp.autocast(
                "cuda", enabled=True, dtype=torch.float16):
            return _latent(sampler.sample(
                S=steps, batch_size=BATCH, shape=runner.shape, eta=0.0,
                verbose=False, **cond)).float().cpu()

    sample(12, SEEDS[0])  # freeze attention scales and warm all selected kernels
    result = {seed: sample(STEPS, seed) for seed in SEEDS}
    del model, sampler, runner
    torch.cuda.empty_cache()
    return result


old = _all_latents(False)
new = _all_latents(True)
passed = True
for seed in SEEDS:
    rel = (new[seed] - old[seed]).norm().item() / (old[seed].norm().item() + 1e-12)
    ok = rel < 0.02
    passed &= ok
    print(f"seed={seed} latent_rel_l2={rel:.6f} {'PASS' if ok else 'FAIL'}", flush=True)
print("ALL PASS" if passed else "QUALITY GATE FAILED")
raise SystemExit(0 if passed else 1)
