"""Three-seed model-quality gate for exact T1024/hd24 INT8 FlashAttention."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 4
STEPS = 50
SEEDS = (1234, 5678, 9012)


def latent(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def all_latents(exact):
    os.environ.update({
        "MODIFF_QUANT_LINEAR": "1",
        "MODIFF_QUANT_ATTN": "1",
        "MODIFF_QUANT_ATTN_STATIC": "1",
        "MODIFF_QATTN_FLASH": "1",
        "MODIFF_FLASH_GATE": "on",
        "MODIFF_LINEAR_OUT_I8": "0",
        "MODIFF_ROUTE1": "0",
        "MODIFF_INT8_QKV_EPILOGUE": "1",
        "MODIFF_INT8_QKV_LAYOUT_EPILOGUE": "1",
        "MODIFF_INT8_QKV_COMPACT_EPILOGUE": "1",
        "MODIFF_INT8_FLASH_HD24_EXACT": "1" if exact else "0",
    })
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="integration/results/quality",
        batch_size=BATCH,
        steps=STEPS,
        shape=(4, 32, 32),
        calibration_path="integration/calibration/int8_calibration.pt",
        linear_backend="int_gemm",
    )
    model, sampler = runner._setup_model("int8")
    cond = runner._cond_kwargs(model, BATCH)

    def sample(steps, seed):
        torch.manual_seed(seed)
        with torch.inference_mode(), torch.amp.autocast(
                "cuda", enabled=True, dtype=torch.float16):
            return latent(sampler.sample(
                S=steps, batch_size=BATCH, shape=runner.shape, eta=0.0,
                verbose=False, **cond)).float().cpu()

    sample(12, SEEDS[0])
    result = {seed: sample(STEPS, seed) for seed in SEEDS}
    del model, sampler, runner
    torch.cuda.empty_cache()
    return result


reference = all_latents(False)
candidate = all_latents(True)
passed = True
for seed in SEEDS:
    delta = candidate[seed] - reference[seed]
    rel = delta.norm().item() / (reference[seed].norm().item() + 1e-12)
    maximum = delta.abs().max().item()
    ok = rel < 0.02
    passed &= ok
    print(
        f"seed={seed} latent_rel_l2={rel:.8f} max_abs={maximum:.8f} "
        f"{'PASS' if ok else 'FAIL'}",
        flush=True)
print("ALL PASS" if passed else "QUALITY GATE FAILED")
raise SystemExit(0 if passed else 1)
