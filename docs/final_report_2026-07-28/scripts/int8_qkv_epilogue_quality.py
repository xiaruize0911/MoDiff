"""Fixed-seed model-level gate for the W8A8 QKV INT8 epilogue route."""
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


def _all_latents(epilogue):
    os.environ.update({
        "MODIFF_QUANT_LINEAR": "1",
        "MODIFF_QUANT_ATTN": "1",
        "MODIFF_QUANT_ATTN_STATIC": "1",
        "MODIFF_QATTN_FLASH": "1",
        "MODIFF_FLASH_GATE": "on",
        "MODIFF_LINEAR_OUT_I8": "0",
        "MODIFF_ROUTE1": "0",
        "MODIFF_INT8_QKV_EPILOGUE": "on" if epilogue else "off",
        "MODIFF_INT8_KV_COMPACT24": "0",
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
            return _latent(sampler.sample(
                S=steps, batch_size=BATCH, shape=runner.shape, eta=0.0,
                verbose=False, **cond)).float().cpu()

    sample(12, SEEDS[0])
    result = {seed: sample(STEPS, seed) for seed in SEEDS}
    del model, sampler, runner
    torch.cuda.empty_cache()
    return result


reference = _all_latents(False)
candidate = _all_latents(True)
passed = True
for seed in SEEDS:
    rel = ((candidate[seed] - reference[seed]).norm().item()
           / (reference[seed].norm().item() + 1e-12))
    ok = rel < 0.02
    passed &= ok
    print(f"seed={seed} latent_rel_l2={rel:.6f} {'PASS' if ok else 'FAIL'}",
          flush=True)
print("ALL PASS" if passed else "QUALITY GATE FAILED")
raise SystemExit(0 if passed else 1)
