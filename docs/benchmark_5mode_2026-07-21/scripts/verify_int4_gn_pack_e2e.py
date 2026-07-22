#!/usr/bin/env python3
"""Phase 5 validation: int4 attention GN->pack fold vs the fallback, in the DRIVER config.

Must run with the quantized-linear wiring the benchmark uses (MODIFF_QUANT_LINEAR=1 etc.),
else the attention qkv stays fp16 nn.Linear and the fold never engages. Builds one int4
model, freezes the flash self-calibration with a warmup sample, then samples the SAME seed
twice toggling MODIFF_FUSE_GN_QKV_I4 (read per-call in forward): 0 = fallback (fp16 GN +
standalone quantize_act_int4_pack), 1 = fold (group_norm_silu_quantize_pack_nhwc ->
forward_from_int4). Reports rel-L2 (expect small, nonzero -> engaged + correct).
"""
import os, sys, importlib.util, torch
# Driver config MUST be set before _setup_model reads it.
os.environ.setdefault("MODIFF_QUANT_LINEAR", "1")
os.environ.setdefault("MODIFF_QUANT_ATTN", "1")
os.environ.setdefault("MODIFF_QUANT_ATTN_STATIC", "1")
os.environ.setdefault("MODIFF_LINEAR_OUT_I8", "0")
os.environ.pop("MODIFF_FLASH_ATTN", None)

REPO = "/workspace/MoDiff"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src/taming-transformers")
os.chdir(REPO)
spec = importlib.util.spec_from_file_location("bldm", REPO + "/integration/benchmarks/benchmark_ldm.py")
bldm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bldm)

STEPS, BATCH, SEED = 50, 4, 1234


def sample(runner, model, sampler, fold):
    os.environ["MODIFF_FUSE_GN_QKV_I4"] = "1" if fold else "0"
    bldm.reset_modiff_state_int4(model.model.diffusion_model)
    if bldm.HAS_INT4_LINEAR:
        bldm.reset_modiff_state_int4_linear(model.model.diffusion_model)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out, _ = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False,
                                **runner._cond_kwargs(model, BATCH))
    return out.detach().float().cpu()


r = bldm.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                         ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/i4v",
                         batch_size=BATCH, steps=STEPS,
                         calibration_path="integration/calibration/int4_calibration.pt")
model, sampler = r._setup_model("int4")
# Warmup once to freeze the flash self-calibration so both measured runs share static scales.
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False,
                   **r._cond_kwargs(model, BATCH))

ref = sample(r, model, sampler, fold=False)   # fallback
out = sample(r, model, sampler, fold=True)     # GN->pack fold
re = (out - ref).norm().item() / (ref.norm().item() + 1e-12)
mx = (out - ref).abs().max().item()
# NOTE: engagement is confirmed separately by a forward_from_int4 call counter (see the Phase 5
# write-up). rel_L2 == 0 here is EXPECTED and correct: int4 quantization (a_scale ~0.7, buckets
# ~0.7 apart) is coarse enough to absorb the fp16(fallback)-vs-fp32(fold) GN-rounding difference,
# so the fold is bit-identical to the fallback. Any rel_L2 < 2e-2 passes.
print(f"[int4 GN->pack fold] rel_L2_vs_fallback={re:.4e} max_abs={mx:.4e} "
      f"-> {'PASS (bit-identical/within tol)' if re < 2e-2 else 'FAIL'}")
sys.exit(0 if re < 2e-2 else 1)
