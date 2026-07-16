"""F3 quality: latent rel-err of the fused qkv-int->flash path vs fp16 attention and
vs the per-token §6 flash, for int8 and int4, on the default C192/T1024 block. Runs 2
samples per config (first completes calibration/warmup, second is measured)."""
import os, sys, importlib.util
import torch
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 8; args.steps = 20
args.linear_backend = "fp16"; args.calibration = None
MODE = "int8_baseline"

def latent(env):
    for k in ("MODIFF_QUANT_ATTN", "MODIFF_QKV_FLASH_FUSED", "MODIFF_FLASH_MIN_T"):
        os.environ.pop(k, None)
    os.environ.update(env)
    runner, model, sampler = abb.build(MODE, args)
    def one():
        torch.manual_seed(0); cond = runner._cond_kwargs(model, args.batch_size)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)
        return (out[0] if isinstance(out, (tuple, list)) else out).float()
    one()                          # calibrate + warm
    lat = one()                    # measured
    del runner, model, sampler; torch.cuda.empty_cache()
    return lat

ref = latent({})                                                     # fp16 attention
print("ref (fp16 attention) computed", flush=True)
configs = [
    ("§6 per-token int8 flash", {"MODIFF_QUANT_ATTN": "1"}),
    ("fused int8 (W8A8->flash)", {"MODIFF_QUANT_ATTN": "1", "MODIFF_QKV_FLASH_FUSED": "8"}),
    ("fused int4 (W4A4->flash)", {"MODIFF_QUANT_ATTN": "1", "MODIFF_QKV_FLASH_FUSED": "4"}),
]
for name, env in configs:
    lat = latent(env)
    rel = (lat - ref).norm().item() / (ref.norm().item() + 1e-12)
    print(f"{name:28s}  latent rel-err vs fp16 = {rel:.4f}", flush=True)
