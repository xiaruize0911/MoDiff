"""D: e2e latent rel-err of int8/int4 quantized STANDARD attention (+ quantized conv) vs
fp16 standard attention. Confirms wiring + measures quality (int4 pre-MoDiff-compensation)."""
import os, sys, importlib.util
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
import torch
class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 8; args.steps = 20
args.linear_backend = "fp16"; args.calibration = None

def latent(mode):
    for k in ("MODIFF_STD_ATTN_BITS",): os.environ.pop(k, None)
    runner, model, sampler = abb.build(mode, args)
    torch.manual_seed(0); cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
        out = sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)
    lat = (out[0] if isinstance(out, (tuple, list)) else out).float()
    pk = torch.cuda.max_memory_allocated() / 1048576
    del runner, model, sampler; torch.cuda.empty_cache()
    return lat, pk

ref, _ = latent("fp16")
print("REF fp16 standard attention computed", flush=True)
for mode in ["int8_baseline", "int4_baseline"]:
    lat, pk = latent(mode)
    rel = (lat - ref).norm().item() / (ref.norm().item() + 1e-12)
    print(f"RESULT {mode:14s} (int{'8' if '8' in mode else '4'} conv + std attn) latent rel-vs-fp16={rel:.4f} peak={pk:.0f}MiB", flush=True)
