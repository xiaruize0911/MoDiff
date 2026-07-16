"""Quality: final-latent rel-err vs fp32 for every mode (dynamic vs static), fixed start noise.

DDIM eta=0 is deterministic given the same initial noise, so seeding before each sample makes the
latents directly comparable. Static trades a calibrated constant for the runtime statistic, so it
is lossier -- especially the static-c softmax on quantized attention (a single c cannot serve rows
whose max varies). Gate: int8 dynamic <= 0.05; everything else reported (int4 + all static are
MoDiff-compensation targets). Emits quality.csv."""
import os, sys, csv, importlib.util
os.environ.setdefault("MODIFF_ATTN_CALIB_STEPS", "16")
import torch
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"
SEED, BS, STEPS = 1234, 8, 20

class A: pass
def mk():
    a = A(); a.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
    a.ckpt = "models/ldm/lsun_churches256/model.ckpt"; a.batch_size = BS; a.steps = STEPS
    a.linear_backend = "int_gemm"; a.calibration = None; return a

def sample(mode):
    runner, model, sampler = abb.build(mode, mk())
    cond = runner._cond_kwargs(model, BS)
    # warm a few runs first so static calibration freezes before the measured (seeded) run
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
        for _ in range(2):
            sampler.sample(S=STEPS, batch_size=BS, shape=runner.shape, eta=0.0, verbose=False, **cond)
        torch.manual_seed(SEED)
        out = sampler.sample(S=STEPS, batch_size=BS, shape=runner.shape, eta=0.0, verbose=False, **cond)
    x = (out[0] if isinstance(out, tuple) else out).float()
    del runner, model, sampler; torch.cuda.empty_cache()
    return x

MODES = ["dynamic_fp16", "static_fp16", "dynamic_int8", "static_int8",
         "dynamic_int8_modiff", "static_int8_modiff", "dynamic_int4", "static_int4",
         "dynamic_int4_modiff", "static_int4_modiff"]
ref = sample("fp32")
rows = []
print(f"{'mode':>22} {'rel-vs-fp32':>12}")
for m in MODES:
    try:
        x = sample(m); rel = ((x - ref).norm() / ref.norm()).item()
    except Exception as e:
        rel = float("nan"); print("  ERR", m, repr(e))
    variant = "static" if m.startswith("static") else "dynamic"
    gate = "int8 gate" if m == "dynamic_int8" else "reported"
    print(f"{m:>22} {rel:12.4f}   ({gate})")
    rows.append({"mode": m, "variant": variant, "rel_vs_fp32": round(rel, 4), "note": gate})
with open(f"{OUT}/quality.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "variant", "rel_vs_fp32", "note"]); w.writeheader(); w.writerows(rows)
print("WROTE quality.csv")
