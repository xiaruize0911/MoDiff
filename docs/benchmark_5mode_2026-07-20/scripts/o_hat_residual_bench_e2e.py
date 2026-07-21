"""e2e speed A/B for the o_hat+residual fusion. argv: <mode:int8|int4> <fuse:on|off>.
GN-modiff fusion stays OFF (default). 20 warm + 4x150 timed, synchronized."""
import os, sys, time, statistics
mode, fuse = sys.argv[1], sys.argv[2]
if fuse == "off":
    os.environ["MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION"] = "1"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
os.environ["MODIFF_QUANT_LINEAR"]="1"; os.environ["MODIFF_QUANT_ATTN"]="1"; os.environ["MODIFF_LINEAR_OUT_I8"]="0"
import torch
import integration.benchmarks.benchmark_ldm as B
import integration.fused_ops.fused_resblock as FR

BATCH, WARMUP, TIMED, RUNS = 128, 20, 150, 4
calib = f"integration/calibration/{mode}_calibration.pt"
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/spres",
    batch_size=BATCH, steps=TIMED, shape=(4,32,32), calibration_path=calib, linear_backend="int_gemm")
model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

def smp(Sn):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=Sn, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
smp(WARMUP); torch.cuda.synchronize()
ms = []
for _ in range(RUNS):
    torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
    ms.append((time.time()-t0)/TIMED*1000)
print(f"RESULT mode={mode} residual_fuse={fuse} flag={FR.HAS_O_HAT_RESIDUAL} "
      f"ms/step mean={statistics.mean(ms):.2f} min={min(ms):.2f}")
