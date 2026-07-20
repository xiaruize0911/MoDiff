"""Rigorous check of the int8 flash-attention e2e effect. Loads the int8 model ONCE and
toggles _flash_bits on the live TokenMajorAttentionBlock modules, so MATH vs flash are timed
with identical weights/clocks, INTERLEAVED (rules out clock drift / run-order artifacts).
Also checks correctness: sampled latent (fixed seed) MATH vs flash rel-L2. b128."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock

BATCH = 128; TIMED = 100; ROUNDS = 4
os.environ["MODIFF_QUANT_LINEAR"] = "1"; os.environ["MODIFF_QUANT_ATTN"] = "0"
os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/e2e_flash_rig",
                      batch_size=BATCH, steps=TIMED, shape=(4, 32, 32),
                      calibration_path="integration/calibration/int8_calibration.pt", linear_backend="int_gemm")
model, sampler = r._setup_model("int8_baseline"); cond = r._cond_kwargs(model, BATCH)
blks = [m for m in model.model.diffusion_model.modules() if isinstance(m, TokenMajorAttentionBlock)]
print(f"loaded int8 model: {len(blks)} TokenMajorAttentionBlocks")

def set_flash(bits):
    for m in blks: m._flash_bits = bits

def smp(S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        return sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

def timed(bits, S=TIMED):
    set_flash(bits); torch.cuda.synchronize(); t0 = time.time(); smp(S); torch.cuda.synchronize()
    return (time.time() - t0) / S * 1000

# GPU burn-in + warmup both paths
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
set_flash(0); smp(20); set_flash(8); smp(20); torch.cuda.synchronize()

# interleaved timing
math_ms, flash_ms = [], []
for i in range(ROUNDS):
    math_ms.append(timed(0)); flash_ms.append(timed(8))
    print(f"  round {i}: MATH {math_ms[-1]:.2f}  flash {flash_ms[-1]:.2f}  delta {math_ms[-1]-flash_ms[-1]:+.2f} ms/step")

mm, fm = statistics.mean(math_ms), statistics.mean(flash_ms)
print(f"\nint8 MATH  : {mm:.2f} ms/step  (min {min(math_ms):.2f}, std {statistics.pstdev(math_ms):.2f})")
print(f"int8 flash : {fm:.2f} ms/step  (min {min(flash_ms):.2f}, std {statistics.pstdev(flash_ms):.2f})")
print(f"flash vs MATH: {mm/fm:.3f}x  ({(1-fm/mm)*100:+.1f}% step time)")

# correctness: same seed, MATH vs flash sampled latent (sample() -> (samples, intermediates))
def latent(x): return x[0] if isinstance(x, (tuple, list)) else x
torch.manual_seed(1234); set_flash(0); lat_m = latent(smp(20))
torch.manual_seed(1234); set_flash(8); lat_f = latent(smp(20))
rel = (lat_m.float() - lat_f.float()).norm().item() / (lat_m.float().norm().item() + 1e-9)
print(f"\ncorrectness: sampled-latent rel-L2 (int8 MATH vs int8 flash) = {rel:.4f}")
print("(if this is small ~0.01-0.05, flash output is sane; if large, the speed would be meaningless)")
