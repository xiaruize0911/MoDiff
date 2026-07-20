"""Verify the NEW DEFAULT pipeline: int8 mode with NO env flags set should now use fused-flash
quantized attention. Confirms the conversion message, that flash is engaged, and the e2e timing."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
# clear any leftover attention env flags -> test the true defaults
for k in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT", "MODIFF_FLASH_ATTN"):
    os.environ.pop(k, None)
import torch
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.quantized_std_attention import QuantizedStandardAttentionBlock
BATCH = 128

def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/verifydef",
                          batch_size=BATCH, steps=40, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    nqb = sum(1 for m in model.model.diffusion_model.modules() if isinstance(m, QuantizedStandardAttentionBlock))
    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()
    # check flash engaged on a quant-attn block
    flash_flag = None
    for m in model.model.diffusion_model.modules():
        if isinstance(m, QuantizedStandardAttentionBlock):
            flash_flag = getattr(m, "_fq_frozen2", "n/a"); break
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(40); torch.cuda.synchronize(); ms.append((time.time() - t0) / 40 * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), nqb, flash_flag

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
tf, _, _ = run("fp16")
ti, nqb, ff = run("int8_baseline")
print(f"\n===== DEFAULT pipeline verification (no attn env flags) @ b{BATCH} =====")
print(f"  int8 QuantizedStandardAttentionBlocks in model: {nqb} (flash static-frozen after warmup: {ff})")
print(f"  fp16                : {tf:.1f} ms/step  1.00x")
print(f"  int8 (DEFAULT)      : {ti:.1f} ms/step  {tf/ti:.3f}x vs fp16")
print("  -> expect ~133 ms / ~1.42x if fused-flash attention is the default path.")
