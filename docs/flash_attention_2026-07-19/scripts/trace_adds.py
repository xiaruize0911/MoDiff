"""Attribute fp16 residual-add bytes to source lines: patch Tensor.__add__/__iadd__/torch.add to
record caller line + output bytes for large adds. One UNet forward at b128 (int8 default)."""
import os, sys, traceback, collections
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
BATCH = 128
os.environ["MODIFF_QUANT_LINEAR"] = "1"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
for kk in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT"): os.environ.pop(kk, None)
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/addtrace",
                      batch_size=BATCH, steps=20, shape=(4, 32, 32),
                      calibration_path="integration/calibration/int8_calibration.pt", linear_backend="int_gemm")
model, sampler = r._setup_model("int8_baseline"); cond = r._cond_kwargs(model, BATCH)
def smp(S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
smp(20); torch.cuda.synchronize()

tally = collections.defaultdict(lambda: [0, 0])
def _site():
    hit = None
    for fr in reversed(traceback.extract_stack()[:-2]):
        f = os.path.basename(fr.filename)
        if f in ("openaimodel.py", "model.py", "fused_resblock.py", "quantized_std_attention.py",
                 "attention.py", "util.py", "token_major_attention.py"):
            hit = f"{f}:{fr.lineno}"; break
    return hit or "other"
_add = torch.Tensor.__add__
_iadd = torch.Tensor.__iadd__
def add(self, other):
    out = _add(self, other)
    if isinstance(other, torch.Tensor) and out.is_cuda and out.numel() >= 128*32*32:
        t = tally[_site()]; t[0] += 1; t[1] += out.numel() * out.element_size()
    return out
def iadd(self, other):
    if isinstance(other, torch.Tensor) and self.is_cuda and self.numel() >= 128*32*32:
        t = tally[_site() + " [+=]"]; t[0] += 1; t[1] += self.numel() * self.element_size()
    return _iadd(self, other)
torch.Tensor.__add__ = add
torch.Tensor.__iadd__ = iadd
smp(1)
torch.Tensor.__add__ = _add; torch.Tensor.__iadd__ = _iadd

print("\n===== large fp16 residual-add bytes by source line (one step) =====")
for site, (cnt, byt) in sorted(tally.items(), key=lambda x: -x[1][1]):
    print(f"  {byt/1e6:8.1f} MB   x{cnt:<4d}  {site}")
print(f"  TOTAL {sum(v[1] for v in tally.values())/1e6:.1f} MB (out) / step")
