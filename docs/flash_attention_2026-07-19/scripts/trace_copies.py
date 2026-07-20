"""Attribute direct_copy bytes to source lines: patch Tensor.contiguous/reshape/clone to record
the caller line + copied bytes whenever an actual copy happens. One UNet forward at b128."""
import os, sys, traceback, collections
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
BATCH = 128
os.environ["MODIFF_QUANT_LINEAR"] = "1"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
for kk in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT"): os.environ.pop(kk, None)
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/copytrace",
                      batch_size=BATCH, steps=20, shape=(4, 32, 32),
                      calibration_path="integration/calibration/int8_calibration.pt", linear_backend="int_gemm")
model, sampler = r._setup_model("int8_baseline"); cond = r._cond_kwargs(model, BATCH)
def smp(S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
smp(20); torch.cuda.synchronize()   # warm + freeze static

# ---- patch to record copies ----
tally = collections.defaultdict(lambda: [0, 0])   # site -> [count, bytes]
_orig_contig = torch.Tensor.contiguous
_orig_reshape = torch.Tensor.reshape
def _site():
    for fr in reversed(traceback.extract_stack()[:-2]):
        if "quantized_std_attention" in fr.filename or "token_major_attention" in fr.filename or \
           ("modiff" in fr.filename.lower() and "benchmark" not in fr.filename):
            return f"{os.path.basename(fr.filename)}:{fr.lineno}"
    return "other"
def contig(self, *a, **k):
    was = self.is_contiguous()
    out = _orig_contig(self, *a, **k)
    if not was and out.data_ptr() != self.data_ptr():
        t = tally[_site() + " [contig]"]; t[0] += 1; t[1] += out.numel() * out.element_size()
    return out
def reshape(self, *a, **k):
    was = self.is_contiguous()
    out = _orig_reshape(self, *a, **k)
    if not was and out.data_ptr() != self.data_ptr():
        t = tally[_site() + " [reshape]"]; t[0] += 1; t[1] += out.numel() * out.element_size()
    return out
torch.Tensor.contiguous = contig
torch.Tensor.reshape = reshape
smp(1)   # ONE step (=UNet forward x1)
torch.Tensor.contiguous = _orig_contig; torch.Tensor.reshape = _orig_reshape

print("\n===== copy bytes by source line (one step) =====")
for site, (cnt, byt) in sorted(tally.items(), key=lambda x: -x[1][1]):
    print(f"  {byt/1e6:8.1f} MB   x{cnt:<4d}  {site}")
print(f"  TOTAL {sum(v[1] for v in tally.values())/1e6:.1f} MB copied/step")
