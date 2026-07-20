"""Measure int4 fused attention after the one-pass mixed quantize (quantize_attn_qkv_i4qk_i8v +
flash_attn_int4_vt). Correctness microbench + e2e vs fp16 and int8 fused. b128."""
import os, sys, math, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn.functional as F, modiff_cutlass as mc
BATCH = 128
print("symbols:", "flash_attn_int4_vt", hasattr(mc, "flash_attn_int4_vt"),
      "| quantize_attn_qkv_i4qk_i8v", hasattr(mc, "quantize_attn_qkv_i4qk_i8v"))

# ---- correctness microbench: int4 fused (new one-pass path) vs fp32 ref ----
torch.manual_seed(0); dev = "cuda"
def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)
for (N, H, T, hd) in [(128, 8, 1024, 24), (128, 8, 256, 48)]:
    BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc, -1), v.float())
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
    q4, k4, vt, sq4, sk4, sv = mc.quantize_attn_qkv_i4qk_i8v(qm, km, vm, hdp4, hdp_v)
    o = mc.flash_attn_int4_vt(q4.view(N, H, T, -1), k4.view(N, H, T, -1), vt.view(N, H, hdp_v, T),
                              sq4.view(N, H, T).contiguous(), sk4.view(N, H, T).contiguous(),
                              sv[..., :hd].contiguous().view(N, H, hd), hdp4, sc)
    print(f"  correctness hd{hd}/T{T}: int4 fused (new) rel-L2 vs fp32 = {relL2(o, ref):.4f}  (eager int4 ~0.14)")

# ---- e2e ----
import integration.benchmarks.benchmark_ldm as B
def setup(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for kk in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT"): os.environ.pop(kk, None)  # defaults
    calib = ("integration/calibration/int4_calibration.pt" if "int4" in mode else
             ("integration/calibration/int8_calibration.pt" if "int8" in mode else None))
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/i4v2",
                          batch_size=BATCH, steps=40, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH); return r, model, sampler, cond
def run(mode):
    r, model, sampler, cond = setup(mode)
    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(40); torch.cuda.synchronize()
        ms.append((time.time() - t0) / 40 * 1000)
    del model, sampler; torch.cuda.empty_cache(); return statistics.mean(ms)
bn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
tf = run("fp16"); t8 = run("int8_baseline"); t4 = run("int4_baseline")
print(f"\n===== e2e @ b{BATCH} (defaults: fused-flash quant attn ON) =====")
print(f"  fp16                : {tf:.1f} ms/step  1.000x")
print(f"  int8 fused (DEFAULT): {t8:.1f} ms/step  {tf/t8:.3f}x")
print(f"  int4 fused (NEW)    : {t4:.1f} ms/step  {tf/t4:.3f}x")
