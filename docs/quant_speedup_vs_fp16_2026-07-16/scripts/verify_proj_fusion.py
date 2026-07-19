"""Verify + benchmark the proj-side quantize fusion (quantize_attn_out_int8) on the int8 churches
UNet. (1) Parity: same DDIM sample with fusion OFF vs ON -> compare final latents. (2) Latency:
per-step ms, fusion OFF vs ON. Toggles TokenMajorAttentionBlock._fuse_proj_quant post-setup so both
share one calibrated model. Batch via E2E_BATCH (default 64)."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
from integration.kernels.wxax_linear import QuantLinearWxAx

BATCH = int(os.environ.get("E2E_BATCH", "64"))
WARM, STEPS, RUNS = 20, 100, 5


def set_fusion(model, on):
    np_, nq = 0, 0
    for m in model.model.diffusion_model.modules():
        if isinstance(m, TokenMajorAttentionBlock):
            m._fuse_proj_quant = on
            m._fuse_qkv_quant = on
            if on and isinstance(m.proj, QuantLinearWxAx) and m.proj.bits == 8 and m.proj.a_scale is not None:
                np_ += 1
            if on and isinstance(m.qkv, QuantLinearWxAx) and m.qkv.bits == 8 and m.qkv.a_scale is not None:
                nq += 1
    return np_, nq


os.environ["MODIFF_QUANT_LINEAR"] = "1"
runner = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                           "models/ldm/lsun_churches256/model.ckpt",
                           output_dir="integration/results/verify_projfuse", batch_size=BATCH,
                           steps=STEPS, shape=(4, 32, 32),
                           calibration_path="integration/calibration/int8_calibration.pt",
                           linear_backend="int_gemm")
model, sampler = runner._setup_model("int8")
cond = runner._cond_kwargs(model, BATCH)
np_, nq = set_fusion(model, True)
print(f"fusion-eligible attention blocks (int8, calibrated): proj={np_}  qkv={nq}")


# --- 1. parity on ONE UNet forward. The int8 fused-conv/CUTLASS kernels are non-deterministic
# run-to-run, so parity must be judged against that noise floor (OFF-vs-OFF), not against 0.
# (Per-block the fused proj is bit-exact; see scripts/verify notes.) A 20-step DDIM trajectory is
# chaotic and amplifies even the OFF-vs-OFF noise to rel~1, so it is NOT a valid parity probe.
dm = model.model.diffusion_model
torch.manual_seed(0)
x = torch.randn(BATCH, *runner.shape, device="cuda")
tt = torch.randint(0, 1000, (BATCH,), device="cuda")


def fwd():
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        return dm(x, tt).float()


def rel(u, v):
    return ((u - v).norm() / u.norm()).item()


set_fusion(model, False); a1 = fwd(); a2 = fwd()
set_fusion(model, True);  b1 = fwd()
print(f"\nPARITY (single UNet forward):")
print(f"  OFF vs OFF (nondeterminism floor): rel-L2={rel(a1, a2):.3e}")
print(f"  OFF vs ON  (fusion effect)       : rel-L2={rel(a1, b1):.3e}  "
      f"-> {'within noise floor, fusion is numerically correct' if rel(a1, b1) <= 3 * rel(a1, a2) + 1e-4 else 'ABOVE floor -- investigate'}")


def bench(on):
    set_fusion(model, on)
    torch.manual_seed(0)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=WARM, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)
        torch.cuda.synchronize(); ms.append((time.time() - t0) / STEPS * 1000)
    return min(ms), statistics.median(ms)


off_min, off_med = bench(False)
on_min, on_med = bench(True)
print(f"\nLATENCY per-step (batch {BATCH}, {RUNS}x{STEPS} steps):")
print(f"  fusion OFF: min={off_min:.3f}  median={off_med:.3f} ms/step")
print(f"  fusion ON : min={on_min:.3f}  median={on_med:.3f} ms/step")
print(f"  delta: {off_min - on_min:+.3f} ms/step  ({(off_min/on_min - 1)*100:+.2f}% e2e, min-of-{RUNS})")
