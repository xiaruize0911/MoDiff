"""Verify + benchmark the int8-score attention wiring (§8 kernels) on the int8 churches UNet.
Baseline = fp16 SDPA (int8-score OFF). Measures: (1) quality — UNet-forward rel-L2 of int8-score
vs baseline, against the model's own nondeterminism floor; (2) latency per-step, OFF vs ON.
Toggles TokenMajorAttentionBlock._int8_score post-setup. Batch via E2E_BATCH (default 64)."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock

BATCH = int(os.environ.get("E2E_BATCH", "64"))
WARM, STEPS, RUNS = 20, 100, 5


def set_score(model, on):
    n = 0
    for m in model.model.diffusion_model.modules():
        if isinstance(m, TokenMajorAttentionBlock):
            m._int8_score = on
            # reset self-calibration so each enable recalibrates cleanly
            m._sc_frozen = False; m._sc_n = 0; m._sc_sS_acc = 0.0; m._sc_c_acc = 0.0
            m._sc_sS = None; m._sc_c = None
            if on: n += 1
    return n


os.environ["MODIFF_QUANT_LINEAR"] = "1"
runner = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                           "models/ldm/lsun_churches256/model.ckpt",
                           output_dir="integration/results/verify_i8score", batch_size=BATCH,
                           steps=STEPS, shape=(4, 32, 32),
                           calibration_path="integration/calibration/int8_calibration.pt",
                           linear_backend="int_gemm")
model, sampler = runner._setup_model("int8")
cond = runner._cond_kwargs(model, BATCH)
print(f"int8-score-eligible attention blocks: {set_score(model, True)}")
dm = model.model.diffusion_model
torch.manual_seed(0)
x = torch.randn(BATCH, *runner.shape, device="cuda")
tt = torch.randint(0, 1000, (BATCH,), device="cuda")


def fwd():
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        return dm(x, tt).float()


def rel(u, v):
    return ((u - v).norm() / u.norm()).item()


# --- quality: baseline (fp16 SDPA) vs int8-score (after self-calib warm) ---
set_score(model, False); a1 = fwd(); a2 = fwd()          # nondeterminism floor
set_score(model, True)
for _ in range(10): fwd()                                 # warm to freeze sS/c
b1 = fwd()
print(f"\nQUALITY (single UNet forward):")
print(f"  OFF vs OFF (nondeterminism floor): rel-L2={rel(a1, a2):.3e}")
print(f"  OFF vs int8-score (attn quant err): rel-L2={rel(a1, b1):.3e}")


def bench(on):
    set_score(model, on)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=WARM, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)  # calib+warm
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
print(f"  int8-score OFF (fp16 SDPA): min={off_min:.3f}  median={off_med:.3f} ms/step")
print(f"  int8-score ON             : min={on_min:.3f}  median={on_med:.3f} ms/step")
print(f"  delta: {off_min - on_min:+.3f} ms/step  ({(off_min/on_min - 1)*100:+.2f}% e2e, min-of-{RUNS})")
