"""A/B e2e benchmark: effect of the MODIFF_FLASH_PACKED default flip (0 -> 1) on the
production LDM-churches pipeline (see integration/fused_ops/quantized_std_attention.py:89).

Only int8 attention blocks (bits=8) can be affected -- fp16 mode doesn't use the flash
quantized-attention class at all, and int4 mode's _flash_packed gate is hardcoded
`bits == 8 and ...`, so int4_baseline/int4_modiff are structurally untouched by this flag.
This script therefore only re-benchmarks int8_baseline and int8_modiff, each under
MODIFF_FLASH_PACKED=0 (old default) and =1 (new default), same methodology as
docs/benchmark_5mode_2026-07-25/scripts/e2e_speed.py (GPU clock burn-in, warmup, N rounds
x M timed steps, mean + min ms/step). fp16/int4 numbers are reused unchanged from that run
(dated 2026-07-25, 2 days prior) since this flag cannot affect them.

Additionally dumps, per QuantizedStandardAttentionBlock instance (i.e. per level/direction),
the frozen score-path decision (_flash_choice: flash vs fp16 SDPA) and the frozen packed-vs-
quantize+flash decision (_packed_choice) under both settings, so we can see exactly which
blocks' internal choice changed, not just the aggregate ms/step.
"""
import os, sys, csv, json, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.quantized_std_attention import QuantizedStandardAttentionBlock

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("int8_baseline", "int8_baseline"), ("int8_modiff", "int8")]


def collect_block_choices(model):
    out = []
    for name, m in model.named_modules():
        if isinstance(m, QuantizedStandardAttentionBlock):
            out.append(dict(
                name=name, channels=m.channels, head_dim=m.head_dim,
                flash_choice=m._flash_choice, packed_choice=m._packed_choice,
                packed_qout_choice=m._packed_qout_choice,
            ))
    return out


def run(mode, flash_packed):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ["MODIFF_FLASH_PACKED"] = str(flash_packed)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    choices = collect_block_choices(model)
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms), choices


os.makedirs(f"{HERE}/data", exist_ok=True)
os.makedirs(f"{HERE}/tmp_out", exist_ok=True)

# GPU clock burn-in
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
block_dump = {}
print(f"A/B e2e speed @ b{BATCH}  ({WARMUP} warm + {RUNS}x{TIMED} steps, MODIFF_FLASH_PACKED 0 vs 1)\n")
print(f"{'mode':16} {'packed':7} {'ms/step':>9} {'min':>8}")
for (label, mode) in VERS:
    for flash_packed in (0, 1):
        mean, mn, choices = run(mode, flash_packed)
        tag = f"{label}_packed{flash_packed}"
        print(f"{label:16} {flash_packed:<7} {mean:9.2f} {mn:8.2f}")
        rows.append(dict(mode=label, flash_packed=flash_packed, ms_step=round(mean, 2), min_ms=round(mn, 2)))
        block_dump[tag] = choices

with open(f"{HERE}/data/ab_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f"{HERE}/data/block_choices.json", "w") as f:
    json.dump(block_dump, f, indent=2)
print(f"\nWROTE {HERE}/data/ab_speed.csv and {HERE}/data/block_choices.json")
