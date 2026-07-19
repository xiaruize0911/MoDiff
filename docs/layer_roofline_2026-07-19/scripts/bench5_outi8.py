"""5-version e2e benchmark with the int8-OUTPUT Linear fix (MODIFF_LINEAR_OUT_I8) wired into the model.
Best batch (128). Protocol: 30 warm-up steps, then 5 runs x 200 DDIM steps, report MEAN per-step ms.
Runs fp16 + {int8,int4}x{baseline,modiff}, each with the Linear-output fix OFF and ON. Attention = fp16
SDPA (the established best). Writes data/bench5_outi8_b128.csv."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = int(os.environ.get("E2E_BATCH", "128"))
WARMUP_STEPS, TIMED_STEPS, RUNS = 30, 200, 5


def run(mode, backend, out_i8):
    if backend == "int_gemm": os.environ["MODIFF_QUANT_LINEAR"] = "1"
    else: os.environ.pop("MODIFF_QUANT_LINEAR", None)
    os.environ["MODIFF_LINEAR_OUT_I8"] = "1" if out_i8 else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"   # fp16 SDPA attention (established best)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/b5outi8",
                          batch_size=BATCH, steps=TIMED_STEPS, shape=(4, 32, 32), calibration_path=calib, linear_backend=backend)
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    # confirm the fix engaged
    from integration.kernels.wxax_linear import QuantLinearWxAx
    neng = sum(1 for m in model.model.diffusion_model.modules()
               if isinstance(m, QuantLinearWxAx) and getattr(m, "_out_i8", False) and m._inv_out_scale is not None)

    def sample(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != "fp32"), dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    sample(WARMUP_STEPS); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); sample(TIMED_STEPS); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED_STEPS * 1000)
    mean = statistics.mean(ms); mn = min(ms)
    del model, sampler; torch.cuda.empty_cache()
    return mean, mn, neng


rows = []
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int8_modiff", "int8"),
        ("int4_baseline", "int4_baseline"), ("int4_modiff", "int4")]
print(f"batch={BATCH}, warmup={WARMUP_STEPS} steps, {RUNS}x{TIMED_STEPS} steps, MEAN ms/step\n")
for (label, mode) in VERS:
    backend = "int_gemm" if "int" in mode else "fp16"
    variants = [("off", False)] if mode == "fp16" else [("off", False), ("on", True)]
    res = {}
    for (tag, oi8) in variants:
        mean, mn, neng = run(mode, backend, oi8)
        res[tag] = (mean, mn, neng)
        print(f"  {label:16s} out_i8={tag:3s}  mean={mean:7.2f}  min={mn:7.2f} ms/step  (fix layers={neng})")
    row = dict(version=label, mean_off=round(res["off"][0], 3), min_off=round(res["off"][1], 3))
    if "on" in res:
        row.update(mean_on=round(res["on"][0], 3), min_on=round(res["on"][1], 3),
                   fix_layers=res["on"][2], speedup_on_over_off=round(res["off"][0] / res["on"][0], 4))
    rows.append(row)

fp = rows[0]["mean_off"]
with open("docs/layer_roofline_2026-07-19/data/bench5_outi8_b128.csv", "w", newline="") as f:
    cols = ["version", "mean_off", "min_off", "mean_on", "min_on", "fix_layers", "speedup_on_over_off"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"\n=== SUMMARY (mean ms/step, batch {BATCH}) — × vs fp16 ===")
for r in rows:
    on = f" | fix ON {r.get('mean_on','-')} ({fp/r['mean_on']:.3f}× fp16, {r.get('speedup_on_over_off','-')}× vs off)" if r.get("mean_on") else ""
    print(f"  {r['version']:16s} off {r['mean_off']:7.2f} ({fp/r['mean_off']:.3f}× fp16){on}")
print("\nWROTE bench5_outi8_b128.csv")
