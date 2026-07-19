"""Rigorous e2e per-step latency: 30 warm-up steps, then 5 runs x 200 DDIM steps, batch_size=64,
for fp16 / int8(port) / int4(port). Reuses BenchmarkRunner._setup_model (model load + conv/attn +
wxax-linear static calibration), then drives the sampler directly for exact warmup/timing control.
Writes data/e2e_bench_b64_s200.txt."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

CONFIG = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
CKPT = "models/ldm/lsun_churches256/model.ckpt"
BATCH = int(os.environ.get("E2E_BATCH", "64"))
WARMUP_STEPS, TIMED_STEPS, RUNS = 30, 200, 5
OUT = f"docs/quant_speedup_vs_fp16_2026-07-16/data/e2e_bench_b{BATCH}_s{TIMED_STEPS}.txt"


def reset_state(model, mode):
    dm = model.model.diffusion_model
    if mode == "int8" and B.HAS_INT8: B.reset_modiff_state_int8(dm)
    elif mode == "int4" and B.HAS_INT4: B.reset_modiff_state_int4(dm)
    if mode == "int8" and B.HAS_INT8_LINEAR: B.reset_modiff_state_linear(dm)
    elif mode == "int4" and B.HAS_INT4_LINEAR: B.reset_modiff_state_int4_linear(dm)


def run_mode(mode):
    quant = mode in ("int8", "int4")
    if quant: os.environ["MODIFF_QUANT_LINEAR"] = "1"
    else: os.environ.pop("MODIFF_QUANT_LINEAR", None)
    calib = {"int8": "integration/calibration/int8_calibration.pt",
             "int4": "integration/calibration/int4_calibration.pt"}.get(mode)
    runner = B.BenchmarkRunner(CONFIG, CKPT, output_dir="integration/results/e2e_b64_s200",
                               batch_size=BATCH, steps=TIMED_STEPS, shape=(4, 32, 32),
                               calibration_path=calib,
                               linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = runner._setup_model(mode)
    # conv/attn static scales: _setup_model already applied calibration files; recalibrate only if absent
    if mode == "int8" and B.HAS_INT8:
        cfg = B.get_calibration_config_int8()
        if not cfg.is_calibrated: runner._calibrate_int8(model, sampler)
    elif mode == "int4" and B.HAS_INT4:
        p = "integration/calibration/int4_calibration.pt"
        if not os.path.exists(p): runner._calibrate_int4(model, sampler)
    cond = runner._cond_kwargs(model, BATCH)
    ac = (mode != "fp32")

    def sample(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=ac, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)

    # 30 warm-up steps (single partial pass; lets cuDNN autotune + caches settle)
    reset_state(model, mode)
    sample(WARMUP_STEPS)
    torch.cuda.synchronize()

    ms_per_step = []
    for r in range(RUNS):
        reset_state(model, mode)
        torch.cuda.synchronize(); t0 = time.time()
        sample(TIMED_STEPS)
        torch.cuda.synchronize(); dt = time.time() - t0
        ms = dt / TIMED_STEPS * 1000.0
        ms_per_step.append(ms)
        print(f"    run{r+1}: {ms:.3f} ms/step  ({dt:.3f}s / {TIMED_STEPS} steps, batch {BATCH})")
    del model, sampler; torch.cuda.empty_cache()
    return ms_per_step


def main():
    lines = [f"E2E per-step latency: {WARMUP_STEPS} warm-up steps, {RUNS} runs x {TIMED_STEPS} steps, batch_size={BATCH}",
             f"GPU: {torch.cuda.get_device_name()}", ""]
    summary = {}
    for mode in ("fp16", "int8", "int4"):
        print(f"\n===== {mode} =====")
        lines.append(f"=== {mode} ===")
        vals = run_mode(mode)
        for r, v in enumerate(vals): lines.append(f"  run{r+1}: {v:.3f} ms/step")
        mn, md, mean = min(vals), statistics.median(vals), statistics.mean(vals)
        summary[mode] = (mn, md, mean)
        lines.append(f"  min={mn:.3f}  median={md:.3f}  mean={mean:.3f} ms/step")
        print(f"  -> min={mn:.3f} median={md:.3f} mean={mean:.3f} ms/step")
    fp = summary["fp16"][0]
    lines.append("")
    lines.append(f"{'mode':6s} {'min':>8s} {'median':>8s} {'mean':>8s} {'min vs fp16':>12s}")
    for mode in ("fp16", "int8", "int4"):
        mn, md, mean = summary[mode]
        rel = f"{mn/fp:.3f}x" if mode != "fp16" else "-"
        lines.append(f"{mode:6s} {mn:8.3f} {md:8.3f} {mean:8.3f} {rel:>12s}")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
