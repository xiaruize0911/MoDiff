"""Is this repo's int8/int4 flash-attention kernel still worth using?

MODIFF_FLASH_GATE=auto (default) MEASURES the custom int flash kernel against fp16 SDPA per
block and picks the winner. That gate used to see a crippled opponent: fp16 SDPA was pinned
to the MATH backend. Now that fp16 SDPA is PyTorch's flash kernel, the comparison is fair and
the gate's verdict is worth reading directly.

  gate=auto  the shipped default -- gate decides per block
  gate=on    force the custom int flash kernel on every eligible block (head_dim<=48, T%64==0)
  gate=off   never use it

Reports e2e ms/step plus how much attention time each path spends, so "our kernel vs
PyTorch's" is a measured statement rather than an assumption. Run with an idle GPU: earlier
numbers for this were wrong because a background profile job was competing for the device.

Writes data/attn_gate_check.json.
"""
import os, sys, json, time, statistics, contextlib, io
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType

HERE = "docs/final_report_2026-07-28"
BATCH, WARMUP, TIMED, RUNS, PROF = 128, 20, 100, 3, 10


def run(mode, gate, calib):
    import integration.benchmarks.benchmark_ldm as B
    os.environ["MODIFF_QUANT_LINEAR"] = "1"
    os.environ["MODIFF_QUANT_ATTN"] = "1"
    os.environ["MODIFF_FLASH_GATE"] = gate
    for k in ("MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND", "MODIFF_FP16_MATERIALIZED"):
        os.environ.pop(k, None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                              "models/ldm/lsun_churches256/model.ckpt",
                              output_dir=f"{HERE}/tmp_out", batch_size=BATCH, steps=TIMED,
                              shape=(4, 32, 32), calibration_path=calib,
                              linear_backend="int_gemm")
        model, sampler = r._setup_model(mode)
        cond = r._cond_kwargs(model, BATCH)

        def smp(S):
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True,
                                                           dtype=torch.float16):
                sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0,
                               verbose=False, **cond)
        smp(WARMUP); torch.cuda.synchronize()
        ms = []
        for _ in range(RUNS):
            torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
            ms.append((time.time() - t0) / TIMED * 1000)
        mean_ms = statistics.mean(ms)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            smp(PROF); torch.cuda.synchronize()
        agg = {}
        for e in prof.key_averages():
            if e.device_type != DeviceType.CUDA or e.self_device_time_total <= 0:
                continue
            key = None
            if "flash_attn_int" in e.key:
                key = "ours_int_flash"
            elif "pytorch_flash" in e.key:
                key = "pytorch_flash"
            elif "aq_qtok" in e.key or "aq_vquant" in e.key or "quantize_attn" in e.key:
                key = "our_qkv_quantize"
            elif "quant_attn_out" in e.key:
                key = "our_attn_out_quantize"
            if key:
                a = agg.setdefault(key, {"ms_step": 0.0, "calls_per_step": 0.0})
                a["ms_step"] += e.self_device_time_total / PROF / 1000
                a["calls_per_step"] += e.count / PROF
        for a in agg.values():
            a["ms_step"] = round(a["ms_step"], 3)
            a["calls_per_step"] = round(a["calls_per_step"], 1)
        del model, sampler, prof
        torch.cuda.empty_cache()
    return dict(ms_step=round(mean_ms, 2), attn=agg)


def main():
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()
    out = {}
    for mode, calib in (("int8_baseline", "integration/calibration/int8_calibration.pt"),
                        ("int4_baseline", "integration/calibration/int4_calibration.pt")):
        for gate in ("auto", "on", "off"):
            res = run(mode, gate, calib)
            out[f"{mode}|{gate}"] = res
            o = res["attn"].get("ours_int_flash", {})
            p = res["attn"].get("pytorch_flash", {})
            print(f"{mode:14s} gate={gate:4s} {res['ms_step']:7.2f} ms/step | "
                  f"ours {o.get('ms_step',0):6.2f} ms x{o.get('calls_per_step',0):4.0f} | "
                  f"pytorch {p.get('ms_step',0):6.2f} ms x{p.get('calls_per_step',0):4.0f}")
    with open(f"{HERE}/data/attn_gate_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {HERE}/data/attn_gate_check.json")


if __name__ == "__main__":
    main()
