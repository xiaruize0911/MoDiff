"""Deep profile of OUR quantized attention path, with the autotune gate FORCED ON.

Purpose: not "should we use it" but "why is it slow and what do we fix". MODIFF_FLASH_GATE=on
so every eligible block (head_dim<=48, T%64==0) runs our int8/int4 flash kernel instead of
letting the gate fall back to PyTorch flash.

The quantized S4 is not one kernel. It is:
  A. q/k/v .contiguous()          -- q,k,v arrive as non-contiguous transposes of the qkv
                                     tensor, so .reshape(BH,T,hd).contiguous() materializes
                                     three full copies before anything else happens
  B. quantize_attn_qkv[_static]   -- per-token Q/K scales + per-channel V scale, and V is
                                     emitted TRANSPOSED ([hd,T]) because the flash kernel
                                     consumes int8 PV that way
  C. flash_attn_int8_vt / int4_vt -- QK^T + online softmax + AV, scores kept in SRAM
  (int8 may instead take the PACKED path, which folds B into C's smem staging)

For each real eligible shape this reports every kernel's us and, for the flash kernel, the
achieved TOPS against the A40 int8/int4 dense peak -- so "4x slower than PyTorch flash"
becomes a specific statement about which stage and how far from the hardware limit.

Writes data/qattn_deep_profile.json.
"""
import os, sys, json, statistics, contextlib, io, re
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType

HERE = "docs/final_report_2026-07-28"
BATCH = int(os.environ.get("QDP_BATCH", "128"))
WARM, ITERS, ROUNDS = 12, 30, 5
# A40 dense tensor-core peaks (no sparsity)
PEAK = {"fp16": 74.8, "int8": 149.7, "int4": 299.3}


def bench(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(ROUNDS):
        s = [torch.cuda.Event(True) for _ in range(ITERS)]
        e = [torch.cuda.Event(True) for _ in range(ITERS)]
        for i in range(ITERS):
            s[i].record(); fn(); e[i].record()
        torch.cuda.synchronize()
        t = sorted(s[i].elapsed_time(e[i]) for i in range(ITERS))
        meds.append(t[len(t) // 2])
    return statistics.median(meds) * 1e3


def clean(name):
    nm = name.split("(")[0]
    for p in ("void ", "at::native::", "(anonymous namespace)::"):
        nm = nm.replace(p, "")
    nm = re.sub(r"<[^<>]*>", "", nm)
    nm = re.sub(r"<.*", "", nm).strip(": ")
    if nm.startswith("_Z"):
        toks = [t for t in re.split(r"\d+", nm) if t]
        cand = [t for t in toks if re.match(r"^[A-Z][A-Za-z_]{9,}$", t)
                and not t.startswith(("ZN", "Kernel"))]
        nm = re.sub(r"(INS_?|IN|EEE?)$", "", cand[0]) if cand else nm[:40]
    return nm[:46]


def kernels(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA or ev.self_device_time_total <= 0:
            continue
        rows.append({"kernel": clean(ev.key), "us": round(ev.self_device_time_total / iters, 2),
                     "calls": round(ev.count / iters, 2)})
    rows.sort(key=lambda r: -r["us"])
    return rows


def main():
    # Wake the CUDA context without power-capping the board before INT8 (the first
    # measured mode). Each block receives its own WARM iterations in bench().
    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    import integration.benchmarks.benchmark_ldm as B
    real = json.load(open(f"{HERE}/data/layer_pipeline_bench.json"))
    shapes = [(r["x_shape"][1], r["x_shape"][2], r["x_shape"][3], r["n_instances"])
              for r in real["modes"]["fp16"] if r["kind"] == "attention"]

    results = {"batch": BATCH, "modes": {}}
    for label, mode, calib, bits in (
            ("int8_baseline", "int8_baseline", "integration/calibration/int8_calibration.pt", 8),
            ("int4_baseline", "int4_baseline", "integration/calibration/int4_calibration.pt", 4)):
        os.environ["MODIFF_QUANT_LINEAR"] = "1"
        os.environ["MODIFF_QUANT_ATTN"] = "1"
        os.environ["MODIFF_FLASH_GATE"] = "on"          # <-- autotune OFF, force our kernel
        for k in ("MODIFF_SDPA_BACKEND", "MODIFF_FP16_MATERIALIZED"):
            os.environ.pop(k, None)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                                  "models/ldm/lsun_churches256/model.ckpt",
                                  output_dir=f"{HERE}/tmp_out", batch_size=BATCH, steps=10,
                                  shape=(4, 32, 32), calibration_path=calib,
                                  linear_backend="int_gemm")
            model, sampler = r._setup_model(mode)
            cond = r._cond_kwargs(model, BATCH)
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True,
                                                           dtype=torch.float16):
                sampler.sample(S=10, batch_size=BATCH, shape=r.shape, eta=0.0,
                               verbose=False, **cond)

        by_c = {}
        for name, m in model.model.diffusion_model.named_modules():
            if type(m).__name__ == "QuantizedStandardAttentionBlock":
                by_c.setdefault(m.channels, []).append(m)

        rows = []
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            for C, H, Wd, n_inst in shapes:
                if C not in by_c:
                    continue
                blk = by_c[C][0]
                T = H * Wd
                nh, hd = blk.num_heads, C // blk.num_heads
                eligible = (hd <= 48 and (T % 64) == 0)
                x = torch.randn(BATCH, C, H, Wd, device="cuda", dtype=torch.float16)
                x = x.contiguous(memory_format=torch.channels_last)
                blk(x)
                full = bench(lambda: blk(x))
                ks = kernels(lambda: blk(x))
                # isolate stage C: time the flash call alone on representative tensors
                attn_flops = 4 * BATCH * nh * T * T * hd
                flash_us = sum(k["us"] for k in ks if "flash_attn_int" in k["kernel"])
                pyflash_us = sum(k["us"] for k in ks if "pytorch_flash" in k["kernel"])
                quant_us = sum(k["us"] for k in ks
                               if any(s in k["kernel"] for s in ("aq_qtok", "aq_vquant", "aq_kquant")))
                copy_us = sum(k["us"] for k in ks
                              if "elementwise" in k["kernel"] or "direct_copy" in k["kernel"])
                which = "ours" if flash_us > 0 else ("pytorch" if pyflash_us > 0 else "?")
                used = flash_us if flash_us > 0 else pyflash_us
                # The hd=24 INT4-value specialization deliberately executes on K=32 INT8
                # MMA to avoid the 62.5% K-padding waste of native K=64 INT4 MMA.
                mixed_i4_i8mma = bits == 4 and T == 1024 and hd == 24 and flash_us > 0
                if flash_us > 0:
                    peak_kind = "int8" if bits == 8 or mixed_i4_i8mma else "int4"
                else:
                    peak_kind = "fp16"
                peak = PEAK[peak_kind]
                rows.append(dict(C=C, H=H, W=Wd, T=T, num_heads=nh, head_dim=hd,
                                 n_instances=n_inst, eligible=eligible, flash_path=which,
                                 tensor_core_peak_kind=peak_kind,
                                 full_us=round(full, 2),
                                 attn_gflop=round(attn_flops / 1e9, 2),
                                 flash_us=round(used, 2),
                                 flash_tops=round(attn_flops / (used * 1e-6) / 1e12, 1) if used else None,
                                 flash_pct_peak=round(attn_flops / (used * 1e-6) / 1e12 / peak * 100, 1)
                                 if used else None,
                                 qkv_quantize_us=round(quant_us, 2),
                                 copy_us=round(copy_us, 2),
                                 kernels=ks))
                print(f"{label:14s} C{C:4d} {H:2d}x{Wd:<2d} T={T:5d} hd={hd:3d} "
                      f"{'ELIG' if eligible else 'skip':4s} {which:7s} full={full:8.1f}us "
                      f"flash={used:8.1f}us ({rows[-1]['flash_pct_peak'] or 0:5.1f}% peak) "
                      f"quant={quant_us:7.1f}us copy={copy_us:7.1f}us")
                del x
                torch.cuda.empty_cache()
        results["modes"][label] = rows
        del model, sampler
        torch.cuda.empty_cache()

    with open(f"{HERE}/data/qattn_deep_profile.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {HERE}/data/qattn_deep_profile.json")


if __name__ == "__main__":
    main()
