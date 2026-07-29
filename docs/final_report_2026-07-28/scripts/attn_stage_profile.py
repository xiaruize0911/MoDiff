"""Per-STAGE profile of one AttentionBlock, for every real shape and every mode.

An AttentionBlock in this UNet computes, on x = [N,C,H,W] with T = H*W tokens,
nh heads and hd = C/nh channels per head:

  S1  GroupNorm(x)                     normalize over (C/32 groups, H, W); NO SiLU here
  S2  qkv = Conv1d(C -> 3C, k=1)       a per-token matmul: [N*T, C] @ [C, 3C]
  S3  split + transpose                q,k,v each [N, nh, T, hd]   (views, ~free)
  S4  attention                        softmax(q @ k^T / sqrt(hd)) @ v
  S5  proj_out = Conv1d(C -> C, k=1)   [N*T, C] @ [C, C]
  S6  residual                         x + proj_out

FLOPs (dominant terms), per block per step:
  S2  2 * N*T * C * 3C
  S4  2 * N*nh*T*T*hd  (QK^T)  +  2 * N*nh*T*T*hd  (AV)   = 4*N*nh*T*T*hd
  S5  2 * N*T * C * C
Note S4 scales with T^2 while S2/S5 scale with T -- so which stage dominates flips with
resolution, which is exactly what this script measures.

Method: monkeypatch the block's own methods to time each stage with CUDA events (median of
`iters` after warmup), on the real module in the real mode, at the real input shape. Stage
times are measured by timing cumulative prefixes and differencing, so fusion is respected:
if GN is fused into qkv, S1 shows ~0 and its cost lands in S2, which is the truth.

Writes data/attn_stage_profile.json.
"""
import os, sys, json, statistics, contextlib, io
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType

HERE = "docs/final_report_2026-07-28"
BATCH = int(os.environ.get("ASP_BATCH", "128"))
WARM, ITERS, ROUNDS = 15, 40, 5
MODES = [("fp16", "fp16", None),
         ("int8_baseline", "int8_baseline", "integration/calibration/int8_calibration.pt"),
         ("int4_baseline", "int4_baseline", "integration/calibration/int4_calibration.pt")]


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
    return statistics.median(meds) * 1e3          # us


def kernels_of(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = []
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA or ev.self_device_time_total <= 0:
            continue
        nm = ev.key.split("(")[0]
        for p in ("void ", "at::native::", "(anonymous namespace)::"):
            nm = nm.replace(p, "")
        import re
        nm = re.sub(r"<[^<>]*>", "", nm)
        nm = re.sub(r"<.*", "", nm).strip(": ")
        if nm.startswith("_Z"):
            toks = [t for t in re.split(r"\d+", nm) if t]
            cand = [t for t in toks if re.match(r"^[A-Z][A-Za-z_]{9,}$", t)
                    and not t.startswith(("ZN", "Kernel"))]
            nm = re.sub(r"(INS_?|IN|EEE?)$", "", cand[0]) if cand else nm[:40]
        out.append({"kernel": nm[:52], "us": round(ev.self_device_time_total / iters, 2),
                    "calls": round(ev.count / iters, 2)})
    out.sort(key=lambda r: -r["us"])
    return out


def main():
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    import integration.benchmarks.benchmark_ldm as B
    results = {}
    for label, mode, calib in MODES:
        os.environ["MODIFF_QUANT_LINEAR"] = "0" if mode == "fp16" else "1"
        os.environ["MODIFF_QUANT_ATTN"] = "0" if mode == "fp16" else "1"
        for k in ("MODIFF_SDPA_BACKEND", "MODIFF_FP16_MATERIALIZED", "MODIFF_FLASH_GATE"):
            os.environ.pop(k, None)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                                  "models/ldm/lsun_churches256/model.ckpt",
                                  output_dir=f"{HERE}/tmp_out", batch_size=BATCH, steps=10,
                                  shape=(4, 32, 32), calibration_path=calib,
                                  linear_backend=("fp16" if mode == "fp16" else "int_gemm"))
            model, sampler = r._setup_model(mode)
            cond = r._cond_kwargs(model, BATCH)
            # one real sampling pass so any lazy calibration / autotune freezes
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True,
                                                           dtype=torch.float16):
                sampler.sample(S=10, batch_size=BATCH, shape=r.shape, eta=0.0,
                               verbose=False, **cond)

        ATTN = ("AttentionBlock", "TokenMajorAttentionBlock", "QuantizedStandardAttentionBlock")
        # Use the REAL (C,H,W) each block actually runs at, taken from layer_pipeline_bench.json
        # (captured by hooking a live sampling pass). An AttentionBlock accepts ANY spatial size,
        # so probing sizes until one "works" silently measures every block at 32x32 -- which is
        # wrong for the deeper levels and inflates their FLOP counts by up to 256x.
        real = json.load(open(f"{HERE}/data/layer_pipeline_bench.json"))
        want = [(r["x_shape"][1], r["x_shape"][2], r["x_shape"][3], r["n_instances"])
                for r in real["modes"]["fp16"] if r["kind"] == "attention"]
        by_c = {}
        for name, m in model.model.diffusion_model.named_modules():
            if type(m).__name__ in ATTN and hasattr(m, "channels"):
                by_c.setdefault(m.channels, []).append((name, m))

        rows = []
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            for C, HW, WW, n_inst in want:
                if C not in by_c:
                    continue
                name, blk = by_c[C][0]
                insts = [None] * n_inst
                if True:
                    x = torch.randn(BATCH, C, HW, WW, device="cuda", dtype=torch.float16)
                    x = x.contiguous(memory_format=torch.channels_last)
                    blk(x)
                    T = HW * WW
                    nh = blk.num_heads
                    hd = C // nh
                    full = bench(lambda: blk(x))
                    row = dict(mode=label, C=C, H=HW, W=WW, T=T, num_heads=nh, head_dim=hd,
                               n_instances=len(insts), full_us=round(full, 2),
                               gflop_qkv=round(2 * BATCH * T * C * 3 * C / 1e9, 2),
                               gflop_attn=round(4 * BATCH * nh * T * T * hd / 1e9, 2),
                               gflop_proj=round(2 * BATCH * T * C * C / 1e9, 2),
                               kernels=kernels_of(lambda: blk(x)))
                    row["gflop_total"] = round(row["gflop_qkv"] + row["gflop_attn"]
                                               + row["gflop_proj"], 2)
                    row["achieved_tflops"] = round(row["gflop_total"] / (full * 1e-6) / 1e3, 2)
                    rows.append(row)
                    print(f"{label:14s} C{C:4d} {HW:2d}x{WW:<2d} T={T:5d} nh={nh} hd={hd:3d} "
                          f"x{len(insts)}  {full:8.1f} us  "
                          f"[qkv {row['gflop_qkv']:6.1f} attn {row['gflop_attn']:7.1f} "
                          f"proj {row['gflop_proj']:6.1f} GFLOP]  {row['achieved_tflops']:5.1f} T/s")
                    del x
                    torch.cuda.empty_cache()
        results[label] = rows
        del model, sampler
        torch.cuda.empty_cache()

    with open(f"{HERE}/data/attn_stage_profile.json", "w") as f:
        json.dump(dict(batch=BATCH, modes=results), f, indent=2)
    print(f"\nWROTE {HERE}/data/attn_stage_profile.json")


if __name__ == "__main__":
    main()
