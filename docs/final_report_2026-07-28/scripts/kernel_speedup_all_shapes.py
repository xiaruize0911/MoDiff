"""Per-kernel speedup vs the fp16 reference, at EVERY shape the model actually runs.

Shapes come from data/all_shapes.json (captured by hooking the live UNet in
capture_all_shapes.py), not a hand-written list -- so coverage is complete by construction
and each row carries its real calls/step weight.

Families measured, each against the fp16 op it replaces:
  conv        : OptimizedInt8/Int4Conv2d (baseline + modiff)     vs  cuDNN fp16 nn.Conv2d
  groupnorm   : group_norm_silu_nhwc / _quantize / _quantize_pack vs  F.group_norm + F.silu
  resize      : upsample2x_quantize / avgpool2x_quantize (fused)  vs  interpolate/avg_pool + quantize
  concat      : cat2_channels_last_fp16                           vs  torch.cat
Timing: CUDA-event median of `iters` per round, median across rounds, after a clock burn-in
and per-shape warmup (same methodology as docs/benchmark_5mode_2026-07-20/scripts/conv_kernel.py).
Writes data/kernel_speedup_all_shapes.json.
"""
import os, sys, json, statistics
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F
import modiff_cutlass as mc
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d

torch.manual_seed(0)
dev = "cuda"
HERE = "docs/final_report_2026-07-28"
WARM, ITERS, ROUNDS = 30, 80, 5


def cuda_bench(fn, warm=WARM, iters=ITERS, rounds=ROUNDS):
    """CUDA-event median (us). Returns None if the op raises (unsupported shape)."""
    try:
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return None
    meds = []
    for _ in range(rounds):
        s = [torch.cuda.Event(True) for _ in range(iters)]
        e = [torch.cuda.Event(True) for _ in range(iters)]
        for i in range(iters):
            s[i].record(); fn(); e[i].record()
        torch.cuda.synchronize()
        t = sorted(s[i].elapsed_time(e[i]) for i in range(iters))
        meds.append(t[len(t) // 2])
    return statistics.median(meds) * 1e3


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def burn_in():
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    for _ in range(60):
        b = b @ b * 1e-4 + 1.0
    torch.cuda.synchronize()


# ---------------------------------------------------------------- conv
def bench_conv(shapes, batch):
    rows = []
    for sh in shapes:
        C, H, W, K, k, st, pad = sh["C"], sh["H"], sh["W"], sh["K"], sh["k"], sh["stride"], sh["pad"]
        N = batch
        conv = nn.Conv2d(C, K, k, stride=st, padding=pad, bias=True).cuda().eval()
        x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16))
        row = dict(family="conv", N=N, C=C, H=H, W=W, K=K, k=k, stride=st, pad=pad,
                   calls_per_step=sh["calls_per_step"])
        with torch.inference_mode():
            ref = conv.half().to(memory_format=torch.channels_last)
            row["fp16_us"] = cuda_bench(lambda: ref(x))
            for mode, Wrap in (("int8", OptimizedInt8Conv2d), ("int4", OptimizedInt4Conv2d)):
                for modiff in (False, True):
                    try:
                        opt = Wrap(nn.Conv2d(C, K, k, stride=st, padding=pad, bias=True).cuda().eval(),
                                   layer_name="bench").cuda().eval()
                        opt.set_static_scale(32.0)
                        opt.set_standard_output_fp16(True)
                        opt.enable_modiff(modiff)
                        us = cuda_bench(lambda: opt(x))
                    except Exception:
                        us = None
                    row[f"{mode}{'_modiff' if modiff else '_baseline'}_us"] = us
                    del opt
        rows.append(row)
        del conv, x
        torch.cuda.empty_cache()
        print(f"  conv N{N} C{C} {H}x{W} -> K{K} k{k} s{st}: "
              f"fp16={row['fp16_us']} int8={row.get('int8_baseline_us')} int4={row.get('int4_baseline_us')}")
    return rows


# ---------------------------------------------------------------- groupnorm (+SiLU [+quantize])
def bench_groupnorm(shapes, batch):
    rows = []
    for sh in shapes:
        C, H, W, G = sh["C"], sh["H"], sh["W"], sh["num_groups"]
        if G is None or C % G:
            continue
        N = batch
        x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16))
        g = torch.randn(C, device=dev, dtype=torch.float16)
        b = torch.randn(C, device=dev, dtype=torch.float16)
        scale = torch.tensor([127.0 / 3.0], device=dev, dtype=torch.float32)
        empty = torch.empty(0, device=dev, dtype=torch.float32)
        empty_h = torch.empty(0, device=dev, dtype=torch.float16)
        row = dict(family="groupnorm", N=N, C=C, H=H, W=W, num_groups=G,
                   calls_per_step=sh["calls_per_step"])
        with torch.inference_mode():
            # fp16 reference: what FusedGroupNormSiLU falls back to (autocast disabled)
            gf, bf = g.float(), b.float()
            row["fp16_gn_silu_us"] = cuda_bench(lambda: F.silu(F.group_norm(x, G, gf.half(), bf.half())))
            row["fused_gn_silu_us"] = cuda_bench(
                lambda: mc.group_norm_silu_nhwc(x, g, b, G, 1e-6, True, empty_h, empty_h))
            # GN+SiLU+int8 quantize, vs the two-kernel fp16 GN then standalone quantize
            row["fused_gn_silu_quant_int8_us"] = cuda_bench(
                lambda: mc.group_norm_silu_quantize_nhwc(x, g, b, G, 1e-6, True, scale, empty,
                                                         empty_h, empty_h))
            row["twostep_gn_then_quant_int8_us"] = cuda_bench(
                lambda: mc.step1_static_quantize_noahat_fprop(
                    cl(mc.group_norm_silu_nhwc(x, g, b, G, 1e-6, True, empty_h, empty_h)),
                    scale, empty))
            if C % 2 == 0 and (C // G) % 2 == 0:
                row["fused_gn_silu_quant_int4_us"] = cuda_bench(
                    lambda: mc.group_norm_silu_quantize_pack_nhwc(x, g, b, G, 1e-6, True, scale,
                                                                  empty, empty_h, empty_h))
                row["twostep_gn_then_quant_int4_us"] = cuda_bench(
                    lambda: mc.step1_static_quantize_pack_int4_noahat_fprop(
                        cl(mc.group_norm_silu_nhwc(x, g, b, G, 1e-6, True, empty_h, empty_h)),
                        scale, empty))
        rows.append(row)
        del x
        torch.cuda.empty_cache()
        print(f"  gn N{N} C{C} {H}x{W} G{G}: fp16={row['fp16_gn_silu_us']} "
              f"fused={row['fused_gn_silu_us']} fused+q8={row['fused_gn_silu_quant_int8_us']}")
    return rows


# ------------------------------------------------- resize+quantize fusions (this session)
def bench_resize(shapes, batch):
    """Fused resize+quantize vs the two-step (resize then quantize) path it replaced."""
    seen, rows = set(), []
    for sh in shapes:
        C, H, W = sh["C"], sh["H"], sh["W"]
        if (C, H, W) in seen:
            continue
        seen.add((C, H, W))
        N = batch
        scale = torch.tensor([127.0 / 3.0], device=dev, dtype=torch.float32)
        s4 = torch.tensor([7.0 / 3.0], device=dev, dtype=torch.float32)
        empty = torch.empty(0, device=dev, dtype=torch.float32)
        with torch.inference_mode():
            # --- upsample 2x (nearest) + quantize
            x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16))
            r = dict(family="resize_upsample", N=N, C=C, H=H, W=W)
            r["fused_int8_us"] = cuda_bench(lambda: mc.upsample2x_quantize_noahat_fprop(x, scale, empty))
            r["twostep_int8_us"] = cuda_bench(lambda: mc.step1_static_quantize_noahat_fprop(
                cl(F.interpolate(x, scale_factor=2, mode="nearest")), scale, empty))
            if C % 2 == 0:
                r["fused_int4_us"] = cuda_bench(
                    lambda: mc.upsample2x_quantize_pack_noahat_fprop(x, s4, empty))
                r["twostep_int4_us"] = cuda_bench(
                    lambda: mc.step1_static_quantize_pack_int4_noahat_fprop(
                        cl(F.interpolate(x, scale_factor=2, mode="nearest")), s4, empty))
            rows.append(r)
            # --- avg_pool 2x2 + quantize (needs even H,W)
            if H % 2 == 0 and W % 2 == 0:
                r2 = dict(family="resize_avgpool", N=N, C=C, H=H, W=W)
                r2["fused_int8_us"] = cuda_bench(
                    lambda: mc.avgpool2x_quantize_noahat_fprop(x, scale, empty))
                r2["twostep_int8_us"] = cuda_bench(lambda: mc.step1_static_quantize_noahat_fprop(
                    cl(F.avg_pool2d(x, 2, 2)), scale, empty))
                if C % 2 == 0:
                    r2["fused_int4_us"] = cuda_bench(
                        lambda: mc.avgpool2x_quantize_pack_noahat_fprop(x, s4, empty))
                    r2["twostep_int4_us"] = cuda_bench(
                        lambda: mc.step1_static_quantize_pack_int4_noahat_fprop(
                            cl(F.avg_pool2d(x, 2, 2)), s4, empty))
                rows.append(r2)
            del x
            torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------ skip-concat (decoder), vs torch.cat
def bench_concat(batch):
    """Real decoder skip-concat channel pairs, cat2_channels_last_fp16 vs torch.cat."""
    PAIRS = [(768, 768, 2, 2), (768, 384, 4, 4), (384, 384, 4, 4),
             (384, 192, 8, 8), (384, 192, 16, 16), (192, 192, 32, 32)]
    rows = []
    for (C1, C2, H, W) in PAIRS:
        N = batch
        a = cl(torch.randn(N, C1, H, W, device=dev, dtype=torch.float16))
        b = cl(torch.randn(N, C2, H, W, device=dev, dtype=torch.float16))
        with torch.inference_mode():
            rows.append(dict(family="skip_concat", N=N, C1=C1, C2=C2, H=H, W=W,
                             torch_cat_us=cuda_bench(lambda: torch.cat([a, b], dim=1)),
                             fused_us=cuda_bench(lambda: mc.cat2_channels_last_fp16(a, b))))
        del a, b
        torch.cuda.empty_cache()
        print(f"  concat C{C1}+{C2} {H}x{W}: cat={rows[-1]['torch_cat_us']} "
              f"fused={rows[-1]['fused_us']}")
    return rows


def main():
    shapes = json.load(open(f"{HERE}/data/all_shapes.json"))
    batch = int(os.environ.get("KBENCH_BATCH", "128"))
    burn_in()
    out = {"batch": batch}
    print("[conv]");      out["conv"] = bench_conv(shapes["conv"], batch)
    print("[groupnorm]"); out["groupnorm"] = bench_groupnorm(shapes["groupnorm"], batch)
    print("[resize]");    out["resize"] = bench_resize(shapes["groupnorm"], batch)
    print("[concat]");    out["concat"] = bench_concat(batch)
    with open(f"{HERE}/data/kernel_speedup_all_shapes.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {HERE}/data/kernel_speedup_all_shapes.json")


if __name__ == "__main__":
    main()
