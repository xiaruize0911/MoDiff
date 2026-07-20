"""Kernel items 1 (speed) — conv kernels across the 5 modes on the churches UNet conv shapes, b128.
fp16 (cuDNN) / int8 no-cache / int4 no-cache / int8+modiff-cache / int4+modiff-cache.
modiff = temporal-delta cache (step1 quantize + conv o_hat + accumulate) via enable_modiff(True).
CUDA-event median, warm 50 + 200 iters x 5 rounds, synchronize each round. Writes data/conv_kernel_speed.csv.
"""
import os, sys, csv, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d

torch.manual_seed(0); dev = "cuda"
B = 128
HERE = "docs/benchmark_5mode_2026-07-20"
# (name, Cin, H, W, Cout, K, stride, pad) — churches ResBlock convs (batch overridden to B)
CONV_SHAPES = [
    ("res_128_64", 128, 64, 64, 128, 3, 1, 1), ("res_128_32", 128, 32, 32, 128, 3, 1, 1),
    ("down_128_256_32", 128, 32, 32, 256, 3, 1, 1), ("res_256_32", 256, 32, 32, 256, 3, 1, 1),
    ("res_256_16", 256, 16, 16, 256, 3, 1, 1), ("down_256_512_16", 256, 16, 16, 512, 3, 1, 1),
    ("mid_512_8", 512, 8, 8, 512, 3, 1, 1), ("up_512_256_16", 512, 16, 16, 256, 3, 1, 1),
    ("up_256_128_32", 256, 32, 32, 128, 3, 1, 1), ("up_128_64", 128, 64, 64, 128, 3, 1, 1),
]
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]


def cuda_bench(fn, warm=50, iters=200, rounds=5):
    meds = []
    for _ in range(rounds):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        s = [torch.cuda.Event(True) for _ in range(iters)]; e = [torch.cuda.Event(True) for _ in range(iters)]
        for i in range(iters):
            s[i].record(); fn(); e[i].record()
        torch.cuda.synchronize()
        t = sorted(s[i].elapsed_time(e[i]) for i in range(iters)); meds.append(t[len(t) // 2])
    return statistics.median(meds) * 1e3          # us


def conv_tops(Cin, H, W, Cout, K, st, pad, us):
    Ho = (H + 2 * pad - K) // st + 1; Wo = (W + 2 * pad - K) // st + 1
    return 2 * B * Ho * Wo * Cout * Cin * K * K / (us * 1e-6) / 1e12


def make(mode, Cin, H, W, Cout, K, st, pad):
    conv = nn.Conv2d(Cin, Cout, K, stride=st, padding=pad, bias=True).cuda().eval()
    x = torch.randn(B, Cin, H, W, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    if mode == "fp16":
        c = conv.half().to(memory_format=torch.channels_last)
        return lambda: c(x)
    Wrap = OptimizedInt8Conv2d if "int8" in mode else OptimizedInt4Conv2d
    opt = Wrap(conv, layer_name="bench").cuda().eval()
    opt.set_static_scale(32.0); opt.set_standard_output_fp16(True)
    opt.enable_modiff("modiff" in mode)
    return lambda: opt(x)


# clock burn-in
_b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _b = _b @ _b * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
print(f"Conv kernel speed @ b{B} (us, median 5x200)\n")
print(f"{'shape':18} " + " ".join(f"{m:>14}" for m in MODES))
for (name, Cin, H, W, Cout, K, st, pad) in CONV_SHAPES:
    rec = {"shape": name, "Cin": Cin, "Cout": Cout, "HW": H}
    us = {}
    for mode in MODES:
        try:
            fn = make(mode, Cin, H, W, Cout, K, st, pad)
            fn(); torch.cuda.synchronize()
            us[mode] = cuda_bench(fn)
        except Exception as ex:
            us[mode] = None; print(f"  {name} {mode}: N/A ({type(ex).__name__}: {ex})")
    f = us["fp16"]
    for mode in MODES:
        rec[f"{mode}_us"] = round(us[mode], 1) if us[mode] else ""
        rec[f"{mode}_vs_fp16"] = round(f / us[mode], 3) if (us[mode] and f) else ""
    rec["fp16_tops"] = round(conv_tops(Cin, H, W, Cout, K, st, pad, us["fp16"]), 1) if us["fp16"] else ""
    rows.append(rec)
    print(f"{name:18} " + " ".join(f"{(us[m] if us[m] else float('nan')):14.1f}" for m in MODES))

with open(f"{HERE}/data/conv_kernel_speed.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nWROTE {HERE}/data/conv_kernel_speed.csv")
