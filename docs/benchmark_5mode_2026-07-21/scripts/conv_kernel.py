"""Conv kernel speed — EVERY conv geometry the churches UNet runs (33 shapes), 5 modes, b128.

Shapes + per-step call counts come from the real-model enumeration (shapes.conv_shapes()). For
each geometry we measure the per-call kernel time and multiply by its per-step count to get the
per-step time contribution. Modes:
  fp16                 = cuDNN fp16 (channels_last)
  int8/int4 _baseline  = OptimizedInt{8,4}Conv2d, modiff OFF  (latest deep-fuse store kernel)
  int8/int4 _modiff    = OptimizedInt{8,4}Conv2d, modiff ON   (temporal-delta o_hat conv + accumulate)
Conv layers that stay fp16 in quant modes (skip / 1x1 pointwise / in_channels<32 / final out —
quant_eligible=False) run cuDNN in every mode, so their quant-mode time == fp16 time.

CUDA-event median, 50 warm + 200 iters x 5 rounds, GPU clock burn-in. The Optimized conv is
configured exactly as the model wires it: static scale + standard_output_fp16 + enable_modiff.
Writes data/conv_kernel_speed.csv (+ per-step rollup rows).
"""
import os, sys, csv, statistics
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "docs/benchmark_5mode_2026-07-21/scripts")
import torch, torch.nn as nn
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d
import shapes as S

torch.manual_seed(0); dev = "cuda"
B = 128
HERE = "docs/benchmark_5mode_2026-07-21"
MODES = S.MODES
torch.backends.cudnn.benchmark = True


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


# GPU clock burn-in
_b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _b = _b @ _b * 1e-4 + 1.0
torch.cuda.synchronize()

conv_shapes = sorted(S.conv_shapes(), key=lambda s: (-s["count"], -s["Cin"] * s["H"] * s["W"]))
rows = []
perstep = {m: 0.0 for m in MODES}
print(f"Conv kernel speed @ b{B} (us/call, median 5x200) — {len(conv_shapes)} geometries\n")
print(f"{'geometry':30}{'cnt':>4}{'quant':>6} | " + " ".join(f"{m.split('_')[0][:4]+m.split('_')[-1][:1]:>9}" for m in MODES))
for sh in conv_shapes:
    Cin, H, W, Cout, K, st, pad, cnt, qe = (sh["Cin"], sh["H"], sh["W"], sh["Cout"], sh["K"],
                                            sh["stride"], sh["pad"], sh["count"], sh["quant_eligible"])
    geom = f"{Cin}->{Cout} {H}x{W} K{K}s{st}"
    us = {}
    # fp16 always
    fn = make("fp16", Cin, H, W, Cout, K, st, pad); fn(); torch.cuda.synchronize()
    us["fp16"] = cuda_bench(fn)
    for mode in MODES[1:]:
        if not qe:
            us[mode] = us["fp16"]          # stays cuDNN fp16 in quant modes
            continue
        try:
            fn = make(mode, Cin, H, W, Cout, K, st, pad); fn(); torch.cuda.synchronize()
            us[mode] = cuda_bench(fn)
        except Exception as ex:
            us[mode] = None; print(f"  {geom} {mode}: N/A ({type(ex).__name__}: {ex})")
    rec = dict(Cin=Cin, H=H, W=W, Cout=Cout, K=K, stride=st, pad=pad, count_per_step=cnt,
               quant_eligible=int(qe))
    f16 = us["fp16"]
    for mode in MODES:
        rec[f"{mode}_us"] = round(us[mode], 1) if us[mode] else ""
        rec[f"{mode}_us_per_step"] = round(us[mode] * cnt, 1) if us[mode] else ""
        rec[f"{mode}_vs_fp16"] = round(f16 / us[mode], 3) if (us[mode] and f16) else ""
        if us[mode]: perstep[mode] += us[mode] * cnt
    rows.append(rec)
    print(f"{geom:30}{cnt:>4}{('Y' if qe else '-'):>6} | " +
          " ".join(f"{(us[m] if us[m] else float('nan')):9.1f}" for m in MODES))

# rollup: per-step total (us) + as ms + speedup vs fp16
roll = dict(Cin="TOTAL_PER_STEP", H="", W="", Cout="", K="", stride="", pad="",
            count_per_step=sum(s["count"] for s in conv_shapes), quant_eligible="")
print(f"\n=== conv per-step total (us, sum count x us/call) ===")
for m in MODES:
    roll[f"{m}_us"] = ""
    roll[f"{m}_us_per_step"] = round(perstep[m], 1)
    roll[f"{m}_vs_fp16"] = round(perstep["fp16"] / perstep[m], 3) if perstep[m] else ""
    print(f"  {m:16} {perstep[m]/1000:8.3f} ms/step  {perstep['fp16']/perstep[m]:.2f}x")
rows.append(roll)

cols = ["Cin", "H", "W", "Cout", "K", "stride", "pad", "count_per_step", "quant_eligible"]
for m in MODES: cols += [f"{m}_us", f"{m}_us_per_step", f"{m}_vs_fp16"]
with open(f"{HERE}/data/conv_kernel_speed.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"\nWROTE {HERE}/data/conv_kernel_speed.csv")
