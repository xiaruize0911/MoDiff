"""INT4 twin of bench.py: same probe body, packed 4-bit code store (0.5 B/elem instead of 1 B).
Validated against the shipped group_norm_silu_delta_quantize_pack_nhwc and the committed
W4A4 rows of the conv-block ablation."""
import os, sys, json, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")
import ahat_probe
sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")
import modiff_cutlass as mc

torch.manual_seed(0)
N, G, PEAK = 128, 32, 696.0
TRIALS, REPS, WARMUP = 5, 30, 10
SHAPES = [(192, 32, 32, 7), (384, 16, 16, 7), (384, 32, 32, 2),
          (576, 32, 32, 1), (768, 16, 16, 2), (768, 2, 2, 12), (384, 8, 8, 8)]
ARMS = [("w1c1", True, True), ("w0c1", False, True)]

def time_arm(fn, a0, a):
    a.copy_(a0)
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(REPS): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / REPS

out = []
for C, H, W, freq in SHAPES:
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    a = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    yq = torch.empty(N * C * H * W // 2, device="cuda", dtype=torch.int8)
    gamma = torch.ones(C, device="cuda", dtype=torch.float16)
    beta = torch.zeros(C, device="cuda", dtype=torch.float16)
    mean = torch.zeros(N * G, device="cuda", dtype=torch.float32)
    inv_std = torch.ones(N * G, device="cuda", dtype=torch.float32)
    scale = torch.tensor([64.0], device="cuda", dtype=torch.float32)
    e16 = torch.empty(0, device="cuda", dtype=torch.float16)
    e32 = torch.empty(0, device="cuda", dtype=torch.float32)
    ei = torch.empty(0, device="cuda", dtype=torch.int32)
    a0 = a.clone(); ne = N * C * H * W; ss = C * H * W

    def mk(wa, wc):
        return lambda: ahat_probe.probe_launch(x, a, yq, gamma, beta, mean, inv_std, scale,
                                               C, G, ss, ne, True, True, wa, wc, True)
    fns = {n: mk(wa, wc) for n, wa, wc in ARMS}
    fns["prod"] = lambda: mc.group_norm_silu_delta_quantize_pack_nhwc(
        x, gamma, beta, a, G, 1e-5, True, scale, e32, e16, e16, e32, e32, e32, ei, 7.0, False, 1.0)
    names = ["w1c1", "w0c1", "prod"]
    samples = {n: [] for n in names}
    for t in range(TRIALS):
        for n in names[t % 3:] + names[:t % 3]:
            samples[n].append(time_arm(fns[n], a0, a))
    med = {n: statistics.median(v) for n, v in samples.items()}
    sd = {n: statistics.stdev(v) for n, v in samples.items()}
    saved = med["w1c1"] - med["w0c1"]
    row = dict(shape=f"{C},{H}x{W}", C=C, H=H, freq=freq, med=med, saved_ms=saved,
               saved_pct_apply=100 * saved / med["w1c1"], saved_pct_gn=100 * saved / med["prod"],
               gbs_w1c1=ne * 6.5 / (med["w1c1"] * 1e-3) / 1e9,
               gbs_w0c1=ne * 4.5 / (med["w0c1"] * 1e-3) / 1e9,
               sd_max_pct=100 * max(sd[n] / med[n] for n in names))
    out.append(row)
    print(f"{row['shape']:>12} freq{freq:>3} | w1c1 {med['w1c1']:.4f} w0c1 {med['w0c1']:.4f} | prod {med['prod']:.4f} | "
          f"saved {saved:.4f} ms ({row['saved_pct_apply']:.1f}% apply, {row['saved_pct_gn']:.1f}% GN) | "
          f"BW {row['gbs_w1c1']:.0f}->{row['gbs_w0c1']:.0f} ({100*row['gbs_w1c1']/PEAK:.0f}%->{100*row['gbs_w0c1']/PEAK:.0f}%) | sd<{row['sd_max_pct']:.2f}%")
    del x, a, a0, yq; torch.cuda.empty_cache()

json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/result_int4.json", "w"), indent=1)
tw = sum(r["saved_ms"] * r["freq"] for r in out)
print(f"\nfreq-weighted saving, 7 shapes: {tw:.3f} ms/step = {100*tw/68.2706:.2f}% of the W4A4 step (68.27 ms)")
