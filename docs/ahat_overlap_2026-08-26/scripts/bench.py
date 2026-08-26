"""Does eliding the a_hat WRITE actually save what the byte model says it should?

The proposed design moves `a_hat += q/scale` out of the bandwidth-bound GN kernel and into the
conv's mainloop, where o_hat's bytes are measured 2.35x/4.06x cheaper. Before touching CUTLASS,
this measures the GN side alone: what the GN kernel gives back if the write disappears. That is
the NUMERATOR of the whole idea -- if it is small, the design dies here for free.

Arms (same kernel body, template flags):
    w1c1  today: read x, read a_hat, write a_hat, write code
    w0c1  proposed GN side: read x, read a_hat, write code
    w1c0 / w0c0  bracketing arms
Order is rotated across trials, medians reported -- the methodology that caught the sign-flipping
o_hat reading in the 08-26 session.
"""
import os, sys, json, statistics, itertools
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")
import ahat_probe

sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")
import modiff_cutlass as mc

torch.manual_seed(0)
N, G = 128, 32
PEAK = 696.0
TRIALS, REPS, WARMUP = 5, 30, 10

# The five shapes that carry 63% of MoDiff's conv-block overhead, plus two references.
# (Cin at the GN's resolution, H, W, calls/step)
SHAPES = [
    (192, 32, 32, 7), (384, 16, 16, 7), (384, 32, 32, 2),
    (576, 32, 32, 1), (768, 16, 16, 2), (768, 2, 2, 12), (384, 8, 8, 8),
]
ARMS = [("w1c1", True, True), ("w0c1", False, True), ("w1c0", True, False), ("w0c0", False, False)]


def make(C, H, W):
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    a = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    yq = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
    gamma = torch.ones(C, device="cuda", dtype=torch.float16)
    beta = torch.zeros(C, device="cuda", dtype=torch.float16)
    mean = torch.zeros(N * G, device="cuda", dtype=torch.float32)
    inv_std = torch.ones(N * G, device="cuda", dtype=torch.float32)
    scale = torch.tensor([64.0], device="cuda", dtype=torch.float32)
    return x, a, yq, gamma, beta, mean, inv_std, scale


def time_arm(fn, a0, a):
    a.copy_(a0)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(REPS):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / REPS


out = []
for C, H, W in [(c, h, w) for c, h, w, _ in SHAPES]:
    freq = dict(((c, h, w), f) for c, h, w, f in SHAPES)[(C, H, W)]
    x, a, yq, gamma, beta, mean, inv_std, scale = make(C, H, W)
    a0 = a.clone()
    ne = N * C * H * W
    ss = C * H * W

    def mk(wa, wc):
        return lambda: ahat_probe.probe_launch(x, a, yq, gamma, beta, mean, inv_std, scale,
                                               C, G, ss, ne, True, False, wa, wc)
    fns = {name: mk(wa, wc) for name, wa, wc in ARMS}
    # the shipped full path (stats pass + this apply kernel), for scale
    empty16 = torch.empty(0, device="cuda", dtype=torch.float16)
    empty32 = torch.empty(0, device="cuda", dtype=torch.float32)
    empty_i32 = torch.empty(0, device="cuda", dtype=torch.int32)
    def prod():
        mc.group_norm_silu_delta_quantize_nhwc(x, gamma, beta, a, G, 1e-5, True,
                                               scale, empty32, empty16, empty16,
                                               empty32, empty32, empty32, empty_i32,
                                               127.0, False, 1.0)
    fns["prod"] = prod

    names = [n for n, _, _ in ARMS] + ["prod"]
    samples = {n: [] for n in names}
    for t in range(TRIALS):
        order = names[t % len(names):] + names[:t % len(names)]   # rotate
        for n in order:
            samples[n].append(time_arm(fns[n], a0, a))
    med = {n: statistics.median(v) for n, v in samples.items()}
    sd = {n: statistics.stdev(v) for n, v in samples.items()}

    row = dict(shape=f"{C},{H}x{W}", C=C, H=H, freq=freq, elem=ne,
               med=med, sd_max_pct=100 * max(sd[n] / med[n] for n in names))
    # bytes/elem in the apply kernel: read x 2 + read a_hat 2 + [write a_hat 2] + [write code 1]
    row["gbs_w1c1"] = ne * 7 / (med["w1c1"] * 1e-3) / 1e9
    row["gbs_w0c1"] = ne * 5 / (med["w0c1"] * 1e-3) / 1e9
    row["saved_ms"] = med["w1c1"] - med["w0c1"]
    row["saved_pct_of_apply"] = 100 * (med["w1c1"] - med["w0c1"]) / med["w1c1"]
    row["saved_pct_of_prod_gn"] = 100 * (med["w1c1"] - med["w0c1"]) / med["prod"]
    row["apply_pct_of_prod"] = 100 * med["w1c1"] / med["prod"]
    out.append(row)
    print(f"{row['shape']:>12} freq{freq:>3} | apply w1c1 {med['w1c1']:.4f} w0c1 {med['w0c1']:.4f} "
          f"w1c0 {med['w1c0']:.4f} w0c0 {med['w0c0']:.4f} | prod {med['prod']:.4f} | "
          f"saved {row['saved_ms']:.4f} ms ({row['saved_pct_of_apply']:.1f}% of apply, "
          f"{row['saved_pct_of_prod_gn']:.1f}% of GN) | BW {row['gbs_w1c1']:.0f}->{row['gbs_w0c1']:.0f} GB/s "
          f"({100*row['gbs_w1c1']/PEAK:.0f}%->{100*row['gbs_w0c1']/PEAK:.0f}% peak) | sd<{row['sd_max_pct']:.2f}%")
    del x, a, a0, yq
    torch.cuda.empty_cache()

json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/result.json", "w"), indent=1)

tot_saved = sum(r["saved_ms"] * r["freq"] for r in out[:5])
tot_prod = sum(r["med"]["prod"] * r["freq"] for r in out[:5])
print(f"\nfreq-weighted over the 5 dominant shapes: saved {tot_saved:.3f} ms/step "
      f"of {tot_prod:.3f} ms/step of MoDiff GN ({100*tot_saved/tot_prod:.1f}%)")
tot_saved_all = sum(r["saved_ms"] * r["freq"] for r in out)
print(f"over all 7 measured shapes: saved {tot_saved_all:.3f} ms/step")
