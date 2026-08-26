"""What does the conv charge to absorb MoDiff's a_hat update?

Arm A  conv2d_int8_evt_o_hat            -- the conv as shipped
Arm B  conv2d_int8_evt_o_hat_ahat       -- same conv, plus `a_hat += code/scale` CTA-partitioned
                                           inside the kernel

`code` is the conv's OWN input tensor -- the int8 delta the GN wrote -- so arm B re-reads a tensor
the mainloop is already streaming. Traffic added per element: read code 1 B + read a_hat 2 B +
write a_hat 2 B = 5 B, against the 2 B (+ RFO) the GN gives back. The bet is that the conv, at
23-25% of peak bandwidth with every SM busy, absorbs 5 B more cheaply than the GN pays for 2.
"""
import os, sys, json, statistics
import torch
sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")
import modiff_cutlass as mc

torch.manual_seed(0)
N = 128
TRIALS, REPS, WARMUP = 5, 30, 10
# (Cin, Cout, H, W, calls/step) -- the five shapes carrying 63% of MoDiff's conv-block overhead
SHAPES = [(192, 192, 32, 32, 7), (384, 384, 16, 16, 7), (384, 192, 32, 32, 2),
          (576, 192, 32, 32, 1), (768, 384, 16, 16, 2)]

def time_arm(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(REPS): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / REPS

out = []
for Cin, Cout, H, W, freq in SHAPES:
    x = torch.randint(-8, 8, (N, Cin, H, W), device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
    wt = torch.randint(-8, 8, (Cout, 3, 3, Cin), device="cuda", dtype=torch.int8).contiguous()
    inv_scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    wscales = torch.ones(Cout, device="cuda", dtype=torch.float32)
    o_hat = torch.zeros(N, Cout, H, W, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    a_hat = (0.1 * torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    code = x.permute(0, 2, 3, 1).contiguous().view(-1)

    A = lambda: mc.conv2d_int8_evt_o_hat(x, wt, inv_scale, wscales, o_hat, 1, 1, 1, 1, 1, 1)
    B = lambda: mc.conv2d_int8_evt_o_hat_ahat(x, wt, inv_scale, wscales, o_hat, a_hat, code,
                                              1.0 / 64.0, 1, 1, 1, 1, 1, 1, False)
    Bp = lambda: mc.conv2d_int8_evt_o_hat_ahat(x, wt, inv_scale, wscales, o_hat, a_hat, code,
                                               1.0 / 64.0, 1, 1, 1, 1, 1, 1, True)
    fns = {"conv": A, "conv+ahat": B, "conv+ahat_post": Bp}
    names = list(fns)
    samples = {n: [] for n in names}
    for t in range(TRIALS):
        for n in (names[t % len(names):] + names[:t % len(names)]):
            samples[n].append(time_arm(fns[n]))
    med = {n: statistics.median(v) for n, v in samples.items()}
    sd = {n: statistics.stdev(v) for n, v in samples.items()}
    cost = min(med["conv+ahat"], med["conv+ahat_post"]) - med["conv"]
    row = dict(shape=f"{Cin}->{Cout},{H}x{W}", freq=freq, med=med, conv_cost_ms=cost,
               cost_pct=100 * cost / med["conv"],
               sd_max_pct=100 * max(sd[n] / med[n] for n in names))
    out.append(row)
    print(f"{row['shape']:>18} freq{freq:>3} | conv {med['conv']:.4f} conv+ahat {med['conv+ahat']:.4f} "
          f"post {med['conv+ahat_post']:.4f} | best charge {cost:+.4f} ms ({row['cost_pct']:+.1f}%) | sd<{row['sd_max_pct']:.2f}%")
    del x, wt, o_hat, a_hat; torch.cuda.empty_cache()

json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/result_conv.json", "w"), indent=1)

# the GN side, from bench.py, over the same five shapes
gn = {r["shape"]: r for r in json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/result.json"))}
key = {"192->192,32x32": "192,32x32", "384->384,16x16": "384,16x16", "384->192,32x32": "384,32x32",
       "576->192,32x32": "576,32x32", "768->384,16x16": "768,16x16"}
saved = sum(gn[key[r["shape"]]]["saved_ms"] * r["freq"] for r in out)
paid = sum(r["conv_cost_ms"] * r["freq"] for r in out)
print(f"\nfreq-weighted over the 5 shapes:")
print(f"  GN gives back : {saved:+.3f} ms/step")
print(f"  conv charges  : {paid:+.3f} ms/step")
print(f"  NET           : {saved - paid:+.3f} ms/step = {100*(saved-paid)/77.0005:+.2f}% of the W8A8 step")
