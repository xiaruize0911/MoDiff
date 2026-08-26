import os, sys, json, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")
import gn_stats_probe as P

torch.manual_seed(0)
N, G, PEAK = 128, 32, 696.0
TRIALS, REPS, WARMUP = 5, 30, 10
SHAPES = [(768, 2, 2, 12), (384, 8, 8, 8), (384, 16, 16, 7), (768, 4, 4, 7), (192, 32, 32, 7), (1536, 2, 2, 3), (768, 8, 8, 2), (1536, 4, 4, 2), (384, 32, 32, 2), (768, 16, 16, 2), (384, 4, 4, 1), (1152, 8, 8, 1), (1152, 4, 4, 1), (192, 16, 16, 1), (576, 32, 32, 1), (768, 8, 8, 1), (576, 16, 16, 1), (384, 32, 32, 1)]
NAMES = {0: "v0 shipped", 1: "v1 hw-unroll", 2: "v2 vec2-chan", 3: "v3 both"}

def time_v(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(REPS): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / REPS

out = []
print("%-12s %5s | %10s %10s %10s %10s | %8s %8s | exact" %
      ("shape", "freq", *NAMES.values(), "best", "speedup"))
for C, H, W, freq in SHAPES:
    HW = H * W
    nblocks = min(HW, 32)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    ps = {v: torch.empty(N * G * nblocks, device="cuda", dtype=torch.float32) for v in range(4)}
    pq = {v: torch.empty(N * G * nblocks, device="cuda", dtype=torch.float32) for v in range(4)}
    fns = {v: (lambda v=v: P.launch(x, ps[v], pq[v], N, C, HW, G, nblocks, v)) for v in range(4)}
    for v in range(4): fns[v]()
    torch.cuda.synchronize()
    exact = all(torch.equal(ps[v], ps[0]) and torch.equal(pq[v], pq[0]) for v in range(1, 4))
    med = {}
    samples = {v: [] for v in range(4)}
    for t in range(TRIALS):
        for v in ([0,1,2,3][t % 4:] + [0,1,2,3][:t % 4]):
            samples[v].append(time_v(fns[v]))
    med = {v: statistics.median(s) for v, s in samples.items()}
    best = min(med, key=med.get)
    ne = N * C * HW
    row = dict(shape=f"{C},{H}x{W}", freq=freq, med={NAMES[v]: med[v] for v in med},
               best=NAMES[best], speedup=med[0] / med[best], bit_exact=bool(exact),
               gbs_v0=ne*2/(med[0]*1e-3)/1e9, gbs_best=ne*2/(med[best]*1e-3)/1e9)
    out.append(row)
    print("%-12s %5d | %10.4f %10.4f %10.4f %10.4f | %8s %7.3fx | %s  (%.0f%%->%.0f%% peak)" %
          (row["shape"], freq, med[0], med[1], med[2], med[3], NAMES[best], row["speedup"],
           "YES" if exact else "NO", 100*row["gbs_v0"]/PEAK, 100*row["gbs_best"]/PEAK))
    del x, ps, pq; torch.cuda.empty_cache()

json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/result.json", "w"), indent=1)
for v in range(4):
    tot = sum(r["med"][NAMES[v]] * r["freq"] for r in out)
    print("%-14s freq-weighted total: %7.3f ms/step" % (NAMES[v], tot))
t0 = sum(r["med"]["v0 shipped"] * r["freq"] for r in out)
tb = sum(min(r["med"].values()) * r["freq"] for r in out)
print("\nper-shape best: %.3f ms/step, saving %.3f (%.1f%% of the stats pass)" % (tb, t0-tb, 100*(t0-tb)/t0))
