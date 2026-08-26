"""Time the vec4 apply-kernel probe against the shipped vec2 kernel, on every real conv shape
where vec4 is structurally safe (CPG % 4 == 0 -- see probe_vec4.cu's header for why), and confirm
bit-identity first. Reuses the vec2 probe from ahat_overlap_2026-08-26/scripts/probe.cu (built
separately; see build_probe.py there) and probe_vec4.cu in this folder.

This is a MEASUREMENT ONLY. Nothing here is wired into csrc/ -- the finding was recorded in
FINDINGS.md and NOT landed, at the project owner's request (small win, 14 real shapes only cover
50 of ~62 calls/step, and the remaining 12 calls -- C=192/576 -- would need a fallback path not
built here).

Run: python bench_vec4_vs_vec2.py
Requires both probes already built (see build_probe.py in ahat_overlap_2026-08-26/scripts/ for the
vec2 probe, build_probe_vec4.py in this folder for vec4).
"""
import statistics
import sys

import torch

VEC2_BUILD = "/tmp/claude-0/-workspace/31e575da-69cf-419d-bc20-66eb029653e9/scratchpad/ahat_probe/build"
VEC4_BUILD = "/tmp/claude-0/-workspace/31e575da-69cf-419d-bc20-66eb029653e9/scratchpad/ahat_probe/build_vec4"
sys.path.insert(0, VEC2_BUILD)
sys.path.insert(0, VEC4_BUILD)
import ahat_probe as P2       # noqa: E402
import ahat_probe_vec4 as P4  # noqa: E402

torch.manual_seed(0)
N, G = 128, 32
CL = torch.channels_last
TRIALS, REPS, WARMUP = 5, 30, 10

# every real (Cin, H, W, freq) from the committed ablation CSV where CPG = Cin/32 is a multiple
# of 4 -- the structural precondition probe_vec4.cu's TORCH_CHECK enforces.
SHAPES = [(768, 2, 2, 12), (384, 8, 8, 8), (384, 16, 16, 7), (768, 4, 4, 7), (1536, 2, 2, 3),
          (768, 8, 8, 2), (1536, 4, 4, 2), (384, 32, 32, 2), (768, 16, 16, 2), (384, 4, 4, 1),
          (1152, 8, 8, 1), (1152, 4, 4, 1), (768, 8, 8, 1), (384, 32, 32, 1)]


def timeit(fn, a, a0):
    a.copy_(a0)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(REPS):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / REPS


def make(C, H, W):
    HW = H * W
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=CL)
    gm = torch.randn(C, device="cuda", dtype=torch.float16).abs() + 0.5
    bt = torch.randn(C, device="cuda", dtype=torch.float16) * 0.1
    mean = torch.zeros(N * G, device="cuda", dtype=torch.float32)
    istd = torch.ones(N * G, device="cuda", dtype=torch.float32)
    sc = torch.tensor([16.0], device="cuda", dtype=torch.float32)
    yq = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=CL)
    return x, a0, gm, bt, mean, istd, sc, yq, N * C * HW, C * HW


print("=== bit-identity check ===")
for C, H, W, _ in SHAPES[:4]:
    x, a0, gm, bt, mean, istd, sc, yq, ne, ss = make(C, H, W)
    a2, yq2 = a0.clone(), yq.clone()
    P2.probe_launch(x, a2, yq2, gm, bt, mean, istd, sc, C, G, ss, ne, True, False, True, True, False, 1)
    a4, yq4 = a0.clone(), yq.clone()
    P4.probe_launch_vec4(x, a4, yq4, gm, bt, mean, istd, sc, C, G, ss, ne, True, False, True, True)
    torch.cuda.synchronize()
    print(f"  {C},{H}x{W}: a_hat equal={torch.equal(a2, a4)}  code equal={torch.equal(yq2, yq4)}")

print("\n=== timing, freq-weighted over all 14 vec4-eligible real shapes ===")
print("%-14s %5s | %9s %9s | %7s" % ("shape", "freq", "vec2 ms", "vec4 ms", "speedup"))
tot2 = tot4 = 0.0
for C, H, W, freq in SHAPES:
    x, a0, gm, bt, mean, istd, sc, yq, ne, ss = make(C, H, W)
    a = a0.clone()

    def f2():
        P2.probe_launch(x, a, yq, gm, bt, mean, istd, sc, C, G, ss, ne, True, False, True, True, False, 1)

    def f4():
        P4.probe_launch_vec4(x, a, yq, gm, bt, mean, istd, sc, C, G, ss, ne, True, False, True, True)

    s2, s4 = [], []
    for t in range(TRIALS):
        order = ["v2", "v4"] if t % 2 == 0 else ["v4", "v2"]
        for tag in order:
            (s2 if tag == "v2" else s4).append(timeit(f2 if tag == "v2" else f4, a, a0))
    m2, m4 = statistics.median(s2), statistics.median(s4)
    tot2 += m2 * freq
    tot4 += m4 * freq
    print("%-14s %5d | %9.4f %9.4f | %6.3fx" % (f"{C},{H}x{W}", freq, m2, m4, m2 / m4))

print()
print("freq-weighted (50 vec4-eligible calls/step): vec2 %.3f ms/step  vec4 %.3f ms/step  "
      "(%+.1f%%, saving %.3f ms/step)" % (tot2, tot4, 100 * (tot4 - tot2) / tot2, tot2 - tot4))
