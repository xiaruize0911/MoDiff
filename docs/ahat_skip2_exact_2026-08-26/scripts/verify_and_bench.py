"""Real CUDA-level check for the K=2 deferred-write a_hat scheme derived and numerically
(Python-level) verified in FINDINGS.md: (1) is it actually bit-identical to the standard kernel
when run on real hardware in fp16, at the real kernel's launch geometry, and (2) does the modelled
~0.564 ms/step net saving survive contact with a real kernel, or does the extra code-read / kernel
overhead eat more than the model assumed.

Two steps of a window are simulated with DIFFERENT x / mean / inv_std / scale (a real window has
different activations and a different per-step scale each step -- reusing the same tensor for both
steps would not exercise the interesting case).

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/verify_and_bench.py
(build the extension first: python docs/ahat_skip2_exact_2026-08-26/scripts/build_probe.py)
"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")
import ahat_skip2_probe as probe  # noqa: E402

torch.manual_seed(0)
N, G = 128, 32
PEAK = 696.0
TRIALS, REPS, WARMUP = 5, 30, 10

SHAPES = [
    (192, 32, 32, 7), (384, 16, 16, 7), (384, 32, 32, 2),
    (576, 32, 32, 1), (768, 16, 16, 2), (768, 2, 2, 12), (384, 8, 8, 8),
]


def make(C, H, W, seed_offset=0):
    g = torch.Generator(device="cuda").manual_seed(42 + seed_offset)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16, generator=g).to(memory_format=torch.channels_last)
    gamma = (0.5 + torch.rand(C, device="cuda", dtype=torch.float16, generator=g))
    beta = (0.1 * torch.randn(C, device="cuda", dtype=torch.float16, generator=g))
    mean = 0.05 * torch.randn(N * G, device="cuda", dtype=torch.float32, generator=g)
    inv_std = 0.8 + 0.4 * torch.rand(N * G, device="cuda", dtype=torch.float32, generator=g)
    return x, gamma, beta, mean, inv_std


def run_correctness(C, H, W):
    ne, ss = N * C * H * W, C * H * W
    x1, gamma1, beta1, mean1, inv_std1 = make(C, H, W, seed_offset=1)
    x2, gamma2, beta2, mean2, inv_std2 = make(C, H, W, seed_offset=2)
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    scale1 = torch.tensor([53.0], device="cuda", dtype=torch.float32)
    scale2 = torch.tensor([71.0], device="cuda", dtype=torch.float32)
    # computed via a float32 tensor op (not a Python float division then cast) so this matches
    # the CUDA kernel's own `1.0f/scale` bit-for-bit -- Python's `1.0/53.0` rounds in float64
    # first, which can differ from a native float32 division by 1 ULP.
    prev_inv_scale = 1.0 / scale1

    # --- standard: two real writes ---
    a_std = a0.clone()
    yq_std = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
    probe.probe_standard_launch(x1, a_std, yq_std, gamma1, beta1, mean1, inv_std1, scale1,
                                C, G, ss, ne, True, False)
    probe.probe_standard_launch(x2, a_std, yq_std, gamma2, beta2, mean2, inv_std2, scale2,
                                C, G, ss, ne, True, False)

    # --- skip2: skip then catch-up ---
    a_s2 = a0.clone()
    yq_s2 = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
    probe.probe_skip_launch(x1, a_s2, yq_s2, gamma1, beta1, mean1, inv_std1, scale1,
                            C, G, ss, ne, True, False)
    probe.probe_catchup_launch(x2, a_s2, yq_s2, gamma2, beta2, mean2, inv_std2, scale2,
                               prev_inv_scale, C, G, ss, ne, True, False)

    a_exact = torch.equal(a_std, a_s2)
    yq_exact = torch.equal(yq_std, yq_s2)
    a_max_diff = (a_std.float() - a_s2.float()).abs().max().item()
    return a_exact, yq_exact, a_max_diff


def time_arm(fn):
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


print("=== Correctness: is the K=2 scheme bit-identical to the standard 2-step kernel? ===")
all_exact = True
for C, H, W, _ in SHAPES:
    a_exact, yq_exact, a_max_diff = run_correctness(C, H, W)
    ok = a_exact and yq_exact
    all_exact = all_exact and ok
    print(f"  C={C:>4} {H}x{W:<3}  a_hat exact={a_exact}  Yq exact={yq_exact}  "
          f"max|diff|={a_max_diff:.3g}")
print(f"\nALL SHAPES BIT-IDENTICAL: {'YES' if all_exact else 'NO -- STOP, do not trust the benchmark below'}")

if not all_exact:
    sys.exit(1)

print("\n=== Benchmark: real ms/step for the K=2 scheme vs the standard baseline ===")
rows = []
for C, H, W, freq in SHAPES:
    ne, ss = N * C * H * W, C * H * W
    x1, gamma1, beta1, mean1, inv_std1 = make(C, H, W, seed_offset=1)
    x2, gamma2, beta2, mean2, inv_std2 = make(C, H, W, seed_offset=2)
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    scale1 = torch.tensor([53.0], device="cuda", dtype=torch.float32)
    scale2 = torch.tensor([71.0], device="cuda", dtype=torch.float32)
    prev_inv_scale = 1.0 / scale1
    a_buf = a0.clone()
    yq_buf = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)

    def reset():
        a_buf.copy_(a0)

    def baseline_pair():
        reset()
        probe.probe_standard_launch(x1, a_buf, yq_buf, gamma1, beta1, mean1, inv_std1, scale1,
                                    C, G, ss, ne, True, False)
        probe.probe_standard_launch(x2, a_buf, yq_buf, gamma2, beta2, mean2, inv_std2, scale2,
                                    C, G, ss, ne, True, False)

    def skip2_pair():
        reset()
        probe.probe_skip_launch(x1, a_buf, yq_buf, gamma1, beta1, mean1, inv_std1, scale1,
                                C, G, ss, ne, True, False)
        probe.probe_catchup_launch(x2, a_buf, yq_buf, gamma2, beta2, mean2, inv_std2, scale2,
                                   prev_inv_scale, C, G, ss, ne, True, False)

    fns = {"baseline_pair": baseline_pair, "skip2_pair": skip2_pair}
    samples = {n: [] for n in fns}
    for t in range(TRIALS):
        order = list(fns) if t % 2 == 0 else list(fns)[::-1]
        for n in order:
            samples[n].append(time_arm(fns[n]))
    med = {n: statistics.median(v) for n, v in samples.items()}
    sd = {n: statistics.stdev(v) for n, v in samples.items()}

    baseline_per_step = med["baseline_pair"] / 2
    skip2_per_step = med["skip2_pair"] / 2
    saved_per_step = baseline_per_step - skip2_per_step
    rows.append(dict(shape=f"{C},{H}x{W}", freq=freq, baseline=baseline_per_step,
                     skip2=skip2_per_step, saved=saved_per_step))
    print(f"  C={C:>4} {H}x{W:<3} freq{freq:>3} | baseline {baseline_per_step:.4f} ms/step | "
          f"skip2 {skip2_per_step:.4f} ms/step | saved {saved_per_step:+.4f} ms/step "
          f"({100*saved_per_step/baseline_per_step:+.1f}%) | "
          f"sd<{100*max(sd.values())/min(med.values()):.2f}%")
    del x1, x2, a0, a_buf, yq_buf
    torch.cuda.empty_cache()

tot_saved = sum(r["saved"] * r["freq"] for r in rows[:5])
tot_baseline = sum(r["baseline"] * r["freq"] for r in rows[:5])
print(f"\nfreq-weighted over the 5 dominant shapes: saved {tot_saved:.4f} ms/step of "
      f"{tot_baseline:.4f} ms/step baseline ({100*tot_saved/tot_baseline:.1f}%)")
print(f"(modelled estimate in FINDINGS.md was +0.564 ms/step, ~28% of the full write-elision ceiling)")
