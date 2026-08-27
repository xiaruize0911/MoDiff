"""Generalizes the K=2 deferred-write scheme to arbitrary K, verifies exactness against the real
production kernel for several K values, then sweeps K to find the empirically optimal window size
and plots ms/step vs K.

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/sweep_k.py
"""
import os
import statistics
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")

import torch  # noqa: E402
import ahat_skip2_probe as probe  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

torch.manual_seed(0)
N, G = 128, 32
TRIALS, REPS, WARMUP = 5, 30, 10
K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
K_MAX = max(K_VALUES)

SHAPES = [
    (192, 32, 32, 7), (384, 16, 16, 7), (384, 32, 32, 2),
    (576, 32, 32, 1), (768, 16, 16, 2), (768, 2, 2, 12), (384, 8, 8, 8),
]

empty16 = None
empty32 = None
empty_i32 = None


def make(C, H, W, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16, generator=g).to(memory_format=torch.channels_last)
    gamma = (0.5 + torch.rand(C, device="cuda", dtype=torch.float16, generator=g))
    beta = (0.1 * torch.randn(C, device="cuda", dtype=torch.float16, generator=g))
    return x, gamma, beta


def run_window_correctness(C, H, W, K, seed0=100):
    """Run K real steps via production, and K steps via probe_window_step; compare a_hat (at the
    final step, which writes) and every code."""
    global empty16, empty32, empty_i32
    ne, ss = N * C * H * W, C * H * W
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)

    xs, gammas, betas, scales = [], [], [], []
    for t in range(K):
        x, gamma, beta = make(C, H, W, seed0 + t)
        xs.append(x); gammas.append(gamma); betas.append(beta)
        scales.append(torch.tensor([50.0 + 7.0 * t], device="cuda", dtype=torch.float32))

    # --- production, K real steps ---
    a_prod = a0.clone()
    codes_prod = []
    for t in range(K):
        c = mc.group_norm_silu_delta_quantize_nhwc(
            xs[t], gammas[t], betas[t], a_prod, G, 1e-5, True, scales[t],
            empty32, empty16, empty16, empty32, empty32, empty32, empty_i32, 127.0, False, 1.0)
        codes_prod.append(c)

    # --- windowed scheme, K steps in one window ---
    a_win = a0.clone()
    pending_codes = torch.zeros(max(K - 1, 1), N, C, H, W, device="cuda", dtype=torch.int8)
    pending_inv_scales = torch.zeros(max(K - 1, 1), device="cuda", dtype=torch.float32)
    codes_win = []
    for t in range(K):
        mean = torch.empty(N * G, device="cuda", dtype=torch.float32)
        inv_std = torch.empty(N * G, device="cuda", dtype=torch.float32)
        probe.stats_launch(xs[t], mean, inv_std, C, G, H * W, 1e-5)
        yq = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
        is_last = (t == K - 1)
        pc_t = pending_codes[t] if t < K - 1 else pending_codes[0]  # unused slot when is_last
        probe.probe_window_step_launch(xs[t], a_win, yq, gammas[t], betas[t], empty16, empty16, empty16,
                                       mean, inv_std, scales[t], pending_codes.view(max(K - 1, 1), -1),
                                       pending_inv_scales, ne, t, is_last, C, G, ss, ne, True, False)
        codes_win.append(yq)
        if not is_last:
            pending_inv_scales[t] = 1.0 / scales[t]

    a_exact = torch.equal(a_prod, a_win)
    codes_exact = all(torch.equal(codes_prod[t], codes_win[t]) for t in range(K))
    return a_exact, codes_exact


print("=== Correctness: windowed scheme vs production, several K values ===")
empty16 = torch.empty(0, device="cuda", dtype=torch.float16)
empty32 = torch.empty(0, device="cuda", dtype=torch.float32)
empty_i32 = torch.empty(0, device="cuda", dtype=torch.int32)
all_ok = True
for K in [2, 3, 5, 8]:
    for C, H, W, _ in SHAPES[:3]:
        a_ok, c_ok = run_window_correctness(C, H, W, K)
        ok = a_ok and c_ok
        all_ok = all_ok and ok
        print(f"  K={K:>2} C={C:>4} {H}x{W:<3}  a_hat exact={a_ok}  all codes exact={c_ok}")

print(f"\nALL K VALUES BIT-IDENTICAL TO PRODUCTION: {'YES' if all_ok else 'NO -- STOP'}")
if not all_ok:
    sys.exit(1)


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


print("\n=== Timing sweep: ms/step vs K, 5 dominant shapes, freq-weighted ===")
results = {K: [] for K in K_VALUES}  # per-shape (freq, ms_per_step)
for C, H, W, freq in SHAPES:
    ne, ss = N * C * H * W, C * H * W
    xs = [make(C, H, W, 200 + t)[0] for t in range(K_MAX)]
    gamma, beta = make(C, H, W, 999)[1:]
    scales = [torch.tensor([50.0 + 7.0 * t], device="cuda", dtype=torch.float32) for t in range(K_MAX)]
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    a_buf = a0.clone()
    yq_buf = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
    mean_buf = torch.empty(N * G, device="cuda", dtype=torch.float32)
    inv_std_buf = torch.empty(N * G, device="cuda", dtype=torch.float32)
    pending_codes = torch.zeros(max(K_MAX - 1, 1), N, C, H, W, device="cuda", dtype=torch.int8).view(max(K_MAX - 1, 1), -1)
    pending_inv_scales = torch.zeros(max(K_MAX - 1, 1), device="cuda", dtype=torch.float32)

    def reset():
        a_buf.copy_(a0)

    def make_fn(K):
        def fn():
            reset()
            for t in range(K):
                probe.stats_launch(xs[t], mean_buf, inv_std_buf, C, G, H * W, 1e-5)
                is_last = (t == K - 1)
                probe.probe_window_step_launch(xs[t], a_buf, yq_buf, gamma, beta, empty16, empty16, empty16,
                                               mean_buf, inv_std_buf, scales[t], pending_codes,
                                               pending_inv_scales, ne, t, is_last, C, G, ss, ne, True, False)
                if not is_last:
                    pending_inv_scales[t] = 1.0 / scales[t]
        return fn

    fns = {K: make_fn(K) for K in K_VALUES}
    samples = {K: [] for K in K_VALUES}
    for trial in range(TRIALS):
        order = K_VALUES if trial % 2 == 0 else K_VALUES[::-1]
        for K in order:
            samples[K].append(time_arm(fns[K]) / K)  # ms PER STEP, not per window
    med = {K: statistics.median(v) for K, v in samples.items()}
    for K in K_VALUES:
        results[K].append((freq, med[K]))
    print(f"  C={C:>4} {H}x{W:<3} freq{freq:>3} | " +
          " | ".join(f"K={K}:{med[K]:.4f}" for K in K_VALUES))
    del xs, a0, a_buf, yq_buf, pending_codes
    torch.cuda.empty_cache()

print(f"\n{'K':>4} {'freq-weighted ms/step (5 dominant shapes)':>45} {'vs K=1':>10}")
freq_weighted = {}
baseline_fw = None
for K in K_VALUES:
    tot = sum(freq * ms for freq, ms in results[K][:5])
    tot_freq = sum(freq for freq, ms in results[K][:5])
    fw = tot / tot_freq
    freq_weighted[K] = fw
    if K == 1:
        baseline_fw = fw
    delta_pct = 100 * (baseline_fw - fw) / baseline_fw if baseline_fw else 0.0
    print(f"{K:>4} {fw:>45.4f} {delta_pct:>+9.2f}%")

best_K = min(freq_weighted, key=freq_weighted.get)
print(f"\nEmpirically optimal K: {best_K} ({freq_weighted[best_K]:.4f} ms/step, "
      f"{100*(baseline_fw-freq_weighted[best_K])/baseline_fw:+.2f}% vs K=1)")

import json
json.dump({"K_VALUES": K_VALUES, "freq_weighted": freq_weighted, "per_shape": results},
          open(os.path.dirname(os.path.abspath(__file__)) + "/../data/k_sweep.json", "w"),
          indent=1, default=str)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
ks = K_VALUES
ms = [freq_weighted[k] for k in ks]
ax.plot(ks, ms, marker="o", color="steelblue", label="measured, freq-weighted")
ax.axhline(baseline_fw, color="gray", linestyle="--", label=f"K=1 baseline ({baseline_fw:.4f} ms/step)")
ax.scatter([best_K], [freq_weighted[best_K]], color="crimson", zorder=5, s=80,
          label=f"optimal K={best_K}")
ax.set_xlabel("K (window size)")
ax.set_ylabel("ms/step (freq-weighted, 5 dominant shapes)")
ax.set_title("Deferred-write a_hat: real measured cost vs window size K")
ax.legend()
ax.set_xticks(ks)
plt.tight_layout()
out_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/k_sweep.png"
plt.savefig(out_path, dpi=130)
print(f"\nsaved plot: {out_path}")
