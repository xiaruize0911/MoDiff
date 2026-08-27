"""Kernel-level benchmark of the deferred-write scheme AT THE REAL REFRESH CADENCE.

Why this exists: sweep_k.py measured K windows of the probe kernel against production called with
ALL-EMPTY reduction buffers, i.e. production's NON-refresh branch only. Production actually takes
the REFRESH branch every R'th step (R = MODIFF_DELTA_REFRESH, default 4), where
_delta_gn_dynamic_args hands it real absmax/scale/inv_scale/retire buffers and it runs an extra
reduction pass. The 2026-08-27 fix therefore delegates refresh steps to production and windows only
the steps between them -- so the honest question is not "K windows vs K production calls" but
"one R-step cadence, patched vs unpatched". That is what this measures.

Both arms run the identical R-step cadence on the identical tensors:

  unpatched : R production calls -- step 0 with REFRESH args, steps 1..R-1 with non-refresh args
  patched   : step 0 production/REFRESH (delegated), steps 1..R-1 through stats + probe_window_step
              with position = (phase-1) % K_eff, is_last at position K_eff-1 or phase == R-1

Correctness is gated BEFORE timing: a_hat after the full cadence, and every emitted code, must be
bit-identical between the arms, or the script exits non-zero.

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/cadence_kernel_bench.py
Env: BENCH_R (default 4), BENCH_KS (default 2,3), BENCH_TRIALS (default 7), BENCH_REPS (default 30)
"""
import json
import os
import statistics
import sys

os.chdir("/workspace/MoDiff")
sys.path[:0] = ["/workspace/MoDiff", "/workspace/MoDiff/src/taming-transformers",
                os.path.dirname(os.path.abspath(__file__)) + "/build"]

import torch  # noqa: E402
import ahat_skip2_probe as probe  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

torch.manual_seed(0)
N, G, EPS = 128, 32, 1e-5
R = int(os.environ.get("BENCH_R", "4"))
KS = [int(k) for k in os.environ.get("BENCH_KS", "2,3").split(",")]
TRIALS = int(os.environ.get("BENCH_TRIALS", "7"))
REPS = int(os.environ.get("BENCH_REPS", "30"))
WARMUP = 10

# (C, H, W, calls/step) -- the shapes and real call frequencies sweep_k.py used, from the
# 2026-08-13 capture. The first five dominate total step time; 768x2x2 and 384x8x8 are kept
# visible because they are the launch-bound counter-examples.
SHAPES = [(192, 32, 32, 7), (384, 16, 16, 7), (384, 32, 32, 2),
          (576, 32, 32, 1), (768, 16, 16, 2), (768, 2, 2, 12), (384, 8, 8, 8)]
DOMINANT = 5

e16 = torch.empty(0, device="cuda", dtype=torch.float16)
e32 = torch.empty(0, device="cuda", dtype=torch.float32)
ei32 = torch.empty(0, device="cuda", dtype=torch.int32)


def make(C, H, W, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16,
                    generator=g).to(memory_format=torch.channels_last)
    gamma = 0.5 + torch.rand(C, device="cuda", dtype=torch.float16, generator=g)
    beta = 0.1 * torch.randn(C, device="cuda", dtype=torch.float16, generator=g)
    return x, gamma, beta


class RefreshBufs:
    """The four buffers _delta_gn_dynamic_args hands the kernel on a refresh step."""

    def __init__(self):
        self.absmax = torch.zeros(1, device="cuda", dtype=torch.float32)
        self.scale = torch.empty(1, device="cuda", dtype=torch.float32)
        self.inv = torch.empty(1, device="cuda", dtype=torch.float32)
        self.retire = torch.zeros(1, device="cuda", dtype=torch.int32)

    def refresh_args(self, act_q=127.0):
        return (self.absmax, self.scale, self.inv, self.retire, act_q, False, 1.0)

    @staticmethod
    def plain_args():
        return (e32, e32, e32, ei32, 127.0, False, 1.0)


def prod_call(x, gamma, beta, a_hat, scale, is_refresh, bufs):
    tail = bufs.refresh_args() if is_refresh else RefreshBufs.plain_args()
    return mc.group_norm_silu_delta_quantize_nhwc(
        x, gamma, beta, a_hat, G, EPS, True, scale, e32, e16, e16, *tail)


def window_step(x, a_hat, yq, gamma, beta, mean, inv_std, scale, pend, pinv,
                ne, ss, position, is_last):
    probe.stats_launch(x, mean, inv_std, x.size(1), G, x.size(2) * x.size(3), EPS)
    probe.probe_window_step_launch(x, a_hat, yq, gamma, beta, e16, e16, e16,
                                   mean, inv_std, scale, pend, pinv,
                                   ne, position, is_last, x.size(1), G, ss, ne, True, False)


def cadence_plan(K):
    """(phase -> ('prod'|('win', position, is_last))) for one R-step cadence under the fix."""
    K_eff = max(1, min(K, R - 1)) if R > 1 else K
    plan = []
    for phase in range(R):
        if R > 1 and phase == 0:
            plan.append(("prod", None, None))
            continue
        pos = (phase - 1) % K_eff
        last = (pos == K_eff - 1) or (R > 1 and phase == R - 1)
        plan.append(("win", pos, last))
    return plan, K_eff


def check(C, H, W, K, seed0=100):
    """Bit-exactness of the full cadence, both arms, same inputs."""
    ne, ss = N * C * H * W, C * H * W
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda",
                            dtype=torch.float16)).to(memory_format=torch.channels_last)
    xs, gs, bs, scs = [], [], [], []
    for t in range(R):
        x, gamma, beta = make(C, H, W, seed0 + t)
        xs.append(x); gs.append(gamma); bs.append(beta)
        scs.append(torch.tensor([50.0 + 7.0 * t], device="cuda", dtype=torch.float32))

    a_ref = a0.clone()
    bufs = RefreshBufs()
    codes_ref = [prod_call(xs[t], gs[t], bs[t], a_ref, scs[t], t == 0, bufs).clone()
                 for t in range(R)]

    plan, K_eff = cadence_plan(K)
    a_got = a0.clone()
    bufs2 = RefreshBufs()
    mean = torch.empty(N * G, device="cuda", dtype=torch.float32)
    inv_std = torch.empty(N * G, device="cuda", dtype=torch.float32)
    pend = torch.zeros(max(K_eff - 1, 1), ne, device="cuda", dtype=torch.int8)
    pinv = torch.zeros(max(K_eff - 1, 1), device="cuda", dtype=torch.float32)
    yq = torch.empty(N, C, H, W, device="cuda",
                     dtype=torch.int8).to(memory_format=torch.channels_last)
    codes_got = []
    for t, (kind, pos, last) in enumerate(plan):
        if kind == "prod":
            codes_got.append(prod_call(xs[t], gs[t], bs[t], a_got, scs[t], t == 0, bufs2).clone())
        else:
            window_step(xs[t], a_got, yq, gs[t], bs[t], mean, inv_std, scs[t],
                        pend, pinv, ne, ss, pos, last)
            codes_got.append(yq.clone())
            if not last:
                pinv[pos].copy_(scs[t].view(-1)[0].reciprocal())
    a_ok = torch.equal(a_ref, a_got)
    c_ok = all(torch.equal(codes_ref[t], codes_got[t]) for t in range(R))
    return a_ok, c_ok, K_eff


def time_fn(fn):
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


print(f"=== Correctness gate: full R={R} cadence, patched vs unpatched, per shape ===")
allok = True
for K in KS:
    for C, H, W, _ in SHAPES[:3]:
        a_ok, c_ok, K_eff = check(C, H, W, K)
        allok = allok and a_ok and c_ok
        print(f"  K={K} (K_eff={K_eff}) C={C:>4} {H}x{W:<3}: a_hat={a_ok} codes={c_ok}")
print(f"CADENCE BIT-EXACT: {'YES' if allok else 'NO -- STOP'}")
if not allok:
    sys.exit(1)

print(f"\n=== Timing: one R={R} cadence, ms per STEP (cadence/R), {TRIALS} trials x {REPS} reps ===")
results = {}
for C, H, W, freq in SHAPES:
    ne, ss = N * C * H * W, C * H * W
    xs, gs, bs, scs = [], [], [], []
    for t in range(R):
        x, gamma, beta = make(C, H, W, 200 + t)
        xs.append(x); gs.append(gamma); bs.append(beta)
        scs.append(torch.tensor([50.0 + 7.0 * t], device="cuda", dtype=torch.float32))
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda",
                            dtype=torch.float16)).to(memory_format=torch.channels_last)
    a_buf = a0.clone()
    mean = torch.empty(N * G, device="cuda", dtype=torch.float32)
    inv_std = torch.empty(N * G, device="cuda", dtype=torch.float32)
    yq = torch.empty(N, C, H, W, device="cuda",
                     dtype=torch.int8).to(memory_format=torch.channels_last)
    bufs = RefreshBufs()

    def unpatched():
        a_buf.copy_(a0)
        for t in range(R):
            prod_call(xs[t], gs[t], bs[t], a_buf, scs[t], t == 0, bufs)

    arms = {"unpatched": unpatched}
    keep = []
    for K in KS:
        plan, K_eff = cadence_plan(K)
        pend = torch.zeros(max(K_eff - 1, 1), ne, device="cuda", dtype=torch.int8)
        pinv = torch.zeros(max(K_eff - 1, 1), device="cuda", dtype=torch.float32)
        keep.append((pend, pinv))

        def make_arm(plan=plan, pend=pend, pinv=pinv):
            def fn():
                a_buf.copy_(a0)
                for t, (kind, pos, last) in enumerate(plan):
                    if kind == "prod":
                        prod_call(xs[t], gs[t], bs[t], a_buf, scs[t], t == 0, bufs)
                    else:
                        window_step(xs[t], a_buf, yq, gs[t], bs[t], mean, inv_std, scs[t],
                                    pend, pinv, ne, ss, pos, last)
                        if not last:
                            pinv[pos].copy_(scs[t].view(-1)[0].reciprocal())
            return fn
        arms[f"K={K}"] = make_arm()

    names = list(arms)
    samples = {n: [] for n in names}
    for tr in range(TRIALS):
        order = names if tr % 2 == 0 else names[::-1]
        for n in order:
            samples[n].append(time_fn(arms[n]) / R)
    med = {n: statistics.median(samples[n]) for n in names}
    for n in names:
        results.setdefault(n, []).append((freq, med[n]))
    base = med["unpatched"]
    txt = " | ".join(f"{n}:{med[n]:.4f}({100*(base-med[n])/base:+.1f}%)" for n in names if n != "unpatched")
    print(f"  C={C:>4} {H}x{W:<3} freq{freq:>3} | unpatched:{base:.4f} | {txt}")
    del xs, a0, a_buf, yq, keep
    torch.cuda.empty_cache()

print(f"\n{'arm':<12}{'freq-weighted ms/step':>24}{'vs unpatched':>14}")
fw = {}
for n, rows in results.items():
    tot = sum(f * m for f, m in rows[:DOMINANT])
    tf = sum(f for f, _ in rows[:DOMINANT])
    fw[n] = tot / tf
base = fw["unpatched"]
for n in results:
    print(f"{n:<12}{fw[n]:>24.4f}{100*(base-fw[n])/base:>+13.2f}%")

out = os.path.dirname(os.path.abspath(__file__)) + "/../data/cadence_kernel_bench.json"
json.dump({"R": R, "KS": KS, "N": N, "trials": TRIALS, "reps": REPS,
           "freq_weighted": fw, "per_shape": {k: v for k, v in results.items()}},
          open(out, "w"), indent=1, default=str)
print(f"\nwrote {out}")
print("NOTE: ms/step here is per BATCH step at batch 128, the same unit as this document's other"
      f" kernel numbers -- divide by {N} to compare against benchmark_ldm's per-sample figures.")
