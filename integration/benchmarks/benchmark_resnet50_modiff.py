#!/usr/bin/env python3
"""MoDiff-mode efficiency on ResNet-50.

MoDiff (temporal error-compensated modulation, `o_hat`/`a_hat` delta caching) is designed
for a CORRELATED SEQUENCE of inputs — it quantizes the small frame-to-frame residual
`a_t - a_hat_{t+1}` and accumulates `o_hat` in place across steps. ResNet has no natural
sequence, so we drive a DENOISING-STYLE trajectory: a fixed image + a fixed noise pattern
scaled by a decreasing schedule (`x_t = base + sigma_t * eps`, sigma_t down), so consecutive
activations converge (a_t -> a_{t+1}).

MoDiff runs ONLY via the plain per-conv `forward()` (bare torchvision model, NOT the fullchain
wrappers, which require modiff off). It is expected to be SLOWER than the fullchain baseline
(the sub+accumulate skips no convs) and to use ~2x activation memory (a_hat + o_hat caches) —
MoDiff buys accuracy on sequences, not speed. This benchmark measures that efficiency tradeoff
(speed + memory); it does NOT tabulate accuracy.

  python integration/benchmarks/benchmark_resnet50_modiff.py --batch 64 --steps 24 --repeats 6
  python integration/benchmarks/benchmark_resnet50_modiff.py --validate
"""
import os, sys, argparse, statistics, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

spec = importlib.util.spec_from_file_location("rn", os.path.join(HERE, "benchmark_resnet50.py"))
rn = importlib.util.module_from_spec(spec); spec.loader.exec_module(rn)
from integration.fused_ops.chained_bottleneck import build_fully_chained, build_fully_chained_int4
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d

QCONV = (OptimizedInt8Conv2d, OptimizedInt4Conv2d)


def make_sequence(N, T, sigma_max=0.6, sigma_min=0.02, seed=0):
    """Denoising-style correlated sequence: fixed base image + fixed noise pattern eps,
    amplitude sigma_t decreasing linearly. Consecutive frames differ by (sigma_t -
    sigma_{t+1})*eps -> small, so a_t -> a_{t+1} (what MoDiff assumes)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(N, 3, 224, 224, device="cuda", generator=g)
    eps = torch.randn(N, 3, 224, 224, device="cuda", generator=g)
    sigmas = torch.linspace(sigma_max, sigma_min, T)
    return [(base + s.item() * eps).contiguous(memory_format=torch.channels_last).half() for s in sigmas]


def build_mode(mode, x_calib):
    if mode == "fp16":
        return rn.build_fp16()
    kind = "int8" if "int8" in mode else "int4"
    if mode.endswith("_modiff"):
        m, _ = rn.build_quantized(kind, x_calib, skip_pointwise=False, modiff=True)
        return m                                  # bare model, stock forward drives MoDiff
    # fullchain reference
    m, _ = rn.build_quantized(kind, x_calib, skip_pointwise=False)
    return build_fully_chained(m) if kind == "int8" else build_fully_chained_int4(m)


def reset_modiff(model):
    for mod in model.modules():
        if isinstance(mod, QCONV) and getattr(mod, "modiff_enabled", False):
            mod.reset_state()


def cache_mib(model):
    b = 0
    for mod in model.modules():
        if isinstance(mod, QCONV):
            for c in (getattr(mod, "a_hat_cache", None), getattr(mod, "o_hat_cache", None)):
                if c is not None:
                    b += c.numel() * c.element_size()
    return b / (1024 ** 2)


def _evt():
    return torch.cuda.Event(enable_timing=True)


@torch.no_grad()
def time_stateless(model, seq, repeats):
    """Per-frame latency for a stateless model (each frame independent)."""
    with torch.autocast("cuda", dtype=torch.float16):
        for f in seq[:3]:
            model(f)
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeats):
            for f in seq:
                s, e = _evt(), _evt(); s.record(); model(f); e.record()
                torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts), statistics.pstdev(ts)


@torch.no_grad()
def time_modiff(model, seq, repeats):
    """Drive the correlated sequence retaining cache; split first-step vs modulated ms."""
    with torch.autocast("cuda", dtype=torch.float16):
        reset_modiff(model)                       # warm-up sequence
        for f in seq:
            model(f)
        torch.cuda.synchronize()
        first, mod = [], []
        for _ in range(repeats):
            reset_modiff(model)
            for i, f in enumerate(seq):
                s, e = _evt(), _evt(); s.record(); model(f); e.record()
                torch.cuda.synchronize()
                (first if i == 0 else mod).append(s.elapsed_time(e))
    return (statistics.median(mod), statistics.pstdev(mod), statistics.median(first))


def run(batch, T, repeats, modes):
    seq = make_sequence(batch, T)
    x_calib = seq[len(seq) // 2]
    print(f"ResNet-50 MoDiff efficiency | A40 batch={batch} T={T} steps repeats={repeats}\n")
    results = {}
    for mode in modes:
        model = build_mode(mode, x_calib)
        if mode.endswith("_modiff"):
            med, sd, first = time_modiff(model, seq, repeats)
            mib = cache_mib(model)
            results[mode] = dict(ms=med, sd=sd, first=first, mib=mib)
        else:
            med, sd = time_stateless(model, seq, repeats)
            results[mode] = dict(ms=med, sd=sd, first=float("nan"), mib=0.0)
        del model; torch.cuda.empty_cache()
    fp = results["fp16"]["ms"]
    print(f"{'mode':<16}{'ms/frame':>10}{'stdev':>8}{'vs fp16':>9}{'cache MiB':>11}{'first-step ms':>15}")
    for m in modes:
        r = results[m]
        fs = f"{r['first']:.2f}" if r['first'] == r['first'] else "-"
        print(f"{m:<16}{r['ms']:>10.3f}{r['sd']:>8.3f}{fp/r['ms']:>8.2f}x{r['mib']:>11.1f}{fs:>15}")
    return results


@torch.no_grad()
def validate(batch=16, T=12):
    """Confirm MoDiff is FUNCTIONING (state machine + non-divergence + memory), not silently
    corrupting. Numerical accuracy vs fp16 is reported but int4 is a known-broken path."""
    print("=== MoDiff validation ===")
    seq = make_sequence(batch, T)
    x_calib = seq[T // 2]
    fp16 = rn.build_fp16()
    with torch.autocast("cuda", dtype=torch.float16):
        fp_out = [fp16(f).float() for f in seq]
    for mode in ("int8_modiff", "int4_modiff"):
        m = build_mode(mode, x_calib)
        reset_modiff(m)
        with torch.autocast("cuda", dtype=torch.float16):
            outs = [m(f).float() for f in seq]
        # dispatch counts: first conv should have seen 1 first-step + (T-1) modulated
        conv0 = next(mod for mod in m.modules() if isinstance(mod, QCONV) and getattr(mod, "modiff_enabled", False))
        step_ct = getattr(conv0, "step_count", None)
        rel = [(outs[t] - fp_out[t]).norm().item() / (fp_out[t].norm().item() + 1e-9) for t in range(T)]
        # reset isolation: rerun once more after reset, compare to first run
        reset_modiff(m)
        with torch.autocast("cuda", dtype=torch.float16):
            outs2 = [m(f).float() for f in seq]
        iso = (outs2[-1] - outs[-1]).norm().item() / (outs[-1].norm().item() + 1e-9)
        mib = cache_mib(m)
        finite = all(torch.isfinite(o).all() for o in outs)
        print(f"\n{mode}: finite={finite}  step_count(conv0)={step_ct}  cache={mib:.1f} MiB  reset_iso_rel={iso:.4f}")
        print(f"  per-frame rel-vs-fp16 (t=0..{T-1}): " + " ".join(f"{r:.2f}" for r in rel))
        print(f"  final-frame rel={rel[-1]:.3f}  {'(OK)' if rel[-1] < 0.5 else '(BROKEN numerics — known int4 issue)' if 'int4' in mode else '(HIGH)'}")
        del m; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=24, help="sequence length T")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--modes", nargs="+",
                    default=["fp16", "int8_fullchain", "int4_fullchain", "int8_modiff", "int4_modiff"])
    a = ap.parse_args()
    if a.validate:
        validate()
        return
    run(a.batch, a.steps, a.repeats, a.modes)


if __name__ == "__main__":
    main()
