"""Capture REAL a_hat values and per-step delta magnitudes from an actual generation, to replace
the synthetic AR(1) proxy trajectory simulate_drift.py used, and to size two untried ideas:

  (1) temporal gating -- use int8 a_hat only for the steps where the calibrated delta_scale table
      says a coarse grid is still adequate, fp16 for the rest.
  (2) non-uniform (companding) quantization -- worth it only if a_hat's real value distribution is
      skewed enough that a non-uniform 8-bit grid meaningfully beats a uniform one.

Hooks group_norm_silu_delta_quantize_nhwc at the Python attribute level (same technique as
measure_zero_code_rate.py) and snapshots a_hat_cache's post-update values every call, along with
which call-order (a proxy for step index, as in that script) it came from.

Run: python docs/int8_ahat_cache_2026-08-26/scripts/capture_real_ahat.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

_orig_flat = mc.group_norm_silu_delta_quantize_nhwc

# Cap how many (layer-call, step) combinations we keep full histograms for -- otherwise this
# would hold onto gigabytes. Track the FIRST layer call seen (called every step, so its call index
# increases by exactly the per-step call count each step) to get a clean per-step trajectory for
# one representative layer, and a global histogram (subsampled) across all layers for the
# companding question.
first_layer_key = [None]
per_step_stats = []   # (call_index, mean, std, p01, p50, p99, max_abs)
global_samples = []   # subsampled raw a_hat values, all layers, for the histogram question
delta_over_quantum = []  # |out - a_hat_prev| * scale, i.e. the actual q magnitude each call


def _wrapped(x, weight, bias, a_hat_cache, num_groups, eps, apply_silu, scale, *rest, **kwargs):
    a_hat_before = a_hat_cache.clone()
    out = _orig_flat(x, weight, bias, a_hat_cache, num_groups, eps, apply_silu, scale, *rest, **kwargs)
    torch.cuda.synchronize()

    key = (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
    if first_layer_key[0] is None:
        first_layer_key[0] = key
    if key == first_layer_key[0]:
        a = a_hat_cache.float()
        flat = a.reshape(-1)
        qs = flat.abs().quantile(torch.tensor([0.01, 0.50, 0.99], device=flat.device))
        per_step_stats.append((len(per_step_stats), float(flat.mean()), float(flat.std()),
                               float(qs[0]), float(qs[1]), float(qs[2]), float(flat.abs().max())))
        # the actual per-element code magnitude this call produced, in delta units (q = code)
        code = out.permute(0, 2, 3, 1).contiguous().reshape(-1).float()
        delta_over_quantum.append((len(delta_over_quantum), float(code.abs().float().mean()),
                                   float(code.abs().float().max())))

    if len(global_samples) < 2_000_000:
        idx = torch.randint(0, a_hat_cache.numel(), (min(2000, a_hat_cache.numel()),), device=a_hat_cache.device)
        global_samples.append(a_hat_cache.reshape(-1)[idx].float().cpu().numpy())

    return out


mc.group_norm_silu_delta_quantize_nhwc = _wrapped
import integration.kernels.int8_optimized as i8opt  # noqa: E402
i8opt.modiff_cutlass.group_norm_silu_delta_quantize_nhwc = _wrapped

import integration.benchmarks.benchmark_ldm as bl  # noqa: E402

STEPS, BATCH = 20, 4
sys.argv = ["benchmark_ldm.py", "--mode", "int8", "--batch_size", str(BATCH),
           "--steps", str(STEPS), "--num_samples", str(BATCH), "--skip_calibration"]

print(f"Running {STEPS}-step int8 generation, batch {BATCH}, to capture real a_hat values...")
try:
    bl.main()
except SystemExit:
    pass

print(f"\nrepresentative layer shape: {first_layer_key[0]}, {len(per_step_stats)} calls captured")
print(f"{'call':>5} {'mean':>9} {'std':>9} {'p01':>9} {'p50':>9} {'p99':>9} {'max|a_hat|':>10}  "
      f"{'mean|code|':>10} {'max|code|':>9}")
for (i, m, sd, p01, p50, p99, mx), (_, cmean, cmax) in zip(per_step_stats, delta_over_quantum):
    print(f"{i:>5} {m:>9.4f} {sd:>9.4f} {p01:>9.4f} {p50:>9.4f} {p99:>9.4f} {mx:>10.4f}  "
          f"{cmean:>10.4f} {cmax:>9.1f}")

allv = np.concatenate(global_samples) if global_samples else np.array([])
print(f"\nglobal a_hat value distribution, {len(allv)} samples across all layers:")
print(f"  mean={allv.mean():.4f}  std={allv.std():.4f}  "
      f"p01={np.percentile(allv,1):.4f}  p50={np.percentile(allv,50):.4f}  "
      f"p99={np.percentile(allv,99):.4f}  min={allv.min():.4f}  max={allv.max():.4f}")
print(f"  fraction within +-1 std of mean: {np.mean(np.abs(allv-allv.mean())<allv.std()):.3f} "
      f"(0.683 would be exactly Gaussian)")
print(f"  fraction within +-0.25*max(|min|,|max|) of ZERO: "
      f"{np.mean(np.abs(allv) < 0.25*max(abs(allv.min()),abs(allv.max()))):.3f}")

np.savez("/workspace/MoDiff/docs/int8_ahat_cache_2026-08-26/data/real_ahat_capture.npz",
        per_step=np.array(per_step_stats), delta=np.array(delta_over_quantum),
        global_samples=allv)
print("\nsaved data/real_ahat_capture.npz")
