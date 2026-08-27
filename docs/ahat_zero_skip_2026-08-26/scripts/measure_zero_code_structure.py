"""Follow-up to measure_zero_code_rate.py: that script showed delta codes are 53% zero overall but
only 1.7% of 64-element FLAT-NHWC warps are all-zero -- zeros looked "scattered." But flat-NHWC
warps mix channels together (channel is the fastest-varying axis in NHWC, so a 64-element warp
spans up to 64 consecutive channels at one spatial position, or wraps into the next position for
narrower layers) -- that grouping was never chosen to test channel- or spatial-locality, only the
kernel's actual memory access unit. This asks the structural question directly: is the zero-rate
UNEVEN across channels, or across spatial positions, enough that a coarser (per-channel or
per-spatial-tile) skip could find a cheaper unit than the warp did?

Hooks the same function as measure_zero_code_rate.py, but instead of scoring flat warps, keeps the
real (N, C, H, W) code tensor for ONE representative layer (matched by shape, same technique as
capture_real_ahat.py) across every call in an actual generation, and computes:
  (a) per-channel zero-rate (over N,H,W, aggregated across all calls) -- does it vary a lot?
  (b) per-channel ALL-zero-this-call rate -- fraction of calls where an entire channel (all N,H,W
      for that call) is exactly zero; this is the unit a per-channel skip would need.
  (c) per-spatial-position (h,w) zero-rate and ALL-channel-zero rate, the dual of (a)/(b).

Run: python docs/ahat_zero_skip_2026-08-26/scripts/measure_zero_code_structure.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")

import torch  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

_orig_flat = mc.group_norm_silu_delta_quantize_nhwc

first_layer_key = [None]
# per-channel accumulators (built once we know the layer's C)
chan_zero_count = [None]   # [C] running count of zero elements per channel (summed over N,H,W,calls)
chan_total_count = [0]
chan_allzero_calls = [None]  # [C] running count of calls where that channel was ALL zero
pos_zero_count = [None]    # [H*W] running count of zero elements per spatial position
pos_total_count = [0]
pos_allzero_calls = [None]
n_calls_this_layer = [0]


def _wrapped(x, weight, bias, a_hat_cache, num_groups, eps, apply_silu, scale, *rest, **kwargs):
    out = _orig_flat(x, weight, bias, a_hat_cache, num_groups, eps, apply_silu, scale, *rest, **kwargs)
    key = (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
    if first_layer_key[0] is None:
        first_layer_key[0] = key
    if key != first_layer_key[0]:
        return out

    codes = out if out.dim() == 4 else out  # (N, C, H, W)
    N, C, H, W = codes.shape
    zero = (codes == 0)

    if chan_zero_count[0] is None:
        chan_zero_count[0] = torch.zeros(C, dtype=torch.long, device=codes.device)
        chan_allzero_calls[0] = torch.zeros(C, dtype=torch.long, device=codes.device)
        pos_zero_count[0] = torch.zeros(H * W, dtype=torch.long, device=codes.device)
        pos_allzero_calls[0] = torch.zeros(H * W, dtype=torch.long, device=codes.device)

    # per-channel: sum zeros over N,H,W -> [C]
    chan_zero_this = zero.sum(dim=(0, 2, 3))
    chan_zero_count[0] += chan_zero_this
    # a channel is "all zero this call" if every one of its N*H*W elements is zero
    chan_all_this = (chan_zero_this == N * H * W)
    chan_allzero_calls[0] += chan_all_this.long()

    # per-spatial-position: sum zeros over N,C -> [H*W]
    pos_zero_this = zero.sum(dim=(0, 1)).reshape(-1)
    pos_zero_count[0] += pos_zero_this
    pos_all_this = (pos_zero_this == N * C)
    pos_allzero_calls[0] += pos_all_this.long()

    chan_total_count[0] = chan_total_count[0] + N * H * W  # per channel, elements contributed this call
    pos_total_count[0] = pos_total_count[0] + N * C        # per position, elements contributed this call
    n_calls_this_layer[0] += 1

    return out


mc.group_norm_silu_delta_quantize_nhwc = _wrapped
import integration.kernels.int8_optimized as i8opt  # noqa: E402
i8opt.modiff_cutlass.group_norm_silu_delta_quantize_nhwc = _wrapped

import integration.benchmarks.benchmark_ldm as bl  # noqa: E402

STEPS, BATCH = 20, 4
sys.argv = ["benchmark_ldm.py", "--mode", "int8", "--batch_size", str(BATCH),
           "--steps", str(STEPS), "--num_samples", str(BATCH), "--skip_calibration"]

print(f"Running {STEPS}-step int8 generation, batch {BATCH}, to capture real delta code structure...")
try:
    bl.main()
except SystemExit:
    pass

C = chan_zero_count[0].numel()
HW = pos_zero_count[0].numel()
n_calls = n_calls_this_layer[0]
print(f"\nrepresentative layer shape: C={first_layer_key[0][0]}, H={first_layer_key[0][1]}, "
      f"W={first_layer_key[0][2]}, {n_calls} calls captured")

chan_rate = (chan_zero_count[0].float() / chan_total_count[0]).cpu().numpy()
pos_rate = (pos_zero_count[0].float() / pos_total_count[0]).cpu().numpy()
chan_allzero_rate = (chan_allzero_calls[0].float() / n_calls).cpu().numpy()
pos_allzero_rate = (pos_allzero_calls[0].float() / n_calls).cpu().numpy()

import numpy as np
print(f"\n--- per-channel zero-rate over {C} channels (aggregated over N,H,W,calls) ---")
print(f"  mean={chan_rate.mean():.4f}  std={chan_rate.std():.4f}  min={chan_rate.min():.4f}  "
      f"max={chan_rate.max():.4f}")
print(f"  per-channel ALL-ZERO-this-call rate (fraction of the {n_calls} calls where the WHOLE "
      f"channel was zero):")
print(f"  mean={chan_allzero_rate.mean():.4f}  std={chan_allzero_rate.std():.4f}  "
      f"min={chan_allzero_rate.min():.4f}  max={chan_allzero_rate.max():.4f}")
print(f"  channels with >50% all-zero-call rate: {int((chan_allzero_rate > 0.5).sum())}/{C}")
print(f"  channels with >90% all-zero-call rate: {int((chan_allzero_rate > 0.9).sum())}/{C}")

print(f"\n--- per-spatial-position zero-rate over {HW} positions (aggregated over N,C,calls) ---")
print(f"  mean={pos_rate.mean():.4f}  std={pos_rate.std():.4f}  min={pos_rate.min():.4f}  "
      f"max={pos_rate.max():.4f}")
print(f"  per-position ALL-ZERO-this-call rate (fraction of the {n_calls} calls where the WHOLE "
      f"spatial position was zero across all channels):")
print(f"  mean={pos_allzero_rate.mean():.4f}  std={pos_allzero_rate.std():.4f}  "
      f"min={pos_allzero_rate.min():.4f}  max={pos_allzero_rate.max():.4f}")

np.savez("/workspace/MoDiff/docs/ahat_zero_skip_2026-08-26/data/zero_code_structure.npz",
         chan_rate=chan_rate, pos_rate=pos_rate,
         chan_allzero_rate=chan_allzero_rate, pos_allzero_rate=pos_allzero_rate,
         layer_shape=np.array(first_layer_key[0]), n_calls=n_calls)
print("\nsaved data/zero_code_structure.npz")
