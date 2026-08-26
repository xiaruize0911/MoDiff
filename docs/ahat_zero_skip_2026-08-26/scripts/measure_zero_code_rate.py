"""Real measurement: what fraction of a_hat's delta codes are exactly zero, and are the zeros
clustered enough (per warp) that a predicated skip of the a_hat WRITE would actually save DRAM
transactions?

Motivation: a_hat's write is `a_hat += q/scale`; when q == 0 that add is a no-op, so skipping the
write is EXACT, not an approximation. But GPUs move memory in fixed-size sectors, not per element
-- skipping only pays off if entire WARPS of adjacent elements are all-zero at once, not just a lot
of scattered individual zeros. `gn_apply_delta_quantize_flat_vec2_kernel` gives one warp (32
threads) 64 CONSECUTIVE flat-NHWC elements (2 per thread, vec2), so that is the natural unit to
score.

Hooks modiff_cutlass.group_norm_silu_delta_quantize_nhwc / _pack_nhwc at the Python attribute
level (both are called as `modiff_cutlass.<name>(...)`, a plain module-attribute lookup at call
time -- confirmed at integration/kernels/int8_optimized.py:963 -- so wrapping the module attribute
before the model runs intercepts every real call) during an ACTUAL short generation, recording the
warp-all-zero fraction per call, IN CALL ORDER (a proxy for DDIM step order, since this UNet calls
the same fixed set of modulated layers once per step). This directly tests what the calibration
data implies (delta_scale grows at the tail => real deltas shrink there): zero-rate should be low
early and high late.

Run: python docs/int8_ahat_cache_2026-08-26/scripts/measure_zero_code_rate.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")

import torch  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

WARP_ELEMS = 64  # 32 threads x 2 elements/thread (vec2), matching the apply kernel's grid unit

records = []  # (call_index, tag, total_warps, zero_warps, total_elems, zero_elems)

_orig_flat = mc.group_norm_silu_delta_quantize_nhwc
_orig_pack = mc.group_norm_silu_delta_quantize_pack_nhwc


def _score(codes_flat, tag):
    n = codes_flat.numel()
    pad = (-n) % WARP_ELEMS
    if pad:
        codes_flat = torch.nn.functional.pad(codes_flat, (0, pad))
    warps = codes_flat.view(-1, WARP_ELEMS)
    zero_warp = (warps == 0).all(dim=1)
    records.append((len(records), tag, int(warps.shape[0]), int(zero_warp.sum()),
                    n, int((codes_flat[:n] == 0).sum())))


def _wrapped_flat(x, *args, **kwargs):
    out = _orig_flat(x, *args, **kwargs)
    codes = out.permute(0, 2, 3, 1).contiguous().reshape(-1) if out.dim() == 4 else out.reshape(-1)
    _score(codes.to(torch.int32), "flat_int8")
    return out


def _wrapped_pack(x, *args, **kwargs):
    out = _orig_pack(x, *args, **kwargs)
    b = out.permute(0, 2, 3, 1).contiguous().reshape(-1) if out.dim() == 4 else out.reshape(-1)
    b = b.to(torch.int32) & 0xFF
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    lo = lo - 16 * (lo > 7)
    hi = hi - 16 * (hi > 7)
    codes = torch.stack([lo, hi], dim=1).reshape(-1)
    _score(codes, "pack_int4")
    return out


mc.group_norm_silu_delta_quantize_nhwc = _wrapped_flat
mc.group_norm_silu_delta_quantize_pack_nhwc = _wrapped_pack

import integration.kernels.int8_optimized as i8opt  # noqa: E402
i8opt.modiff_cutlass.group_norm_silu_delta_quantize_nhwc = _wrapped_flat
i8opt.modiff_cutlass.group_norm_silu_delta_quantize_pack_nhwc = _wrapped_pack

import integration.benchmarks.benchmark_ldm as bl  # noqa: E402

STEPS, BATCH = 20, 4
sys.argv = ["benchmark_ldm.py", "--mode", "int8", "--batch_size", str(BATCH),
           "--steps", str(STEPS), "--num_samples", str(BATCH), "--skip_calibration"]

print(f"Running {STEPS}-step int8 generation, batch {BATCH}, to capture real delta codes...")
try:
    bl.main()
except SystemExit:
    pass

print(f"\ncaptured {len(records)} kernel calls")
if not records:
    print("NOTHING CAPTURED -- the hook did not fire.")
    sys.exit(1)

total_calls = len(records)
first_half = records[: total_calls // 2]
second_half = records[total_calls // 2:]


def agg(rows, label):
    tw = sum(r[2] for r in rows)
    zw = sum(r[3] for r in rows)
    te = sum(r[4] for r in rows)
    ze = sum(r[5] for r in rows)
    print(f"{label:>20}  calls={len(rows):4d}  elem zero-rate {100*ze/te:6.2f}%   "
          f"warp-all-zero rate {100*zw/tw:6.2f}%  ({zw}/{tw} warps)")


print()
agg(records, "ALL CALLS")
agg(first_half, "FIRST HALF (early)")
agg(second_half, "SECOND HALF (late)")

print("\nInterpretation: 'elem zero-rate' is the fraction of individual delta codes that are")
print("exactly 0. 'warp-all-zero rate' is the fraction of 64-element groups where EVERY code in")
print("the group is 0 -- the quantity that actually determines whether a predicated skip saves a")
print("real memory transaction. If warp-all-zero is much lower than elem zero-rate, zeros are")
print("scattered rather than clustered and this optimization has little to work with regardless")
print("of how high the raw zero-rate looks.")
