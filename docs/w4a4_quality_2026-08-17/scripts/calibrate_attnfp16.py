"""Re-fit the int4 activation scales with the attention SCORE PATH in fp16, then report the difference.

WHY. `ATTN_FP16=1` measured +4.1% on the 16-image screen, which §0 shows is INSIDE that screen's 8.7/255
cross-process floor -- so the axis is unresolved, not refuted. And the test was confounded:
`int4_calibration_realckpt.pt` was fitted with attention QUANTIZED, so turning the score path off changes
the activations every conv sees while the scales stay fitted to the old distribution. This produces the
matching calibration so the axis can be measured instead of guessed.

The axis matters because it is a real structural difference from the reference, not a knob: qdiff's
`QuantAttnBlock` builds its q/k/v quantizers at `sm_abit` (8) rather than `act_bit` under the comment
"we do not reduce the bit in attention in this work", and its forward() never calls them at all. So the
paper's "W4A4" has a full-precision score path where ours runs int4 Q/K on the MMA kernels.

RESIDUAL CONFOUND, STATED. The static delta table was ALSO fitted with attention quantized, and this
script does not re-fit it -- that needs the qdiff export pipeline. So a static-delta arm using these
scales is matched on the activation grid and mismatched on the delta grid. A dynamic-delta arm has no
table and is fully matched, at the cost of needing its own baseline.

Run: python docs/w4a4_quality_2026-08-17/scripts/calibrate_attnfp16.py
"""
import os
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch                                                            # noqa: E402
from integration.utils.preflight import preflight, MODEL                # noqa: E402
preflight(*MODEL, what="calibrate_attnfp16.py")

OUT = "integration/calibration/int4_calibration_attnfp16.pt"
REF = "integration/calibration/int4_calibration_realckpt.pt"

import kernel_suites_bench as ks                                        # noqa: E402
import integration.benchmarks.benchmark_ldm as B                        # noqa: E402

ks.set_env("int4")
os.environ["MODIFF_QUANT_ATTN"] = "0"        # AFTER set_env, which writes 1 unconditionally
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_DELTA_REFRESH"] = "4"
os.environ["MODIFF_LINEAR"] = "1"            # L1, the arm being measured
os.environ["MODIFF_ACT_BITS"] = "8"
os.environ["MODIFF_WARMUP_STEPS"] = "5"

if os.path.exists(OUT):
    os.remove(OUT)                            # a present file would make _setup_model LOAD, not fit

runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="/workspace/fid_det/tmp", batch_size=16, steps=50,
    shape=(4, 32, 32), calibration_path=OUT)
model, sampler = runner._setup_model("int4")

# NON-VACUITY on the flag: the build log must not have installed quantized attention
runner._calibrate_int4(model, sampler)
assert os.path.exists(OUT), f"{OUT} was not written"

new = torch.load(OUT, weights_only=True)
ref = torch.load(REF, weights_only=True)
conv_new = {k: v for k, v in new.items() if not k.startswith("linear:")}
conv_ref = {k: v for k, v in ref.items() if not k.startswith("linear:")}
shared = sorted(set(conv_new) & set(conv_ref))
print(f"\nconv scales: {len(conv_new)} new, {len(conv_ref)} reference, {len(shared)} shared")
assert shared, "no shared layer names -- the two files are not comparable"

import statistics                                                        # noqa: E402
def _scale(entry):
    """Each entry is {'static_scale': float, 'smooth_scale': Tensor} -- the scalar is the grid."""
    return float(entry["static_scale"]) if isinstance(entry, dict) else float(
        torch.as_tensor(entry).flatten()[0])


ratios = [_scale(conv_new[k]) / max(_scale(conv_ref[k]), 1e-30) for k in shared]
changed = sum(1 for r in ratios if abs(r - 1.0) > 0.01)
print(f"  median new/ref scale ratio: {statistics.median(ratios):.4f}")
print(f"  layers changed by >1%:      {changed}/{len(shared)}")
print(f"  range:                      {min(ratios):.4f} .. {max(ratios):.4f}")

# If the calibration is identical the whole experiment is vacuous -- the flag did not reach the convs.
assert changed > len(shared) // 4, (
    f"only {changed}/{len(shared)} scales moved by >1%. Turning the attention score path off barely "
    f"changed the activations the convs see, so re-calibrating cannot explain the earlier +4.1% and the "
    f"confound this script exists to remove was not the confound.")
print(f"\nWROTE {OUT}")
print("Use with:  ATTN_FP16=1 FID_CALIB4=" + OUT)
