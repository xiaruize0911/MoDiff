"""P2 / fix #4: the end-to-end prize for AdaRound W4, measured with no kernel at all.

weight_zp_output_error.py established that on CONV OUTPUT error -- the metric AdaRound optimises, and
the one docs/paper_repro_2026-08-12/FINDINGS.md section 5 admitted it could not measure -- the paper's
AdaRound W4 beats our RTN+MSE by 1.35x (median 0.0504 vs 0.0680, winning on 54 of 70 convs), even
though ours wins on ||W - Q(W)|| (0.1293 vs 0.1506, both reproducing the committed values exactly).

A layer-output win is not an end-to-end win, and fix #4's cost is a CUDA kernel that does not exist, so
the number that should decide it is end-to-end. THAT NUMBER IS MEASURABLE TODAY WITHOUT THE KERNEL:
weight-only quantization needs no zero-point support anywhere in the datapath. Dequantize the 4-bit
weights back to fp16, run the model in fp16, and the sampled latent tells you what the WEIGHT CHOICE is
worth. The missing kernel is only needed to make it FAST, not to make it measurable.

This is a different situation from the activation fake-quant harness, which is untrustworthy (three
failed self-checks; see docs/zp_coverage_2026-08-13/FINDINGS_HARNESS.md). That harness had to emulate a
quantization GRID and got the grid wrong. Here nothing is emulated: the weights are bit-for-bit the ones
a deployed W4 kernel would multiply.

TWO CHECKPOINTS, TWO BASELINES -- the correctness point of this file. The AdaRound weights come from
Q-Diffusion's church_w4a8_ckpt.pth, whose fp32 weights differ from ours by a median relative 0.086.
Scoring both quantizations against OUR fp16 reference would charge AdaRound for that checkpoint
difference and call it quantization error. So each arm is scored against a reference built from ITS OWN
fp32 weights, in the same process:

    ref_ours   fp16 sample with our fp32 weights        arm_ours   fp16 sample with our W4
    ref_ada    fp16 sample with their fp32 weights      arm_ada    fp16 sample with their W4

and the comparison is of the two RATIOS, not of the two absolute relL2 values.

Run: python docs/zp_coverage_2026-08-13/scripts/weight_zp_end_to_end.py    # ~8 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/zp_coverage_2026-08-13/scripts")]
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402
from weight_zp_output_error import load_adaround, match, q_ours          # noqa: E402

D = "docs/zp_coverage_2026-08-13"
CKPT = "/workspace/quant_models/church_w4a8_ckpt.pth"
SEEDS = [1234, 20260805]


def sample(r, m, s, seeds):
    out = {}
    for sd in seeds:
        H.SEED = sd
        H.latent(r, m, s)                      # discard, as every harness here does
        lat, _ = H.latent(r, m, s)
        out[sd] = lat.float()
    return out


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    H.STEPS, H.BATCH = 50, 8
    H.AUTO_DELTA_TABLE = True
    import act_fake_quant as A

    r, m, s = H.build("fp16", None, "static")
    unet = m.model.diffusion_model
    convs = A.target_convs(unet)
    ada = load_adaround(CKPT)
    print(f"{len(convs)} convs, {len(ada)} adaround entries", flush=True)

    ours_fp32 = {k: convs[k].weight.data.clone() for k in convs}
    ada_fp32, ada_w4, missing = {}, {}, []
    for k in convs:
        key = match(k, ada)
        if key is None or tuple(ada[key]["weight"].shape) != tuple(ours_fp32[k].shape):
            missing.append(k)
            continue
        dev, dt = ours_fp32[k].device, ours_fp32[k].dtype
        ada_fp32[k] = ada[key]["weight"].to(dev, dt)
        ada_w4[k] = ada[key]["w_q"].to(dev, dt)
    print(f"matched {len(ada_w4)}/{len(convs)} convs; missing {len(missing)}", flush=True)
    if missing:
        print("REFUSING: not every conv matched, so the two arms would quantize different layer sets.")
        return 1

    def load(d):
        for k in convs:
            convs[k].weight.data.copy_(d[k])

    res = {}
    print("\nreference: our fp32 weights", flush=True)
    ref_ours = sample(r, m, s, SEEDS)
    print("arm: our W4 (RTN + MSE scale)", flush=True)
    load({k: q_ours(ours_fp32[k].float()).to(ours_fp32[k].dtype) for k in convs})
    arm_ours = sample(r, m, s, SEEDS)
    res["ours"] = [rel(arm_ours[sd], ref_ours[sd]) for sd in SEEDS]

    print("reference: their fp32 weights", flush=True)
    load(ada_fp32)
    ref_ada = sample(r, m, s, SEEDS)
    print("arm: their W4 (AdaRound + per-channel weight zero point)", flush=True)
    load(ada_w4)
    arm_ada = sample(r, m, s, SEEDS)
    res["adaround"] = [rel(arm_ada[sd], ref_ada[sd]) for sd in SEEDS]

    load(ours_fp32)                            # leave the model as we found it

    #: how different the two fp32 checkpoints are, as context for why each needs its own baseline
    ckpt_gap = [rel(ref_ada[sd], ref_ours[sd]) for sd in SEEDS]

    mo, ma = statistics.mean(res["ours"]), statistics.mean(res["adaround"])
    print(f"\n{'arm':34}{'relL2 vs own fp32':>20}{'per seed':>26}")
    print(f"{'ours (RTN + MSE scale)':34}{mo:20.4f}{str([round(x, 4) for x in res['ours']]):>26}")
    print(f"{'adaround + weight zero point':34}{ma:20.4f}"
          f"{str([round(x, 4) for x in res['adaround']]):>26}")
    print(f"\nthe two fp32 checkpoints differ by relL2 {statistics.mean(ckpt_gap):.4f} on the sampled "
          f"latent,\nwhich is why each arm is scored against its own baseline rather than a shared one")

    out = {"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "results": res,
           "ckpt_gap": ckpt_gap, "ours_mean": mo, "adaround_mean": ma,
           "adaround_gain": mo / ma if ma else None,
           "layer_output_gain": 1.35}
    json.dump(out, open(f"{D}/data/weight_zp_end_to_end.json", "w"), indent=1)
    print(f"wrote {D}/data/weight_zp_end_to_end.json")

    gain = mo / ma if ma else float("nan")
    print(f"\nAdaRound is worth {gain:.2f}x END TO END on weight-only W4 "
          f"(layer-output error said 1.35x)")
    #: W4A4's cross-process floor is 0.13% (FINDINGS_NOISE_FLOOR.md) and both arms here are measured in
    #: ONE process in a fixed order, which is the regime that file says is comparable.
    if gain > 1.15:
        print("\nFIX #4 IS WORTH REVISITING, and the earlier deprioritisation rested on the wrong\n"
              "metric: ||W - Q(W)|| favours ours, output error and the latent both favour AdaRound.\n"
              "The cost is a windowed per-output-pixel reduction for z_w*Sigma(a) -- the SAME missing\n"
              "capability fix #2's padding defect needed. Price them together: one kernel, two levers.")
    elif gain > 1.02:
        print("\nREAL BUT SMALL end to end. The layer-output win does not survive to the latent at full\n"
              "size, so fix #4 stays deprioritised -- now on the metric that was missing, rather than\n"
              "on ||W - Q(W)|| which was the wrong one.")
    else:
        print("\nNO END-TO-END WIN. The layer-output advantage does not reach the latent at all, so the\n"
              "deprioritisation of fix #4 stands and is now supported by the right metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
