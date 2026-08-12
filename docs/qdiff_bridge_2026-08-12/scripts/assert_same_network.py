"""A0 gate: do `scripts/` and `integration/` calibrate the SAME network?

WHY THIS RUNS FIRST. The Q-Diffusion bridge derives per-layer activation scales in the `scripts/`
world and consumes them in the `integration/` world. If the two build different weights, every scale
is silently wrong and nothing downstream can detect it -- the model still runs, still samples, and the
numbers still look plausible.

There is a concrete reason to suspect they differ. `scripts/sample_diffusion_ldm.py:527-529` does

    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

unconditionally, right after loading. `integration/benchmarks/benchmark_ldm.py:152` (`load_model`)
does not. LSUN-churches ships EMA weights in the same checkpoint, so the two worlds would be running
different networks that load from the same file.

WHAT THIS ASSERTS
  non-EMA build vs integration build : all 70 conv weights bit-identical  (this is the bridge premise)
  EMA build     vs integration build : NOT identical                      (this is why --no_ema exists)

WHAT IT DOES NOT ASSERT, AND WHY THE W4A4 BRIDGE FAILED ANYWAY.

This compares FP weights, and passing it does NOT license the bridge at every bit width. Two
explanations were tried for the W4A4 failure and only the second survived measurement:

  REFUTED -- "the quantized weights differ". integration uses per-output-channel SYMMETRIC MSE at
  Q=7 (int4_optimized.py:59); qdiff defaulted to ASYMMETRIC. Plausible, and wrong: adding --w_sym so
  both are symmetric per-channel MSE moved the assumed activation range only 3.769 -> 3.586 and the
  PTQ relL2 not at all (1.1667 -> 1.2200). The weight scheme is not the cause.

  CONFIRMED -- the CALIBRATION DATA comes from the wrong trajectory. sample_diffusion_ldm.py's
  --generate branch runs at :553 and exit()s at :565, BEFORE `if opt.ptq:` at :568, so the latents
  it saves are the FP16 model's trajectory. integration's _calibrate_int8/_calibrate_int4 instead
  call sampler.sample() on the model BEING calibrated, so they observe the quantized model's own
  trajectory.

    W8A8: the quantized trajectory tracks fp16 (relL2 ~0.11), so fp16 latents are representative and
          qdiff's better statistic wins -- 2.29x on the PTQ baseline.
    W4A4: the quantized trajectory DIVERGES (relL2 0.42-0.79). The activations the model actually
          sees at run time are ~4x larger than what qdiff measured on fp16 latents (assumed range
          3.6 against an optimum near 14.8), so the scale clips catastrophically: 1.19 against a
          0.4885 control.

So the bridge is valid exactly where quantization does not move the sampling trajectory. Making it
work at 4 bits needs a two-pass bootstrap -- quantize with a rough scale, generate calibration data
from THAT model, recalibrate -- not a better statistic.

The second assertion matters as much as the first. If EMA turned out to be a no-op here, `--no_ema`
would be dead code and the whole EMA concern would be a misreading -- better to find that out now than
to carry a flag nobody needs.

Everything is on CPU: this compares weight VALUES, and two 2.7 GB models on the GPU would contend with
nothing else usefully.

Run: python docs/qdiff_bridge_2026-08-12/scripts/assert_same_network.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
# `taming` is a vendored submodule, not an installed package -- ldm.models.autoencoder imports it at
# module scope, so every entry point in this repo prepends it. Same list dynamic_delta_ab.py:31 uses.
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch                                                                # noqa: E402
from omegaconf import OmegaConf                                             # noqa: E402
from ldm.util import instantiate_from_config                                # noqa: E402

CKPT = "models/ldm/lsun_churches256/model.ckpt"
#: integration/ loads this one (benchmark_ldm.py's config_path default)
CFG_INTEGRATION = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
#: scripts/ derives its config from the CHECKPOINT's directory, not from configs/ --
#: sample_diffusion_ldm.py:471-486 globs `<ckpt_dir>/config.yaml`. Verified 2026-08-12 that the two
#: files' unet_config params are identical, so any weight difference is EMA and not architecture.
CFG_SCRIPTS = "models/ldm/lsun_churches256/config.yaml"
OUT = "docs/qdiff_bridge_2026-08-12/data/assert_same_network.json"


def build(cfg_path, sd, apply_ema):
    conf = OmegaConf.load(cfg_path)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    if apply_ema:
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    return model.eval()


def conv_names(unet):
    """The 70 convs integration actually quantizes, by integration's own predicate.

    Mirrors convert_model_to_optimized_int8: in_channels >= 32, groups == 1, kernel > 1x1, name has no
    'skip', not under `out.`. Returns raw LDM paths (`...in_layers.2`), which is what the qdiff side
    would see -- the `.in_conv` rename happens later, inside FusedResBlock.
    """
    import torch.nn as nn
    out = []
    for name, m in unet.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.in_channels < 32 or m.groups != 1 or m.kernel_size == (1, 1):
            continue
        if "skip" in name or name.startswith("out."):
            continue
        out.append(name)
    return sorted(out)


def compare(a, b, names):
    """Per-layer bit-equality plus relative L2, so a near-miss is distinguishable from a mismatch."""
    ma, mb = dict(a.named_modules()), dict(b.named_modules())
    rows = []
    for n in names:
        wa, wb = ma[n].weight.detach().float(), mb[n].weight.detach().float()
        same = bool(torch.equal(wa, wb))
        rel = float((wa - wb).norm() / wa.norm().clamp_min(1e-12))
        rows.append({"layer": n, "identical": same, "rel_l2": rel})
    return rows


def main():
    if not os.path.exists(CKPT):
        print(f"FAIL: {CKPT} missing")
        return 1
    print(f"loading {CKPT} ...", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)

    print("building integration-side model (no EMA) ...", flush=True)
    m_int = build(CFG_INTEGRATION, sd, apply_ema=False)
    unet = m_int.model.diffusion_model
    names = conv_names(unet)
    print(f"  {len(names)} quantizable convs by integration's predicate")

    print("building scripts-side model, EMA OFF ...", flush=True)
    m_noema = build(CFG_SCRIPTS, sd, apply_ema=False)
    print("building scripts-side model, EMA ON ...", flush=True)
    m_ema = build(CFG_SCRIPTS, sd, apply_ema=True)

    r_noema = compare(unet, m_noema.model.diffusion_model, names)
    r_ema = compare(unet, m_ema.model.diffusion_model, names)

    n_same_noema = sum(r["identical"] for r in r_noema)
    n_same_ema = sum(r["identical"] for r in r_ema)
    worst_ema = max(r_ema, key=lambda r: r["rel_l2"])

    print()
    print(f"  non-EMA vs integration : {n_same_noema}/{len(names)} identical")
    print(f"  EMA     vs integration : {n_same_ema}/{len(names)} identical  "
          f"(worst rel_L2 {worst_ema['rel_l2']:.4f} on {worst_ema['layer']})")

    ok_premise = n_same_noema == len(names)
    ok_ema_matters = n_same_ema < len(names)
    print()
    print(f"  [{'PASS' if ok_premise else 'FAIL'}] bridge premise: non-EMA builds agree bit-for-bit")
    print(f"  [{'PASS' if ok_ema_matters else 'FAIL'}] --no_ema is load-bearing: EMA changes the weights")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"n_convs": len(names), "n_identical_noema": n_same_noema,
               "n_identical_ema": n_same_ema,
               "worst_ema_rel_l2": worst_ema["rel_l2"],
               "worst_ema_layer": worst_ema["layer"],
               "premise_ok": ok_premise, "ema_matters": ok_ema_matters,
               "per_layer_ema": r_ema}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0 if (ok_premise and ok_ema_matters) else 1


if __name__ == "__main__":
    sys.exit(main())
