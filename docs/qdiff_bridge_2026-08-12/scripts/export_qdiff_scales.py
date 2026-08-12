"""A6: convert a Q-Diffusion calibration checkpoint into integration/'s scale format.

The two worlds share no code, so this is the whole bridge, and everything in it is arithmetic that
"runs" whether or not it is correct. The assertions below are the point of the script; the conversion
itself is four lines.

WHAT COMES IN. `ckpt.pth` from scripts/sample_diffusion_ldm.py, a flat state_dict with keys like
`model.<unet-path>.act_quantizer.{delta,zero_point}`. qdiff stores a STEP SIZE (`x_int = round(x/delta)
+ zero_point`); asymmetric unless --a_sym, per-tensor unless --act_tensor.

WHAT GOES OUT. integration/'s `apply_static_scales` format: `{dotted-module-path: float}` where the
float is a MULTIPLIER, `127.0 / absmax` -- the reciprocal of a step size. Or, for --kind delta,
`apply_int8_delta_scales`'s format: `{name: Tensor[256]}` of per-step multipliers.

THE TRAP THIS SCRIPT EXISTS TO PREVENT. Under --modulate, `act_quantizer.delta` is the step size of
the temporal delta `a_t - a_hat_{t-1}`, NOT of the activation `a_t`; its consumer is
apply_int8_delta_scales, not apply_static_scales. Feeding one to the other produces a model that runs,
samples, and is wrong. So --kind is cross-checked against the run's own `modulate` setting and a
mismatch is refused, rather than left to discipline.

Run:
  python .../export_qdiff_scales.py --run <qdiff_runs/act_sym> --kind static --out <path.pt> --dry-run
"""
import argparse
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch                                                                # noqa: E402
import yaml                                                                 # noqa: E402

SHIPPED_STATIC = "integration/calibration/int8_calibration_realckpt.pt"
SHIPPED_INT4 = "integration/calibration/int4_calibration_realckpt.pt"
#: integration/kernels/int8_optimized.py's MODIFF_MAX_STEPS -- the delta table length it forward-fills
MAX_STEPS = 256

#: The ONLY two rename rules. FusedResBlock re-registers original.in_layers[-1] as .in_conv and
#: original.out_layers[3] as .out_conv (fused_resblock.py:756,768); qdiff wraps in place and so reports
#: the raw LDM path. Verified set-equal to the shipped 70 by scripts/smoke_qdiff.py.
RENAME = ((".in_layers.2", ".in_conv"), (".out_layers.3", ".out_conv"))
#: One OR two index levels. input_blocks/output_blocks nest as <container>.<block>.<child>, but
#: middle_block is a single TimestepEmbedSequential whose ResBlocks sit at .0 and .2 -- one level.
KEEP = re.compile(r"^(input_blocks|middle_block|output_blocks)(\.\d+){1,2}\.(in|out)_conv$")


def map_name(raw):
    n = raw
    for a, b in RENAME:
        n = n.replace(a, b)
    return n if KEEP.match(n) else None


def read_run_config(run_dir):
    """The run writes sampling_config.yaml with mode 'a+', so repeated runs APPEND documents.
    Take the last one, not the first."""
    hits = []
    for dirpath, _, files in os.walk(run_dir):
        if "sampling_config.yaml" in files:
            hits.append(os.path.join(dirpath, "sampling_config.yaml"))
    if not hits:
        raise SystemExit(f"no sampling_config.yaml under {run_dir}")
    docs = [d for d in yaml.safe_load_all(open(sorted(hits)[0])) if d]
    return docs[-1], sorted(hits)[0]


def find_ckpt(run_dir):
    hits = [os.path.join(dp, "ckpt.pth") for dp, _, f in os.walk(run_dir) if "ckpt.pth" in f]
    if not hits:
        raise SystemExit(f"no ckpt.pth under {run_dir}")
    return sorted(hits)[0]


def collect(ckpt_path):
    """-> {raw_name: (delta_tensor, zero_point_float)} for every act quantizer in the checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in sd.items():
        if not k.endswith("act_quantizer.delta"):
            continue
        raw = k[len("model."):] if k.startswith("model.") else k
        raw = raw[:-len(".act_quantizer.delta")]
        zp = sd.get(k[:-len("delta")] + "zero_point")
        out[raw] = (v, float(zp) if zp is not None else 0.0)
    return out


def to_multiplier(delta, zp, act_bit, sym):
    """qdiff step size -> integration multiplier, always int8-normalised.

    set_static_scale (int8_optimized.py:1822) rescales by act_q/127 itself and end_delta_calibration
    hardcodes q=127, so emitting anything but the 127-based value double-applies the bit width.
    """
    d = float(delta)
    n_levels = (2 ** (act_bit - 1) - 1) if sym else (2 ** act_bit)
    if sym:
        absmax = d * n_levels
    else:
        x_min = -zp * d
        x_max = (n_levels - 1 - zp) * d
        absmax = max(abs(x_min), abs(x_max))
    return 127.0 / max(absmax, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a qdiff_runs/<name> directory")
    ap.add_argument("--kind", required=True, choices=["static", "delta"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--act-scales", default=None,
                    help="for --kind delta: the static .pt, used by the head policy")
    ap.add_argument("--delta-head", type=int, default=2,
                    help="steps at the head of the delta table clamped to act_scale/2 "
                         "(provably non-clipping: |a_t - a_hat_t+1| <= 2*act_absmax). 0 = flat.")
    ap.add_argument("--target", default="int8", choices=["int8", "int4"],
                    help="which shipped key set to validate against, and which act_bit to expect")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg, cfg_path = read_run_config(a.run)
    ckpt = find_ckpt(a.run)
    modulate = bool(cfg.get("modulate", False))
    act_bit = int(cfg.get("act_bit", 8))
    sym = bool(cfg.get("a_sym", False))
    print(f"  run      : {a.run}")
    print(f"  config   : {cfg_path}")
    print(f"  ckpt     : {ckpt}")
    print(f"  modulate={modulate}  act_bit={act_bit}  a_sym={sym}  a_min_max={cfg.get('a_min_max')}")

    # --- trap 2, enforced ---------------------------------------------------------------------
    expect = "delta" if modulate else "static"
    if a.kind != expect:
        raise SystemExit(
            f"REFUSED: --kind {a.kind} but this run has modulate={modulate}, which produces "
            f"'{expect}' scales.\n"
            "  Under --modulate, act_quantizer.delta is the step size of the TEMPORAL DELTA "
            "(a_t - a_hat_{t-1}); QuantModule.forward never quantizes a_T at all.\n"
            "  Exporting it as an activation scale would produce a model that runs and is wrong.")
    if sym and not cfg.get("a_min_max"):
        raise SystemExit(
            "REFUSED: --a_sym without --a_min_max. UniformAffineQuantizer's 'mse' branch computes "
            "zero_point without checking self.sym while sym sets n_levels=127, so quantize() clamps "
            "to [0,126] with zp~128 and the search optimises garbage.")

    raw = collect(ckpt)
    print(f"  act quantizers in ckpt: {len(raw)}")

    kept, dropped = {}, []
    for rname, (delta, zp) in raw.items():
        n = map_name(rname)
        if n is None:
            dropped.append({"raw_name": rname, "delta": float(delta.reshape(-1)[0]),
                            "zero_point": zp, "numel": int(delta.numel()),
                            "excluded_because": "not one of integration's 70 quantized convs"})
            continue
        if delta.numel() != 1:
            raise SystemExit(
                f"REFUSED: {rname} has a per-channel delta (numel={delta.numel()}). "
                "--act_tensor is wired to channel_wise (the opposite of its help text); "
                "apply_static_scales cannot consume that.")
        kept[n] = to_multiplier(delta, zp, act_bit, sym)

    # SMOOTHQUANT, and why the int4 export is bare floats rather than a dict.
    #
    # int4's shipped file is {name: {"static_scale", "smooth_scale"}} with smoothing LIVE
    # (per-input-channel, 2.96-5.39); int8's is bare floats with smoothing identity. The kernel
    # applies `x * smooth_inv` and THEN the scale, and int8_optimized.py:1616-1618 derives the
    # shipped static_scale from the SMOOTHED range.
    #
    # qdiff has no SmoothQuant, so its delta measures the UNSMOOTHED activation. Grafting a qdiff
    # static_scale onto the shipped smooth_scale would therefore be wrong by the per-channel smooth
    # factor -- the scale would assume a smoothing that its calibration never saw. And qdiff cannot
    # supply a smoothed range: it only reports a per-TENSOR delta, and asking for per-channel
    # (--act_tensor, which confusingly sets channel_wise) produces a shape apply_*_static_scales
    # cannot consume.
    #
    # So a qdiff int4 file is bare floats and SmoothQuant is off. That is SELF-CONSISTENT -- scale
    # from unsmoothed range, applied unsmoothed -- but it bundles two changes against the shipped
    # arm. The A/B therefore carries a control arm (shipped static_scale re-emitted as bare floats)
    # to bound how much of any difference is SmoothQuant rather than calibration.
    shipped = torch.load(SHIPPED_INT4 if a.target == "int4" else SHIPPED_STATIC,
                         map_location="cpu", weights_only=False)
    if a.target == "int4" and act_bit != 4:
        raise SystemExit(f"REFUSED: --target int4 but the run calibrated at act_bit={act_bit}. "
                         "An 8-bit-optimal clip rescaled to 15 levels is not clip-optimal; measured, "
                         "it loses to the shipped scale (data/a4_clip_optimal_ab.json).")
    missing = sorted(set(shipped) - set(kept))
    extra = sorted(set(kept) - set(shipped))
    if missing or extra:
        raise SystemExit(f"REFUSED: key set differs from the shipped 70.\n"
                         f"  missing {len(missing)}: {missing[:5]}\n  extra {len(extra)}: {extra[:5]}")
    bad = {k: v for k, v in kept.items() if not math.isfinite(v) or v <= 0}
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} non-finite or non-positive scales: {list(bad)[:5]}")
    print(f"  mapped   : {len(kept)}/70, all finite and positive; {len(dropped)} dropped to sidecar")

    # --- ratio audit: the characteristic failure is a plausible number ---------------------------
    # EXPECTED DIRECTION, and the plan for this work had it backwards. Clipping means
    # max|x| * static_scale > 127, so relieving it requires a SMALLER scale: ratio below 1, not above.
    # Measured 2026-08-12: median 0.348, 57 of 70 layers shrink. The mechanism is that the shipped
    # scale is mean(127/absmax_i) over calibration calls (int8_optimized.py:1620) -- a mean of
    # reciprocals, dominated by the calls with the SMALLEST absmax -- so a layer whose activation
    # range swings across timesteps gets a wildly inflated scale. The worst layer is
    # middle_block.0.out_conv at 14.5x, deep in the net where the range swings most.
    # The [0.02, 50] band below is a unit-error guard, not a direction check.
    def _old(v):
        return float(v["static_scale"]) if isinstance(v, dict) else float(v)
    rows = sorted(((k, kept[k], _old(shipped[k]), kept[k] / _old(shipped[k])) for k in kept),
                  key=lambda r: -abs(math.log(r[3])))
    print(f"\n  new/old ratio: min {min(r[3] for r in rows):.3f}  "
          f"median {sorted(r[3] for r in rows)[len(rows)//2]:.3f}  max {max(r[3] for r in rows):.3f}")
    print("  most-moved layers:")
    for k, new, old, ratio in rows[:6]:
        print(f"    {k:42s} {old:10.3f} -> {new:10.3f}   x{ratio:.3f}")
    out_of_band = [r for r in rows if not (0.02 <= r[3] <= 50.0)]
    if out_of_band:
        print(f"\n  WARNING: {len(out_of_band)} layers outside [0.02, 50]: "
              f"{[(r[0], round(r[3], 4)) for r in out_of_band[:5]]}")
        print("  ratio ~1.0 means the export silently reproduced the old value; "
              "ratio ~1000 means a unit error.")

    if a.dry_run:
        print("\n  --dry-run: nothing written")
        return 1 if out_of_band else 0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.kind == "static":
        torch.save(kept, a.out)
    else:
        if a.delta_head > 0:
            if not a.act_scales:
                raise SystemExit("--kind delta with --delta-head > 0 needs --act-scales")
            act = torch.load(a.act_scales, map_location="cpu", weights_only=False)
        table = {}
        for k, v in kept.items():
            t = torch.full((MAX_STEPS,), float(v), dtype=torch.float32)
            if a.delta_head > 0:
                # |a_t - a_hat_{t+1}| <= 2*act_absmax, so act_scale/2 cannot clip. qdiff's a_hat_T is
                # the UNQUANTIZED a_T, so its step-1 delta carries none of the t=T quantization error
                # the integration kernel's does -- the pooled scalar is too coarse for the head.
                t[:a.delta_head] = min(float(v), float(act[k]) / 2.0)
            table[k] = t
        torch.save(table, a.out)
    print(f"\n  wrote {a.out}")

    side = os.path.join(os.path.dirname(a.out), "qdiff_all_168.json")
    json.dump({"kind": a.kind, "run": a.run, "act_bit": act_bit, "a_sym": sym,
               "modulate": modulate, "n_kept": len(kept), "n_dropped": len(dropped),
               "kept": kept, "ratio_vs_shipped": {r[0]: r[3] for r in rows},
               "dropped": dropped}, open(side, "w"), indent=1)
    print(f"  wrote {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
