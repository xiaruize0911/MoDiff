"""Why does folding SmoothQuant help one conv's first step and hurt the 50-step trajectory?

THE DISAGREEMENT. d77c516 measured that restoring `smooth_scale` on checkpoint apply is worth ~2x
on one layer's MoDiff first step (rel ~0.20 against ~0.40), and test_int4_export_apply still asserts
it (`apply_acc < legacy_acc - 0.05`). docs/qdiff_bridge_2026-08-12/FINDINGS.md §5a/§5e measured the
opposite over 50 DDIM steps on the real checkpoint: SmoothQuant OFF takes W4A4 PTQ from 0.7121 to
0.4823 and MoDiff from 0.4220 to 0.3540, and the int4 defaults now ship bare floats for that reason.

The recorded candidate is that folding s widens each output channel's weight range, so at 15 weight
levels the added weight error exceeds the activation clipping it prevents. This measures both terms.

WHAT MAKES IT MEASURABLE WITHOUT A FORWARD PASS. The shipped calibration carries s per input channel
and the per-tensor static scale, and s is defined as sqrt(act_max_c / w_max_c) with w_max_c taken
from the weights. The weights are still on the module (`_orig_weight`, an alias kept because the
file-load path never calls end_calibration), so:

    act_max_c = s_c^2 * w_max_c

recovers the per-input-channel activation range the calibration actually observed. That is a
recovery, not an assumption, and §3's gate below proves it: static_scale is 7/max_c(act_max_c/s_c) by
construction, so if the recovery is right, max_c of the SMOOTHED quantized range must come back as
exactly 7.0. If it does not, this script refuses to report the clipping half.

Three sections:
  1. the synthetic fixture the gate uses -- does it contain the effect at all?
  2. weight reconstruction error, Q(W) against Q(W*s), 70 real convs
  3. implied activation clipping, smoothed against unsmoothed, 70 real convs

Run: python docs/smoothquant_fold_2026-08-12/scripts/smoothquant_fold_probe.py   # ~2 min, needs GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import torch.nn as nn                                                       # noqa: E402

SHIPPED = "integration/calibration/int4_calibration_realckpt.pt"
OUT = "docs/smoothquant_fold_2026-08-12/data/fold_probe.json"
QW = 7.0        #: int4 weight levels, +-7
QA = 7.0        #: int4 activation levels, +-7


def spread(v):
    """max/median, the statistic SmoothQuant exists to reduce, plus the raw ends."""
    v = sorted(float(x) for x in v)
    med = statistics.median(v)
    return {"min": v[0], "median": med, "max": v[-1],
            "max_over_median": v[-1] / med if med > 0 else float("inf")}


def w_err(w_flat, scale_fn):
    """Relative Frobenius error of the reconstructed weight, per output channel scale."""
    sc = scale_fn(w_flat)
    q = (w_flat / sc[:, None]).round().clamp(-QW, QW) * sc[:, None]
    return float((q - w_flat).norm() / w_flat.norm())


def section1_fixture():
    """The gate's own fixture: random Gaussian weights, random Gaussian input.

    SmoothQuant migrates PER-INPUT-CHANNEL outliers out of the activation and into the weights. A
    Gaussian activation has no channel that is an outlier -- every channel's absmax concentrates on
    the same value as the channel count grows -- so s should come back nearly uniform and folding it
    should be nearly inert. That is a property of the fixture, not of SmoothQuant.
    """
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d, _int4_weight_scale
    torch.manual_seed(0)
    ref_conv = nn.Conv2d(256, 256, 3, padding=1).to("cuda")
    m = OptimizedInt4Conv2d(ref_conv).to("cuda")
    m.layer_name = "fixture"
    m.set_calibrating(True)
    _ = m(torch.randn(16, 256, 32, 32, device="cuda"))
    act_max = m._act_channel_max.detach().float().clone()      # captured before end_calibration
    m.set_calibrating(False)
    s = m.smooth_scale.detach().float().reshape(-1).clone()

    w = ref_conv.weight.data.detach().float()
    K, C = w.shape[0], w.shape[1]
    w_flat = w.reshape(K, -1)
    w_folded = (w * s.view(1, -1, 1, 1)).reshape(K, -1)
    return {"act_max_per_in_channel": spread(act_max), "smooth_scale": spread(s),
            "w_err_plain": w_err(w_flat, _int4_weight_scale),
            "w_err_folded": w_err(w_folded, _int4_weight_scale),
            "shape": [K, C]}


def section23_real():
    """The real checkpoint's 70 calibrated convs, from the shipped file's own s and static_scale."""
    import dynamic_delta_ab as H
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d, _int4_weight_scale
    H.STEPS, H.BATCH = 50, 8
    os.environ["MODIFF_LINEAR"] = "0"
    runner, model, sampler = H.build("int4_baseline", SHIPPED, "static")
    shipped = torch.load(SHIPPED, map_location="cpu", weights_only=True)

    rows, bad_recovery, dead_total = [], [], 0
    for mod in model.model.diffusion_model.modules():
        if not isinstance(mod, OptimizedInt4Conv2d) or mod.layer_name not in shipped:
            continue
        entry = shipped[mod.layer_name]
        if not isinstance(entry, dict):
            continue
        if mod._orig_weight is None:
            bad_recovery.append(f"{mod.layer_name}: _orig_weight released, cannot recover")
            continue
        w = mod._orig_weight.detach().float()
        K = w.shape[0]
        s = mod.smooth_scale.detach().float().reshape(-1)
        static = float(mod.static_input_scale.item())

        # --- recover the observed per-input-channel activation range ---
        w_max = w.abs().amax(dim=(0, 2, 3))                    # [C_in], as _apply_smoothquant took it
        act_max = s.pow(2) * torch.clamp(w_max, min=1e-8)
        # channels where the fold substituted identity (dead weights) or hit a clamp carry no
        # recoverable range; count them rather than letting them silently distort the aggregate.
        dead = (w_max <= 1e-12) | (s <= 1.0001e-4) | (s >= 0.9999e4)
        dead_total += int(dead.sum())
        live = ~dead

        # --- section 3 gate: static_scale IS 7/max(act_max/s), so this must return 7.0 ---
        q_smoothed = (act_max[live] / s[live]) * static
        gate = float(q_smoothed.max()) if int(live.sum()) else float("nan")
        q_unsmoothed = act_max[live] * static
        clipping = q_unsmoothed[q_unsmoothed > QA]

        w_flat = w.reshape(K, -1)
        w_folded = (w * s.view(1, -1, 1, 1)).reshape(K, -1)
        rows.append({
            "layer": mod.layer_name, "C_in": int(w.shape[1]), "K": K,
            "static_scale": static, "smooth_scale": spread(s[live].tolist() or [1.0]),
            "act_max": spread(act_max[live].tolist() or [1.0]),
            "gate_smoothed_qmax": gate,
            "clip_frac_smoothed": float((q_smoothed > QA).float().mean()),
            "clip_frac_unsmoothed": float((q_unsmoothed > QA).float().mean()),
            #: how far past the ceiling the CLIPPING channels sit -- the median over all channels
            #: would sit below 1.0 and read as "no clipping" while 43% of them clip.
            "overshoot_of_clipping_median": float(clipping.median() / QA) if clipping.numel() else 0.0,
            "overshoot_of_clipping_max": float(clipping.max() / QA) if clipping.numel() else 0.0,
            #: the scale a correct UNsmoothed calibration would have used, for the arm that removes
            #: the widened weight range and the clipping at the same time (see nosmooth_recal.py)
            "static_scale_unsmoothed": QA / float(act_max[live].max()),
            #: CLIPPING SELECTIVITY. fold_plus_clip_ab.py found the fold's sign flips with the
            #: clipping regime: better at k=1, worse at k=5. This is the candidate reason, and it is
            #: purely distributional. Over-scale by k and a channel clips iff its max exceeds 1/k of
            #: the layer's global max, so what matters is how SPREAD the channel maxima are.
            #: SmoothQuant equalises them by construction -- smoothed_max_c = act_max_c / s_c =
            #: sqrt(act_max_c * w_max_c), a geometric mean -- so under the fold the same k clips
            #: nearly every channel, while unsmoothed it clips only the outlier tail.
            "clip_selectivity": {
                str(k): {"folded": float(((act_max[live] / s[live]) /
                                          (act_max[live] / s[live]).max() > 1.0 / k).float().mean()),
                         "unfolded": float((act_max[live] / act_max[live].max() > 1.0 / k)
                                           .float().mean())}
                for k in (1.0, 2.5, 5.0, 10.0)},
            "w_err_plain": w_err(w_flat, _int4_weight_scale),
            "w_err_folded": w_err(w_folded, _int4_weight_scale),
        })
    del runner, model, sampler
    torch.cuda.empty_cache()
    return rows, bad_recovery, dead_total


def main():
    print("section 1: the fixture test_int4_export_apply actually measures", flush=True)
    fx = section1_fixture()
    print(f"  activation absmax per input channel : max/median {fx['act_max_per_in_channel']['max_over_median']:.3f}"
          f"   ({fx['act_max_per_in_channel']['min']:.3f} .. {fx['act_max_per_in_channel']['max']:.3f})")
    print(f"  smooth_scale s                      : max/median {fx['smooth_scale']['max_over_median']:.3f}"
          f"   ({fx['smooth_scale']['min']:.3f} .. {fx['smooth_scale']['max']:.3f})")
    print(f"  weight recon error  plain {fx['w_err_plain']:.4f}  folded {fx['w_err_folded']:.4f}"
          f"   ({fx['w_err_folded'] / fx['w_err_plain']:.3f}x)")

    print("\nsections 2-3: the real checkpoint's 70 calibrated convs", flush=True)
    rows, bad, dead = section23_real()
    if not rows:
        print("FAIL: no layers recovered")
        return 1

    gates = [r["gate_smoothed_qmax"] for r in rows]
    off = [r["layer"] for r in rows if abs(r["gate_smoothed_qmax"] - QA) > 0.05 * QA]
    print(f"  gate: max smoothed quantized range = {statistics.median(gates):.4f} median "
          f"(must be {QA} -- static_scale is defined as {QA:.0f}/that max)")
    print(f"        {len(rows) - len(off)}/{len(rows)} layers within 5% of {QA:.0f}; "
          f"{dead} identity/clamped channels excluded")
    if len(off) > len(rows) * 0.1:
        print(f"REFUSED: the act_max recovery does not reproduce static_scale on {len(off)} layers, "
              f"e.g. {off[:4]}. The clipping half below would be arithmetic on a wrong quantity.")
        return 1

    wp = [r["w_err_plain"] for r in rows]
    wf = [r["w_err_folded"] for r in rows]
    ratio = [r["w_err_folded"] / r["w_err_plain"] for r in rows]
    worse = sum(1 for r in ratio if r > 1.0)
    print(f"\n  WEIGHT half -- relative Frobenius error of the reconstructed 4-bit weight")
    print(f"    unfolded W      median {statistics.median(wp):.4f}   worst {max(wp):.4f}")
    print(f"    folded   W*s    median {statistics.median(wf):.4f}   worst {max(wf):.4f}")
    print(f"    folded/unfolded median {statistics.median(ratio):.3f}x, folding is worse on "
          f"{worse}/{len(rows)} layers")

    cs = [r["clip_frac_smoothed"] for r in rows]
    cu = [r["clip_frac_unsmoothed"] for r in rows]
    ov = [r["overshoot_of_clipping_median"] for r in rows]
    ovx = [r["overshoot_of_clipping_max"] for r in rows]
    print(f"\n  CLIPPING half -- input channels whose observed max exceeds the +-{QA:.0f} grid")
    print(f"    smoothed        median {statistics.median(cs) * 100:.2f}% of channels")
    print(f"    unsmoothed      median {statistics.median(cu) * 100:.2f}% of channels; of those, "
          f"median {statistics.median(ov):.2f}x past the ceiling, worst layer {max(ovx):.2f}x")

    print(f"\n  So dropping the fold TRADES {statistics.median(ratio):.3f}x weight error for "
          f"{statistics.median(cu) * 100:.1f}% clipped channels, and §5e measured that trade as a win.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"fixture": fx, "layers": rows, "bad_recovery": bad,
               "dead_channels_excluded": dead,
               "summary": {"w_err_plain_median": statistics.median(wp),
                           "w_err_folded_median": statistics.median(wf),
                           "w_err_ratio_median": statistics.median(ratio),
                           "layers_fold_worse": worse, "layers": len(rows),
                           "clip_frac_smoothed_median": statistics.median(cs),
                           "clip_frac_unsmoothed_median": statistics.median(cu),
                           "overshoot_unsmoothed_median": statistics.median(ov)}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
