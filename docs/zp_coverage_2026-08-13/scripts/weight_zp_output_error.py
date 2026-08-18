"""P2 / fix #4: does the paper's AdaRound W4 beat ours on the metric AdaRound actually optimises?

WHERE THE EVIDENCE STOPS TODAY. docs/paper_repro_2026-08-12/FINDINGS.md section 5 deprioritised fix #4
on weight RECONSTRUCTION error over the 70 convs:

    qdiff AdaRound                  median 0.1506   worst 0.3110
    ours, RTN + MSE                 median 0.1296   worst 0.2588
    AdaRound re-quantised on ours   median 0.1581   worst 0.3235

and then said the honest thing: "AdaRound optimises block OUTPUT error rather than ||W-Q(W)||, so this
does not prove ours is better end to end." That caveat is the whole remaining gap in fix #4's case, and
it is measurable offline -- no kernel, no sampler, no per-output-pixel reduction -- because the question
is about the WEIGHTS, not about how they are multiplied.

WHAT THIS MEASURES. Per conv, on REAL captured input activations, the relative error of the conv output
against an fp32-weight reference:

    err(W_q) = ||conv(x, W_q) - conv(x, W_fp32)|| / ||conv(x, W_fp32)||

for three weight sets:

    ours        RTN + the shipped MSE-optimal per-channel scale (_int4_weight_scale)
    adaround    the paper's church_w4a8_ckpt.pth weights, dequantized with THEIR per-channel scale
                AND per-channel zero point (z_w in 1..14, so it is genuinely asymmetric)
    rtn_sym     plain absmax RTN, as a floor to show the MSE scale is doing something

ACTIVATIONS ARE REAL AND ARE THE CONV'S OWN INPUT. Weight-only error is measured, so x is held fixed
across arms: any difference is the weight choice. Captured with a forward_pre_hook on the fp16 model,
where ResBlocks call `self.in_conv(h)` normally -- NOT on the int4 model, where the fused path enters
the conv through forward_from_int4() and a hook never fires. That trap has already cost this project a
whole run (docs/zero_point_2026-08-13/FINDINGS.md, mistake #3), so it is worth restating.

WHY THIS DECIDES FIX #4 EITHER WAY:

  * If ours wins on OUTPUT error too, fix #4 is closed on evidence: importing AdaRound would need a new
    windowed reduction kernel (z_w * Sigma(a) is per-output-PIXEL, so it cannot fold into the bias) to
    buy something that measures worse than what already ships.
  * If AdaRound wins by enough to matter, fix #4 is back on, and the cost is then known precisely --
    and it is the SAME capability fix #2's padding defect needed (a per-output-pixel correction), which
    would change the economics of both at once.

THE ARITHMETIC THAT MAKES THIS EXPENSIVE, restated so the conclusion is checkable:
    sum_i (w_q[k,i] - z_w[k]) * a[i] = sum_i w_q[k,i]*a[i] - z_w[k] * sum_i a[i]
`sum_i a[i]` runs over the conv WINDOW at each output pixel, so it is per-output-pixel and cannot fold
into a per-output-channel bias. It is cheap in principle -- it does not depend on the output channel --
but it is a kernel that does not exist.

Run: python docs/zp_coverage_2026-08-13/scripts/weight_zp_output_error.py   # ~5 min, needs the GPU
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
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
import dynamic_delta_ab as H                                              # noqa: E402

D = "docs/zp_coverage_2026-08-13"
CKPT = "/workspace/quant_models/church_w4a8_ckpt.pth"
Q = 7.0
#: cap the captured activation per conv; the metric is a ratio of norms and converges fast, and the
#: full batch of every conv's input does not fit
MAX_N = 2


class Grab:
    """Capture one conv input. See the docstring: fp16 model only, where the call is an ordinary one."""

    def __init__(self):
        self.x = None

    def __call__(self, mod, args):
        if self.x is None:
            self.x = args[0].detach()[:MAX_N].float().clone()
        return None


def int4_weight_scale(wf):
    """The shipped per-channel MSE-optimal scale, imported rather than reimplemented."""
    from integration.kernels.int4_optimized import _int4_weight_scale
    return _int4_weight_scale(wf)


def q_ours(w):
    K = w.shape[0]
    wf = w.reshape(K, -1).float()
    sc = int4_weight_scale(wf)
    return ((wf / sc[:, None]).round().clamp(-Q, Q) * sc[:, None]).reshape_as(w)


def q_rtn_sym(w):
    K = w.shape[0]
    wf = w.reshape(K, -1).float()
    sc = wf.abs().amax(dim=1).clamp(min=1e-9) / Q
    return ((wf / sc[:, None]).round().clamp(-Q, Q) * sc[:, None]).reshape_as(w)


def load_adaround(path, n_bits=4):
    """Apply Q-Diffusion's AdaRoundQuantizer to each conv's weight. APPLY, not invert.

    THE FIRST VERSION OF THIS FUNCTION WAS WRONG AND ITS OUTPUT LOOKED LIKE A FINDING. It treated the
    checkpoint's `weight` as integer codes and computed (weight - z) * delta, which produced a median
    weight-reconstruction error of 3.27 and a conv-output error of 10.33 -- and the script duly printed
    "FIX #4 IS CLOSED ON EVIDENCE, AdaRound loses by 152x". 3.27 is impossible: zeroing the weights
    entirely gives 1.0. The tell was there in the number, which is the same magnitude test this tree
    already uses ("anything past 3x is bug magnitudes, not a result").

    What the checkpoint actually holds, per quantized module:
        weight                    the ORIGINAL fp32 weight (absmax 0.374 here)
        weight_quantizer.alpha    AdaRound's learned rounding parameter, weight-shaped
        weight_quantizer.delta    per-output-channel step
        weight_quantizer.zero_point   per-output-channel offset, 6..12 on this layer

    So the quantizer has to be run, exactly as AdaRoundQuantizer.forward does at inference with the
    hard rounding target:

        x_floor = floor(w / delta)
        x_int   = x_floor + (alpha >= 0)              # the learned rounding decision
        x_quant = clamp(x_int + zero_point, 0, 2**n_bits - 1)
        w_q     = (x_quant - zero_point) * delta

    `alpha >= 0` is the hard target: AdaRound's h(alpha) = clamp(sigmoid(alpha)*1.2 - 0.1, 0, 1) is
    thresholded at 0.5, and sigmoid(alpha) >= 0.5 exactly when alpha >= 0.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    levels = 2 ** n_bits - 1
    out = {}
    for k in sd:
        if not k.endswith("weight_quantizer.delta"):
            continue
        base = k[: -len("weight_quantizer.delta")]
        d = sd.get(k)
        z = sd.get(base + "weight_quantizer.zero_point")
        w = sd.get(base + "weight")
        alpha = sd.get(base + "weight_quantizer.alpha")
        if d is None or z is None or w is None:
            continue
        w = w.float()
        shape = (-1,) + (1,) * (w.dim() - 1)
        d = d.float().reshape(shape)
        z = z.float().reshape(shape)
        x_floor = torch.floor(w / d)
        if alpha is not None:
            x_int = x_floor + (alpha.float() >= 0).to(w.dtype)
        else:
            #: no alpha -> plain nearest rounding, and the row is flagged so the summary can say so
            x_int = torch.round(w / d)
        x_quant = torch.clamp(x_int + z, 0, levels)
        out[base.rstrip(".")] = {"w_q": (x_quant - z) * d, "weight": w,
                                 "zero_point": z, "adaround": alpha is not None}
    return out


def match(name, ada):
    """Map OUR conv name to the qdiff checkpoint's, EXPLICITLY.

    The two namings differ by more than a prefix, which is why suffix matching found 0 of 70 on the
    first run (and the script refused a verdict rather than reporting nan, which is the only reason
    that was visible):

        ours                          qdiff checkpoint
        input_blocks.1.0.in_conv      model.input_blocks.1.0.in_layers.2
        input_blocks.1.0.out_conv     model.input_blocks.1.0.out_layers.3
        input_blocks.0.0              model.input_blocks.0.0

    `in_conv`/`out_conv` are FusedResBlock's aliases for `original.in_layers[-1]` and
    `original.out_layers[-1]` (fused_resblock.py:756) -- the same aliasing that caused the
    double-wrapping bug. The trailing index is theirs: in_layers is [GroupNorm, SiLU, Conv] so the conv
    is 2, out_layers is [GroupNorm, SiLU, Dropout, Conv] so it is 3.
    """
    base = name
    if base.endswith(".in_conv"):
        base = base[: -len(".in_conv")] + ".in_layers.2"
    elif base.endswith(".out_conv"):
        base = base[: -len(".out_conv")] + ".out_layers.3"
    for cand in (f"model.{base}", base):
        if cand in ada:
            return cand
    return None


def main():
    import act_fake_quant as A
    H.STEPS, H.BATCH = 8, 2
    H.AUTO_DELTA_TABLE = True
    r, m, s = H.build("fp16", None, "static")
    convs = A.target_convs(m.model.diffusion_model)
    print(f"{len(convs)} target convs", flush=True)

    grabs = {k: Grab() for k in convs}
    hs = [convs[k].register_forward_pre_hook(grabs[k]) for k in convs]
    H.SEED = 1234
    H.latent(r, m, s)
    for h in hs:
        h.remove()
    got = sum(1 for g in grabs.values() if g.x is not None)
    print(f"activations captured for {got}/{len(convs)} convs", flush=True)

    ada = load_adaround(CKPT) if os.path.exists(CKPT) else {}
    print(f"adaround entries in {os.path.basename(CKPT)}: {len(ada)}", flush=True)

    rows, matched = [], 0
    for name, mod in convs.items():
        x = grabs[name].x
        if x is None:
            continue
        w = mod.weight.data.float()
        b = mod.bias.data.float() if mod.bias is not None else None
        st = (mod.stride, mod.padding, mod.dilation, mod.groups)

        def out(wq):
            return F.conv2d(x, wq, b, st[0], st[1], st[2], st[3])

        ref = out(w)
        nrm = float(ref.norm()) + 1e-12
        row = {"layer": name,
               "ours": float((out(q_ours(w)) - ref).norm()) / nrm,
               "rtn_sym": float((out(q_rtn_sym(w)) - ref).norm()) / nrm,
               "w_recon_ours": float((q_ours(w) - w).norm() / (w.norm() + 1e-12))}
        key = match(name, ada)
        if key is not None:
            e = ada[key]
            if tuple(e["weight"].shape) == tuple(w.shape):
                wq = e["w_q"].to(w.device)
                # THE PAPER'S WEIGHTS ARE NOT THIS MODEL'S WEIGHTS unless the checkpoint matches the
                # one we loaded, so score AdaRound against ITS OWN fp32 weight and ours against ours.
                # Scoring both against our fp32 weight would charge AdaRound for a checkpoint
                # difference and call it quantization error.
                w_ada = e["weight"].to(w.device)
                ref_ada = F.conv2d(x, w_ada, b, st[0], st[1], st[2], st[3])
                nrm_ada = float(ref_ada.norm()) + 1e-12
                row["adaround"] = float((out(wq) - ref_ada).norm()) / nrm_ada
                # ADDED 2026-08-17. `adaround` above is AdaRound WITH its per-channel zero point, which
                # needs fix #4's windowed reduction. What SHIPS today is AdaRound re-quantized onto our
                # symmetric grid: generate_fid_samples.install_adaround() substitutes the dequantized
                # W_q into the state dict and the int4 conversion re-rounds it, dropping z_w. Scoring
                # that against the SAME ref_ada isolates the zero point -- the only thing fix #4 adds
                # over the arm already measured at FID 181.5 -> 140.2 -- from AdaRound's rounding, which
                # is already available with no kernel work. The ratio of these two IS fix #4's prize.
                row["adaround_on_our_grid"] = float((out(q_ours(wq)) - ref_ada).norm()) / nrm_ada
                row["w_recon_adaround"] = float((wq - w_ada).norm() / (w_ada.norm() + 1e-12))
                row["fp32_weight_delta"] = float((w_ada - w).norm() / (w.norm() + 1e-12))
                row["zp_min"] = float(e["zero_point"].min())
                row["zp_max"] = float(e["zero_point"].max())
                row["is_adaround"] = bool(e["adaround"])
                matched += 1
        rows.append(row)

    print(f"adaround matched to {matched}/{len(rows)} convs", flush=True)
    if matched == 0:
        print("REFUSING A VERDICT: 0 convs matched the AdaRound checkpoint, so the comparison this "
              "script exists for did not happen. The 'ours' column below is still valid on its own.")

    def med(key):
        v = [r_[key] for r_ in rows if key in r_]
        return (statistics.median(v), max(v), len(v)) if v else (float("nan"), float("nan"), 0)

    print(f"\n{'weight set':14}{'median out err':>16}{'worst':>10}{'n':>5}")
    summary = {}
    for key in ("rtn_sym", "ours", "adaround", "adaround_on_our_grid"):
        mm, wo, n = med(key)
        summary[key] = {"median": mm, "worst": wo, "n": n}
        print(f"{key:14}{mm:16.4f}{wo:10.4f}{n:5d}")
    print(f"\n{'weight set':14}{'median ||W-Wq||':>17}   (the metric fix #4 was deprioritised on)")
    for key in ("w_recon_ours", "w_recon_adaround"):
        mm, wo, n = med(key)
        summary[key] = {"median": mm, "worst": wo, "n": n}
        print(f"{key:14}{mm:17.4f}")

    zmin = [r_["zp_min"] for r_ in rows if "zp_min" in r_]
    zmax = [r_["zp_max"] for r_ in rows if "zp_max" in r_]
    if zmin:
        print(f"\nAdaRound per-channel weight zero point spans [{min(zmin):.0f}, {max(zmax):.0f}] "
              f"-- not centred, so it cannot be waved away")

    json.dump({"max_n": MAX_N, "matched": matched, "summary": summary, "rows": rows},
              open(f"{D}/data/weight_zp_output_error.json", "w"), indent=1)
    print(f"wrote {D}/data/weight_zp_output_error.json")

    # ---- SANITY GATE ON THE INSTRUMENT, before any verdict ---------------------------------------
    # A 4-bit weight cannot reconstruct worse than zeroing the weight, which is 1.0. If it does, the
    # dequantization is wrong -- which is exactly what happened here on the first run (3.27, reported
    # by this very script as "AdaRound loses by 152x"). Refuse rather than conclude.
    bad = [r_ for r_ in rows if r_.get("w_recon_adaround", 0) > 1.0]
    if bad:
        print(f"\nREFUSING A VERDICT: {len(bad)} of {matched} convs have AdaRound weight-reconstruction "
              f"error > 1.0, i.e. worse than zeroing the weights (worst "
              f"{max(r_['w_recon_adaround'] for r_ in bad):.2f}). That is a dequantization bug, not a "
              f"property of AdaRound. Fix the quantizer application before reading the table above.")
        return 1
    drift = [r_["fp32_weight_delta"] for r_ in rows if "fp32_weight_delta" in r_]
    if drift:
        print(f"\nfp32 weight difference between the two checkpoints: median "
              f"{statistics.median(drift):.4f}   (each arm is scored against its OWN fp32 weight, so "
              f"this does not enter the comparison)")

    if matched:
        o, a = summary["ours"]["median"], summary["adaround"]["median"]
        wins = sum(1 for r_ in rows if "adaround" in r_ and r_["ours"] < r_["adaround"])
        print(f"\nours wins on OUTPUT error for {wins}/{matched} convs; medians "
              f"{o:.4f} (ours) vs {a:.4f} (adaround), ratio {a / o:.2f}x")

        # ADDED 2026-08-17: fix #4's prize, isolated. Both arms start from AdaRound's W_q and are scored
        # against AdaRound's own fp32 weight, so the ONLY difference is whether z_w survives.
        g = summary.get("adaround_on_our_grid", {}).get("median")
        if g:
            zp_wins = sum(1 for r_ in rows
                          if "adaround" in r_ and r_["adaround"] < r_["adaround_on_our_grid"])
            print(f"\nFIX #4 ISOLATED -- AdaRound with z_w vs the same weights on our symmetric grid\n"
                  f"  with z_w (needs fix #4)   median {a:.4f}\n"
                  f"  on our grid (SHIPS TODAY) median {g:.4f}\n"
                  f"  ratio {g / a:.2f}x, and the zero point wins on {zp_wins}/{matched} convs.\n"
                  f"This is what a windowed reduction for z_w*Sigma(a) buys ON TOP OF the AdaRound arm\n"
                  f"already measured end to end -- not on top of the shipped RTN+MSE weights.")
        if o <= a:
            print("\nFIX #4 IS CLOSED ON EVIDENCE. AdaRound loses on OUTPUT error too -- the metric it\n"
                  "optimises and the one the earlier deprioritisation could not measure. Adopting it\n"
                  "would need a windowed per-output-pixel reduction for z_w*Sigma(a), a kernel that does\n"
                  "not exist, to buy something measurably worse than what ships.")
        else:
            print(f"\nFIX #4 IS BACK ON THE TABLE: AdaRound is {o / a:.2f}x better on output error. The\n"
                  f"cost is a windowed reduction for z_w*Sigma(a) -- the SAME per-output-pixel\n"
                  f"capability fix #2's padding defect needed, so the two should be priced together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
