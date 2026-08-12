"""ATTEMPTED and DID NOT WORK -- kept as a refutation so the next person does not repeat it.

GOAL: measure what the W4A4 convs actually see, to settle why the Q-Diffusion scales lose.

OUTCOME: 0 of 70 convs ever receive a floating-point input, with BOTH techniques:
  register_forward_pre_hook  -> 0/70. FusedResBlock calls conv methods directly, never __call__.
  patching the 10 dispatch methods (the layer harness's own technique) -> also 0/70.

The reason is structural: in W4A4 the quantize is FUSED INTO THE PROLOGUE. _prequant_gn_conv and
the Upsample fusion quantize the GroupNorm+SiLU output and hand the conv packed int4 via
forward_from_int4, so the float activation never touches OptimizedInt4Conv2d at all. int4's
effective_code_utilisation (int4_optimized.py:287) is therefore unreachable on this path.

To actually measure it, instrument fused_resblock.py's _prequant_gn_conv -- that is where the float
tensor is, and it is a module-level function, so it can be wrapped the way the layer harness wraps
_prequant_gn_resize_conv. Not done here.

Original docstring follows.

Stop inferring: measure what the W4A4 convs ACTUALLY see, and compare each scale file to it.

Three explanations for the W4A4 bridge failure have now been offered and refuted:

  1. "the quantized weights differ"      -> --w_sym matched them; relL2 1.1667 -> 1.2200. No.
  2. "the wrong statistic (absmax)"      -> the clip-search variant also lost. No.
  3. "the calibration trajectory is fp16, and W4A4's activations are larger"
                                         -> the W4A4 LATENTS are 0.51-0.95x fp16's, i.e. SMALLER.

Explanation 3 compared latents, which is not what the quantizer sees. The conv inputs sit after
GroupNorm+SiLU and are on a completely different scale. So this measures the thing directly:
per-layer max|x| at the 70 int4 conv inputs over a real sampling run, then asks which scale file's
assumed range (127/static_scale, times the smooth factor where smoothing is live) actually matches.

A scale file is only right if its assumed range tracks the measured one. Utilisation is
measured_absmax * static_scale * (act_q/127) with act_q=7 -- above 7 means clipping, far below 7
means wasted levels.

Run: python docs/qdiff_bridge_2026-08-12/scripts/int4_actual_ranges.py
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
import dynamic_delta_ab as H                                               # noqa: E402

D = "docs/qdiff_bridge_2026-08-12/data"
FILES = {"shipped": "integration/calibration/int4_calibration_realckpt.pt",
         "shipped_nosmooth": f"{D}/int4_shipped_nosmooth.pt",
         "qdiff_sym": f"{D}/qdiff_w4a4_sym.pt",
         "qdiff_wsym_sym": f"{D}/qdiff_w4a4_wsym_sym.pt",
         "qdiff_mse": f"{D}/qdiff_w4a4_mse.pt"}
OUT = f"{D}/int4_actual_ranges.json"
ACT_Q = 7.0


def main():
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    H.STEPS, H.BATCH, H.SEED = 50, 8, 1234
    r, m, s = H.build("int4_baseline", H.CALIB["int4"], "static")
    unet = m.model.diffusion_model

    convs = {c.layer_name: c for c in unet.modules()
             if isinstance(c, OptimizedInt4Conv2d) and getattr(c, "is_calibrated", False)}
    print(f"  {len(convs)} calibrated int4 convs")

    # METHOD PATCHING, not hooks. FusedResBlock calls conv methods directly
    # (self.in_conv.forward_from_int4(...)), never __call__, so register_forward_pre_hook fires
    # ZERO times -- measured, 0/70. This is the documented pitfall in
    # profile_layers_and_model.py's docstring, which recorded pre-hooks missing 62 of 70 convs.
    # The layer harness's answer is to wrap the real dispatch targets, and int4 already ships the
    # right metric: effective_code_utilisation (int4_optimized.py:287) returns max|x*smooth_inv| in
    # CODE units as the kernel sees it, so it accounts for smoothing and SiLU in the right order.
    UTIL = {k: 0.0 for k in convs}
    patched = []
    for nm in [n for n in dir(OptimizedInt4Conv2d)
               if n == "forward" or n.startswith(("forward_", "_forward"))]:
        fn = getattr(OptimizedInt4Conv2d, nm, None)
        if not callable(fn):
            continue

        def mk(fn, nm):
            def w(self, x, *aa, **kk):
                if torch.is_tensor(x) and x.is_floating_point() and getattr(self, "is_calibrated", False):
                    try:
                        u = self.effective_code_utilisation(
                            x, fused_silu=getattr(self, "fuse_input_silu", False))
                        if u > UTIL.get(self.layer_name, 0.0):
                            UTIL[self.layer_name] = float(u)
                    except Exception:
                        pass
                return fn(self, x, *aa, **kk)
            return w
        setattr(OptimizedInt4Conv2d, nm, mk(fn, nm))
        patched.append((nm, fn))
    print(f"  patched {len(patched)} dispatch methods")

    H.latent(r, m, s)          # discard: attention self-calibration
    for k in UTIL:
        UTIL[k] = 0.0
    H.latent(r, m, s)          # the measured run
    for nm, fn in patched:
        setattr(OptimizedInt4Conv2d, nm, fn)

    # utilisation is measured against the SHIPPED file (that is the model we ran). Convert back to a
    # measured absmax so every other file can be scored against the same physical quantity.
    shipped = torch.load(FILES["shipped"], map_location="cpu", weights_only=False)
    seen = {}
    for k, u in UTIL.items():
        if u <= 0 or k not in shipped:
            continue
        e = shipped[k]
        sc = float(e["static_scale"]) if isinstance(e, dict) else float(e)
        sm = float(e["smooth_scale"].median()) if isinstance(e, dict) else 1.0
        # u = (mx/sm) * sc * act_q/127  ->  mx = u * sm * 127 / (sc * act_q)
        seen[k] = u * sm * 127.0 / (sc * ACT_Q)

    live = {k: v for k, v in seen.items() if v > 0}
    print(f"  {len(live)}/{len(convs)} convs saw a floating-point input")
    if not live:
        print("FAIL: no float inputs observed -- the fused prologues feed codes, as in the int8 case")
        return 1
    print(f"  measured max|x|: median {statistics.median(live.values()):.3f}  "
          f"min {min(live.values()):.3f}  max {max(live.values()):.3f}")

    print(f"\n{'file':20s} {'median assumed':>15} {'median util':>12} {'clipping':>10} {'wasted':>8}")
    rows = {}
    for name, path in FILES.items():
        if not os.path.exists(path):
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        util, assumed = [], []
        for k, mx in live.items():
            if k not in d:
                continue
            e = d[k]
            sc = float(e["static_scale"]) if isinstance(e, dict) else float(e)
            smooth = e.get("smooth_scale") if isinstance(e, dict) else None
            # The kernel quantizes x*smooth_inv, so smoothing shrinks what the scale multiplies.
            eff = mx / float(smooth.median()) if smooth is not None else mx
            util.append(eff * sc * (ACT_Q / 127.0))
            assumed.append(127.0 / sc * (float(smooth.median()) if smooth is not None else 1.0))
        clip = sum(1 for u in util if u > ACT_Q)
        waste = sum(1 for u in util if u < ACT_Q / 4)
        rows[name] = {"median_assumed_range": statistics.median(assumed),
                      "median_utilisation": statistics.median(util),
                      "n_clipping": clip, "n_wasted": waste, "n": len(util)}
        print(f"{name:20s} {statistics.median(assumed):15.3f} {statistics.median(util):12.3f} "
              f"{clip:6d}/{len(util)} {waste:5d}/{len(util)}")

    print(f"\n  Q_b = {ACT_Q:.0f}. utilisation > 7 clips; << 7 wastes levels.")
    print("  measured relL2 (PTQ): shipped 0.7119 | nosmooth 0.4885 | qdiff_sym 1.1945 | "
          "qdiff_mse 1.5203")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"measured_absmax": live, "files": rows, "act_q": ACT_Q}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
