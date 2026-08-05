"""Is the reason for disabling MoDiff on the LINEAR path still true?

`benchmark_ldm.py` hard-disabled MoDiff for qkv/proj with the note "rel-err diverges 0.06 -> 3.2 as
quant error accumulates over DDIM steps". That divergence had one cause: Bug 2 -- wxax_linear.py
passed the already-quantized codes `q` into `_gemm()`, which re-quantized them with 1/d_scale ~ 1e4
and saturated every nonzero delta to +-127. a_hat stayed correct, o_hat was poisoned, and the error
compounded over steps -- exactly the reported signature.

Bug 2 was fixed 2026-08-03. This measures whether the divergence went with it, instead of leaving a
paper-incomplete implementation justified by a stale comment. Per the paper, A(.) in Eqs 8-17 is any
linear operator, so leaving the Linear layers out is an incompleteness, not a design choice.

Latent relL2 vs fp16 at steady state (run 1 discarded -- see FINDINGS 2026-08-04), for
MODIFF_LINEAR=0 and 1, at both bit-widths. A divergence would show up as relL2 >> 1.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import torch

from dynamic_delta_ab import CALIB, build, latent


def main():
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    os.environ["MODIFF_LINEAR"] = "0"
    r, m, s = build("fp16", None, "dynamic")
    latent(r, m, s)
    ref, _ = latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()
    print(f"fp16 reference |x|max {float(ref.abs().max()):.4f}\n", flush=True)

    out = {}
    for bits in ("int8", "int4"):
        print(f"{'=' * 72}\n{bits}  (conv MoDiff dynamic in both rows; only the LINEAR "
              f"path differs)\n{'=' * 72}", flush=True)
        for lin in ("0", "1"):
            os.environ["MODIFF_LINEAR"] = lin
            r, m, s = build(bits, CALIB[bits], "dynamic")
            latent(r, m, s)
            lat, ms = latent(r, m, s)
            rel = float((lat - ref).norm() / ref.norm())
            out[f"{bits}_linear_modiff_{lin}"] = {
                "rel_l2_vs_fp16": rel, "ms_per_step": ms,
                "latent_absmax": float(lat.abs().max())}
            print(f"  MODIFF_LINEAR={lin}   relL2 {rel:.4f}   {ms:7.2f} ms/step   "
                  f"|x|max {float(lat.abs().max()):.4f}", flush=True)
            del m, s, r
            torch.cuda.empty_cache()
        a = out[f"{bits}_linear_modiff_0"]["rel_l2_vs_fp16"]
        b = out[f"{bits}_linear_modiff_1"]["rel_l2_vs_fp16"]
        verdict = ("DIVERGES -- the old comment still holds" if b > 1.0 else
                   "helps" if b < a else "no divergence, but no gain either")
        print(f"\n  {a:.4f} -> {b:.4f}   {verdict}\n", flush=True)
        os.environ["MODIFF_LINEAR"] = "0"

    with open("docs/modiff_correctness_2026-08-03/data/linear_modiff_ab.json", "w") as f:
        json.dump({"results": out}, f, indent=2)
    print("wrote docs/modiff_correctness_2026-08-03/data/linear_modiff_ab.json")


if __name__ == "__main__":
    main()
