"""Where does W4A4's error come from? Localise it before trying to fix it.

Why this matters for the speed goal. MoDiff W4A4 already runs at 62.3 ms/step against
int8_baseline's 71.0 -- it is **1.14x FASTER than the W8A8 baseline**. It is just less accurate
(relL2 0.4979 vs 0.2376). If W4A4+MoDiff reached 0.2376 it would deliver W8A8-baseline quality at
1.14x the speed, which is a real speedup *versus baseline* and is exactly the paper's thesis: MoDiff
lets you drop activation bits, and the speed comes from the lower bit-width.

So the question is not "can MoDiff be faster than its own baseline" (it cannot -- more bytes by
construction) but "can MoDiff W4A4 reach W8A8-baseline quality". That is an accuracy question, and
the first step is to find out which quantized component is responsible for the 0.4979.

MoDiff currently covers the CONV path only. Attention and the Linear layers are quantized to W4A4
with a static scale and no modulation at all, so they are the prime suspects. This measures each by
turning them off (leaving them fp16) one at a time:

  all quantized          the shipped int4 mode
  attention fp16         MODIFF_QUANT_ATTN=0
  linear fp16            MODIFF_QUANT_LINEAR=0
  linear + MoDiff        MODIFF_LINEAR=1 (the delta path, now that Bug 2 is fixed)
  attn + linear fp16     conv-only quantization -- the part MoDiff actually covers

The last row is the important one: it is the error floor MoDiff's current scope can reach. If it is
already near 0.2376, extending MoDiff to attention/proj is worth doing. If it is not, the conv path
itself is the limit and W4A4 cannot get there.
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

# (label, env overrides applied AFTER ks.set_env inside build())
CASES = [
    ("int4 all quantized (shipped)",      {}),
    ("int4, attention fp16",              {"MODIFF_QUANT_ATTN": "0"}),
    ("int4, linear fp16",                 {"MODIFF_QUANT_LINEAR": "0"}),
    ("int4, linear + MoDiff",             {"MODIFF_LINEAR": "1"}),
    ("int4, attn+linear fp16 (conv only)", {"MODIFF_QUANT_ATTN": "0",
                                            "MODIFF_QUANT_LINEAR": "0"}),
]
INT8_BASELINE_TARGET = None      # measured in-process below


def measure(bits, overrides, ref):
    # build() calls kernel_suites_bench.set_env, which REWRITES every MODIFF_QUANT_* key from its own
    # QUANT_ENV table. Setting them beforehand is therefore silently undone -- the first version of
    # this script did exactly that and produced five identical rows. Wrap set_env so the overrides
    # are re-applied immediately after it, before _setup_model reads them.
    import kernel_suites_bench as ks
    base_set_env = getattr(ks, "_orig_set_env", ks.set_env)
    ks._orig_set_env = base_set_env

    def patched(mode):
        base_set_env(mode)
        for k, v in overrides.items():
            os.environ[k] = v
    ks.set_env = patched
    try:
        r, m, s = build(bits, CALIB[bits], "dynamic")
    finally:
        ks.set_env = base_set_env
    latent(r, m, s)                       # warm-up
    lat, ms = latent(r, m, s)
    rel = float((lat - ref).norm() / ref.norm())
    del m, s, r
    torch.cuda.empty_cache()
    for k in overrides:
        os.environ.pop(k, None)
    return rel, ms


def main():
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    os.environ["MODIFF_DELTA_REPORT"] = "0"   # free reporting diverges at W4A4; see FINDINGS
    r, m, s = build("fp16", None, "dynamic")
    latent(r, m, s)
    ref, _ = latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()

    # The bar: W8A8 baseline, same process and discipline.
    r, m, s = build("int8_baseline", CALIB["int8"], "static")
    latent(r, m, s)
    lat, ms8 = latent(r, m, s)
    target = float((lat - ref).norm() / ref.norm())
    del m, s, r
    torch.cuda.empty_cache()
    print(f"  BAR: int8_baseline relL2 {target:.4f} at {ms8:.2f} ms/step\n", flush=True)

    out = {"int8_baseline": {"rel_l2_vs_fp16": target, "ms_per_step": ms8}}
    for label, ov in CASES:
        rel, ms = measure("int4", ov, ref)
        out[label] = {"rel_l2_vs_fp16": rel, "ms_per_step": ms, "env": ov}
        print(f"  {label:38s} relL2 {rel:.4f}   {ms:6.2f} ms/step   "
              f"{'BEATS the W8A8 bar' if rel < target else f'{rel/target:.2f}x the bar'}",
              flush=True)

    full = out["int4 all quantized (shipped)"]["rel_l2_vs_fp16"]
    print(f"\n{'=' * 78}\nError attribution (relL2 removed by leaving a component in fp16)\n{'=' * 78}")
    for label, _ in CASES[1:]:
        d = full - out[label]["rel_l2_vs_fp16"]
        print(f"  {label:38s} {d:+.4f}   ({d / full * 100:+.0f}% of the total)")

    with open("docs/modiff_correctness_2026-08-03/data/int4_error_attribution.json", "w") as f:
        json.dump({"bar_int8_baseline": target, "results": out}, f, indent=2)
    print("\nwrote docs/modiff_correctness_2026-08-03/data/int4_error_attribution.json")


if __name__ == "__main__":
    main()
