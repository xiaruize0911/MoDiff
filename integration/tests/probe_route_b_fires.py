"""Does route (b) actually fire inside quality_route_b_paired.py's configuration?

WHY THIS EXISTS. quality_route_b_paired.py reports BIT-IDENTICAL at 8 seeds and asserts nothing about
whether the two arms ran different code. That is the exact signature of a vacuous gate, and this repo
has already shipped three of them (b2525d5). On 2026-08-16 the same failure was found in a fresh
harness: quality_gn_fast_paired.py's first version ran mode "int8", where the flag it was toggling is
inert, and confidently reported BIT-IDENTICAL while measuring nothing.

So before "route (b) is bit-identical" can be read as a green light to make MODIFF_FUSE_QKV_I8 default
(OPEN_ITEMS B6), the alternative has to be ruled out: that the flag is inert in this harness's
configuration and the 8 identical latents mean the same code ran twice.

Counts the three kernels route (b) adds or removes, in the SAME configuration
quality_route_b_paired.py builds (H.build("int8", CALIB, "dynamic")), one latent per arm.

    ON  : gemm_w8a8_awq_o_hat_out_i8 > 0 and flash_attn_int8_packed_vt > 0
    OFF : both 0

Run: python integration/tests/probe_route_b_fires.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))
sys.path.insert(0, os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"))

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

WATCH = ["gemm_w8a8_awq_o_hat_out_i8", "flash_attn_int8_packed_vt",
         "quantize_attn_qkv_packed_static", "quantize_attn_qkv_packed"]


def main():
    hits = {n: 0 for n in WATCH if hasattr(mc, n)}
    missing = [n for n in WATCH if not hasattr(mc, n)]
    if missing:
        print(f"not exported by this build: {missing}")
    for n in list(hits):
        orig = getattr(mc, n)

        def counting(*a, _n=n, _f=orig, **kw):
            hits[_n] += 1
            return _f(*a, **kw)
        setattr(mc, n, counting)

    out = {}
    for on in (False, True):
        os.environ["MODIFF_DELTA_REFRESH"] = "4"
        os.environ["MODIFF_LINEAR_DELTA_REFRESH"] = "4"
        os.environ["MODIFF_FUSE_QKV_I8"] = "1" if on else "0"
        r, m, s = H.build("int8", H.CALIB["int8"], "dynamic")
        H.latent(r, m, s)                                  # discard, as the quality harness does
        for k in hits:
            hits[k] = 0
        H.SEED = 1234
        H.latent(r, m, s)
        out["ON" if on else "OFF"] = dict(hits)
        del r, m, s
        torch.cuda.empty_cache()

    print(f"\n{'kernel':38s}{'OFF':>10}{'ON':>10}")
    for k in sorted(set(out["OFF"]) | set(out["ON"])):
        print(f"{k:38s}{out['OFF'].get(k, 0):>10}{out['ON'].get(k, 0):>10}")

    fired = out["ON"].get("gemm_w8a8_awq_o_hat_out_i8", 0) > 0
    off_clean = out["OFF"].get("gemm_w8a8_awq_o_hat_out_i8", 0) == 0
    print()
    if fired and off_clean:
        print("VERDICT: route (b) FIRES in this configuration. A bit-identical latent is therefore a "
              "real result -- the two arms ran different kernels and produced the same output.")
        return 0
    print("VERDICT: route (b) DOES NOT FIRE here. quality_route_b_paired.py's BIT-IDENTICAL is "
          "VACUOUS -- it timed the same code twice. OPEN_ITEMS B6 cannot be closed on it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
