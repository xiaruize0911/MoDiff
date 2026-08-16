"""The attention block's score path vs its projections, from the existing capture. No instrumentation.

OPEN_ITEMS B10 was filed as "the PTQ attn/proj split -- needs `_flash_proj_qout` instrumented to
separate the projection GEMM from the score path". It does not. `_flash_proj_qout` already issues the
two halves as SEPARATE kernel launches with distinct names -- the flash `_qout` kernel and
`gemm_wXaX_awq_bias_res` -- and the suite capture already times both. What was missing was arithmetic,
not instrumentation: deciding which `linear` records are attention projections rather than embedding
linears.

TWO CLASSIFIER TRAPS, both of which produced a wrong table before this file was right:

  1. fp16 passes the activation 3-D as [b, T, c]; the quantized arms flatten it to [b*T, c]. Keying on
     shape[0][0] reads fp16's M as 128 and files every fp16 out-projection as an embedding linear --
     which put fp16's projections at 3.57 ms/step instead of 9.96.
  2. fp16's two largest projections are `fused_gn_qkv`, and in captures before 2026-08-16 that record
     sits in the `other` suite, not `linear` (OPEN_ITEMS A1). Iterating only `linear` drops 6.4 ms/step
     of fp16 projection work. Detected by NAME here, wherever the capture filed it.

Both traps push the same way: they understate fp16 and so overstate the speedup. That is the third time
in one session that suite membership rather than the measurement was the error.

Run: python docs/bench_report_2026-08-16_gnfast/scripts/attn_proj_split.py [path/to/kernel_suites.json]
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

D = "docs/bench_report_2026-08-16_gnfast"
ARMS = [("fp16", "fp16"), ("int8_baseline", "W8A8"), ("int4_baseline", "W4A4")]
#: the token counts this model's attention blocks run at; anything else is an embedding linear
TOK = {1024, 256, 64, 16, 4}


def tok_of(rec):
    """T if this record is an attention projection, else None."""
    a = (rec.get("arg_shapes") or [[]])[0]
    if "gn_qkv" in rec["entry"]:
        return a[2] * a[3] if len(a) == 4 else None          # NHWC image: T = H*W
    if len(a) == 3:
        return a[1] if a[1] in TOK else None                 # fp16 [b, T, c]
    if len(a) == 2 and a[0] % 128 == 0 and a[0] // 128 in TOK:
        return a[0] // 128                                   # quantized [b*T, c]
    return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else f"{D}/data/kernel_suites.json"
    d = json.load(open(src))
    cs = d.get("capture_steps") or 1

    def ms_step(r):
        return r["stats"]["median"] * r["calls_per_sample"] / 1000.0 / cs

    rows = {}
    print(f"{src}\nattention block, ms/step at batch {d.get('batch')}\n")
    print(f"{'arm':7s}{'score':>8}{'proj':>8}{'total':>8}   split     projections by T")
    for key, lab in ARMS:
        score = sum(ms_step(r) for r in (d["modes"][key].get("attention") or []))
        proj = collections.Counter()
        for suite, recs in d["modes"][key].items():
            if not isinstance(recs, list) or suite not in ("linear", "other"):
                continue
            for r in recs:
                t = tok_of(r)
                if t:
                    proj[t] += ms_step(r)
        p = sum(proj.values())
        tot = score + p
        rows[lab] = (score, p, tot)
        print(f"{lab:7s}{score:8.2f}{p:8.2f}{tot:8.2f}   {100 * score / tot:2.0f}%/"
              f"{100 * p / tot:2.0f}%   "
              + ", ".join(f"T={t}:{v:.2f}" for t, v in sorted(proj.items(), reverse=True)))

    print()
    f = rows["fp16"]
    out = {"source": src, "rows": rows, "vs_fp16": {}}
    for lab in ("W8A8", "W4A4"):
        q = rows[lab]
        out["vs_fp16"][lab] = {"score": f[0] / q[0], "proj": f[1] / q[1], "block": f[2] / q[2]}
        print(f"{lab}: score {f[0] / q[0]:.2f}x, projections {f[1] / q[1]:.2f}x, "
              f"whole block {f[2] / q[2]:.2f}x vs fp16")
    print("\nThe split barely moves with precision (56/44 at fp16 and W8A8, 58/42 at W4A4) and the two "
          "halves gain almost equally. So the attention block is not a fast score path dragged down by "
          "slow projections, or the reverse -- it is uniformly ~1.25x, and any fix has to move both.")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump(out, open(f"{D}/data/attn_proj_split.json", "w"), indent=1)
    print(f"\nwrote {D}/data/attn_proj_split.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
