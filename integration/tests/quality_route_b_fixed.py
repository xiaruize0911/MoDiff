"""Route (b)'s quality, measured in a configuration where route (b) actually runs.

WHY A SECOND FILE. `quality_route_b_paired.py` reports BIT-IDENTICAL at 3 and at 8 seeds, and it is
VACUOUS: route (b) never fires in the arm it builds. Counted directly
(`integration/tests/probe_route_b_fires.py`): `gemm_w8a8_awq_o_hat_out_i8` 0 calls in BOTH arms, and so
is every packed-qkv kernel. What that harness does run is the `_qout` family --
`flash_attn_int8_qi8_kv_static_qout[_hd24]` fed by `gemm_w8a8_awq_qkv_i8_layouts[_compact]` -- i.e. the
PTQ attention path. `_qout` is mutually exclusive with MoDiff's fp16 o_hat state by construction (all 21
blocks report `qout_eligible == 0`), so in that arm route (b) has no o_hat to advance and nothing to do.

THE DIFFERENCE IS ONE ENV VAR. `dynamic_delta_ab.build()` goes through `kernel_suites_bench.set_env`,
which does not set `MODIFF_LINEAR=1`. The TIMING A/B (`ab_route_b_qkv_i8.py`) sets its own MODIFF_ENV
with `MODIFF_LINEAR=1`, which is what puts the qkv projection on the MoDiff o_hat path -- and its
counters confirm 10 route-(b) calls/step. So the +0.79 ms/step is real and the quality number never
existed.

This file uses the TIMING A/B's env, so the two arms differ in the same way the timing arms did, and it
asserts the counts rather than trusting them. OPEN_ITEMS B6 was filed as "needs more seeds"; the actual
state is "never measured".

Run: python integration/tests/quality_route_b_fixed.py [--seeds 8]
Writes docs/gn_fast_reduce_2026-08-16/data/route_b_quality_fixed.json
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))
sys.path.insert(0, os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"))

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"

#: verbatim from ab_route_b_qkv_i8.py, the harness whose counters prove route (b) fires
MODIFF_ENV = {
    "MODIFF_QUANT_LINEAR": "1", "MODIFF_QUANT_ATTN": "1", "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1", "MODIFF_FLASH_GATE": "on", "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0", "MODIFF_FUSE_PROJ_QUANT": "1", "MODIFF_LINEAR": "1",
    "MODIFF_ACT_BITS": "8", "MODIFF_DELTA_MODE": "dynamic", "MODIFF_WARMUP_STEPS": "5",
    "MODIFF_DELTA_REFRESH": "4", "MODIFF_LINEAR_DELTA_REFRESH": "4",
}

COUNTED = ["gemm_w8a8_awq_o_hat_out_i8", "flash_attn_int8_packed_vt"]


def _apply_env():
    for k, v in MODIFF_ENV.items():
        os.environ[k] = v


def fp16_refs(seeds):
    r, m, s = H.build("fp16", None, "dynamic")
    H.latent(r, m, s)
    refs = {}
    for seed in seeds:
        H.SEED = seed
        refs[seed], _ = H.latent(r, m, s)
    del r, m, s
    torch.cuda.empty_cache()
    return refs


def arm(route_b, seeds, refs):
    _apply_env()
    os.environ["MODIFF_FUSE_QKV_I8"] = "1" if route_b else "0"
    r, m, s = H.build("int8", H.CALIB["int8"], "dynamic")
    _apply_env()                       # build() calls set_env(), which overwrites some of these
    os.environ["MODIFF_FUSE_QKV_I8"] = "1" if route_b else "0"
    H.latent(r, m, s)                                    # discard: not steady state
    hits = {n: 0 for n in COUNTED if hasattr(mc, n)}
    origs = {}
    for n in list(hits):
        origs[n] = getattr(mc, n)

        def counting(*a, _n=n, _f=origs[n], **kw):
            hits[_n] += 1
            return _f(*a, **kw)
        setattr(mc, n, counting)
    rel = {}
    for seed in seeds:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        rel[seed] = float((lat - refs[seed]).norm() / refs[seed].norm())
    for n, o in origs.items():
        setattr(mc, n, o)
    del r, m, s
    torch.cuda.empty_cache()
    return rel, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = [1234 + i for i in range(args.seeds)]

    refs = fp16_refs(seeds)
    off, off_c = arm(False, seeds, refs)
    on, on_c = arm(True, seeds, refs)

    po = [off[s] for s in seeds]
    pn = [on[s] for s in seeds]
    mo, mn = statistics.mean(po), statistics.mean(pn)
    per = [(y - x) / x * 100.0 for x, y in zip(po, pn)]
    dm = statistics.mean(per)
    sd = statistics.stdev(per) if len(per) > 1 else 0.0
    sem = sd / len(per) ** 0.5 if len(per) > 1 else 0.0
    ident = all(abs(x - y) < 1e-12 for x, y in zip(po, pn))

    print("\n=== route (b): latent relL2 vs fp16, lower is better ===")
    print(f"{'arm':>16} | " + " ".join(f"{s:>8}" for s in seeds) + f" | {'mean':>8}")
    print(f"{'OFF':>16} | " + " ".join(f"{v:>8.4f}" for v in po) + f" | {mo:>8.4f}")
    print(f"{'ON (route b)':>16} | " + " ".join(f"{v:>8.4f}" for v in pn) + f" | {mn:>8.4f}")
    print(f"paired per-seed diff: {dm:+.3f}% +- {sem:.3f}% (SEM), stdev {sd:.3f}%")

    fired = on_c.get("gemm_w8a8_awq_o_hat_out_i8", 0)
    off_fired = off_c.get("gemm_w8a8_awq_o_hat_out_i8", 0)
    print(f"\nnon-vacuity: gemm_w8a8_awq_o_hat_out_i8  OFF {off_fired}  ON {fired}")
    vacuous = fired == 0 or off_fired != 0
    if vacuous:
        print("  FAIL: route (b) did not fire (or fired in both) -- this verdict means NOTHING.")
    else:
        print("  PASS: the arms ran different kernels.")
        if ident:
            print("  -> BIT-IDENTICAL, and now that is a real result: route (b) changes no output.")
        elif abs(dm) <= 2 * sem:
            print(f"  -> NOT RESOLVED: |{dm:+.2f}%| inside 2*SEM ({2 * sem:.2f}%)")
        else:
            print(f"  -> RESOLVED at {len(seeds)} seeds: {dm:+.2f}%")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"seeds": seeds, "per_seed_off": off, "per_seed_on": on, "mean_off": mo,
               "mean_on": mn, "paired_diff_pct_mean": dm, "paired_diff_pct_sem": sem,
               "identical": ident, "counts_off": off_c, "counts_on": on_c,
               "vacuous": bool(vacuous)},
              open(f"{D}/data/route_b_quality_fixed.json", "w"), indent=1)
    print(f"wrote {D}/data/route_b_quality_fixed.json")
    return 1 if vacuous else 0


if __name__ == "__main__":
    sys.exit(main())
