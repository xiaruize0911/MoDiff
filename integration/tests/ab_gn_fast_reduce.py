"""Paired A/B of the GN fast-reduce swap in fused_resblock.py.

  arm ON  : MODIFF_GN_FAST=1 -- the ResBlock's GN+quantize takes `..._fast` (fast_reduce=true:
            128-512 threads, pair-major pass 1)
  arm OFF : MODIFF_GN_FAST=0 -- what this file did until 2026-08-16: the plain entry point, whose
            generic heuristic launches up to 1024 threads

PREDICTION, from the kernel microbenchmark and written here before this ran
(docs/gn_fast_reduce_2026-08-16/data/gn_fast_mod.json, every captured ResBlock GN shape at its real
call count, production modulation):

    int8_baseline   71.23 -> 64.32 ms/step   saves 6.907   (1.446x -> 1.601x vs fp16)
    int4_baseline   57.85 -> 50.29 ms/step   saves 7.562   (1.780x -> 2.048x vs fp16)

Both arms run on the SAME model object, alternating ON/OFF/ON/OFF so a thermal or clock trend splits
evenly, and each ON is differenced against the OFF measured immediately after it. That matters more
here than the effect size might suggest: arm ORDER alone moves W4A4 by up to 28%
(docs/zp_coverage_2026-08-13/FINDINGS_NOISE_FLOOR.md), which is why this is a paired in-process A/B and
not two runs of the e2e bench.

WHY THE COUNTERS. _gnq() declines silently if the `_fast` symbol is missing, and the arm would then
time the same code twice and report a believable ~0. So both entry points are counted and two
invariants asserted per arm:

    ON  : plain == 0                    -- every ResBlock call took the fast path
    OFF : plain == <the captured count> -- and none of them did

The `_fast` count is NOT expected to be zero in the OFF arm: the attention paths have always called it
via getattr(_mc, "..._fast", <plain>) and this flag does not reach them. What must hold is that
plain + fast is CONSERVED across arms -- the swap moves calls between two entry points, it does not
add or remove any.

Run: python integration/tests/ab_gn_fast_reduce.py [--mode int8_baseline] [--steps 200]
Writes docs/gn_fast_reduce_2026-08-16/data/ab_<mode>.json
"""
import argparse
import json
import os
import statistics
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "integration/benchmarks/report"))

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"

#: mode -> (plain entry point the ResBlock uses, its _fast sibling)
ENTRIES = {
    "int8_baseline": "group_norm_silu_quantize_nhwc",
    "int4_baseline": "group_norm_silu_quantize_pack_nhwc",
}

#: ms/step of record for each arm, from REPORT.md section 1, so the delta can be put in context
BASELINE_MS = {"int8_baseline": 71.23, "int4_baseline": 57.85, "fp16": 103.00}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8_baseline", choices=sorted(ENTRIES))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="per arm, alternated")
    ap.add_argument("--warmups", type=int, default=2)
    args = ap.parse_args()

    plain = ENTRIES[args.mode]
    fast = plain + "_fast"
    if not hasattr(mc, fast):
        print(f"FAIL: {fast} is not exported -- nothing to A/B")
        return 1

    #: expected plain calls/step in the OFF arm, derived from the 2026-08-13 capture rather than
    #: hardcoded, so a change in the model or the capture window shows up as a count mismatch.
    ks = json.load(open("docs/bench_report_2026-08-13_postzp/data/kernel_suites.json"))
    cap_steps = ks["capture_steps"]
    exp_plain_off = sum(r["calls_per_sample"] for r in ks["modes"][args.mode]["norm_quantize"]
                        if r["entry"] == plain) / cap_steps
    exp_fast_any = sum(r["calls_per_sample"] for r in ks["modes"][args.mode]["norm_quantize"]
                       if r["entry"] == fast) / cap_steps
    print(f"from the capture: {plain} {exp_plain_off:.0f}/step, {fast} {exp_fast_any:.0f}/step "
          f"(attention, not reached by this flag)")

    import integration.benchmarks.benchmark_ldm as B
    from kernel_suites_bench import CALIB, set_env
    set_env(args.mode)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir=f"{D}/tmp_out",
                          batch_size=args.batch, steps=args.steps, shape=(4, 32, 32),
                          calibration_path=CALIB.get(args.mode), linear_backend="fp16")
    model, sampler = r._setup_model(args.mode)
    cond = r._cond_kwargs(model, args.batch)

    hits = {plain: 0, fast: 0}
    for name in list(hits):
        orig = getattr(mc, name)

        def counting(*a, _n=name, _f=orig, **kw):
            hits[_n] += 1
            return _f(*a, **kw)
        setattr(mc, name, counting)

    def one(on):
        # _gnq reads the env per call, so flipping arms needs no re-setup: same model object, same
        # weights, same calibration -- only which of two entry points 62 ResBlock calls take.
        os.environ["MODIFF_GN_FAST"] = "1" if on else "0"
        for k in hits:
            hits[k] = 0
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                        dtype=torch.float16):
            sampler.sample(S=args.steps, batch_size=args.batch, shape=r.shape, eta=0.0,
                           verbose=False, **cond)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / args.steps, {k: v / args.steps for k, v in hits.items()}

    for _ in range(args.warmups):
        one(True)
        one(False)

    on_ms, off_ms, on_c, off_c = [], [], [], []
    for _ in range(args.repeats):
        for on, ml, cl in ((True, on_ms, on_c), (False, off_ms, off_c)):
            ms, c = one(on)
            ml.append(ms)
            cl.append(c)

    def summ(v):
        return statistics.median(v), (statistics.stdev(v) / statistics.mean(v) * 100
                                      if len(v) > 1 else 0.0)

    on_med, on_cv = summ(on_ms)
    off_med, off_cv = summ(off_ms)
    pairs = [off - on for on, off in zip(on_ms, off_ms)]
    delta = statistics.median(pairs)

    print(f"\n{args.mode}, batch {args.batch}, {args.steps} steps, {args.repeats} paired repeats, "
          f"{torch.cuda.get_device_name(0)}\n")
    print("| arm | ms/step (median) | CV% |")
    print("|---|---:|---:|")
    print(f"| ON  (fast_reduce) | {on_med:.2f} | {on_cv:.3f} |")
    print(f"| OFF (until today) | {off_med:.2f} | {off_cv:.3f} |")
    print(f"\npaired deltas (OFF - ON), ms/step: {', '.join(f'{p:+.3f}' for p in pairs)}")
    pred = {"int8_baseline": 6.907, "int4_baseline": 7.562}[args.mode]
    print(f"median {delta:+.3f} ms/step recovered   (prediction was {pred:+.3f})")
    if len(pairs) > 1:
        print(f"stdev of the paired delta: {statistics.stdev(pairs):.3f} ms")
    print(f"\nvs fp16: {BASELINE_MS['fp16'] / off_med:.3f}x -> {BASELINE_MS['fp16'] / on_med:.3f}x")

    ok = True
    print(f"\n| entry point | ON /step | OFF /step | expected |")
    print("|---|---:|---:|---|")
    on_p, off_p = on_c[-1][plain], off_c[-1][plain]
    on_f, off_f = on_c[-1][fast], off_c[-1][fast]
    print(f"| `{plain}` | {on_p:.1f} | {off_p:.1f} | ON 0, OFF {exp_plain_off:.0f} |")
    print(f"| `{fast}` | {on_f:.1f} | {off_f:.1f} | attention keeps {exp_fast_any:.0f} in both |")
    if on_p > 0.01:
        print(f"FAIL: the ON arm still made {on_p:.1f} plain calls/step -- _gnq declined")
        ok = False
    if abs(off_p - exp_plain_off) > 0.05:
        print(f"FAIL: OFF arm made {off_p:.1f} plain calls/step, capture says {exp_plain_off:.0f}")
        ok = False
    if abs((on_p + on_f) - (off_p + off_f)) > 0.05:
        print(f"FAIL: total GN calls not conserved: ON {on_p + on_f:.1f} vs OFF {off_p + off_f:.1f} "
              f"-- the swap moved work rather than relocating it")
        ok = False
    print("\nALL COUNTS PASS" if ok else "\nCOUNTS FAILED -- the timing above is not what it claims")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"mode": args.mode, "batch": args.batch, "steps": args.steps,
               "gpu": torch.cuda.get_device_name(0), "on_ms": on_ms, "off_ms": off_ms,
               "on_median": on_med, "off_median": off_med, "paired_deltas": pairs,
               "delta_median": delta, "prediction": pred, "counts_pass": ok,
               "counts": {"on": on_c[-1], "off": off_c[-1],
                          "expected_plain_off": exp_plain_off, "expected_fast_any": exp_fast_any}},
              open(f"{D}/data/ab_{args.mode}.json", "w"), indent=1)
    print(f"wrote {D}/data/ab_{args.mode}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
