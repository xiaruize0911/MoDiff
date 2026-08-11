"""Paired A/B of route (b): the dual-output qkv GEMM feeding flash's int8 gather path.

  arm ON  : MODIFF_FUSE_QKV_I8=1 -- the 10 hd=48 blocks emit int8 from the GEMM epilogue straight
            into flash_attn_int8_packed_vt, and their three aq_* re-quantize passes disappear
  arm OFF : MODIFF_FUSE_QKV_I8=0 -- production: fp16 qkv, quantize_attn_qkv_packed_static, mma flash

PREDICTION, from the kernel microbenchmark and written here before this ran: +0.79 ms/step at batch
128 (integration/tests/bench_flash_packed_vs_unpacked.py -- 10 blocks, aq_* removed worth ~1.9 ms
against ~1.1 ms paid back by the slower gather kernel). The 5 hd=24 blocks are NOT eligible, so this
is not the whole 4.60 ms attn_quantize bucket; see the gate's docstring.

Both arms run on the SAME model object, alternating ON/OFF/ON/OFF so a thermal or clock trend splits
evenly, because session drift here is the same order as the effect: the cross-session comparison that
preceded ab_updown_fusion_refresh.py moved its own control arm by 0.43 ms.

WHY THE COUNTERS. Route (b) is selected per block by a gate that can silently decline (unfrozen
scales, wrong shape, missing symbol) and the arm would then time production twice and report a
believable ~0. Each arm asserts its kernel counts instead of being trusted:

    gemm_w8a8_awq_o_hat_out_i8            ON 10/step   OFF 0/step
    quantize_attn_qkv_packed_static       ON 5/step    OFF 15/step   (only hd=24 keeps it)
    flash_attn_int8_packed_vt             ON 10/step   OFF 0/step

Run: python integration/tests/ab_route_b_qkv_i8.py [--k 4] [--batch 128] [--steps 200]
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

MODIFF_ENV = {
    "MODIFF_QUANT_LINEAR": "1", "MODIFF_QUANT_ATTN": "1", "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1", "MODIFF_FLASH_GATE": "on", "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0", "MODIFF_FUSE_PROJ_QUANT": "1", "MODIFF_LINEAR": "1",
    "MODIFF_ACT_BITS": "8", "MODIFF_DELTA_MODE": "dynamic",
    "MODIFF_WARMUP_STEPS": "5",
}

#: name -> (expected per step with the fusion ON, expected with it OFF)
COUNTED = {
    "gemm_w8a8_awq_o_hat_out_i8": (10, 0),
    "quantize_attn_qkv_packed_static": (5, 15),
    "flash_attn_int8_packed_vt": (10, 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="MODIFF_DELTA_REFRESH")
    ap.add_argument("--proj-k", type=int, default=4, help="MODIFF_LINEAR_DELTA_REFRESH")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=4, help="per arm, alternated")
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    for k, v in MODIFF_ENV.items():
        os.environ[k] = v
    os.environ["MODIFF_DELTA_REFRESH"] = str(args.k)
    os.environ["MODIFF_LINEAR_DELTA_REFRESH"] = str(args.proj_k)
    os.environ["MODIFF_FUSE_QKV_I8"] = "1"

    import integration.benchmarks.benchmark_ldm as B
    from kernel_suites_bench import CALIB
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="docs/component_attribution_2026-08-07/tmp_out",
                          batch_size=args.batch, steps=args.steps, shape=(4, 32, 32),
                          calibration_path=CALIB.get("int8"), linear_backend="fp16")
    model, sampler = r._setup_model("int8")
    cond = r._cond_kwargs(model, args.batch)

    hits = {name: 0 for name in COUNTED}
    for name in COUNTED:
        orig = getattr(mc, name)

        def counting(*a, _n=name, _f=orig, **kw):
            hits[_n] += 1
            return _f(*a, **kw)
        setattr(mc, name, counting)

    def one(on):
        # Read per call by the gate, so flipping it needs no re-setup: the two arms are the same
        # model object differing only in which branch ten blocks take.
        os.environ["MODIFF_FUSE_QKV_I8"] = "1" if on else "0"
        for name in hits:
            hits[name] = 0
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                        dtype=torch.float16):
            sampler.sample(S=args.steps, batch_size=args.batch, shape=r.shape, eta=0.0,
                           verbose=False, **cond)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / args.steps, {n: c / args.steps for n, c in hits.items()}

    for _ in range(args.warmups):
        one(True)
        one(False)

    on_ms, off_ms, on_cnt, off_cnt = [], [], [], []
    for _ in range(args.repeats):
        for on, ms_l, c_l in ((True, on_ms, on_cnt), (False, off_ms, off_cnt)):
            ms, c = one(on)
            ms_l.append(ms)
            c_l.append(c)

    def summarize(v):
        return statistics.median(v), (statistics.stdev(v) / statistics.mean(v) * 100
                                      if len(v) > 1 else 0.0)

    on_med, on_cv = summarize(on_ms)
    off_med, off_cv = summarize(off_ms)
    # Each ON differenced against the OFF measured immediately after it, so a monotone drift cancels
    # rather than landing on whichever arm ran later.
    pairs = [off - on for on, off in zip(on_ms, off_ms)]

    print(f"\nK={args.k}, projK={args.proj_k}, batch {args.batch}, {args.steps} steps, "
          f"{args.repeats} paired repeats, {torch.cuda.get_device_name(0)}\n")
    print("| arm | ms/step (median) | CV% |")
    print("|---|---:|---:|")
    print(f"| ON  (route b) | {on_med:.2f} | {on_cv:.3f} |")
    print(f"| OFF (today)   | {off_med:.2f} | {off_cv:.3f} |")
    print(f"\npaired deltas (OFF - ON), ms/step: {', '.join(f'{p:+.2f}' for p in pairs)}")
    print(f"median {statistics.median(pairs):+.2f} ms/step recovered by route (b)  "
          f"(prediction was +0.79)")
    if len(pairs) > 1:
        print(f"stdev of the paired delta: {statistics.stdev(pairs):.3f} ms")

    print("\n| kernel | ON /step | expected | OFF /step | expected |")
    print("|---|---:|---:|---:|---:|")
    # The ON arm cannot fire before the attention scales freeze (_fq_frozen2, MODIFF_ATTN_CALIB_STEPS
    # forwards), so with --warmups 0 the first timed run is short by that window: a 20-step smoke run
    # measured 8.50/step against 10, i.e. exactly 17 of 20 steps fused. With one warm-up or more the
    # freeze has already happened and the counts are exact, which is why the tolerance stays tight
    # rather than being widened to accommodate a configuration nobody should trust for timing.
    tol = 0.01 if args.warmups > 0 else max(exp for exp, _ in COUNTED.values()) * 6.0 / args.steps
    if args.warmups == 0:
        print(f"\n(--warmups 0: counts are expected to run short by the calibration window; "
              f"tolerance widened to {tol:.2f}/step and the TIMING IS NOT USABLE -- the ON arm "
              f"absorbs every one-time allocation)")
    warn = []
    for name, (exp_on, exp_off) in COUNTED.items():
        got_on = statistics.median(c[name] for c in on_cnt)
        got_off = statistics.median(c[name] for c in off_cnt)
        print(f"| {name} | {got_on:.2f} | {exp_on} | {got_off:.2f} | {exp_off} |")
        if got_on > exp_on + 0.01 or got_on < exp_on - tol:
            warn.append(f"{name}: ON arm ran it {got_on:.2f}/step, expected {exp_on}")
        if abs(got_off - exp_off) > 0.01:
            warn.append(f"{name}: OFF arm ran it {got_off:.2f}/step, expected {exp_off}")
    if warn:
        print("\nWARNING: the arms are not the arms they claim to be -- the delta above is not "
              "measuring route (b):")
        for w in warn:
            print("  -", w)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(0), batch=args.batch, steps=args.steps,
                           k=args.k, proj_k=args.proj_k, repeats=args.repeats,
                           on_ms=on_ms, off_ms=off_ms, pairs=pairs,
                           on_median=on_med, off_median=off_med,
                           paired_median=statistics.median(pairs),
                           prediction_ms=0.79,
                           counts_on={n: statistics.median(c[n] for c in on_cnt) for n in COUNTED},
                           counts_off={n: statistics.median(c[n] for c in off_cnt) for n in COUNTED},
                           warnings=warn), f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
