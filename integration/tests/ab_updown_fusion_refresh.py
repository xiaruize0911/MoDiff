"""Paired A/B of the updown refresh fusion, both arms in ONE process.

The cross-session comparison against docs/component_attribution_2026-08-07 put the fix at
-1.05 ms/step on modiff_full_k1 -- but the int8_ptq control, which contains no MoDiff resize path
at all, moved -0.43 ms over the same pair of runs. Session drift is the same order as the effect,
so subtracting one from the other is not a measurement.

This runs both arms on the SAME model object, alternating A/B/A/B so any thermal or clock trend
splits evenly between them, and reports the paired difference.

  arm ON  : MODIFF_UPDOWN_FUSE_REFRESH=1 (the fix) -- 8/8 updown blocks fused every step
  arm OFF : MODIFF_UPDOWN_FUSE_REFRESH=0 (before)  -- fused only on non-refresh steps, so 0/8 at K=1

The flag is read per call, so flipping it needs no re-setup: the two arms are byte-identical
models differing only in which branch eight ResBlocks take.

Run: python integration/tests/ab_updown_fusion_refresh.py [--k 1] [--batch 128] [--steps 200]
"""
import argparse
import os
import statistics
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "integration/benchmarks/report"))

import torch
import modiff_cutlass as mc

MODIFF_ENV = {
    "MODIFF_QUANT_LINEAR": "1", "MODIFF_QUANT_ATTN": "1", "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1", "MODIFF_FLASH_GATE": "on", "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0", "MODIFF_FUSE_PROJ_QUANT": "1", "MODIFF_LINEAR": "1",
    "MODIFF_ACT_BITS": "8", "MODIFF_DELTA_MODE": "dynamic",
    "MODIFF_WARMUP_STEPS": "5",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1, help="MODIFF_DELTA_REFRESH")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=4, help="per arm, alternated")
    ap.add_argument("--warmups", type=int, default=2)
    args = ap.parse_args()

    for k, v in MODIFF_ENV.items():
        os.environ[k] = v
    os.environ["MODIFF_DELTA_REFRESH"] = str(args.k)
    os.environ["MODIFF_UPDOWN_FUSE_REFRESH"] = "1"

    import integration.benchmarks.benchmark_ldm as B
    from kernel_suites_bench import CALIB
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="docs/component_attribution_2026-08-07/tmp_out",
                          batch_size=args.batch, steps=args.steps, shape=(4, 32, 32),
                          calibration_path=CALIB.get("int8"), linear_backend="fp16")
    model, sampler = r._setup_model("int8")
    cond = r._cond_kwargs(model, args.batch)

    # Count the fused entry so each arm proves it is the arm it claims to be, rather than being
    # trusted to be. This is the guard the cross-session comparison could not have.
    fused_name = "group_norm_silu_delta_quantize_resize_nhwc"
    orig_fused = getattr(mc, fused_name)
    hits = {"n": 0}

    def counting(*a, **kw):
        hits["n"] += 1
        return orig_fused(*a, **kw)
    setattr(mc, fused_name, counting)

    def one(on):
        os.environ["MODIFF_UPDOWN_FUSE_REFRESH"] = "1" if on else "0"
        hits["n"] = 0
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                        dtype=torch.float16):
            sampler.sample(S=args.steps, batch_size=args.batch, shape=r.shape, eta=0.0,
                           verbose=False, **cond)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / args.steps, hits["n"] / args.steps

    for _ in range(args.warmups):
        one(True)
        one(False)

    on_ms, off_ms, on_fused, off_fused = [], [], [], []
    for _ in range(args.repeats):
        for on, ms_l, f_l in ((True, on_ms, on_fused), (False, off_ms, off_fused)):
            ms, f = one(on)
            ms_l.append(ms)
            f_l.append(f)

    def summarize(v):
        return statistics.median(v), (statistics.stdev(v) / statistics.mean(v) * 100
                                      if len(v) > 1 else 0.0)

    on_med, on_cv = summarize(on_ms)
    off_med, off_cv = summarize(off_ms)
    # Paired: each ON is differenced against the OFF measured immediately after it, so a monotone
    # drift over the run cancels rather than landing on whichever arm ran later.
    pairs = [off - on for on, off in zip(on_ms, off_ms)]

    print(f"\nK={args.k}, batch {args.batch}, {args.steps} steps, {args.repeats} paired repeats, "
          f"{torch.cuda.get_device_name(0)}\n")
    print("| arm | ms/step (median) | CV% | updown fused/step |")
    print("|---|---:|---:|---:|")
    print(f"| ON  (fix)    | {on_med:.2f} | {on_cv:.3f} | {statistics.median(on_fused):.2f} |")
    print(f"| OFF (before) | {off_med:.2f} | {off_cv:.3f} | {statistics.median(off_fused):.2f} |")
    print(f"\npaired deltas (OFF - ON), ms/step: "
          f"{', '.join(f'{p:+.2f}' for p in pairs)}")
    print(f"median {statistics.median(pairs):+.2f} ms/step recovered by the fix")
    if len(pairs) > 1:
        print(f"stdev of the paired delta: {statistics.stdev(pairs):.3f} ms")
    if statistics.median(on_fused) < 7.99:
        print("\nWARNING: the ON arm did not fuse 8/8 -- the comparison is not what it claims")
    if statistics.median(off_fused) > 0.01 and args.k == 1:
        print("\nWARNING: the OFF arm fused something at K=1 -- the flag did not take effect")


if __name__ == "__main__":
    main()
