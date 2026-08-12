"""Pass 1 of a two-pass W4A4 calibration: generate calibration data FROM the quantized model.

THE PROBLEM THIS SOLVES. sample_diffusion_ldm.py's --generate branch runs at :553 and exit()s at
:565, BEFORE `if opt.ptq:` at :568 -- so Q-Diffusion's calibration latents are the FP16 model's
trajectory. At W8A8 that is fine, because the quantized trajectory tracks fp16 (relL2 ~0.11) and the
bridge delivered 2.29x. At W4A4 the quantized trajectory DIVERGES (relL2 0.42-0.79), so those latents
do not describe what the model actually sees: qdiff measured an activation range of 3.6 where the
optimum is near 14.8, and the resulting scales clipped catastrophically (relL2 1.19 against a 0.4885
control).

Two other explanations were tried first and both were refuted by measurement -- a mismatched weight
quantizer (fixed with --w_sym, changed nothing: 1.1667 -> 1.2200) and the choice of statistic
(absmax vs an 80-candidate clip search, both lost). The data provenance is the cause.

WHAT THIS DOES. Builds integration's OWN W4A4 model -- the network that will consume the scales --
runs DDIM with log_every_t=1, and writes the same {xs, ts, xs_prev, ts_prev} structure qdiff's
get_train_samples expects. Pass 2 is then an ordinary qdiff calibration against this file.

Bucketing mirrors generate() in sample_diffusion_ldm.py:185-205 exactly, including its t<=1 special
case, so the two files are interchangeable.

Run: python docs/qdiff_bridge_2026-08-12/scripts/gen_cali_from_int4.py [--cali-n 64] [--cali-st 10]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

OUT = "docs/qdiff_bridge_2026-08-12/data/cali_churches_w4a4.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cali-n", type=int, default=64)
    ap.add_argument("--cali-st", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mode", default="int4_baseline",
                    help="pass-1 model. int4_baseline is the current W4A4 default; the point is that "
                         "it is a QUANTIZED model rather than fp16, not that it is optimal.")
    a = ap.parse_args()
    assert a.steps % a.cali_st == 0, "cali_st must divide steps exactly (generate() indexes t//interval)"

    H.STEPS, H.BATCH, H.SEED = a.steps, a.batch, 1234
    calib = H.CALIB["int4"]
    print(f"pass-1 model: {a.mode}, calibration {calib}", flush=True)
    runner, model, sampler = H.build(a.mode, calib, "static")

    # Warm-up and discard: the quantized attention blocks self-calibrate over their first
    # MODIFF_ATTN_CALIB_STEPS forwards, so a first sample is not steady state and its trajectory is
    # not the one the model will actually follow.
    H.latent(runner, model, sampler)

    xs_l = [[] for _ in range(a.cali_st)]
    ts_l = [[] for _ in range(a.cali_st)]
    xp_l = [[] for _ in range(a.cali_st)]
    tp_l = [[] for _ in range(a.cali_st)]

    n_rounds = (a.cali_n + a.batch - 1) // a.batch
    cond = runner._cond_kwargs(model, a.batch)
    for rnd in range(n_rounds):
        torch.manual_seed(20260812 + rnd)
        torch.cuda.manual_seed_all(20260812 + rnd)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            _, inter = sampler.sample(S=a.steps, batch_size=a.batch, shape=runner.shape,
                                      eta=0.0, verbose=False, log_every_t=1, **cond)
        steps = len(inter["ts"])
        interval = steps // a.cali_st
        for t in range(steps):
            if t % interval:
                continue
            b = t // interval
            if b >= a.cali_st:
                continue
            # Same t<=1 special case as generate(): at the very start there is no earlier step to
            # pair with, so it pairs (t+1, t) instead of (t, t-1).
            i_cur, i_prev = (t + 1, t) if t <= 1 else (t, t - 1)
            xs_l[b].append(inter["x_inter"][i_cur].clone().cpu())
            ts_l[b].append(inter["ts"][i_cur].clone().cpu())
            xp_l[b].append(inter["x_inter"][i_prev].clone().cpu())
            tp_l[b].append(inter["ts"][i_prev].clone().cpu())
        print(f"  round {rnd + 1}/{n_rounds} done", flush=True)

    def stack(lst):
        return torch.stack([torch.cat(b, 0)[:a.cali_n] for b in lst], 0)

    out = {"xs": stack(xs_l), "ts": stack(ts_l),
           "xs_prev": stack(xp_l), "ts_prev": stack(tp_l)}
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert all(not torch.allclose(out["xs"][i], out["xs_prev"][i]) for i in range(a.cali_st)), \
        "xs == xs_prev in some bucket -- the residual pairing is degenerate"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(out, OUT)
    print(f"\nwrote {OUT}")

    # The whole point, quantified: how far has the W4A4 trajectory moved from fp16's?
    fp = "docs/qdiff_bridge_2026-08-12/data/cali_churches_residual.pt"
    if os.path.exists(fp):
        f = torch.load(fp, map_location="cpu", weights_only=False)
        n = min(out["xs"].shape[1], f["xs"].shape[1])
        print("\n  per-bucket activation absmax, fp16-trajectory latents vs W4A4's own:")
        print(f"    {'t':>6} {'fp16':>9} {'W4A4':>9} {'ratio':>7}")
        for i in range(a.cali_st):
            a1 = float(f["xs"][i, :n].abs().max())
            a2 = float(out["xs"][i, :n].abs().max())
            print(f"    {int(out['ts'][i, 0]):6d} {a1:9.3f} {a2:9.3f} {a2 / a1:7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
