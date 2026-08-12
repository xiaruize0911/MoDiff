"""Does ONE global clip ratio on the delta table reach the paper's per-layer result?

Fix #1 in the plan is "clip-search the delta table instead of sizing it to the observed absmax".
Before implementing a per-layer MSE search there is a cheaper question: the paper's per-layer values
are ~21x tighter than ours and reach 0.2529 in our datapath. How much of that is available from a
single global ratio applied to the table we already have?

This matters for the implementation, not just the number. A per-layer MSE search needs the delta
DISTRIBUTION, and the fused GN kernel never materialises silu(gn(x)) -- which is exactly why
_observe_delta_absmax records only the absmax and why int4 calibration runs in dynamic mode at all.
Getting a per-layer histogram means forcing calibration onto the non-fused path
(fuse_input_silu = False for the pass, relying on forward_gn_fused_modiff's documented
bit-identity with the two-kernel path). That is worth doing if and only if a global ratio leaves
something on the table.

Arms, all act-only with fp16 weights so the delta grid is the only variable:

  ours            the shipped table                                  expect ~0.4946
  ours / k        the same table with every entry scaled by k        the sweep
  paper           the paper's per-layer delta, as a target           expect ~0.2529

Run: python docs/paper_repro_2026-08-12/scripts/delta_clip_sweep.py    # ~15 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import act_fake_quant as A                                                 # noqa: E402

Q = 7.0
ACT = "integration/calibration/int4_calibration_qdiff.pt"
DELTA = "integration/calibration/int4_delta_qdiff.pt"
PAPER = "docs/paper_repro_2026-08-12/data/paper_act_params.json"
OUT = "docs/paper_repro_2026-08-12/data/delta_clip_sweep.json"
SEEDS = [1234, 20260805, 777]
#: scale multipliers. k>1 tightens the grid (scale = Q/absmax, so x k means absmax / k).
KS = [1.0, 2.0, 4.0, 8.0, 16.0, 21.0, 32.0]


def q_sym(v, s):
    return torch.clamp(torch.round(v * s), -Q, Q) / s


class Hook:
    """integration's seeding (4-bit a_T + 5 warm-up rounds), then a symmetric delta grid."""

    def __init__(self, s_act, s_delta):
        self.s_act, self.s_delta, self.a_hat = s_act, s_delta, None

    def reset(self):
        self.a_hat = None

    def __call__(self, mod, args):
        x = args[0].float()
        if self.a_hat is None or self.a_hat.shape != x.shape:
            a = q_sym(x, self.s_act)
            for _ in range(4):
                r = x - a
                a = a + q_sym(r, Q / r.abs().max().clamp_min(1e-6))
            self.a_hat = a
        else:
            self.a_hat = self.a_hat + q_sym(x - self.a_hat, self.s_delta)
        return (self.a_hat.to(args[0].dtype),) + args[1:]


def main():
    H.STEPS, H.BATCH = 50, 6
    os.environ["MODIFF_LINEAR"] = "0"
    sact = {k: float(v) for k, v in torch.load(ACT, map_location="cpu", weights_only=True).items()}
    dtab = {k: float(v.median()) for k, v in
            torch.load(DELTA, map_location="cpu", weights_only=True).items()}
    paper = json.load(open(PAPER))

    runner, model, sampler = H.build("fp16", None, "static")
    convs = {k: v for k, v in A.target_convs(model.model.diffusion_model).items() if k in sact}
    print(f"{len(convs)} convs\n")

    print("fp16 references ...", flush=True)
    refs = {}
    for sd in SEEDS:
        H.SEED = sd
        H.latent(runner, model, sampler)
        refs[sd] = H.latent(runner, model, sampler)[0].float()

    def run(scale_of):
        rels = []
        for sd in SEEDS:
            hooks = [Hook(sact[k], scale_of(k)) for k in convs]
            hs = [m.register_forward_pre_hook(h) for m, h in zip(convs.values(), hooks)]
            H.SEED = sd
            for h in hooks:
                h.reset()
            H.latent(runner, model, sampler)
            for h in hooks:
                h.reset()
            lat, _ = H.latent(runner, model, sampler)
            for x in hs:
                x.remove()
            rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
        return statistics.mean(rels), rels

    out = {}
    print("global ratio on the shipped delta table:", flush=True)
    for k in KS:
        m, rels = run(lambda n, k=k: dtab[n] * k)
        out[f"ours_x{k:g}"] = {"mean": m, "relL2": rels, "k": k}
        print(f"  ours x{k:<5g} (absmax /{k:g})   {m:.4f}   {[round(x, 3) for x in rels]}", flush=True)

    #: the paper's per-layer delta expressed as our symmetric scale: absmax = delta*(Q+1)
    m, rels = run(lambda n: Q / max(paper[n]["delta"] * (Q + 1), 1e-9))
    out["paper_per_layer"] = {"mean": m, "relL2": rels}
    print(f"  paper per-layer            {m:.4f}   {[round(x, 3) for x in rels]}", flush=True)

    best = min((v["mean"], k) for k, v in out.items() if k.startswith("ours_x"))
    pap = out["paper_per_layer"]["mean"]
    print(f"\nbest global ratio: {best[1]} at {best[0]:.4f}")
    print(f"paper per-layer  : {pap:.4f}")
    gap = best[0] / pap
    if gap < 1.10:
        print(f"A GLOBAL RATIO IS ENOUGH ({gap:.2f}x of the paper's). Implement fix #1 as a swept "
              f"constant in end_delta_calibration -- no per-layer histogram, no non-fused "
              f"calibration path, no kernel change.")
    else:
        print(f"PER-LAYER IS WORTH IT: the best global ratio is {gap:.2f}x off the paper's per-layer "
              f"result. Fix #1 needs the distribution, so calibration has to run on the non-fused "
              f"path (fuse_input_silu=False for the pass) to materialise the delta.")

    json.dump({"seeds": SEEDS, "ks": KS, "results": out}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
