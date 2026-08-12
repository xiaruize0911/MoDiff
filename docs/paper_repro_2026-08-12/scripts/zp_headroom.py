"""How much is a zero point still worth, now that both clip ratios have landed?

Fix #2 in the plan is the expensive one: a zero point on the activation grid touches the .pt format,
the module state, up to 15 CUDA quantize entry points, and needs a Sigma(w_q) fold in the epilogue.
It was justified by a 1.76x measured BEFORE fixes #1 and #3, on a grid that was not yet clipped. Both
of those exploited the same slack the zero point does -- saturating a one-sided distribution's tail --
so the remaining headroom is almost certainly smaller than 1.76x and has to be re-measured before
committing to that much kernel work.

TWO CORRECTIONS TO HOW THIS IS MEASURED, both learned the hard way this session:

  4-BIT WEIGHTS. The earlier sweep ran fp16 weights, which made it read optimistic once the grid got
  fine: it predicted 0.1147 for fix #1 where the kernels delivered 0.3099. Weights are quantized here
  using the shipped rule (_int4_weight_scale) so the comparison sits in the regime that ships.

  PTQ ONLY. The real-kernel sweep showed MoDiff is insensitive to the activation grid -- 1.09x across
  a 10x ratio range, against the W4A4 noise floor of 0.6% -- because it reads the grid only at t=T and
  then refines with 5 warm-up rounds. A zero point on that grid therefore cannot help MoDiff much, and
  the decision rests on the PTQ axis.

Arms, all with 4-bit weights, PTQ semantics (every step quantized on the static grid):

  sym  @ shipped ratio 4.5     what ships today
  sym  @ its own best ratio    the clip lever alone, re-swept here for a like-for-like baseline
  asym @ matching ratios       the zero point on top

Decision rule, stated before the run: if asym beats the best sym by less than 1.15x, fix #2 is not
worth 15 kernel entry points and gets deprioritised the way #4 did -- on evidence, with the number
recorded.

Run: python docs/paper_repro_2026-08-12/scripts/zp_headroom.py    # ~12 min, needs the GPU
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
OUT = "docs/paper_repro_2026-08-12/data/zp_headroom.json"
SEEDS = [1234, 20260805, 777]
RATIOS = [1.0, 3.0, 4.5, 6.7]


class Col:
    def __init__(self):
        self.vmin, self.vmax, self.absmax = float("inf"), float("-inf"), 0.0

    def __call__(self, mod, args):
        v = args[0].detach().float()
        self.vmin = min(self.vmin, float(v.min()))
        self.vmax = max(self.vmax, float(v.max()))
        self.absmax = max(self.absmax, float(v.abs().max()))
        return None


class AQ:
    """PTQ semantics: replace the input with its 4-bit reconstruction every call."""

    def __init__(self, kind, c, r):
        if kind == "sym":
            self.s, self.lo = Q / max(c.absmax / r, 1e-9), None
        else:
            lo, hi = c.vmin, c.vmax / r
            self.s, self.lo = (2 * Q) / max(hi - lo, 1e-9), lo

    def __call__(self, mod, args):
        v = args[0].float()
        if self.lo is None:
            qv = torch.clamp(torch.round(v * self.s), -Q, Q) / self.s
        else:
            qv = (torch.clamp(torch.round((v - self.lo) * self.s) - Q, -Q, Q) + Q) / self.s + self.lo
        return (qv.to(args[0].dtype),) + args[1:]


def main():
    from integration.kernels.int4_optimized import _int4_weight_scale
    H.STEPS, H.BATCH = 50, 6
    os.environ["MODIFF_LINEAR"] = "0"
    runner, model, sampler = H.build("fp16", None, "static")
    convs = A.target_convs(model.model.diffusion_model)

    print("fp16 references ...", flush=True)
    refs = {}
    for sd in SEEDS:
        H.SEED = sd
        H.latent(runner, model, sampler)
        refs[sd] = H.latent(runner, model, sampler)[0].float()

    # WEIGHTS FIRST, THEN RANGES. The first version of this script collected activation ranges on
    # the fp16 model and quantized the weights afterwards, so every arm ran on ranges that did not
    # belong to the model being measured. That is why it disagreed with the real kernels not just in
    # magnitude but in ORDERING -- it put the symmetric optimum past 6.7 where the kernels put it at
    # 4.5. In the shipped path calibration and deployment both see quantized weights; so must this.
    saved = {}
    for k, m in convs.items():
        w = m.weight.data
        saved[k] = w.clone()
        K = w.shape[0]
        wf = w.reshape(K, -1).float()
        sc = _int4_weight_scale(wf)
        m.weight.data = ((wf / sc[:, None]).round().clamp(-7, 7) * sc[:, None]).reshape_as(w).to(w.dtype)
    print("weights quantized to 4 bits (shipped rule)", flush=True)

    print("collecting activation ranges ON THE QUANTIZED-WEIGHT MODEL ...", flush=True)
    cols = {k: Col() for k in convs}
    hs = [convs[k].register_forward_pre_hook(cols[k]) for k in convs]
    H.SEED = SEEDS[0]
    H.latent(runner, model, sampler)
    for h in hs:
        h.remove()

    out = {}
    for kind in ("sym", "asym"):
        for r in RATIOS:
            rels = []
            for sd in SEEDS:
                hs = [convs[k].register_forward_pre_hook(AQ(kind, cols[k], r)) for k in convs]
                H.SEED = sd
                H.latent(runner, model, sampler)
                lat, _ = H.latent(runner, model, sampler)
                for h in hs:
                    h.remove()
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            out[f"{kind}_r{r:g}"] = {"mean": statistics.mean(rels), "relL2": rels}
            print(f"  {kind:4s} ratio {r:<5g} {statistics.mean(rels):.4f}   "
                  f"{[round(x, 3) for x in rels]}", flush=True)
    for k, m in convs.items():
        m.weight.data = saved[k]

    # SELF-CHECK BEFORE THE VERDICT. The real-kernel sweep is authoritative and puts the symmetric
    # optimum at 4.5 (0.4692) with 6.7 worse (0.5312). If this harness's symmetric arm does not
    # reproduce that ORDERING, it is not measuring the shipped regime and its asymmetric number means
    # nothing -- it has already been wrong twice (2.7x on fix #1, and the ordering above).
    sym_min = min(RATIOS, key=lambda r: out[f"sym_r{r:g}"]["mean"])
    trust = sym_min == 4.5 and out["sym_r4.5"]["mean"] < out["sym_r6.7"]["mean"]
    print(f"\n  self-check: symmetric optimum at {sym_min:g}, real kernels say 4.5 -> "
          f"{'AGREES, verdict below is usable' if trust else 'DISAGREES, verdict below is NOT usable'}")

    bs = min((out[f"sym_r{r:g}"]["mean"], r) for r in RATIOS)
    ba = min((out[f"asym_r{r:g}"]["mean"], r) for r in RATIOS)
    gain = bs[0] / ba[0]
    print(f"\nbest sym  ratio {bs[1]:g} at {bs[0]:.4f}")
    print(f"best asym ratio {ba[1]:g} at {ba[0]:.4f}")
    print(f"zero point is worth {gain:.2f}x on top of the best clip")
    if not trust:
        print("HARNESS NOT TRUSTED: it does not reproduce the real kernels' ordering, so this "
              "zero-point number cannot decide fix #2. Deciding it needs the kernel change itself.")
    elif gain < 1.15:
        print("BELOW THE BAR: fix #2 does not justify 15 CUDA entry points plus a Sigma(w_q) fold. "
              "Deprioritise it the way #4 was, with this number on the record.")
    else:
        print("WORTH IT: implement fix #2. Reachable quantize kernels first, the rest refusing a "
              "non-zero zp rather than ignoring it.")
    json.dump({"seeds": SEEDS, "ratios": RATIOS, "results": out, "harness_trusted": trust,
               "best_sym": bs, "best_asym": ba, "zp_gain": gain}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
