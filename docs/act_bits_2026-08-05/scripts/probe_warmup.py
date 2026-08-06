"""Does the t=T warm-up loop actually contract the quantization error?

The paper's warm-up (Appendix D.5): "warm-up can be achieved by repeatedly inputting a_T. This
process converges to the full-precision activation due to the contraction of the quantization error.
... approximately 4 to 5 steps are sufficient to reduce the quantization error to a negligible level
on CIFAR-10 using 4-bit precision."

That contraction needs each round to quantize the SHRUNKEN residual on a grid matched to it, i.e. a
dynamic scale. `_forward_first_step` runs `warmup_steps = 3` rounds but, on the calibrated path that
ships, passes `r_scale = input_scale` -- the full-activation grid -- to every round. If the residual
after round 1 is below half an LSB of that grid, round 2 quantizes it to zero and the loop is a no-op
for everything except elements that CLIPPED in round 1.

This measures, on real activations from the real checkpoint, the relative error of a_hat after each
round under three schemes:

    static   the shipped calibrated path: same static scale every round
    dynamic  a fresh absmax scale per round (what the contraction argument assumes)
    paper    dynamic, run out to 5 rounds

Reported at A8 (Q=127) and A4 (Q=7), because the whole question is whether a rule tuned at 8 bits
still holds at 4.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.dirname(os.path.abspath(__file__))]

import torch                                                                    # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402
from weight_ceiling import conv_eligible                                        # noqa: E402


def warmup(x, Q, rounds, mode, static_scale):
    """Return relative error of a_hat vs x after each round."""
    a_hat = torch.zeros_like(x)
    errs = []
    for r in range(rounds):
        resid = x - a_hat
        if mode == "static":
            s = static_scale                       # the shipped calibrated path
        else:
            s = Q / resid.abs().max().clamp_min(1e-8)
        q = (resid * s).round().clamp(-Q, Q)
        a_hat = a_hat + q / s
        errs.append(float((a_hat - x).norm() / x.norm()))
    return errs


def main():
    r, m, s = H.build("fp16", None, "static")
    unet = m.model.diffusion_model
    names = [n for n, mod in unet.named_modules() if conv_eligible(n, mod)]
    grab = {}
    hooks = [dict(unet.named_modules())[n].register_forward_pre_hook(
        (lambda nm: lambda mod, inp: grab.setdefault(nm, inp[0].detach().float()))(n))
        for n in names[:6]]
    H.SEED = 4242
    H.latent(r, m, s)
    for h in hooks:
        h.remove()

    print(f"{'layer':28s} {'Q':>4} {'scheme':>8} " + " ".join(f"{'r' + str(i + 1):>8}" for i in range(5)))
    print("-" * 100)
    for name in list(grab)[:4]:
        x = grab[name]
        for Q in (127.0, 7.0):
            # the calibrated static grid is Q/range, so build it from the tensor's own range --
            # this is the best case for `static`, a real calibration is fitted over many samples
            static_scale = Q / x.abs().max().clamp_min(1e-8)
            for mode in ("static", "dynamic"):
                e = warmup(x, Q, 5, mode, static_scale)
                print(f"{name[:28]:28s} {int(Q):>4} {mode:>8} "
                      + " ".join(f"{v:8.5f}" for v in e))
        print()


if __name__ == "__main__":
    main()
