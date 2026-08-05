"""How much of W4A4's error is the WEIGHTS? Fake-quantize weights, leave activations in fp16.

W4A4+MoDiff sits at FID 200 / latent relL2 0.44-0.47 while W8A4+MoDiff (paper anchor) is 0.153, and
the standing inference is "W4A4 is weight-limited, so MoDiff cannot reach it". That inference has
never been measured directly -- it was read off two numbers that differ in both weights AND the
activation datapath. This measures the weight term on its own.

Method: build the fp16 model, then replace each eligible layer's weight with Q(W) dequantized back to
fp16, using the SAME scale rules the shipped kernels use. Activations are never quantized, so the
result is the error floor that no activation-side work -- MoDiff, per-token scales, clip tuning --
can get below. If that floor is already ~0.44, the weights are the ceiling and learned rounding
(AdaRound) is the only way down. If it is ~0.1, the activation side still has room and the
priority order changes.

Layer sets mirror the shipped predicates exactly, so the floor corresponds to a real mode:
  conv    convert_model_to_optimized_int4's filter: Conv2d, in_ch >= 32 and even, not 'skip',
          not under out., groups == 1, not 1x1        -> 87 layers
  linear  wxax_linear._eligible(bits=4): out_f % 64 == 0 and in_f % 64 == 0        -> 42 layers
Scale rules: convs use _int4_weight_scale (MSE search, the shipped default, or absmax via the same
env var); linears use per-output-channel ABSMAX, which is what QuantLinearWxAx does -- the MSE fix
landed on the conv path only, and the `linear_mse` arm below is what extending it would buy.

Same harness, seeds and protocol as act_bit_sweep.py: batch 8, DDIM 50, 3 seeds, paired against a
per-seed fp16 reference, one warm-up run per arm discarded.
"""

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                    # noqa: E402
import torch.nn as nn                                                           # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402
from integration.kernels.int4_optimized import _int4_weight_scale               # noqa: E402

SEEDS = [1234, 20260805, 777]
OUT = "docs/act_bits_2026-08-05/data/weight_ceiling.json"


def conv_eligible(name, m):
    return (isinstance(m, nn.Conv2d) and m.in_channels >= 32 and m.in_channels % 2 == 0
            and "skip" not in name.split(".")[-1] and not name.startswith("out.")
            and m.groups == 1 and m.kernel_size != (1, 1))


def linear_eligible(name, m):
    return isinstance(m, nn.Linear) and m.out_features % 64 == 0 and m.in_features % 64 == 0


def fake_quant_(w, q, rule):
    """Q(W) dequantized in place, per output channel, symmetric. Matches the kernels' layout:
    one scale per output row of the flattened [out, rest] weight."""
    flat = w.detach().reshape(w.shape[0], -1).float()
    if rule == "mse":
        os.environ["MODIFF_INT4_WSCALE"] = "mse"
        s = _int4_weight_scale(flat, Q=q)                     # 13-candidate clip search
    else:
        s = torch.clamp(flat.abs().max(dim=1).values / q, min=1e-8)
    s = s.unsqueeze(1)
    deq = (flat / s).round().clamp(-q, q) * s
    w.data.copy_(deq.reshape(w.shape).to(w.dtype))


def patch(model, conv=None, linear=None):
    """conv/linear are (Q, rule) or None. Returns how many layers each touched."""
    nc = nl = 0
    for name, m in model.named_modules():
        if conv and conv_eligible(name, m):
            fake_quant_(m.weight, *conv); nc += 1
        elif linear and linear_eligible(name, m):
            fake_quant_(m.weight, *linear); nl += 1
    return nc, nl


def arm(label, refs, conv=None, linear=None):
    r, m, s = H.build("fp16", None, "static")               # fp16 activations throughout
    counts = patch(m.model.diffusion_model, conv, linear) if (conv or linear) else (0, 0)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                                       # warm-up, discarded
    rel = {}
    for seed in SEEDS:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        rel[seed] = float((lat - refs[seed]).norm() / refs[seed].norm())
    del m, s, r
    torch.cuda.empty_cache()
    v = list(rel.values())
    mean, sd = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
    print(f"{label:34s} convs={counts[0]:<3d} linears={counts[1]:<3d} "
          f"relL2 {mean:.4f} +- {sd:.4f}  {[round(x, 4) for x in v]}", flush=True)
    return {"label": label, "n_conv": counts[0], "n_linear": counts[1], "per_seed": rel,
            "mean": mean, "stdev": sd}


def main():
    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}   "
          f"(weights fake-quantized, activations fp16)\n", flush=True)
    r, m, s = H.build("fp16", None, "static")
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
    refs = {}
    for seed in SEEDS:
        H.SEED = seed
        refs[seed], _ = H.latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()

    rows = [
        arm("int8 convs (control)", refs, conv=(127.0, "absmax")),
        arm("int4 convs, MSE (shipped)", refs, conv=(7.0, "mse")),
        arm("int4 convs, absmax (old)", refs, conv=(7.0, "absmax")),
        arm("int4 convs MSE + linears absmax", refs, conv=(7.0, "mse"), linear=(7.0, "absmax")),
        arm("int4 convs MSE + linears MSE", refs, conv=(7.0, "mse"), linear=(7.0, "mse")),
    ]
    with open(OUT, "w") as f:
        json.dump({"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS, "rows": rows}, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
