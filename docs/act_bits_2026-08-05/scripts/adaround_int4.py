"""Learned rounding (AdaRound) for the int4 conv weights, scored against the weight floor.

Why. weight_ceiling.py measured that int4 weights alone cost latent relL2 0.244 with a perfect
activation datapath, which sits on the 0.238 -> FID 16.4 anchor. No activation-side work can go below
that, so W4A4 cannot approach fp16 until the weight quantizer improves. The paper does not hit this
wall because its W4 weights come from Q-Diffusion checkpoints, which are produced with AdaRound plus
layer-wise reconstruction (paper Appendix B); this tree has no learned rounding at all.

What AdaRound does. Nearest rounding minimises |W - Q(W)| per weight, but the quantity that matters is
the LAYER OUTPUT error, and per-weight errors partly cancel inside the sum over K. So instead of
round(W/s), take floor(W/s) + h, and learn h in [0,1] per weight against the real input distribution:

    minimise  || conv(x, Q(W)) - conv(x, W) ||^2  +  lambda * sum(1 - |2h-1|^beta)

h is a rectified sigmoid of a free parameter V, and beta anneals 20 -> 2 so h is free to move early
and is pushed to 0/1 by the end. The second term is what makes the final hard rounding lossless.
This is offline only: the product is int4 codes plus one fp16 scale per output channel, exactly the
shipped layout, so INFERENCE AND KERNELS ARE UNCHANGED.

Scale rules are swept because they interact with rounding: the MSE clip search wins end to end at
W4A4 but LOSES on weight-only error (0.2443 vs absmax 0.2184), since clipping discards outliers that
learned rounding cannot recover. Whether that trade survives AdaRound is a measurement, not a guess.

Protocol notes, both load-bearing:
  * Inputs are collected from the FP16 model, which is self-consistent with the evaluation (weights
    quantized, activations fp16). Q-Diffusion instead feeds each layer the already-quantized
    network's activations; that is a strictly better match for the real W4A4 mode and is the obvious
    round 2 if round 1 pays.
  * Calibration inputs are sampled across timesteps, not from one step. The activation distribution
    of a diffusion UNet changes over the trajectory, which is the whole premise of this project.
"""

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.dirname(os.path.abspath(__file__))]

import torch                                                                    # noqa: E402
import torch.nn.functional as F                                                 # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402
from integration.kernels.int4_optimized import _int4_weight_scale               # noqa: E402
from weight_ceiling import conv_eligible                                        # noqa: E402

SEEDS = [1234, 20260805, 777]
Q = 7.0
ITERS = int(os.environ.get("ADAROUND_ITERS", "1500"))
N_CALIB_STEPS = int(os.environ.get("ADAROUND_CALIB_STEPS", "6"))   # timesteps sampled
CALIB_BATCH = int(os.environ.get("ADAROUND_CALIB_BATCH", "2"))
LAMBDA = float(os.environ.get("ADAROUND_LAMBDA", "0.01"))
#: Adam on V. A weight can only flip if its V crosses 0, and V init lives in about [-2.4, 2.4], so
#: the run has to afford ~2 units of travel: Adam moves ~lr per step, hence lr*iters >> 2. At the
#: paper's lr=1e-3 that needs ~10k iterations; 1e-2 x 2000 gets the same travel in a fifth of the
#: time. `frac_flipped_vs_nearest` in the output is the check that this actually happened -- if it
#: reads 0%, the optimiser never moved and the arm is meaningless rather than negative.
LR = float(os.environ.get("ADAROUND_LR", "0.01"))
GAMMA, ZETA = -0.1, 1.1                                            # rectified-sigmoid support
OUT = "docs/act_bits_2026-08-05/data/adaround_int4.json"


# --------------------------------------------------------------------------- calibration capture
def collect(model, want_steps):
    """{layer_name: input tensor [n, C, H, W] on CPU fp16} for every eligible conv.

    One forward pre-hook per layer, firing on `want_steps` of the DDIM trajectory. Storing every
    step would be ~8x the memory for a distribution that barely differs between adjacent steps.
    """
    unet = model.model.diffusion_model
    store, hooks, state = {}, [], {"call": {}}

    def mk(name):
        def hook(mod, inp):
            i = state["call"].get(name, 0)
            state["call"][name] = i + 1
            if i in want_steps:
                store.setdefault(name, []).append(inp[0].detach().half().cpu())
        return hook

    for name, m in unet.named_modules():
        if conv_eligible(name, m):
            hooks.append(m.register_forward_pre_hook(mk(name)))
    return store, hooks


# --------------------------------------------------------------------------- AdaRound, one layer
def adaround_layer(w, x, conv_kwargs, iters=ITERS, scale_rule="mse"):
    """Learn the rounding of `w` against inputs `x`. Returns (dequantized weight, stats)."""
    w = w.detach().float()
    flat = w.reshape(w.shape[0], -1)
    if scale_rule == "mse":
        os.environ["MODIFF_INT4_WSCALE"] = "mse"
        s = _int4_weight_scale(flat, Q=Q)
    else:
        s = torch.clamp(flat.abs().max(dim=1).values / Q, min=1e-8)
    s_w = s.reshape(-1, *([1] * (w.dim() - 1)))                    # broadcast over the kernel

    ws = w / s_w
    lo = torch.floor(ws)
    frac = (ws - lo).clamp(1e-4, 1 - 1e-4)                         # nearest rounding is frac>0.5
    # Init V so h(V) == frac, i.e. the optimisation STARTS at soft-nearest-rounding and can only
    # improve on it. This is the initialisation from the AdaRound paper.
    V = (-torch.log((ZETA - GAMMA) / (frac - GAMMA) - 1)).clone().requires_grad_(True)

    with torch.no_grad():
        ref = [F.conv2d(xi, w, None, **conv_kwargs) for xi in x]
        nearest = (torch.round(ws).clamp(-Q, Q) * s_w)
        base_err = sum(float(((F.conv2d(xi, nearest, None, **conv_kwargs) - ri) ** 2).sum())
                       for xi, ri in zip(x, ref))
        denom = sum(float((ri ** 2).sum()) for ri in ref) + 1e-12

    opt = torch.optim.Adam([V], lr=LR)
    for it in range(iters):
        # beta anneal: h is free for the first fifth, then increasingly pushed to 0/1.
        p = it / iters
        beta = 20.0 if p < 0.2 else 2.0 + (20.0 - 2.0) * max(0.0, (1 - (p - 0.2) / 0.8))
        h = torch.clamp(torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA, 0, 1)
        wq = ((lo + h).clamp(-Q, Q)) * s_w
        loss = sum(((F.conv2d(xi, wq, None, **conv_kwargs) - ri) ** 2).sum()
                   for xi, ri in zip(x, ref)) / denom
        reg = (1 - (2 * h - 1).abs().pow(beta)).sum() / h.numel()
        opt.zero_grad(set_to_none=True)
        (loss + LAMBDA * reg).backward()
        opt.step()

    with torch.no_grad():
        h = torch.clamp(torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA, 0, 1)
        codes = (lo + (h >= 0.5).float()).clamp(-Q, Q)              # hard rounding
        learned = codes * s_w
        learn_err = sum(float(((F.conv2d(xi, learned, None, **conv_kwargs) - ri) ** 2).sum())
                        for xi, ri in zip(x, ref))
        flipped = float((codes != torch.round(ws).clamp(-Q, Q)).float().mean())
    return learned.to(torch.float32), {
        "output_err_nearest": base_err / denom, "output_err_learned": learn_err / denom,
        "frac_flipped_vs_nearest": flipped}


# --------------------------------------------------------------------------- evaluation
def measure(refs, weights=None, rule=None):
    """relL2 of an fp16-activation model whose eligible conv weights are replaced.

    weights: {name: tensor} from AdaRound. rule: ("mse"|"absmax") for plain nearest rounding.
    """
    r, m, s = H.build("fp16", None, "static")
    n = 0
    with torch.no_grad():
        for name, mod in m.model.diffusion_model.named_modules():
            if not conv_eligible(name, mod):
                continue
            if weights is not None and name in weights:
                mod.weight.data.copy_(weights[name].to(mod.weight.device, mod.weight.dtype))
                n += 1
            elif rule is not None:
                flat = mod.weight.detach().reshape(mod.weight.shape[0], -1).float()
                if rule == "mse":
                    os.environ["MODIFF_INT4_WSCALE"] = "mse"
                    sc = _int4_weight_scale(flat, Q=Q)
                else:
                    sc = torch.clamp(flat.abs().max(dim=1).values / Q, min=1e-8)
                sc = sc.unsqueeze(1)
                deq = (flat / sc).round().clamp(-Q, Q) * sc
                mod.weight.data.copy_(deq.reshape(mod.weight.shape).to(mod.weight.dtype))
                n += 1
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                                               # warm-up, discarded
    rel = {}
    for seed in SEEDS:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        rel[seed] = float((lat - refs[seed]).norm() / refs[seed].norm())
    del m, s, r
    torch.cuda.empty_cache()
    v = list(rel.values())
    return {"n_layers": n, "per_seed": rel, "mean": statistics.mean(v),
            "stdev": statistics.stdev(v) if len(v) > 1 else 0.0}


def main():
    print(f"AdaRound int4 conv weights | iters {ITERS}, lambda {LAMBDA}, "
          f"{N_CALIB_STEPS} timesteps x batch {CALIB_BATCH}, seeds {SEEDS}\n", flush=True)

    # ---- fp16 references (per seed) and the calibration capture, from one build
    r, m, s = H.build("fp16", None, "static")
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
    refs = {}
    for seed in SEEDS:
        H.SEED = seed
        refs[seed], _ = H.latent(r, m, s)

    want = set(range(0, H.STEPS, max(1, H.STEPS // N_CALIB_STEPS)))
    store, hooks = collect(m, want)
    bs, H.BATCH = H.BATCH, CALIB_BATCH
    H.SEED = 4242                                                   # not one of the eval seeds
    H.latent(r, m, s)
    H.BATCH = bs
    for h in hooks:
        h.remove()
    shapes = {k: tuple(v[0].shape) for k, v in store.items()}
    mb = sum(sum(t.numel() * 2 for t in v) for v in store.values()) / 2**20
    print(f"captured {len(store)} layers, {len(next(iter(store.values())))} tensors each, "
          f"{mb:.0f} MB\n", flush=True)
    # Keep the module metadata AdaRound needs, then release the model before optimising.
    meta = {name: (mod.weight.detach().float().cpu(),
                   dict(stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                        groups=mod.groups))
            for name, mod in m.model.diffusion_model.named_modules() if conv_eligible(name, mod)}
    del m, s, r
    torch.cuda.empty_cache()

    out = {"iters": ITERS, "lambda": LAMBDA, "seeds": SEEDS, "calib": {"steps": sorted(want),
           "batch": CALIB_BATCH, "shapes": {k: list(v) for k, v in shapes.items()}}, "arms": {}}

    for rule in ("mse", "absmax"):
        print(f"=== scale rule: {rule} ===", flush=True)
        learned, stats = {}, {}
        for i, (name, (w, kw)) in enumerate(meta.items()):
            if name not in store:
                continue
            x = [t.to("cuda", torch.float32) for t in store[name]]
            lw, st = adaround_layer(w.to("cuda"), x, kw, scale_rule=rule)
            learned[name] = lw.cpu()
            stats[name] = st
            del x
            if (i + 1) % 20 == 0 or i + 1 == len(meta):
                imp = statistics.mean([1 - s2["output_err_learned"] / max(s2["output_err_nearest"], 1e-12)
                                       for s2 in stats.values()])
                print(f"  {i + 1}/{len(meta)} layers | mean layer-output error reduction "
                      f"{imp * 100:.1f}%", flush=True)
        torch.cuda.empty_cache()
        near = measure(refs, rule=rule)
        ada = measure(refs, weights=learned)
        out["arms"][rule] = {
            "nearest": near, "adaround": ada,
            "mean_layer_output_err_nearest": statistics.mean(
                [s2["output_err_nearest"] for s2 in stats.values()]),
            "mean_layer_output_err_adaround": statistics.mean(
                [s2["output_err_learned"] for s2 in stats.values()]),
            "mean_frac_flipped": statistics.mean(
                [s2["frac_flipped_vs_nearest"] for s2 in stats.values()]),
            "per_layer": stats,
        }
        print(f"  nearest  relL2 {near['mean']:.4f} +- {near['stdev']:.4f}\n"
              f"  AdaRound relL2 {ada['mean']:.4f} +- {ada['stdev']:.4f}   "
              f"({near['mean'] / max(ada['mean'], 1e-9):.2f}x)   "
              f"flipped {out['arms'][rule]['mean_frac_flipped'] * 100:.1f}% of codes\n", flush=True)
        torch.save(learned, f"docs/act_bits_2026-08-05/data/adaround_int4_{rule}.pt")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
