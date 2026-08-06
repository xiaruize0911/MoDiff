"""Sequential AdaRound done correctly: optimise layers in forward order, replacing as you go.

adaround_int4_full.py's `sequential` arm came back at floor 0.4893 -- worse than doing nothing
(nearest rounding is 0.2442) and 3x worse than its own `fp16_inputs` arm (0.1531). The cause was a
structural mistake in that arm, not in the idea:

    it captured EVERY layer's input from the fully nearest-rounded model up front, then optimised all
    70 layers against those inputs simultaneously.

So every layer learned a rounding that compensates the error of a NEAREST-ROUNDED upstream, while in
the finished model the upstream is AdaRound'd and its error is ~74% smaller. Each layer was correcting
an error that no longer existed, and 70 layers of over-correction compounded. The symptom fits: it
flipped 27.8% of codes against `fp16_inputs`'s 14.2%, moving hard in a direction calibrated to the
wrong input distribution, while its own layer-error metric looked healthy because that metric used the
same wrong target.

What this does instead, which is what BRECQ and Q-Diffusion actually do:

    for each quantized conv, in FORWARD EXECUTION ORDER:
        replay the recorded UNet inputs through the model as it stands  -> x_q for this layer
        learn its rounding against (x_q, the FP model's output for this layer)
        WRITE the learned weight back into the model before moving on

so layer k's input reflects the already-corrected layers 1..k-1 -- the error it will actually receive
at inference. Layers after k are still fp16 at that moment, which is fine: they cannot influence k's
input.

Cost is the same order as the broken arm. A replay is 17 UNet forwards (~1 s), so 70 of them is a
couple of minutes; the 10k-iteration per-layer optimisation dominates either way.

Execution order is taken from the order the hooks actually fire, not from named_modules(): a UNet with
skip connections does not necessarily execute in definition order, and getting this backwards would
silently reintroduce a milder version of the same bug.

Acceptance test is unchanged -- the weight floor from weight_ceiling.py. Reference points:
nearest 0.2442, cheap AdaRound 0.1833, fp16-input AdaRound 0.1531, broken sequential 0.4893.
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
import dynamic_delta_ab as H                                                    # noqa: E402
from weight_ceiling import conv_eligible                                        # noqa: E402
from adaround_int4 import measure                                               # noqa: E402
from adaround_int4_full import (adaround_layer, record_unet_inputs, replay,      # noqa: E402
                                ITERS, LR, MB, N_STEPS, CALIB_BATCH, SEEDS)

OUT = "docs/act_bits_2026-08-05/data/adaround_sequential.json"


def firing_order(model, rec):
    """Layer names in the order their forward hooks fire during one replayed UNet call."""
    order, seen, hooks = [], set(), []

    def mk(name):
        def hook(mod, inp, out):
            if name not in seen:
                seen.add(name)
                order.append(name)
        return hook

    for name, m in model.model.diffusion_model.named_modules():
        if conv_eligible(name, m):
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        args, kwargs = rec[0]
        model.model.diffusion_model(*args, **kwargs)
    for h in hooks:
        h.remove()
    return order


def capture_one(model, rec, name):
    """This layer's input under the model as it currently stands. Hooks only the layer we need."""
    mod = dict(model.model.diffusion_model.named_modules())[name]
    got = []
    h = mod.register_forward_pre_hook(lambda m, inp: got.append(inp[0].detach().float()))
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for args, kwargs in rec:
            model.model.diffusion_model(*args, **kwargs)
    h.remove()
    return torch.cat(got)


def main():
    print(f"Sequential AdaRound (in-order, progressive replacement)\n"
          f"iters {ITERS}, lr {LR}, minibatch {MB}, {N_STEPS} timesteps x batch {CALIB_BATCH}, "
          f"seeds {SEEDS}\n", flush=True)

    # ---- fp16 references for the floor, recorded UNet inputs, and the FP targets
    r, m, s = H.build("fp16", None, "static")
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
    refs = {}
    for seed in SEEDS:
        H.SEED = seed
        refs[seed], _ = H.latent(r, m, s)

    bs, H.BATCH = H.BATCH, CALIB_BATCH
    want = set(range(0, H.STEPS, max(1, H.STEPS // N_STEPS)))
    rec = record_unet_inputs(m, r, s, want)
    H.BATCH = bs
    order = firing_order(m, rec)
    fp = replay(m, rec, "in_out")
    y_fp = {k: [b for _, b in v] for k, v in fp.items()}
    del fp
    print(f"recorded {len(rec)} UNet invocations, {len(order)} layers in execution order, "
          f"targets captured\n", flush=True)

    # The model we mutate as we go. Same build, so the same weights to start from.
    mods = dict(m.model.diffusion_model.named_modules())
    learned, stats = {}, {}
    for i, name in enumerate(order):
        mod = mods[name]
        X = capture_one(m, rec, name)                      # <- reflects layers already replaced
        Y = torch.cat([t.float() for t in y_fp[name]]).to("cuda")
        lw, st = adaround_layer(
            mod.weight.detach().float(), X, Y,
            dict(stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups),
            bias=None if mod.bias is None else mod.bias.detach().float(),
            rule="mse", gen=torch.Generator().manual_seed(20260805 + i))
        with torch.no_grad():                              # progressive replacement
            mod.weight.data.copy_(lw.to(mod.weight.device, mod.weight.dtype))
        learned[name], stats[name] = lw.cpu(), st
        del X, Y
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == len(order):
            red = statistics.mean([1 - v["output_err_learned"] / max(v["output_err_nearest"], 1e-12)
                                   for v in stats.values()])
            fl = statistics.mean([v["frac_flipped_vs_nearest"] for v in stats.values()])
            print(f"  {i + 1}/{len(order)} layers | mean layer-output error reduction "
                  f"{red * 100:.1f}% | flipped {fl * 100:.1f}%", flush=True)

    del m, s, r
    torch.cuda.empty_cache()
    floor = measure(refs, weights=learned)
    out = {"iters": ITERS, "lr": LR, "minibatch": MB, "seeds": SEEDS, "n_samples": len(rec),
           "order": order, "floor": floor,
           "mean_layer_output_err_nearest": statistics.mean(
               [v["output_err_nearest"] for v in stats.values()]),
           "mean_layer_output_err_learned": statistics.mean(
               [v["output_err_learned"] for v in stats.values()]),
           "mean_frac_flipped": statistics.mean(
               [v["frac_flipped_vs_nearest"] for v in stats.values()]),
           "per_layer": stats}
    torch.save(learned, "docs/act_bits_2026-08-05/data/adaround_sequential.pt")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFLOOR relL2 {floor['mean']:.4f} +- {floor['stdev']:.4f}   "
          f"{[round(v, 4) for v in floor['per_seed'].values()]}", flush=True)
    print("reference floors: nearest 0.2442 | cheap 0.1833 | fp16-input 0.1531 | "
          "broken-sequential 0.4893", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
