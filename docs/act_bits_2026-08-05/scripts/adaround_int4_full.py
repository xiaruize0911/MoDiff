"""AdaRound at full strength: more calibration data, 10k iterations, sequential reconstruction.

adaround_int4.py deliberately ran the cheap corner -- 12 calibration samples, 2000 iterations,
layer-wise against fp16 inputs -- and moved the int4 weight floor 0.2442 -> 0.1833. This pulls the
three levers Q-Diffusion uses and that run left alone, and keeps the same acceptance test: the weight
floor from weight_ceiling.py (weights quantized, activations fp16, so no MoDiff and no activation
confounder). The decision it exists to settle: if the floor reaches ~0.12 then W4A4 has real quality
headroom and the int4-kernel injection work is justified; if it stalls near 0.17 then W4A4 should be
treated as a speed-only configuration and its quality target dropped.

The three levers, and why each is expected to matter:

1. DATA -- 104 samples (13 timesteps x batch 8) instead of 12. A diffusion UNet's activation
   distribution changes along the trajectory, so the rounding that minimises output error at t=T is
   not the one that minimises it at t=0; more of the trajectory means a rounding that serves all of it.
   Iterations now draw MINI-BATCHES, so more data does not multiply the per-iteration cost.

2. ITERATIONS -- 10k instead of 2k, which is what a weight can afford to travel: a gate only flips
   when its V crosses 0, V starts in about [-2.4, 2.4], and Adam moves ~lr per step.

3. SEQUENTIAL INPUTS -- the layer's input comes from the ALREADY-QUANTIZED network while the target
   stays the FP network's output, so each layer's rounding also compensates the error it inherits.
   Q-Diffusion does this; the previous run did not.

   Alignment matters here and is easy to get wrong. Once weights change, a DDIM trajectory diverges,
   so "the same step" of two models is not the same input. This teacher-forces instead: the UNet's
   own inputs (latent, timestep, cond) are recorded from one FP sampling run and REPLAYED through both
   models, so x_q and y_fp are the same position of the same trajectory and the pairing is exact.

Arms, all with the shipped MSE clip scale (which AdaRound made the better rule -- see FINDINGS):
  fp16_inputs   levers 1+2. Attributes how much is data/iterations alone.
  sequential    levers 1+2+3, inputs from the nearest-rounded model.
  sequential_r2 one more round: inputs re-collected from the `sequential` model, since its inputs
                were captured from a model whose weights it then changed.
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
from adaround_int4 import measure                                               # noqa: E402  (same floor eval)

Q = 7.0
SEEDS = [1234, 20260805, 777]
ITERS = int(os.environ.get("ADAROUND_ITERS", "10000"))
LR = float(os.environ.get("ADAROUND_LR", "1e-3"))
LAMBDA = float(os.environ.get("ADAROUND_LAMBDA", "0.01"))
MB = int(os.environ.get("ADAROUND_MINIBATCH", "8"))
N_STEPS = int(os.environ.get("ADAROUND_CALIB_STEPS", "13"))
CALIB_BATCH = int(os.environ.get("ADAROUND_CALIB_BATCH", "8"))
ARMS = os.environ.get("ADAROUND_ARMS", "fp16_inputs,sequential,sequential_r2").split(",")
GAMMA, ZETA = -0.1, 1.1
OUT = "docs/act_bits_2026-08-05/data/adaround_int4_full.json"


# ----------------------------------------------------------------- teacher-forced calibration data
def record_unet_inputs(model, runner, sampler, want):
    """The UNet's own (args, kwargs) at selected trajectory positions of one FP sampling run."""
    unet = model.model.diffusion_model
    rec, state = [], {"i": 0}

    def hook(mod, args, kwargs):
        i = state["i"]; state["i"] += 1
        if i in want:
            rec.append(([a.detach().clone() if torch.is_tensor(a) else a for a in args],
                        {k: (v.detach().clone() if torch.is_tensor(v) else v)
                         for k, v in kwargs.items()}))
    h = unet.register_forward_pre_hook(hook, with_kwargs=True)
    H.SEED = 4242                                     # not one of the eval seeds
    H.latent(runner, model, sampler)
    h.remove()
    return rec


def replay(model, rec, capture):
    """Replay recorded UNet inputs, collecting per-layer tensors. capture in {"in_out", "in"}.

    Returns {layer_name: [tensors]} -- inputs, or (input, output) pairs, in fp16 on the CPU.
    """
    unet = model.model.diffusion_model
    store, hooks = {}, []

    def mk(name):
        def hook(mod, inp, out):
            if capture == "in_out":
                store.setdefault(name, []).append((inp[0].detach().half().cpu(),
                                                   out.detach().half().cpu()))
            else:
                store.setdefault(name, []).append(inp[0].detach().half().cpu())
        return hook

    for name, m in unet.named_modules():
        if conv_eligible(name, m):
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for args, kwargs in rec:
            unet(*args, **kwargs)
    for h in hooks:
        h.remove()
    return store


def nearest_int4(w, rule="mse"):
    flat = w.detach().reshape(w.shape[0], -1).float()
    if rule == "mse":
        os.environ["MODIFF_INT4_WSCALE"] = "mse"
        s = _int4_weight_scale(flat, Q=Q)
    else:
        s = torch.clamp(flat.abs().max(dim=1).values / Q, min=1e-8)
    s = s.unsqueeze(1)
    return ((flat / s).round().clamp(-Q, Q) * s).reshape(w.shape)


def build_patched(weights=None, rule=None):
    """fp16 model with eligible conv weights replaced (learned dict, or nearest rounding)."""
    r, m, s = H.build("fp16", None, "static")
    with torch.no_grad():
        for name, mod in m.model.diffusion_model.named_modules():
            if not conv_eligible(name, mod):
                continue
            if weights is not None and name in weights:
                mod.weight.data.copy_(weights[name].to(mod.weight.device, mod.weight.dtype))
            elif rule is not None:
                mod.weight.data.copy_(nearest_int4(mod.weight, rule).to(mod.weight.dtype))
    return r, m, s


# ----------------------------------------------------------------------------- AdaRound, one layer
def adaround_layer(w, X, Y, conv_kwargs, bias=None, iters=ITERS, rule="mse", gen=None, lr=None):
    """Learn the rounding of `w` so that conv(x, Q(w)) matches the FP targets `Y`.

    `bias` is NOT optional in practice and is the subtle part: Y is captured with a forward hook, so
    it is the module's output and INCLUDES the bias. Predicting with bias=None asks learned rounding
    to cancel a constant offset -- it cannot, and trying wrecks the solution. Measured: the first run
    of this script omitted it and came back at floor 0.2879, WORSE than nearest rounding's 0.2442,
    while its own layer-error metric still looked like a 61% improvement (both sides of that ratio
    carried the same offset, so it could not see the fault). Bias is a constant here, never learned.

    X/Y are single stacked tensors [n, C, H, W] on the GPU in fp32 -- every calibration sample for a
    layer has the same shape, so a mini-batch is one index_select plus ONE conv launch rather than
    `MB` of them. That is the difference between ~2 ms and ~0.4 ms per iteration at 10k iterations x
    70 layers x 3 arms. Iterations draw a mini-batch, so the per-iteration cost does not grow with
    how much calibration data was collected.
    """
    w = w.detach().float()
    flat = w.reshape(w.shape[0], -1)
    if rule == "mse":
        os.environ["MODIFF_INT4_WSCALE"] = "mse"
        s = _int4_weight_scale(flat, Q=Q)
    else:
        s = torch.clamp(flat.abs().max(dim=1).values / Q, min=1e-8)
    s_w = s.reshape(-1, *([1] * (w.dim() - 1)))

    ws = w / s_w
    lo = torch.floor(ws)
    frac = (ws - lo).clamp(1e-4, 1 - 1e-4)
    V = (-torch.log((ZETA - GAMMA) / (frac - GAMMA) - 1)).clone().requires_grad_(True)
    near = torch.round(ws).clamp(-Q, Q)

    def err(weight, chunk=16):
        """Relative output error over ALL samples, chunked so the eval never spikes memory."""
        with torch.no_grad():
            num = 0.0
            for i in range(0, X.shape[0], chunk):
                num += float(((F.conv2d(X[i:i + chunk], weight, bias, **conv_kwargs)
                               - Y[i:i + chunk]) ** 2).sum())
            den = float((Y ** 2).sum()) + 1e-12
        return num / den

    base_err = err(near * s_w)
    n = X.shape[0]
    opt = torch.optim.Adam([V], lr=LR if lr is None else lr)
    for it in range(iters):
        p = it / iters
        beta = 20.0 if p < 0.2 else 2.0 + 18.0 * max(0.0, 1 - (p - 0.2) / 0.8)
        idx = torch.randint(0, n, (min(MB, n),), generator=gen).to(X.device)
        xb, yb = X.index_select(0, idx), Y.index_select(0, idx)
        h = torch.clamp(torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA, 0, 1)
        wq = (lo + h).clamp(-Q, Q) * s_w
        num = ((F.conv2d(xb, wq, bias, **conv_kwargs) - yb) ** 2).sum()
        den = float((yb ** 2).sum()) + 1e-12
        reg = (1 - (2 * h - 1).abs().pow(beta)).sum() / h.numel()
        opt.zero_grad(set_to_none=True)
        (num / den + LAMBDA * reg).backward()
        opt.step()

    with torch.no_grad():
        h = torch.clamp(torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA, 0, 1)
        codes = (lo + (h >= 0.5).float()).clamp(-Q, Q)
        learned = codes * s_w
        flipped = float((codes != near).float().mean())
    return learned.cpu(), {"output_err_nearest": base_err, "output_err_learned": err(learned),
                           "frac_flipped_vs_nearest": flipped}


# --------------------------------------------------------------------------------------- one arm
def run_arm(label, meta, targets, inputs, refs, out):
    """inputs/targets: {name: [cpu fp16 tensors]}. Optimise every layer, then measure the floor."""
    learned, stats = {}, {}
    gen = torch.Generator().manual_seed(20260805)
    for i, (name, (w, kw, bias)) in enumerate(meta.items()):
        if name not in inputs or name not in targets:
            continue
        X = torch.cat([t.float() for t in inputs[name]]).to("cuda")
        Y = torch.cat([t.float() for t in targets[name]]).to("cuda")
        lw, st = adaround_layer(w.to("cuda"), X, Y, kw,
                                bias=None if bias is None else bias.to("cuda"),
                                rule="mse", gen=gen)
        learned[name], stats[name] = lw, st
        del X, Y
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == len(meta):
            red = statistics.mean([1 - v["output_err_learned"] / max(v["output_err_nearest"], 1e-12)
                                   for v in stats.values()])
            print(f"  [{label}] {i + 1}/{len(meta)} layers | mean layer-output error reduction "
                  f"{red * 100:.1f}% | flipped "
                  f"{statistics.mean([v['frac_flipped_vs_nearest'] for v in stats.values()]) * 100:.1f}%",
                  flush=True)
    floor = measure(refs, weights=learned)
    out["arms"][label] = {
        "floor": floor,
        "mean_layer_output_err_nearest": statistics.mean(
            [v["output_err_nearest"] for v in stats.values()]),
        "mean_layer_output_err_learned": statistics.mean(
            [v["output_err_learned"] for v in stats.values()]),
        "mean_frac_flipped": statistics.mean(
            [v["frac_flipped_vs_nearest"] for v in stats.values()]),
        "per_layer": stats}
    print(f"  [{label}] FLOOR relL2 {floor['mean']:.4f} +- {floor['stdev']:.4f}   "
          f"{[round(v, 4) for v in floor['per_seed'].values()]}\n", flush=True)
    torch.save(learned, f"docs/act_bits_2026-08-05/data/adaround_full_{label}.pt")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    return learned


def main():
    print(f"AdaRound FULL | iters {ITERS}, lr {LR}, minibatch {MB}, lambda {LAMBDA}\n"
          f"calibration {N_STEPS} timesteps x batch {CALIB_BATCH}, arms {ARMS}, seeds {SEEDS}\n",
          flush=True)

    # ---- fp16 references for the floor, and the recorded UNet inputs, from one fp16 build
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
    print(f"recorded {len(rec)} UNet invocations (batch {CALIB_BATCH}) at steps {sorted(want)}",
          flush=True)

    # ---- FP inputs and targets, from the same replay
    fp = replay(m, rec, "in_out")
    x_fp = {k: [a for a, _ in v] for k, v in fp.items()}
    y_fp = {k: [b for _, b in v] for k, v in fp.items()}
    del fp
    meta = {name: (mod.weight.detach().float().cpu(),
                   dict(stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                        groups=mod.groups),
                   None if mod.bias is None else mod.bias.detach().float().cpu())
            for name, mod in m.model.diffusion_model.named_modules() if conv_eligible(name, mod)}
    gb = sum(sum(t.numel() * 2 for t in v) for v in x_fp.values()) / 2**30
    print(f"captured {len(x_fp)} layers x {len(rec)} tensors, {gb:.2f} GB inputs "
          f"(+ the same again in targets)\n", flush=True)
    del m, s, r
    torch.cuda.empty_cache()

    out = {"iters": ITERS, "lr": LR, "minibatch": MB, "lambda": LAMBDA, "seeds": SEEDS,
           "calib": {"steps": sorted(want), "batch": CALIB_BATCH, "n_samples": len(rec)},
           "arms": {}}

    # ---- optional LR probe. The cheap run used lr=1e-2/2k iterations and the default here is
    # 1e-3/10k; both afford enough travel for a gate to cross zero, but mini-batch noise makes the
    # right choice an empirical question. Probing a few layers on the DIRECT objective (layer output
    # error) costs ~2 minutes and is a valid way to pick an optimiser hyperparameter, where picking
    # it on the floor would be tuning on the acceptance test.
    probe = int(os.environ.get("ADAROUND_PROBE", "0"))
    if probe:
        names = [n for n in list(meta)[:probe] if n in x_fp]
        for lr in [float(v) for v in os.environ.get("ADAROUND_PROBE_LRS", "1e-3,1e-2").split(",")]:
            ratios, flips, gen = [], [], torch.Generator().manual_seed(20260805)
            for n in names:
                w, kw, bias = meta[n]
                X = torch.cat([q.float() for q in x_fp[n]]).to("cuda")
                Y = torch.cat([q.float() for q in y_fp[n]]).to("cuda")
                _, st = adaround_layer(w.to("cuda"), X, Y, kw,
                                       bias=None if bias is None else bias.to("cuda"),
                                       gen=gen, lr=lr)
                ratios.append(st["output_err_learned"] / max(st["output_err_nearest"], 1e-12))
                flips.append(st["frac_flipped_vs_nearest"])
                del X, Y
                torch.cuda.empty_cache()
            print(f"probe lr={lr:<8g} layer output err {statistics.mean(ratios):.3f}x of nearest, "
                  f"flipped {statistics.mean(flips) * 100:.1f}%  ({len(names)} layers)", flush=True)
        return

    if "fp16_inputs" in ARMS:
        print("=== arm fp16_inputs (levers 1+2: data and iterations only) ===", flush=True)
        run_arm("fp16_inputs", meta, y_fp, x_fp, refs, out)

    seq_weights = None
    if "sequential" in ARMS or "sequential_r2" in ARMS:
        print("=== capturing inputs from the nearest-rounded model ===", flush=True)
        r, m, s = build_patched(rule="mse")
        x_q = replay(m, rec, "in")
        del m, s, r
        torch.cuda.empty_cache()
        print("=== arm sequential (levers 1+2+3) ===", flush=True)
        seq_weights = run_arm("sequential", meta, y_fp, x_q, refs, out)
        del x_q
        torch.cuda.empty_cache()

    if "sequential_r2" in ARMS and seq_weights is not None:
        print("=== capturing inputs from the round-1 AdaRound model ===", flush=True)
        r, m, s = build_patched(weights=seq_weights)
        x_q2 = replay(m, rec, "in")
        del m, s, r
        torch.cuda.empty_cache()
        print("=== arm sequential_r2 (one more round) ===", flush=True)
        run_arm("sequential_r2", meta, y_fp, x_q2, refs, out)

    print(f"wrote {OUT}", flush=True)
    print("\nfloors:  nearest 0.2442 (weight_ceiling.py)   cheap AdaRound 0.1833 "
          "(adaround_int4.py)", flush=True)
    for k, v in out["arms"].items():
        print(f"         {k:16s} {v['floor']['mean']:.4f} +- {v['floor']['stdev']:.4f}", flush=True)


if __name__ == "__main__":
    main()
