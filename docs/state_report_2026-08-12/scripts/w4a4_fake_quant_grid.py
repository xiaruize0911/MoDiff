"""W4A4 in FAKE quantization: which part of the damage is the scale, and which is the kernel?

The real W4A4 arms produce fog (PTQ, relL2 0.8642) and partially-recovered structure (MoDiff,
0.6122). That is a single number over a stack of four different things -- the 4-bit activation
grid, the 4-bit weights, the MoDiff recursion, and the int4 CUTLASS datapath. Fake quantization
separates them, because it runs the ORDINARY fp16 model and simulates one piece at a time.

SIX ARMS, each adding one thing:

  1  fp16                       nothing simulated -- the reference
  2  act 4-bit only             the W4A4 activation scale, weights left fp16
  3  weight 4-bit only          the shipped per-output-channel MSE weight rule, activations fp16
  4  act + weight               full W4A4 PTQ, simulated
  5  act + weight, MoDiff       the same, plus the delta recursion on the qdiff delta table
  6  real int4 kernels (PTQ)    the actual datapath, for the simulation-vs-reality gap

Reading it: 2 vs 3 says whether the scale or the weights dominate. 4 vs 6 says how much the int4
CUTLASS path costs beyond what the arithmetic demands -- if they agree, the kernels are faithful and
the damage is inherent to 4 bits; if 6 is worse, the kernels are adding something.

WHY THE HOOK IS EXACT, not an approximation. A convolution is linear, so MoDiff's recursion
telescopes: o_hat_t = o_hat_{t-1} + A(d_t) = A(a_hat_t) + b. Substituting a_hat for the conv's INPUT
reproduces the whole MoDiff datapath without touching the conv. That is act_fake_quant.py's trick and
this file reuses its DeltaHook rather than reimplementing it -- with QMAX patched from 127 to 7,
which is the only thing that differs at 4 bits. Reimplementing a tested recursion to change one
constant is how the two versions drift apart.

The weight side mirrors OptimizedInt4Conv2d.__init__ exactly by calling the same _int4_weight_scale,
so "weight 4-bit" here means what it means in the shipped kernel, not a second opinion about it.

Run: python docs/state_report_2026-08-12/scripts/w4a4_fake_quant_grid.py   # ~12 min, needs the GPU
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import act_fake_quant as A                                                 # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

#: 4-bit codes are [-7, 7]. Patching the module global is deliberate -- see the docstring.
A.QMAX = 7.0
ACT = "integration/calibration/int4_calibration_qdiff.pt"
DELTA = "integration/calibration/int4_delta_qdiff.pt"
OUT = "docs/state_report_2026-08-12/plots/w4a4_fake_quant.png"
JSON = "docs/state_report_2026-08-12/data/w4a4_fake_quant.json"


def quantize_weights_(convs):
    """4-bit per-output-channel weights, in place. Returns the originals for restore."""
    from integration.kernels.int4_optimized import _int4_weight_scale
    saved = {}
    for name, m in convs.items():
        w = m.weight.data
        saved[name] = w.clone()
        K = w.shape[0]
        wf = w.reshape(K, -1).float()
        sc = _int4_weight_scale(wf)                       # the shipped rule, not a second opinion
        wq = (wf / sc[:, None]).round().clamp(-7.0, 7.0) * sc[:, None]
        m.weight.data = wq.reshape_as(w).to(w.dtype)
    return saved


def restore_weights_(convs, saved):
    for name, m in convs.items():
        m.weight.data = saved[name]


def decode(model, lat, chunk=8):
    lat = lat.to("cuda", torch.float16)
    out = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for i in range(0, lat.shape[0], chunk):
            d = model.decode_first_stage(lat[i:i + chunk])
            out.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu())
            del d
    return (torch.cat(out, 0).numpy() * 255).round().astype("uint8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    H.STEPS, H.BATCH, H.SEED = a.steps, a.n, a.seed
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"

    for p in (ACT, DELTA):
        if not os.path.exists(p):
            print(f"FAIL: missing {p}")
            return 1
    scales = {k: float(v) for k, v in
              torch.load(ACT, map_location="cpu", weights_only=True).items()}
    dtab = torch.load(DELTA, map_location="cpu", weights_only=True)

    rows, quality, ref = [], {}, None
    runner, model, sampler = H.build("fp16", None, "static")
    convs = A.target_convs(model.model.diffusion_model)
    print(f"fake-quant targets: {len(convs)} convs, {len(scales)} scales, "
          f"{len(set(convs) & set(scales))} matched", flush=True)
    if len(set(convs) & set(scales)) != len(scales):
        print("FAIL: the scale file does not match the model's conv names")
        return 1

    #: (label, quantize activations, quantize weights, MoDiff recursion)
    FAKE = [("fp16 reference", False, False, False),
            ("fake: activations 4-bit only", True, False, False),
            ("fake: weights 4-bit only", False, True, False),
            ("fake: act + weight  (W4A4 PTQ simulated)", True, True, False),
            ("fake: act + weight + MoDiff", True, True, True)]

    for label, qa, qw, modiff in FAKE:
        saved = quantize_weights_(convs) if qw else None
        handles = []
        if qa:
            for key, mod in convs.items():
                h = A.DeltaHook(key, scales[key],
                                dtab[key].float() if (modiff and key in dtab) else None,
                                "modiff" if modiff else "baseline")
                handles.append(mod.register_forward_pre_hook(h))
        H.SEED = a.seed
        H.latent(runner, model, sampler)                  # discard: warm-up
        H.SEED = a.seed
        lat, _ = H.latent(runner, model, sampler)
        for hd in handles:
            hd.remove()
        if saved is not None:
            restore_weights_(convs, saved)
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        quality[label] = rel
        print(f"  {label:44s} relL2 {rel:.4f}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, decode(model, lat)))
    del runner, model, sampler
    torch.cuda.empty_cache()

    # The real kernels, same seed, for the simulation-vs-reality gap.
    for label, mode in (("real int4 kernels  (W4A4 PTQ)", "int4_baseline"),
                        ("real int4 kernels  (W4A4 MoDiff)", "int4")):
        r, m, s = H.build(mode, B._default_calibration_path(mode), "static")
        H.SEED = a.seed
        H.latent(r, m, s)
        H.SEED = a.seed
        lat, _ = H.latent(r, m, s)
        rel = float((lat.float() - ref).norm() / ref.norm())
        quality[label] = rel
        print(f"  {label:44s} relL2 {rel:.4f}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}", decode(m, lat)))
        del r, m, s
        torch.cuda.empty_cache()

    cell, pad, lab = a.cell, 6, 26
    W = pad + a.n * (cell + pad)
    Hh = len(rows) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for label, arr in rows:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(a.n, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT, "PNG")
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "QMAX": A.QMAX,
               "act_scales": ACT, "delta_table": DELTA, "relL2": quality},
              open(JSON, "w"), indent=1)
    print(f"\nwrote {OUT}\nwrote {JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
