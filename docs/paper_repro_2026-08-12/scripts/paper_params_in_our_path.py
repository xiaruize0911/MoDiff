"""Put the PAPER's calibrated parameters into OUR datapath, one difference at a time.

The README command reproduces cleanly (docs/paper_repro_2026-08-12/paper_w4a4_samples.png) while
integration's W4A4 is fog. Everything measured so far says the gap is configuration rather than
kernel arithmetic -- fake quant and the real kernels agree to 4-18% -- but "configuration" is four
things at once. This separates them by running each in the same harness.

WHAT THE PAPER'S NUMBERS ACTUALLY ARE. Under --modulate, qdiff's `act_quantizer` holds the step size
of the TEMPORAL DELTA, not of the activation (export_qdiff_scales.py documents this as the trap the
script exists to prevent). Extracted from its ckpt.pth, mapped onto integration's 70 conv names,
all 70 resolve:

    delta      median 0.011957
    zero_point median 7.50, non-zero on 168/168 -- ASYMMETRIC
    representable [-0.0878, +0.0888]

integration's delta table assumes a half-width of 1.84. That is 21x wider on the same quantity.

AND THE PAPER DOES NOT QUANTIZE a_T AT ALL. Under --modulate, QuantModule.forward's `a_hat is None`
branch skips the activation quantizer, so a_hat_T is the exact fp16 activation. integration
quantizes it on the 4-bit activation grid and then refines it with 5 warm-up rounds.

FIVE ARMS, each isolating one thing:

  1  fp16                                                     the reference
  2  ours: a_T on the 4-bit grid + our delta table            what integration ships
  3  ours' seeding + PAPER delta/zp                           does the delta grid explain it?
  4  PAPER seeding (a_T exact) + PAPER delta/zp               the paper's datapath in our harness
  5  PAPER seeding + paper delta magnitude, SYMMETRIC         what the discarded zero point costs

3 vs 2 isolates the delta grid. 4 vs 3 isolates the unquantized a_T. 5 vs 4 isolates the zero point.
If 4 comes out clean, the gap is entirely parameters and the fix is a file format plus a zero-point
term in the kernel. If 4 is still fog, something in our datapath differs beyond these parameters.

Run: python docs/paper_repro_2026-08-12/scripts/paper_params_in_our_path.py   # ~12 min, needs GPU
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
from PIL import Image, ImageDraw                                            # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import act_fake_quant as A                                                 # noqa: E402

PAPER = "docs/paper_repro_2026-08-12/data/paper_act_params.json"
ACT = "integration/calibration/int4_calibration_qdiff.pt"
DELTA = "integration/calibration/int4_delta_qdiff.pt"
OUT_IMG = "docs/paper_repro_2026-08-12/paper_params_grid.png"
OUT_JSON = "docs/paper_repro_2026-08-12/data/paper_params_in_our_path.json"
Q = 7.0
SEEDS = [1234, 20260805, 777]


def q_sym(v, s):
    return torch.clamp(torch.round(v * s), -Q, Q) / s


def q_asym(v, delta, zp):
    """qdiff's UniformAffineQuantizer: x_int = round(x/delta) + zp, clamped to [0, 15]."""
    xi = torch.clamp(torch.round(v / delta) + zp, 0, 2 * Q + 1)
    return (xi - zp) * delta


class Hook:
    """MoDiff recursion with selectable seeding and delta quantizer."""

    def __init__(self, s_act, dtab, pd, pz, seed_exact, delta_kind):
        self.s_act, self.dtab, self.pd, self.pz = s_act, dtab, pd, pz
        self.seed_exact, self.delta_kind = seed_exact, delta_kind
        self.a_hat = None

    def reset(self):
        self.a_hat = None

    def __call__(self, mod, args):
        x = args[0].float()
        if self.a_hat is None or self.a_hat.shape != x.shape:
            if self.seed_exact:
                self.a_hat = x.clone()                      # the paper: a_hat_T = a_T, exact
            else:
                a = q_sym(x, self.s_act)                    # integration: 4-bit grid + warm-up
                for _ in range(4):
                    r = x - a
                    a = a + q_sym(r, Q / r.abs().max().clamp_min(1e-6))
                self.a_hat = a
            return (self.a_hat.to(args[0].dtype),) + args[1:]
        d = x - self.a_hat
        if self.delta_kind == "ours":
            dq = q_sym(d, self.dtab)
        elif self.delta_kind == "paper":
            dq = q_asym(d, self.pd, self.pz)
        else:                                               # paper magnitude, symmetric
            dq = q_sym(d, Q / max(self.pd * (Q + 1), 1e-9))
        self.a_hat = self.a_hat + dq
        return (self.a_hat.to(args[0].dtype),) + args[1:]


def decode(model, lat, chunk=8):
    lat = lat.to("cuda", torch.float16)
    out = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for i in range(0, lat.shape[0], chunk):
            d = model.decode_first_stage(lat[i:i + chunk])
            out.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu())
    return (torch.cat(out, 0).numpy() * 255).round().astype("uint8")


def main():
    H.STEPS, H.BATCH = 50, 6
    os.environ["MODIFF_LINEAR"] = "0"
    paper = json.load(open(PAPER))
    sact = {k: float(v) for k, v in torch.load(ACT, map_location="cpu", weights_only=True).items()}
    dtab = torch.load(DELTA, map_location="cpu", weights_only=True)

    runner, model, sampler = H.build("fp16", None, "static")
    convs = {k: v for k, v in A.target_convs(model.model.diffusion_model).items() if k in paper}
    if len(convs) != len(paper):
        print(f"FAIL: {len(convs)}/{len(paper)} matched")
        return 1
    print(f"{len(convs)} convs matched\n")

    ARMS = [("fp16 reference", None),
            ("2  ours: 4-bit a_T + our delta table", ("ours", False)),
            ("3  ours' a_T + PAPER delta/zp", ("paper", False)),
            ("4  PAPER a_T exact + PAPER delta/zp", ("paper", True)),
            ("5  PAPER a_T exact + paper magnitude, SYMMETRIC", ("sym", True))]

    rows, quality, refs = [], {}, {}
    for label, cfg in ARMS:
        rels = []
        for sd in SEEDS:
            hooks, handles = [], []
            if cfg is not None:
                kind, seed_exact = cfg
                for k, mod in convs.items():
                    h = Hook(sact[k], float(dtab[k].median()), paper[k]["delta"], paper[k]["zp"],
                             seed_exact, kind)
                    hooks.append(h)
                    handles.append(mod.register_forward_pre_hook(h))
            H.SEED = sd
            for h in hooks:
                h.reset()
            H.latent(runner, model, sampler)
            for h in hooks:
                h.reset()
            lat, _ = H.latent(runner, model, sampler)
            for hd in handles:
                hd.remove()
            if cfg is None:
                refs[sd] = lat.float().clone()
            else:
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            if sd == SEEDS[0]:
                keep = lat
        m = statistics.mean(rels) if rels else 0.0
        quality[label] = m
        print(f"  {label:48s} relL2 {m:.4f}" + (f"   {[round(x, 3) for x in rels]}" if rels else ""),
              flush=True)
        rows.append((f"{label}    relL2 {m:.4f}" if rels else label, decode(model, keep)))

    cell, pad, lab = 256, 6, 26
    W = pad + 6 * (cell + pad)
    Hh = len(rows) * (cell + lab + pad) + pad
    c = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(c)
    y = pad
    for label, arr in rows:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(6, arr.shape[0])):
            c.paste(Image.fromarray(arr[i]), (pad + i * (cell + pad), y))
        y += cell + pad
    c.save(OUT_IMG)
    json.dump({"seeds": SEEDS, "relL2": quality}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_IMG}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
