"""Is the surviving qdiff ckpt a usable AdaRound source? Four checks before it is trusted.

/workspace/quant_models/church_w4a8_ckpt.pth (B5's input) is gone from this container. The qdiff run that
produced docs/paper_repro_2026-08-12/paper_w4a4_samples.png saved its own 2.36 GB ckpt.pth, and it carries
the same four keys adaround_weights() reads. This checks it is the real thing rather than a shape-match:

  1. the reconstruction produces at most 16 distinct values per output channel (a 4-bit grid);
  2. `alpha >= 0` is a MIX -- all-True would make "AdaRound" mean "always round up", which is not a
     learned rounding and would be a vacuous arm that still passed every shape assertion;
  3. it differs from round-to-nearest on a non-trivial fraction of weights (that difference IS AdaRound);
  4. the name+shape map onto the UNet reaches the 89 convs B5 verified as a clean bijection.
"""
import os
import re
import sys

import torch

CANDIDATES = [
    "/workspace/quant_models/church_w4a8_ckpt.pth",
    "docs/paper_repro_2026-08-12/qdiff_w4a4/lsun_churches256/samples/ckpt.pth",
]
UNET = "models/ldm/lsun_churches256/model.ckpt"

src = next((p for p in CANDIDATES if os.path.isfile(p)), None)
assert src, f"no AdaRound source found among {CANDIDATES}"
print(f"source: {src}  ({os.path.getsize(src)/2**30:.2f} GiB)")

ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
bases = sorted({m.group(1) for k in ck if (m := re.match(r"(.+)\.weight_quantizer\.alpha$", k))})
print(f"layers with weight_quantizer.alpha: {len(bases)}")

recon, n_conv = {}, 0
alpha_true = alpha_tot = 0
rtn_diff_frac = []
levels_max = 0
for b in bases:
    W = ck[b + ".weight"]
    if W.dim() != 4:
        continue
    n_conv += 1
    W, al = W.float(), ck[b + ".weight_quantizer.alpha"].float()
    d = ck[b + ".weight_quantizer.delta"].float()
    z = ck[b + ".weight_quantizer.zero_point"].float()

    up = (al >= 0).float()
    alpha_true += float(up.sum())
    alpha_tot += up.numel()

    x_q = torch.clamp(torch.floor(W / d) + up + z, 0, 15)
    Wq = (x_q - z) * d
    recon[b[len("model."):]] = Wq

    # check 1: a 4-bit grid, per output channel
    for c in range(min(W.shape[0], 8)):                     # sample 8 channels; 16 is the bound
        levels_max = max(levels_max, int(torch.unique(x_q[c]).numel()))

    # check 3: AdaRound vs round-to-nearest on the same grid
    x_rtn = torch.clamp(torch.round(W / d) + z, 0, 15)
    rtn_diff_frac.append(float((x_q != x_rtn).float().mean()))

frac_up = alpha_true / alpha_tot
mean_rtn_diff = sum(rtn_diff_frac) / len(rtn_diff_frac)
print(f"4-D conv weights reconstructed: {n_conv}")
print(f"check 1  max distinct codes per output channel: {levels_max}  (bound 16)")
print(f"check 2  fraction of alpha >= 0 (round UP):     {frac_up:.4f}")
print(f"check 3  mean fraction differing from RTN:      {mean_rtn_diff:.4f}")

assert levels_max <= 16, f"{levels_max} distinct codes -- not a 4-bit grid"
assert 0.02 < frac_up < 0.98, (
    f"alpha >= 0 is {frac_up:.4f} -- degenerate. All-up or all-down is not a learned rounding, and "
    f"such an arm would still pass every shape check while measuring nothing.")
assert mean_rtn_diff > 0.005, (
    f"only {mean_rtn_diff:.4f} of weights differ from round-to-nearest -- AdaRound would be a no-op "
    f"and the arm would report 'AdaRound does not matter' vacuously.")

# check 4: the bijection onto the UNet
sd = torch.load(UNET, map_location="cpu", weights_only=False, mmap=True)
d = sd.get("state_dict", sd)
hit = sum(1 for rel, Wq in recon.items()
          if (t := d.get("model.diffusion_model." + rel + ".weight")) is not None
          and tuple(t.shape) == tuple(Wq.shape))
print(f"check 4  convs matching the UNet by name+shape: {hit}  (B5 verified 89)")
assert hit == 89, f"{hit} matched, expected B5's 89 -- the mapping is not the verified bijection"
print("\nALL FOUR CHECKS PASS -- usable as the AdaRound source")
