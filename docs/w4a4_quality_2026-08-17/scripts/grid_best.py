"""All six W4A4+MoDiff arms. L0 vs L1 (MoDiff on the 42 attention projections) x AdaRound x fp16 attn.

benchmark_5mode records L1 as "recovering structure L0 loses entirely" at W4A4 -- the committed FID
table is L0 throughout, so L1 at W4A4 has never had an image or an FID row.

mean |delta| is a SCREEN, not the verdict. AdaRound restores high-frequency detail, which can raise
pixel distance while improving perceptual quality -- the same nonlinearity that made relL2 disagree with
FID twice in this project (B3, C7). FID decides; this ranks candidates for it.
"""
import itertools
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

B = "/workspace/fid_warmup5"
ARMS = [
    ("int4_modiff_l0", "L0", "conv-only MoDiff (shipped)"),
    ("int4_modiff_l0_adaround", "L0 + AdaRound", "AdaRound W4 weights"),
    ("int4_modiff_l0_attnfp16", "L0 + fp16 attn", "score path in fp16"),
    ("int4_modiff_l0_adaround_attnfp16", "L0 + AdaRound + fp16 attn", "both"),
    ("int4_modiff_l1", "L1", "MoDiff on the 42 projections too"),
    ("int4_modiff_l1_adaround", "L1 + AdaRound", "projections + AdaRound"),
    ("int4_modiff_l1_static", "L1 + static delta", "static per-step delta table"),
    ("int4_modiff_l1_adaround_static", "L1 + AdaRound + static", "all three"),
]
SHOW = ["int4_modiff_l0", "int4_modiff_l0_adaround", "int4_modiff_l1_adaround", "int4_modiff_l1_static"]
OUT = os.path.dirname(os.path.abspath(__file__))
TILE, PAD, LABEL_W, HEADER = 224, 6, 270, 26
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def names(folder):
    d = os.path.join(B, folder)
    assert os.path.isdir(d), f"{d} missing"
    f = sorted(x for x in os.listdir(d) if x.endswith(".png"))
    assert f, f"{d} holds no PNG"
    return f


def arr(folder, nm):
    p = os.path.join(B, folder, nm)
    assert os.path.isfile(p), f"missing {p}"
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def delta(fa, fb, ns):
    tot = npix = 0
    for nm in ns:
        a, b = arr(fa, nm), arr(fb, nm)
        tot += float(np.abs(a - b).sum())
        npix += a.size
    return tot / npix


ns = names("fp16")
for folder, _, _ in ARMS:
    assert not [c for c in ns if c not in set(names(folder))], f"{folder} not paired"
D = {f: delta("fp16", f, ns) for f, _, _ in ARMS}
base = D["int4_modiff_l0"]

print(f"{len(ns)} paired images per arm.  W8A8+MoDiff sits at 2.80/255 for scale.\n")
print(f"{'arm':44} {'mean |Δ| vs fp16':>16} {'vs L0':>8}")
for folder, lab, sub in sorted(ARMS, key=lambda r: D[r[0]]):
    rel = "--" if folder == "int4_modiff_l0" else f"{(D[folder]/base - 1)*100:+.1f}%"
    print(f"  {lab:42} {D[folder]:8.3f}/255 {rel:>8}   ({sub})")

for folder, lab, _ in ARMS:
    assert D[folder] > 0, f"{lab} identical to fp16"
for (fa, la, _), (fb, lb, _) in itertools.combinations(ARMS, 2):
    assert delta(fa, fb, ns) > 0, f"{la} and {lb} byte-identical -- a flag did not apply"
print("\nnon-vacuity OK: all six arms mutually distinct")

fnt_b = ImageFont.truetype(FONT_B, 16)
fnt_s = ImageFont.truetype(FONT, 12)
lut = {f: (l, s) for f, l, s in ARMS}
rows = [("fp16", "fp16", "reference")] + [(f, *lut[f]) for f in SHOW]


def grid(cols, out_png, crop=None):
    W = LABEL_W + len(cols) * (TILE + PAD) + PAD
    H = HEADER + len(rows) * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for j, nm in enumerate(cols):
        dr.text((LABEL_W + j * (TILE + PAD), 6), f"#{int(nm.split('.')[0])}", font=fnt_s,
                fill=(90, 90, 90))
    for i, (folder, lab, sub) in enumerate(rows):
        y = HEADER + i * (TILE + PAD)
        dr.text((PAD, y + 4), lab, font=fnt_b, fill=(0, 0, 0))
        dr.text((PAD, y + 26), sub, font=fnt_s, fill=(110, 110, 110))
        if folder != "fp16":
            dr.text((PAD, y + 44), f"mean |Δ| vs fp16  {D[folder]:.2f}/255", font=fnt_s,
                    fill=(110, 110, 110))
        for j, nm in enumerate(cols):
            im = Image.open(os.path.join(B, folder, nm)).convert("RGB")
            if crop:
                cx, cy, side = crop
                im = im.crop((cx, cy, cx + side, cy + side))
            canvas.paste(im.resize((TILE, TILE), Image.LANCZOS), (LABEL_W + j * (TILE + PAD), y))
    canvas.save(out_png)
    print(f"wrote {out_png}  ({canvas.size[0]}x{canvas.size[1]})")


grid(ns[:6], f"{OUT}/best_arms_w4a4.png")
grid(ns[:4], f"{OUT}/best_arms_w4a4_zoom.png", crop=(64, 64, 128))
json.dump({"dir": B, "n_paired": len(ns), "mean_abs_delta_vs_fp16_255": D,
           "w8a8_modiff_reference_255": 2.803,
           "config": "DDIM 50, batch 16, dynamic delta, MODIFF_WARMUP_STEPS=5, seed0=20260805"},
          open(f"{OUT}/best_arms_w4a4.json", "w"), indent=1)
print("OK")
