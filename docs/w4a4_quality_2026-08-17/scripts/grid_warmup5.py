"""Paired sample grid for the warmup=5 run, with the pixel distance that makes "looks the same" a number.

Pairing: generate_fid_samples.py drives every mode with the same seed sequence, so <mode>/000003.png is
the same noise draw in all three folders. Column j is one latent rendered three ways; any difference down
a column is the mode.

Every check here tests that both files EXIST before comparing them. A previous non-vacuity check in this
project used `cmp -s a b || echo differs` and reported three PASSes on an empty directory, because `||`
takes the branch when a file is missing, not only when it differs. A check whose failure mode is a PASS
is worse than no check.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FID = os.environ.get("FID_DIR", "/workspace/fid_warmup5")
OUT = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
MODES = [("fp16", "fp16  (reference)"),
         ("int8_modiff_l0", "W8A8 + MoDiff"),
         ("int4_modiff_l0", "W4A4 + MoDiff")]
NCOL = int(os.environ.get("SAMPLE_COLS", "6"))
TILE, PAD, LABEL_W, HEADER = 224, 6, 210, 26
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def names(folder, n=None):
    d = os.path.join(FID, folder)
    assert os.path.isdir(d), f"{d} does not exist -- nothing was generated for {folder}"
    f = sorted(x for x in os.listdir(d) if x.endswith(".png"))
    assert f, f"{d} exists but holds no PNG"
    return f[:n] if n else f


def arr(folder, name):
    p = os.path.join(FID, folder, name)
    assert os.path.isfile(p), f"missing {p}"          # EXISTS before COMPARE
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def mean_abs_delta(fa, fb, ns):
    tot = npix = 0
    for nm in ns:
        a, b = arr(fa, nm), arr(fb, nm)
        tot += float(np.abs(a - b).sum())
        npix += a.size
    return tot / npix


all_names = names("fp16")
for folder, _ in MODES:                                # pairing is an assumption; check it
    have = set(names(folder))
    missing = [c for c in all_names if c not in have]
    assert not missing, f"{folder} is missing {missing} -- the columns would not be paired"
print(f"{len(all_names)} paired images per mode, from {FID}")

DELTA = {f: mean_abs_delta("fp16", f, all_names) for f, _ in MODES}
for folder, label in MODES:
    print(f"  {label:22} mean |delta| vs fp16 over {len(all_names)} paired images: "
          f"{DELTA[folder]:.3f}/255")

# --- non-vacuity: three DISTINCT arms, not three renders of one -------------------------------------
assert DELTA["fp16"] == 0.0, "the fp16 control must be exactly 0, else the pairing is broken"
cross = mean_abs_delta("int8_modiff_l0", "int4_modiff_l0", all_names)
print(f"  W8A8 vs W4A4 (arms against each other):  {cross:.3f}/255")
for f, _ in MODES[1:]:
    assert DELTA[f] > 0.0, f"{f} is byte-identical to fp16 -- the arm did not engage"
assert cross > 0.0, "the two quantized arms are byte-identical to each other -- one env var did not take"

fnt_b = ImageFont.truetype(FONT_B, 16)
fnt_s = ImageFont.truetype(FONT, 12)


def grid(cols, out_png, crop=None):
    W = LABEL_W + len(cols) * (TILE + PAD) + PAD
    H = HEADER + len(MODES) * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for j, nm in enumerate(cols):
        dr.text((LABEL_W + j * (TILE + PAD), 6), f"#{int(nm.split('.')[0])}",
                font=fnt_s, fill=(90, 90, 90))
    for i, (folder, label) in enumerate(MODES):
        y = HEADER + i * (TILE + PAD)
        dr.text((PAD, y + 4), label, font=fnt_b, fill=(0, 0, 0))
        sub = ("control" if folder == "fp16"
               else f"mean |Δ| vs fp16  {DELTA[folder]:.2f}/255")
        dr.text((PAD, y + 26), sub, font=fnt_s, fill=(110, 110, 110))
        dr.text((PAD, y + 44), "MODIFF_WARMUP_STEPS=5" if folder != "fp16" else "",
                font=fnt_s, fill=(110, 110, 110))
        for j, nm in enumerate(cols):
            im = Image.open(os.path.join(FID, folder, nm)).convert("RGB")
            if crop:
                cx, cy, side = crop
                im = im.crop((cx, cy, cx + side, cy + side))
            canvas.paste(im.resize((TILE, TILE), Image.LANCZOS), (LABEL_W + j * (TILE + PAD), y))
    canvas.save(out_png)
    print(f"wrote {out_png}  ({canvas.size[0]}x{canvas.size[1]})")


cols = all_names[:NCOL]
grid(cols, f"{OUT}/warmup5_samples.png")
grid(cols[:4], f"{OUT}/warmup5_samples_zoom.png", crop=(64, 64, 128))
json.dump({"fid_dir": FID, "n_paired": len(all_names), "columns": cols,
           "mean_abs_delta_vs_fp16_255": DELTA, "w8a8_vs_w4a4_255": cross,
           "config": "DDIM 50, batch 16, dynamic delta, MODIFF_LINEAR=0 (conv-only MoDiff), "
                     "MODIFF_WARMUP_STEPS=5, real-checkpoint calibration, seed0=20260805"},
          open(f"{OUT}/warmup5_samples.json", "w"), indent=1)
print("OK")
