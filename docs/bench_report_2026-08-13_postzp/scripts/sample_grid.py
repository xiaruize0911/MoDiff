"""Generation samples, paired across modes, plus the pixel distance that makes "looks the same" a number.

WHY THIS IS A FAIR COMPARISON AND NOT A PICKED ONE. generate_fid_samples.py drives every mode with
THE SAME seed sequence (--seed0 20260805, a different seed per batch, the same sequence per mode), so
/workspace/fid/<mode>/000123.png is the same noise draw in all five folders. Column j of the grid is
therefore one latent rendered five ways, and any difference down a column is the mode. The indices are
the first N in sorted order -- no selection, because with a paired set there is nothing to select for.

WHAT THE NUMBER UNDER EACH ROW IS. mean |pixel - fp16 pixel| over --metric-n paired images, 0-255. It is
not a perceptual metric and is not FID; it exists so the visual claim has an anchor that cannot be
argued with, and so that "W8A8+MoDiff is indistinguishable from fp16" is falsifiable. The fp16 row is 0
by construction and is printed as the control.

PROVENANCE OF THE IMAGES. These are the 2026-08-05 FID run: DDIM 50, batch 128, dynamic delta,
MODIFF_LINEAR=0 (conv-only MoDiff), real-checkpoint calibration. That is NOT the configuration
REPORT.md times (static delta, 200 steps). They are answering different questions -- these show what
comes out, REPORT.md shows what it costs -- and mixing the two configurations in one claim would be
wrong, so the doc keeps them in separate sections.

Run: python docs/bench_report_2026-08-13_postzp/scripts/sample_grid.py
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
D = "docs/bench_report_2026-08-13_postzp"
FID = os.environ.get("FID_DIR", "/workspace/fid")

#: (folder, label) -- fp16 first because every other row is measured against it
MODES = [("fp16", "fp16"),
         ("int8_baseline", "W8A8 PTQ"),
         ("int8_modiff", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"),
         ("int4_modiff", "W4A4 MoDiff")]
NCOL = int(os.environ.get("SAMPLE_COLS", "6"))
METRIC_N = int(os.environ.get("METRIC_N", "500"))
TILE = 224
PAD = 6
LABEL_W = 200
HEADER = 26
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def names(folder, n):
    f = sorted(x for x in os.listdir(os.path.join(FID, folder)) if x.endswith(".png"))
    return f[:n]


def load(folder, name):
    return Image.open(os.path.join(FID, folder, name)).convert("RGB")


def mean_abs_delta(folder, ref_names):
    """mean |x - fp16| in 0-255 over the same paired indices, streamed so memory stays flat."""
    if folder == "fp16":
        return 0.0
    tot, npix = 0.0, 0
    for nm in ref_names:
        a = np.asarray(load("fp16", nm), dtype=np.int16)
        b = np.asarray(load(folder, nm), dtype=np.int16)
        tot += float(np.abs(a - b).sum())
        npix += a.size
    return tot / npix


def grid(cols, out_png, crop=None):
    """crop=(x, y, side) renders a zoom of that box instead of the whole image."""
    fnt = ImageFont.truetype(FONT, 15)
    fnt_b = ImageFont.truetype(FONT_B, 16)
    fnt_s = ImageFont.truetype(FONT, 12)
    W = LABEL_W + len(cols) * (TILE + PAD) + PAD
    H = HEADER + len(MODES) * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for j, nm in enumerate(cols):
        x = LABEL_W + j * (TILE + PAD)
        dr.text((x, 6), f"seed #{int(nm.split('.')[0])}", font=fnt_s, fill=(90, 90, 90))
    for i, (folder, label) in enumerate(MODES):
        y = HEADER + i * (TILE + PAD)
        dr.text((PAD, y + 4), label, font=fnt_b, fill=(0, 0, 0))
        d = DELTA[folder]
        sub = "control (reference)" if folder == "fp16" else f"mean |Δ| vs fp16  {d:.2f}/255"
        dr.text((PAD, y + 26), sub, font=fnt_s, fill=(110, 110, 110))
        for j, nm in enumerate(cols):
            im = load(folder, nm)
            if crop:
                cx, cy, side = crop
                im = im.crop((cx, cy, cx + side, cy + side))
            canvas.paste(im.resize((TILE, TILE), Image.LANCZOS),
                         (LABEL_W + j * (TILE + PAD), y))
    canvas.save(out_png)
    print(f"wrote {out_png}  ({canvas.size[0]}x{canvas.size[1]})")


cols = names("fp16", NCOL)
for folder, _ in MODES:                                  # the pairing is an assumption; check it
    have = set(names(folder, NCOL))
    missing = [c for c in cols if c not in have]
    assert not missing, f"{folder} is missing {missing} -- the columns would not be paired"

metric_names = names("fp16", METRIC_N)
DELTA = {}
for folder, label in MODES:
    DELTA[folder] = mean_abs_delta(folder, metric_names)
    print(f"{label:14} mean |Δ| vs fp16 over {len(metric_names)} paired images: {DELTA[folder]:.3f}/255")
assert DELTA["fp16"] == 0.0, "the fp16 control must be exactly 0 -- otherwise the pairing is broken"

os.makedirs(f"{D}/plots", exist_ok=True)
grid(cols, f"{D}/plots/06_samples.png")
grid(cols[:4], f"{D}/plots/07_samples_zoom.png", crop=(64, 64, 128))

json.dump({"fid_dir": FID, "columns": cols, "metric_n": len(metric_names),
           "mean_abs_delta_vs_fp16_255": DELTA,
           "provenance": "2026-08-05 FID run: DDIM 50, batch 128, dynamic delta, MODIFF_LINEAR=0, "
                         "real-checkpoint calibration"},
          open(f"{D}/data/samples.json", "w"), indent=1)
print(f"wrote {D}/data/samples.json")
