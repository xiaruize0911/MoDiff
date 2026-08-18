"""Does W4A4+MoDiff recover at the paper's step count? 50 vs 200 vs 500 on our real int4 datapath.

Each arm is scored against the fp16 reference AT ITS OWN STEP COUNT. Scoring a 500-step arm against a
50-step reference would fold the schedule change into mean|delta| and report it as quantization error --
the same error class as grading an EMA arm against a non-EMA reference (paper_repro FINDINGS section 9).

Two non-vacuity checks, because a flag that silently does not apply is this project's recurring failure:
  * fp16 must DIFFER across step counts, else --steps never took effect;
  * each quantized arm must differ from fp16 at its own step count, else the arm never engaged.
Both test that files EXIST before comparing them.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIRS = {50: "/workspace/fid_warmup5", 200: "/workspace/fid_warmup5_s200",
        500: "/workspace/fid_warmup5_s500"}
STEPS = [50, 200, 500]
ARMS = [("int8_modiff_l0", "W8A8 + MoDiff"), ("int4_modiff_l0", "W4A4 + MoDiff")]
OUT = os.path.dirname(os.path.abspath(__file__))
TILE, PAD, LABEL_W, HEADER = 224, 6, 230, 26
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def names(s, folder):
    d = os.path.join(DIRS[s], folder)
    assert os.path.isdir(d), f"{d} missing -- nothing generated for {folder} at {s} steps"
    f = sorted(x for x in os.listdir(d) if x.endswith(".png"))
    assert f, f"{d} holds no PNG"
    return f


def arr(s, folder, nm):
    p = os.path.join(DIRS[s], folder, nm)
    assert os.path.isfile(p), f"missing {p}"
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def delta(sa, fa, sb, fb, ns):
    tot = npix = 0
    for nm in ns:
        a, b = arr(sa, fa, nm), arr(sb, fb, nm)
        tot += float(np.abs(a - b).sum())
        npix += a.size
    return tot / npix


cols_all = names(50, "fp16")
for s in STEPS:
    for folder, _ in [("fp16", "fp16")] + ARMS:
        have = set(names(s, folder))
        miss = [c for c in cols_all if c not in have]
        assert not miss, f"{s} steps / {folder} missing {miss}"
print(f"{len(cols_all)} paired images per (arm, step count)")

# --- did --steps actually take effect? -------------------------------------------------------------
sched = {(a, b): delta(a, "fp16", b, "fp16", cols_all) for a, b in [(50, 200), (200, 500), (50, 500)]}
for (a, b), d in sched.items():
    print(f"  fp16 {a} vs {b} steps: {d:.3f}/255")
    assert d > 0.0, f"fp16 is byte-identical at {a} and {b} steps -- --steps never took effect"

# --- each arm vs fp16 at its OWN step count -------------------------------------------------------
D = {}
for s in STEPS:
    for folder, label in ARMS:
        D[(s, folder)] = delta(s, "fp16", s, folder, cols_all)
        assert D[(s, folder)] > 0.0, f"{folder} identical to fp16 at {s} steps -- arm did not engage"

print("\n  steps |  W8A8+MoDiff  |  W4A4+MoDiff   (mean |delta| vs fp16 at the SAME step count)")
for s in STEPS:
    print(f"   {s:4d} |    {D[(s,'int8_modiff_l0')]:6.3f}     |    {D[(s,'int4_modiff_l0')]:6.3f}")

fnt_b = ImageFont.truetype(FONT_B, 16)
fnt_s = ImageFont.truetype(FONT, 12)


def grid(rows, cols, out_png, crop=None):
    """rows: list of (step, folder, bold_label, sub_label)"""
    W = LABEL_W + len(cols) * (TILE + PAD) + PAD
    H = HEADER + len(rows) * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for j, nm in enumerate(cols):
        dr.text((LABEL_W + j * (TILE + PAD), 6), f"#{int(nm.split('.')[0])}", font=fnt_s,
                fill=(90, 90, 90))
    for i, (s, folder, lab, sub) in enumerate(rows):
        y = HEADER + i * (TILE + PAD)
        dr.text((PAD, y + 4), lab, font=fnt_b, fill=(0, 0, 0))
        dr.text((PAD, y + 26), sub, font=fnt_s, fill=(110, 110, 110))
        for j, nm in enumerate(cols):
            im = Image.open(os.path.join(DIRS[s], folder, nm)).convert("RGB")
            if crop:
                cx, cy, side = crop
                im = im.crop((cx, cy, cx + side, cy + side))
            canvas.paste(im.resize((TILE, TILE), Image.LANCZOS), (LABEL_W + j * (TILE + PAD), y))
    canvas.save(out_png)
    print(f"wrote {out_png}  ({canvas.size[0]}x{canvas.size[1]})")


cols = cols_all[:6]
w4 = [(500, "fp16", "fp16 @ 500", "reference for the bottom row")]
w4 += [(s, "int4_modiff_l0", f"W4A4+MoDiff @ {s}",
        f"mean |Δ| vs fp16@{s}   {D[(s,'int4_modiff_l0')]:.2f}/255") for s in STEPS]
grid(w4, cols, f"{OUT}/steps_w4a4.png")

ctl = [(500, "fp16", "fp16 @ 500", "reference")]
ctl += [(s, "int8_modiff_l0", f"W8A8+MoDiff @ {s}",
         f"mean |Δ| vs fp16@{s}   {D[(s,'int8_modiff_l0')]:.2f}/255") for s in STEPS]
grid(ctl, cols, f"{OUT}/steps_w8a8_control.png")

json.dump({"dirs": DIRS, "n_paired": len(cols_all), "columns": cols,
           "mean_abs_delta_vs_fp16_same_steps_255":
               {f"{s}_{f}": D[(s, f)] for s in STEPS for f, _ in ARMS},
           "fp16_schedule_delta_255": {f"{a}_vs_{b}": d for (a, b), d in sched.items()},
           "config": "batch 16, dynamic delta, MODIFF_LINEAR=0, MODIFF_WARMUP_STEPS=5, "
                     "real-checkpoint calibration, seed0=20260805"},
          open(f"{OUT}/steps_sweep.json", "w"), indent=1)
print("OK")
