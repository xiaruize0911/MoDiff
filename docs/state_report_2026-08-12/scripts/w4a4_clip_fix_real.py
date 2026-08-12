"""Land the clip fix on the REAL int4 kernels, and render it.

The activation-grid sweep (w4a4_quantizer_fix.py, act-only fake quant, 3 seeds) separated the two
available fixes:

    shipped symmetric grid                     0.9294
    symmetric + clip x0.15                     0.4519    2.06x, PURE CALIBRATION
    asymmetric + clip x0.25                    0.3668    2.53x, but needs a kernel zero point

Most of it is the clipping, and clipping is free: the scale is just a number in the calibration
file, so it can be shipped and tested end to end without touching a kernel. That is what this does.

WHY CLIPPING HELPS SO MUCH AT 4 BITS. probe_int4_code_use.py measured the shipped grid using 5 of
its 15 codes for >0.1% of the mass, on an activation whose |max|/|min| is 20x. Sizing the grid to
the positive tail spends the resolution on outliers; shrinking it by 0.15 concentrates all 15 codes
on the bulk and lets the tail saturate. The shipped absmax file has been pulling this lever by
ACCIDENT all along -- its scale is 5.13x too large for unsmoothed input, which is why it beat the
correctly-sized qdiff scale 0.71 to 0.86.

The ratio is re-swept here rather than assumed: the act-only optimum was found with weights in
fp16, and 4-bit weights add their own error, which can move it.

Run: python docs/state_report_2026-08-12/scripts/w4a4_clip_fix_real.py    # ~18 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

RANGES = "docs/state_report_2026-08-12/data/int4_code_use.json"
SHIPPED = "integration/calibration/int4_calibration_qdiff.pt"
TMP = "docs/state_report_2026-08-12/data/int4_calibration_clip{}.pt"
OUT = "docs/state_report_2026-08-12/data/w4a4_clip_fix_real.json"
GRID = "docs/state_report_2026-08-12/plots/w4a4_clip_fix.png"
SEEDS = [1234, 20260805, 777]
Q = 7.0
#: EXTENDED DOWNWARD after the first pass: PTQ was still falling monotonically at 0.15
#: (0.25 -> 0.9970, 0.20 -> 0.6865, 0.15 -> 0.4931), i.e. the sweep had not bracketed the
#: minimum and 0.15 was only "best of what was tried". The act-only fake-quant curve turned
#: back up by 0.09, so the optimum should be in here somewhere.
RATIOS = [0.15, 0.12, 0.10, 0.08]


def clipped_file(ratio, ranges):
    """scale = Q / (observed_absmax * ratio). Bare floats, the 7-based int4 convention."""
    out = {r["layer"]: Q / (max(abs(r["vmin"]), abs(r["vmax"])) * ratio) for r in ranges}
    p = TMP.format(str(ratio).replace(".", ""))
    torch.save(out, p)
    return p, out


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
    if not os.path.exists(RANGES):
        print(f"FAIL: missing {RANGES} -- run probe_int4_code_use.py first")
        return 1
    ranges = json.load(open(RANGES))["layers"]
    shipped = torch.load(SHIPPED, map_location="cpu", weights_only=True)
    if {r["layer"] for r in ranges} != set(shipped):
        print("FAIL: the measured ranges do not cover the shipped 70 layers")
        return 1

    H.STEPS, H.BATCH = 50, 6
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"

    print("fp16 reference ...", flush=True)
    rf, mf, sf = H.build("fp16", None, "static")
    refs, ref_img = {}, None
    for sd in SEEDS:
        H.SEED = sd
        H.latent(rf, mf, sf)
        lat, _ = H.latent(rf, mf, sf)
        refs[sd] = lat.float()
        if sd == SEEDS[0]:
            ref_img = decode(mf, lat)
    del rf, mf, sf
    torch.cuda.empty_cache()

    def arm(mode, cal):
        r, m, s = H.build(mode, cal, "static")
        H.SEED = SEEDS[0]
        H.latent(r, m, s)                     # discard: attention self-calibration
        rels, img = [], None
        for sd in SEEDS:
            H.SEED = sd
            H.latent(r, m, s)
            lat, _ = H.latent(r, m, s)
            rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            if sd == SEEDS[0]:
                img = decode(m, lat)
        del r, m, s
        torch.cuda.empty_cache()
        return statistics.mean(rels), rels, img

    out, rows = {}, [("fp16 reference", ref_img)]
    for mode, tag in (("int4_baseline", "PTQ"),):   # PTQ only: MoDiff moved 1.04x
        m, rels, img = arm(mode, SHIPPED)
        out[f"{tag}/shipped"] = {"mean": m, "relL2": rels}
        print(f"  W4A4 {tag:6s} shipped grid      {m:.4f}  {[round(x,3) for x in rels]}", flush=True)
        rows.append((f"W4A4 {tag} — shipped grid    relL2 {m:.4f}", img))
        for ratio in RATIOS:
            p, _ = clipped_file(ratio, ranges)
            m, rels, img = arm(mode, p)
            out[f"{tag}/clip{ratio}"] = {"mean": m, "relL2": rels, "file": p}
            print(f"  W4A4 {tag:6s} clip x{ratio:<5}      {m:.4f}  {[round(x,3) for x in rels]}",
                  flush=True)
            rows.append((f"W4A4 {tag} — clip x{ratio}    relL2 {m:.4f}", img))

    print()
    # Only tags that were actually measured. Iterating a hardcoded pair after narrowing the arm
    # list is a KeyError raised AFTER every sample has been drawn, which discards the whole run.
    for tag in sorted({k.split("/")[0] for k in out}):
        base = out[f"{tag}/shipped"]["mean"]
        best_r = min(RATIOS, key=lambda r: out[f"{tag}/clip{r}"]["mean"])
        best = out[f"{tag}/clip{best_r}"]["mean"]
        print(f"W4A4 {tag:6s}: {base:.4f} -> {best:.4f} at clip x{best_r}  ({base/best:.2f}x)")
        out[f"{tag}/best"] = {"ratio": best_r, "mean": best, "gain": base / best}

    cell, pad, lab = 256, 6, 26
    n = min(6, rows[0][1].shape[0])
    W = pad + n * (cell + pad)
    Hh = len(rows) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for label, arr in rows:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(n):
            canvas.paste(Image.fromarray(arr[i]), (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(GRID), exist_ok=True)
    canvas.save(GRID, "PNG")
    json.dump({"seeds": SEEDS, "ratios": RATIOS, "results": out}, open(OUT, "w"), indent=1)
    print(f"\nwrote {GRID}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
