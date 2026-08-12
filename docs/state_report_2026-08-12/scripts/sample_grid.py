"""Decoded samples for the five shipped modes, plus the latent relL2 that goes with them.

Everything here resolves from the SHIPPED DEFAULTS -- no hand-passed calibration paths, no
MODIFF_DELTA_MODE override -- so the grid shows what `benchmark_ldm.py --mode X` actually produces
today. That is the point: a sample grid built from explicitly-passed files documents a configuration
nobody runs.

Images and numbers come from ONE process and one seed per column, so a column is the same noise
sampled five ways and the relL2 column grades the image directly above it.

Run: python docs/state_report_2026-08-12/scripts/sample_grid.py    # ~8 min, needs the GPU
"""
import argparse
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

OUT = "docs/state_report_2026-08-12/plots/samples.png"
JSON = "docs/state_report_2026-08-12/data/samples_quality.json"
#: (row label, mode). Delta mode comes from the shipped default for the MoDiff arms; the PTQ
#: baselines have no modulated steps so the setting cannot reach them.
ARMS = [("fp16 reference", "fp16"),
        ("W8A8 PTQ  (int8_baseline)", "int8_baseline"),
        ("W8A8 MoDiff  (int8)", "int8"),
        ("W4A4 PTQ  (int4_baseline)", "int4_baseline"),
        ("W4A4 MoDiff  (int4)", "int4")]


def decode(model, lat, chunk=8):
    """Latent -> uint8 RGB. Chunked, and under the SAME autocast the sampler ran in: the VAE holds
    fp32 params, so an fp16 latent outside that context raises on the bias dtype."""
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
    ap.add_argument("--n", type=int, default=6, help="columns, i.e. samples per mode")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    H.STEPS, H.BATCH, H.SEED = a.steps, a.n, a.seed
    H.AUTO_DELTA_TABLE = True          # the shipped default; dynamic_delta_ab keeps it off for itself

    rows, quality, ref = [], {}, None
    for label, mode in ARMS:
        cal = B._default_calibration_path(mode)
        dm = os.environ.get("MODIFF_DELTA_MODE", "static")
        print(f"=== {label}   calib={os.path.basename(cal) if cal else '-'}  delta={dm}", flush=True)
        r, m, s = H.build(mode, cal, dm)
        H.SEED = a.seed
        H.latent(r, m, s)                       # discard: attention self-calibration + clock ramp
        H.SEED = a.seed
        lat, ms = H.latent(r, m, s)
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        quality[mode] = {"label": label, "relL2_vs_fp16": rel, "ms_per_step": ms,
                         "calibration": cal, "delta_mode": dm if "baseline" not in mode
                         and mode != "fp16" else None}
        print(f"    relL2 vs fp16 {rel:.4f}   {ms:.2f} ms/step", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, decode(m, lat)))
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
    print(f"\nwrote {OUT}  ({W}x{Hh})")

    os.makedirs(os.path.dirname(JSON), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "modes": quality}, open(JSON, "w"), indent=1)
    print(f"wrote {JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
