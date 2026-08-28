"""Replay residual: skip GN+conv, read frozen o_hat [+ live ResBlock skip].

Commit every K steps (t=T always). Skip steps do not launch quantize/conv.
Env: MODIFF_REPLAY_K. Orthogonal to MODIFF_CACHE_SKIP_K (leave that at 1).

Run: python docs/residual_replay_2026-08-27/scripts/replay_gen.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402
import ahat_fake_quant_grid as G                                           # noqa: E402
import integration.kernels.int8_optimized as i8opt                         # noqa: E402
import integration.kernels.int4_optimized as i4opt                         # noqa: E402

OUT = "docs/residual_replay_2026-08-27/plots/replay_grid.png"
JSON_OUT = "docs/residual_replay_2026-08-27/data/replay_gen.json"
MODE_TAG = {"int8": "W8A8", "int4": "W4A4"}
SHAPE = (4, 32, 32)

_STATS = {"replay": 0, "compute": 0}
_ORIG = {}


def _wrap_replay(orig):
    def wrapped(self):
        r = orig(self)
        if r:
            _STATS["replay"] += 1
        elif self.step_count > 0:
            _STATS["compute"] += 1
        return r
    return wrapped


def install_counters():
    uninstall_counters()
    _STATS["replay"] = _STATS["compute"] = 0
    _ORIG["i8"] = i8opt.OptimizedInt8Conv2d._replay_residual
    _ORIG["i4"] = i4opt.OptimizedInt4Conv2d._replay_residual
    i8opt.OptimizedInt8Conv2d._replay_residual = _wrap_replay(_ORIG["i8"])
    i4opt.OptimizedInt4Conv2d._replay_residual = _wrap_replay(_ORIG["i4"])


def uninstall_counters():
    if "i8" in _ORIG:
        i8opt.OptimizedInt8Conv2d._replay_residual = _ORIG["i8"]
    if "i4" in _ORIG:
        i4opt.OptimizedInt4Conv2d._replay_residual = _ORIG["i4"]
    _ORIG.clear()


def _reset(model, mode):
    unet = model.model.diffusion_model
    if "int8" in mode:
        B.reset_modiff_state_int8(unet)
    elif "int4" in mode:
        B.reset_modiff_state_int4(unet)
    B._reset_wxax_modiff_safe(model)


def sample(model, sampler, steps, batch, seed, mode):
    _reset(model, mode)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=batch, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--ks", type=str, default="1,2,4,8")
    ap.add_argument("--modes", type=str, default="int8,int4")
    a = ap.parse_args()
    ks = tuple(int(x) for x in a.ks.split(",") if x.strip())
    modes = tuple(m.strip() for m in a.modes.split(",") if m.strip())
    os.environ.setdefault("MODIFF_DELTA_MODE", "static")
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"

    print(f"GPU: {torch.cuda.get_device_name()}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/residual_replay_2026-08-27/tmp_gen",
        batch_size=a.n, steps=a.steps, shape=SHAPE, calibration_path=None,
        auto_delta_table=True)

    install_counters()
    rows, quality, ref = [], {}, None

    def add_arm(label, lat, extra=None):
        nonlocal ref
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        rec = {"relL2_vs_fp16": rel}
        if extra:
            rec.update(extra)
        quality[label] = rec
        print(f"  {label:28s} relL2 {rel:.4f}"
              + (f"  compute {extra['n_compute']} replay {extra['n_replay']}" if extra else ""),
              flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, G.decode(model, lat)))

    print("===== fp16 =====", flush=True)
    os.environ["MODIFF_REPLAY_K"] = "1"
    model, sampler = runner._setup_model("fp16")
    sample(model, sampler, a.steps, a.n, a.seed, "fp16")
    lat = sample(model, sampler, a.steps, a.n, a.seed, "fp16")
    add_arm("fp16 reference", lat)
    del model, sampler
    torch.cuda.empty_cache()

    for mode in modes:
        tag = MODE_TAG[mode]
        print(f"===== {mode} =====", flush=True)
        runner.calibration_path = B._default_calibration_path(mode)
        model, sampler = runner._setup_model(mode)
        for k in ks:
            os.environ["MODIFF_REPLAY_K"] = str(k)
            _STATS["replay"] = _STATS["compute"] = 0
            sample(model, sampler, a.steps, a.n, a.seed, mode)
            _STATS["replay"] = _STATS["compute"] = 0
            lat = sample(model, sampler, a.steps, a.n, a.seed, mode)
            extra = {"replay_k": k, "n_compute": _STATS["compute"], "n_replay": _STATS["replay"]}
            add_arm(f"{tag} replay-K={k}", lat, extra)
        os.environ["MODIFF_REPLAY_K"] = "1"
        del model, sampler
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
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "K": list(ks),
               "modes": list(modes), "relL2": quality, "out": OUT},
              open(JSON_OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  ({W}x{Hh})\nwrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        uninstall_counters()
        os.environ["MODIFF_REPLAY_K"] = "1"
