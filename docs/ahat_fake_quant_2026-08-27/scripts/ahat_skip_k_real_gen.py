"""Real-kernel skip-K generation: freeze a_hat/o_hat for K-1 of every K steps.

The CUDA path still computes a fresh residual and a fresh output every step
(so this step's conv/o_hat math matches fake-quant's `used = a_hat + Q(delta)`).
On a skip step the in-place cache writes are rolled back, so the next residual
is still against the stale checkpoint.

This is a quality experiment, not a bandwidth win (the rollback clones caches).
K=1 is a no-op (identical to shipped MoDiff).

Run: python docs/ahat_fake_quant_2026-08-27/scripts/ahat_skip_k_real_gen.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402
import integration.fused_ops.fused_resblock as FR                          # noqa: E402
import integration.kernels.int8_optimized as i8opt                         # noqa: E402
import integration.kernels.int4_optimized as i4opt                         # noqa: E402
import ahat_fake_quant_grid as G                                           # noqa: E402

OUT_DEFAULT = "docs/ahat_fake_quant_2026-08-27/plots/ahat_skip_k_real_grid.png"
JSON_DEFAULT = "docs/ahat_fake_quant_2026-08-27/data/ahat_skip_k_real.json"
KS_DEFAULT = (1, 2, 4)
MODE_TAG = {"int8": "W8A8", "int4": "W4A4"}

_K = 1
_STATS = {"commit": 0, "skip": 0}
_ORIGINALS = {}


def _snap(conv):
    a = getattr(conv, "a_hat_cache", None)
    o = getattr(conv, "o_hat_cache", None)
    return (None if a is None else a.clone(),
            None if o is None else o.clone(),
            int(getattr(conv, "step_count", 0)))


def _maybe_restore(conv, a_snap, o_snap, sc_before):
    sc = int(getattr(conv, "step_count", 0))
    if _K <= 1 or a_snap is None or sc == sc_before:
        if sc != sc_before:
            _STATS["commit"] += 1
        return
    if sc % _K == 0:
        _STATS["commit"] += 1
        return
    conv.a_hat_cache.copy_(a_snap)
    if o_snap is not None and getattr(conv, "o_hat_cache", None) is not None:
        conv.o_hat_cache.copy_(o_snap)
    _STATS["skip"] += 1


def _detach_if_cache(conv, out):
    # Non-residual MoDiff convs return `self.o_hat_cache`. Restoring that tensor
    # in place would replace this step's `out = o_hat_old + conv` with o_hat_old.
    oh = getattr(conv, "o_hat_cache", None)
    if out is oh:
        return out.clone()
    return out


def _wrap_conv_method(orig):
    def wrapped(self, *args, **kwargs):
        if not getattr(self, "modiff_enabled", False) or _K <= 1:
            return orig(self, *args, **kwargs)
        a_snap, o_snap, sc0 = _snap(self)
        out = orig(self, *args, **kwargs)
        out = _detach_if_cache(self, out)
        _maybe_restore(self, a_snap, o_snap, sc0)
        return out
    return wrapped


def _wrap_resize(orig):
    def wrapped(x, gn, conv, *args, **kwargs):
        if _K <= 1 or conv is None or not getattr(conv, "modiff_enabled", False):
            return orig(x, gn, conv, *args, **kwargs)
        a_snap, o_snap, sc0 = _snap(conv)
        out = orig(x, gn, conv, *args, **kwargs)
        if out is not None:
            out = _detach_if_cache(conv, out)
            _maybe_restore(conv, a_snap, o_snap, sc0)
        return out
    return wrapped


def _wrap_upsample_forward(orig):
    def wrapped(self, x):
        conv = getattr(getattr(self, "orig", None), "conv", None)
        if _K <= 1 or conv is None or not getattr(conv, "modiff_enabled", False):
            return orig(self, x)
        a_snap, o_snap, sc0 = _snap(conv)
        out = orig(self, x)
        out = _detach_if_cache(conv, out)
        _maybe_restore(conv, a_snap, o_snap, sc0)
        return out
    return wrapped


_CONV_METHODS = ("forward", "forward_gn_fused_modiff", "forward_modiff_fused_silu_residual")


def install(k: int):
    global _K
    uninstall()
    _K = int(k)
    _STATS["commit"] = _STATS["skip"] = 0
    for cls, tag in ((i8opt.OptimizedInt8Conv2d, "i8"), (i4opt.OptimizedInt4Conv2d, "i4")):
        for name in _CONV_METHODS:
            key = (tag, name)
            if not hasattr(cls, name):
                continue
            _ORIGINALS[key] = getattr(cls, name)
            setattr(cls, name, _wrap_conv_method(_ORIGINALS[key]))
    _ORIGINALS["resize"] = FR._prequant_gn_resize_conv_modiff
    FR._prequant_gn_resize_conv_modiff = _wrap_resize(_ORIGINALS["resize"])
    _ORIGINALS["upsample"] = FR.FusedUpsample.forward
    FR.FusedUpsample.forward = _wrap_upsample_forward(_ORIGINALS["upsample"])


def uninstall():
    global _K
    _K = 1
    for cls, tag in ((i8opt.OptimizedInt8Conv2d, "i8"), (i4opt.OptimizedInt4Conv2d, "i4")):
        for name in _CONV_METHODS:
            key = (tag, name)
            if key in _ORIGINALS:
                setattr(cls, name, _ORIGINALS[key])
    if "resize" in _ORIGINALS:
        FR._prequant_gn_resize_conv_modiff = _ORIGINALS["resize"]
    if "upsample" in _ORIGINALS:
        FR.FusedUpsample.forward = _ORIGINALS["upsample"]
    _ORIGINALS.clear()


def _reset(model, mode):
    unet = model.model.diffusion_model
    if "int8" in mode:
        B.reset_modiff_state_int8(unet)
        if B.HAS_INT8_LINEAR:
            B.reset_modiff_state_linear(unet)
    elif "int4" in mode:
        B.reset_modiff_state_int4(unet)
        if B.HAS_INT4_LINEAR:
            B.reset_modiff_state_int4_linear(unet)
    B._reset_wxax_modiff_safe(model)


SHAPE = (4, 32, 32)


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
    ap.add_argument("--ks", type=str, default=",".join(str(k) for k in KS_DEFAULT),
                    help="comma-separated skip-K values, e.g. 1,8,16,32")
    ap.add_argument("--modes", type=str, default="int8,int4",
                    help="comma-separated modes: int8, int4")
    ap.add_argument("--out", type=str, default=OUT_DEFAULT)
    ap.add_argument("--json", type=str, default=JSON_DEFAULT)
    ap.add_argument("--no_fp16", action="store_true")
    a = ap.parse_args()
    ks = tuple(int(x) for x in a.ks.split(",") if x.strip())
    modes = tuple(m.strip() for m in a.modes.split(",") if m.strip())
    os.environ.setdefault("MODIFF_DELTA_MODE", "static")
    os.environ["MODIFF_LINEAR"] = "0"

    print(f"GPU: {torch.cuda.get_device_name()}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_fake_quant_2026-08-27/tmp_skip_real",
        batch_size=a.n, steps=a.steps, shape=SHAPE, calibration_path=None,
        auto_delta_table=True)

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
              + (f"  commit {extra['n_commit']} skip {extra['n_skip']}" if extra else ""),
              flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, G.decode(model, lat)))

    if not a.no_fp16:
        print("===== fp16 =====", flush=True)
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
            install(k)
            _STATS["commit"] = _STATS["skip"] = 0
            sample(model, sampler, a.steps, a.n, a.seed, mode)  # warmup
            _STATS["commit"] = _STATS["skip"] = 0
            lat = sample(model, sampler, a.steps, a.n, a.seed, mode)
            extra = {"skip_k": k, "n_commit": _STATS["commit"], "n_skip": _STATS["skip"]}
            add_arm(f"{tag} MoDiff  skip-K={k}", lat, extra)
            uninstall()
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
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    canvas.save(a.out, "PNG")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "K": list(ks),
               "modes": list(modes), "relL2": quality, "out": a.out},
              open(a.json, "w"), indent=1)
    print(f"\nwrote {a.out}  ({W}x{Hh})\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        uninstall()
