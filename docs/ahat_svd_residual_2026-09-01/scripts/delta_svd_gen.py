"""Quality probe: per-step SVD of the temporal delta on three residual convs.

Drop the orthogonal tail (error feedback). Both a_hat and o_hat see Q(delta_k).
Existing C-channel INT8 conv (no speed claim — Python SVD + lifted codes).

k=64 is the operating point from the offline proxy; k=16 is the 'too small' control.

  python docs/ahat_svd_residual_2026-09-01/scripts/delta_svd_gen.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [
    ROOT, os.path.join(ROOT, "src/taming-transformers"),
    os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts"),
]

os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
os.environ["MODIFF_QUANT_LINEAR"] = "1"
os.environ["MODIFF_QUANT_ATTN"] = "1"
os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
os.environ["MODIFF_QATTN_FLASH"] = "1"
os.environ["MODIFF_FLASH_GATE"] = "on"
os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
os.environ["MODIFF_WARMUP_STEPS"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import integration.benchmarks.benchmark_ldm as B
import integration.kernels.int8_optimized as i8
import ahat_fake_quant_grid as G

OUT_DIR = "docs/ahat_svd_residual_2026-09-01"
OUT_PNG = f"{OUT_DIR}/plots/delta_svd_grid.png"
OUT_JSON = f"{OUT_DIR}/data/delta_svd_gen.json"
SHAPE = (4, 32, 32)
SEED = 20260805
NQ = 4
STEPS = 50
TARGET = (
    "input_blocks.1.0.in_conv",
    "input_blocks.4.0.in_conv",
    "input_blocks.4.0.out_conv",
)
K_ARMS = (64, 16)

ORIG = None
STATE = {"k": 64, "svd": True, "hits": 0, "layers": set()}


def target_activation(self, x, gn_weight, gn_bias, num_groups, eps, ms2d, sh2d):
    xf = x.float()
    n = F.group_norm(xf, num_groups, gn_weight.float(), gn_bias.float(), eps)
    if ms2d is not None and ms2d.numel() > 0:
        n = n * (1.0 + ms2d.float().view(x.shape[0], x.shape[1], 1, 1))
        n = n + sh2d.float().view(x.shape[0], x.shape[1], 1, 1)
    o = F.silu(n.half().float())
    if (not self._smooth_is_identity and getattr(self, "_smooth_inv_flat", None) is not None
            and self._smooth_inv_flat.numel() > 0):
        o = o * self._smooth_inv_flat.float().view(1, -1, 1, 1)
    return o


def project_delta(d: torch.Tensor, k: int) -> torch.Tensor:
    """Shared channel subspace over the batch: d_k = U U^T d, U is C×k."""
    n, c, h, w = d.shape
    k = min(k, c, n * h * w)
    m = d.permute(1, 0, 2, 3).reshape(c, -1)
    u, _, _ = torch.linalg.svd(m, full_matrices=False)
    u = u[:, :k]
    z = torch.einsum("ck,nchw->nkhw", u, d)
    return torch.einsum("ck,nkhw->nchw", u, z)


def patched_gn_fused(self, x, gn_weight, gn_bias, num_groups, eps,
                     mod_scale2d, mod_shift2d, residual=None):
    name = self.layer_name or ""
    if name not in TARGET:
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    self.step_count += 1
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    self._ensure_state_buffers(x)
    with torch.no_grad():
        tgt = target_activation(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d)
        d = tgt - self.a_hat_cache.float()
        d_k = project_delta(d, STATE["k"]) if STATE["svd"] else d
        amax = d_k.abs().amax().clamp_min(1e-8)
        scale = 127.0 / amax
        q = torch.clamp(torch.round(d_k * scale), -127, 127)
        deq = q / scale
        if self._write_ahat_now():
            self.a_hat_cache.add_(deq.half())
        x_int8 = q.to(torch.int8).contiguous(memory_format=torch.channels_last)
        alpha = getattr(self, "_dsvd_alpha", None)
        if alpha is None or alpha.device != x.device:
            self._dsvd_alpha = torch.empty(1, device=x.device, dtype=torch.float32)
            alpha = self._dsvd_alpha
        alpha.fill_(float(1.0 / scale))
    STATE["hits"] += 1
    STATE["layers"].add(name)
    if residual is not None:
        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
        out = self._layer_out_buf()
        self._evt_ohat_residual(x_int8, alpha, residual, out)
        return self._after_ahat_write(out)
    out = self._evt_ohat(x_int8, alpha)
    return self._after_ahat_write(out)


def install(k: int, svd: bool = True):
    global ORIG
    STATE["k"] = k
    STATE["svd"] = svd
    STATE["hits"] = 0
    STATE["layers"] = set()
    if ORIG is None:
        ORIG = i8.OptimizedInt8Conv2d.forward_gn_fused_modiff
    i8.OptimizedInt8Conv2d.forward_gn_fused_modiff = patched_gn_fused


def uninstall():
    if ORIG is not None:
        i8.OptimizedInt8Conv2d.forward_gn_fused_modiff = ORIG


def reset_int8(model):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def gen_lat(model, sampler, quantized, seed=SEED):
    if quantized:
        reset_int8(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=NQ, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def relL2(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = float(b.norm())
    return float((a - b).norm() / denom) if denom > 0 else float("nan")


def grid(path, images, cell=256, pad=12, lab=48):
    nq = images[0][2].shape[0]
    rows = len(images)
    W = pad + nq * (cell + pad)
    H = pad + rows * (cell + lab + pad)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = font_sm = ImageFont.load_default()
    y = pad
    for title, sub, arr in images:
        dr.text((pad, y + 4), title, fill=(11, 11, 11), font=font)
        dr.text((pad, y + 24), sub, fill=(70, 70, 70), font=font_sm)
        y += lab
        for i in range(min(nq, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path, "PNG")
    print("wrote", path, flush=True)


def setup(mode):
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{OUT_DIR}/tmp_dsvd",
        batch_size=NQ, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def main():
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    lats, images, rows = {}, [], []

    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    lat = gen_lat(model, sampler, quantized=False)
    lats["fp16"] = lat
    images.append(("fp16  S=50", "reference", G.decode(model, lat)))
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 MoDiff =====", flush=True)
    model, sampler = setup("int8")
    uninstall()
    lat = gen_lat(model, sampler, quantized=True)
    lats["A"] = lat
    r_fp = relL2(lat, lats["fp16"])
    rows.append({"arm": "A", "k": None, "svd": False, "relL2_vs_fp16": r_fp, "relL2_vs_A": 0.0,
                 "hits": 0, "layers": []})
    print(f"  A  relL2 vs fp16 {r_fp:.4f}", flush=True)
    images.append(("A  full MoDiff  S=50", f"relL2 vs fp16 {r_fp:.3f}",
                   G.decode(model, lat)))

    extra = [
        ("py_full", None, False, "A  Python GN+absmax  (no SVD)"),
        ("k64", 64, True, "A + delta-SVD k=64  (3 layers)"),
        ("k16", 16, True, "A + delta-SVD k=16  (3 layers)"),
    ]
    for aid, k, svd, title in extra:
        install(k if k else 64, svd=svd)
        lat = gen_lat(model, sampler, quantized=True)
        lats[aid] = lat
        r_fp = relL2(lat, lats["fp16"])
        r_a = relL2(lat, lats["A"])
        rec = {"arm": aid, "k": k, "svd": svd, "relL2_vs_fp16": r_fp,
               "relL2_vs_A": r_a, "hits": STATE["hits"],
               "layers": sorted(STATE["layers"])}
        rows.append(rec)
        print(f"  {aid}  svd={svd} k={k}  hits={STATE['hits']}\n"
              f"       relL2 vs fp16 {r_fp:.4f}  vs A {r_a:.4f}", flush=True)
        images.append((title,
                       f"vs fp16 {r_fp:.3f}   vs A {r_a:.3f}   hits {STATE['hits']}",
                       G.decode(model, lat)))
    uninstall()
    del model, sampler
    torch.cuda.empty_cache()

    grid(OUT_PNG, images)
    payload = {
        "seed": SEED, "n": NQ, "steps": STEPS, "target_layers": list(TARGET),
        "note": "Python SVD on delta, drop tail, Q(d_k) through existing INT8 conv. Quality only.",
        "rows": rows, "png": OUT_PNG,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
