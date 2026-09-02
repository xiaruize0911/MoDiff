"""Fused range-finder k-conv: native GN + delta_lowrank_fprop CUDA op + Cin=k INT8.

  python docs/ahat_svd_residual_2026-09-01/scripts/delta_svd_kconv_fused.py
"""
from __future__ import annotations

import json
import os
import statistics
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
from PIL import Image, ImageDraw, ImageFont

import integration.benchmarks.benchmark_ldm as B
import integration.kernels.int8_optimized as i8
import ahat_fake_quant_grid as G

OUT_DIR = "docs/ahat_svd_residual_2026-09-01"
OUT_PNG = f"{OUT_DIR}/plots/delta_svd_kconv_fused_grid.png"
OUT_JSON = f"{OUT_DIR}/data/delta_svd_kconv_fused.json"
SHAPE = (4, 32, 32)
SEED = 20260805
NQ = 4
STEPS = 50
BATCH = 128
THREE = (
    "input_blocks.1.0.in_conv",
    "input_blocks.4.0.in_conv",
    "input_blocks.4.0.out_conv",
)
RETAIN = (0.25, 0.33, 0.50)

ORIG = None
STATE = {"scope": "three", "retain": 0.33, "hits": 0, "layers": {}, "fail": 0}
CL = torch.channels_last
E16 = None
E32 = None


def align16(k: int, c: int) -> int:
    if c < 16:
        return c
    k = max(16, min(c, int(round(k / 16.0) * 16)))
    return c if k > c else k


def eligible(mod) -> bool:
    if not isinstance(mod, i8.OptimizedInt8Conv2d):
        return False
    if not getattr(mod, "modiff_enabled", False) or getattr(mod, "groups", 1) != 1:
        return False
    ks = mod.kernel_size
    if ks != (3, 3) and ks != 3:
        return False
    if mod.in_channels < 32:
        return False
    name = mod.layer_name or ""
    if STATE["scope"] == "three":
        return name in THREE
    return True


def k_for(mod) -> int:
    return align16(int(STATE["retain"] * mod.in_channels), mod.in_channels)


def empty_mod(x):
    global E16, E32
    if x.dtype == torch.float16:
        if E16 is None or E16.device != x.device:
            E16 = torch.empty(0, device=x.device, dtype=torch.float16)
        return E16
    if E32 is None or E32.device != x.device:
        E32 = torch.empty(0, device=x.device, dtype=torch.float32)
    return E32


def evt_k(self, x_q, alpha, w_k, wscale, residual=None):
    mc = i8.modiff_cutlass
    strides = (self.stride[0], self.stride[1], self.padding[0], self.padding[1],
               self.dilation[0], self.dilation[1])
    if residual is not None:
        residual = residual.to(torch.float16).contiguous(memory_format=CL)
        out = self._layer_out_buf()
        skip = (self._skip_cache_store()
                and hasattr(mc, "conv2d_int8_evt_o_hat_residual_skip"))
        fn = (mc.conv2d_int8_evt_o_hat_residual_skip if skip
              else mc.conv2d_int8_evt_o_hat_residual)
        fn(x_q, w_k, alpha, wscale, self.o_hat_cache, residual, out, *strides)
        return out
    if self._skip_cache_store():
        out = self._skip_out_buf()
        mc.conv2d_int8_evt_o_hat_skip(
            x_q, w_k, alpha, wscale, self.o_hat_cache, out, *strides)
        return out
    mc.conv2d_int8_evt_o_hat(
        x_q, w_k, alpha, wscale, self.o_hat_cache, *strides)
    return self._module_output()


def w_fp_cached(self):
    buf = getattr(self, "_dsvd_wfp", None)
    if buf is None or buf.device != self.weight_int8.device:
        self._dsvd_wfp = self.dequantized_weight().contiguous()
    return self._dsvd_wfp


def patched_gn_fused(self, x, gn_weight, gn_bias, num_groups, eps,
                     mod_scale2d, mod_shift2d, residual=None):
    if not eligible(self):
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    self.step_count += 1
    if not x.is_contiguous(memory_format=CL):
        x = x.contiguous(memory_format=CL)
    self._ensure_state_buffers(x)
    k = k_for(self)
    if k >= self.in_channels:
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    mc = i8.modiff_cutlass
    ms = mod_scale2d if mod_scale2d is not None and mod_scale2d.numel() > 0 else empty_mod(x)
    sh = mod_shift2d if mod_shift2d is not None and mod_shift2d.numel() > 0 else empty_mod(x)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        tgt = mc.group_norm_silu_nhwc(
            x, gn_weight, gn_bias, int(num_groups), float(eps), True, ms, sh)
        if (not self._smooth_is_identity
                and getattr(self, "_smooth_inv", None) is not None):
            tgt = tgt * self._smooth_inv.to(tgt.dtype)
        d = (tgt.half() - self.a_hat_cache.half()).contiguous(memory_format=CL)
        z_int8, w_k, wscale, alpha = mc.delta_lowrank_fprop(
            d, w_fp_cached(self), self.a_hat_cache, int(k))
    STATE["hits"] += 1
    STATE["layers"][self.layer_name] = {
        "Cin": int(self.in_channels), "k": int(z_int8.shape[1]),
        "retain": round(z_int8.shape[1] / self.in_channels, 4),
    }
    return self._after_ahat_write(evt_k(self, z_int8, alpha, w_k, wscale, residual))


def install(scope: str, retain: float):
    global ORIG
    STATE["scope"] = scope
    STATE["retain"] = retain
    STATE["hits"] = 0
    STATE["layers"] = {}
    if ORIG is None:
        ORIG = i8.OptimizedInt8Conv2d.forward_gn_fused_modiff
    i8.OptimizedInt8Conv2d.forward_gn_fused_modiff = patched_gn_fused


def uninstall():
    if ORIG is not None:
        i8.OptimizedInt8Conv2d.forward_gn_fused_modiff = ORIG


def reset_int8(model):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def time_n(model, sampler, steps=STEPS, n=2, warm=1, quantized=True):
    def once():
        if quantized:
            reset_int8(model)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=steps, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    for _ in range(warm):
        once()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        once()
        e.record()
        torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / steps)
    return statistics.median(xs), xs


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
        output_dir=f"{OUT_DIR}/tmp_kconv_fused",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def isolated():
    mc = i8.modiff_cutlass
    n = BATCH
    rows = []
    for cin, cout, hw, k in [(192, 192, 32, 64), (192, 192, 16, 64), (384, 384, 16, 64)]:
        d = torch.randn(n, cin, hw, hw, device="cuda", dtype=torch.float16).contiguous(memory_format=CL)
        a = torch.zeros_like(d)
        w = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32)
        def go():
            a.zero_()
            mc.delta_lowrank_fprop(d, w, a, k)
        for _ in range(8):
            go()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(20):
            go()
        e.record()
        torch.cuda.synchronize()
        tax = s.elapsed_time(e) / 20
        xq = torch.randint(-8, 8, (n, k, hw, hw), device="cuda", dtype=torch.int8).contiguous(memory_format=CL)
        wk = torch.randint(-8, 8, (cout, 3, 3, k), device="cuda", dtype=torch.int8).contiguous()
        inv = torch.tensor([1.0 / 16.0], device="cuda")
        wsc = torch.full((cout,), 0.02, device="cuda")
        ohat = (0.1 * torch.randn(n, cout, hw, hw, device="cuda", dtype=torch.float16)).contiguous(memory_format=CL)
        xf = torch.randint(-8, 8, (n, cin, hw, hw), device="cuda", dtype=torch.int8).contiguous(memory_format=CL)
        wf = torch.randint(-8, 8, (cout, 3, 3, cin), device="cuda", dtype=torch.int8).contiguous()
        ohatf = ohat.clone()
        def ck():
            mc.conv2d_int8_evt_o_hat(xq, wk, inv, wsc, ohat, 1, 1, 1, 1, 1, 1)
        def cf():
            mc.conv2d_int8_evt_o_hat(xf, wf, inv, wsc, ohatf, 1, 1, 1, 1, 1, 1)
        for _ in range(8):
            ck(); cf()
        torch.cuda.synchronize()
        s.record()
        for _ in range(40):
            ck()
        e.record()
        torch.cuda.synchronize()
        ms_k = s.elapsed_time(e) / 40
        s.record()
        for _ in range(40):
            cf()
        e.record()
        torch.cuda.synchronize()
        ms_f = s.elapsed_time(e) / 40
        rec = {"Cin": cin, "Cout": cout, "HW": hw, "k": k,
               "fused_tax_ms": round(tax, 4),
               "conv_k_ms": round(ms_k, 4), "conv_full_ms": round(ms_f, 4),
               "conv_save_ms": round(ms_f - ms_k, 4),
               "net_ms": round(tax + ms_k - ms_f, 4)}
        rows.append(rec)
        print(f"  {cout}@{hw} k={k}  tax {tax:.3f}  conv {ms_f:.3f}->{ms_k:.3f}  "
              f"net {rec['net_ms']:+.3f} ms", flush=True)
    return rows


def main():
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    assert hasattr(i8.modiff_cutlass, "delta_lowrank_fprop"), "rebuild modiff_cutlass"

    print("===== fused tax vs conv =====", flush=True)
    micro = isolated()

    lats, images, rows = {}, [], []
    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    lat = gen_lat(model, sampler, quantized=False)
    lats["fp16"] = lat
    images.append(("fp16  S=50", "reference", G.decode(model, lat)))
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 =====", flush=True)
    model, sampler = setup("int8")
    uninstall()
    ms_a, trials_a = time_n(model, sampler, quantized=True)
    lat = gen_lat(model, sampler, quantized=True)
    lats["A"] = lat
    r_fp = relL2(lat, lats["fp16"])
    rows.append({"arm": "A", "scope": None, "retain": None, "relL2_vs_fp16": r_fp,
                 "relL2_vs_A": 0.0, "ms_step": ms_a, "trials": trials_a, "vs_A": 1.0,
                 "hits": 0, "n_layers": 0})
    print(f"  A  {ms_a:.2f} ms/step  relL2 {r_fp:.4f}", flush=True)
    images.append(("A  full MoDiff", f"{ms_a:.1f} ms/step  relL2 {r_fp:.3f}",
                   G.decode(model, lat)))

    extra = []
    for frac in RETAIN:
        extra.append((f"three_f{frac:.2f}", "three", frac, f"3 layers fused retain {frac:.0%}"))
    extra.append(("all_f0.33", "all", 0.33, "all 3x3 fused retain 33%"))

    for aid, scope, retain, title in extra:
        install(scope, retain)
        print(f"----- {aid} -----", flush=True)
        try:
            ms, trials = time_n(model, sampler, quantized=True)
            STATE["hits"] = 0
            STATE["layers"] = {}
            lat = gen_lat(model, sampler, quantized=True)
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            uninstall()
            rows.append({"arm": aid, "error": f"{type(e).__name__}: {e}"})
            continue
        r_fp = relL2(lat, lats["fp16"])
        r_a = relL2(lat, lats["A"])
        rec = {"arm": aid, "scope": scope, "retain": retain,
               "relL2_vs_fp16": r_fp, "relL2_vs_A": r_a,
               "ms_step": ms, "trials": trials, "vs_A": ms_a / ms if ms else None,
               "hits": STATE["hits"], "n_layers": len(STATE["layers"]),
               "layers": STATE["layers"]}
        rows.append(rec)
        print(f"  {ms:.2f} ms/step  {ms_a/ms:.3f}x vs A  relL2 {r_fp:.4f}  "
              f"hits={STATE['hits']} layers={len(STATE['layers'])}", flush=True)
        images.append((title, f"{ms:.1f} ms/step  {ms_a/ms:.2f}x  vs fp16 {r_fp:.3f}",
                       G.decode(model, lat)))
        uninstall()

    uninstall()
    del model, sampler
    torch.cuda.empty_cache()
    grid(OUT_PNG, images)
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "seed": SEED, "n_quality": NQ, "batch_time": BATCH, "steps": STEPS,
        "note": "Native GN + fused delta_lowrank_fprop + Cin=k CUTLASS.",
        "micro": micro, "A_ms_step": ms_a, "rows": rows, "png": OUT_PNG,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
