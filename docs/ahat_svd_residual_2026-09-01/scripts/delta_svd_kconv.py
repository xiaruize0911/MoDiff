"""Folded rank-k INT8 residual conv: quality + e2e speed, 3 layers then all.

Per modulated step on a patched layer:
  U = range_finder(delta, k)     # one U for the batch
  Z = U^T delta                  # Cin = k
  q = Q_absmax(Z)
  W_k = quant(W_fp @ U)          # fold, requantize
  a_hat += U @ dequant(q)
  o_hat += conv_k(q; W_k)        # CUTLASS Cin=k

k = align16(retain * Cin). Tail of delta is dropped (error feedback).
Python GN+absmax on patched layers (same confound as delta_svd_gen.py);
compare k-conv against py_full (no SVD, full Cin) and production A.

  python docs/ahat_svd_residual_2026-09-01/scripts/delta_svd_kconv.py
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import integration.benchmarks.benchmark_ldm as B
import integration.kernels.int8_optimized as i8
import ahat_fake_quant_grid as G

OUT_DIR = "docs/ahat_svd_residual_2026-09-01"
OUT_PNG = f"{OUT_DIR}/plots/delta_svd_kconv_grid.png"
OUT_JSON = f"{OUT_DIR}/data/delta_svd_kconv.json"
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
RETAIN = (0.25, 0.33, 0.50, 0.67)

ORIG = None
STATE = {
    "scope": "three",      # "three" | "all"
    "retain": 0.33,
    "kconv": True,         # False = py_full (Python GN+absmax, full Cin)
    "hits": 0,
    "layers": {},          # name -> {Cin, k, retain}
    "fail": 0,
}

CL = torch.channels_last


def align16(k: int, c: int) -> int:
    if c < 16:
        return c
    k = max(16, min(c, int(round(k / 16.0) * 16)))
    return c if k > c else k


def eligible(mod) -> bool:
    if not isinstance(mod, i8.OptimizedInt8Conv2d):
        return False
    if not getattr(mod, "modiff_enabled", False):
        return False
    if getattr(mod, "groups", 1) != 1:
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


def range_finder(d: torch.Tensor, k: int):
    """Batch-shared U (C×k) and Z (N×k×H×W). None, None if k cannot be 16-aligned."""
    d = d.float().contiguous()
    n, c, h, w = d.shape
    k = min(int(k), c, n * h * w)
    k = (k // 16) * 16
    if k < 16:
        return None, None
    m = d.permute(1, 0, 2, 3).reshape(c, -1).contiguous()
    omega = torch.randn(m.shape[1], k, device=d.device, dtype=torch.float32)
    y = m @ omega
    q, _ = torch.linalg.qr(y, mode="reduced")
    u = q[:, :k].contiguous()
    z = (u.T @ m).reshape(k, n, h, w).permute(1, 0, 2, 3).contiguous()
    return u, z


def fold_weight(w_fp: torch.Tensor, u: torch.Tensor):
    """W (Cout,Cin,R,S) @ U (Cin,k) → INT8 KRSC + per-out-channel scale."""
    w_k = torch.einsum("oirs,ik->okrs", w_fp, u)
    cout, k, r, s = w_k.shape
    flat = w_k.reshape(cout, -1)
    ch_max = flat.abs().amax(dim=1).clamp_min(1e-8)
    ch_scale = ch_max / 127.0
    wq = (flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    w_nhwc = wq.reshape(cout, k, r, s).permute(0, 2, 3, 1).contiguous()
    return w_nhwc, ch_scale.contiguous()


def w_fp_cached(self):
    buf = getattr(self, "_dsvd_wfp", None)
    if buf is None or buf.device != self.weight_int8.device:
        self._dsvd_wfp = self.dequantized_weight().contiguous()
    return self._dsvd_wfp


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


def patched_gn_fused(self, x, gn_weight, gn_bias, num_groups, eps,
                     mod_scale2d, mod_shift2d, residual=None):
    if not eligible(self):
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    self.step_count += 1
    if not x.is_contiguous(memory_format=CL):
        x = x.contiguous(memory_format=CL)
    self._ensure_state_buffers(x)
    if (not hasattr(self, "_smooth_inv_flat") or self._smooth_inv_flat.device != x.device):
        if not self._smooth_is_identity:
            self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
        else:
            self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)
    with torch.no_grad():
        tgt = target_activation(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d)
        d = tgt.float() - self.a_hat_cache.float()
        k_req = k_for(self)
        do_k = STATE["kconv"] and k_req < self.in_channels
        u = z = None
        if do_k:
            with torch.amp.autocast("cuda", enabled=False):
                u, z = range_finder(d, k_req)
            do_k = u is not None
        if do_k:
            k = int(z.shape[1])
            with torch.amp.autocast("cuda", enabled=False):
                amax = z.abs().amax().clamp_min(1e-8)
                scale = 127.0 / amax
                q = torch.clamp(torch.round(z * scale), -127, 127)
                deq = q / scale
                if self._write_ahat_now():
                    inc = torch.einsum("ck,nkhw->nchw", u, deq)
                    self.a_hat_cache.add_(inc.half())
                x_int8 = q.to(torch.int8).contiguous(memory_format=CL)
                w_k, wscale = fold_weight(w_fp_cached(self).float(), u)
            alpha = getattr(self, "_dsvd_alpha", None)
            if alpha is None or alpha.device != x.device:
                self._dsvd_alpha = torch.empty(1, device=x.device, dtype=torch.float32)
                alpha = self._dsvd_alpha
            alpha.copy_(torch.reciprocal(scale).reshape(1).float())
            STATE["hits"] += 1
            STATE["layers"][self.layer_name] = {
                "Cin": int(self.in_channels), "Cout": int(self.out_channels),
                "k": int(k), "retain": round(k / self.in_channels, 4),
                "ks": list(self.kernel_size) if isinstance(self.kernel_size, tuple)
                else [self.kernel_size, self.kernel_size],
                "stride": list(self.stride) if isinstance(self.stride, tuple)
                else [self.stride, self.stride],
            }
            return self._after_ahat_write(evt_k(self, x_int8, alpha, w_k, wscale, residual))
        # py_full: same Python GN+absmax, full-C INT8 conv
        amax = d.abs().amax().clamp_min(1e-8)
        scale = 127.0 / amax
        q = torch.clamp(torch.round(d * scale), -127, 127)
        deq = q / scale
        if self._write_ahat_now():
            self.a_hat_cache.add_(deq.half())
        x_int8 = q.to(torch.int8).contiguous(memory_format=CL)
        alpha = getattr(self, "_dsvd_alpha", None)
        if alpha is None or alpha.device != x.device:
            self._dsvd_alpha = torch.empty(1, device=x.device, dtype=torch.float32)
            alpha = self._dsvd_alpha
        alpha.fill_(float(1.0 / scale))
        STATE["hits"] += 1
        STATE["layers"][self.layer_name] = {
            "Cin": int(self.in_channels), "Cout": int(self.out_channels),
            "k": int(self.in_channels), "retain": 1.0,
        }
        if residual is not None:
            residual = residual.to(torch.float16).contiguous(memory_format=CL)
            out = self._layer_out_buf()
            self._evt_ohat_residual(x_int8, alpha, residual, out)
            return self._after_ahat_write(out)
        return self._after_ahat_write(self._evt_ohat(x_int8, alpha))


def install(scope: str, retain: float, kconv: bool):
    global ORIG
    STATE["scope"] = scope
    STATE["retain"] = retain
    STATE["kconv"] = kconv
    STATE["hits"] = 0
    STATE["layers"] = {}
    STATE["fail"] = 0
    if ORIG is None:
        ORIG = i8.OptimizedInt8Conv2d.forward_gn_fused_modiff
    i8.OptimizedInt8Conv2d.forward_gn_fused_modiff = patched_gn_fused


def uninstall():
    if ORIG is not None:
        i8.OptimizedInt8Conv2d.forward_gn_fused_modiff = ORIG


def reset_int8(model):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def census(model):
    rows = []
    for m in model.modules():
        if not isinstance(m, i8.OptimizedInt8Conv2d):
            continue
        ks = m.kernel_size
        rows.append({
            "name": m.layer_name,
            "Cin": m.in_channels, "Cout": m.out_channels,
            "ks": list(ks) if isinstance(ks, tuple) else [ks, ks],
            "stride": list(m.stride) if isinstance(m.stride, tuple) else [m.stride, m.stride],
            "groups": m.groups, "modiff": bool(m.modiff_enabled),
            "fuse_silu": bool(getattr(m, "fuse_input_silu", False)),
        })
    return rows


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
        output_dir=f"{OUT_DIR}/tmp_kconv",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def isolated_kconv():
    """CUTLASS-only Cin sweep + one-layer tax (range finder + fold) at batch 128."""
    mc = i8.modiff_cutlass
    rows = []
    shapes = [(192, 192, 32), (192, 192, 16), (384, 384, 16)]
    ks = (32, 48, 64, 96, 128, 192, 256)
    n, reps, warm = BATCH, 40, 8
    for cin, cout, hw in shapes:
        for k in ks:
            if k > cin or (k % 16) != 0:
                continue
            x = torch.randint(-8, 8, (n, k, hw, hw), device="cuda", dtype=torch.int8)
            x = x.contiguous(memory_format=CL)
            w = torch.randint(-8, 8, (cout, 3, 3, k), device="cuda", dtype=torch.int8).contiguous()
            inv = torch.tensor([1.0 / 16.0], device="cuda")
            wscale = torch.full((cout,), 0.02, device="cuda")
            ohat = (0.1 * torch.randn(n, cout, hw, hw, device="cuda", dtype=torch.float16)
                    ).contiguous(memory_format=CL)
            def go():
                mc.conv2d_int8_evt_o_hat(x, w, inv, wscale, ohat, 1, 1, 1, 1, 1, 1)
            for _ in range(warm):
                go()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            for _ in range(reps):
                go()
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / reps
            rows.append({
                "Cin": k, "Cout": cout, "HW": hw, "ms": round(ms, 4),
                "full_Cin": cin, "flop": round(k / cin, 4),
            })
            print(f"  micro {cout}@{hw} Cin={k:3d}  {ms:.3f} ms", flush=True)
    # tax: range finder + Z + fold on a representative tensor
    tax = []
    for cin, cout, hw in shapes:
        k = 64 if cin >= 64 else 32
        d = torch.randn(n, cin, hw, hw, device="cuda", dtype=torch.float32)
        wfp = torch.randn(cout, cin, 3, 3, device="cuda", dtype=torch.float32)
        def tax_fn():
            u, z = range_finder(d, k)
            fold_weight(wfp, u)
            _ = z
        for _ in range(4):
            tax_fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(20):
            tax_fn()
        e.record()
        torch.cuda.synchronize()
        tax.append({"Cin": cin, "Cout": cout, "HW": hw, "k": k,
                    "range_finder_fold_ms": round(s.elapsed_time(e) / 20, 4)})
        print(f"  tax   {cout}@{hw} k={k}  {tax[-1]['range_finder_fold_ms']:.3f} ms",
              flush=True)
    return {"conv": rows, "tax": tax}


def main():
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

    print("===== isolated k-conv microbench =====", flush=True)
    micro = isolated_kconv()

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
    layers = census(model.model.diffusion_model)
    n_three = sum(1 for r in layers if r["name"] in THREE)
    n_elig = sum(1 for r in layers
                 if r["modiff"] and r["groups"] == 1 and r["ks"] == [3, 3]
                 and r["Cin"] >= 32)
    print(f"  convs={len(layers)}  three={n_three}  eligible_3x3={n_elig}", flush=True)

    uninstall()
    ms_a, trials_a = time_n(model, sampler, quantized=True)
    lat = gen_lat(model, sampler, quantized=True)
    lats["A"] = lat
    r_fp = relL2(lat, lats["fp16"])
    rec_a = {
        "arm": "A", "scope": None, "retain": None, "kconv": False,
        "relL2_vs_fp16": r_fp, "relL2_vs_A": 0.0,
        "ms_step": ms_a, "trials": trials_a, "vs_A": 1.0,
        "hits": 0, "n_layers": 0, "layers": {},
    }
    rows.append(rec_a)
    print(f"  A  {ms_a:.2f} ms/step  relL2 vs fp16 {r_fp:.4f}", flush=True)
    images.append(("A  full MoDiff  S=50",
                   f"{ms_a:.1f} ms/step   relL2 {r_fp:.3f}",
                   G.decode(model, lat)))

    extra = [("three_py", "three", 1.0, False, "3 layers  Python GN+absmax  full Cin")]
    for frac in RETAIN:
        extra.append((f"three_r{frac:.2f}", "three", frac, True,
                      f"3 layers  k-conv retain {frac:.0%}"))
    extra.append(("all_py", "all", 1.0, False, "all 3x3  Python GN+absmax  full Cin"))
    for frac in RETAIN:
        extra.append((f"all_r{frac:.2f}", "all", frac, True,
                      f"all 3x3  k-conv retain {frac:.0%}"))

    grid_ids = {"A", "three_r0.33", "three_r0.50", "all_r0.33", "all_r0.50"}

    for aid, scope, retain, kconv, title in extra:
        install(scope, retain, kconv)
        print(f"----- {aid}  scope={scope} retain={retain} kconv={kconv} -----",
              flush=True)
        try:
            ms, trials = time_n(model, sampler, quantized=True)
            STATE["hits"] = 0
            STATE["layers"] = {}
            lat = gen_lat(model, sampler, quantized=True)
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            uninstall()
            rows.append({
                "arm": aid, "scope": scope, "retain": retain, "kconv": kconv,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        lats[aid] = lat
        r_fp = relL2(lat, lats["fp16"])
        r_a = relL2(lat, lats["A"])
        rec = {
            "arm": aid, "scope": scope, "retain": retain, "kconv": kconv,
            "relL2_vs_fp16": r_fp, "relL2_vs_A": r_a,
            "ms_step": ms, "trials": trials, "vs_A": (ms_a / ms) if ms else None,
            "hits": STATE["hits"], "n_layers": len(STATE["layers"]),
            "layers": STATE["layers"],
        }
        rows.append(rec)
        print(f"  {ms:.2f} ms/step  {ms_a/ms:.3f}x vs A  "
              f"relL2 fp16 {r_fp:.4f}  vs A {r_a:.4f}  "
              f"hits={STATE['hits']} layers={len(STATE['layers'])}", flush=True)
        if aid in grid_ids:
            images.append((title,
                           f"{ms:.1f} ms/step  {ms_a/ms:.2f}x  vs fp16 {r_fp:.3f}",
                           G.decode(model, lat)))
        uninstall()

    uninstall()
    del model, sampler
    torch.cuda.empty_cache()

    grid(OUT_PNG, images)
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "seed": SEED, "n_quality": NQ, "batch_time": BATCH, "steps": STEPS,
        "three": list(THREE), "retain": list(RETAIN),
        "note": ("Range-finder U + folded Cin=k CUTLASS. Quality n=4; "
                 "timing batch 128 median of 2 after 1 warmup. "
                 "Patched layers use Python GN+absmax."),
        "census": {
            "n_conv": len(layers), "n_three": n_three,
            "n_eligible_3x3": n_elig,
            "n_fuse_silu_3x3": sum(1 for r in layers if r["fuse_silu"]
                                   and r["ks"] == [3, 3] and r["Cin"] >= 32),
            "names": [r["name"] for r in layers if r["ks"] == [3, 3]],
        },
        "microbench": micro,
        "fp16_ms_step": None,
        "A_ms_step": ms_a,
        "rows": rows,
        "png": OUT_PNG,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
