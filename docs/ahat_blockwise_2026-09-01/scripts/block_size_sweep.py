"""Block-size screen for tokenwise / groupwise int8 a_hat storage.

Per-tensor 8-bit failed because max|a_hat| / tail_quantum needs ~11.6 bits
(C15). Blockwise cannot shrink the global max — the worst block still holds
it — but it gives every other location a local scale. This run measures that
on a real W8A8 generation, without mutating the live cache.

Schemes (symmetric absmax, qmax=127, no zero-point):
  per_tensor     one scale for the whole NCHW tensor          (C15 baseline)
  tokenwise      one scale per spatial location (n,h,w) over C
  along_c/B      NHWC groups of B consecutive channels / token
  per_channel    one scale per (n,c) over HW
  along_hw/B     groups of B spatial sites at each channel

Metrics per write, vs the live delta quantum 1/δ:
  bits      log2(2 * block_amax / quantum)     (element-weighted)
  frac_le8  fraction of elements whose block fits in 8 bits
  err_*_q   |Q8(a) − a| / quantum              (rms / p99 / max)

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/ahat_blockwise_2026-09-01/scripts/block_size_sweep.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_DELTA_FREEZE"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="block_size_sweep.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.kernels.int8_optimized import OptimizedInt8Conv2d  # noqa: E402

SHAPE = (4, 32, 32)
QMAX = 127.0
BLOCK_C = (8, 16, 32, 64, 96, 192, 384)
BLOCK_HW = (16, 64, 256)
PLOT_LAYERS = (
    "input_blocks.1.0.in_conv",    # 192×32×32
    "input_blocks.4.0.in_conv",    # 192×16×16
    "input_blocks.4.0.out_conv",   # 384×16×16
)
OUT_JSON = "docs/ahat_blockwise_2026-09-01/data/block_size_sweep.json"


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pack_err(err, quantum):
    flat = err.reshape(-1)
    q = max(quantum, 1e-30)
    p99 = torch.quantile(flat, 0.99).item() if flat.numel() > 1 else flat.item()
    return {
        "err_rms_q": float(err.pow(2).mean().sqrt().item() / q),
        "err_p99_q": float(p99 / q),
        "err_max_q": float(err.max().item() / q),
    }


def _pack_bits(bits_elem, quantum, err):
    flat = bits_elem.reshape(-1)
    rec = {
        "bits_median": float(flat.median().item()),
        "bits_p99": float(torch.quantile(flat, 0.99).item()) if flat.numel() > 1 else float(flat.item()),
        "bits_max": float(flat.max().item()),
        "frac_le8": float((flat <= 8.0).float().mean().item()),
    }
    rec.update(_pack_err(err, quantum))
    return rec


def metrics_per_tensor(x, quantum):
    amax = x.abs().amax().clamp_min(1e-12)
    s = amax / QMAX
    q = (x / s).round().clamp(-QMAX, QMAX)
    err = (q * s - x).abs()
    bits = math.log2(max(2.0 * float(amax) / quantum, 1.0))
    rec = {
        "bits_median": bits,
        "bits_p99": bits,
        "bits_max": bits,
        "frac_le8": 1.0 if bits <= 8.0 else 0.0,
    }
    rec.update(_pack_err(err, quantum))
    return rec


def metrics_along_c(x_nhwc, B, quantum):
    n, h, w, c = x_nhwc.shape
    B = min(int(B), c)
    pad = (B - c % B) % B
    xp = F.pad(x_nhwc, (0, pad)) if pad else x_nhwc
    g = xp.shape[-1] // B
    blk = xp.reshape(n, h, w, g, B)
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    s = amax / QMAX
    q = (blk / s).round().clamp(-QMAX, QMAX)
    recon = (q * s).reshape(n, h, w, -1)[..., :c]
    err = (recon - x_nhwc).abs()
    bits = torch.log2((2.0 * amax.squeeze(-1) / quantum).clamp_min(1.0))
    bits_elem = bits.unsqueeze(-1).expand(-1, -1, -1, -1, B).reshape(n, h, w, -1)[..., :c]
    return _pack_bits(bits_elem, quantum, err)


def metrics_tokenwise(x_nhwc, quantum):
    return metrics_along_c(x_nhwc, x_nhwc.shape[-1], quantum)


def metrics_along_hw(x_nchw, B, quantum):
    n, c, h, w = x_nchw.shape
    hw = h * w
    B = min(int(B), hw)
    y = x_nchw.reshape(n, c, hw)
    pad = (B - hw % B) % B
    yp = F.pad(y, (0, pad)) if pad else y
    g = yp.shape[-1] // B
    blk = yp.reshape(n, c, g, B)
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    s = amax / QMAX
    q = (blk / s).round().clamp(-QMAX, QMAX)
    recon = (q * s).reshape(n, c, -1)[..., :hw]
    err = (recon - y).abs().reshape(n, c, h, w)
    bits = torch.log2((2.0 * amax.squeeze(-1) / quantum).clamp_min(1.0))
    bits_elem = bits.unsqueeze(-1).expand(-1, -1, -1, B).reshape(n, c, -1)[..., :hw]
    return _pack_bits(bits_elem.reshape(n, c, h, w), quantum, err)


def metrics_per_channel(x_nchw, quantum):
    return metrics_along_hw(x_nchw, x_nchw.shape[2] * x_nchw.shape[3], quantum)


# layer -> scheme -> list of per-step recs
ROWS = defaultdict(lambda: defaultdict(list))
SHAPES = {}
ORIG = OptimizedInt8Conv2d._after_ahat_write


def _hook(self, out):
    ret = ORIG(self, out)
    a = self.a_hat_cache
    if (a is None or a.numel() == 0 or int(self.step_count) == 0
            or a.dtype not in (torch.float16, torch.float32)):
        return ret
    name = self.layer_name or "?"
    sc, _ = self._delta_scale_args(a.device)
    quantum = 1.0 / max(float(sc.view(-1)[0].item()), 1e-12)
    x = a.detach().float()
    n, c, h, w = x.shape
    SHAPES[name] = [c, h, w]
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    step = int(self.step_count)
    recs = {
        "per_tensor": metrics_per_tensor(x, quantum),
        "tokenwise": metrics_tokenwise(x_nhwc, quantum),
        "per_channel": metrics_per_channel(x, quantum),
    }
    for B in BLOCK_C:
        recs[f"along_c/{B}"] = metrics_along_c(x_nhwc, B, quantum)
    for B in BLOCK_HW:
        recs[f"along_hw/{B}"] = metrics_along_hw(x, B, quantum)
    for sch, rec in recs.items():
        rec["step"] = step
        rec["quantum"] = quantum
        ROWS[name][sch].append(rec)
    return ret


def _summarize_series(series, tail_frac=1.0 / 3.0):
    if not series:
        return {}
    n = len(series)
    tail = series[max(1, int(n * (1.0 - tail_frac))):]

    def pack(rows):
        keys = ("bits_median", "bits_p99", "bits_max", "frac_le8",
                "err_rms_q", "err_p99_q", "err_max_q")
        return {k: _median([r[k] for r in rows]) for k in keys}

    return {"n": n, "all": pack(series), "tail": pack(tail)}


def summarize():
    layers = {}
    for name, schemes in ROWS.items():
        layers[name] = {"shape": SHAPES.get(name), "schemes": {}}
        for sch, series in schemes.items():
            rec = _summarize_series(series)
            if name in PLOT_LAYERS:
                rec["per_step"] = [
                    {"step": r["step"], "bits_median": r["bits_median"],
                     "frac_le8": r["frac_le8"], "err_rms_q": r["err_rms_q"]}
                    for r in series
                ]
            layers[name]["schemes"][sch] = rec

    scheme_names = []
    if ROWS:
        scheme_names = list(next(iter(ROWS.values())).keys())

    def across(sch, which, key):
        vals = []
        for name in layers:
            node = layers[name]["schemes"].get(sch, {}).get(which)
            if node and node.get(key) is not None:
                vals.append(node[key])
        return _median(vals)

    n_layers = len(layers)

    def n_ok(sch, which, key, pred):
        c = 0
        for name in layers:
            node = layers[name]["schemes"].get(sch, {}).get(which)
            if node and node.get(key) is not None and pred(node[key]):
                c += 1
        return c

    grid = {}
    for sch in scheme_names:
        grid[sch] = {
            "tail_bits_median": across(sch, "tail", "bits_median"),
            "tail_bits_p99": across(sch, "tail", "bits_p99"),
            "tail_bits_max": across(sch, "tail", "bits_max"),
            "tail_frac_le8": across(sch, "tail", "frac_le8"),
            "tail_err_rms_q": across(sch, "tail", "err_rms_q"),
            "tail_err_p99_q": across(sch, "tail", "err_p99_q"),
            "tail_err_max_q": across(sch, "tail", "err_max_q"),
            "n_layers_frac_le8_ge_0.99": n_ok(sch, "tail", "frac_le8", lambda v: v >= 0.99),
            "n_layers_bits_med_le8": n_ok(sch, "tail", "bits_median", lambda v: v <= 8.0),
            "n_layers": n_layers,
        }
    return {"layers": layers, "grid": grid, "scheme_names": scheme_names}


def _print_grid(grid):
    hdr = (f"{'scheme':<16} {'bits med':>9} {'bits p99':>9} {'bits max':>9} "
           f"{'frac≤8':>8} {'rms/q':>8} {'p99/q':>8} {'max/q':>8} "
           f"{'L med≤8':>8} {'L 99%≤8':>8}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for sch, r in grid.items():
        print(
            f"{sch:<16} {r['tail_bits_median']:9.2f} {r['tail_bits_p99']:9.2f} "
            f"{r['tail_bits_max']:9.2f} {r['tail_frac_le8']:8.3f} "
            f"{r['tail_err_rms_q']:8.2f} {r['tail_err_p99_q']:8.2f} "
            f"{r['tail_err_max_q']:8.2f} "
            f"{r['n_layers_bits_med_le8']:8d} {r['n_layers_frac_le8_ge_0.99']:8d}",
            flush=True,
        )


def sample(runner, model, sampler, n, seed, steps):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_sweep",
        batch_size=a.n, steps=a.steps, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    OptimizedInt8Conv2d._after_ahat_write = _hook
    try:
        model, sampler = runner._setup_model("int8")
        sample(runner, model, sampler, a.n, a.seed, a.steps)
        ROWS.clear()
        SHAPES.clear()
        lat = sample(runner, model, sampler, a.n, a.seed, a.steps)
        print(f"captured {len(ROWS)} layers, latent {tuple(lat.shape)}", flush=True)
    finally:
        OptimizedInt8Conv2d._after_ahat_write = ORIG

    payload = summarize()
    payload["meta"] = {
        "n": a.n, "steps": a.steps, "seed": a.seed, "qmax": QMAX,
        "block_c": list(BLOCK_C), "block_hw": list(BLOCK_HW),
        "device": torch.cuda.get_device_name(0),
        "note": "live cache not mutated; one-step snap of the production fp16 a_hat",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f)
    print(f"\n=== tail (last 1/3 of schedule), median across {payload['grid'][next(iter(payload['grid']))]['n_layers']} layers ===",
          flush=True)
    _print_grid(payload["grid"])
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        OptimizedInt8Conv2d._after_ahat_write = ORIG
