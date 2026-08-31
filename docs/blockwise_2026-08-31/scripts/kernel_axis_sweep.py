"""Every shipped/reference kernel, swept one axis at a time over (B, N, H, W).

Six kernels, all 3x3 pad=1 stride=1, channels_last, same default point as
docs/conv_kernel_sweep_2026-08-28 (B=128, N=Cin=Cout=384, H=16, W=16):

  fp16 GN+SiLU        group_norm_silu_nhwc                       (fp16 reference stage 1)
  fp16 conv           F.conv2d fp16, cudnn.benchmark on           (fp16 reference stage 2)
  baseline GN+quant   group_norm_silu_quantize_nhwc_fast          (SHIPPED baseline stage 1)
  baseline conv (D1)  conv2d_int8_evt_bias_residual_fp16          (SHIPPED baseline stage 2)
  MoDiff GN+delta     group_norm_silu_delta_quantize_nhwc         (SHIPPED MoDiff stage 1)
  MoDiff conv (D2)    conv2d_int8_evt_o_hat                       (SHIPPED MoDiff stage 2)

Both production paths are 2 fused kernels each (see docs/blockwise_2026-08-31/FINDINGS.md
"Fused pair vs fused pair"); this sweeps all 6 kernels (2 fp16 + 2 baseline + 2 MoDiff)
independently across B, N, H, W so the per-kernel behavior vs shape is visible, not just
a frequency-weighted total.

N is both Cin and Cout, matching the committed conv_kernel_sweep convention.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/kernel_axis_sweep.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from integration.utils.preflight import preflight  # noqa: E402

preflight("torch", what="kernel_axis_sweep.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

torch.backends.cudnn.benchmark = True

DEV, CL = "cuda", torch.channels_last
JSON_OUT = "docs/blockwise_2026-08-31/data/kernel_axis_sweep.json"
G_NORM, EPS = 32, 1e-5
DEFAULT = {"B": 128, "N": 384, "H": 16, "W": 16}
AXES = {
    "B": (8, 16, 32, 64, 128, 256),
    "N": (128, 192, 256, 384, 512, 768, 1152, 1536),
    "H": (2, 4, 8, 16, 32),
    "W": (2, 4, 8, 16, 32),
}

E32 = torch.empty(0, device=DEV, dtype=torch.float32)
E16 = torch.empty(0, device=DEV, dtype=torch.float16)
EI = torch.empty(0, device=DEV, dtype=torch.int32)


def cl(t):
    return t.contiguous(memory_format=CL)


def _time(fn, reps, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def measure(b, n, h, w, reps, trials):
    """One (B, N, H, W) point. N is both Cin and Cout."""
    cin = cout = n
    st = (1, 1, 1, 1, 1, 1)
    x = cl(torch.randn(b, cin, h, w, device=DEV, dtype=torch.float16))
    gamma = torch.randn(cin, device=DEV, dtype=torch.float16).abs() + 0.5
    beta = torch.randn(cin, device=DEV, dtype=torch.float16) * 0.1
    scale = torch.tensor([16.0], device=DEV, dtype=torch.float32)
    inv_scale = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
    wq = torch.randint(-8, 8, (cout, 3, 3, cin), device=DEV, dtype=torch.int8).contiguous()
    wsc = torch.full((cout,), 0.02, device=DEV, dtype=torch.float32)
    a_hat = cl(0.1 * torch.randn(b, cin, h, w, device=DEV, dtype=torch.float16))
    o_hat = cl(0.1 * torch.randn(b, cout, h, w, device=DEV, dtype=torch.float16))
    residual = cl(0.1 * torch.randn(b, cout, h, w, device=DEV, dtype=torch.float16))
    out = cl(torch.empty(b, cout, h, w, device=DEV, dtype=torch.float16))
    bias = torch.zeros(cout, device=DEV, dtype=torch.float32)
    wh = cl(torch.randn(cout, cin, 3, 3, device=DEV, dtype=torch.float16))

    def f1():
        return mc.group_norm_silu_nhwc(x, gamma, beta, G_NORM, EPS, True, E16, E16)

    normed = f1()

    def f2():
        F.conv2d(normed, wh, None, 1, 1, 1, 1)

    def b1():
        return mc.group_norm_silu_quantize_nhwc_fast(
            x, gamma, beta, G_NORM, EPS, True, scale, E32, E16, E16)

    xq_b = b1()

    def b2():
        mc.conv2d_int8_evt_bias_residual_fp16(
            xq_b, wq, inv_scale, wsc, bias, residual, out, *st)

    def m1():
        return mc.group_norm_silu_delta_quantize_nhwc(
            x, gamma, beta, a_hat, G_NORM, EPS, True, scale,
            E32, E16, E16, E32, E32, E32, EI, 127.0, False, 1.0, False, True, E32)

    xq_m = m1()

    def m2():
        mc.conv2d_int8_evt_o_hat(xq_m, wq, inv_scale, wsc, o_hat, *st)

    rec = {"B": b, "N": n, "H": h, "W": w}
    rec["fp16_gn_ms"] = min(_time(f1, reps) for _ in range(trials))
    rec["fp16_conv_ms"] = min(_time(f2, reps) for _ in range(trials))
    rec["base_gn_ms"] = min(_time(b1, reps) for _ in range(trials))
    rec["base_conv_ms"] = min(_time(b2, reps) for _ in range(trials))
    rec["modiff_gn_ms"] = min(_time(m1, reps) for _ in range(trials))
    rec["modiff_conv_ms"] = min(_time(m2, reps) for _ in range(trials))
    rec["fp16_total"] = rec["fp16_gn_ms"] + rec["fp16_conv_ms"]
    rec["base_total"] = rec["base_gn_ms"] + rec["base_conv_ms"]
    rec["modiff_total"] = rec["modiff_gn_ms"] + rec["modiff_conv_ms"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  default "
          f"B={DEFAULT['B']} N={DEFAULT['N']} H={DEFAULT['H']} W={DEFAULT['W']}  "
          f"reps={a.reps} trials={a.trials}  GN groups={G_NORM}", flush=True)

    out = {"gpu": torch.cuda.get_device_name(0), "default": DEFAULT,
           "axes": {k: list(v) for k, v in AXES.items()},
           "method": "3x3 pad=1 stride=1, channels_last, N=Cin=Cout. fp16 GN = "
                     "group_norm_silu_nhwc; fp16 conv = F.conv2d (cudnn.benchmark); "
                     "baseline = group_norm_silu_quantize_nhwc_fast + "
                     "conv2d_int8_evt_bias_residual_fp16 (D1, shipped); MoDiff = "
                     "group_norm_silu_delta_quantize_nhwc + conv2d_int8_evt_o_hat "
                     "(D2, shipped).",
           "sweeps": {}}

    for axis, vals in AXES.items():
        print(f"\n=== sweep {axis} ===", flush=True)
        print(f"  {axis:>5s} {'fp16_gn':>8s} {'fp16_cv':>8s} {'base_gn':>8s} "
              f"{'base_cv':>8s} {'md_gn':>8s} {'md_cv':>8s}  |  "
              f"{'fp16_tot':>8s} {'base_tot':>8s} {'md_tot':>8s}", flush=True)
        rows = []
        for v in vals:
            p = dict(DEFAULT)
            p[axis] = v
            r = measure(p["B"], p["N"], p["H"], p["W"], a.reps, a.trials)
            rows.append(r)
            print(f"  {v:5d} {r['fp16_gn_ms']:8.3f} {r['fp16_conv_ms']:8.3f} "
                  f"{r['base_gn_ms']:8.3f} {r['base_conv_ms']:8.3f} "
                  f"{r['modiff_gn_ms']:8.3f} {r['modiff_conv_ms']:8.3f}  |  "
                  f"{r['fp16_total']:8.3f} {r['base_total']:8.3f} {r['modiff_total']:8.3f}",
                  flush=True)
        out["sweeps"][axis] = rows

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
