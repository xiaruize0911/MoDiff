"""The full per-step layer path, timed kernel by kernel -- not just the conv.

WHY THIS EXISTS. The cost numbers in FINDINGS section 4 timed ONE kernel, the conv.
That flatters int8: the fp16 arm needs no quantize step at all, so charging int8
only for its conv hides the work that buys the cheaper conv. A ResBlock conv in
this tree is three stages, and only the int8 arm pays all three:

  K1  GroupNorm + SiLU
  K2  quantize (delta against a_hat, + the a_hat write)   <- int8 only
  K3  conv

The shipped MoDiff path fuses K1+K2 into group_norm_silu_delta_quantize_nhwc, which
is why docs/cache_schemes_report_2026-08-28's conv-path figure is two kernels. Both
decompositions are measured here so the fusion's value is visible too:

  fp16          K1 group_norm_silu_nhwc            + K3 F.conv2d
  int8 3-kernel K1 group_norm_silu_nhwc  + K2 step1_static_quantize_fprop_silu
                                                    + K3 conv2d_int8_evt_o_hat
  int8 fused    K1+K2 group_norm_silu_delta_quantize_nhwc
                                                    + K3 conv2d_int8_evt_o_hat
  blockwise     K1+K2 as above, + K3 run Cin/G times (channel-block split-K)

Blockwise caveat, stated because it matters for the totals: a real blockwise
implementation also needs K2 to emit a per-block absmax instead of one scalar. That
extra reduction is NOT modelled -- K2 is timed unchanged -- so the blockwise totals
here are a floor on a floor.

Speedup is fp16_total / arm_total over the 20 UNet shapes, freq-weighted.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/path_kernels.py
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

preflight("torch", what="path_kernels.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

torch.backends.cudnn.benchmark = True

DEV = "cuda"
CL = torch.channels_last
JSON_OUT = "docs/blockwise_2026-08-31/data/path_kernels.json"
G_NORM, EPS = 32, 1e-5          # GroupNorm32, as the LDM UNet uses
BLOCKS = (64, 32)
#: (Cin, Cout, H, W, freq) -- verbatim from conv_layer_microbench.py
SHAPES = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]

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


def measure(b, cin, cout, h, w, reps, trials):
    x = cl(torch.randn(b, cin, h, w, device=DEV, dtype=torch.float16))
    gamma = torch.randn(cin, device=DEV, dtype=torch.float16).abs() + 0.5
    beta = torch.randn(cin, device=DEV, dtype=torch.float16) * 0.1
    scale = torch.tensor([16.0], device=DEV, dtype=torch.float32)
    inv_scale = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
    wq = torch.randint(-8, 8, (cout, 3, 3, cin), device=DEV, dtype=torch.int8).contiguous()
    wsc = torch.full((cout,), 0.02, device=DEV, dtype=torch.float32)
    a_hat = cl(0.1 * torch.randn(b, cin, h, w, device=DEV, dtype=torch.float16))
    o_hat = cl(0.1 * torch.randn(b, cout, h, w, device=DEV, dtype=torch.float16))
    wh = cl(torch.randn(cout, cin, 3, 3, device=DEV, dtype=torch.float16))
    st = (1, 1, 1, 1, 1, 1)

    def k1_gn():
        return mc.group_norm_silu_nhwc(x, gamma, beta, G_NORM, EPS, True, E16, E16)

    normed = k1_gn()

    def k2_quant():
        return mc.step1_static_quantize_fprop_silu(normed, a_hat, scale, E32, False, True, E32)

    xq = k2_quant()

    def k12_fused():
        return mc.group_norm_silu_delta_quantize_nhwc(
            x, gamma, beta, a_hat, G_NORM, EPS, True, scale,
            E32, E16, E16, E32, E32, E32, EI, 127.0, False, 1.0, False, True, E32)

    def k3_conv():
        mc.conv2d_int8_evt_o_hat(xq, wq, inv_scale, wsc, o_hat, *st)

    rec = {"B": b, "cin": cin, "cout": cout, "H": h, "W": w}
    rec["k1_gn_ms"] = min(_time(k1_gn, reps) for _ in range(trials))
    rec["k2_quant_ms"] = min(_time(k2_quant, reps) for _ in range(trials))
    rec["k12_fused_ms"] = min(_time(k12_fused, reps) for _ in range(trials))
    rec["k3_conv_ms"] = min(_time(k3_conv, reps) for _ in range(trials))
    # fp16 arm: same GroupNorm+SiLU kernel, then an fp16 conv. No quantize.
    rec["fp16_conv_ms"] = min(_time(lambda: F.conv2d(normed, wh, None, 1, 1, 1, 1), reps)
                              for _ in range(trials))

    for g in BLOCKS:
        if g > cin or cin % g:
            continue
        nb = cin // g
        xs = [xq[:, i * g:(i + 1) * g].contiguous(memory_format=CL) for i in range(nb)]
        ws = [wq[:, :, :, i * g:(i + 1) * g].contiguous() for i in range(nb)]
        als = [inv_scale.clone() for _ in range(nb)]
        wss = [wsc.clone() for _ in range(nb)]

        def split():
            for i in range(nb):
                mc.conv2d_int8_evt_o_hat(xs[i], ws[i], als[i], wss[i], o_hat, *st)

        rec[f"bw{g}_conv_ms"] = min(_time(split, reps) for _ in range(trials))
        rec[f"bw{g}_blocks"] = nb

    # path totals
    rec["fp16_total"] = rec["k1_gn_ms"] + rec["fp16_conv_ms"]
    rec["int8_3k_total"] = rec["k1_gn_ms"] + rec["k2_quant_ms"] + rec["k3_conv_ms"]
    rec["int8_fused_total"] = rec["k12_fused_ms"] + rec["k3_conv_ms"]
    for g in BLOCKS:
        if f"bw{g}_conv_ms" in rec:
            rec[f"bw{g}_total"] = rec["k12_fused_ms"] + rec[f"bw{g}_conv_ms"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  B={a.batch}  reps={a.reps} "
          f"trials={a.trials}  GroupNorm groups={G_NORM}", flush=True)
    rows = []
    for cin, cout, h, w, freq in SHAPES:
        r = measure(a.batch, cin, cout, h, w, a.reps, a.trials)
        r["freq"] = freq
        rows.append(r)
        print(f"  {cin}->{cout} {h}x{w} (f{freq}): "
              f"K1 {r['k1_gn_ms']:.3f} K2 {r['k2_quant_ms']:.3f} "
              f"K1+2fused {r['k12_fused_ms']:.3f} K3 {r['k3_conv_ms']:.3f} | "
              f"fp16 {r['fp16_total']:.3f} int8-3k {r['int8_3k_total']:.3f} "
              f"int8-fused {r['int8_fused_total']:.3f}", flush=True)

    def wsum(key):
        if not all(key in r for r in rows):
            return None
        return sum(r[key] * r["freq"] for r in rows)

    tot = {k: wsum(k) for k in
           ("k1_gn_ms", "k2_quant_ms", "k12_fused_ms", "k3_conv_ms", "fp16_conv_ms",
            "fp16_total", "int8_3k_total", "int8_fused_total")}
    for g in BLOCKS:
        tot[f"bw{g}_conv_ms"] = wsum(f"bw{g}_conv_ms")
        tot[f"bw{g}_total"] = wsum(f"bw{g}_total")
    tot = {k: v for k, v in tot.items() if v is not None}

    print("\nfreq-weighted over the 20 UNet shapes, 62 calls/step (ms/step):", flush=True)
    print(f"  K1 GN+SiLU                 {tot['k1_gn_ms']:8.2f}", flush=True)
    print(f"  K2 quantize (int8 only)    {tot['k2_quant_ms']:8.2f}", flush=True)
    print(f"  K1+K2 fused                {tot['k12_fused_ms']:8.2f}   "
          f"(saves {tot['k1_gn_ms'] + tot['k2_quant_ms'] - tot['k12_fused_ms']:.2f})", flush=True)
    print(f"  K3 conv int8               {tot['k3_conv_ms']:8.2f}", flush=True)
    print(f"  K3 conv fp16               {tot['fp16_conv_ms']:8.2f}", flush=True)
    print("\n  path totals and speedup vs the fp16 path:", flush=True)
    base = tot["fp16_total"]
    for k in ("fp16_total", "int8_3k_total", "int8_fused_total") + \
             tuple(f"bw{g}_total" for g in BLOCKS):
        if k in tot:
            print(f"    {k:20s} {tot[k]:8.2f} ms   {base / tot[k]:6.3f}x", flush=True)
    print(f"\n  conv kernel alone, int8 vs fp16: "
          f"{tot['fp16_conv_ms'] / tot['k3_conv_ms']:.3f}x  "
          f"<- what FINDINGS section 4 reported", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "reps": a.reps,
               "trials": a.trials, "gn_groups": G_NORM, "blocks": list(BLOCKS),
               "method": "K1 group_norm_silu_nhwc; K2 step1_static_quantize_fprop_silu; "
                         "K1+K2 group_norm_silu_delta_quantize_nhwc; K3 conv2d_int8_evt_o_hat "
                         "or F.conv2d fp16. Blockwise K3 = channel-block split-K with the "
                         "slicing hoisted, and K2 is NOT charged for producing per-block "
                         "scales, so blockwise totals are a floor.",
               "shapes": rows, "freq_weighted_ms": tot}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
