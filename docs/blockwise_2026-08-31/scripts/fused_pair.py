"""MoDiff's two fused kernels vs the baseline's two fused kernels.

Both arms in their FUSED production form, so the comparison is 2 kernels against
2 kernels rather than a fused arm against an unfused one:

  baseline (plain int8, no temporal cache)
    B1  group_norm_silu_quantize_nhwc_fast     GN + SiLU + quantize   <- what SHIPS
    B1p group_norm_silu_quantize_nhwc          the same, plain reduction (MODIFF_GN_FAST=0)
    B2  conv2d_int8_evt_bias_residual_fp16     out = acc*alpha*ws + bias + residual  (D1)

  MoDiff (temporal delta cache)
    M1  group_norm_silu_delta_quantize_nhwc    GN + SiLU + delta-vs-a_hat + a_hat write
    M2  conv2d_int8_evt_o_hat                  o_hat += acc*alpha*ws                 (D2)

  NOTE, and it matters for reading M1 vs B1: the fast-reduce variant
  (docs/gn_fast_reduce_2026-08-16 -- 128-512 threads, pair-major pass 1, worth 1.91x on
  the baseline kernel) was only ever ported to the BASELINE entry point. There is no
  group_norm_silu_delta_quantize_nhwc_fast in the tree, so the shipped MoDiff arm runs
  the plain reduction while the shipped baseline arm runs the fast one. Both baseline
  variants are timed so that asymmetry is visible rather than buried in one ratio.

  fp16 reference (nothing to fuse a quantize into)
    F1  group_norm_silu_nhwc
    F2  F.conv2d in fp16

Why this and not the earlier numbers. docs/conv_kernel_sweep_2026-08-28 compared B2
against M2 and put the fused EVT at 0.966x, but never compared B1 against M1 -- and
M1 does strictly more work (it reads a_hat and writes it back, which B1 does not).
The earlier path_kernels.py run compared MoDiff's fused pair against an fp16 arm,
which answers "does quantizing pay" but not "what does the temporal cache cost".
This answers both, on the same operands.

Blockwise is applied to whichever conv the arm uses, by channel-block split-K. The
slicing is hoisted out of the timed region and K1 is NOT charged for producing
per-block scales, so blockwise rows are a floor. For the MoDiff arm the split is
numerically exact (D2 accumulates into o_hat); for the baseline arm it is a TIMING
PROXY only, because D1 writes its output instead of accumulating -- see _split_conv.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/fused_pair.py
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

preflight("torch", what="fused_pair.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

torch.backends.cudnn.benchmark = True

DEV, CL = "cuda", torch.channels_last
JSON_OUT = "docs/blockwise_2026-08-31/data/fused_pair.json"
G_NORM, EPS = 32, 1e-5
BLOCKS = (64, 32)
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


def _split_conv(fn_kind, xq, wq, alpha, wsc, o_hat, bias, residual, out, g, reps, trials):
    """Channel-block split-K over the arm's own conv."""
    cin = xq.shape[1]
    nb = cin // g
    st = (1, 1, 1, 1, 1, 1)
    xs = [xq[:, i * g:(i + 1) * g].contiguous(memory_format=CL) for i in range(nb)]
    ws = [wq[:, :, :, i * g:(i + 1) * g].contiguous() for i in range(nb)]
    als = [alpha.clone() for _ in range(nb)]
    wss = [wsc.clone() for _ in range(nb)]

    if fn_kind == "modiff":
        def split():
            for i in range(nb):
                mc.conv2d_int8_evt_o_hat(xs[i], ws[i], als[i], wss[i], o_hat, *st)
    else:
        # TIMING PROXY ONLY, not a correct blockwise baseline. D1 WRITES `out` instead of
        # accumulating into it, so nb calls overwrite each other rather than summing --
        # the result is wrong. The per-call arithmetic and memory traffic are what a real
        # blockwise D1 would do, so the TIME is representative, but a correct version needs
        # either an accumulating D1 variant (which does not exist) or a separate reduction
        # pass, and would therefore cost MORE than this. The MoDiff arm has no such caveat:
        # D2's o_hat read-modify-write makes its split exact.
        def split():
            for i in range(nb):
                mc.conv2d_int8_evt_bias_residual_fp16(
                    xs[i], ws[i], als[i], wss[i], bias, residual, out, *st)

    return min(_time(split, reps) for _ in range(trials)), nb


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
    residual = cl(0.1 * torch.randn(b, cout, h, w, device=DEV, dtype=torch.float16))
    out = cl(torch.empty(b, cout, h, w, device=DEV, dtype=torch.float16))
    bias = torch.zeros(cout, device=DEV, dtype=torch.float32)
    wh = cl(torch.randn(cout, cin, 3, 3, device=DEV, dtype=torch.float16))
    st = (1, 1, 1, 1, 1, 1)

    # ---- baseline: 2 fused kernels ----
    def b1_fast():
        return mc.group_norm_silu_quantize_nhwc_fast(
            x, gamma, beta, G_NORM, EPS, True, scale, E32, E16, E16)

    def b1_plain():
        return mc.group_norm_silu_quantize_nhwc(
            x, gamma, beta, G_NORM, EPS, True, scale, E32, E16, E16)

    xq_b = b1_fast()

    def b2():
        mc.conv2d_int8_evt_bias_residual_fp16(
            xq_b, wq, inv_scale, wsc, bias, residual, out, *st)

    # ---- MoDiff: 2 fused kernels ----
    def m1():
        return mc.group_norm_silu_delta_quantize_nhwc(
            x, gamma, beta, a_hat, G_NORM, EPS, True, scale,
            E32, E16, E16, E32, E32, E32, EI, 127.0, False, 1.0, False, True, E32)

    xq_m = m1()

    def m2():
        mc.conv2d_int8_evt_o_hat(xq_m, wq, inv_scale, wsc, o_hat, *st)

    # ---- fp16 reference ----
    def f1():
        return mc.group_norm_silu_nhwc(x, gamma, beta, G_NORM, EPS, True, E16, E16)

    normed = f1()

    rec = {"B": b, "cin": cin, "cout": cout, "H": h, "W": w}
    rec["b1_ms"] = min(_time(b1_fast, reps) for _ in range(trials))
    rec["b1_plain_ms"] = min(_time(b1_plain, reps) for _ in range(trials))
    rec["b2_ms"] = min(_time(b2, reps) for _ in range(trials))
    rec["m1_ms"] = min(_time(m1, reps) for _ in range(trials))
    rec["m2_ms"] = min(_time(m2, reps) for _ in range(trials))
    rec["f1_ms"] = min(_time(f1, reps) for _ in range(trials))
    rec["f2_ms"] = min(_time(lambda: F.conv2d(normed, wh, None, 1, 1, 1, 1), reps)
                       for _ in range(trials))
    rec["baseline_total"] = rec["b1_ms"] + rec["b2_ms"]
    rec["baseline_plain_total"] = rec["b1_plain_ms"] + rec["b2_ms"]
    rec["modiff_total"] = rec["m1_ms"] + rec["m2_ms"]
    rec["fp16_total"] = rec["f1_ms"] + rec["f2_ms"]

    for g in BLOCKS:
        if g > cin or cin % g:
            continue
        t_m, nb = _split_conv("modiff", xq_m, wq, inv_scale, wsc, o_hat, bias,
                              residual, out, g, reps, trials)
        t_b, _ = _split_conv("baseline", xq_b, wq, inv_scale, wsc, o_hat, bias,
                             residual, out, g, reps, trials)
        rec[f"m2_bw{g}_ms"] = t_m
        rec[f"b2_bw{g}_ms"] = t_b
        rec[f"modiff_bw{g}_total"] = rec["m1_ms"] + t_m
        rec[f"baseline_bw{g}_total"] = rec["b1_ms"] + t_b
        rec[f"bw{g}_blocks"] = nb
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  B={a.batch}  reps={a.reps} "
          f"trials={a.trials}  GN groups={G_NORM}", flush=True)
    rows = []
    for cin, cout, h, w, freq in SHAPES:
        r = measure(a.batch, cin, cout, h, w, a.reps, a.trials)
        r["freq"] = freq
        rows.append(r)
        print(f"  {cin}->{cout} {h}x{w} (f{freq}): "
              f"B1 {r['b1_ms']:.3f} B2 {r['b2_ms']:.3f} = {r['baseline_total']:.3f} | "
              f"M1 {r['m1_ms']:.3f} M2 {r['m2_ms']:.3f} = {r['modiff_total']:.3f} | "
              f"MoDiff/base {r['baseline_total'] / r['modiff_total']:.3f}x", flush=True)

    def wsum(key):
        if not all(key in r for r in rows):
            return None
        return sum(r[key] * r["freq"] for r in rows)

    keys = ["b1_ms", "b1_plain_ms", "b2_ms", "m1_ms", "m2_ms", "f1_ms", "f2_ms",
            "baseline_total", "baseline_plain_total", "modiff_total", "fp16_total"]
    for g in BLOCKS:
        keys += [f"m2_bw{g}_ms", f"b2_bw{g}_ms",
                 f"modiff_bw{g}_total", f"baseline_bw{g}_total"]
    tot = {k: wsum(k) for k in keys}
    tot = {k: v for k, v in tot.items() if v is not None}

    print("\nfreq-weighted over the 20 UNet shapes, 62 calls/step (ms/step):", flush=True)
    print(f"  baseline  B1 GN+SiLU+quant FAST {tot['b1_ms']:8.2f}   (shipped)", flush=True)
    print(f"            B1 same, plain reduce {tot['b1_plain_ms']:8.2f}   "
          f"fast is {tot['b1_plain_ms'] / tot['b1_ms']:.3f}x", flush=True)
    print(f"            B2 conv (D1)          {tot['b2_ms']:8.2f}   "
          f"total {tot['baseline_total']:8.2f}", flush=True)
    print(f"  MoDiff    M1 GN+SiLU+delta+ahat {tot['m1_ms']:8.2f}   "
          f"{tot['b1_ms'] / tot['m1_ms']:.3f}x vs B1", flush=True)
    print(f"            M2 conv (D2, o_hat)   {tot['m2_ms']:8.2f}   "
          f"{tot['b2_ms'] / tot['m2_ms']:.3f}x vs B2   "
          f"total {tot['modiff_total']:8.2f}", flush=True)
    print(f"  fp16      F1 GN+SiLU            {tot['f1_ms']:8.2f}", flush=True)
    print(f"            F2 conv fp16          {tot['f2_ms']:8.2f}   "
          f"total {tot['fp16_total']:8.2f}", flush=True)

    print("\n  2-kernel path totals:", flush=True)
    for lab, k in (("fp16", "fp16_total"),
                   ("baseline (fast)", "baseline_total"),
                   ("baseline (plain)", "baseline_plain_total"),
                   ("MoDiff int8", "modiff_total")):
        print(f"    {lab:16s} {tot[k]:8.2f} ms   "
              f"{tot['fp16_total'] / tot[k]:6.3f}x vs fp16   "
              f"{tot['baseline_total'] / tot[k]:6.3f}x vs baseline", flush=True)
    for g in BLOCKS:
        for lab, k in ((f"baseline bw G={g}", f"baseline_bw{g}_total"),
                       (f"MoDiff   bw G={g}", f"modiff_bw{g}_total")):
            if k in tot:
                print(f"    {lab:16s} {tot[k]:8.2f} ms   "
                      f"{tot['fp16_total'] / tot[k]:6.3f}x vs fp16   "
                      f"{tot['baseline_total'] / tot[k]:6.3f}x vs baseline", flush=True)

    # ---- roofline: is M1 slow because it is untuned, or because it moves more bytes? ----
    # bytes per INPUT element for each GN-stage kernel:
    #   F1 fp16   read x fp16 2 + write normed fp16 2                       = 4
    #   B1        read x fp16 2 + write int8 codes 1                        = 3
    #   M1        read x 2 + read a_hat 2 + write codes 1 + write a_hat 2   = 7
    PEAK_GBS = 696.0                      # A40 HBM2, as quoted in fused_resblock.py
    el = sum(a.batch * r["cin"] * r["H"] * r["W"] * r["freq"] for r in rows)
    BPE = {"f1_ms": 4, "b1_plain_ms": 3, "b1_ms": 3, "m1_ms": 7}
    roof = {"input_elems": el, "peak_gbs": PEAK_GBS, "bytes_per_elem": BPE, "gbs": {}}
    print("\n  effective bandwidth of the GN stage (A40 peak 696 GB/s):", flush=True)
    for k, bpe in BPE.items():
        g = el * bpe / (tot[k] * 1e-3) / 1e9
        roof["gbs"][k] = g
        print(f"    {k:14s} {tot[k]:6.2f} ms  {bpe}B/elem  {g:6.1f} GB/s  "
              f"{g / PEAK_GBS * 100:5.1f}% of peak", flush=True)
    m1_roof = el * BPE["m1_ms"] / (PEAK_GBS * 1e9) * 1e3
    roof["m1_at_roofline_ms"] = m1_roof
    roof["modiff_total_if_m1_roofline"] = m1_roof + tot["m2_ms"]
    print(f"    M1 at roofline would be {m1_roof:.2f} ms -> MoDiff total "
          f"{m1_roof + tot['m2_ms']:.2f} ms = "
          f"{tot['baseline_total'] / (m1_roof + tot['m2_ms']):.3f}x baseline "
          f"(so even perfect M1 only draws level)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "reps": a.reps,
               "trials": a.trials, "gn_groups": G_NORM, "blocks": list(BLOCKS),
               "method": "both arms in fused 2-kernel form. baseline = "
                         "group_norm_silu_quantize_nhwc + conv2d_int8_evt_bias_residual_fp16 "
                         "(D1). MoDiff = group_norm_silu_delta_quantize_nhwc + "
                         "conv2d_int8_evt_o_hat (D2). fp16 = group_norm_silu_nhwc + F.conv2d. "
                         "Blockwise = channel-block split-K on the arm's own conv, slicing "
                         "hoisted and K1 not charged for per-block scales, so a floor.",
               "shapes": rows, "freq_weighted_ms": tot, "roofline": roof},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
