"""Independent re-derivation of ../data/combined_w8a8_w4a4.csv and ../TABLE.md.

No generator for that data was ever committed to this repo -- checked by `git log -S` across all
history for the CSV's filename and its column names: the only hits are two later CONSUMERS
(docs/perf_report_2026-08-26/scripts/analyze.py and docs/ahat_overlap_2026-08-26/scripts/
make_findings.py), never a writer. This rebuilds the measurement from the methodology TABLE.md
documents and the production kernel entry points, so the committed numbers can be checked rather
than trusted on faith.

METHODOLOGY, as stated in TABLE.md:
  - GN base / MoDiff:   group_norm_silu_quantize_nhwc_fast      vs group_norm_silu_delta_quantize_nhwc      (W8A8)
                        group_norm_silu_quantize_pack_nhwc_fast vs group_norm_silu_delta_quantize_pack_nhwc (W4A4)
  - conv base / MoDiff: conv2d_int{8,4}_evt_bias_residual_fp16  vs conv2d_int{8,4}_evt_o_hat
  - "independent-layers chained (no same-buffer RAW hazard)": L=16 independently-allocated layers
    per shape/arm, called back to back and timed as one CUDA-event-bracketed chain, so no launch
    reads a buffer a sibling launch is still writing.
  - REPS=40 chain repetitions per trial, 5 trials per shape, order of the 4 arms ROTATED each
    trial (the discipline that caught a sign-flipping reading earlier in this project) to cancel
    systematic GPU drift between arms.
  - all 20 real conv shapes this LSUN-churches UNet uses, both precisions -- freq/Cin/Cout/H/W
    read from the committed CSV's shape column (structure only; none of ITS timing numbers feed
    this script).

a_hat_ms = gn_modiff - gn_base. o_hat_ms = conv_modiff - conv_base. Percentages are each cost as
a share of MoDiff's own (GN+conv) total. block_ratio = MoDiff total / baseline total. All four
definitions copied verbatim from TABLE.md's own text.

Writes data/combined_w8a8_w4a4_REDERIVED.csv and prints a shape-by-shape, column-by-column diff
against the committed data/combined_w8a8_w4a4.csv.
"""
import csv
import os
import statistics
import sys

import torch

sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")
import modiff_cutlass as mc

torch.manual_seed(0)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, G = 128, 32
L, REPS, WARMUP, TRIALS = 16, 40, 5, 5
CL = torch.channels_last

# (Cin, Cout, H, W, freq) -- read from the committed CSV's shape/freq columns only.
SHAPES = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]

E32 = torch.empty(0, device="cuda", dtype=torch.float32)
E16 = torch.empty(0, device="cuda", dtype=torch.float16)
EI = torch.empty(0, device="cuda", dtype=torch.int32)


def pack4(codes_nchw_or_kcrs):
    """[*, C, *, *] int codes in [-8,7] -> [*, *, *, C/2] int8, low nibble = channel 2i.
    Matches integration/tests/test_zpw_additive.py's pack_act/pack_w exactly."""
    c = codes_nchw_or_kcrs.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


def make_gn_layer(Cin, Cout, H, W, prec):
    x = torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    gamma = torch.randn(Cin, device="cuda", dtype=torch.float16).abs() + 0.5
    beta = torch.randn(Cin, device="cuda", dtype=torch.float16) * 0.1
    scale = torch.tensor([16.0 if prec == "W8A8" else 3.0], device="cuda", dtype=torch.float32)
    a_hat = (0.1 * torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float16)).to(memory_format=CL)
    return dict(x=x, gamma=gamma, beta=beta, scale=scale, a_hat=a_hat)


def make_conv_layer(Cin, Cout, H, W, prec):
    if prec == "W8A8":
        xq = torch.randint(-8, 8, (N, Cin, H, W), device="cuda", dtype=torch.int8).to(memory_format=CL)
        wq = torch.randint(-8, 8, (Cout, 3, 3, Cin), device="cuda", dtype=torch.int8).contiguous()
    else:
        x_raw = torch.randint(-7, 8, (N, Cin, H, W), device="cuda", dtype=torch.int64)
        w_raw = torch.randint(-7, 8, (Cout, Cin, 3, 3), device="cuda", dtype=torch.int64)
        xq = pack4(x_raw)                                   # [N,H,W,Cin/2]
        wq = pack4(w_raw)                                   # [Cout,3,3,Cin/2]
    inv_scale = torch.tensor([1.0 / 16.0 if prec == "W8A8" else 1.0 / 3.0],
                             device="cuda", dtype=torch.float32)
    wscale = torch.full((Cout,), 0.02, device="cuda", dtype=torch.float32)
    o_hat = (0.1 * torch.randn(N, Cout, H, W, device="cuda", dtype=torch.float16)).to(memory_format=CL)
    output = torch.empty(N, Cout, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    return dict(x=xq, w=wq, inv_scale=inv_scale, wscale=wscale, o_hat=o_hat, output=output)


def call_gn_base(l, prec):
    if prec == "W8A8":
        mc.group_norm_silu_quantize_nhwc_fast(l["x"], l["gamma"], l["beta"], G, 1e-5, True,
                                              l["scale"], E32, E16, E16)
    else:
        mc.group_norm_silu_quantize_pack_nhwc_fast(l["x"], l["gamma"], l["beta"], G, 1e-5, True,
                                                    l["scale"], E32, E16, E16, 0)


def call_gn_modiff(l, prec):
    if prec == "W8A8":
        mc.group_norm_silu_delta_quantize_nhwc(l["x"], l["gamma"], l["beta"], l["a_hat"], G, 1e-5,
                                               True, l["scale"], E32, E16, E16,
                                               E32, E32, E32, EI, 127.0, False, 1.0)
    else:
        mc.group_norm_silu_delta_quantize_pack_nhwc(l["x"], l["gamma"], l["beta"], l["a_hat"], G,
                                                     1e-5, True, l["scale"], E32, E16, E16,
                                                     E32, E32, E32, EI, 7.0, False, 1.0)


def call_conv_base(l, prec):
    fn = mc.conv2d_int8_evt_bias_residual_fp16 if prec == "W8A8" else mc.conv2d_int4_evt_bias_residual_fp16
    fn(l["x"], l["w"], l["inv_scale"], l["wscale"], E32, E16, l["output"], 1, 1, 1, 1, 1, 1)


def call_conv_modiff(l, prec):
    fn = mc.conv2d_int8_evt_o_hat if prec == "W8A8" else mc.conv2d_int4_evt_o_hat
    fn(l["x"], l["w"], l["inv_scale"], l["wscale"], l["o_hat"], 1, 1, 1, 1, 1, 1)


ARMS = {
    "gn_base": (make_gn_layer, call_gn_base),
    "gn_modiff": (make_gn_layer, call_gn_modiff),
    "conv_base": (make_conv_layer, call_conv_base),
    "conv_modiff": (make_conv_layer, call_conv_modiff),
}


def time_chain(make_fn, call_fn, Cin, Cout, H, W, prec):
    layers = [make_fn(Cin, Cout, H, W, prec) for _ in range(L)]

    def chain():
        for lyr in layers:
            call_fn(lyr, prec)

    for _ in range(WARMUP):
        chain()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(REPS):
        chain()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (L * REPS)


rows_out = []
for Cin, Cout, H, W, freq in SHAPES:
    for prec in ("W8A8", "W4A4"):
        samples = {a: [] for a in ARMS}
        arm_names = list(ARMS)
        for t in range(TRIALS):
            order = arm_names[t % len(arm_names):] + arm_names[:t % len(arm_names)]
            for a in order:
                make_fn, call_fn = ARMS[a]
                samples[a].append(time_chain(make_fn, call_fn, Cin, Cout, H, W, prec))
        med = {a: statistics.median(v) for a, v in samples.items()}
        gn_b, gn_m = med["gn_base"], med["gn_modiff"]
        cv_b, cv_m = med["conv_base"], med["conv_modiff"]
        a_hat, o_hat = gn_m - gn_b, cv_m - cv_b
        modiff_tot, base_tot = gn_m + cv_m, gn_b + cv_b
        row = dict(shape=f"{Cin}->{Cout},{H}x{W}", Cin=Cin, Cout=Cout, H=H, W=W, freq=freq,
                  precision=prec, gn_base=gn_b, gn_modiff=gn_m, conv_base=cv_b, conv_modiff=cv_m,
                  a_hat_ms=a_hat, o_hat_ms=o_hat, modiff_total_ms=modiff_tot, base_total_ms=base_tot,
                  a_hat_pct=100 * a_hat / modiff_tot, o_hat_pct=100 * o_hat / modiff_tot,
                  block_ratio=modiff_tot / base_tot)
        rows_out.append(row)
        print(f"{row['shape']:>18} {prec}  gn {gn_b:.4f}->{gn_m:.4f}  conv {cv_b:.4f}->{cv_m:.4f}  "
              f"a_hat% {row['a_hat_pct']:5.2f}  o_hat% {row['o_hat_pct']:5.2f}  "
              f"ratio {row['block_ratio']:.4f}")

out_path = f"{HERE}/data/combined_w8a8_w4a4_REDERIVED.csv"
fields = list(rows_out[0].keys())
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)
print(f"\nwrote {out_path}")

# ---- diff against the committed file ----
committed_path = f"{HERE}/data/combined_w8a8_w4a4.csv"
if os.path.exists(committed_path):
    committed = {(r["shape"], r["precision"]): r for r in csv.DictReader(open(committed_path))}
    print(f"\n=== diff vs {committed_path} ===")
    print(f"{'shape':>18} {'prec':<5} {'field':<14} {'committed':>10} {'rederived':>10} {'rel diff':>9}")
    CHECK = ["gn_base", "gn_modiff", "conv_base", "conv_modiff", "a_hat_ms", "o_hat_ms",
             "a_hat_pct", "o_hat_pct", "block_ratio"]
    n_flag = 0
    for r in rows_out:
        key = (r["shape"], r["precision"])
        if key not in committed:
            print(f"{r['shape']:>18} {r['precision']:<5}  NOT IN COMMITTED FILE")
            continue
        c = committed[key]
        for f_ in CHECK:
            cv, rv = float(c[f_]), float(r[f_])
            rel = (rv - cv) / cv if abs(cv) > 1e-9 else float("nan")
            flag = "  <<<" if abs(rel) > 0.15 else ""
            if flag:
                n_flag += 1
                print(f"{r['shape']:>18} {r['precision']:<5} {f_:<14} {cv:>10.4f} {rv:>10.4f} "
                      f"{100*rel:>8.1f}%{flag}")
    print(f"\n{n_flag} (shape, field) pairs differ by more than 15% relative.")
else:
    print(f"\nno committed file at {committed_path} to diff against")
