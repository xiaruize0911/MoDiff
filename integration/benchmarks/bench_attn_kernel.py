"""
bench_attn_kernel.py — Isolated attention Conv1d kernel microbenchmark.

Compares the per-call speed of every attention projection (Conv1d ks=1) across
all precision modes with and without MoDiff temporal delta-caching:

    FP32          — torch F.conv1d, float32
    FP16          — torch F.conv1d, float16
    INT8 no-MoDiff — CUTLASS INT8:  K1+K2 → K3(quant) → K4(GEMM) → K5 → K7+K8
    INT8 w/MoDiff  — CUTLASS INT8:  K1+K2+K3(fused delta+cache) → K4(GEMM+acc) → K7+K8
    INT4 no-MoDiff — CUTLASS INT4:  K1+K2 → K3(pack) → K4(GEMM) → K5 → K7+K8
    INT4 w/MoDiff  — CUTLASS INT4:  K1+K2 → K3(delta+pack) → K4(GEMM+acc) → K7+K8

Also reports isolated timings for kernel sub-groups:
    K1+K2  — fp16 NCW → fp32 channels-last (tiled layout transpose + dtype cast)
    K7+K8  — fp32 CL  → fp16 NCW           (tiled layout transpose + dtype cast)
    K4     — INT8/INT4 CUTLASS GEMM only (no layout overhead)

All 6 Conv1d projection shapes present in the LSUN Churches 256 UNet:
    C=192: qkv (192→576) and proj_out (192→192)  — 5 AttentionBlocks each
    C=384: qkv (384→1152) and proj_out (384→384) — 10 AttentionBlocks each
    C=768: qkv (768→2304) and proj_out (768→768) — 6 AttentionBlocks each

Benchmark config: B=42, L=1024 (32×32 spatial), 200 warmup + 1000 measured iterations.

Usage:
    cd /workspace/MoDiff
    python integration/benchmarks/bench_attn_kernel.py
"""

import sys, math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/MoDiff")

# ── Extension probe ────────────────────────────────────────────────────────
try:
    import modiff_cutlass as mc
    HAS_CUTLASS        = True
    HAS_FUSED_CAST     = hasattr(mc, 'fp16_ncw_to_fp32_cl')
    HAS_FUSED_DELTA    = hasattr(mc, 'fp16_ncw_delta_to_int8_cl')
    HAS_SCALE_QUANT_I8 = hasattr(mc, 'scale_quantize_int8')
    HAS_SCALE_QUANT_I4 = hasattr(mc, 'scale_quantize_and_pack')
    HAS_STEP1_STATIC   = hasattr(mc, 'step1_static_quantize_fprop')
    HAS_STEP1_INT4     = hasattr(mc, 'step1_static_quantize_pack_int4_fprop')
    HAS_CONV_FPROP_I8  = hasattr(mc, 'conv2d_int8_fprop')
    HAS_CONV_FPROP_I4  = hasattr(mc, 'conv2d_int4_fprop')
    HAS_CONV_OHAT_I8   = hasattr(mc, 'conv2d_int8_fprop_o_hat')
    HAS_CONV_OHAT_I4   = hasattr(mc, 'conv2d_int4_fprop_o_hat')
except ImportError:
    HAS_CUTLASS = False
    HAS_FUSED_CAST = HAS_FUSED_DELTA = False
    HAS_SCALE_QUANT_I8 = HAS_SCALE_QUANT_I4 = False
    HAS_STEP1_STATIC = HAS_STEP1_INT4 = False
    HAS_CONV_FPROP_I8 = HAS_CONV_FPROP_I4 = False
    HAS_CONV_OHAT_I8 = HAS_CONV_OHAT_I4 = False
    print("WARNING: modiff_cutlass not found — INT8/INT4 paths skipped")

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE = 'cuda'
B, L   = 42, 1024   # batch=42; 32×32 spatial flattened = 1024
WARMUP = 200
ITERS  = 1000

# (C_in, C_out, label, n_blocks)
# n_blocks = number of AttentionBlocks with this C; each block has qkv + proj_out
SHAPES = [
    (192,  576, "C192→qkv",   5),
    (192,  192, "C192→proj",  5),
    (384, 1152, "C384→qkv",  10),
    (384,  384, "C384→proj", 10),
    (768, 2304, "C768→qkv",   6),
    (768,  768, "C768→proj",  6),
]

# ── Timing helper ──────────────────────────────────────────────────────────
def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(ITERS):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / ITERS * 1000   # µs per call


# ── Per-shape benchmark ────────────────────────────────────────────────────
def bench_one(C_in: int, C_out: int) -> dict:
    res = {}
    x32 = torch.randn(B, C_in, L, device=DEVICE, dtype=torch.float32)
    x16 = x32.half()

    # FP32/FP16 weights (no bias, matches AttentionBlock projection style)
    sc  = 1.0 / math.sqrt(C_in)
    w32 = torch.randn(C_out, C_in, 1, device=DEVICE) * sc
    w16 = w32.half()

    # ── FP32 F.conv1d ─────────────────────────────────────────────────────
    res['fp32'] = bench(lambda: F.conv1d(x32, w32))

    # ── FP16 F.conv1d ─────────────────────────────────────────────────────
    res['fp16'] = bench(lambda: F.conv1d(x16, w16))

    if not HAS_CUTLASS:
        return res

    # ── Shared CUTLASS setup ───────────────────────────────────────────────
    # INT8 weights: [C_out, C_in] → INT8 → reshape to KRSC [C_out,1,1,C_in]
    w2d        = w32.view(C_out, C_in)
    w_amax     = float(w2d.abs().max().clamp(1e-8))
    w_scale_v  = w_amax / 127.0
    w_i8       = (w2d / w_scale_v).round_().clamp_(-127, 127).to(torch.int8)
    w_i8_krsc  = w_i8.view(C_out, 1, 1, C_in).contiguous()
    w_scale_ch    = torch.full((C_out,), w_scale_v, device=DEVICE)           # [C_out] for o_hat
    w_scale_ch_4d = w_scale_ch.view(1, C_out, 1, 1)                          # [1,C_out,1,1] for multiply

    # INT4 weights: pack 2×INT4 per byte → KRSC [C_out,1,1,C_in//2]
    if C_in % 2 == 0 and C_out % 2 == 0:
        w_i4_raw = (w2d / (w_amax / 7.0)).round_().clamp_(-7, 7).to(torch.int8)
        # Pack pairs of INT4 values: lo = w[i], hi = w[i+1]
        w_i4_pairs = w_i4_raw.view(C_out, C_in // 2, 2)
        w_i4_packed = ((w_i4_pairs[:, :, 0] & 0x0F) |
                       ((w_i4_pairs[:, :, 1] & 0x0F) << 4)).to(torch.int8)
        w_i4_krsc  = w_i4_packed.view(C_out, 1, 1, C_in // 2).contiguous()
        w4_scale_v = w_amax / 7.0
        w4_scale_ch    = torch.full((C_out,), w4_scale_v, device=DEVICE)
        w4_scale_ch_4d = w4_scale_ch.view(1, C_out, 1, 1)
    else:
        w_i4_krsc = None

    # Static activation scale
    act_amax   = float(x32.abs().max().clamp(1e-8))
    act_s_i8   = 127.0 / act_amax          # static_input_scale for INT8
    act_s_i4   = 7.0   / act_amax          # static_input_scale for INT4
    act_sc_i8  = torch.tensor([act_s_i8],   device=DEVICE)
    act_sc_i4  = torch.tensor([act_s_i4],   device=DEVICE)
    alpha_i8   = torch.tensor([1.0 / act_s_i8], device=DEVICE)
    alpha_i4   = torch.tensor([1.0 / act_s_i4], device=DEVICE)

    empty_bias   = torch.empty(0, device=DEVICE)
    empty_smooth = torch.empty(0, device=DEVICE)

    # ── K1+K2: FP16 NCW → FP32 CL ────────────────────────────────────────
    if HAS_FUSED_CAST:
        res['k1k2'] = bench(lambda: mc.fp16_ncw_to_fp32_cl(x16, B, C_in, L))

    # ── K7+K8: FP32 CL → FP16 NCW (measured on C_out channels, output side) ─
    if HAS_FUSED_CAST:
        # Simulate the C_out-channel output as produced by the GEMM
        x_out_cl = torch.randn(B * L, C_out, 1, 1, device=DEVICE).contiguous(
            memory_format=torch.channels_last)
        res['k7k8'] = bench(lambda: mc.fp32_cl_to_fp16_ncw(x_out_cl, B, C_out, L))

    # Build CL input for GEMM benchmarks
    if HAS_FUSED_CAST:
        x32_cl = mc.fp16_ncw_to_fp32_cl(x16, B, C_in, L)   # [B*L, C_in, 1,1] FP32 CL
    else:
        x32_cl = x32.permute(0, 2, 1).reshape(B * L, C_in, 1, 1).contiguous(
            memory_format=torch.channels_last)

    # ── K4 INT8 GEMM only ─────────────────────────────────────────────────
    if HAS_SCALE_QUANT_I8 and HAS_CONV_FPROP_I8:
        x_i8_cl = mc.scale_quantize_int8(x32_cl, act_sc_i8)
        def k4_i8_only():
            out = mc.conv2d_int8_fprop(x_i8_cl, w_i8_krsc, alpha_i8, empty_bias,
                                       1, 1, 0, 0, 1, 1)
            return out * w_scale_ch_4d
        res['k4_int8'] = bench(k4_i8_only)

    # ── K4 INT4 GEMM only ─────────────────────────────────────────────────
    if HAS_SCALE_QUANT_I4 and HAS_CONV_FPROP_I4 and w_i4_krsc is not None:
        x_i4_cl = mc.scale_quantize_and_pack(x32_cl, act_sc_i4)
        def k4_i4_only():
            out = mc.conv2d_int4_fprop(x_i4_cl, w_i4_krsc, alpha_i4, empty_bias,
                                       1, 1, 0, 0, 1, 1)
            return out * w4_scale_ch_4d
        res['k4_int4'] = bench(k4_i4_only)

    # ── INT8 full pipeline, WITHOUT MoDiff ────────────────────────────────
    # K1+K2 → K3(scale_quantize) → K4(GEMM) → K5(per-ch scale) → K7+K8
    if HAS_FUSED_CAST and HAS_SCALE_QUANT_I8 and HAS_CONV_FPROP_I8:
        def int8_no_modiff():
            x_cl   = mc.fp16_ncw_to_fp32_cl(x16, B, C_in, L)
            x_i8   = mc.scale_quantize_int8(x_cl, act_sc_i8)
            out_cl = mc.conv2d_int8_fprop(x_i8, w_i8_krsc, alpha_i8, empty_bias,
                                          1, 1, 0, 0, 1, 1)
            out_cl = out_cl * w_scale_ch_4d
            return mc.fp32_cl_to_fp16_ncw(out_cl, B, C_out, L)
        res['int8_no_modiff'] = bench(int8_no_modiff)

    # ── INT8 full pipeline, WITH MoDiff (delta caching) ───────────────────
    # K1+K2+K3(fused delta + a_hat update) → K4(GEMM + accumulate) → K7+K8
    if HAS_FUSED_CAST and HAS_CONV_OHAT_I8:
        o_hat_i8 = torch.zeros(B * L, C_out, 1, 1, device=DEVICE).contiguous(
                               memory_format=torch.channels_last)
        if HAS_FUSED_DELTA:
            # Single fused K1+K2+K3 kernel
            a_hat_i8 = torch.zeros(B * L, C_in, 1, 1, device=DEVICE).contiguous(
                                   memory_format=torch.channels_last)
            def int8_modiff():
                x_i8 = mc.fp16_ncw_delta_to_int8_cl(
                    x16, a_hat_i8, act_sc_i8, B, C_in, L)
                mc.conv2d_int8_fprop_o_hat(
                    x_i8, w_i8_krsc, alpha_i8, w_scale_ch, o_hat_i8,
                    1, 1, 0, 0, 1, 1)
                return mc.fp32_cl_to_fp16_ncw(o_hat_i8, B, C_out, L)
        elif HAS_STEP1_STATIC:
            # Separate K1+K2 then K3
            a_hat_i8 = torch.zeros(B * L, C_in, 1, 1, device=DEVICE).contiguous(
                                   memory_format=torch.channels_last)
            def int8_modiff():
                x_cl = mc.fp16_ncw_to_fp32_cl(x16, B, C_in, L)
                x_i8 = mc.step1_static_quantize_fprop(
                    x_cl, a_hat_i8, act_sc_i8, empty_smooth)
                mc.conv2d_int8_fprop_o_hat(
                    x_i8, w_i8_krsc, alpha_i8, w_scale_ch, o_hat_i8,
                    1, 1, 0, 0, 1, 1)
                return mc.fp32_cl_to_fp16_ncw(o_hat_i8, B, C_out, L)
        else:
            int8_modiff = None
        if int8_modiff is not None:
            res['int8_modiff'] = bench(int8_modiff)

    # ── INT4 full pipeline, WITHOUT MoDiff ────────────────────────────────
    # K1+K2 → K3(scale_quantize_and_pack) → K4(GEMM) → K5 → K7+K8
    if HAS_FUSED_CAST and HAS_SCALE_QUANT_I4 and HAS_CONV_FPROP_I4 and w_i4_krsc is not None:
        def int4_no_modiff():
            x_cl    = mc.fp16_ncw_to_fp32_cl(x16, B, C_in, L)
            x_i4    = mc.scale_quantize_and_pack(x_cl, act_sc_i4)
            out_cl  = mc.conv2d_int4_fprop(x_i4, w_i4_krsc, alpha_i4, empty_bias,
                                           1, 1, 0, 0, 1, 1)
            out_cl  = out_cl * w4_scale_ch_4d
            return mc.fp32_cl_to_fp16_ncw(out_cl, B, C_out, L)
        res['int4_no_modiff'] = bench(int4_no_modiff)

    # ── INT4 full pipeline, WITH MoDiff ───────────────────────────────────
    # K1+K2 → K3(delta+pack) → K4(GEMM+acc) → K7+K8
    if HAS_FUSED_CAST and HAS_STEP1_INT4 and HAS_CONV_OHAT_I4 and w_i4_krsc is not None:
        a_hat_i4 = torch.zeros(B * L, C_in, 1, 1, device=DEVICE).contiguous(
                               memory_format=torch.channels_last)
        o_hat_i4 = torch.zeros(B * L, C_out, 1, 1, device=DEVICE).contiguous(
                               memory_format=torch.channels_last)
        def int4_modiff():
            x_cl = mc.fp16_ncw_to_fp32_cl(x16, B, C_in, L)
            x_i4 = mc.step1_static_quantize_pack_int4_fprop(
                x_cl, a_hat_i4, act_sc_i4, empty_smooth)
            mc.conv2d_int4_fprop_o_hat(
                x_i4, w_i4_krsc, alpha_i4, w4_scale_ch, o_hat_i4,
                1, 1, 0, 0, 1, 1)
            return mc.fp32_cl_to_fp16_ncw(o_hat_i4, B, C_out, L)
        res['int4_modiff'] = bench(int4_modiff)

    return res


# ── Formatting helpers ─────────────────────────────────────────────────────
COLUMN_ORDER = [
    'fp32', 'fp16',
    'k1k2', 'k7k8',
    'k4_int8', 'k4_int4',
    'int8_no_modiff', 'int8_modiff',
    'int4_no_modiff', 'int4_modiff',
]
COLUMN_LABEL = {
    'fp32':           'FP32 conv1d',
    'fp16':           'FP16 conv1d',
    'k1k2':           'K1+K2 (cast in)',
    'k7k8':           'K7+K8 (cast out)',
    'k4_int8':        'K4 INT8 GEMM',
    'k4_int4':        'K4 INT4 GEMM',
    'int8_no_modiff': 'INT8  no-MoDiff',
    'int8_modiff':    'INT8  w/ MoDiff',
    'int4_no_modiff': 'INT4  no-MoDiff',
    'int4_modiff':    'INT4  w/ MoDiff',
}

def fµs(v):
    return f"{v:7.1f}" if v is not None else f"{'—':>7}"

def fspd(v, base):
    if v is None or base is None or v == 0:
        return f"{'—':>6}"
    return f"{base / v:6.2f}×"


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    prop = torch.cuda.get_device_properties(0)
    print()
    print("=" * 76)
    print(" Attention Conv1d Kernel Microbenchmark — Isolated per-call latency")
    print(f" GPU : {prop.name}  (sm_{prop.major}{prop.minor})")
    print(f" B={B}, L={L} (32×32 spatial),  warmup={WARMUP}, iters={ITERS}")
    print("=" * 76)

    print("\n Extension capabilities detected:")
    cap = {
        "K1+K2 (fp16_ncw_to_fp32_cl)":        HAS_FUSED_CAST,
        "K1+K2+K3 (fp16_ncw_delta_to_int8)":  HAS_FUSED_DELTA,
        "INT8 scale_quantize_int8":            HAS_SCALE_QUANT_I8,
        "INT8 step1_static_quantize_fprop":    HAS_STEP1_STATIC,
        "INT8 conv2d_int8_fprop":              HAS_CONV_FPROP_I8,
        "INT8 conv2d_int8_fprop_o_hat":        HAS_CONV_OHAT_I8,
        "INT4 scale_quantize_and_pack":        HAS_SCALE_QUANT_I4,
        "INT4 step1_static_quantize_pack":     HAS_STEP1_INT4,
        "INT4 conv2d_int4_fprop":              HAS_CONV_FPROP_I4,
        "INT4 conv2d_int4_fprop_o_hat":        HAS_CONV_OHAT_I4,
    }
    for name, ok in cap.items():
        print(f"   {'✓' if ok else '✗'}  {name}")

    print()

    # ── Run benchmarks ────────────────────────────────────────────────────
    all_res = {}
    for C_in, C_out, label, _ in SHAPES:
        print(f"  Benchmarking {label:<12} ({C_in:>3}→{C_out:>4}) ...", flush=True, end="")
        all_res[label] = (C_in, C_out, bench_one(C_in, C_out))
        print(" done")

    # ── Per-layer detailed tables ─────────────────────────────────────────
    print()
    for C_in, C_out, label, _ in SHAPES:
        _, _, res = all_res[label]
        fp32_t = res.get('fp32')
        fp16_t = res.get('fp16')
        macs   = 2 * B * L * C_in * C_out
        print()
        print(f"  ┌─ {label}  [B={B}, C_in={C_in}, C_out={C_out}, L={L}]  "
              f"MACs: {macs / 1e9:.2f} G")
        print(f"  │  {'Mode':<22}  {'µs/call':>7}  {'vs fp32':>8}  {'vs fp16':>8}")
        print(f"  │  {'─'*52}")
        for mode in COLUMN_ORDER:
            if mode not in res:
                continue
            t = res[mode]
            print(f"  │  {COLUMN_LABEL[mode]:<22}  {fµs(t)}  {fspd(t, fp32_t)}  {fspd(t, fp16_t)}")
        print(f"  └{'─'*55}")

    # ── Summary table: µs per call ────────────────────────────────────────
    present = [m for m in COLUMN_ORDER
               if any(m in r for _, _, r in all_res.values())]
    CW, LW = 18, 13

    def summary_table(title, value_fn):
        print()
        print(f"  {title}")
        hdr = f"  {'Layer':<{LW}}" + "".join(f"{COLUMN_LABEL[m]:>{CW}}" for m in present)
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for C_in, C_out, label, _ in SHAPES:
            _, _, res = all_res[label]
            row = f"  {label:<{LW}}"
            for m in present:
                v = value_fn(res.get(m), res.get('fp16'))
                row += f"{v:>{CW}}"
            print(row)

    print()
    print("=" * 76)
    print(" SUMMARY TABLES")
    print("=" * 76)

    summary_table(
        "Latency (µs per call)",
        lambda t, fp16: fµs(t).strip() if t is not None else "—"
    )

    summary_table(
        "Speedup relative to FP16 F.conv1d (higher is faster)",
        lambda t, fp16: fspd(t, fp16).strip() if t is not None and fp16 else "—"
    )

    # ── Per-kernel breakdown ───────────────────────────────────────────────
    print()
    print("=" * 76)
    print(" KERNEL BREAKDOWN — avg µs across all 6 layers")
    print("=" * 76)

    for mode in present:
        vals = [all_res[lbl][2][mode]
                for *_, lbl, _ in [(None, None, lbl, None) for *_, lbl, _ in SHAPES]
                if mode in all_res[lbl][2]]
        # rebuild properly
        vals = []
        for C_in, C_out, label, _ in SHAPES:
            v = all_res[label][2].get(mode)
            if v is not None:
                vals.append(v)
        if vals:
            print(f"  {COLUMN_LABEL[mode]:<22}: avg={sum(vals)/len(vals):6.1f} µs  "
                  f"min={min(vals):6.1f}  max={max(vals):6.1f}")

    # ── Pipeline overhead vs raw FP16 ─────────────────────────────────────
    print()
    print("=" * 76)
    print(" PIPELINE AGGREGATION — total for all 42 Conv1d calls per diffusion step")
    print(" (21 AttentionBlocks × 2 projections each: qkv + proj_out)")
    print("=" * 76)

    # n_blocks × 1 call each for qkv and proj_out
    BLOCK_COUNTS = {192: 5, 384: 10, 768: 6}
    SHAPE_MAP = {}
    for C_in, C_out, label, n_blocks in SHAPES:
        SHAPE_MAP[label] = n_blocks

    for mode in present:
        total_µs = 0.0
        for C_in, C_out, label, n_blocks in SHAPES:
            t = all_res[label][2].get(mode)
            if t is not None:
                total_µs += t * n_blocks
        if total_µs > 0:
            total_ms = total_µs / 1000.0
            print(f"  {COLUMN_LABEL[mode]:<22}: {total_ms:.3f} ms / step")

    print()
    print("  Reference full-pipeline measurements (200-step benchmark, bs=42):")
    print("    fp32            : 4.52 ms/step   (incl. VAE + all UNet ops)")
    print("    fp16            : 1.63 ms/step")
    print("    int8_attn_modiff: 2.10 ms/step")
    print("    int4_attn_modiff: 2.00 ms/step")
    print()
    print("  Note: aggregated single-layer timings underestimate real pipeline cost;")
    print("  full-pipeline includes LayerNorm, softmax, non-attn UNet layers, etc.")

    print()
    print("=" * 76)
    print(" LEGEND")
    print("=" * 76)
    print("  K1+K2  : fp16 [B,C,L] NCW → fp32 [B*L,C,1,1] channels-last")
    print("           (tiled shared-memory transpose + dtype cast, single CUDA kernel)")
    print("  K7+K8  : fp32 [B*L,C,1,1] CL → fp16 [B,C,L] NCW")
    print("           (tiled shared-memory transpose + dtype cast, single CUDA kernel)")
    print("  K4     : CUTLASS INT8/INT4 GEMM + dequantize (no layout overhead)")
    print()
    print("  INT8 no-MoDiff : K1+K2 → K3(scale+quant) → K4(GEMM) → K5(scale) → K7+K8")
    print("  INT8 w/MoDiff  : K1+K2+K3(fused delta,update_ahat) → K4(GEMM+acc) → K7+K8")
    print("  INT4 no-MoDiff : K1+K2 → K3(scale+quant+pack) → K4(GEMM) → K5 → K7+K8")
    print("  INT4 w/MoDiff  : K1+K2 → K3(delta+pack,update_ahat) → K4(GEMM+acc) → K7+K8")
    print()
    print("  MoDiff temporal delta caching: instead of quantizing the full activation,")
    print("  quantize only the temporal difference (input_t - a_hat_{t+1}) and accumulate")
    print("  into o_hat cache. Reduces quantization error for temporally coherent inputs.")
    print("  Reference: Gao et al. 'MoDiff' ICML 2025.")
    print()


if __name__ == '__main__':
    main()
