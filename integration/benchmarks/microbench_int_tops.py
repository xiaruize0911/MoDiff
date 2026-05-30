"""
Microbenchmark: INT8 vs INT4 conv2d TOPS on the current GPU.

Sections:
  1. Per-layer TOPS at batch=32 (original experiment)
  2. Batch-size sweep on three representative layers — shows how batch size
     transitions a layer from latency-bound → compute-bound

Usage:
    cd /workspace/MoDiff
    python integration/benchmarks/microbench_int_tops.py
"""
import sys
import os
import time
import torch
import numpy as np

# ── locate modiff_cutlass ──────────────────────────────────────────────────
sys.path.insert(0, "/workspace/MoDiff")
import modiff_cutlass

# ── GPU info ───────────────────────────────────────────────────────────────
prop = torch.cuda.get_device_properties(0)
gpu_name = prop.name
sm_major = prop.major
sm_minor = prop.minor
# GDDR6 uses 8 transfers per base-clock cycle (WCK×4 + DDR×2)
# so effective_BW = 8 × base_clock_kHz × 1000 × bus_width_bits/8 / 1e9 GB/s
bw_gbps = 8 * prop.memory_clock_rate * 1000 * prop.memory_bus_width / 8 / 1e9
print(f"GPU: {gpu_name}  (sm_{sm_major}{sm_minor})")
print(f"SMs: {prop.multi_processor_count}  |  "
      f"Clock: {prop.clock_rate/1e6:.2f} GHz  |  "
      f"Mem BW: {bw_gbps:.0f} GB/s"
      f"  (base clock {prop.memory_clock_rate/1e3:.0f} MHz × {prop.memory_bus_width}-bit)")

# A40 theoretical peak (from NVIDIA datasheet)
A40_SPECS = {
    "FP32 (no TC)":  37.4,   # TFLOPS
    "FP16 TC":      149.7,   # TFLOPS
    "INT8 TC":      299.3,   # TOPS
    "INT4 TC":      597.7,   # TOPS  (2× INT8 per NVIDIA datasheet)
}
print("\nA40 theoretical peak TOPS (NVIDIA datasheet):")
for k, v in A40_SPECS.items():
    print(f"  {k:16s}: {v:6.1f} {'TFLOPS' if 'F' in k else 'TOPS'}")

# ── helper: prepare INT8 buffers ───────────────────────────────────────────
def make_int8_buffers(N, C_in, H, W, C_out, kH=3, kW=3):
    # Input: channels_last (N, C, H, W) in NHWC physical layout, int8
    act = torch.randint(-127, 127, (N, C_in, H, W), dtype=torch.int8, device="cuda"
                        ).to(memory_format=torch.channels_last)
    # Weight: (K, R, S, C) contiguous — KRSC layout expected by CUTLASS
    wgt  = torch.randint(-127, 127, (C_out, kH, kW, C_in), dtype=torch.int8, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    bias  = torch.empty(0, dtype=torch.float32, device="cuda")
    return act, wgt, scale, bias

def make_int4_buffers(N, C_in, H, W, C_out, kH=3, kW=3):
    # Input x_packed: (N, H, W, C/2) contiguous int8 — packed INT4
    assert C_in % 2 == 0 and C_out % 2 == 0
    act  = torch.randint(-128, 127, (N, H, W, C_in//2),  dtype=torch.int8, device="cuda").contiguous()
    # Weight: (K, R, S, C/2) contiguous int8 — packed INT4 KRSC/2 layout
    wgt  = torch.randint(-128, 127, (C_out, kH, kW, C_in//2), dtype=torch.int8, device="cuda").contiguous()
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    bias  = torch.empty(0, dtype=torch.float32, device="cuda")
    return act, wgt, scale, bias

# ── TOPS calculation ───────────────────────────────────────────────────────
def conv_ops(N, C_in, H_out, W_out, C_out, kH=3, kW=3):
    """Multiply-accumulates = N * H_out * W_out * C_out * C_in * kH * kW
       Each MAC = 2 ops (mul + add).
    """
    macs = N * H_out * W_out * C_out * C_in * kH * kW
    return 2 * macs

# ── benchmark helper ───────────────────────────────────────────────────────
def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters  # seconds per call

# ── layer shapes from LSUN-churches LDM UNet ──────────────────────────────
# Only layers with C_in >= 64 (CUTLASS INT8/INT4 require K ≥ instruction tile)
# The first conv (C_in=4) and last conv (C_out=4) are never quantized.
SHAPES = [
    # name,             N,  C_in,  H,  W,  C_out, kH, kW,  stride, pad
    ("res_128x128",     32, 128,   32, 32,  128,   3,  3,   1,  1),
    ("res_128x256_ds",  32, 128,   32, 32,  256,   3,  3,   2,  1),
    ("res_256x256",     32, 256,   16, 16,  256,   3,  3,   1,  1),
    ("res_256x512_ds",  32, 256,   16, 16,  512,   3,  3,   2,  1),
    ("mid_512x512",     32, 512,    8,  8,  512,   3,  3,   1,  1),
    ("mid_512x512_2",   32, 512,    8,  8,  512,   3,  3,   1,  1),
    ("res_512x256_us",  32, 512,   16, 16,  256,   3,  3,   1,  1),
    ("res_256x128_us",  32, 256,   32, 32,  128,   3,  3,   1,  1),
    ("res_128x128_us",  32, 128,   64, 64,  128,   3,  3,   1,  1),
]

print("\n" + "="*100)
print(f"{'Layer':<22} {'Shape (N,Ci,H,W→Co)':<28} {'INT8 ms':>8} {'INT4 ms':>8} "
      f"{'INT8 TOPS':>10} {'INT4 TOPS':>10} {'INT4/INT8':>10} {'BW ratio':>10}")
print("="*100)

int8_tops_all = []
int4_tops_all = []

for (name, N, Cin, H, W, Cout, kH, kW, stride, pad) in SHAPES:
    Hout = (H + 2*pad - kH) // stride + 1
    Wout = (W + 2*pad - kW) // stride + 1
    ops  = conv_ops(N, Cin, Hout, Wout, Cout, kH, kW)

    # ── INT8 ──
    act8, wgt8, scale8, bias8 = make_int8_buffers(N, Cin, H, W, Cout, kH, kW)

    def run_int8():
        modiff_cutlass.conv2d_int8_fprop(
            act8, wgt8, scale8, bias8,
            stride, stride,
            pad, pad,
            1, 1  # dilation
        )

    t8 = bench(run_int8) * 1000  # ms
    tops8 = ops / (t8 * 1e-3) / 1e12

    # ── INT4 ──
    act4, wgt4, scale4, bias4 = make_int4_buffers(N, Cin, H, W, Cout, kH, kW)

    def run_int4():
        modiff_cutlass.conv2d_int4_fprop(
            act4, wgt4, scale4, bias4,
            stride, stride,
            pad, pad,
            1, 1  # dilation
        )

    t4 = bench(run_int4) * 1000  # ms
    tops4 = ops / (t4 * 1e-3) / 1e12

    ratio  = tops4 / tops8
    # bytes read: INT8 = Cin*N*H*W + Cout*kH*kW*Cin, INT4 = half of that
    bw_ratio = (Cin*N*H*W + Cout*kH*kW*Cin) / ((Cin//2)*N*H*W + Cout*kH*kW*(Cin//2))

    int8_tops_all.append(tops8)
    int4_tops_all.append(tops4)

    print(f"{name:<22} N={N} ({Cin},{H},{W})→{Cout}  "
          f"{t8:>8.3f} {t4:>8.3f} "
          f"{tops8:>10.2f} {tops4:>10.2f} {ratio:>10.2f}x {bw_ratio:>9.1f}x")

print("="*100)
avg8 = np.mean(int8_tops_all)
avg4 = np.mean(int4_tops_all)
print(f"{'MEAN':<22} {'':28} {'':8} {'':8} "
      f"{avg8:>10.2f} {avg4:>10.2f} {avg4/avg8:>10.2f}x")

print(f"\n{'% of A40 peak INT8 TOPS achieved':>40}: {avg8/A40_SPECS['INT8 TC']*100:.1f}%")
print(f"{'% of A40 peak INT4 TOPS achieved':>40}: {avg4/A40_SPECS['INT4 TC']*100:.1f}%")

# ── Memory bandwidth bound analysis ───────────────────────────────────────
print("\n" + "="*100)
print("ROOFLINE ANALYSIS")
print("="*100)
print(f"  A40 INT8  peak compute : {A40_SPECS['INT8 TC']:.1f} TOPS")
print(f"  A40 INT4  peak compute : {A40_SPECS['INT4 TC']:.1f} TOPS")
# GDDR6: effective_BW = 8 × base_clock_kHz × 1000 × bus_width_bits/8 / 1e9 GB/s
bw_gbps_roofline = 8 * prop.memory_clock_rate * 1000 * prop.memory_bus_width / 8 / 1e9
print(f"  A40 memory bandwidth   : {bw_gbps_roofline:.0f} GB/s")
# arithmetic intensity: ops / bytes
print(f"  {'Layer':<22} {'INT8 OPs/Byte':>14} {'INT4 OPs/Byte':>14} "
      f"{'INT8 BW-limit TOPS':>18} {'INT4 BW-limit TOPS':>18}")
for (name, N, Cin, H, W, Cout, kH, kW, stride, pad) in SHAPES:
    Hout = (H + 2*pad - kH) // stride + 1
    Wout = (W + 2*pad - kW) // stride + 1
    ops = conv_ops(N, Cin, Hout, Wout, Cout, kH, kW)
    # bytes read/written: activations + weights + output (fp32)
    bytes_int8  = N*H*W*Cin + Cout*kH*kW*Cin + N*Hout*Wout*Cout*4   # act+wgt in int8, out in fp32
    bytes_int4  = N*H*W*(Cin//2) + Cout*kH*kW*(Cin//2) + N*Hout*Wout*Cout*4
    ai8 = ops / bytes_int8
    ai4 = ops / bytes_int4
    bw_top8 = bw_gbps_roofline * ai8 / 1e3   # TOPS if bandwidth-bound
    bw_top4 = bw_gbps_roofline * ai4 / 1e3
    print(f"  {name:<22} {ai8:>14.1f} {ai4:>14.1f} {bw_top8:>18.2f} {bw_top4:>18.2f}")

print()
print("  If measured TOPS << bandwidth-limited TOPS: compute-bound")
print("  If measured TOPS ≈  bandwidth-limited TOPS: memory-bandwidth bound")
print()
print("  Key insight: INT4 uses 2× fewer bytes for weights/activations than INT8.")
print("  If the layer is memory-bandwidth bound, INT4 SHOULD be ~2× faster.")
print("  If INT4/INT8 ≈ 1×, it means the kernel is NOT reaching theoretical throughput")
print("  (likely due to: instruction pipeline, epilogue overhead, or sub-optimal tiling).")

# ==========================================================================
# SECTION 2: Batch-size sweep
# ==========================================================================
# Three representative layers: small spatial (mid block), mid (256-ch), large (upsampler)
SWEEP_LAYERS = [
    # name,              Cin,  H,  W, Cout, kH, kW, stride, pad
    ("mid_512x512",      512,   8,  8,  512,  3,  3,   1,  1),  # compute-heavy, small spatial
    ("res_256x256",      256,  16, 16,  256,  3,  3,   1,  1),  # mid-size
    ("res_128x128",      128,  32, 32,  128,  3,  3,   1,  1),  # large spatial, low channels
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 21, 42, 84]

print("\n\n" + "="*120)
print("SECTION 2 — BATCH SIZE SWEEP: Effect of batch size on INT8 vs INT4 throughput")
print("="*120)
print(f"  INT8 peak: {A40_SPECS['INT8 TC']:.0f} TOPS  |  INT4 peak: {A40_SPECS['INT4 TC']:.0f} TOPS")
print()

for (lname, Cin, H, W, Cout, kH, kW, stride, pad) in SWEEP_LAYERS:
    Hout = (H + 2*pad - kH) // stride + 1
    Wout = (W + 2*pad - kW) // stride + 1

    print(f"  Layer: {lname}  ({Cin}×{H}×{W} → {Cout}, 3×3, s={stride})")
    print(f"  {'Batch':>6}  {'INT8 ms':>9}  {'INT4 ms':>9}  "
          f"{'INT8 TOPS':>10}  {'INT4 TOPS':>10}  {'INT4/INT8':>10}  "
          f"{'INT8 % peak':>12}  {'INT4 % peak':>12}  {'Regime':>16}")
    print("  " + "-"*110)

    for N in BATCH_SIZES:
        ops = conv_ops(N, Cin, Hout, Wout, Cout, kH, kW)

        act8, wgt8, scale8, bias8 = make_int8_buffers(N, Cin, H, W, Cout, kH, kW)
        def run_int8():
            modiff_cutlass.conv2d_int8_fprop(
                act8, wgt8, scale8, bias8,
                stride, stride, pad, pad, 1, 1)
        t8 = bench(run_int8, warmup=10, iters=100) * 1000

        act4, wgt4, scale4, bias4 = make_int4_buffers(N, Cin, H, W, Cout, kH, kW)
        def run_int4():
            modiff_cutlass.conv2d_int4_fprop(
                act4, wgt4, scale4, bias4,
                stride, stride, pad, pad, 1, 1)
        t4 = bench(run_int4, warmup=10, iters=100) * 1000

        tops8 = ops / (t8 * 1e-3) / 1e12
        tops4 = ops / (t4 * 1e-3) / 1e12
        pct8  = tops8 / A40_SPECS["INT8 TC"] * 100
        pct4  = tops4 / A40_SPECS["INT4 TC"] * 100
        ratio = tops4 / tops8

        # Classify regime based on INT8 peak utilization
        if pct8 > 40:
            regime = "compute-bound"
        elif pct8 > 15:
            regime = "transitioning"
        else:
            regime = "latency-bound"

        print(f"  {N:>6}  {t8:>9.3f}  {t4:>9.3f}  "
              f"{tops8:>10.1f}  {tops4:>10.1f}  {ratio:>10.2f}x  "
              f"{pct8:>11.1f}%  {pct4:>11.1f}%  {regime:>16}")
    print()
