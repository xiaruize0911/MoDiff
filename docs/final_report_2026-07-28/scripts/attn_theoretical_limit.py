"""Derive the theoretical floor for quantized flash attention from MEASURED unit peaks.

Why measured and not datasheet: the binding constraint for this kernel turns out not to be the
tensor core, and two of the units behave differently from what the spec sheet implies on this
card (fp16x2 is only 1.24x fp32, not 2x; HMAX2 issues *slower* per instruction than FMNMX, so
half2-packing the softmax is nearly worthless here). A floor built on datasheet ratios would
point the optimization work in the wrong direction.

A flash-attention kernel has five independent floors; the real floor is the max:

  1. tensor core -- 2*T^2*hd_pad MACs (QK^T + PV). hd_pad is what the mma actually issues, so
                    padding waste is part of the floor, not something an implementation can
                    optimize away (m16n8k32 has no shorter K).
  2. SFU         -- exactly T^2 ex2 instructions, one per score element. fp32-only, unpackable,
                    unskippable. On this card 2.21 T/s, which is the same order as the tensor
                    core term -- that is the whole story of why quantizing attention is hard.
  3. fp32 ALU    -- the irreducible online-softmax recurrence per score element:
                      m' = max(m, s)   |   p = exp2(s - m')   |   l = l*a + p   |   s = acc*scale
                    = 4 fp32 lane-ops. Nothing fuses these away.
  4. HBM         -- Q+K+V read once + O written once, assuming perfect L2 reuse of K/V across
                    the CTAs sharing an (n,h). Optimistic by construction.
  5. issue       -- 4 instructions/cycle/SM against an irreducible instruction budget.

Everything a real kernel does beyond items 1-5 (dequant scaling, P requantize, address
arithmetic, smem staging, loop control) is implementation overhead. "Approach the limit" means
drive the measured SASS instruction count down toward the item-5 budget.

Peaks come from scripts/unit_peaks.cu, compiled and run here so the numbers are this card's.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "data", "attn_theoretical_limit.json")
CU = os.path.join(HERE, "unit_peaks.cu")
BIN = "/tmp/claude-0/-workspace/dc9b6ff3-f870-42dd-9bc1-5e0353efa0e7/scratchpad/unit_peaks_bin"

# Shapes this model's AttentionBlocks actually use, plus two probes (hd=32, hd=64) that isolate
# the hd dependence. N=128 H=8 matches the benchmark batch.
SHAPES = [(128, 8, 1024, 24), (128, 8, 256, 48), (128, 8, 64, 48),
          (128, 8, 1024, 32), (128, 8, 1024, 64)]

# Current measured kernel times (docs/.../data/qattn_qhoist_bench.json) and PyTorch fp16 SDPA.
MEASURED_I8 = {(1024, 24): 2303.3, (256, 48): 296.2, (64, 48): 52.8,
               (1024, 32): 2469.9, (1024, 64): 4055.1}
MEASURED_I4 = {(1024, 24): 2277.5, (256, 48): 265.2, (64, 48): 44.2,
               (1024, 32): 2417.4, (1024, 64): 3400.6}
PT_FP16 = {(1024, 24): 1830.8, (256, 48): 226.8, (64, 48): 67.5,
           (1024, 32): 1593.6, (1024, 64): 2790.5}

FP32_OPS_PER_ELEM = 4          # see item 3 above
BR, BC = 16, 64                # kernel tile: rows per warp, keys per tile
ISSUE_PER_SM_PER_CYCLE = 4     # one per sub-partition


def peaks():
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < os.path.getmtime(CU):
        env = dict(os.environ)
        env.setdefault("CUDA_HOME", "/usr/local/cuda-12.4")
        subprocess.run([f"{env['CUDA_HOME']}/bin/nvcc", "-O3", "-arch=sm_86", "--std=c++17",
                        "--extended-lambda", "-o", BIN, CU], check=True, env=env)
    txt = subprocess.run([BIN], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^JSON (\{.*\})$", txt, re.M)
    if not m:
        raise RuntimeError(f"unit_peaks produced no JSON line:\n{txt}")
    return json.loads(m.group(1)), txt


def main():
    pk, raw = peaks()
    print(raw.split("=== 实测单元峰值 ===")[1].split("JSON")[0].strip())
    clock = 1.74e9                      # reported by unit_peaks
    issue_per_s = pk["sm"] * ISSUE_PER_SM_PER_CYCLE * clock

    rows = []
    print(f"\n各 shape 的理论下界 (us)。下界 = 五项之 max\n")
    print(f"{'T':>5s} {'hd':>3s} {'pad':>4s} | {'mma i8':>7s} {'mma i4':>7s} {'SFU':>7s} "
          f"{'fp32':>6s} {'HBM':>6s} {'issue':>6s} | {'底i8':>6s} {'底i4':>6s} | "
          f"{'实测i8':>7s} {'差':>5s} | {'实测i4':>7s} {'差':>5s} | {'PT':>7s} {'底/PT':>6s}")
    for N, H, T, hd in SHAPES:
        hp = ((hd + 31) // 32) * 32
        elems = N * H * T * T                       # score elements
        # 1. tensor core
        t_mma8 = 2 * elems * hp / pk["int8_mac"] * 1e6
        # int4 pads QK's K to 64 (m16n8k64 minimum); PV stays int8 in this design
        t_mma4 = (elems * 64 / pk["int4_mac"] + elems * hp / pk["int8_mac"]) * 1e6
        # 2. SFU
        t_sfu = elems / pk["ex2"] * 1e6
        # 3. fp32 ALU (FFMA TFLOPS counts 2 flop/instr -> lane-op rate is flop/2)
        t_f32 = elems * FP32_OPS_PER_ELEM / (pk["fp32_flop"] / 2) * 1e6
        # 4. HBM
        byts = 3 * N * H * T * hp + N * H * T * hd * 2
        t_hbm = byts / pk["hbm"] * 1e6
        # 5. issue: irreducible warp-instructions per (warp, key-tile)
        per_lane = (hp // 32) * (BC // 8) + (hd // 8) * (BC // 32)      # mma
        per_lane += BC // 2                                             # ex2, one per element
        per_lane += FP32_OPS_PER_ELEM * (BC // 2)                       # softmax recurrence
        per_lane += BC // 2 + (hd // 8) * 4                             # I2FP on both accumulators
        per_lane += (BC // 8) * (hp // 32) * 2 + (hd // 8) * 4          # K and V smem fragments
        per_lane += BC // 4 + (BC // 32) * 4                            # P store + reload
        n_wkt = (N * H * T / BR) * (T / BC)
        t_issue = per_lane * n_wkt / issue_per_s * 1e6
        lo8 = max(t_mma8, t_sfu, t_f32, t_hbm, t_issue)
        lo4 = max(t_mma4, t_sfu, t_f32, t_hbm, t_issue)
        m8, m4, pt = MEASURED_I8[(T, hd)], MEASURED_I4[(T, hd)], PT_FP16[(T, hd)]
        rows.append(dict(T=T, hd=hd, hd_pad=hp, t_mma_i8=t_mma8, t_mma_i4=t_mma4, t_sfu=t_sfu,
                         t_fp32=t_f32, t_hbm=t_hbm, t_issue=t_issue,
                         irreducible_instr_per_lane_per_tile=per_lane,
                         floor_i8=lo8, floor_i4=lo4, measured_i8=m8, measured_i4=m4,
                         gap_i8=m8 / lo8, gap_i4=m4 / lo4, pt_fp16=pt,
                         floor_vs_pt=pt / lo8))
        print(f"{T:5d} {hd:3d} {hp:4d} | {t_mma8:7.0f} {t_mma4:7.0f} {t_sfu:7.0f} "
              f"{t_f32:6.0f} {t_hbm:6.0f} {t_issue:6.0f} | {lo8:6.0f} {lo4:6.0f} | "
              f"{m8:7.0f} {m8/lo8:4.1f}x | {m4:7.0f} {m4/lo4:4.1f}x | {pt:7.0f} {pt/lo8:5.2f}x")

    with open(OUT, "w") as f:
        json.dump({"peaks": pk, "fp32_ops_per_score_element": FP32_OPS_PER_ELEM,
                   "issue_per_s": issue_per_s, "shapes": rows}, f, indent=2)
    print(f"\n底/PT = 理论下界相对 PyTorch fp16 的加速上限")
    print(f"WROTE {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
