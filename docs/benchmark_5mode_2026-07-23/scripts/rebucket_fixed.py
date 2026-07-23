"""Re-bucket the raw 145-kernel profile with a CORRECTED, order-safe cat():
quantize is tested BEFORE group_norm/gn_/conv, so fused GN+quant and delta-quant land in quantize
(not GroupNorm). Also splits quantize into standalone vs GN-fused for transparency. Pure CSV, no GPU.
Reads data/e2e_kernel_profile_raw.csv -> data/e2e_timing_profile_fixed.csv + prints."""
import csv
HERE = "/workspace/MoDiff/docs/benchmark_5mode_2026-07-23"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
WALL = {"fp16":187.11,"int8_baseline":114.86,"int4_baseline":104.20,"int8_modiff":124.35,"int4_modiff":120.92}

def cat(name):
    l = name.lower()
    # 1) attention: fused flash, fp16 softmax, fp16 QK/PV batched matmul (wmma)
    if "flash_attn" in l or "softmax" in l or "scaled_dot" in l or "wmma_tensorop_f16" in l:
        return "attention"
    # 2) quantize/dequant — TESTED BEFORE groupnorm/conv so fused kernels land here (the fix).
    #    fused GN+SiLU+quant and GN delta-quant are quant-family; broken out separately below.
    if "group_norm_silu_quantize" in l or "delta_quantize" in l or "delta_quant" in l:
        return "quantize (fused into GN)"
    if ("quantize" in l or "aq_qtok" in l or "aq_vquant" in l or "quant_act" in l
            or "quant_attn_out" in l or "update_ahat" in l or "dequant" in l
            or ("quant" in l and "gemm" not in l)):
        return "quantize (standalone)"
    # 3) qkv/proj int GEMM (AWQ W8A8 / W4A4)
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l:
        return "qkv/proj int GEMM"
    # 4) conv: int implicit-GEMM (EVT + plain) and fp16 cudnn/xmma fprop
    if ("implicitgemm" in l or "implicit_gemm" in l or "xmma_fprop" in l or "cudnn" in l
            or "scudnn" in l or "wgrad" in l or "convolution" in l or "fprop" in l):
        return "conv"
    # 5) other fp16 GEMM (fp16 linear/proj: cublas/cutlass f16 gemm)
    if ("s1688gemm" in l or "tensorop_f16" in l or "xmma_gemm" in l or "cublas" in l
            or "ampere_fp16" in l or ("gemm" in l and "f16" in l)):
        return "other fp16 GEMM"
    # 6) GroupNorm (pure: stats/apply/moments, NO quant)
    if ("group_norm" in l or "groupnorm" in l or "gn_group_stats" in l or "gn_accum" in l
            or "rowwisemoments" in l or "gn_" in l):
        return "GroupNorm (pure)"
    # 7) resample / concat
    if "upsample" in l or "interpolate" in l or "catarraybatched" in l.replace("_", "") or "avg_pool" in l:
        return "resample/concat"
    # 8) elementwise / copy
    if ("elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "fill" in l
            or "add" in l or "index" in l or "clamp" in l or "round" in l or "reduce" in l or "cat" in l):
        return "elementwise/copy"
    return "other"

BUCKETS = ["attention", "conv", "qkv/proj int GEMM", "other fp16 GEMM", "GroupNorm (pure)",
           "quantize (standalone)", "quantize (fused into GN)", "resample/concat", "elementwise/copy", "other"]
agg = {m: {b: 0.0 for b in BUCKETS} for m in MODES}
for r in csv.DictReader(open(f"{HERE}/data/e2e_kernel_profile_raw.csv")):
    b = cat(r["kernel"])
    for m in MODES:
        agg[m][b] += float(r[m])

rows = []
for m in MODES:
    busy = sum(agg[m].values())
    row = {"mode": m, **{b: round(agg[m][b], 2) for b in BUCKETS},
           "quantize_total": round(agg[m]["quantize (standalone)"] + agg[m]["quantize (fused into GN)"], 2),
           "gpu_busy": round(busy, 2), "wall": WALL[m]}
    rows.append(row)

with open(f"{HERE}/data/e2e_timing_profile_fixed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

w1 = 26
print(f"CORRECTED bucketed GPU time (ms/step, b128)\n")
print(f"{'bucket':<{w1}}" + "".join(f"{m:>15}" for m in MODES))
for b in BUCKETS + ["quantize_total", "gpu_busy"]:
    key = b
    print(f"{key:<{w1}}" + "".join(f"{[r for r in rows if r['mode']==m][0][key]:15.2f}" for m in MODES))
print(f"\nWROTE {HERE}/data/e2e_timing_profile_fixed.csv")
