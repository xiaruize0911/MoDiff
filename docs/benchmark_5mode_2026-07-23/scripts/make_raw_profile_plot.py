"""Per-kernel e2e profile plot — NO categorization. One bar per individual CUDA kernel (top N by
total time across modes), grouped by mode. Reads e2e_kernel_profile_raw.csv -> figs/fig_e2e_kernel_profile_raw.png."""
import os, csv, re
os.chdir("/workspace/MoDiff")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = "docs/benchmark_5mode_2026-07-23"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
COL = {"fp16": "#2563eb", "int8_baseline": "#f59e0b", "int4_baseline": "#dc2626",
       "int8_modiff": "#fbbf24", "int4_modiff": "#f87171"}
TOPN = 34


def short(raw):
    """1:1 readable name (NOT a bucket) — just demangle/trim the symbol."""
    s = raw
    if "ImplicitGemmConvolutionEVT" in s: return "cutlass modiff-EVT conv"
    if "ImplicitGemmConvoluti" in s:      return "cutlass conv fprop (int)"
    if "wmma_tensorop_f16" in s:          return "cutlass wmma f16 (attn bmm)"
    if "cutlass__5x_cudnn" in s:          return "cudnn cutlass f16 gemm"
    if "xmma_fprop_implicit_gemm_f16" in s: return "cudnn xmma fprop f16 conv"
    if "xmma_gemm_f16" in s:              return "cudnn xmma f16 gemm"
    if "tensorop_f16_s16816gemm" in s:    return "cutlass f16 tensorop gemm"
    if s.startswith("ampere_fp16_s1688gemm"): return "ampere f16 s1688 gemm"
    m = re.search(r"(flash_attn_int[48]_mma_kernel|gemm_w8a8_kernel_awq|gemm_w4a4_kernel_awq|_gemm_w4a4_kernel|"
                  r"group_norm_silu_quantize_pack_nhwc_kernel|group_norm_silu_quantize_nhwc_kernel|"
                  r"group_norm_silu_nhwc_kernel|gn_group_stats_kernel|gn_accum_kernel|"
                  r"gn_apply_delta_quantize_pack_flat_kernel|gn_apply_delta_quantize_flat_kernel|"
                  r"aq_qtok_packed_static_kernel|aq_vquant_trans_packed_kernel|quant_act_int4_pack_kernel|"
                  r"quant_act_int8_kernel|scale_quantize_pack_kernel|scale_quantize_int8_kernel|"
                  r"static_quantize_pack_and_update_ahat_kernel_int4_half|static_quantize_and_update_ahat_kernel_int8_half|"
                  r"quant_attn_out_int8_kernel|softmax_warp_forward|upsample_nearest2|CatArrayBatchedCopy|"
                  r"avg_pool2d_out_cuda|RowwiseMoments)", s)
    if m:
        n = m.group(1)
        if "group_norm_silu_quantize_pack" in n: return "GN+SiLU+quant+pack (fused)"
        if "group_norm_silu_quantize" in n:      return "GN+SiLU+quant (fused)"
        if "group_norm_silu_nhwc" in n:          return "GN+SiLU"
        if "gn_apply_delta_quantize_pack" in n:  return "GN delta-quant+pack (flat)"
        if "gn_apply_delta_quantize_flat" in n:  return "GN delta-quant (flat)"
        if "gn_group_stats" in n:                return "GN group stats"
        if "gn_accum" in n:                      return "GN accum"
        if "softmax_warp_forward" in n:          return "softmax (fp16 SDPA)"
        if "upsample_nearest2" in n:             return "upsample nearest2d"
        if "CatArrayBatchedCopy" in n:           return "cat (concat)"
        if "avg_pool2d" in n:                    return "avg_pool2d"
        if "RowwiseMoments" in n:                return "RowwiseMoments"
        return n.replace("_kernel", "")
    if "vectorized_elementwise_kernel" in s: return "vectorized elementwise"
    if "unrolled_elementwise_kernel" in s:   return "unrolled elementwise"
    if "elementwise_kernel" in s:            return "elementwise"
    if "reduce_kernel" in s:                 return "reduce"
    return (s[:34] + "...") if len(s) > 34 else s


rows = []
with open(f"{HERE}/data/e2e_kernel_profile_raw.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)
rows = rows[:TOPN]

# unique labels (append #k on collision so distinct kernels stay distinct — no bucketing)
labels, seen = [], {}
for r in rows:
    base = short(r["kernel"])
    seen[base] = seen.get(base, 0) + 1
    labels.append(base if seen[base] == 1 and [short(x["kernel"]) for x in rows].count(base) == 1
                  else f"{base} #{seen[base]}")

y = np.arange(len(rows))[::-1]; h = 0.16
fig, ax = plt.subplots(figsize=(12, 15))
for i, m in enumerate(MODES):
    vals = [float(r[m]) for r in rows]
    ax.barh(y + (i - 2) * h, vals, h, label=m, color=COL[m], edgecolor="white", lw=0.3)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("ms / step (GPU self time)"); ax.set_title(f"E2E per-kernel GPU time (top {TOPN} of 145, NO bucketing) — b128")
ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(f"{HERE}/figs/fig_e2e_kernel_profile_raw.png", dpi=150); plt.close(fig)
print("wrote figs/fig_e2e_kernel_profile_raw.png")
