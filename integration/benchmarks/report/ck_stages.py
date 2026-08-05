"""Kernel-name -> pipeline-stage attribution, shared by the checkpoint report's figures and tables.

This lived inside make_checkpoint_report_plots.py, where it was the only copy. The report's
prose tables were then transcribed by hand from the figures, so a change to the mapping moved
the plots but not the numbers quoted next to them. Both now read the same mapping from here.

The name tokens are ORDER-SENSITIVE and the names are adversarial -- see the comments inside
STAGES before editing.
"""
from collections import defaultdict

STAGES = [
    ("attn", "attention core", "#c0392b",
     ("flash_attn_int8_mma", "flash_attn_int4_mma", "pytorch_flash", "flash_fwd",
      "qi8packed_small", "bmm", "softmax")),
    # conv vs GEMM is decided by NAME TOKENS, and the names are adversarial: CUTLASS calls a
    # convolution "..._s16816fprop_..." and a matmul "..._s16816gemm_...", while the fused
    # GN->QKV projection is called "ImplicitGemmConvolutionFusionPerSample" -- a projection with
    # "Convolution" in its name. Matching loosely on "cutlass" or "ImplicitGemm" puts the wrong
    # kernels in the wrong bucket, which made FP16's attention bars show phantom convolution.
    # So: GEMM is matched on gemm-specific tokens FIRST, conv on fprop/convolve tokens after.
    ("gemm", "QKV / output projection", "#2f6fb2",
     ("ImplicitGemmConvolutionFusionPerSample",   # fused GN->QKV; attention-only, verified
      # "s161616gemm" added 2026-08-04: cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2
      # (the fp16 timestep-embedding Linear) contains neither "s16816gemm" nor "s1688gemm" -- the
      # digit runs differ -- so it was falling through to elementwise/other in every mode.
      "s16816gemm", "s1688gemm", "s161616gemm", "xmma_gemm", "gemm_w8a8_kernel",
      "gemm_w4a4_kernel", "gemm_w4a4_awq", "gemv")),
    ("conv", "convolution", "#1b7f79",
     ("ImplicitGemmConvolution", "fprop", "implicit_convolve", "conv2d", "cudnn",
      "Kernel_conv", "wgrad", "dgrad", "xmma_conv")),
    # The MoDiff tokens were added 2026-08-04. Without them the whole MoDiff GN chain --
    # gn_apply_delta_quantize_flat_vec2, gn_stats_partials_chanmajor, gn_stats_reduce_partials,
    # delta_absmax_fp16 -- matched no bucket and fell through to "elementwise / copies / other",
    # which is where the single largest MoDiff-specific kernel (8.2 ms/step) would have been filed.
    # They are GroupNorm/quantizer work: two are the GN statistics pass, one is the fused
    # GN+SiLU+delta-quantize store, one computes the delta quantizer's scale.
    # Every kernel this tree can emit from csrc/kernels/norm/*.cu, plus torch's own GN moments,
    # enumerated rather than guessed: three separate rounds of "one more name matched nothing and
    # silently became elementwise/other" cost more than listing them. The gn_* family does NOT share
    # a matchable prefix with group_norm_*, and a bare "gn_" token cannot be used -- it also matches
    # at::native's sign_kernel ("si-gn_-kernel").
    ("norm", "GroupNorm + quantize", "#e6a020",
     ("group_norm", "gn_accum", "gn_finalize", "quantize_act", "quant_act",
      "gn_apply", "gn_stats", "gn_group_stats", "gn_mean", "gn_invstd",
      "delta_absmax", "delta_quantize", "update_ahat", "rowwisemoments")),
    # Split from a single "K/V prep + out quantize" bucket. They are different operations with
    # different causes: the K/V producer is what the layout epilogues removed, while the output
    # quantize exists only where attention itself is not quantized. Lumping them made INT4's
    # residual look like leftover producer work when no producer remains.
    ("kvprep", "K/V gather + transpose", "#7d5ba6",
     ("aq_kv_packed", "qkv_i4codes", "from_i8_kv_tiled", "quantize_attn_kv", "layout_transform",
      "modiff_delta")),
    ("outq", "attention output quantize", "#c98bdb",
     ("quant_attn_out", "quantize_attn_out")),
    ("misc", "elementwise / copies / other", "#b9c2cc", ()),
]

STAGE_LABEL = {k: lbl for k, lbl, _, _ in STAGES}


def stage_of(name):
    for key, _, _, frags in STAGES:
        if frags and any(f.lower() in name.lower() for f in frags):
            return key
    return "misc"


def split(kernels, keyname="kernel", usname="us"):
    t = defaultdict(float)
    for k in kernels:
        t[stage_of(k[keyname])] += k[usname]
    return t
