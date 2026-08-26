// Plain host-facing prototypes for every function pybind.cpp exposes to Python.
// Grouped to mirror the files under csrc/kernels/. Only included by pybind.cpp
// (a plain .cpp, not .cu), so nothing here uses CUDA-only syntax.
//
// ---------------------------------------------------------------------------------------
// DEAD-CODE POLICY for csrc/ -- read before deleting an unreferenced kernel.
//
// The rule (stated in full at csrc/kernels/norm/group_norm_silu.cu, above the pair-major
// stats kernel): a superseded-but-correct kernel is DELETED once its replacement ships,
// because git history already keeps it. A kernel is kept unreferenced only when the reason
// it is unused is itself a finding worth not rediscovering.
//
// Exactly two blocks currently qualify, and both say so in their own header comment. Do
// not "clean up" either one:
//   * group_norm_silu.cu -- the int8-INPUT GN kernel: built, verified, measured ~1.03-1.09x
//     faster, unreachable only because the upstream conv does not yet write int8 directly.
//   * group_norm_silu.cu -- the pair-major vectorized stats kernel: a FAILED experiment.
//     Changing the fp32 reduction order flipped int8 codes on real input
//     (max_code_diff=1) while passing random-data checks. Kept as executable evidence for
//     why the stats reduction stays scalar.
//
// Everything else unreferenced is deletable. The 2026-08-01 pass removed, with a runtime
// probe as evidence rather than an assumption: three flash_attn_int*_qpacked_kv_static*
// entry points (superseded by their _qout siblings, which are live), the two fp32-input
// int4 conv epilogue kernels (superseded by the _from_half ones), and csrc/kernels/backup/
// (never in setup.py sources, so it never compiled).
// ---------------------------------------------------------------------------------------
#pragma once

#include <torch/extension.h>

// ---- csrc/kernels/quantize/quantize.cu ----
torch::Tensor quantize_and_pack(torch::Tensor input);
torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale);
//: Activation-zero-point variant (plan fix #2). MoDiff's t=T entry point -- see the kernel comment
//: in quantize.cu for why this specific one is load-bearing and was missing from the first census.
torch::Tensor scale_quantize_and_pack_zp(torch::Tensor input, torch::Tensor scale,
                                         double zero_point);
torch::Tensor scale_quantize_int8(torch::Tensor input, torch::Tensor scale);
torch::Tensor quantize_attn_out_int8(torch::Tensor a, double a_scale);
torch::Tensor quantize_attn_out_int4_pack(torch::Tensor a, double a_scale, int64_t k_pad);
torch::Tensor dequant_bias_i8(torch::Tensor in, torch::Tensor out_scale, torch::Tensor bias);
void dequant_accumulate_int4(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale);
void dequant_accumulate_int8(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale);
void dequant_accumulate_and_return_int4(torch::Tensor residual, torch::Tensor a_hat_cache,
                                         torch::Tensor scale, torch::Tensor r_dq_out);
void dequant_accumulate_and_return_int8(torch::Tensor residual, torch::Tensor a_hat_cache,
                                         torch::Tensor scale, torch::Tensor r_dq_out);

// Cache-free dynamic scale discovery for the plain (non-MoDiff) baseline.
// (compute_dynamic_scale itself is an internal helper, defined+used within
// modiff_delta_quantize.cu; not declared here since pybind.cpp never binds it.)
torch::Tensor dynamic_quantize_int8_fprop(torch::Tensor x, torch::Tensor absmax_buf, torch::Tensor scale_buf,
                                           torch::Tensor inv_scale_buf, torch::Tensor retire_count);
torch::Tensor dynamic_quantize_pack_int4_fprop(torch::Tensor x, torch::Tensor absmax_buf, torch::Tensor scale_buf,
                                                torch::Tensor inv_scale_buf, torch::Tensor retire_count);

// ---- csrc/kernels/quantize/modiff_delta_quantize.cu (note: dynamic_quantize_* above are defined here) ----
void sub_absmax_scale(torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual,
                      torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
                      torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

// FP16-cache twin of the above, reduction only (no residual store). Supplies the dynamic
// MoDiff delta scale to the calibrated path, whose caches are fp16 and whose conv is the
// EVT in-place o_hat RMW. Q_level selects int8 (127) or int4 (7).
void delta_absmax_fp16(torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor absmax_buf,
                       torch::Tensor scale_out, torch::Tensor inv_scale_out,
                       torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv,
                       bool fused_silu);

torch::Tensor step1_quantize_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

torch::Tensor step1_quantize_no_ahat_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

// a4: the activation bit-width of this datapath (true => 4 bits, Q_b = 7; false => 8, Q_b = 127).
// It replaced a `float code_ceiling`. The ceiling was a magnitude, so its failure mode was a
// plausible-but-wrong number -- pass 127, or forget the argument, and a 4-bit layer silently stayed
// 8-bit, which is exactly what group_norm_silu_delta_quantize_resize_nhwc did until 2026-08-10. A
// bool has no such value to get wrong, and the saturation limit is now derived inside each kernel
// from the datapath it was told it is on. Saturating at Q_b rather than 127 is what a reused
// (MODIFF_DELTA_REFRESH > 1) scale depends on once the delta has outgrown it -- see
// docs/delta_clip_2026-08-06/FINDINGS.md, whose clip ratio is itself now retired.
torch::Tensor step1_static_quantize_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv, bool a4 = false);

torch::Tensor step1_static_quantize_fprop_silu(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv, bool a4 = false);

torch::Tensor step1_quantize_pack_int4_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

torch::Tensor step1_static_quantize_pack_int4_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv);

// cache-free static quantize (baseline conv, NO a_hat read/write)
torch::Tensor step1_static_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv);
torch::Tensor step1_static_quantize_pack_int4_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv);
//: Activation-zero-point variant (plan fix #2). No a_hat, so this always quantizes the activation on
//: the activation grid and z applies unconditionally.
torch::Tensor step1_static_quantize_pack_int4_noahat_fprop_zp(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv, double zero_point);

// Upsample(nearest,2x) + static quantize fusion (baseline conv, NO a_hat): fold Upsample.forward's
// F.interpolate into the following conv's quantize prologue, never materializing the fp16 upsampled
// intermediate. x is the SMALL pre-upsample [N,C,H,W]; output is [N,C,2H,2W] (int4: packed [N,2H,2W,C/2]).
// `a_hat_cache` is OPTIONAL: empty => the cache-free baseline these were named for, bit-identical
// to before it existed; fp16 [N,C,2H,2W] => MoDiff (subtract a_hat, quantize the delta, advance the
// cache in place). One kernel serves both modes -- there is no cloned MoDiff twin, because the loop
// already grids over output elements and so already visits each a_hat entry exactly once.
torch::Tensor upsample2x_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache);
torch::Tensor upsample2x_quantize_pack_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache);
//: Activation-zero-point variant (plan fix #2). Only legal with an EMPTY a_hat_cache: with a cache
//: this kernel quantizes a delta, where z is undefined and would also corrupt the cache update.
torch::Tensor upsample2x_quantize_pack_noahat_fprop_zp(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv,
    torch::Tensor a_hat_cache, double zero_point);

// Downsample(avg_pool,2x2,stride2) + static quantize fusion (baseline conv, NO a_hat): fold
// Downsample.forward's nn.AvgPool2d into the following conv's quantize prologue, never
// materializing the fp16 pooled intermediate. x is the LARGE pre-pool [N,C,H,W]; output is
// [N,C,H/2,W/2] (int4: packed [N,H/2,W/2,C/2]). Bit-exact to nn.AvgPool2d -> quantize.
torch::Tensor avgpool2x_quantize_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv);
torch::Tensor avgpool2x_quantize_pack_noahat_fprop(
    torch::Tensor x, torch::Tensor scale_buf, torch::Tensor smooth_inv);

torch::Tensor step1_static_quantize_pack_int4_fprop_silu(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv);

torch::Tensor step1_quantize_pack_int4_no_ahat_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

// ---- csrc/kernels/conv/conv2d_int8.cu ----
torch::Tensor conv2d_int8_fprop(
    torch::Tensor input, torch::Tensor weight, torch::Tensor scales, torch::Tensor bias,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_dequant_fp16_prealloc(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

int64_t conv2d_int8_num_tuned_configs();
torch::Tensor conv2d_int8_dequant_fp16_tuned(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_o_hat(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor o_hat_cache,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_o_hat_residual(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor o_hat_cache,
    torch::Tensor residual, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_no_ohat_prealloc(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_no_ohat_prealloc_bias(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_no_ohat_prealloc_bias_residual(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_relu_requant_int8(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor requant_scale, torch::Tensor output,
    bool apply_relu, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_deepfuse_relu_requant_int8(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor bias, torch::Tensor requant_scale, torch::Tensor output,
    bool apply_relu, int64_t config_id, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_deepfuse_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int64_t config_id, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_deepfuse_bias_residual_dual(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor bias, torch::Tensor residual,
    torch::Tensor requant_scale, torch::Tensor out_half, torch::Tensor out_int8, bool apply_relu,
    int64_t config_id, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_no_ohat(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

// ---- csrc/kernels/conv/conv2d_evt.cu (EVT-fused conv epilogues; scale+bias+residual / o_hat dual-store) ----
// weight_scales + bias are FP32 [K]; residual/o_hat/output are FP16.
torch::Tensor conv2d_int8_evt_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw);
torch::Tensor conv2d_int4_evt_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw);
//: fix #4 step 1: S[p] = sum over the conv window of the int4 activation codes, over all input
//: channels. One scalar per output pixel, shared by every output channel -- the reduction the weight
//: zero point needs. See csrc/modiff/conv/zpw_window_sum.cu for why it is padding-clean.
torch::Tensor int4_window_sum(
    torch::Tensor x_packed, int64_t R, int64_t S,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w);
torch::Tensor conv2d_int8_evt_o_hat_residual(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw);
torch::Tensor conv2d_int4_evt_o_hat_residual(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, torch::Tensor residual, torch::Tensor output,
    int sh, int sw, int ph, int pw, int dh, int dw);
// D2 no-residual: o_hat RMW in place (no `out`).
torch::Tensor conv2d_int8_evt_o_hat(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, int sh, int sw, int ph, int pw, int dh, int dw);
torch::Tensor conv2d_int4_evt_o_hat(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    torch::Tensor o_hat, int sh, int sw, int ph, int pw, int dh, int dw);

// ---- csrc/kernels/conv/conv2d_int4.cu ----
torch::Tensor conv2d_int4_fprop(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor scales, torch::Tensor bias,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_no_ohat_prealloc(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_no_ohat_prealloc_bias(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_no_ohat_prealloc_bias_residual(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor residual, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

int64_t conv2d_int4_num_tuned_configs();

torch::Tensor conv2d_int4_fprop_tuned(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_dequant_fp16_tuned(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_deepfuse_bias_residual_fp16(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor bias, torch::Tensor residual,
    torch::Tensor output, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_relu_requant_int4(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor requant_scale, torch::Tensor output,
    bool apply_relu, int64_t config_id, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_bias_residual_dual(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor bias, torch::Tensor residual, torch::Tensor requant_scale,
    torch::Tensor out_half, torch::Tensor out_packed, bool apply_relu, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_no_ohat(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_o_hat(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor o_hat_cache,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_o_hat_residual(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor o_hat_cache,
    torch::Tensor residual, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

// ---- csrc/kernels/util/layout_transform.cu ----
torch::Tensor fp16_ncw_to_fp32_cl(torch::Tensor src, int N, int C, int L);
torch::Tensor fp32_cl_to_fp16_ncw(torch::Tensor src, int N, int C, int L);
torch::Tensor fp16_ncw_delta_to_int8_cl(torch::Tensor x, torch::Tensor a_hat, torch::Tensor scale_t, int N, int C, int L);
// packed-INT4 -> INT8 widening, NHWC-physical. Value-preserving (a nibble code becomes the same
// int8 code), so the consumer's dequant `alpha` is unchanged. This is what lets W8A4 route its
// activation through int4 STORAGE -- where 4 bits is a property of the format, not of a clamp --
// and feed the int8 weight GEMM anyway. See the kernel header for why no mixed s8xs4 MMA exists.
torch::Tensor unpack_int4_to_int8_cl(torch::Tensor packed, int64_t N, int64_t C,
                                     int64_t H, int64_t W);

torch::Tensor cat2_channels_last_fp16(torch::Tensor a, torch::Tensor b);
//: The decoder skip-concat fold: concatenation + GroupNorm stats in one pass over the data,
//: so the pass cat2 used to cost is absorbed by a read that was happening anyway.
//: Returns {cat, mean, inv_std}. See csrc/modiff/norm/group_norm_silu.cu.
std::vector<torch::Tensor> cat2_gn_stats_fp16(
    torch::Tensor a, torch::Tensor b, int64_t num_groups, double eps);
//: The shipped GroupNorm stats pass, exposed so it can be TIMED IN ISOLATION -- its absence is what
//: made an inherited (and 3.8x wrong) number the basis of a claim that had to be retracted on
//: 2026-08-13. Returns {mean, inv_std}.
std::vector<torch::Tensor> gn_stats_fp16(torch::Tensor x, int64_t num_groups, double eps);
//: The decoder skip-concat fold wired end to end: the two halves in, {packed, cat} out, with the
//: concatenation and the GroupNorm statistics produced in one pass. Returns `cat` because the
//: ResBlock still consumes it (skip conv + residual), which is why the skip conv does NOT need
//: splitting. See csrc/modiff/norm/group_norm_silu.cu.
std::vector<torch::Tensor> group_norm_silu_delta_quantize_pack_cat2_nhwc(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor a_hat_cache, int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv, torch::Tensor mod_scale,
    torch::Tensor mod_shift, torch::Tensor absmax_buf, torch::Tensor scale_out,
    torch::Tensor inv_scale_out, torch::Tensor retire_count, double Q_level,
    bool report_next, double safety);

// ---- csrc/kernels/norm/group_norm_silu.cu ----
torch::Tensor group_norm_silu_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor mod_scale, torch::Tensor mod_shift);

torch::Tensor group_norm_silu_quantize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift);
torch::Tensor group_norm_silu_quantize_nhwc_fast(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift);

// k_pad: padded row width in channels for the int4 GEMM's K alignment (<=C -> no padding). Lets a
// block whose C is not a multiple of the GEMM's K tile (C=192 -> K_pad 256) keep the fused
// GN->quantize->pack path instead of GN + F.pad + a standalone quantize_act_int4_pack.
torch::Tensor group_norm_silu_quantize_pack_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad = 0);
torch::Tensor group_norm_silu_quantize_resize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack);
//: Activation-zero-point variant (plan fix #2). PACKED INT4 ONLY -- the int8 output has no matching
//: bias correction, and the wrapper TORCH_CHECKs that.
torch::Tensor group_norm_silu_quantize_resize_nhwc_zp(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack, double zero_point);
torch::Tensor group_norm_silu_quantize_pack_nhwc_fast(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad = 0);
//: Activation-zero-point variants (plan fix #2). a_q = clamp(round(a*s) + z, -7, 7); the
//: dequantization's -z*sum(w_q) term is folded into the conv bias at calibration time, so the
//: GEMM and the epilogue never see z. zero_point = 0.0 reproduces the symmetric kernel exactly.
torch::Tensor group_norm_silu_quantize_pack_nhwc_zp(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad, double zero_point);
torch::Tensor group_norm_silu_quantize_pack_nhwc_fast_zp(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups,
    double eps, bool apply_silu, torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, int64_t k_pad, double zero_point);

torch::Tensor group_norm_silu_dequant_quantize_nhwc(
    torch::Tensor x_int8, double in_dequant, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift);

// MoDiff-fused GN(+mod)+SiLU + temporal-delta quantize + in-place a_hat update.
torch::Tensor group_norm_silu_delta_quantize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor a_hat_cache,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
    torch::Tensor retire_count, double Q_level, bool report_next, double safety,
    bool a4 = false);
// SINGLE-KERNEL (group-major) twin of the above's static path. **A FAILED EXPERIMENT, kept as
// executable evidence under the dead-code policy at the top of this file** -- it is 0.44x-0.98x the
// two-kernel path (slower everywhere, worst at the CPG the UNet uses most), because one-block-
// per-group forces group-major NHWC access and de-coalesces a_hat's read/write, while the only
// saving available was ~1-3% of one kernel launch. Full measurement + mechanism in the kernel's
// header comment in csrc/modiff/norm/group_norm_silu.cu. Unreferenced on purpose; do not wire in.
torch::Tensor group_norm_silu_delta_quantize_nhwc_fused(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor a_hat_cache,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift, bool a4 = false);
// MoDiff twin of group_norm_silu_quantize_resize_nhwc: fuses GN(+mod)(+SiLU)+2x resize+delta
// quantize+in-place a_hat for the eight updown ResBlocks, which get no fusion otherwise.
// The trailing eight arguments are the same dynamic-scale contract as the non-resize sibling
// above: empty absmax_buf/scale_out/inv_scale_out/retire_count => static (the original
// behaviour), real 1-element buffers => the scale is measured here. Without them the kernel
// could not serve a refresh step, so the caller declined on every one of them -- which at
// MODIFF_DELTA_REFRESH=1 is every step, i.e. the fusion never fired.
std::vector<torch::Tensor> gn_stats_from_tiles(torch::Tensor x, int64_t num_groups,
                                               int64_t Mt, int64_t Nt);
torch::Tensor group_norm_silu_delta_quantize_resize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    int64_t k_pad, int64_t resize, bool pack, torch::Tensor a_hat_cache,
    torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
    torch::Tensor retire_count, double Q_level, bool report_next, double safety,
    bool a4);

torch::Tensor group_norm_silu_delta_quantize_pack_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor a_hat_cache,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift,
    torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
    torch::Tensor retire_count, double Q_level, bool report_next, double safety);

// ---- csrc/kernels/norm/fused_gn_qkv.cu ----
torch::Tensor fused_gn_qkv(
    torch::Tensor x, torch::Tensor weight, torch::Tensor epi_bias,
    int groups, double eps, double shift);
torch::Tensor fused_gn_qkv_int8(
    torch::Tensor x, torch::Tensor weight, torch::Tensor epi_bias,
    int groups, double eps, double shift);
// int8 fused GN->qkv with a custom fp32-bias/int8-clamp EVT epilogue (fixes fused_gn_qkv_int8's
// signed-qkv overflow). bias_f32 is fp32 [3C]; output int8 [N,3C,H,W] channels_last.
torch::Tensor fused_gn_qkv_i8evt(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias_f32,
    int groups, double eps, double shift);

// ---- csrc/kernels/linear/awq_w8a8_gemm_cuda.cu (vendored from llm-awq, MIT) ----

// ---- csrc/kernels/linear/gemm_wxax.cu ----
// Production Linear GEMM backend: AWQ-tiling-scheme ports (CTA_M/N=128, CTA_K=64,
// WARP_N=32, 4 warps, ldmatrix.m8n8.x4 + XOR bank-swizzle shared-mem reads).
// C[M,N] fp16 = (A int8/int4 . B^T) * a_scale * w_scale[n]. Require N%128==0 and
// K%64 (int8) / K%128 (int4); callers pad weight/scale offline + activation at call
// time. (The prior hand-written gemm_w8a8/gemm_w4a4[/_out_int8] family was retired
// 2026-07-18; a non-compiled copy is at csrc/kernels/backup/.)
torch::Tensor gemm_w8a8_awq(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale);
torch::Tensor gemm_w8a8_awq_nout(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t n_out);
torch::Tensor gemm_w4a4_awq_nout(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K, int64_t n_out);
// MoDiff o_hat accumulate in the Linear GEMM epilogue (Eq 9). No bias parameter: on a modulated
// step the bias belongs to o_hat_T only. `residual` empty => returns o_hat itself; given => also
// returns o_hat_t + residual as a separate tensor (the ResBlock/attention skip, which must NOT be
// folded into the temporal state).
// a_scale is a 1-element fp32 DEVICE tensor, not a double: the MoDiff delta scale is produced on
// device each call, and taking it by value would force a host sync per linear per step.
torch::Tensor gemm_w8a8_awq_o_hat(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                  torch::Tensor a_scale, int64_t n_out, torch::Tensor o_hat,
                                  torch::Tensor residual, torch::Tensor bias);
torch::Tensor gemm_w4a4_awq_o_hat(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                  torch::Tensor a_scale, int64_t n_out, torch::Tensor o_hat,
                                  torch::Tensor residual, torch::Tensor bias);

torch::Tensor gemm_w8a8_awq_bias_res(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t n_out, torch::Tensor bias, torch::Tensor residual);
torch::Tensor gemm_w4a4_awq_bias_res(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K, int64_t n_out, torch::Tensor bias, torch::Tensor residual);
torch::Tensor gemm_w8a8_awq_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, torch::Tensor inv_out_scale);
// MoDiff DUAL output: advances the fp16 o_hat state (Eq 9, bias-free) AND returns int8 codes of
// o_hat+bias, in one pass. Lets a MoDiff'd projection keep feeding an int8 consumer (flash), which
// the single-output variants made mutually exclusive. a_scale is a 1-element float32 DEVICE tensor.
torch::Tensor gemm_w8a8_awq_o_hat_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale,
                                        torch::Tensor a_scale, int64_t n_out, torch::Tensor o_hat,
                                        torch::Tensor bias, torch::Tensor inv_out_scale);
torch::Tensor gemm_w8a8_awq_out_i8_bias_nout(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
    torch::Tensor inv_out_scale, torch::Tensor bias, int64_t n_out);
std::vector<torch::Tensor> gemm_w8a8_awq_qkv_i8_layouts(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
    torch::Tensor inv_out_scale, torch::Tensor bias, int64_t nh,
    int64_t T, int64_t hd, int64_t hp);
std::vector<torch::Tensor> gemm_w8a8_awq_qkv_i8_layouts_compact(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale,
    torch::Tensor inv_out_scale, torch::Tensor bias, int64_t nh,
    int64_t T, int64_t hd, int64_t hp);
torch::Tensor gemm_w4a4_awq(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K);
torch::Tensor gemm_w4a4_awq_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K, torch::Tensor inv_out_scale);
std::vector<torch::Tensor> gemm_w4a4_awq_qkv_i4qk_i8v(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K,
    int64_t n_out, torch::Tensor bias, int64_t nh, int64_t T, int64_t hd,
    int64_t hp_qk, int64_t hp_v, int64_t storage_mode,
    double sq, double sk, torch::Tensor sv);
torch::Tensor gemm_w4a4_awq_qkv_codes(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K,
    int64_t n_out_, torch::Tensor bias, int64_t nh, int64_t hd,
    torch::Tensor inv_out, torch::Tensor lim);
std::vector<torch::Tensor> gemm_w4a4_awq_qkv_i4qk_i8v_layouts(
    torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K,
    torch::Tensor inv_out, torch::Tensor lim, torch::Tensor bias,
    int64_t nh_, int64_t T_, int64_t hd_, int64_t hp_, torch::Tensor sv,
    int64_t packed_qk);
torch::Tensor quantize_act_int8(torch::Tensor x, double a_scale);
torch::Tensor quantize_act_int4_pack(torch::Tensor x, double a_scale);

// ---- csrc/kernels/attention/flash_attn_int8.cu (fused int8/int4 flash attention) ----
torch::Tensor flash_attn_int8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, double softmax_scale);
torch::Tensor flash_attn_int8_vt(torch::Tensor q, torch::Tensor k, torch::Tensor vt,
                                 torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, double softmax_scale);
torch::Tensor flash_attn_int8_vt_static(torch::Tensor q, torch::Tensor k, torch::Tensor vt,
                                        torch::Tensor sv, double sq, double sk, double softmax_scale);
torch::Tensor flash_attn_int8_qpacked_kv_static_qout(
    torch::Tensor qkv, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int8_qi8packed_kv_static_qout(
    torch::Tensor qkv_i8, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int8_qi8packed_kv_static_qout_preg(
    torch::Tensor qkv_i8, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int8_qi8_kv_static_qout(
    torch::Tensor q_i8, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int8_qi8_kv_static_qout_hd24(
    torch::Tensor q_i8, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int8_qi8packed_small_qout(
    torch::Tensor qkv_i8, torch::Tensor sv, double sq, double sk,
    double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_i4values_i8mma_qpacked_kv_static_qout(
    torch::Tensor qkv, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale,
    double proj_a_scale, int64_t k_pad);
torch::Tensor flash_attn_i4values_i8mma_vt_static_qout(
    torch::Tensor q, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale,
    double proj_a_scale, int64_t k_pad);
torch::Tensor flash_attn_i4values_small_qout(
    torch::Tensor qkv_i8, torch::Tensor sv, double sq, double sk,
    double softmax_scale, double proj_a_scale, int64_t k_pad);
torch::Tensor flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24(
    torch::Tensor q_i8, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    int64_t hd_pad, double sq, double sk, double softmax_scale,
    double proj_a_scale, int64_t k_pad);
torch::Tensor flash_attn_int4(torch::Tensor q4, torch::Tensor k4, torch::Tensor v,
                              torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, int64_t hdp4, double softmax_scale);
torch::Tensor flash_attn_int4_vt(torch::Tensor q4, torch::Tensor k4, torch::Tensor vt,
                                 torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, int64_t hdp4, double softmax_scale);
torch::Tensor flash_attn_int4_vt_static(torch::Tensor q4, torch::Tensor k4, torch::Tensor vt,
                                        torch::Tensor sv, int64_t hdp4, double sq, double sk,
                                        double softmax_scale);
torch::Tensor flash_attn_int4_qpacked_kv_static_qout(
    torch::Tensor qkv, torch::Tensor k4, torch::Tensor vt, torch::Tensor sv,
    int64_t hdp4, double sq, double sk, double softmax_scale,
    double proj_a_scale, int64_t k_pad);
// Fused proj-quantize flash variants: emit the attention output already quantized token-major
// (int8 [b*T,C] / packed-int4 [b*T,k_pad/2]) by the calibrated proj scale, so the separate
// quantize_attn_out_int{8,4} pass + fp16 attn-output round-trip are eliminated.
torch::Tensor flash_attn_int8_vt_qout(torch::Tensor q, torch::Tensor k, torch::Tensor vt,
                                      torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                      double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int8_vt_static_qout(
    torch::Tensor q, torch::Tensor k, torch::Tensor vt, torch::Tensor sv,
    double sq, double sk, double softmax_scale, double proj_a_scale);
torch::Tensor flash_attn_int4_vt_qout(torch::Tensor q4, torch::Tensor k4, torch::Tensor vt,
                                      torch::Tensor sq, torch::Tensor sk, torch::Tensor sv,
                                      int64_t hdp4, double softmax_scale, double proj_a_scale, int64_t k_pad);
torch::Tensor flash_attn_int4_vt_static_qout(
    torch::Tensor q4, torch::Tensor k4, torch::Tensor vt, torch::Tensor sv,
    int64_t hdp4, double sq, double sk, double softmax_scale,
    double proj_a_scale, int64_t k_pad);
// PACKED-input int8 flash: read interleaved qkv [b,T,nh,3,hd] directly (fp16 -> quantize on load with
// frozen per-tensor sq_c/sk_c + per-channel sv[hd]; int8 -> plain gather), doing hd->hd_pad pad + the
// V-transpose in smem. Replaces the aq_qtok/aq_vquant (or Route-1 from_i8) reshuffle + qi/ki/vt HBM
// round-trip. Dispatches on qkv.dtype(). _qout variant emits proj-quantized int8 token-major [b*T,C].
torch::Tensor flash_attn_int8_packed_vt(torch::Tensor qkv, torch::Tensor sv, int64_t hd_pad,
                                        double sq_c, double sk_c, double softmax_scale);
torch::Tensor flash_attn_int8_packed_vt_qout(torch::Tensor qkv, torch::Tensor sv, int64_t hd_pad,
                                             double sq_c, double sk_c, double softmax_scale,
                                             double proj_a_scale);
torch::Tensor flash_attn_int8_packed_persistent_qout(
    torch::Tensor qkv, torch::Tensor sv, int64_t hd_pad,
    double sq_c, double sk_c, double softmax_scale, double proj_a_scale);
torch::Tensor mma_smoke(torch::Tensor A, torch::Tensor B);

// ---- csrc/kernels/attention/attn_quant_gemm.cu (quantize prologue for FUSED flash attention) ----
// Per-token int8/int4 Q/K + per-channel int8 (transposed) V + scales, feeding
// flash_attn_int8_vt / flash_attn_int4_vt. (The materialized QKᵀ/softmax/AV int GEMM
// attention path was removed; flash is the sole quantized-attention path.)
std::vector<torch::Tensor> quantize_attn_qkv(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t hp_qk, int64_t hp_av, int64_t bits);
std::vector<torch::Tensor> quantize_attn_qkv_i4qk_i8v(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t hp_qk, int64_t hp_av);
std::vector<torch::Tensor> quantize_attn_qkv_i4qk_i8v_static(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t hp_qk, int64_t hp_av, double sq_c, double sk_c, torch::Tensor sv_vec);
std::vector<torch::Tensor> quantize_attn_qkv_static(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t hp_qk, int64_t hp_av, int64_t bits, double sq_c, double sk_c, torch::Tensor sv_vec);
// PACKED-qkv quantize (reads interleaved [b,T,nh,3,hd], no transpose copy). QK int8/int4, V int8.
std::vector<torch::Tensor> quantize_attn_qkv_packed(torch::Tensor qkv, int64_t nh, int64_t T, int64_t hd, int64_t hp_qk, int64_t hp_av, int64_t qk_bits);
std::vector<torch::Tensor> quantize_attn_qkv_packed_static(torch::Tensor qkv, int64_t nh, int64_t T, int64_t hd, int64_t hp_qk, int64_t hp_av, int64_t qk_bits, double sq_c, double sk_c, torch::Tensor sv_vec);
// Production static API: omits dead per-token sq/sk tensors and returns a broadcast [hp_av]
// V-scale vector: {qi, ki, vt, sv}.
std::vector<torch::Tensor> quantize_attn_qkv_packed_static_compact(torch::Tensor qkv, int64_t nh, int64_t T, int64_t hd, int64_t hp_qk, int64_t hp_av, int64_t qk_bits, double sq_c, double sk_c, torch::Tensor sv_vec);
std::vector<torch::Tensor> quantize_attn_kv_packed_static(torch::Tensor qkv, int64_t nh, int64_t T, int64_t hd, int64_t hp_qk, int64_t hp_av, int64_t qk_bits, double sk_c, torch::Tensor sv_vec);
// int8 reshuffle consumer for fused_gn_qkv_i8evt: gather Q/K + transpose V (int8->int8, no requant).
std::vector<torch::Tensor> quantize_attn_qkv_from_i8(torch::Tensor qkv_i8, int64_t nh, int64_t T, int64_t hd, int64_t hp_qk, int64_t hp_av);
std::vector<torch::Tensor> quantize_attn_kv_from_i8(torch::Tensor qkv_i8, int64_t nh, int64_t T, int64_t hd, int64_t hp_qk, int64_t hp_av);
// fp16 (reference) materialized softmax — used by the fp16-materialized attention path (not int8/int4)
std::vector<torch::Tensor> attn_softmax_fp16(torch::Tensor S, bool static_c, double c);
