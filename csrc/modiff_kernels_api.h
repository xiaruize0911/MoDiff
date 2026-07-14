// Plain host-facing prototypes for every function pybind.cpp exposes to Python.
// Grouped to mirror the files under csrc/kernels/. Only included by pybind.cpp
// (a plain .cpp, not .cu), so nothing here uses CUDA-only syntax.
#pragma once

#include <torch/extension.h>

// ---- csrc/kernels/quantize.cu ----
torch::Tensor quantize_and_pack(torch::Tensor input);
torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale);
torch::Tensor scale_quantize_int8(torch::Tensor input, torch::Tensor scale);
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

// ---- csrc/kernels/modiff_delta_quantize.cu ----
void sub_absmax_scale(torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual,
                      torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
                      torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

torch::Tensor step1_quantize_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

torch::Tensor step1_quantize_no_ahat_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

torch::Tensor step1_static_quantize_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv);

torch::Tensor step1_static_quantize_fprop_silu(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv);

torch::Tensor step1_quantize_pack_int4_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

torch::Tensor step1_static_quantize_pack_int4_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv);

torch::Tensor step1_static_quantize_pack_int4_fprop_silu(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor scale_buf, torch::Tensor smooth_inv);

torch::Tensor step1_quantize_pack_int4_no_ahat_fprop(
    torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual_buf,
    torch::Tensor absmax_buf, torch::Tensor scale_buf, torch::Tensor inv_scale_buf,
    torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);

// ---- csrc/kernels/conv2d_int8.cu ----
torch::Tensor conv2d_int8_fprop(
    torch::Tensor input, torch::Tensor weight, torch::Tensor scales, torch::Tensor bias,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_dequant_fp16_prealloc(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int8_fprop_o_hat(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor o_hat_cache,
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

torch::Tensor conv2d_int8_fprop_no_ohat(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale, torch::Tensor weight_scales,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

// ---- csrc/kernels/conv2d_int4.cu ----
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

torch::Tensor conv2d_int4_fprop_no_ohat(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, torch::Tensor weight_scales,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_fprop_o_hat(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales, torch::Tensor o_hat_cache,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

// ---- csrc/kernels/layout_transform.cu ----
torch::Tensor fp16_ncw_to_fp32_cl(torch::Tensor src, int N, int C, int L);
torch::Tensor fp32_cl_to_fp16_ncw(torch::Tensor src, int N, int C, int L);
torch::Tensor fp16_ncw_delta_to_int8_cl(torch::Tensor x, torch::Tensor a_hat, torch::Tensor scale_t, int N, int C, int L);

// ---- csrc/kernels/group_norm_silu.cu ----
torch::Tensor group_norm_silu_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor mod_scale, torch::Tensor mod_shift);

torch::Tensor group_norm_silu_quantize_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift);

torch::Tensor group_norm_silu_quantize_pack_nhwc(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift);
