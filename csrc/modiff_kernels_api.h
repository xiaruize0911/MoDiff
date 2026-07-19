// Plain host-facing prototypes for every function pybind.cpp exposes to Python.
// Grouped to mirror the files under csrc/kernels/. Only included by pybind.cpp
// (a plain .cpp, not .cu), so nothing here uses CUDA-only syntax.
#pragma once

#include <torch/extension.h>

// ---- csrc/kernels/quantize.cu ----
torch::Tensor quantize_and_pack(torch::Tensor input);
torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale);
torch::Tensor scale_quantize_int8(torch::Tensor input, torch::Tensor scale);
torch::Tensor quantize_attn_out_int8(torch::Tensor a, double a_scale);
torch::Tensor quantize_attn_out_int4_pack(torch::Tensor a, double a_scale);
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

int64_t conv2d_int8_num_tuned_configs();
torch::Tensor conv2d_int8_dequant_fp16_tuned(
    torch::Tensor input, torch::Tensor weight, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output, int64_t config_id,
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

int64_t conv2d_int4_num_tuned_configs();

torch::Tensor conv2d_int4_fprop_tuned(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale, int64_t config_id,
    int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w);

torch::Tensor conv2d_int4_dequant_fp16_tuned(
    torch::Tensor input, torch::Tensor weight_packed, torch::Tensor inv_scale,
    torch::Tensor weight_scales_half, torch::Tensor output, int64_t config_id,
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

torch::Tensor group_norm_silu_dequant_quantize_nhwc(
    torch::Tensor x_int8, double in_dequant, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, double eps, bool apply_silu,
    torch::Tensor scale, torch::Tensor smooth_inv,
    torch::Tensor mod_scale, torch::Tensor mod_shift);

// ---- csrc/kernels/fused_gn_qkv.cu ----
torch::Tensor fused_gn_qkv(
    torch::Tensor x, torch::Tensor weight, torch::Tensor epi_bias,
    int groups, double eps, double shift);
torch::Tensor fused_gn_qkv_int8(
    torch::Tensor x, torch::Tensor weight, torch::Tensor epi_bias,
    int groups, double eps, double shift);

// ---- csrc/kernels/awq_w8a8_gemm_cuda.cu (vendored from llm-awq, MIT) ----
void w8a8_gemm_forward_cuda(torch::Tensor in_feats, torch::Tensor kernel, torch::Tensor wscales, torch::Tensor ascales, torch::Tensor out_feats);

// ---- csrc/kernels/gemm_wxax.cu ----
// Production Linear GEMM backend: AWQ-tiling-scheme ports (CTA_M/N=128, CTA_K=64,
// WARP_N=32, 4 warps, ldmatrix.m8n8.x4 + XOR bank-swizzle shared-mem reads).
// C[M,N] fp16 = (A int8/int4 . B^T) * a_scale * w_scale[n]. Require N%128==0 and
// K%64 (int8) / K%128 (int4); callers pad weight/scale offline + activation at call
// time. (The prior hand-written gemm_w8a8/gemm_w4a4[/_out_int8] family was retired
// 2026-07-18; a non-compiled copy is at csrc/kernels/backup/.)
torch::Tensor gemm_w8a8_awq(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale);
torch::Tensor gemm_w8a8_awq_nout(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t n_out);
torch::Tensor gemm_w4a4_awq_nout(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K, int64_t n_out);
torch::Tensor gemm_w8a8_awq_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, torch::Tensor inv_out_scale);
torch::Tensor gemm_w4a4_awq(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K);
torch::Tensor gemm_w4a4_awq_out_i8(torch::Tensor A, torch::Tensor B, torch::Tensor w_scale, double a_scale, int64_t K, torch::Tensor inv_out_scale);
torch::Tensor quantize_act_int8(torch::Tensor x, double a_scale);
torch::Tensor quantize_act_int4_pack(torch::Tensor x, double a_scale);

// ---- csrc/kernels/quantize_qkv.cu ----
std::vector<torch::Tensor> quantize_qkv_int8(torch::Tensor qkv, int64_t nh, int64_t hd_pad);
std::vector<torch::Tensor> transpose_qkv_int8(torch::Tensor qkv_i8, int64_t nh, int64_t hd_pad);

// ---- csrc/kernels/flash_attn_int8.cu (fused int8 flash attention) ----
torch::Tensor flash_attn_int8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, double softmax_scale);
torch::Tensor flash_attn_int4(torch::Tensor q4, torch::Tensor k4, torch::Tensor v,
                              torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, int64_t hdp4, double softmax_scale);
torch::Tensor mma_smoke(torch::Tensor A, torch::Tensor B);

// ---- csrc/kernels/attn_quant_gemm.cu (standard quantized attention; fp16 scores) ----
torch::Tensor attn_qk_int8(torch::Tensor Q, torch::Tensor K, torch::Tensor sq, torch::Tensor sk, double scale);
std::vector<torch::Tensor> attn_softmax_requant(torch::Tensor S);
torch::Tensor attn_av_int8(torch::Tensor P, torch::Tensor Vt, torch::Tensor sp, torch::Tensor sv);
torch::Tensor attn_qk_int4(torch::Tensor Q, torch::Tensor K, int64_t hd_pad, torch::Tensor sq, torch::Tensor sk, double scale);
std::vector<torch::Tensor> attn_softmax_requant4(torch::Tensor S);
torch::Tensor attn_av_int4(torch::Tensor P, torch::Tensor Vt, torch::Tensor sp, torch::Tensor sv, int64_t T);
std::vector<torch::Tensor> quantize_attn_qkv(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t hp_qk, int64_t hp_av, int64_t bits);
// static (calibrated) score path: no runtime max/absmax reductions
std::vector<torch::Tensor> attn_softmax_requant_static(torch::Tensor S, double c);
std::vector<torch::Tensor> attn_softmax_requant_s8(torch::Tensor S, double sS, double c);  // int8-score softmax
std::vector<torch::Tensor> attn_softmax_requant_s8_dyn(torch::Tensor S, double sS);  // int8-score softmax, DYNAMIC per-row max
torch::Tensor attn_qk_int8_s8out(torch::Tensor Q, torch::Tensor K, torch::Tensor sq, torch::Tensor sk, double scale, double sS);  // QKᵀ -> int8 S
std::vector<torch::Tensor> attn_softmax_requant4_static(torch::Tensor S, double c);
std::vector<torch::Tensor> attn_softmax_fp16(torch::Tensor S, bool static_c, double c);
std::vector<torch::Tensor> quantize_attn_qkv_static(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t hp_qk, int64_t hp_av, int64_t bits, double sq_c, double sk_c, torch::Tensor sv_vec);
// int8-output qkv-linear -> attention quantize fusion (consume int8 qkv directly; no fp16 round-trip)
std::vector<torch::Tensor> quantize_attn_qkv_from_i8(torch::Tensor qkv_i8, torch::Tensor oscale, int64_t nh, int64_t T, int64_t hp_qk, int64_t hp_av);
