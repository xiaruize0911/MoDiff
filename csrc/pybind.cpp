#include <torch/extension.h>

#include "modiff_kernels_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Standalone elementwise quantize/pack/dequant-accumulate (kernels/quantize.cu)
    m.def("quantize_and_pack", &quantize_and_pack, "Fast Quantization and Packing for INT4");
    m.def("scale_quantize_and_pack", &scale_quantize_and_pack, "Fused Scale + Quantize + Pack for INT4");
    m.def("scale_quantize_int8", &scale_quantize_int8, "Fused Scale + Quantize for INT8");
    m.def("dequant_accumulate_int4", &dequant_accumulate_int4, "Fused Dequant + Accumulate for INT4 cache");
    m.def("dequant_accumulate_int8", &dequant_accumulate_int8, "Fused Dequant + Accumulate for INT8 cache");
    m.def("dequant_accumulate_and_return_int4", &dequant_accumulate_and_return_int4,
          "Fused Dequant + Accumulate for INT4 cache, also returning the dequantized value");
    m.def("dequant_accumulate_and_return_int8", &dequant_accumulate_and_return_int8,
          "Fused Dequant + Accumulate for INT8 cache, also returning the dequantized value");
    // compute_dynamic_scale is intentionally NOT bound here: it's an internal C++
    // helper used by dynamic_quantize_int8_fprop/dynamic_quantize_pack_int4_fprop
    // below, never called directly from Python.
    m.def("dynamic_quantize_int8_fprop", &dynamic_quantize_int8_fprop, "Cache-free dynamic INT8 quantize (plain baseline, no a_hat)");
    m.def("dynamic_quantize_pack_int4_fprop", &dynamic_quantize_pack_int4_fprop, "Cache-free dynamic INT4 quantize+pack (plain baseline, no a_hat)");

    // MoDiff temporal-delta cache update (kernels/modiff_delta_quantize.cu)
    m.def("sub_absmax_scale", &sub_absmax_scale, "Fused Subtract + AbsMax + Scale computation");
    m.def("step1_quantize_fprop", &step1_quantize_fprop, "Fused sub_absmax_scale + dequant + quantize for step 1");
    m.def("step1_quantize_no_ahat_fprop", &step1_quantize_no_ahat_fprop,
          "Benchmark-only: same as step1_quantize_fprop but skips the a_hat_cache write, to isolate the "
          "write's cost in microbenchmarks. Still READS a_hat_cache (required to form the residual) -- "
          "this is NOT a no-cache substitute. For a true cache-free baseline use dynamic_quantize_int8_fprop.");
    m.def("step1_static_quantize_fprop", &step1_static_quantize_fprop, "Fused static-scale subtract + dequant + quantize for INT8 step 1");
    m.def("step1_static_quantize_fprop_silu", &step1_static_quantize_fprop_silu,
          "Same as step1_static_quantize_fprop but applies SiLU to x inline (FP16 a_hat_cache only) -- "
          "fuses a ResBlock's activation function into the quantize step");
    m.def("step1_quantize_pack_int4_fprop", &step1_quantize_pack_int4_fprop, "Fused sub_absmax_scale + dequant + quantize+pack for INT4 step 1");
    m.def("step1_static_quantize_pack_int4_fprop", &step1_static_quantize_pack_int4_fprop, "Fused static-scale subtract + dequant + quantize+pack for INT4 step 1");
    m.def("step1_static_quantize_pack_int4_fprop_silu", &step1_static_quantize_pack_int4_fprop_silu,
          "Same as step1_static_quantize_pack_int4_fprop but applies SiLU to x inline (FP16 a_hat_cache only)");
    m.def("step1_quantize_pack_int4_no_ahat_fprop", &step1_quantize_pack_int4_no_ahat_fprop,
          "Benchmark-only: INT4 counterpart of step1_quantize_no_ahat_fprop -- see that entry for why this "
          "still takes a_hat_cache. For a true cache-free baseline use dynamic_quantize_pack_int4_fprop.");

    // CUTLASS INT8 Conv2d (kernels/conv2d_int8.cu)
    m.def("conv2d_int8_fprop", &conv2d_int8_fprop, "Conv2d INT8 Forward (CUTLASS)");
    m.def("conv2d_int8_fprop_dequant_fp16_prealloc", &conv2d_int8_fprop_dequant_fp16_prealloc, "CUTLASS INT8 conv with FP16 dequantizing epilogue into a preallocated output buffer");
    m.def("conv2d_int8_num_tuned_configs", &conv2d_int8_num_tuned_configs, "Number of autotunable tile configs for the deep-fuse INT8 conv");
    m.def("conv2d_int8_dequant_fp16_tuned", &conv2d_int8_dequant_fp16_tuned, "Deep-fuse INT8 conv (FP16 out) using tile config `config_id` (for per-shape autotuning)");
    m.def("conv2d_int8_fprop_o_hat", &conv2d_int8_fprop_o_hat, "Fused INT8 Conv + o_hat accumulate");
    m.def("conv2d_int8_fprop_no_ohat_prealloc", &conv2d_int8_fprop_no_ohat_prealloc, "Fused INT8 conv + dequant into a preallocated output buffer");
    m.def("conv2d_int8_fprop_no_ohat_prealloc_bias", &conv2d_int8_fprop_no_ohat_prealloc_bias, "Fused INT8 conv + dequant + bias into a preallocated output buffer");
    m.def("conv2d_int8_fprop_no_ohat_prealloc_bias_residual", &conv2d_int8_fprop_no_ohat_prealloc_bias_residual, "Fused INT8 conv + dequant + bias + residual (skip-add) into a preallocated output buffer");
    m.def("conv2d_int8_fprop_relu_requant_int8", &conv2d_int8_fprop_relu_requant_int8, "INT8 conv + dequant + bias + optional ReLU, requantized to INT8 output (for int8 conv->conv chaining)");
    m.def("conv2d_int8_fprop_deepfuse_relu_requant_int8", &conv2d_int8_fprop_deepfuse_relu_requant_int8, "Deep-fuse INT8 conv (weight_scale in CUTLASS epilogue, no fp32 temp) + bias + ReLU + requant to INT8");
    m.def("conv2d_int8_fprop_deepfuse_bias_residual_fp16", &conv2d_int8_fprop_deepfuse_bias_residual_fp16, "Deep-fuse + tunable INT8 conv (no fp32 temp) + bias + skip residual, FP16 out");
    m.def("conv2d_int8_fprop_deepfuse_bias_residual_dual", &conv2d_int8_fprop_deepfuse_bias_residual_dual, "Deep-fuse INT8 conv3 + bias + skip residual + ReLU, DUAL output: FP16 (x_{N+1}) + requantized INT8 (next block conv1 input) -- fuses the block-entry quantize");
    m.def("conv2d_int8_fprop_no_ohat", &conv2d_int8_fprop_no_ohat, "Fused INT8 conv + dequant without o_hat update");

    // CUTLASS INT4 Conv2d (kernels/conv2d_int4.cu)
    m.def("conv2d_int4_fprop", &conv2d_int4_fprop, "Conv2d INT4 Forward (CUTLASS)");
    m.def("conv2d_int4_fprop_no_ohat_prealloc", &conv2d_int4_fprop_no_ohat_prealloc, "Fused INT4 conv + dequant into a preallocated output buffer");
    m.def("conv2d_int4_fprop_no_ohat_prealloc_bias", &conv2d_int4_fprop_no_ohat_prealloc_bias, "Fused INT4 conv + dequant + bias into a preallocated output buffer");
    m.def("conv2d_int4_fprop_no_ohat_prealloc_bias_residual", &conv2d_int4_fprop_no_ohat_prealloc_bias_residual, "Fused INT4 conv + dequant + bias + residual (skip-add) into a preallocated output buffer");
    m.def("conv2d_int4_num_tuned_configs", &conv2d_int4_num_tuned_configs, "Number of autotunable tile configs for the INT4 conv");
    m.def("conv2d_int4_fprop_tuned", &conv2d_int4_fprop_tuned, "INT4 conv (FP32 out) using tile config `config_id` (per-shape autotuning; <0 = default tile)");
    m.def("conv2d_int4_dequant_fp16_tuned", &conv2d_int4_dequant_fp16_tuned, "Deep-fuse INT4 conv (per-channel weight_scale folded into CUTLASS epilogue -> FP16 out, no fp32 temp) using tile config `config_id`");
    m.def("conv2d_int4_fprop_relu_requant_int4", &conv2d_int4_fprop_relu_requant_int4, "INT4 conv + dequant + bias + ReLU, requantized+packed to INT4 output (int4 conv->conv chaining)");
    m.def("conv2d_int4_fprop_bias_residual_dual", &conv2d_int4_fprop_bias_residual_dual, "INT4 conv3 + bias + residual + ReLU, DUAL output: FP16 (x_{N+1}) + requantized packed INT4 (next block conv1) -- fuses the block-entry quantize");
    m.def("conv2d_int4_fprop_no_ohat", &conv2d_int4_fprop_no_ohat, "Fused INT4 conv + dequant without o_hat update");
    m.def("conv2d_int4_fprop_o_hat", &conv2d_int4_fprop_o_hat, "Fused INT4 Conv + o_hat accumulate");

    // Attention Conv1d layout-transform fusions (kernels/layout_transform.cu)
    m.def("fp16_ncw_to_fp32_cl", &fp16_ncw_to_fp32_cl,
          "Fused FP16 [N,C,L] → FP32 [N*L,C,1,1] channels-last (K1+K2 fusion)");
    m.def("fp32_cl_to_fp16_ncw", &fp32_cl_to_fp16_ncw,
          "Fused FP32 [N*L,C,1,1] channels-last → FP16 [N,C,L] (K7+K8 fusion)");
    m.def("fp16_ncw_delta_to_int8_cl", &fp16_ncw_delta_to_int8_cl,
          "Fused FP16 [N,C,L] → INT8 [N*L,C,1,1] CL with MoDiff delta subtract+quantize (K1+K2+K3 fusion)");

    // Native channels_last GroupNorm(+SiLU) (kernels/group_norm_silu.cu)
    m.def("group_norm_silu_nhwc", &group_norm_silu_nhwc,
          "GroupNorm (+ optional fused SiLU) operating natively on NHWC-physical memory, "
          "never materializing an NCHW intermediate");
    m.def("group_norm_silu_quantize_nhwc", &group_norm_silu_quantize_nhwc,
          "GroupNorm (+ optional SiLU) that quantizes its output to INT8 inline (out*scale, "
          "clamp/round; optional per-channel smooth_inv), fusing away the separate quantize kernel");
    m.def("group_norm_silu_quantize_pack_nhwc", &group_norm_silu_quantize_pack_nhwc,
          "GroupNorm (+ optional SiLU) that quantizes to INT4 and packs channel pairs inline "
          "([N,H,W,C/2] byte layout matching scale_quantize_and_pack); requires even CPG");
    m.def("group_norm_silu_dequant_quantize_nhwc", &group_norm_silu_dequant_quantize_nhwc,
          "INT8-in GroupNorm(+SiLU): reads int8 activation + dequant scale (upstream conv's "
          "int8 output), computes GN from dequantized values, requantizes to int8 output");
    m.def("fused_gn_qkv", &fused_gn_qkv, "Fused GroupNorm->qkv (per-sample scale/bias mainloop fusion)");
    m.def("fused_gn_qkv_int8", &fused_gn_qkv_int8, "Fused GroupNorm->qkv with int8-clamp output (oscale folded into weight/bias)");

    // Fused int8 flash attention (kernels/flash_attn_int8.cu)
    m.def("flash_attn_int8", &flash_attn_int8,
          "Fused int8 flash attention (QK^T/softmax/AV, no T x T materialization); "
          "q/k/v [N,H,T,hd_pad] int8, sq/sk [N,H,T], sv [N,H,hd] f32 -> fp16 [N,H,T,hd]");
    m.def("mma_smoke", &mma_smoke, "Debug: C[16,N]=A[16,K].B[N,K]^T via m16n8k32.s8 tensor cores");
    m.def("gemm_w8a8", &gemm_w8a8,
          "W8A8 linear: C[M,N] fp16 = (A[M,K] int8 . B[N,K]^T int8) * a_scale * w_scale[n]");
    m.def("gemm_w4a4", &gemm_w4a4,
          "W4A4 linear: C[M,N] fp16 = (A[M,K/2] . B[N,K/2]^T packed int4) * a_scale * w_scale[n]");
    m.def("gemm_w8a8_out_int8", &gemm_w8a8_out_int8,
          "W8A8 linear, int8 output = round(acc*a_scale*w_scale[c]*oscale[c]) (fused qkv->flash path)");
    m.def("gemm_w4a4_out_int8", &gemm_w4a4_out_int8,
          "W4A4 linear, int8 output = round(acc*a_scale*w_scale[c]*oscale[c]) (fused qkv->flash path)");
    m.def("quantize_act_int8", &quantize_act_int8, "Fused fp16->int8 activation quantize (static scale)");
    m.def("quantize_act_int4_pack", &quantize_act_int4_pack, "Fused fp16->packed-int4 activation quantize (static scale)");

    // Fused qkv quantize for the flash score path (kernels/quantize_qkv.cu)
    m.def("quantize_qkv_int8", &quantize_qkv_int8,
          "packed qkv [B,T,nh,3,hd] fp16 -> {qi,ki,vi [B,nh,T,hd_pad] int8, sq,sk [B,nh,T], sv [B,nh,hd] f32}");
    m.def("transpose_qkv_int8", &transpose_qkv_int8,
          "int8 packed qkv [B,T,nh,3,hd] -> {qi,ki,vi [B,nh,T,hd_pad] int8} head-major, hd zero-padded");

    // Standard quantized attention (attn_quant_gemm.cu)
    m.def("attn_qk_int8", &attn_qk_int8,
          "batched int8 QKᵀ: Q,K int8 [BH,T,hd_pad] -> S [BH,T,T] fp32 (raw accumulator)");
    m.def("attn_softmax_requant", &attn_softmax_requant,
          "softmax(dequant S · sq·sk·scale) -> {P int8 [BH,T,T] in [0,127], sp [BH,T]}");
    m.def("attn_av_int8", &attn_av_int8,
          "batched int8 AV: P[BH,T,T]·Vtᵀ[BH,hd_pad,T] -> O fp16 [BH,T,hd_pad], dequant sp[row]·sv[col]");
    m.def("attn_qk_int4", &attn_qk_int4, "batched int4 QKᵀ (packed) -> S [BH,T,T] fp16 scaled logits");
    m.def("attn_softmax_requant4", &attn_softmax_requant4, "softmax(fp16 S) -> PACKED int4 P [BH,T,T/2] + sp");
    m.def("attn_av_int4", &attn_av_int4, "batched int4 AV (packed P·Vt) -> O fp16 [BH,T,hd_pad]");
    m.def("quantize_attn_qkv", &quantize_attn_qkv,
          "fused Q/K/V quantize: fp16 [BH,T,hd] -> {qi,ki [BH,T,hp_qk], vt [BH,hp_av,T], sq,sk, sv}");
    // static (calibrated) score path -- no runtime reductions
    m.def("attn_softmax_requant_static", &attn_softmax_requant_static,
          "static-max softmax(S, c) -> {P int8 [BH,T,T], sp}; single read of S (no max pass)");
    m.def("attn_softmax_requant_s8", &attn_softmax_requant_s8,
          "int8-SCORE softmax(S_int8, sS, c) -> {P int8, sp}; halves the T*T score read");
    m.def("attn_softmax_requant4_static", &attn_softmax_requant4_static,
          "static-max softmax(S, c) -> {PACKED int4 P [BH,T,T/2], sp}");
    m.def("attn_softmax_fp16", &attn_softmax_fp16,
          "fp16 materialized softmax(S, static_c, c) -> {P fp16 unnormalized [BH,T,T], rowsum [BH,T]}");
    m.def("quantize_attn_qkv_static", &quantize_attn_qkv_static,
          "static Q/K/V quantize (calibrated sq_c,sk_c per-tensor + sv_vec per-channel; no absmax)");
    m.def("quantize_attn_qkv_from_i8", &quantize_attn_qkv_from_i8,
          "consume int8 qkv-linear output (+oscale) directly -> attention int8 {qi,ki,vt,sq,sk,sv}");
}
