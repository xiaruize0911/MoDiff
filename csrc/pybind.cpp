#include <torch/extension.h>

#include "modiff_kernels_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Standalone elementwise quantize/pack/dequant-accumulate (kernels/quantize.cu)
    m.def("quantize_and_pack", &quantize_and_pack, "Fast Quantization and Packing for INT4");
    m.def("scale_quantize_and_pack", &scale_quantize_and_pack, "Fused Scale + Quantize + Pack for INT4");
    m.def("scale_quantize_int8", &scale_quantize_int8, "Fused Scale + Quantize for INT8");
    m.def("dequant_bias_i8", &dequant_bias_i8, "fused dequant + per-col bias for int8-output GEMM: in_i8*out_scale+bias -> fp16");
    m.def("quantize_attn_out_int4_pack", &quantize_attn_out_int4_pack, "int4 variant of quantize_attn_out_int8: transpose + int4 quantize + pack -> int8 [b*T,C/2]");
    m.def("quantize_attn_out_int8", &quantize_attn_out_int8,
          "Fused attention-output transpose + int8 quantize (proj-side fusion): [b,nh,T,hd] fp16 -> int8 [b*T,nh*hd]");
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
    m.def("step1_static_quantize_noahat_fprop", &step1_static_quantize_noahat_fprop, "cache-free static int8 quantize (baseline conv, no a_hat)");
    m.def("step1_static_quantize_pack_int4_noahat_fprop", &step1_static_quantize_pack_int4_noahat_fprop, "cache-free static int4 quantize+pack (baseline conv, no a_hat)");
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
    m.def("group_norm_silu_delta_quantize_nhwc", &group_norm_silu_delta_quantize_nhwc,
          "MoDiff-fused GroupNorm(+mod)+SiLU + INT8 temporal-delta quantize + in-place a_hat "
          "update (fuses the modiff GN+step1_static_quantize_fprop_silu two-kernel pass)");
    m.def("group_norm_silu_delta_quantize_pack_nhwc", &group_norm_silu_delta_quantize_pack_nhwc,
          "MoDiff-fused GroupNorm(+mod)+SiLU + INT4 delta-quantize+pack + in-place a_hat update "
          "(int4 counterpart; requires even channels-per-group)");
    m.def("fused_gn_qkv", &fused_gn_qkv, "Fused GroupNorm->qkv (per-sample scale/bias mainloop fusion)");
    m.def("fused_gn_qkv_int8", &fused_gn_qkv_int8, "Fused GroupNorm->qkv with int8-clamp output (oscale folded into weight/bias)");

    // Fused int8/int4 flash attention (tensor-core, scores kept in SRAM, fp32 online softmax).
    m.def("flash_attn_int8", &flash_attn_int8,
          "Fused int8 flash attention: q,k,v int8 [N,H,T,hd_pad], sq,sk [N,H,T], sv [N,H,hd] -> out fp16 [N,H,T,hd]");
    m.def("flash_attn_int8_vt", &flash_attn_int8_vt,
          "Fused int8 flash attention, V PRE-TRANSPOSED [N,H,hd_pad,T] (skips internal transpose; mma path only)");
    m.def("flash_attn_int4", &flash_attn_int4,
          "Fused int4 flash attention (int4 QKᵀ, int8 PV): q4,k4 packed [N,H,T,hdp4/2], v int8 [N,H,T,hdp_v] -> fp16");
    m.def("flash_attn_int4_vt", &flash_attn_int4_vt,
          "Fused int4 flash attention, V PRE-TRANSPOSED int8 [N,H,hdp_v,T] (skips internal transpose)");
    m.def("quantize_attn_qkv_i4qk_i8v", &quantize_attn_qkv_i4qk_i8v,
          "One-pass quantize for int4 flash: Q/K packed int4 + V int8-transposed -> {q4,k4,vt,sq,sk,sv}");
    m.def("quantize_attn_qkv_i4qk_i8v_static", &quantize_attn_qkv_i4qk_i8v_static,
          "Static (calibrated, single-pass) int4 Q/K + int8 V quantize for int4 flash");
    m.def("mma_smoke", &mma_smoke, "m16n8k32.s8 fragment-mapping smoke test");

    // AWQ w8a8 GEMM, vendored verbatim from llm-awq (MIT, (c) 2023 MIT HAN Lab) -- faster than our
    // gemm_w8a8_awq on the qkv/proj shapes. in[M,K] int8, weight[N,K] int8, wscales[N] half,
    // ascales[M] half (per-token), out[M,N] fp16 PREALLOCATED. N%128==0, K%64==0.
    m.def("awq_w8a8_gemm", &w8a8_gemm_forward_cuda,
          "AWQ w8a8 GEMM (vendored from llm-awq, MIT): int8xint8 -> fp16, per-token ascale + per-channel wscale");
    m.def("gemm_w8a8_awq", &gemm_w8a8_awq,
          "W8A8 Linear (production): AWQ-tiling-scheme GEMM (CTA_M/N=128, ldmatrix+swizzle) -- "
          "C[M,N] fp16 = (A[M,K] int8 . B[N,K]^T int8) * a_scale * w_scale[n]; requires N%128==0, K%64==0");
    m.def("gemm_w8a8_awq_nout", &gemm_w8a8_awq_nout,
          "W8A8 Linear, unpadded output: B/w_scale padded to N%128==0 but writes [M,n_out] directly "
          "(n_out even), skipping padded cols -- removes the downstream slice+.contiguous() copy");
    m.def("gemm_w4a4_awq_nout", &gemm_w4a4_awq_nout,
          "W4A4 Linear, unpadded output: writes [M,n_out] directly (n_out even), skipping padded cols");
    m.def("gemm_w8a8_awq_bias_res", &gemm_w8a8_awq_bias_res,
          "W8A8 Linear + fused bias + optional residual in the epilogue (empty tensor = skip); removes the separate bias/residual add");
    m.def("gemm_w4a4_awq_bias_res", &gemm_w4a4_awq_bias_res,
          "W4A4 Linear + fused bias + optional residual in the epilogue (empty tensor = skip)");
    m.def("gemm_w8a8_awq_out_i8", &gemm_w8a8_awq_out_i8,
          "W8A8 Linear, INT8 output (output-fusion): same mainloop, epilogue requantizes to int8 with "
          "inv_out_scale[N]=127/absmax_col -> C[M,N] int8; halves the output write");
    m.def("gemm_w4a4_awq", &gemm_w4a4_awq,
          "W4A4 Linear (production): AWQ-tiling-scheme int4 port (CTA_M/N=128, ldmatrix+swizzle) -- "
          "packed int4 A/B; requires N%128==0, K%128==0");
    m.def("gemm_w4a4_awq_out_i8", &gemm_w4a4_awq_out_i8,
          "W4A4 Linear, INT8 output (output-fusion): int4 GEMM, epilogue requantizes to int8; halves the output write");
    m.def("quantize_act_int8", &quantize_act_int8, "Fused fp16->int8 activation quantize (static scale)");
    m.def("quantize_act_int4_pack", &quantize_act_int4_pack, "Fused fp16->packed-int4 activation quantize (static scale)");

    // Quantize prologue for the FUSED (flash) quantized attention path (attn_quant_gemm.cu).
    // Produce per-token int8/int4 Q/K + per-channel int8 (transposed) V + scales for
    // flash_attn_int8_vt / flash_attn_int4_vt. (The materialized QKᵀ/softmax/AV int GEMM
    // attention path was removed; flash is the sole quantized-attention path.)
    m.def("quantize_attn_qkv", &quantize_attn_qkv,
          "fused Q/K/V quantize: fp16 [BH,T,hd] -> {qi,ki [BH,T,hp_qk], vt [BH,hp_av,T], sq,sk, sv}");
    m.def("quantize_attn_qkv_static", &quantize_attn_qkv_static,
          "static Q/K/V quantize (calibrated sq_c,sk_c per-tensor + sv_vec per-channel; no absmax)");
    m.def("quantize_attn_qkv_packed", &quantize_attn_qkv_packed,
          "packed-qkv dynamic quantize (reads interleaved [b,T,nh,3,hd], no transpose copy; QK int8/int4, V int8)");
    m.def("quantize_attn_qkv_packed_static", &quantize_attn_qkv_packed_static,
          "packed-qkv static quantize (calibrated; reads interleaved qkv, no transpose copy)");
    // fp16 (reference) materialized softmax — used by the fp16-materialized attention path
    // (token_major_attention, MODIFF_FP16_MATERIALIZED; the static-vs-dynamic study). Not int8/int4.
    m.def("attn_softmax_fp16", &attn_softmax_fp16,
          "fp16 materialized softmax(S, static_c, c) -> {P fp16 unnormalized [BH,T,T], rowsum [BH,T]}");
}
