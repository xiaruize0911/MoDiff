#include <torch/extension.h>

#include "modiff_kernels_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Standalone elementwise quantize/pack/dequant-accumulate (kernels/quantize.cu)
    m.def("quantize_and_pack", &quantize_and_pack, "Fast Quantization and Packing for INT4");
    m.def("scale_quantize_and_pack", &scale_quantize_and_pack, "Fused Scale + Quantize + Pack for INT4");
    m.def("scale_quantize_int8", &scale_quantize_int8, "Fused Scale + Quantize for INT8");
    m.def("dequant_bias_i8", &dequant_bias_i8, "fused dequant + per-col bias for int8-output GEMM: in_i8*out_scale+bias -> fp16");
    m.def("quantize_attn_out_int4_pack", &quantize_attn_out_int4_pack, "int4 variant of quantize_attn_out_int8: transpose + int4 quantize + pack -> int8 [b*T,k_pad/2] (real ch 0..C-1 packed, C..k_pad-1 zero-filled; k_pad<=0 -> C)");
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
    m.def("delta_absmax_fp16", &delta_absmax_fp16, "FP16-cache delta absmax + dynamic scale (reduction only)");
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
    m.def("upsample2x_quantize_noahat_fprop", &upsample2x_quantize_noahat_fprop, "fused Upsample(nearest,2x) + static int8 quantize (baseline conv, no a_hat)");
    m.def("upsample2x_quantize_pack_noahat_fprop", &upsample2x_quantize_pack_noahat_fprop, "fused Upsample(nearest,2x) + static int4 quantize+pack (baseline conv, no a_hat)");
    m.def("avgpool2x_quantize_noahat_fprop", &avgpool2x_quantize_noahat_fprop, "fused Downsample(avg_pool,2x2) + static int8 quantize (baseline conv, no a_hat)");
    m.def("avgpool2x_quantize_pack_noahat_fprop", &avgpool2x_quantize_pack_noahat_fprop, "fused Downsample(avg_pool,2x2) + static int4 quantize+pack (baseline conv, no a_hat)");
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
    m.def("conv2d_int8_fprop_o_hat_residual", &conv2d_int8_fprop_o_hat_residual,
          "INT8 Conv + o_hat accumulate + fused ResBlock skip-add into a separate output");
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
    m.def("conv2d_int4_dequant_fp16_tuned", &conv2d_int4_dequant_fp16_tuned, "Deep-fuse INT4 conv (per-channel weight_scale folded into CUTLASS epilogue -> FP16 out, no fp32 temp) using tile config `config_id`");
    m.def("conv2d_int4_fprop_deepfuse_bias_residual_fp16", &conv2d_int4_fprop_deepfuse_bias_residual_fp16,
          "Deep-fuse INT4 conv (weight_scale in epilogue -> fp16) + per-channel bias + optional skip residual via a from_half store (no fp32 temp)");
    m.def("conv2d_int4_fprop_relu_requant_int4", &conv2d_int4_fprop_relu_requant_int4, "INT4 conv + dequant + bias + ReLU, requantized+packed to INT4 output (int4 conv->conv chaining)");
    m.def("conv2d_int4_fprop_bias_residual_dual", &conv2d_int4_fprop_bias_residual_dual, "INT4 conv3 + bias + residual + ReLU, DUAL output: FP16 (x_{N+1}) + requantized packed INT4 (next block conv1) -- fuses the block-entry quantize");
    m.def("conv2d_int4_fprop_no_ohat", &conv2d_int4_fprop_no_ohat, "Fused INT4 conv + dequant without o_hat update");
    m.def("conv2d_int4_fprop_o_hat", &conv2d_int4_fprop_o_hat, "Fused INT4 Conv + o_hat accumulate");
    m.def("conv2d_int4_fprop_o_hat_residual", &conv2d_int4_fprop_o_hat_residual,
          "INT4 Conv + o_hat accumulate + fused ResBlock skip-add into a separate output");

    // EVT-fused conv epilogues (kernels/conv2d_evt.cu): scale+bias+residual / o_hat dual-store,
    // no post-conv scratch tensor. Bit-exact replacements for the deepfuse+store / o_hat paths.
    m.def("conv2d_int8_evt_bias_residual_fp16", &conv2d_int8_evt_bias_residual_fp16,
          "EVT INT8 conv: acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16 (single kernel, no scratch)");
    m.def("conv2d_int4_evt_bias_residual_fp16", &conv2d_int4_evt_bias_residual_fp16,
          "EVT INT4 conv: acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16 (single kernel, no scratch)");
    m.def("conv2d_int8_evt_o_hat_residual", &conv2d_int8_evt_o_hat_residual,
          "EVT INT8 conv: o_hat[elem] += acc*alpha*weight_scale[k] (in place) ; out = o_hat_new + residual -> fp16 (dual store, no fp32 round-trip)");
    m.def("conv2d_int4_evt_o_hat_residual", &conv2d_int4_evt_o_hat_residual,
          "EVT INT4 conv: o_hat[elem] += acc*alpha*weight_scale[k] (in place) ; out = o_hat_new + residual -> fp16 (dual store, no fp32 round-trip)");
    m.def("conv2d_int8_evt_o_hat", &conv2d_int8_evt_o_hat,
          "EVT INT8 conv: o_hat[elem] += acc*alpha*weight_scale[k] (in place, no residual, no fp32 round-trip)");
    m.def("conv2d_int4_evt_o_hat", &conv2d_int4_evt_o_hat,
          "EVT INT4 conv: o_hat[elem] += acc*alpha*weight_scale[k] (in place, no residual, no fp32 round-trip)");

    // Attention Conv1d layout-transform fusions (kernels/layout_transform.cu)
    m.def("fp16_ncw_to_fp32_cl", &fp16_ncw_to_fp32_cl,
          "Fused FP16 [N,C,L] → FP32 [N*L,C,1,1] channels-last (K1+K2 fusion)");
    m.def("fp32_cl_to_fp16_ncw", &fp32_cl_to_fp16_ncw,
          "Fused FP32 [N*L,C,1,1] channels-last → FP16 [N,C,L] (K7+K8 fusion)");
    m.def("fp16_ncw_delta_to_int8_cl", &fp16_ncw_delta_to_int8_cl,
          "Fused FP16 [N,C,L] → INT8 [N*L,C,1,1] CL with MoDiff delta subtract+quantize (K1+K2+K3 fusion)");
    m.def("cat2_channels_last_fp16", &cat2_channels_last_fp16,
          "Specialized vectorized 2-tensor channels-last FP16 concat along dim=1 (replaces torch.cat for UNetModel's decoder skip-concat)");

    // Native channels_last GroupNorm(+SiLU) (kernels/group_norm_silu.cu)
    m.def("group_norm_silu_nhwc", &group_norm_silu_nhwc,
          "GroupNorm (+ optional fused SiLU) operating natively on NHWC-physical memory, "
          "never materializing an NCHW intermediate");
    m.def("group_norm_silu_quantize_nhwc", &group_norm_silu_quantize_nhwc,
          "GroupNorm (+ optional SiLU) that quantizes its output to INT8 inline (out*scale, "
          "clamp/round; optional per-channel smooth_inv), fusing away the separate quantize kernel");
    m.def("group_norm_silu_quantize_nhwc_fast", &group_norm_silu_quantize_nhwc_fast,
          "Attention GroupNorm+INT8 quantize with pair-vectorized reduction");
    // k_pad is the only optional arg in this module: taking &f drops the C++ default, and every
    // existing caller (fused_resblock.py, token_major_attention.py, the docs/ scripts) passes 10
    // args, so the default has to be declared here for them to keep working.
    m.def("group_norm_silu_quantize_pack_nhwc", &group_norm_silu_quantize_pack_nhwc,
          "GroupNorm (+ optional SiLU) that quantizes to INT4 and packs channel pairs inline "
          "([N,H,W,k_pad/2] byte layout matching scale_quantize_and_pack); requires even CPG. "
          "k_pad (default 0 = no padding) zero-pads the row to the int4 GEMM's K alignment.",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("num_groups"), pybind11::arg("eps"), pybind11::arg("apply_silu"),
          pybind11::arg("scale"), pybind11::arg("smooth_inv"),
          pybind11::arg("mod_scale"), pybind11::arg("mod_shift"),
          pybind11::arg("k_pad") = 0);
    m.def("group_norm_silu_quantize_resize_nhwc", &group_norm_silu_quantize_resize_nhwc,
          "Fused GroupNorm+SiLU+quantize+2x resize; resize=+1 up, -1 down; pack=int4 nibbles");
    m.def("group_norm_silu_quantize_pack_nhwc_fast", &group_norm_silu_quantize_pack_nhwc_fast,
          "Attention-only INT4 GroupNorm+pack with pair-vectorized warp reduction.",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("num_groups"), pybind11::arg("eps"), pybind11::arg("apply_silu"),
          pybind11::arg("scale"), pybind11::arg("smooth_inv"),
          pybind11::arg("mod_scale"), pybind11::arg("mod_shift"),
          pybind11::arg("k_pad") = 0);
    m.def("group_norm_silu_dequant_quantize_nhwc", &group_norm_silu_dequant_quantize_nhwc,
          "INT8-in GroupNorm(+SiLU): reads int8 activation + dequant scale (upstream conv's "
          "int8 output), computes GN from dequantized values, requantizes to int8 output");
    m.def("group_norm_silu_delta_quantize_nhwc", &group_norm_silu_delta_quantize_nhwc,
          "MoDiff-fused GroupNorm(+mod)+SiLU + INT8 temporal-delta quantize + in-place a_hat "
          "update (fuses the modiff GN+step1_static_quantize_fprop_silu two-kernel pass). Pass "
          "empty absmax_buf/scale_out/inv_scale_out/retire_count for the static scale, or real "
          "1-element buffers for the dynamic per-call scale (adds one reduction pass, cannot clip)");
    m.def("group_norm_silu_delta_quantize_resize_nhwc", &group_norm_silu_delta_quantize_resize_nhwc,
          "MoDiff-fused GroupNorm(+mod)+SiLU + 2x resize + temporal-delta quantize + in-place "
          "a_hat update (the updown ResBlocks' fusion, previously modiff-excluded)");
    m.def("group_norm_silu_delta_quantize_pack_nhwc", &group_norm_silu_delta_quantize_pack_nhwc,
          "MoDiff-fused GroupNorm(+mod)+SiLU + INT4 delta-quantize+pack + in-place a_hat update "
          "(int4 counterpart; requires even channels-per-group). Same static/dynamic scale "
          "contract as the INT8 sibling, Q_level 7.0");
    m.def("fused_gn_qkv", &fused_gn_qkv, "Fused GroupNorm->qkv (per-sample scale/bias mainloop fusion)");
    m.def("fused_gn_qkv_i8evt", &fused_gn_qkv_i8evt, "Fused GroupNorm->qkv, int8 output via fp32-bias/int8-clamp EVT epilogue (signed-qkv-correct)");

    // Fused int8/int4 flash attention (tensor-core, scores kept in SRAM, fp32 online softmax).
    m.def("flash_attn_int8_vt", &flash_attn_int8_vt,
          "Fused int8 flash attention, V PRE-TRANSPOSED [N,H,hd_pad,T] (skips internal transpose; mma path only)");
    m.def("flash_attn_int8_vt_static", &flash_attn_int8_vt_static,
          "Steady-state int8 flash with frozen scalar Q/K scales folded into the kernel");
    m.def("flash_attn_int8_qpacked_kv_static_qout", &flash_attn_int8_qpacked_kv_static_qout,
          "Hybrid int8 flash with fused calibrated int8 projection input");
    m.def("flash_attn_int8_qi8packed_kv_static_qout",
          &flash_attn_int8_qi8packed_kv_static_qout,
          "Hybrid int8 flash reading Q directly from a pre-quantized packed-int8 QKV tensor");
    m.def("flash_attn_int8_qi8packed_kv_static_qout_preg",
          &flash_attn_int8_qi8packed_kv_static_qout_preg,
          "T1024/hd24 register-P int8 flash specialization");
    m.def("flash_attn_int8_qi8_kv_static_qout",
          &flash_attn_int8_qi8_kv_static_qout,
          "INT8 flash reading compact Q plus head-major padded K and transposed V");
    m.def("flash_attn_int8_qi8_kv_static_qout_hd24",
          &flash_attn_int8_qi8_kv_static_qout_hd24,
          "Exact T1024/hd24 INT8 shared-P FlashAttention specialization");
    m.def("flash_attn_int8_qi8packed_small_qout",
          &flash_attn_int8_qi8packed_small_qout,
          "Packed-int8 small-sequence attention with fused calibrated projection quantization");
    m.def("flash_attn_i4values_i8mma_qpacked_kv_static_qout",
          &flash_attn_i4values_i8mma_qpacked_kv_static_qout,
          "Signed-int4 Q/K values executed by the K=32 int8 MMA path; packed-int4 projection input");
    m.def("flash_attn_i4values_i8mma_vt_static_qout",
          &flash_attn_i4values_i8mma_vt_static_qout,
          "Materialized signed-int4 Q/K values through int8 MMA; packed-int4 projection input");
    m.def("flash_attn_i4values_small_qout", &flash_attn_i4values_small_qout,
          "INT4 small-shape attention (T4/T16, hd96): dp4a warp-per-query, packed-int4 output");
    m.def("flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24",
          &flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24,
          "Exact T1024/hd24 signed-int4-values route: token-major direct Q from the W4A4 layout "
          "epilogue, int8 MMA, packed-int4 projection input");
    m.def("flash_attn_int4_vt", &flash_attn_int4_vt,
          "Fused int4 flash attention, V PRE-TRANSPOSED int8 [N,H,hdp_v,T] (skips internal transpose)");
    m.def("flash_attn_int4_vt_static", &flash_attn_int4_vt_static,
          "Steady-state int4 flash with frozen scalar Q/K scales folded into the kernel");
    m.def("flash_attn_int4_qpacked_kv_static_qout", &flash_attn_int4_qpacked_kv_static_qout,
          "Hybrid int4 flash with fused calibrated packed-int4 projection input");
    m.def("flash_attn_int8_vt_qout", &flash_attn_int8_vt_qout,
          "int8 flash attention emitting proj-quantized int8 token-major [b*T,C] (fuses quantize_attn_out_int8)");
    m.def("flash_attn_int8_vt_static_qout", &flash_attn_int8_vt_static_qout,
          "Static-scale int8 flash emitting proj-quantized int8 token-major");
    m.def("flash_attn_int4_vt_qout", &flash_attn_int4_vt_qout,
          "int4 flash attention emitting proj-quantized packed-int4 token-major [b*T,k_pad/2] (fuses quantize_attn_out_int4_pack)");
    m.def("flash_attn_int4_vt_static_qout", &flash_attn_int4_vt_static_qout,
          "Static-scale int4 flash emitting proj-quantized packed-int4 token-major");
    m.def("flash_attn_int8_packed_vt", &flash_attn_int8_packed_vt,
          "int8 flash reading packed qkv [b,T,nh,3,hd] directly (fp16->quantize/int8->gather on load); sv f32 [hd] -> fp16 [N,H,T,hd]");
    m.def("flash_attn_int8_packed_vt_qout", &flash_attn_int8_packed_vt_qout,
          "packed-qkv int8 flash emitting proj-quantized int8 token-major [b*T,C] (fuses the flash Q/K/V quantize + attn-out quantize)");
    m.def("flash_attn_int8_packed_persistent_qout",
          &flash_attn_int8_packed_persistent_qout,
          "shape-specialized persistent packed-qkv int8 flash with qout");
    m.def("quantize_attn_qkv_i4qk_i8v", &quantize_attn_qkv_i4qk_i8v,
          "One-pass quantize for int4 flash: Q/K packed int4 + V int8-transposed -> {q4,k4,vt,sq,sk,sv}");
    m.def("quantize_attn_qkv_i4qk_i8v_static", &quantize_attn_qkv_i4qk_i8v_static,
          "Static (calibrated, single-pass) int4 Q/K + int8 V quantize for int4 flash");

    // AWQ w8a8 GEMM, vendored verbatim from llm-awq (MIT, (c) 2023 MIT HAN Lab) -- faster than our
    // gemm_w8a8_awq on the qkv/proj shapes. in[M,K] int8, weight[N,K] int8, wscales[N] half,
    // ascales[M] half (per-token), out[M,N] fp16 PREALLOCATED. N%128==0, K%64==0.
    m.def("gemm_w8a8_awq", &gemm_w8a8_awq,
          "W8A8 Linear (production): AWQ-tiling-scheme GEMM (CTA_M/N=128, ldmatrix+swizzle) -- "
          "C[M,N] fp16 = (A[M,K] int8 . B[N,K]^T int8) * a_scale * w_scale[n]; requires N%128==0, K%64==0");
    m.def("gemm_w8a8_awq_nout", &gemm_w8a8_awq_nout,
          "W8A8 Linear, unpadded output: B/w_scale padded to N%128==0 but writes [M,n_out] directly "
          "(n_out even), skipping padded cols -- removes the downstream slice+.contiguous() copy");
    m.def("gemm_w4a4_awq_nout", &gemm_w4a4_awq_nout,
          "W4A4 Linear, unpadded output: writes [M,n_out] directly (n_out even), skipping padded cols");
    m.def("gemm_w8a8_awq_o_hat", &gemm_w8a8_awq_o_hat,
          "Linear W8A8 GEMM + MoDiff o_hat accumulate in the epilogue (Eq 9, no bias)");
    m.def("gemm_w4a4_awq_o_hat", &gemm_w4a4_awq_o_hat,
          "Linear W4A4 GEMM + MoDiff o_hat accumulate in the epilogue (Eq 9, no bias)");
    m.def("gemm_w8a8_awq_bias_res", &gemm_w8a8_awq_bias_res,
          "W8A8 Linear + fused bias + optional residual in the epilogue (empty tensor = skip); removes the separate bias/residual add");
    m.def("gemm_w4a4_awq_bias_res", &gemm_w4a4_awq_bias_res,
          "W4A4 Linear + fused bias + optional residual in the epilogue (empty tensor = skip)");
    m.def("gemm_w8a8_awq_out_i8", &gemm_w8a8_awq_out_i8,
          "W8A8 Linear, INT8 output (output-fusion): same mainloop, epilogue requantizes to int8 with "
          "inv_out_scale[N]=127/absmax_col -> C[M,N] int8; halves the output write");
    m.def("gemm_w8a8_awq_out_i8_bias_nout",
          &gemm_w8a8_awq_out_i8_bias_nout,
          "W8A8 Linear with bias and per-column INT8 requantization, writing unpadded n_out");
    m.def("gemm_w8a8_awq_qkv_i8_layouts",
          &gemm_w8a8_awq_qkv_i8_layouts,
          "W8A8 QKV GEMM with direct compact-Q, padded-K and transposed-V INT8 epilogue");
    m.def("gemm_w8a8_awq_qkv_i8_layouts_compact",
          &gemm_w8a8_awq_qkv_i8_layouts_compact,
          "W8A8 compact-column QKV GEMM with direct Q, padded-K and transposed-V epilogue");
    m.def("gemm_w4a4_awq", &gemm_w4a4_awq,
          "W4A4 Linear (production): AWQ-tiling-scheme int4 port (CTA_M/N=128, ldmatrix+swizzle) -- "
          "packed int4 A/B; requires N%128==0, K%128==0");
    m.def("gemm_w4a4_awq_out_i8", &gemm_w4a4_awq_out_i8,
          "W4A4 Linear, INT8 output (output-fusion): int4 GEMM, epilogue requantizes to int8; halves the output write");
    m.def("gemm_w4a4_awq_qkv_i4qk_i8v", &gemm_w4a4_awq_qkv_i4qk_i8v,
          "Experimental W4A4 QKV epilogue: bias/dequant + static signed-I4 Q/K and I8 V, "
          "with packed/native or unpacked-I4-value output layouts");
    m.def("gemm_w4a4_awq_qkv_codes", &gemm_w4a4_awq_qkv_codes,
          "W4A4 QKV GEMM emitting compact token-major INT4-value codes, one launch, no rearrange");
    m.def("gemm_w4a4_awq_qkv_i4qk_i8v_layouts", &gemm_w4a4_awq_qkv_i4qk_i8v_layouts,
          "Direct-layout W4A4 QKV epilogue: ONE launch emits token-major Q, padded K and "
          "transposed Vt for Flash (unpacked signed-I4 values); no rearrange pass");
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
    m.def("quantize_attn_qkv_packed_static_compact",
          &quantize_attn_qkv_packed_static_compact,
          "compact packed-qkv static quantize -> {qi,ki,vt,broadcast sv}; omits dead sq/sk");
    m.def("quantize_attn_kv_packed_static", &quantize_attn_kv_packed_static,
          "packed-qkv static K/V-only quantize for the hybrid Q-in-flash path");
    m.def("quantize_attn_qkv_from_i8", &quantize_attn_qkv_from_i8,
          "reshuffle a pre-quantized int8 qkv (fused_gn_qkv_i8evt output) into flash qi/ki/vt (int8->int8, no requant)");
    m.def("quantize_attn_kv_from_i8", &quantize_attn_kv_from_i8,
          "fused K gather and V transpose for pre-quantized packed qkv int8");
    // fp16 (reference) materialized softmax — used by the fp16-materialized attention path
    // (token_major_attention, MODIFF_FP16_MATERIALIZED; the static-vs-dynamic study). Not int8/int4.
    m.def("attn_softmax_fp16", &attn_softmax_fp16,
          "fp16 materialized softmax(S, static_c, c) -> {P fp16 unnormalized [BH,T,T], rowsum [BH,T]}");
}
