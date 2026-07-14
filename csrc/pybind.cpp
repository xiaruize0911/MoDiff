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
    m.def("conv2d_int8_fprop_o_hat", &conv2d_int8_fprop_o_hat, "Fused INT8 Conv + o_hat accumulate");
    m.def("conv2d_int8_fprop_no_ohat_prealloc", &conv2d_int8_fprop_no_ohat_prealloc, "Fused INT8 conv + dequant into a preallocated output buffer");
    m.def("conv2d_int8_fprop_no_ohat_prealloc_bias", &conv2d_int8_fprop_no_ohat_prealloc_bias, "Fused INT8 conv + dequant + bias into a preallocated output buffer");
    m.def("conv2d_int8_fprop_no_ohat_prealloc_bias_residual", &conv2d_int8_fprop_no_ohat_prealloc_bias_residual, "Fused INT8 conv + dequant + bias + residual (skip-add) into a preallocated output buffer");
    m.def("conv2d_int8_fprop_relu_requant_int8", &conv2d_int8_fprop_relu_requant_int8, "INT8 conv + dequant + bias + optional ReLU, requantized to INT8 output (for int8 conv->conv chaining)");
    m.def("conv2d_int8_fprop_deepfuse_relu_requant_int8", &conv2d_int8_fprop_deepfuse_relu_requant_int8, "Deep-fuse INT8 conv (weight_scale in CUTLASS epilogue, no fp32 temp) + bias + ReLU + requant to INT8");
    m.def("conv2d_int8_fprop_no_ohat", &conv2d_int8_fprop_no_ohat, "Fused INT8 conv + dequant without o_hat update");

    // CUTLASS INT4 Conv2d (kernels/conv2d_int4.cu)
    m.def("conv2d_int4_fprop", &conv2d_int4_fprop, "Conv2d INT4 Forward (CUTLASS)");
    m.def("conv2d_int4_fprop_no_ohat_prealloc", &conv2d_int4_fprop_no_ohat_prealloc, "Fused INT4 conv + dequant into a preallocated output buffer");
    m.def("conv2d_int4_fprop_no_ohat_prealloc_bias", &conv2d_int4_fprop_no_ohat_prealloc_bias, "Fused INT4 conv + dequant + bias into a preallocated output buffer");
    m.def("conv2d_int4_fprop_no_ohat_prealloc_bias_residual", &conv2d_int4_fprop_no_ohat_prealloc_bias_residual, "Fused INT4 conv + dequant + bias + residual (skip-add) into a preallocated output buffer");
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
}
