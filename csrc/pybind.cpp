#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA functions
torch::Tensor conv2d_int4_fprop(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor conv2d_int8_fprop(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor quantize_and_pack(torch::Tensor input);
torch::Tensor scale_quantize_and_pack(torch::Tensor input, torch::Tensor scale);
torch::Tensor scale_quantize_int8(torch::Tensor input, torch::Tensor scale);
void dequant_accumulate_int4(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale);
void dequant_accumulate_int8(torch::Tensor residual, torch::Tensor a_hat_cache, torch::Tensor scale);
void sub_absmax_scale(torch::Tensor x, torch::Tensor a_hat_cache, torch::Tensor residual,
                      torch::Tensor absmax_buf, torch::Tensor scale_out, torch::Tensor inv_scale_out,
                      torch::Tensor retire_count, float Q_level, torch::Tensor smooth_inv);
void scale_accumulate(torch::Tensor conv_output, torch::Tensor weight_scale, torch::Tensor o_hat_cache);

torch::Tensor conv2d_int8_fprop_o_hat(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor o_hat_cache,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);


torch::Tensor step1_quantize_pack_int4_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
);

torch::Tensor conv2d_int4_fprop_o_hat(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor o_hat_cache,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor step1_quantize_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
);

torch::Tensor step1_static_quantize_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv
);

torch::Tensor step1_quantize_no_ahat_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
);

torch::Tensor step1_quantize_pack_int4_no_ahat_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor residual_buf,
    torch::Tensor absmax_buf,
    torch::Tensor scale_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor retire_count,
    float Q_level,
    torch::Tensor smooth_inv
);

torch::Tensor step1_static_quantize_pack_int4_fprop(
    torch::Tensor x,
    torch::Tensor a_hat_cache,
    torch::Tensor scale_buf,
    torch::Tensor smooth_inv
);

torch::Tensor conv2d_int8_fprop_no_ohat(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor conv2d_int8_fprop_no_ohat_prealloc(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor conv2d_int4_fprop_no_ohat(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

torch::Tensor conv2d_int4_fprop_no_ohat_prealloc(
    torch::Tensor input,
    torch::Tensor weight_packed,
    torch::Tensor inv_scale,
    torch::Tensor weight_scales,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_int4_fprop", &conv2d_int4_fprop, "Conv2d INT4 Forward (CUTLASS)");
    m.def("conv2d_int8_fprop", &conv2d_int8_fprop, "Conv2d INT8 Forward (CUTLASS)");
    m.def("quantize_and_pack", &quantize_and_pack, "Fast Quantization and Packing for INT4");
    m.def("scale_quantize_and_pack", &scale_quantize_and_pack, "Fused Scale + Quantize + Pack for INT4");
    m.def("scale_quantize_int8", &scale_quantize_int8, "Fused Scale + Quantize for INT8");
    m.def("dequant_accumulate_int4", &dequant_accumulate_int4, "Fused Dequant + Accumulate for INT4 cache");
    m.def("dequant_accumulate_int8", &dequant_accumulate_int8, "Fused Dequant + Accumulate for INT8 cache");
    m.def("sub_absmax_scale", &sub_absmax_scale, "Fused Subtract + AbsMax + Scale computation");
    m.def("scale_accumulate", &scale_accumulate, "Fused Scale + Accumulate (o_hat += conv * weight_scale)");
    m.def("conv2d_int8_fprop_o_hat", &conv2d_int8_fprop_o_hat, "Fused INT8 Conv + o_hat accumulate");
    
    m.def("step1_quantize_pack_int4_fprop", &step1_quantize_pack_int4_fprop, "Fused sub_absmax_scale + dequant + quantize+pack for INT4 step 1");
    m.def("conv2d_int4_fprop_o_hat", &conv2d_int4_fprop_o_hat, "Fused INT4 Conv + o_hat accumulate");

    m.def("step1_quantize_fprop", &step1_quantize_fprop, "Fused sub_absmax_scale + dequant + quantize for step 1");
    m.def("step1_static_quantize_fprop", &step1_static_quantize_fprop, "Fused static-scale subtract + dequant + quantize for INT8 step 1");
    m.def("step1_quantize_no_ahat_fprop", &step1_quantize_no_ahat_fprop, "Fused sub_absmax_scale + quantize for INT8 step 1 without a_hat update");
    m.def("step1_quantize_pack_int4_no_ahat_fprop", &step1_quantize_pack_int4_no_ahat_fprop, "Fused sub_absmax_scale + quantize+pack for INT4 step 1 without a_hat update");
    m.def("step1_static_quantize_pack_int4_fprop", &step1_static_quantize_pack_int4_fprop, "Fused static-scale subtract + dequant + quantize+pack for INT4 step 1");
    m.def("conv2d_int8_fprop_no_ohat", &conv2d_int8_fprop_no_ohat, "Fused INT8 conv + dequant without o_hat update");
    m.def("conv2d_int4_fprop_no_ohat", &conv2d_int4_fprop_no_ohat, "Fused INT4 conv + dequant without o_hat update");
    m.def("conv2d_int8_fprop_no_ohat_prealloc", &conv2d_int8_fprop_no_ohat_prealloc, "Fused INT8 conv + dequant into a preallocated output buffer");
    m.def("conv2d_int4_fprop_no_ohat_prealloc", &conv2d_int4_fprop_no_ohat_prealloc, "Fused INT4 conv + dequant into a preallocated output buffer");
}
