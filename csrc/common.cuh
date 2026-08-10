// Shared macros used across the modiff_cutlass kernel translation units.
#pragma once

#include <torch/extension.h>

// All CUTLASS conv wrappers in this project expect NHWC-physical (channels_last)
// tensors: the raw pointer arithmetic in conv2d_int8.cu / conv2d_int4.cu and the
// packing kernels in quantize.cu treat flat memory as [N, H, W, C], which is only
// true when the tensor is channels_last-contiguous.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::ChannelsLast), #x " must be channels_last contiguous")

