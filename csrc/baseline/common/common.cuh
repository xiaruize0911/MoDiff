// ============================================================================================
// DUPLICATED HEADER -- BASELINE copy.  Twin: csrc/modiff/common/common.cuh
//
// cp.async / smem-pointer / misc device helpers.
//
// This file is device-side only (templates and __device__ inlines), so both datapaths can carry
// their own copy without any symbol collision. The copies were made deliberately: csrc/ is split
// into a MoDiff tree and a baseline tree so each datapath can be read, edited and profiled without
// the other in the way, and a shared include directory would have re-coupled them.
//
// THE COST, stated because it is real: these copies can DIVERGE. Anything numerical changed here
// must be changed in the twin, or the two datapaths stop being comparable -- and every A/B in
// docs/ compares them. `diff csrc/baseline/... csrc/modiff/...` is the check; it should come back
// empty for every file whose header says "identical to twin" below.
//
// STATUS: identical to twin (byte-for-byte at the time of the split, 2026-08-12).
// ============================================================================================
// Shared macros used across the modiff_cutlass kernel translation units.
#pragma once

#include <torch/extension.h>

// All CUTLASS conv wrappers in this project expect NHWC-physical (channels_last)
// tensors: the raw pointer arithmetic in conv2d_int8.cu / conv2d_int4.cu and the
// packing kernels in quantize.cu treat flat memory as [N, H, W, C], which is only
// true when the tensor is channels_last-contiguous.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::ChannelsLast), #x " must be channels_last contiguous")

