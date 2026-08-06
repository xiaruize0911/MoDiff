// Shared macros used across the modiff_cutlass kernel translation units.
#pragma once

#include <torch/extension.h>

// All CUTLASS conv wrappers in this project expect NHWC-physical (channels_last)
// tensors: the raw pointer arithmetic in conv2d_int8.cu / conv2d_int4.cu and the
// packing kernels in quantize.cu treat flat memory as [N, H, W, C], which is only
// true when the tensor is channels_last-contiguous.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::ChannelsLast), #x " must be channels_last contiguous")

#ifdef __CUDACC__
// Symmetric code ceiling for the activation quantizers.
//
// Every quantize kernel here used to clamp at a literal -- 127.0f for the int8 paths, 7.0f for the
// int4 ones -- which is correct exactly while the scale is built from the tensor's own absmax, since
// then no code can exceed the ceiling anyway. It stops being correct as soon as the scale is
// deliberately larger than that, which is what a clip ratio is: `MODIFF_DELTA_CLIP=r` sets
// `scale = Q_b/(r*absmax)`, and the codes it produces for the top `(1-r)` of the range are meant to
// SATURATE at +-Q_b. Clamping them at 127 instead lets them through, so below A8 the knob silently
// became a finer grid rather than a clip (measured: docs/delta_clip_2026-08-06/FINDINGS.md).
//
// Callers pass the ceiling explicitly. A value <= 0 means "this datapath's native ceiling", i.e. the
// literal the call site used before this parameter existed -- so an un-migrated caller, and every
// caller at r=1.0, stays bit-identical.
__device__ __forceinline__ float clamp_code(float v, float ceiling, float native) {
    const float c = (ceiling > 0.0f) ? ceiling : native;
    return fmaxf(-c, fminf(c, v));
}
#endif  // __CUDACC__
