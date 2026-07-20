// =============================================================================
// VENDORED FROM llm-awq (https://github.com/mit-han-lab/llm-awq), file
// awq/kernels/csrc/w8a8/w8a8_gemm_cuda.{cu,h}, copied verbatim on 2026-07-19 and
// used directly (only the #include path was adjusted to the local header name).
//
// llm-awq is MIT-licensed:
//   MIT License. Copyright (c) 2023 MIT HAN Lab.
//   Permission is hereby granted, free of charge, ... (full text: LICENSES/LICENSE-llm-awq)
// The original AWQ attribution/citation is preserved in the header comment below.
// =============================================================================
#include <torch/extension.h>

void w8a8_gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _wscales, torch::Tensor _ascales, torch::Tensor _out_feats);
void w8a8_gemm_fuse_bias_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _wscales, torch::Tensor _ascales, torch::Tensor _out_feats, torch::Tensor _bias);