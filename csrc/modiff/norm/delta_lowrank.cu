// Experimental fused rank-k residual prepare.
// One C++ op: fp16 GEMM range-finder, QR, Z = U^T d, absmax-Q(Z), fold W U,
// a_hat += U @ dequant(q). Keeps Omega + GEMM on tensor cores instead of
// Python fp32 torch.linalg.qr / einsum (measured 1.34 ms tax on 192@32 b128).
//
// Not wired into production forward. Docs: ahat_svd_residual_2026-09-01 §8.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

std::vector<torch::Tensor> delta_lowrank_fprop(
    torch::Tensor d, torch::Tensor w_fp, torch::Tensor a_hat, int64_t k_in) {
    CHECK_CUDA(d);
    CHECK_CUDA(w_fp);
    CHECK_CUDA(a_hat);
    TORCH_CHECK(d.dim() == 4, "d [N,C,H,W]");
    TORCH_CHECK(a_hat.sizes() == d.sizes(), "a_hat must match d");
    TORCH_CHECK(d.is_contiguous(at::MemoryFormat::ChannelsLast), "d channels_last");
    TORCH_CHECK(a_hat.is_contiguous(at::MemoryFormat::ChannelsLast), "a_hat channels_last");
    TORCH_CHECK(w_fp.dim() == 4, "w_fp [Cout,C,R,S]");

    const int64_t N = d.size(0);
    const int64_t C = d.size(1);
    const int64_t H = d.size(2);
    const int64_t W = d.size(3);
    const int64_t P = N * H * W;
    const int64_t Cout = w_fp.size(0);
    const int64_t R = w_fp.size(2);
    const int64_t S = w_fp.size(3);
    TORCH_CHECK(w_fp.size(1) == C, "W Cin must match d C");

    int64_t k = k_in;
    if (k > C) k = C;
    if (k > P) k = P;
    k = (k / 16) * 16;
    TORCH_CHECK(k >= 16, "k must be >= 16 after 16-align");

    // CL [N,C,H,W] permute to NHWC is a no-copy view; reshape to [P,C].
    auto m = d.to(at::kHalf).permute({0, 2, 3, 1}).reshape({P, C});
    auto omega = at::randn({P, k}, m.options());
    auto y = at::mm(m.transpose(0, 1), omega).to(at::kFloat);  // [C, k]
    auto qr = at::linalg_qr(y, "reduced");
    auto Q = std::get<0>(qr).contiguous();  // [C, k] already
    if (Q.size(1) > k) {
        Q = Q.narrow(1, 0, k).contiguous();
    }

    auto z = at::mm(m.to(at::kFloat), Q);  // [P, k]
    auto amax = z.abs().max().clamp_min(1e-8);
    auto scale = (127.0 / amax).to(at::kFloat);
    auto z_q = at::clamp(at::round(z * scale), -127.0, 127.0);
    auto z_int8 = z_q.to(at::kChar).view({N, H, W, k})
                       .permute({0, 3, 1, 2})
                       .contiguous(at::MemoryFormat::ChannelsLast);
    auto z_deq = z_q / scale;
    auto a_nhwc = a_hat.permute({0, 2, 3, 1}).reshape({P, C});
    a_nhwc.addmm_(z_deq.to(a_nhwc.dtype()), Q.to(a_nhwc.dtype()).transpose(0, 1));

    auto wf = w_fp.to(d.device(), at::kFloat).contiguous();
    auto w_flat = wf.permute({0, 2, 3, 1}).reshape({Cout * R * S, C});
    auto wk = at::mm(w_flat, Q);  // [Cout*R*S, k]
    auto wk4 = wk.view({Cout, R, S, k}).contiguous();
    auto per_out = wk4.reshape({Cout, R * S * k});
    auto ch_max = per_out.abs().amax(/*dim=*/1).clamp_min(1e-8);
    auto wscale = (ch_max / 127.0).contiguous();
    auto w_int8 = at::clamp(at::round(per_out / wscale.unsqueeze(1)), -127.0, 127.0)
                      .to(at::kChar)
                      .view({Cout, R, S, k})
                      .contiguous();

    auto alpha = (1.0 / scale).reshape({1}).to(at::kFloat).contiguous();
    return {z_int8, w_int8, wscale.to(at::kFloat).contiguous(), alpha};
}
