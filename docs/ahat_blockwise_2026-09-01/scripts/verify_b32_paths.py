"""Numerical check of the along-C B=32 int8 a_hat path against a PyTorch reference.

Covers both dispatches of group_norm_silu_delta_quantize_nhwc:
  CPG % 4 == 0 -> gn_apply_delta_quantize_flat_vec4_b32_kernel (4 ch/thread)
  CPG % 4 != 0 -> gn_apply_delta_quantize_flat_vec2_kernel<..., AhatB32=true>

Both use PRMT/FADD magic-number int8 conversion and __fdividef, so exact equality is
not expected: the tolerance below is one code (an approximate reciprocal moves values
sitting on a rounding boundary), on a small fraction of elements.
"""
from __future__ import annotations
import os, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

N, H, W, G = 4, 16, 16, 32
DEV = "cuda"


def reference(x, gamma, beta, codes, s_old, dscale, apply_silu, lim=127.0):
    """Mirrors the kernel op-for-op, including its fp16 round-trip before SiLU."""
    n, c, h, w = x.shape
    o = F.group_norm(x.float(), G, gamma.float(), beta.float(), 1e-5)
    o = o.half().float()
    if apply_silu:
        o = o * torch.sigmoid(o)
    o_nhwc = o.permute(0, 2, 3, 1).contiguous()                      # [N,H,W,C]
    grp = c // 32
    cod = codes.permute(0, 2, 3, 1).contiguous().float().view(n, h, w, grp, 32)
    cfp = (cod * s_old.unsqueeze(-1)).view(n, h, w, c)
    d = o_nhwc - cfp
    # kernel uses roundf: half away from zero, not torch's half to even
    q = torch.sign(d * dscale) * torch.floor((d * dscale).abs() + 0.5)
    q = q.clamp(-lim, lim)
    nc = o_nhwc - d + q / dscale
    g = nc.view(n, h, w, grp, 32).abs().amax(-1).clamp_min(1e-12)
    new_code = torch.round(nc.view(n, h, w, grp, 32) * (127.0 / g).unsqueeze(-1))
    new_code = new_code.clamp(-127, 127).view(n, h, w, c)
    return q, new_code, g / 127.0


def check(C, label):
    torch.manual_seed(7)
    grp = C // 32
    x = torch.randn(N, C, H, W, device=DEV, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    gamma = torch.randn(C, device=DEV, dtype=torch.float16)
    beta = torch.randn(C, device=DEV, dtype=torch.float16) * 0.1
    codes = torch.randint(-127, 128, (N, C, H, W), device=DEV, dtype=torch.int8).contiguous(
        memory_format=torch.channels_last)
    s_old = (torch.rand(N, H, W, grp, device=DEV, dtype=torch.float32) * 0.05 + 0.01)
    dscale = 6.0
    scale = torch.tensor([dscale], device=DEV, dtype=torch.float32)
    empty = torch.empty(0, device=DEV)
    empty_i = torch.empty(0, device=DEV, dtype=torch.int32)

    cache, sc = codes.clone(), s_old.clone()
    yq = mc.group_norm_silu_delta_quantize_nhwc(
        x, gamma, beta, cache, G, 1e-5, True, scale, empty, empty, empty,
        empty, empty, empty, empty_i, 127.0, False, 1.0, False, True, sc)

    q_ref, code_ref, s_ref = reference(x, gamma, beta, codes, s_old, dscale, True)
    yq_k = yq.permute(0, 2, 3, 1).contiguous().float()
    code_k = cache.permute(0, 2, 3, 1).contiguous().float()

    def report(name, k, r, tol):
        diff = (k - r).abs()
        bad = (diff > tol).float().mean().item()
        print(f"    {name:12s} max|diff| {diff.max().item():.4g}   "
              f"frac>|{tol}| {bad:.2e}", flush=True)
        return diff.max().item(), bad

    cpg = C // G
    print(f"  {label}: C={C} CPG={cpg} -> {'vec4' if cpg % 4 == 0 else 'vec2'}", flush=True)
    dy, by = report("yq", yq_k, q_ref, 1.0)
    dc, bc = report("a_hat code", code_k, code_ref, 1.0)
    ds, bs = report("scale", sc, s_ref, 1e-6)
    ok = dy <= 1.0 and dc <= 1.0 and by < 2e-3 and bc < 2e-3 and ds <= 1e-6
    print(f"    -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    print("along-C B=32 int8 a_hat vs PyTorch reference", flush=True)
    allok = True
    for C, label in [(384, "vec4 path"), (192, "vec2 path"),
                     (768, "vec4 path"), (576, "vec2 path")]:
        allok &= check(C, label)
    print("ALL PASS" if allok else "FAILURES", flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
