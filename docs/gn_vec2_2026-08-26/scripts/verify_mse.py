"""Numerical correctness check: MSE against an independent fp64 reference.

Everything committed under gn_vec2_2026-08-26 so far is a BIT-IDENTITY gate (torch.equal against
the OLD kernel this change replaces). That proves "vec2 changed nothing", but never checks the old
kernel was computing the right thing in the first place. This does that check independently, by
recomputing GroupNorm + SiLU + delta-quantize + the a_hat update in float64 pure PyTorch and
scoring the actual CUDA kernels (scalar AND vec2, where both exist) against it with MSE / max-abs-
error -- the same "vs fp64" convention test_cat2_gn_fold.py already uses in this repo.

Three parts:
  A. GN stats alone (mean, inv_std): scalar (v0) and vec2 (v3) chanmajor kernels from the
     gn_stats probe built earlier this session, vs a pure-torch fp64 group mean/var.
  B. The full flat delta-quantize pipeline (group_norm_silu_delta_quantize_nhwc), scalar
     (MODIFF_GN_STATS_VEC2=0) and vec2 (=1) arms, vs an fp64 replay of the kernel's own formula
     (mean/inv_std/normed/silu/delta/round/dequant), including the __half round-trips the kernel
     itself performs. Codes are checked for EXACT match (they are integers); a_hat is scored by
     MSE/max-abs since it carries fp16/fp32 rounding.
  C. The resize kernel's a_hat, both directions (UP/DOWN), current build only (this fix has no
     runtime toggle) -- same fp64 replay, same scoring.

A small residual MSE (float32-epsilon scale, ~1e-13 to 1e-7 depending on the quantity) is EXPECTED
and is not a bug: the kernel's own internal arithmetic is float32, not float64. What is being
verified is that no larger, structural error is present in whichever kernel is checked.
"""
import math
import os
import sys

import torch

sys.path.insert(0, "/tmp/claude-0/-workspace/31e575da-69cf-419d-bc20-66eb029653e9/scratchpad/gn_stats/build")
import gn_stats_probe as P

sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")

torch.manual_seed(0)
N, G, EPS = 8, 32, 1e-5
CL = torch.channels_last
SHAPES = [(192, 32, 32), (384, 16, 16), (768, 4, 4), (576, 32, 32), (1152, 8, 8), (768, 2, 2)]



def round_half_away_from_zero(x):
    """Matches CUDA's roundf(), NOT torch.round()'s round-half-to-even. They differ only at exact
    .5 boundaries, which floating-point delta*scale hits often enough (scale is a power of two
    here) to matter: this was the entire cause of a spurious ~0.1-0.15% code mismatch rate in an
    earlier version of this script, confirmed by switching the reference's rounding convention and
    watching the mismatch collapse to (near) zero. Not a kernel bug -- a reference-script bug."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

def mse(a, b):
    d = (a.double() - b.double())
    return (d * d).mean().item(), d.abs().max().item()


def boundary_report(qraw_flat, yq_flat, q_ref_flat):
    """For elements where the kernel's code and the fp64-replay's code disagree, report how close
    the RAW (pre-round) value sat to a .5 decision boundary. If every disagreement sits within a
    few percent of X.5, the mismatch is floating-point rounding-boundary sensitivity (CUDA's
    single-precision expf/sigmoid vs the host double-precision one landing on opposite sides of an
    infinitesimally-thin decision line) -- not a structural error. A real bug would show large,
    boundary-INDEPENDENT disagreements instead."""
    diff = (yq_flat != q_ref_flat)
    n = int(diff.sum())
    if n == 0:
        return 0, None, None
    idx = diff.nonzero().flatten()
    dist = (qraw_flat[idx].abs() - qraw_flat[idx].abs().floor() - 0.5).abs()
    code_diff = (yq_flat[idx] - q_ref_flat[idx]).abs()
    return n, dist.max().item(), int(code_diff.max().item())


def fp64_group_stats(x):
    """x: [N,C,H,W] any layout. Returns mean, inv_std as [N*G] float64, matching the kernel's
    NG index convention (i = n*G + g) and its EXACT finalize formula (population var, clamp>=0,
    rsqrt(var+eps)) -- just carried out in float64 instead of float32."""
    Nn, C, H, W = x.shape
    CPG = C // G
    xr = x.double().reshape(Nn, G, CPG, H, W)
    mean = xr.mean(dim=(2, 3, 4))
    var = (xr * xr).mean(dim=(2, 3, 4)) - mean * mean
    var = var.clamp(min=0.0)
    inv_std = 1.0 / torch.sqrt(var + EPS)
    return mean.reshape(-1), inv_std.reshape(-1)


# =====================================================================================
# A. GN stats kernels (scalar v0, vec2 v3) vs fp64
# =====================================================================================
print("=== A. GN stats (mean, inv_std) vs fp64 reference ===")
print(f"{'shape':>14} {'variant':>8}  {'mean MSE':>12} {'mean maxerr':>12}  "
      f"{'inv_std MSE':>12} {'inv_std maxerr':>14}")
for C, H, W in SHAPES:
    HW = H * W
    nblocks = min(HW, 32)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    mean_ref, istd_ref = fp64_group_stats(x)
    for variant, tag in ((0, "scalar"), (3, "vec2")):
        ps = torch.empty(N * G * nblocks, device="cuda", dtype=torch.float32)
        pq = torch.empty(N * G * nblocks, device="cuda", dtype=torch.float32)
        P.launch(x, ps, pq, N, C, HW, G, nblocks, variant)
        torch.cuda.synchronize()
        group_size = (C // G) * HW
        s = ps.reshape(N * G, nblocks).sum(dim=1).double()
        sq = pq.reshape(N * G, nblocks).sum(dim=1).double()
        mean_k = s / group_size
        var_k = (sq / group_size - mean_k * mean_k).clamp(min=0.0)
        istd_k = 1.0 / torch.sqrt(var_k + EPS)
        m_mse, m_max = mse(mean_k, mean_ref)
        i_mse, i_max = mse(istd_k, istd_ref)
        print(f"{f'{C},{H}x{W}':>14} {tag:>8}  {m_mse:>12.3e} {m_max:>12.3e}  "
              f"{i_mse:>12.3e} {i_max:>14.3e}")

# =====================================================================================
# B. Full flat delta-quantize pipeline (production kernel) vs fp64
# =====================================================================================

def kernel_group_stats(x, G):
    """The mean/inv_std the ACTUAL CUDA kernel computes internally (float32), not an idealized
    fp64 value -- reusing the gn_stats probe from part A, which is a verbatim copy of the same
    source gn_launch_group_stats dispatches to. Used from here on as the reference's starting
    point, because comparing the REST of the pipeline (normed/silu/delta/quantize/a_hat) against
    an fp64-EXACT mean introduces a confound: any of the ~1e-8-level mean/inv_std discrepancy
    quantified in part A gets amplified through delta*scale, and near a rounding-cell boundary a
    perturbation that small is enough to flip a code by +-1 -- a property of comparing against the
    WRONG (too-exact) upstream stats, not a kernel defect. Isolate the rest of the formula by
    handing it the SAME stats the kernel used."""
    Nn, C, H, W = x.shape
    HW = H * W
    nblocks = min(HW, 32)
    ps = torch.empty(Nn * G * nblocks, device="cuda", dtype=torch.float32)
    pq = torch.empty(Nn * G * nblocks, device="cuda", dtype=torch.float32)
    P.launch(x, ps, pq, Nn, C, HW, G, nblocks, 3)
    torch.cuda.synchronize()
    group_size = (C // G) * HW
    s = ps.reshape(Nn * G, nblocks).sum(dim=1).double()
    sq = pq.reshape(Nn * G, nblocks).sum(dim=1).double()
    mean_k = (s / group_size).float()
    var_k = (sq / group_size - mean_k.double() * mean_k.double()).clamp(min=0.0)
    istd_k = (1.0 / torch.sqrt(var_k + EPS)).float()
    return mean_k.double(), istd_k.double()


print("\n=== B. group_norm_silu_delta_quantize_nhwc (flat, int8) vs fp64 reference ===")


def fp64_delta_quantize(x, gamma, beta, a_hat_prev, scale, lim, mean_ref, istd_ref, G):
    Nn, C, H, W = x.shape
    CPG = C // G
    xr = x.double()
    mean = mean_ref.reshape(Nn, G, 1, 1, 1).expand(Nn, G, CPG, H, W).reshape(Nn, C, H, W)
    istd = istd_ref.reshape(Nn, G, 1, 1, 1).expand(Nn, G, CPG, H, W).reshape(Nn, C, H, W)
    g = gamma.double().view(1, C, 1, 1)
    b = beta.double().view(1, C, 1, 1)
    normed = (xr - mean) * istd * g + b
    normed_h = normed.half().double()          # __half2float(__float2half(n))
    out = normed_h * torch.sigmoid(normed_h)   # SiLU
    delta = out - a_hat_prev.double()
    qraw = delta * scale
    q = round_half_away_from_zero(qraw).clamp(-lim, lim)
    new_ahat = (a_hat_prev.double() + q / scale)
    new_ahat_fp16 = new_ahat.half().double()   # __float2half_rn
    return q.to(torch.int64), new_ahat_fp16, qraw


import modiff_cutlass as mc  # noqa: E402

E32 = torch.empty(0, device="cuda", dtype=torch.float32)
E16 = torch.empty(0, device="cuda", dtype=torch.float16)
EI = torch.empty(0, device="cuda", dtype=torch.int32)
Q_LEVEL, SAFETY, LIM = 127.0, 1.0, 127.0
print(f"{'shape':>14} {'arm':>8}  {'code exact match':>17}  {'a_hat MSE':>12} {'a_hat maxerr':>13}  {'max dist to .5':>14} {'max |code diff|':>15}")
for C, H, W in SHAPES:
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    gamma = (torch.randn(C, device="cuda", dtype=torch.float16).abs() + 0.5)
    beta = torch.randn(C, device="cuda", dtype=torch.float16) * 0.1
    a_hat0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=CL)
    scale_val = 16.0
    scale = torch.tensor([scale_val], device="cuda", dtype=torch.float32)
    mean_ref, istd_ref = kernel_group_stats(x, G)
    q_ref, ahat_ref, qraw = fp64_delta_quantize(x, gamma, beta, a_hat0, scale_val, LIM, mean_ref, istd_ref, G)

    for env_val, tag in (("0", "scalar"), ("1", "vec2")):
        os.environ["MODIFF_GN_STATS_VEC2"] = env_val
        a_hat = a_hat0.clone()
        yq = mc.group_norm_silu_delta_quantize_nhwc(x, gamma, beta, a_hat, G, EPS, True, scale,
                                                     E32, E16, E16, E32, E32, E32, EI,
                                                     Q_LEVEL, False, SAFETY)
        torch.cuda.synchronize()
        yq_flat = yq.permute(0, 2, 3, 1).contiguous().reshape(-1).to(torch.int64)
        q_ref_flat = q_ref.permute(0, 2, 3, 1).contiguous().reshape(-1)
        qraw_flat = qraw.permute(0, 2, 3, 1).contiguous().reshape(-1)
        code_match = torch.equal(yq_flat, q_ref_flat)
        n_diff, max_dist, max_cdiff = boundary_report(qraw_flat, yq_flat, q_ref_flat)
        a_mse, a_max = mse(a_hat, ahat_ref)
        tagstr = "EXACT" if code_match else f"{n_diff} of {yq_flat.numel()} differ"
        dist_s = f"{max_dist:.4f}" if max_dist is not None else "--"
        cdiff_s = str(max_cdiff) if max_cdiff is not None else "--"
        print(f"{f'{C},{H}x{W}':>14} {tag:>8}  {tagstr:>17}  {a_mse:>12.3e} {a_max:>13.3e}  "
              f"{dist_s:>14} {cdiff_s:>15}")
del os.environ["MODIFF_GN_STATS_VEC2"]

# =====================================================================================
# C. Resize kernel a_hat (UP + DOWN), current build, vs fp64
# =====================================================================================
print("\n=== C. group_norm_silu_delta_quantize_resize_nhwc a_hat vs fp64 reference ===")
print(f"{'shape':>14} {'dir':>5}  {'code exact match':>17}  {'a_hat MSE':>12} {'a_hat maxerr':>13}  {'max dist to .5':>14} {'max |code diff|':>15}")

for C, H, W in SHAPES[:4]:
    for resize, tag in ((1, "UP"), (-1, "DOWN")):
        Ho, Wo = (H * 2, W * 2) if resize > 0 else (H // 2, W // 2)
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
        gamma = (torch.randn(C, device="cuda", dtype=torch.float16).abs() + 0.5)
        beta = torch.randn(C, device="cuda", dtype=torch.float16) * 0.1
        a_hat0 = (0.1 * torch.randn(N, C, Ho, Wo, device="cuda", dtype=torch.float16)).to(memory_format=CL)
        scale_val = 16.0
        scale = torch.tensor([scale_val], device="cuda", dtype=torch.float32)
        mean_ref, istd_ref = kernel_group_stats(x, G)

        # fp64 replay of the resize math: nearest-neighbour (UP, each input pixel copied to a
        # 2x2 block) / 2x2 average pool (DOWN), THEN the same delta-quantize as part B.
        CPG = C // G
        mean = mean_ref.reshape(N, G, 1, 1, 1).expand(N, G, CPG, H, W).reshape(N, C, H, W)
        istd = istd_ref.reshape(N, G, 1, 1, 1).expand(N, G, CPG, H, W).reshape(N, C, H, W)
        normed = (x.double() - mean) * istd * gamma.double().view(1, C, 1, 1) + beta.double().view(1, C, 1, 1)
        normed_h = normed.half().double()
        out = normed_h * torch.sigmoid(normed_h)
        if resize > 0:
            out_r = out.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        else:
            out_r = out.reshape(N, C, Ho, 2, Wo, 2).mean(dim=(3, 5))
        delta = out_r - a_hat0.double()
        q_ref = round_half_away_from_zero(delta * scale_val).clamp(-LIM, LIM)
        ahat_ref = (a_hat0.double() + q_ref / scale_val).half().double()

        a_hat = a_hat0.clone()
        yq = mc.group_norm_silu_delta_quantize_resize_nhwc(
            x, gamma, beta, G, EPS, True, scale, E32, E16, E16, 0, resize, False, a_hat,
            E32, E32, E32, EI, Q_LEVEL, False, SAFETY, False)
        torch.cuda.synchronize()
        yq_flat = yq.permute(0, 2, 3, 1).contiguous().reshape(-1).to(torch.int64)
        q_ref_flat = q_ref.permute(0, 2, 3, 1).contiguous().reshape(-1)
        qraw_flat = (delta * scale_val).permute(0, 2, 3, 1).contiguous().reshape(-1)
        code_match = torch.equal(yq_flat, q_ref_flat)
        n_diff, max_dist, max_cdiff = boundary_report(qraw_flat, yq_flat, q_ref_flat)
        a_mse, a_max = mse(a_hat, ahat_ref)
        tagstr = "EXACT" if code_match else f"{n_diff} of {yq_flat.numel()} differ"
        dist_s = f"{max_dist:.4f}" if max_dist is not None else "--"
        cdiff_s = str(max_cdiff) if max_cdiff is not None else "--"
        print(f"{f'{C},{H}x{W}':>14} {tag:>5}  {tagstr:>17}  {a_mse:>12.3e} {a_max:>13.3e}  "
              f"{dist_s:>14} {cdiff_s:>15}")

print("\nDone. Expected: part A near machine precision (~1e-9 to 1e-19). Parts B/C: a small,")
print("nonzero rate of +-1 code mismatches is EXPECTED wherever the raw pre-round value sits within")
print("a few percent of a .5 decision boundary ('max dist to .5' column) -- that is floating-point")
print("rounding-boundary sensitivity between CUDA's single-precision expf/sigmoid and the host's")
print("double-precision one, not a defect. 'max |code diff|' should be exactly 1 whenever nonzero;")
print("anything larger, or a mismatch far from a boundary, would indicate a real problem. This is")
print("the same convention test_cat2_gn_fold.py's own 'vs fp64' column and OPEN_ITEMS A0's")
print("'matches to <=1 code' criterion already use elsewhere in this project.")
