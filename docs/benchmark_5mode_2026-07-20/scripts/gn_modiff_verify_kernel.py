"""Bit-exact check: fused group_norm_silu_delta_quantize[_pack]_nhwc vs the
reference two-kernel modiff path (group_norm_silu_nhwc(apply_silu=False) ->
step1_static_quantize[_pack_int4]_fprop_silu), over multiple iterations so the
fp16 a_hat cache evolves (a 1-step check would miss cache-drift bugs)."""
import torch, modiff_cutlass as m

torch.manual_seed(0)
dev = 'cuda'

def cl(t): return t.contiguous(memory_format=torch.channels_last)

def run_case(is_int4, use_mod, use_smooth, N=2, C=128, H=16, W=16, ng=32, iters=5):
    eps = 1e-5
    gamma = torch.randn(C, device=dev, dtype=torch.float16)
    beta  = torch.randn(C, device=dev, dtype=torch.float16)
    Q = 7.0 if is_int4 else 127.0
    scale = torch.tensor([Q / 3.0], device=dev, dtype=torch.float32)  # static calib scale
    if use_smooth:
        smooth = (0.5 + torch.rand(C, device=dev, dtype=torch.float32))
    else:
        smooth = torch.empty(0, device=dev, dtype=torch.float32)
    if use_mod:
        ms = torch.randn(N, C, device=dev, dtype=torch.float16)
        sh = torch.randn(N, C, device=dev, dtype=torch.float16)
    else:
        ms = sh = torch.empty(0, device=dev, dtype=torch.float16)

    a_ref = cl(torch.zeros(N, C, H, W, device=dev, dtype=torch.float16))
    a_fus = cl(torch.zeros(N, C, H, W, device=dev, dtype=torch.float16))

    max_code_diff = 0
    max_ahat_diff = 0.0
    for it in range(iters):
        x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16) * (1.0 + 0.3 * it))
        # --- reference: standalone GN (no silu) then step1 silu delta-quantize ---
        normed = m.group_norm_silu_nhwc(x, gamma, beta, ng, eps, False, ms, sh)
        if is_int4:
            q_ref = m.step1_static_quantize_pack_int4_fprop_silu(normed, a_ref, scale, smooth)
        else:
            q_ref = m.step1_static_quantize_fprop_silu(normed, a_ref, scale, smooth)
        # --- fused ---
        if is_int4:
            q_fus = m.group_norm_silu_delta_quantize_pack_nhwc(x, gamma, beta, a_fus, ng, eps, True, scale, smooth, ms, sh)
        else:
            q_fus = m.group_norm_silu_delta_quantize_nhwc(x, gamma, beta, a_fus, ng, eps, True, scale, smooth, ms, sh)
        torch.cuda.synchronize()
        cd = (q_ref.int() - q_fus.int()).abs().max().item()
        ad = (a_ref.float() - a_fus.float()).abs().max().item()
        max_code_diff = max(max_code_diff, cd)
        max_ahat_diff = max(max_ahat_diff, ad)
    tag = f"{'int4' if is_int4 else 'int8'} mod={int(use_mod)} smooth={int(use_smooth)}"
    ok = (max_code_diff == 0 and max_ahat_diff == 0.0)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:26s} max_code_diff={max_code_diff}  max_ahat_diff={max_ahat_diff:.3e}")
    return ok

allok = True
for is_int4 in (False, True):
    for use_mod in (False, True):
        for use_smooth in (False, True):
            allok &= run_case(is_int4, use_mod, use_smooth)
print("ALL PASS" if allok else "SOME FAILED")
