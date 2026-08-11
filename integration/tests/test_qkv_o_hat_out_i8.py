"""Does the dual-output qkv GEMM emit the same int8 the three aq_* kernels produce?

This is the gate on the attn_quantize fusion (4.60 ms/step). If it fails, the fusion is dead and no
route restructuring is worth writing; if it passes, what remains is wiring on a proven foundation.

The claim under test: `gemm_w8a8_awq_o_hat_out_i8` advances the fp16 o_hat state AND emits int8 codes
of `o_hat + bias` at a PER-COLUMN scale, such that those codes equal
`quantize(gemm_w8a8_awq_o_hat(...), same per-column scale)`. Per-column matters because flash wants
three different scales -- sq_c and sk_c scalar, sv per-channel -- and the qkv columns are interleaved
(nh, 3, hd), so the scale vector has to be built at that stride.

Deliberately standalone: no model, no route change, nothing that can leave the live qkv path
half-wired. That failure mode is silent -- wrong int8 into attention, no crash -- and this datapath has
paid for it twice (bias dropped rather than moved: latent relL2 0.039 -> 0.300; _scale_buf aliasing:
-> 10.32, both with every kernel unit test green).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import torch, modiff_cutlass as mc

DEV = "cuda"


def build_inv_out_scale(n_pad, C, nh, hd, sq_c, sk_c, sv):
    """[n_pad] f32. Column c of the interleaved (nh, 3, hd) layout belongs to q/k/v by (c//hd)%3."""
    inv = torch.ones(n_pad, device=DEV, dtype=torch.float32)
    for c in range(3 * C):
        which, h = (c // hd) % 3, c % hd
        inv[c] = 1.0 / (sq_c if which == 0 else sk_c if which == 1 else float(sv[h]))
    return inv.contiguous()


def one(M, C, nh, seed):
    hd = C // nh
    torch.manual_seed(seed)
    K = C
    n_out = 3 * C
    n_pad = ((n_out + 127) // 128) * 128
    A = torch.randint(-127, 127, (M, K), device=DEV, dtype=torch.int8)
    B = torch.randint(-127, 127, (n_pad, K), device=DEV, dtype=torch.int8)
    w_scale = (torch.rand(n_pad, device=DEV, dtype=torch.float32) * 0.01 + 0.001).contiguous()
    a_scale = torch.tensor([0.02], device=DEV, dtype=torch.float32)
    bias = (torch.randn(n_out, device=DEV, dtype=torch.float16) * 0.1).contiguous()
    sq_c, sk_c = 0.031, 0.027
    sv = (torch.rand(hd, device=DEV, dtype=torch.float32) * 0.02 + 0.01)
    inv_out = build_inv_out_scale(n_pad, C, nh, hd, sq_c, sk_c, sv)

    o0 = (torch.randn(M, n_out, device=DEV, dtype=torch.float16) * 0.3).contiguous()

    with torch.inference_mode():
        # reference: the o_hat GEMM that ships, then quantize its fp16 output per column
        oh_ref = o0.clone()
        fp16 = mc.gemm_w8a8_awq_o_hat(A, B, w_scale, a_scale, n_out, oh_ref,
                                      torch.empty(0, device=DEV, dtype=torch.float16), bias)
        ref_i8 = torch.clamp(torch.round(fp16.float() * inv_out[:n_out]), -127, 127).to(torch.int8)
        # under test: one pass, o_hat advanced AND int8 emitted
        oh_new = o0.clone()
        got = mc.gemm_w8a8_awq_o_hat_out_i8(A, B, w_scale, a_scale, n_out, oh_new, bias, inv_out)

    got_i8 = got[:, :n_out].to(torch.int8) if got.shape[1] != n_out else got.to(torch.int8)
    d = (got_i8.to(torch.int16) - ref_i8.to(torch.int16)).abs()
    oh_d = (oh_new.float() - oh_ref.float()).abs().max().item()
    return int(d.max()), float((d > 0).float().mean() * 100), oh_d, int(ref_i8.abs().max())


def main():
    print("| M x C (nh) | max|code diff| | codes differing | max|o_hat diff| | max|ref code| |")
    print("|---|---:|---:|---:|---:|")
    bad = []
    for M, C, nh in ((1024, 192, 8), (1024, 384, 8), (256, 768, 8)):
        try:
            md, pct, ohd, mx = one(M, C, nh, seed=7)
        except Exception as e:
            print(f"| {M}x{C} ({nh}) | ERROR: {type(e).__name__}: {str(e)[:70]} |")
            bad.append(f"{M}x{C}: {e}")
            continue
        print(f"| {M}x{C} ({nh}) | {md} | {pct:.3f}% | {ohd:.3e} | {mx} |")
        if md > 1:
            bad.append(f"{M}x{C}: max code diff {md}")
        if ohd > 2e-3:
            bad.append(f"{M}x{C}: o_hat diff {ohd:.3e}")
    print()
    if bad:
        print("FAILED:"); [print("  -", b) for b in bad]; return 1
    print("PASS -- the dual-output GEMM reproduces quantize(o_hat GEMM) per column, and advances o_hat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
