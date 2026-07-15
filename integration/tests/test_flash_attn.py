"""Correctness gates for the fused int8 flash-attention score path.
  1. flash_attn_int8 kernel vs fp32 SDPA reference (all churches shapes)
  2. QuantizedTokenMajorAttentionBlock vs fp16 TokenMajorAttentionBlock

Run: PYTHONPATH=src/taming-transformers python integration/tests/test_flash_attn.py
"""
import os, sys, math
import torch
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import modiff_cutlass as mc

torch.manual_seed(0)
dev = "cuda"
SHAPES = [(192, 1024, 24), (384, 256, 48), (384, 64, 48), (768, 16, 96), (768, 4, 96)]
GATE_FP32 = 0.05   # int8 quant of Q/K/V/P vs true fp32 softmax(QKᵀ)·V


def relerr(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def test_kernel():
    print("[1] flash_attn_int8 kernel vs fp32 SDPA")
    allpass = True
    for (C, T, hd) in SHAPES:
        N, H = 2, 8
        q = torch.randn(N, H, T, hd, device=dev); k = torch.randn(N, H, T, hd, device=dev)
        v = torch.randn(N, H, T, hd, device=dev); scale = 1.0 / math.sqrt(hd)
        ref = torch.einsum("nhij,nhjd->nhid",
                           torch.softmax(torch.einsum("nhid,nhjd->nhij", q, k) * scale, -1), v)
        hd_pad = (hd + 31) // 32 * 32

        def qtok(x):
            sc = x.abs().amax(-1).clamp_min(1e-8) / 127.0
            xi = torch.round(x / sc.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            if hd_pad > hd: xi = torch.nn.functional.pad(xi, (0, hd_pad - hd))
            return xi.contiguous(), sc.float().contiguous()
        qi, sq = qtok(q); ki, sk = qtok(k)
        scv = (v.abs().amax(2).clamp_min(1e-8) / 127.0)
        vi = torch.round(v / scv.unsqueeze(2)).clamp(-127, 127).to(torch.int8)
        if hd_pad > hd: vi = torch.nn.functional.pad(vi, (0, hd_pad - hd))
        out = mc.flash_attn_int8(qi, ki, vi.contiguous(), sq, sk, scv.float().contiguous(), scale)
        e = relerr(out, ref); ok = e < GATE_FP32; allpass &= ok
        print(f"    C{C}/T{T}/hd{hd:<3} rel_vs_fp32={e:.4f}  {'PASS' if ok else 'FAIL'}")
    return allpass


def test_block():
    print("[2] QuantizedTokenMajorAttentionBlock vs fp16 TokenMajorAttentionBlock")
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
    from integration.fused_ops.quantized_attention import QuantizedTokenMajorAttentionBlock
    allpass = True
    for (C, S) in [(192, 32), (384, 16), (384, 8), (768, 4), (768, 2)]:
        T = S * S
        ab = AttentionBlock(C, num_heads=8, use_new_attention_order=False).to(dev).half().eval()
        with torch.no_grad():   # proj_out is zero-init -> randomize so attention isn't a no-op
            torch.nn.init.normal_(ab.proj_out.weight, std=0.1); torch.nn.init.normal_(ab.proj_out.bias, std=0.1)
            torch.nn.init.normal_(ab.qkv.weight, std=0.1); torch.nn.init.normal_(ab.qkv.bias, std=0.1)
        ref_b = TokenMajorAttentionBlock(ab).to(dev).eval()
        q_b = QuantizedTokenMajorAttentionBlock(ab, score_bits=8, proj_bits=16).to(dev).eval()
        x = torch.randn(2, C, S, S, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            e = relerr(q_b(x), ref_b(x))
        path = "int8" if (q_b.score_bits == 8 and T >= 64) else "fp16"
        ok = e < (0.03 if path == "int8" else 1e-3); allpass &= ok
        print(f"    C{C}/T{T:<4} ({path}) rel_vs_fp16={e:.4f}  {'PASS' if ok else 'FAIL'}")
    return allpass


if __name__ == "__main__":
    p1 = test_kernel(); p2 = test_block()
    print("\nALL PASS" if (p1 and p2) else "\nSOME FAILED")
