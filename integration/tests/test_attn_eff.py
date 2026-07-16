"""Effective-path quantized standard attention: fused quantize + fp16 pre-scaled scores +
fused softmax + int8/int4 AV, validated against an fp32 reference.

This supersedes the stage-wise test_qk/test_sm/test_av/test_av4 scripts (which targeted the
earlier fp32-raw-score signatures). Run:
    PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass python3.11 \
        integration/tests/test_attn_eff.py
Gate: int8 rel <= 0.05 (quality-safe); int4 rel reported (lossy by design, MoDiff-compensated).
"""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc

INT8_GATE = 0.05
ok = True
torch.manual_seed(0)
# real churches attention shapes (BH = N*nh at batch 32) with T % 64 == 0
for (BH, T, hd) in [(32, 1024, 24), (32, 256, 48), (32, 64, 96)]:
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(hd)
    ref = torch.bmm(F.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale, -1), V.float())
    for bits in (8, 4):
        hp_qk = (hd + 31) // 32 * 32 if bits == 8 else (hd + 63) // 64 * 64
        hp_av = (hd + 63) // 64 * 64
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hp_qk, hp_av, bits)
        if bits == 8:
            S = mc.attn_qk_int8(qi, ki, sq, sk, scale)
            P, sp = mc.attn_softmax_requant(S)
            O = mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]
        else:
            S = mc.attn_qk_int4(qi, ki, hp_qk, sq, sk, scale)
            P, sp = mc.attn_softmax_requant4(S)
            O = mc.attn_av_int4(P, vt, sp, sv, T)[:, :, :hd]
        assert S.dtype == torch.float16, f"scores must be fp16, got {S.dtype}"
        rel = ((O.float() - ref).norm() / ref.norm()).item()
        gated = " (gate)" if bits == 8 else " (reported)"
        status = "OK" if (bits == 4 or rel <= INT8_GATE) else "FAIL"
        if bits == 8 and rel > INT8_GATE:
            ok = False
        print(f"BH{BH:>3} T{T:>4} hd{hd:>2} int{bits}: rel-vs-fp32 = {rel:.4f}{gated}  [{status}]")
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
