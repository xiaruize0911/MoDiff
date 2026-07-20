"""Per-kernel time breakdown of the FUSED int8 attention chain, at the two quant-eligible
shapes (hd24/T1024, hd48/T256). Each kernel timed in isolation with pre-built inputs.
Classifies each as COMPUTE (matmul) vs DATA-MOVE (quantize / softmax over the T*T scores),
and shows the HBM traffic on the [BH,T,T] score matrix (the 3 round-trips).
"""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc

dev = "cuda"; torch.manual_seed(0); NG = 32; SHIFT = 16.0
def bench(fn, it=100, warm=30, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]
_b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _b = _b @ _b * 1e-4 + 1.0
torch.cuda.synchronize()

N, H = 128, 8
for (C, hd, T, Hs) in [(192, 24, 1024, 32), (384, 48, 256, 16)]:
    BH = N * H; scale = 1.0 / math.sqrt(hd)
    hp_qk = (hd + 31) // 32 * 32; hp_av = (hd + 63) // 64 * 64
    x = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    qkv_wi8 = (torch.randn(3 * C, 1, 1, C, device=dev) * 0.02).half().contiguous()
    epi_i8 = torch.randint(-127, 127, (3 * C,), device=dev, dtype=torch.int8)
    oscale = (torch.rand(3 * C, device=dev) + 0.5).float().contiguous()
    proj = nn.Linear(C, C).to(dev).half()

    # build inputs stage by stage
    def k1():
        qkv_i8 = mc.fused_gn_qkv_int8(x, qkv_wi8, epi_i8, NG, 1e-5, SHIFT)
        return qkv_i8.permute(0, 2, 3, 1).reshape(N * T, 3 * C).contiguous()
    flat = k1()
    def k2(): return mc.quantize_attn_qkv_from_i8(flat, oscale, H, T, hp_qk, hp_av)
    qi, ki, vt, sq, sk, sv = k2()
    S = mc.attn_qk_int8(qi, ki, sq, sk, scale)
    c_static = float(S.float().amax(-1).mean().item())
    P, sp = mc.attn_softmax_requant_static(S, c_static)
    O = mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]
    a = O.reshape(N, H, T, hd).transpose(1, 2).reshape(N, T, C).contiguous()
    a_scale = a.abs().max().item() / 127.0

    t1 = bench(k1)
    t2 = bench(k2)
    t3 = bench(lambda: mc.attn_qk_int8(qi, ki, sq, sk, scale))
    t4 = bench(lambda: mc.attn_softmax_requant_static(S, c_static))
    t5 = bench(lambda: mc.attn_av_int8(P, vt, sp, sv))

    chain = [
        ("[1] fused_gn_qkv_int8", t1, "GN+qkv GEMM+int8out", "compute+write"),
        ("[2] quantize_attn_qkv_from_i8", t2, "int8 qkv -> attn operands", "DATA-MOVE"),
        ("[3] attn_qk_int8  (QKᵀ)", t3, "int8 matmul -> writes S[BH,T,T]", "compute+WRITE T²"),
        ("[4] attn_softmax_requant", t4, "read S, softmax, write int8 P", "DATA-MOVE T² (2 trips)"),
        ("[5] attn_av_int8  (AV)", t5, "read P[BH,T,T], int8 matmul", "compute+READ T²"),
    ]
    tot = sum(c[1] for c in chain)
    TT = BH * T * T
    print(f"\n=========== hd{hd}/T{T}  (BH={BH}, scores [BH,T,T]={BH}x{T}x{T} = {TT/1e9:.2f}G elems) ===========")
    print(f"{'kernel':32} {'us':>8} {'%':>6}  {'what it does':30} {'type':22}")
    for nm, t, what, typ in chain:
        print(f"{nm:32} {t:8.1f} {100*t/tot:5.1f}%  {what:30} {typ:22}")
    print(f"{'TOTAL int8 attn chain':32} {tot:8.1f}")
    # HBM traffic on the score matrix
    wr_qk = TT * 2 / 1e9; rw_sm = TT * 2 / 1e9 + TT * 1 / 1e9; rd_av = TT * 1 / 1e9
    print(f"  score-matrix HBM traffic: QKᵀ write {wr_qk:.2f}GB + softmax read+write {rw_sm:.2f}GB + AV read {rd_av:.2f}GB "
          f"= {wr_qk+rw_sm+rd_av:.2f}GB moved over [BH,T,T] ({t3+t4+t5:.0f}us in [3][4][5])")
