"""Benchmark the PROPERLY-FUSED quantized attention block (no unfused path) vs fp16,
full block front-to-back, on the shapes the model actually quantizes (T>=256, T%64==0).

FUSED int8 block:
  fused_gn_qkv_int8 (GN+qkv -> int8) -> quantize_attn_qkv_from_i8 -> attn_qk_int8
  -> attn_softmax_requant_static -> attn_av_int8 -> proj (fp16) -> residual
fp16 block:
  group_norm_silu -> qkv (fp16) -> SDPA MATH -> proj (fp16) -> residual

T<256 / hd=96 blocks are not quant-eligible -> fp16 in BOTH (the model falls back), so they
contribute 1.00x. Weighted over all 21 blocks. Writes data/fused_attn_block_b<B>.csv
"""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.fused_ops.fused_resblock import _group_norm_silu

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"; NG = 32; SHIFT = 16.0
# (C, nh, hd, T, H, count, quant_eligible)
SHAPES = [(192, 8, 24, 1024, 32, 5, True), (384, 8, 48, 256, 16, 5, True),
          (384, 8, 48, 64, 8, 5, False), (768, 8, 96, 16, 4, 5, False), (768, 8, 96, 4, 2, 1, False)]

def bench(fn, it=100, warm=30, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]

_burn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _burn = _burn @ _burn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []; tot_fp16 = 0.0; tot_q = 0.0
print(f"Properly-fused quantized attention block vs fp16 (full block) @ b{B}")
print(f"{'C/hd/T':>12} {'cnt':>3} {'elig':>5} | {'fp16 blk':>9} {'int8 blk':>9} | {'speedup':>8}")
for (C, nh, hd, T, Hs, cnt, elig) in SHAPES:
    N = B; H = nh; c = C; scale = 1.0 / math.sqrt(hd)
    hp_qk = (hd + 31) // 32 * 32; hp_av = (hd + 63) // 64 * 64
    x = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    x_tok = x.permute(0, 2, 3, 1).reshape(N, T, C)
    gw = torch.randn(C, device=dev, dtype=torch.float16); gb = torch.randn(C, device=dev, dtype=torch.float16)
    qkv_lin = nn.Linear(C, 3 * C).to(dev).half(); proj = nn.Linear(C, C).to(dev).half()

    def fp16_block():
        xn = _group_norm_silu(x, NG, gw, gb, 1e-5, False)
        qkv = qkv_lin(xn.permute(0, 2, 3, 1).reshape(N, T, C)).view(N, T, nh, 3, hd)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(3))
        with sdpa_kernel(SDPBackend.MATH):
            a = F.scaled_dot_product_attention(q, k, v, scale=scale)
        a = a.transpose(1, 2).reshape(N, T, C)
        return x_tok + proj(a)
    t_fp16 = bench(fp16_block)

    if elig:
        qkv_wi8 = (torch.randn(3 * C, 1, 1, C, device=dev) * 0.02).half().contiguous()
        epi_i8 = torch.randint(-127, 127, (3 * C,), device=dev, dtype=torch.int8)
        oscale = (torch.rand(3 * C, device=dev) + 0.5).float().contiguous()
        # static softmax constant c: calibrate once from a real S
        qkv_i8_0 = mc.fused_gn_qkv_int8(x, qkv_wi8, epi_i8, NG, 1e-5, SHIFT)
        flat0 = qkv_i8_0.permute(0, 2, 3, 1).reshape(N * T, 3 * C).contiguous()
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_from_i8(flat0, oscale, nh, T, hp_qk, hp_av)
        S0 = mc.attn_qk_int8(qi, ki, sq, sk, scale)
        c_static = float(S0.float().amax(-1).mean().item())

        def int8_block():
            qkv_i8 = mc.fused_gn_qkv_int8(x, qkv_wi8, epi_i8, NG, 1e-5, SHIFT)
            flat = qkv_i8.permute(0, 2, 3, 1).reshape(N * T, 3 * C).contiguous()
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_from_i8(flat, oscale, nh, T, hp_qk, hp_av)
            S = mc.attn_qk_int8(qi, ki, sq, sk, scale)
            P, sp = mc.attn_softmax_requant_static(S, c_static)
            O = mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]
            a = O.reshape(N, nh, T, hd).transpose(1, 2).reshape(N, T, C)
            return x_tok + proj(a)
        int8_block()  # smoke
        t_q = bench(int8_block)
    else:
        t_q = t_fp16  # model falls back to fp16 -> identical

    sp = t_fp16 / t_q
    print(f"{f'{C}/{hd}/{T}':>12} {cnt:3d} {str(elig):>5} | {t_fp16:9.1f} {t_q:9.1f} | {sp:7.2f}x")
    rows.append(dict(C=C, hd=hd, T=T, count=cnt, quant_eligible=int(elig),
                     fp16_block_us=round(t_fp16, 1), int8_block_us=round(t_q, 1), speedup=round(sp, 3)))
    tot_fp16 += cnt * t_fp16; tot_q += cnt * t_q

print(f"\n=== weighted over all 21 blocks / forward @ b{B} ===")
print(f"fp16 attention blocks           {tot_fp16:9.1f} us   1.00x")
print(f"fused-int8 (elig) + fp16 (rest) {tot_q:9.1f} us   {tot_fp16/tot_q:.3f}x")
# eligible-only view
efp = sum(r['count'] * r['fp16_block_us'] for r in rows if r['quant_eligible'])
eq = sum(r['count'] * r['int8_block_us'] for r in rows if r['quant_eligible'])
print(f"  [eligible blocks only (T>=256): fp16 {efp:.0f}us -> fused-int8 {eq:.0f}us = {efp/eq:.3f}x]")
with open(f"docs/flash_attention_2026-07-19/data/fused_attn_block_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE data/fused_attn_block_b{B}.csv")
