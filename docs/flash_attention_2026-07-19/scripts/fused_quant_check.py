"""Is the qkv->attention int8 fusion done properly? Compare, on the shapes where the
model actually runs quantized attention (T>=256, T%64==0 => level0 T=1024, level1 T=256):

  NON-FUSED (what my fair bench used):
    group_norm_silu (fp16) + qkv Linear (fp16) + quantize_attn_qkv (fp16->int8 attn operands)
  FUSED (MODIFF_FUSE_QKV_I8 path):
    fused_gn_qkv_int8 (GN+qkv, int8 output) + quantize_attn_qkv_from_i8 (int8 qkv -> attn operands)
  fp16 reference block front-end:
    group_norm_silu (fp16) + qkv Linear (fp16)   [feeds fp16 SDPA]

Reports the front-end (norm+qkv+quantize) cost of each, so we can see how much the fusion
saves vs the standalone quantize. Writes data/fused_quant_check_b<B>.csv
"""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from integration.fused_ops.fused_resblock import _group_norm_silu

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"
SHAPES = [(192, 8, 24, 1024, 32, 5), (384, 8, 48, 256, 16, 5)]  # only T>=256 quant-eligible
NG = 32; SHIFT = 16.0

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

rows = []
print(f"qkv->attn int8 fusion check @ b{B}  (front-end = norm + qkv + quantize)")
print(f"{'C/T':>10} | {'fp16 front':>10} | {'nonfused q':>10} {'nonfused FE':>11} | {'fused q':>9} {'fused FE':>9} | {'fused/nonfused FE':>17}")
for (C, nh, hd, T, Hs, cnt) in SHAPES:
    N = B; H = nh; c = C; hp_qk = (hd + 31) // 32 * 32; hp_av = (hd + 63) // 64 * 64
    x = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    gw = torch.randn(C, device=dev, dtype=torch.float16); gb = torch.randn(C, device=dev, dtype=torch.float16)
    qkv_lin = nn.Linear(C, 3 * C).to(dev).half()

    # ---- fp16 front-end: GN + qkv ----
    def fp16_front():
        xn = _group_norm_silu(x, NG, gw, gb, 1e-5, False)
        return qkv_lin(xn.permute(0, 2, 3, 1).reshape(N, T, C))
    t_fp16_fe = bench(fp16_front)

    # ---- NON-FUSED quant front-end: GN + qkv + quantize_attn_qkv ----
    def nonfused_full():
        xn = _group_norm_silu(x, NG, gw, gb, 1e-5, False)
        qkv = qkv_lin(xn.permute(0, 2, 3, 1).reshape(N, T, C)).view(N, T, nh, 3, hd)
        q, k, v = qkv.unbind(3)
        q = q.transpose(1, 2).reshape(N * H, T, hd).contiguous()
        k = k.transpose(1, 2).reshape(N * H, T, hd).contiguous()
        v = v.transpose(1, 2).reshape(N * H, T, hd).contiguous()
        return mc.quantize_attn_qkv(q, k, v, hp_qk, hp_av, 8)
    t_nf_full = bench(nonfused_full)
    # isolate the quantize alone
    xn0 = _group_norm_silu(x, NG, gw, gb, 1e-5, False)
    qkv0 = qkv_lin(xn0.permute(0, 2, 3, 1).reshape(N, T, C)).view(N, T, nh, 3, hd)
    q0, k0, v0 = (t.transpose(1, 2).reshape(N * H, T, hd).contiguous() for t in qkv0.unbind(3))
    t_nf_q = bench(lambda: mc.quantize_attn_qkv(q0, k0, v0, hp_qk, hp_av, 8))

    # ---- FUSED path: fused_gn_qkv_int8 + quantize_attn_qkv_from_i8 ----
    # synthetic frozen weights: fused conv weight [3C,1,1,C] fp16, epilogue bias [3C] int8, oscale [3C] f32
    qkv_wi8 = (torch.randn(3 * C, 1, 1, C, device=dev) * 0.02).half().contiguous()
    epi_i8 = torch.randint(-127, 127, (3 * C,), device=dev, dtype=torch.int8)
    oscale = (torch.rand(3 * C, device=dev) + 0.5).float().contiguous()
    try:
        def fused_full():
            qkv_i8 = mc.fused_gn_qkv_int8(x, qkv_wi8, epi_i8, NG, 1e-5, SHIFT)  # [N,3C,H,W] i8 CL
            flat = qkv_i8.permute(0, 2, 3, 1).reshape(N * T, 3 * C).contiguous()
            return mc.quantize_attn_qkv_from_i8(flat, oscale, nh, T, hp_qk, hp_av)
        out = fused_full()  # smoke
        t_f_full = bench(fused_full)
        # isolate the from_i8 quantize
        qkv_i8_0 = mc.fused_gn_qkv_int8(x, qkv_wi8, epi_i8, NG, 1e-5, SHIFT)
        flat0 = qkv_i8_0.permute(0, 2, 3, 1).reshape(N * T, 3 * C).contiguous()
        t_f_q = bench(lambda: mc.quantize_attn_qkv_from_i8(flat0, oscale, nh, T, hp_qk, hp_av))
        ok = True
    except Exception as ex:
        t_f_full = t_f_q = float("nan"); ok = False
        print(f"  FUSED path error: {type(ex).__name__}: {ex}")
    print(f"{f'{C}/{T}':>10} | {t_fp16_fe:10.1f} | {t_nf_q:10.1f} {t_nf_full:11.1f} | {t_f_q:9.1f} {t_f_full:9.1f} | "
          f"{(t_f_full/t_nf_full if ok else float('nan')):16.2f}x")
    rows.append(dict(C=C, T=T, count=cnt, fp16_front_us=round(t_fp16_fe, 1),
                     nonfused_quant_us=round(t_nf_q, 1), nonfused_front_us=round(t_nf_full, 1),
                     fused_quant_us=round(t_f_q, 1) if ok else "", fused_front_us=round(t_f_full, 1) if ok else "",
                     fused_vs_nonfused=round(t_f_full / t_nf_full, 3) if ok else ""))

with open(f"docs/flash_attention_2026-07-19/data/fused_quant_check_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nNote: 'front-end' = norm+qkv(+quantize). fp16 front-end has NO quantize.")
print("Fusion is 'proper' if fused_front ~ fp16_front (quantize nearly hidden) and << nonfused_front.")
print(f"WROTE data/fused_quant_check_b{B}.csv")
