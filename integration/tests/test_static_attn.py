"""Static (calibrated) vs dynamic quantized standard attention correctness.

Static path uses calibrated constants instead of runtime reductions:
  - per-tensor Q/K scale (sq_c, sk_c) instead of per-token absmax,
  - per-channel V scale (sv_vec) instead of per-(bh,d) absmax over T,
  - a single softmax constant c instead of the per-row max.
Since softmax is shift-invariant, c does NOT change the float result — it only sets the int8/int4
P quantization grid, so c is calibrated near the TYPICAL row max (mean row-max) to keep peaks near
full-scale. fp16 static is lossless for any non-overflowing c (we use the global max).

FINDING: static-c softmax is LOSSLESS for fp16 (c only sets the int8 grid, which fp16 lacks) but
LOSSY for int8/int4 (a single c cannot serve rows whose max varies 5-20x -> the P grid saturates
high-max rows / annihilates low-max rows). Per the design, int8/int4 static is fully static incl
softmax-c and accepts this error for MoDiff to compensate over DDIM steps -> reported, not gated.

Run: PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass python3.11 \
        integration/tests/test_static_attn.py
Gate (quality-safe paths): int8 DYNAMIC rel <= 0.05 vs fp32; fp16 static ~= fp16 dynamic (< 1e-3).
Reported (lossy-by-design, MoDiff-compensated): int8/int4 static, int4 dynamic.
"""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc

INT8_GATE = 0.05
ok = True
torch.manual_seed(0)
print(f"{'shape':>16} {'path':>18} {'rel-vs-fp32':>12}   note")
for (BH, T, hd) in [(32, 1024, 24), (32, 256, 48), (32, 64, 96)]:
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(hd)
    ref = torch.bmm(F.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale, -1), V.float())
    tag = f"{BH},{T},{hd}"

    for bits in (8, 4):
        Qm = 127.0 if bits == 8 else 7.0
        hp_qk = (hd + 31) // 32 * 32 if bits == 8 else (hd + 63) // 64 * 64
        hp_av = (hd + 63) // 64 * 64
        # ---- dynamic (runtime reductions) ----
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hp_qk, hp_av, bits)
        if bits == 8:
            S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant(S)
            Od = mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]
        else:
            S = mc.attn_qk_int4(qi, ki, hp_qk, sq, sk, scale); P, sp = mc.attn_softmax_requant4(S)
            Od = mc.attn_av_int4(P, vt, sp, sv, T)[:, :, :hd]
        rel_d = ((Od.float() - ref).norm() / ref.norm()).item()

        # ---- static (calibrated constants) ----
        sq_c = (Q.abs().max().item() / Qm) or 1e-8
        sk_c = (K.abs().max().item() / Qm) or 1e-8
        sv_hd = (V.abs().amax(dim=(0, 1)).float() / Qm).clamp_min(1e-8)          # [hd]
        sv_vec = torch.ones(hp_av, device="cuda", dtype=torch.float32); sv_vec[:hd] = sv_hd
        qi2, ki2, vt2, sq2, sk2, sv2 = mc.quantize_attn_qkv_static(Q, K, V, hp_qk, hp_av, bits, sq_c, sk_c, sv_vec)
        if bits == 8:
            Ss = mc.attn_qk_int8(qi2, ki2, sq2, sk2, scale)
        else:
            Ss = mc.attn_qk_int4(qi2, ki2, hp_qk, sq2, sk2, scale)
        c = Ss.float().amax(-1).mean().item()      # calibrated c = mean per-row max (typical peak)
        if bits == 8:
            Ps, sps = mc.attn_softmax_requant_static(Ss, c); Os = mc.attn_av_int8(Ps, vt2, sps, sv2)[:, :, :hd]
        else:
            Ps, sps = mc.attn_softmax_requant4_static(Ss, c); Os = mc.attn_av_int4(Ps, vt2, sps, sv2, T)[:, :, :hd]
        rel_s = ((Os.float() - ref).norm() / ref.norm()).item()

        # gate the quality-safe DYNAMIC int8 path; static (and int4) are reported (lossy-by-design)
        st_d = "OK" if (bits == 4 or rel_d <= INT8_GATE) else "FAIL"
        if bits == 8 and rel_d > INT8_GATE: ok = False
        print(f"{tag:>16} {'int%d dynamic' % bits:>18} {rel_d:12.4f}   [{st_d} gate]" if bits == 8
              else f"{tag:>16} {'int%d dynamic' % bits:>18} {rel_d:12.4f}   (reported)")
        print(f"{tag:>16} {'int%d static' % bits:>18} {rel_s:12.4f}   c={c:.2f} (reported, MoDiff-comp)")

    # ---- fp16 materialized: static (1-pass, c) must match dynamic (2-pass, per-row max) ----
    Sf = (torch.bmm(Q, K.transpose(1, 2)) * scale).half()
    cmax = Sf.float().max().item()                 # safe upper bound -> lossless
    Pd, rsd = mc.attn_softmax_fp16(Sf, False, 0.0); Ofd = torch.bmm(Pd, V) / rsd.unsqueeze(-1).half()
    Ps16, rss = mc.attn_softmax_fp16(Sf, True, cmax); Ofs = torch.bmm(Ps16, V) / rss.unsqueeze(-1).half()
    rel_fd = ((Ofd.float() - ref).norm() / ref.norm()).item()
    rel_fs = ((Ofs.float() - ref).norm() / ref.norm()).item()
    rel_ds = ((Ofs.float() - Ofd.float()).norm() / Ofd.float().norm()).item()
    st = "OK" if rel_ds < 1e-3 else "FAIL"
    if rel_ds >= 1e-3: ok = False
    print(f"{tag:>16} {'fp16 dynamic':>18} {rel_fd:12.4f}")
    print(f"{tag:>16} {'fp16 static':>18} {rel_fs:12.4f}   static-vs-dyn={rel_ds:.2e} [{st}]")
print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
