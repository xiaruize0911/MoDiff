"""ncu harness: launch each key kernel exactly once (dominant T=1024 attention shape) so Nsight
Compute can measure real per-kernel DRAM bytes / throughput / bound. Run under ncu (see
ncu_profile.py). Not a benchmark -- ncu replays each kernel internally for accurate counters."""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, modiff_cutlass as mc

BH, T, hd = 256, 1024, 24
scale = 1.0 / math.sqrt(hd)
Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
hpq8, hpq4, hpa = 32, 64, 64
Qm8, Qm4 = 127.0, 7.0
sq_c = Q.abs().max().item() / Qm8; sk_c = K.abs().max().item() / Qm8
sv8 = torch.ones(hpa, device="cuda"); sv8[:hd] = (V.abs().amax(dim=(0, 1)).float() / Qm8).clamp_min(1e-8)
sv4 = torch.ones(hpa, device="cuda"); sv4[:hd] = (V.abs().amax(dim=(0, 1)).float() / Qm4).clamp_min(1e-8)

# --- Q/K/V quantize: dynamic (with absmax reductions) vs static (calibrated) ---
qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hpq8, hpa, 8)                       # DYNAMIC int8 quant
qi2, ki2, vt2, sq2, sk2, sv2 = mc.quantize_attn_qkv_static(Q, K, V, hpq8, hpa, 8, sq_c, sk_c, sv8)  # STATIC int8 quant
# --- QKᵀ (int8) ---
S = mc.attn_qk_int8(qi, ki, sq, sk, scale)
# --- softmax int8: dynamic (2-pass) vs static (1-pass) ---
c = S.float().amax(-1).mean().item()
P, sp = mc.attn_softmax_requant(S)             # DYNAMIC int8 softmax
Ps, sps = mc.attn_softmax_requant_static(S, c) # STATIC int8 softmax
# --- AV (int8) ---
O = mc.attn_av_int8(P, vt, sp, sv)
# --- softmax int4: dynamic vs static ---
qi4, ki4, vt4, sq4, sk4, sv4b = mc.quantize_attn_qkv(Q, K, V, hpq4, hpa, 4)
S4 = mc.attn_qk_int4(qi4, ki4, hpq4, sq4, sk4, scale)
P4, sp4 = mc.attn_softmax_requant4(S4)              # DYNAMIC int4 softmax
P4s, sp4s = mc.attn_softmax_requant4_static(S4, c)  # STATIC int4 softmax
# --- softmax fp16: dynamic (2-pass) vs static (1-pass) ---
Pf, rf = mc.attn_softmax_fp16(S, False, 0.0)   # DYNAMIC fp16 softmax
Pfs, rfs = mc.attn_softmax_fp16(S, True, c)    # STATIC fp16 softmax
torch.cuda.synchronize()
print("harness done")
