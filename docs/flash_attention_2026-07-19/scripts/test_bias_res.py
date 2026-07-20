"""Verify gemm_w8a8/w4a4_awq_bias_res == plain GEMM + separate bias/residual adds (fp16-rounding level).
Fused adds bias+residual in fp32 before the half cast (1 rounding) vs reference's 3 fp16 roundings, so
rel-L2 ~1e-3 is expected/better, not bit-identical."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

def rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)

def check8(M, K, N, use_bias, use_res):
    torch.manual_seed(0)
    A = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    ws = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001
    asc = 0.02
    bias = torch.randn(N, device="cuda", dtype=torch.float16) if use_bias else torch.empty(0, device="cuda", dtype=torch.float16)
    res = torch.randn(M, N, device="cuda", dtype=torch.float16) if use_res else torch.empty(0, device="cuda", dtype=torch.float16)
    base = mc.gemm_w8a8_awq_nout(A, B, ws, asc, N).float()
    ref = base + (bias.float() if use_bias else 0) + (res.float() if use_res else 0)
    fused = mc.gemm_w8a8_awq_bias_res(A, B, ws, asc, N, bias, res).float()
    print(f"  w8a8 M{M} K{K} N{N} bias={use_bias} res={use_res}  rel-L2={rel(fused, ref):.2e}")

def check4(M, K, N, use_bias, use_res):
    torch.manual_seed(0)
    A = torch.randint(-127, 128, (M, K // 2), device="cuda", dtype=torch.int8)
    B = torch.randint(-127, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    ws = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001
    asc = 0.02
    bias = torch.randn(N, device="cuda", dtype=torch.float16) if use_bias else torch.empty(0, device="cuda", dtype=torch.float16)
    res = torch.randn(M, N, device="cuda", dtype=torch.float16) if use_res else torch.empty(0, device="cuda", dtype=torch.float16)
    base = mc.gemm_w4a4_awq_nout(A, B, ws, asc, K, N).float()
    ref = base + (bias.float() if use_bias else 0) + (res.float() if use_res else 0)
    fused = mc.gemm_w4a4_awq_bias_res(A, B, ws, asc, K, N, bias, res).float()
    print(f"  w4a4 M{M} K{K} N{N} bias={use_bias} res={use_res}  rel-L2={rel(fused, ref):.2e}")

print("== w8a8 ==")
for ub, ur in [(True, False), (False, True), (True, True)]:
    check8(1024, 256, 384, ub, ur); check8(512, 512, 512, ub, ur)
print("== w4a4 ==")
for ub, ur in [(True, False), (False, True), (True, True)]:
    check4(1024, 256, 384, ub, ur); check4(512, 512, 512, ub, ur)
print("done")
