"""Minimal harness for nsys: run each GEMM backend (fp16 / ours w8a8 / AWQ w8a8 / ours w4a4) a few
times on C192 qkv and C768 qkv so nsys can name the actual kernels each backend dispatches."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
import awq_inference_engine as _awq
for (M, K, N) in [(32768, 192, 576), (2048, 768, 2304)]:
    W = torch.randn(N, K, device="cuda", dtype=torch.float16); x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    asc = x.abs().max().item() / 127.0; ws = torch.randn(N, device="cuda").abs().float() / 127
    xq = mc.quantize_act_int8(x, asc); Wq = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    Np = ((N + 127) // 128) * 128; Wqp = F.pad(Wq, (0, 0, 0, Np - N)); wsh = F.pad(ws, (0, Np - N), value=1).half()
    ascv = torch.full((M,), asc, device="cuda", dtype=torch.float16); outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
    xq4 = mc.quantize_act_int4_pack(x, asc / 18); Wq4 = torch.randint(-7, 7, (N, K // 2), device="cuda", dtype=torch.int8)
    for _ in range(20):
        F.linear(x, W); mc.gemm_w8a8(xq, Wq, ws, asc)
        _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp); mc.gemm_w4a4(xq4, Wq4, ws, asc, K)
torch.cuda.synchronize(); print("harness done")
