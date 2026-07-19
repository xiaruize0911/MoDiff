"""nsys profiling driver: runs the C192 qkv GEMM (the biggest-win shape, M=32768,K=192,N=576)
through each backend under a cudaProfilerApi capture range so nsys attributes kernel time cleanly.
Backend chosen by argv[1]: ours8awq | awqref | ours4awq. Run under nsys profile
--capture-range=cudaProfilerApi. (The retired ours8/gemm_w8a8 backend was removed 2026-07-18.)"""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/llm-awq/awq/kernels")
import torch, torch.nn.functional as F, modiff_cutlass as mc
import torch.cuda.profiler as prof
try: import awq_inference_engine as _awq
except Exception: _awq = None

backend = sys.argv[1] if len(sys.argv) > 1 else "ours8awq"
M, K, N = 32768, 192, 576
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.float16)
asc = x.abs().max().item() / 127.0
ws = torch.randn(N, device="cuda").abs().float() / 127
xq = mc.quantize_act_int8(x, asc)
Wq = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
Np = ((N + 127) // 128) * 128
Wqp = F.pad(Wq, (0, 0, 0, Np - N)); wsp = F.pad(ws, (0, Np - N), value=1.0)


def pack4(q):
    q = q.to(torch.int8); lo = q[..., 0::2] & 0xF; hi = q[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()


if backend == "awqref":
    wsh = wsp.half(); ascv = torch.full((M,), asc, device="cuda", dtype=torch.float16)
    outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
    fn = lambda: _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp)
elif backend == "ours4awq":
    Kp = 256
    xq4 = pack4(torch.randint(-7, 7, (M, Kp), device="cuda", dtype=torch.int8))
    Wq4 = pack4(torch.randint(-7, 7, (Np, Kp), device="cuda", dtype=torch.int8))
    ws4 = F.pad(torch.randn(N, device="cuda").abs().float() / 7, (0, Np - N), value=1.0)
    fn = lambda: mc.gemm_w4a4_awq(xq4, Wq4, ws4, asc / 18, Kp)
else:  # ours8awq
    fn = lambda: mc.gemm_w8a8_awq(xq, Wqp, wsp, asc)

for _ in range(50): fn()
torch.cuda.synchronize()
prof.start()
for _ in range(200): fn()
torch.cuda.synchronize()
prof.stop()
print(f"done backend={backend}")
