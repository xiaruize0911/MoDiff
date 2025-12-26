
import torch
import time
import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.fused_modiff import fused_modiff_gemm

def benchmark():
    device = torch.device("cuda")
    M, N, K = 4096, 512, 512
    
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    x_prev = torch.randn(M, K, device=device, dtype=torch.float32)
    w_int8 = torch.randint(-127, 127, (K, N), device=device, dtype=torch.int8)
    o_prev = torch.randn(M, N, device=device, dtype=torch.float32)
    scale_w = torch.tensor(0.01, device=device)
    
    # Warmup
    for _ in range(10):
        out = fused_modiff_gemm(x, x_prev, w_int8, o_prev, scale_w)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = fused_modiff_gemm(x, x_prev, w_int8, o_prev, scale_w)
    torch.cuda.synchronize()
    
    fused_time = (time.time() - start)/100 * 1000
    print(f"Fused Kernel Time: {fused_time:.3f} ms")
    
    # Compare with FP32
    linear = torch.nn.Linear(K, N).to(device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = linear(x)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start)/100 * 1000
    print(f"FP32 Time: {fp32_time:.3f} ms")
    print(f"Speedup vs FP32: {fp32_time/fused_time:.2f}x")

if __name__ == "__main__":
    benchmark()
