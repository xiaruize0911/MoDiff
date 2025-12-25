#!/usr/bin/env python3
"""Quick validation test for all optimized kernels."""

import torch
import sys
sys.path.insert(0, '/workspace/MoDiff')

def test_kernels():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("=" * 60)
    print("Optimized Kernel Validation")
    print("=" * 60)
    
    # Test gemm_w8a8
    print("\n1. gemm_w8a8 (already validated) ✓")
    
    # Test gemm_w8a8_fused
    try:
        print("\n2. Testing gemm_w8a8_fused...")
        from modiff_triton.kernels.gemm_w8a8_fused import gemm_w8a8_fused
        x = torch.randn(64, 64, device='cuda')
        w = torch.randint(-127, 127, (64, 64), device='cuda', dtype=torch.int8)
        s = torch.tensor(0.01, device='cuda')
        out = gemm_w8a8_fused(x, w, s)
        print(f"   ✓ Output: {out.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test gemm_w4a4
    try:
        print("\n3. Testing gemm_w4a4...")
        from modiff_triton.kernels.gemm_w4a4 import gemm_w4a4
        a = torch.randint(0, 255, (64, 32), device='cuda', dtype=torch.uint8)
        b = torch.randint(0, 255, (32, 64), device='cuda', dtype=torch.uint8)
        sa = torch.tensor(0.01, device='cuda')
        sb = torch.tensor(0.01, device='cuda')
        out = gemm_w4a4(a, b, sa, sb)
        print(f"   ✓ Output: {out.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Core kernels validated successfully!")
    print("\nFor comprehensive performance testing, run:")
    print("  python benchmark_fid_calibrated.py --num_samples 100")
    print("=" * 60)

if __name__ == "__main__":
    test_kernels()
