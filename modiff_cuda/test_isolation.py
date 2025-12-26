import torch
import modiff_cuda

print("="*60)
print("Debugging Conv2d Kernel - Isolation Tests")
print("="*60)

M, N = 1024, 128  # Same dimensions as Conv test

print("\n[Test 1] Global memory write (no shared memory, no tensor cores)")
print("-" * 60)
try:
    out = modiff_cuda.test_global_write(M, N)
    torch.cuda.synchronize()
    print(f"✓ SUCCESS - Output shape: {out.shape}, mean: {out.mean().item():.2f}")
    assert out.mean().item() == 42.0, "Values should all be 42.0"
    print("✓ Values are correct (all 42.0)")
except RuntimeError as e:
    print(f"✗ FAILED: {e}")
    exit(1)

print("\n[Test 2] Shared memory usage (34KB allocation)")
print("-" * 60)
try:
    out = modiff_cuda.test_shared_memory(M, N)
    torch.cuda.synchronize()
    print(f"✓ SUCCESS - Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.2f}, {out.max().item():.2f}]")
except RuntimeError as e:
    print(f"✗ FAILED: {e}")
    print("  Issue is in shared memory allocation or access pattern")
    exit(1)

print("\n[Test 3] Conv kernel's exact shared memory pattern")
print("-" * 60)
try:
    out = modiff_cuda.test_conv_smem_pattern(M, N)
    torch.cuda.synchronize()
    print(f"✓ SUCCESS - Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.2f}, {out.max().item():.2f}]")
except RuntimeError as e:
    print(f"✗ FAILED: {e}")
    print("  Issue is in Conv's specific shared memory declaration/pattern")
    exit(1)

print("\n" + "="*60)
print("All isolation tests PASSED!")
print("The infrastructure is working - issue must be in Conv kernel logic")
print("="*60)
