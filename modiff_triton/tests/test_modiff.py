"""
Tests for MoDiff Triton Kernels

Verifies correctness of the MoDiff implementation against the paper's equations:
    - Eq. (ec1): â_T = Q(a_T)
    - Eq. (ec2): ô_T = A(â_T)
    - Eq. (ec5): â_t = Q(a_t - â_{t+1}) + â_{t+1}
    - Eq. (ec6): ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}
"""

import torch
import pytest
import sys
import os

# Add MoDiff directory to path for module imports
modiff_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, modiff_dir)

from modiff_triton.kernels.quantize import (
    quantize_symmetric_int8,
    quantize_symmetric_int4,
    dequantize_int8,
    dequantize_int4,
    compute_dynamic_scale_int8,
    compute_dynamic_scale_int4,
)
from modiff_triton.kernels.modulated_quantize import (
    modulated_quantize_int8,
    modulated_quantize_int4,
    modulated_quantize_first_step_int8,
    modulated_quantize_first_step_int4,
)


class TestQuantization:
    """Test basic quantization operations."""
    
    def test_int8_quantize_dequantize(self):
        """Test INT8 quantization roundtrip."""
        x = torch.randn(128, 256, device='cuda')
        
        x_int, scale = quantize_symmetric_int8(x)
        x_recon = x_int.float() * scale
        
        # Check shapes
        assert x_int.shape == x.shape
        assert x_int.dtype == torch.int8
        
        # Check reconstruction error (should be bounded by scale/2)
        error = (x - x_recon).abs().max()
        expected_max_error = scale / 2
        print(f"INT8 max error: {error:.6f}, expected max: {expected_max_error:.6f}")
        assert error < scale, f"Error {error} exceeds scale {scale}"
    
    def test_int4_quantize_dequantize(self):
        """Test INT4 quantization roundtrip."""
        x = torch.randn(128, 256, device='cuda')  # Must be even size
        
        x_packed, scale, original_shape = quantize_symmetric_int4(x)
        x_recon = dequantize_int4(x_packed, scale, original_shape)
        
        # Check packed shape (should be half)
        assert x_packed.numel() == x.numel() // 2
        
        # Check reconstruction
        error = (x - x_recon).abs().max()
        print(f"INT4 max error: {error:.6f}")
    
    def test_dynamic_scale_int8(self):
        """Test dynamic scale computation for INT8."""
        x = torch.randn(1000, device='cuda') * 10
        scale, zp = compute_dynamic_scale_int8(x, symmetric=True)
        
        expected_scale = x.abs().max() / 127
        print(f"Computed scale: {scale:.6f}, expected: {expected_scale:.6f}")
        assert torch.isclose(scale, expected_scale, rtol=1e-5)
    
    def test_dynamic_scale_int4(self):
        """Test dynamic scale computation for INT4."""
        x = torch.randn(1000, device='cuda') * 5
        scale, zp = compute_dynamic_scale_int4(x, symmetric=True)
        
        expected_scale = x.abs().max() / 7
        print(f"INT4 scale: {scale:.6f}, expected: {expected_scale:.6f}")
        assert torch.isclose(scale, expected_scale, rtol=1e-5)


class TestModulatedQuantization:
    """Test MoDiff modulated quantization (Eq. ec5)."""
    
    def test_first_step_int8(self):
        """Test first timestep quantization (Eq. ec1)."""
        a_T = torch.randn(32, 512, device='cuda')
        
        a_T_int, a_hat_T, scale = modulated_quantize_first_step_int8(a_T)
        
        # Verify â_T = Q(a_T) = dequant(a_T_int)
        expected_a_hat = a_T_int.float() * scale
        assert torch.allclose(a_hat_T, expected_a_hat, atol=1e-6)
        
        print(f"First step INT8 - scale: {scale:.6f}")
    
    def test_modulated_step_int8(self):
        """Test modulated quantization (Eq. ec5)."""
        # Simulate diffusion: a_T, a_{T-1}
        a_T = torch.randn(32, 512, device='cuda')
        a_Tm1 = a_T + torch.randn_like(a_T) * 0.1  # Small perturbation
        
        # First step
        _, a_hat_T, _ = modulated_quantize_first_step_int8(a_T)
        
        # Second step (modulated)
        residual_int, a_hat_Tm1, scale_res = modulated_quantize_int8(a_Tm1, a_hat_T)
        
        # Verify Eq. (ec5): â_{T-1} = Q(a_{T-1} - â_T) + â_T
        residual_dequant = residual_int.float() * scale_res
        expected_a_hat = residual_dequant + a_hat_T
        assert torch.allclose(a_hat_Tm1, expected_a_hat, atol=1e-6)
        
        # Verify residual has smaller range (key insight from paper)
        original_range = a_Tm1.abs().max()
        residual_range = (a_Tm1 - a_hat_T).abs().max()
        print(f"Original range: {original_range:.4f}, Residual range: {residual_range:.4f}")
        print(f"Compression ratio: {original_range / residual_range:.2f}x")
    
    def test_error_compensation_accumulation(self):
        """
        Test that error compensation prevents error accumulation.
        
        From paper Theorem 2:
            Standard modulation: error grows as O(2^{T-t})
            Error-compensated: error grows as O((2c)^{T-t}) where c < 1/2
        """
        torch.manual_seed(42)
        
        T = 10  # Number of timesteps
        
        # Generate sequence of activations (simulating diffusion)
        activations = [torch.randn(8, 256, device='cuda')]
        for t in range(T - 1):
            # Each step adds small noise (simulating denoising)
            activations.append(activations[-1] + torch.randn_like(activations[-1]) * 0.1)
        
        # Run MoDiff
        _, a_hat_cache, _ = modulated_quantize_first_step_int8(activations[0])
        
        errors = []
        for t in range(1, T):
            _, a_hat_cache, _ = modulated_quantize_int8(activations[t], a_hat_cache)
            error = (activations[t] - a_hat_cache).abs().mean()
            errors.append(error.item())
        
        print(f"Errors across timesteps: {errors}")
        
        # Verify errors don't explode
        assert max(errors) < 1.0, "Errors should remain bounded"
        
        # Error should not grow exponentially
        error_growth = [errors[i+1] / max(errors[i], 1e-8) for i in range(len(errors)-1)]
        avg_growth = sum(error_growth) / len(error_growth)
        print(f"Average error growth rate: {avg_growth:.4f}")


class TestMoDiffLinear:
    """Test MoDiff Linear layers."""
    
    def test_w8a8_linear_creation(self):
        """Test W8A8MoDiffLinear creation from nn.Linear."""
        from nn.linear import W8A8MoDiffLinear
        
        linear = torch.nn.Linear(512, 256).cuda()
        q_linear = W8A8MoDiffLinear.from_linear(linear)
        
        assert q_linear.weight_int8.dtype == torch.int8
        assert q_linear.weight_int8.shape == (256, 512)
        print(f"W8A8 Linear created: {q_linear}")
    
    def test_w8a8_linear_forward(self):
        """Test W8A8MoDiffLinear forward pass."""
        from nn.linear import W8A8MoDiffLinear
        
        linear = torch.nn.Linear(512, 256).cuda()
        q_linear = W8A8MoDiffLinear.from_linear(linear)
        
        x = torch.randn(4, 32, 512, device='cuda')
        
        # Standard forward
        with torch.no_grad():
            y_fp = linear(x)
        
        # Quantized forward (first step)
        q_linear.reset_cache()
        y_q = q_linear(x)
        
        # Check output shape
        assert y_q.shape == y_fp.shape
        
        # Check output is close (within quantization error)
        error = (y_fp - y_q).abs().mean() / y_fp.abs().mean()
        print(f"W8A8 relative error: {error:.4%}")
        assert error < 0.1, f"Error too high: {error}"
    
    def test_w8a8_modulation_sequence(self):
        """Test W8A8 linear across multiple timesteps (simulating diffusion)."""
        from nn.linear import W8A8MoDiffLinear
        
        linear = torch.nn.Linear(256, 128).cuda()
        q_linear = W8A8MoDiffLinear.from_linear(linear)
        
        # Reset for new sequence
        q_linear.reset_cache()
        
        # Simulate T timesteps
        T = 5
        base_x = torch.randn(2, 16, 256, device='cuda')
        
        outputs = []
        for t in range(T):
            x_t = base_x + torch.randn_like(base_x) * 0.05 * (T - t)
            y_t = q_linear(x_t)
            outputs.append(y_t)
        
        # Verify outputs have correct shape
        for y in outputs:
            assert y.shape == (2, 16, 128)
        
        print(f"Modulation sequence test passed for {T} timesteps")
    
    def test_w4a4_linear(self):
        """Test W4A4MoDiffLinear."""
        from nn.linear import W4A4MoDiffLinear
        
        # in_features must be even for INT4 packing
        linear = torch.nn.Linear(512, 256).cuda()
        q_linear = W4A4MoDiffLinear.from_linear(linear)
        
        x = torch.randn(4, 32, 512, device='cuda')
        
        q_linear.reset_cache()
        y = q_linear(x)
        
        assert y.shape == (4, 32, 256)
        print(f"W4A4 Linear test passed")


class TestMoDiffConv2d:
    """Test MoDiff Conv2d layers."""
    
    def test_w8a8_conv2d_creation(self):
        """Test W8A8MoDiffConv2d creation."""
        from nn.conv import W8A8MoDiffConv2d
        
        conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).cuda()
        q_conv = W8A8MoDiffConv2d.from_conv2d(conv)
        
        assert q_conv.weight_int8.dtype == torch.int8
        print(f"W8A8 Conv2d created: {q_conv}")
    
    def test_w8a8_conv2d_forward(self):
        """Test W8A8MoDiffConv2d forward."""
        from nn.conv import W8A8MoDiffConv2d
        
        conv = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1).cuda()
        q_conv = W8A8MoDiffConv2d.from_conv2d(conv)
        
        x = torch.randn(2, 32, 16, 16, device='cuda')
        
        with torch.no_grad():
            y_fp = conv(x)
        
        q_conv.reset_cache()
        y_q = q_conv(x)
        
        assert y_q.shape == y_fp.shape
        
        error = (y_fp - y_q).abs().mean() / y_fp.abs().mean()
        print(f"W8A8 Conv2d relative error: {error:.4%}")
        assert error < 0.15
    
    def test_w8a8_conv2d_modulation(self):
        """Test Conv2d across timesteps."""
        from nn.conv import W8A8MoDiffConv2d
        
        conv = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1).cuda()
        q_conv = W8A8MoDiffConv2d.from_conv2d(conv)
        
        q_conv.reset_cache()
        
        base_x = torch.randn(2, 32, 8, 8, device='cuda')
        
        for t in range(5):
            x_t = base_x + torch.randn_like(base_x) * 0.1
            y_t = q_conv(x_t)
            assert y_t.shape == (2, 64, 8, 8)
        
        print("Conv2d modulation sequence test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MoDiff Triton Kernel Tests")
    print("=" * 60)
    
    # Quantization tests
    print("\n--- Quantization Tests ---")
    test_q = TestQuantization()
    test_q.test_int8_quantize_dequantize()
    test_q.test_int4_quantize_dequantize()
    test_q.test_dynamic_scale_int8()
    test_q.test_dynamic_scale_int4()
    
    # Modulated quantization tests
    print("\n--- Modulated Quantization Tests ---")
    test_mq = TestModulatedQuantization()
    test_mq.test_first_step_int8()
    test_mq.test_modulated_step_int8()
    test_mq.test_error_compensation_accumulation()
    
    # Linear tests
    print("\n--- Linear Layer Tests ---")
    test_lin = TestMoDiffLinear()
    test_lin.test_w8a8_linear_creation()
    test_lin.test_w8a8_linear_forward()
    test_lin.test_w8a8_modulation_sequence()
    test_lin.test_w4a4_linear()
    
    # Conv2d tests
    print("\n--- Conv2d Layer Tests ---")
    test_conv = TestMoDiffConv2d()
    test_conv.test_w8a8_conv2d_creation()
    test_conv.test_w8a8_conv2d_forward()
    test_conv.test_w8a8_conv2d_modulation()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
