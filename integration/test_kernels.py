
import torch
import modiff_cuda_backend
import unittest

class TestKernels(unittest.TestCase):
    # def test_permute(self):
    #     N, C, H, W = 8, 192, 32, 32
    #     input_nhwc = torch.randn(N, H, W, C, dtype=torch.float16, device='cuda')
        
    #     # Test with compute_max=False
    #     out_nchw, _ = modiff_cuda_backend.permute_half_nhwc_nchw(input_nhwc, False)
        
    #     expected = input_nhwc.permute(0, 3, 1, 2).contiguous()
        
    #     self.assertTrue(torch.allclose(out_nchw, expected))
    #     print("Permute test passed")

    def test_conv(self):
        N, C_in, H, W = 8, 32, 32, 32
        C_out = 192
        K = 3
        stride = 1
        padding = 1
        
        # Input (Int8)
        input_int8 = torch.randint(-127, 127, (N, H, W, C_in), dtype=torch.int8, device='cuda')
        
        # Weight (Int8) - Layout [C_out, C_in * K * K]
        weight = torch.randint(-127, 127, (C_out, C_in * K * K), dtype=torch.int8, device='cuda')
        
        # Scales
        act_scale = torch.tensor(0.1, device='cuda')
        weight_scales = torch.ones(C_out, device='cuda') * 0.01
        
        # Run conv
        out_nhwc, _ = modiff_cuda_backend.conv2d_fast_w8a8(
            input_int8,
            weight,
            act_scale,
            weight_scales,
            K,
            stride,
            padding,
            False # compute_max
        )
        
        print(f"Conv output shape: {out_nhwc.shape}, dtype: {out_nhwc.dtype}")
        self.assertEqual(out_nhwc.shape, (N, H, W, C_out))
        self.assertEqual(out_nhwc.dtype, torch.float16)
        
        # Run permute on output
        out_nchw, _ = modiff_cuda_backend.permute_half_nhwc_nchw(out_nhwc, False)
        print("Conv + Permute passed")

if __name__ == '__main__':
    unittest.main()
