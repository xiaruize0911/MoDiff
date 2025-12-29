
import torch
import torch.nn as nn
import unittest
from modiff_utils import MoDiffConv2dWrapper, convert_model_to_modiff, enable_modiff_mode, reset_modiff_state

class TestMoDiffIntegration(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("Warning: CUDA not available, testing on CPU (kernel will be simulated)")
            
    def test_wrapper_basic(self):
        # Create a simple Conv2d
        conv = nn.Conv2d(3, 16, 3, padding=1).to(self.device)
        wrapper = MoDiffConv2dWrapper(conv, use_cuda_kernel=False).to(self.device) # Force simulation for logic test
        
        x = torch.randn(1, 3, 32, 32).to(self.device)
        
        # Test disabled (standard forward)
        wrapper.enable_modiff(False)
        out_std = wrapper(x)
        self.assertEqual(out_std.shape, (1, 16, 32, 32))
        
        # Test enabled (MoDiff logic)
        wrapper.enable_modiff(True)
        
        # Step 1: Full compute
        out_1 = wrapper(x)
        self.assertTrue(torch.allclose(out_std, out_1, atol=1e-5))
        self.assertIsNotNone(wrapper.last_input)
        self.assertIsNotNone(wrapper.last_output)
        
        # Step 2: Delta compute
        # Create a small delta
        x2 = x + 0.1 * torch.randn_like(x)
        out_std_2 = conv(x2)
        
        out_2 = wrapper(x2)
        
        # In simulation mode (FP32), it should be exactly equal (linearity)
        # out_2 = conv(x2 - x) + out_1
        #       = conv(x2) - conv(x) + conv(x) = conv(x2)
        self.assertTrue(torch.allclose(out_std_2, out_2, atol=1e-5))
        
    def test_model_conversion(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1)
        ).to(self.device)
        
        convert_model_to_modiff(model, use_cuda_kernel=False)
        
        # Check if layers are replaced
        self.assertTrue(isinstance(model[0], MoDiffConv2dWrapper))
        self.assertTrue(isinstance(model[2], MoDiffConv2dWrapper))
        
        x = torch.randn(1, 3, 32, 32).to(self.device)
        
        # Test forward pass
        enable_modiff_mode(model, True)
        out = model(x)
        self.assertEqual(out.shape, (1, 32, 32, 32))

if __name__ == '__main__':
    unittest.main()
