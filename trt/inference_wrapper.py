"""
TensorRT Inference Wrapper for MoDiff UNet

This module provides a PyTorch-compatible interface to run
TensorRT engines for fast inference.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tensorrt as trt
import torch.cuda as cuda


class TRTEngineWrapper:
    """
    Wrapper around TensorRT engine for MoDiff UNet inference.
    
    Loads a .plan file and provides a PyTorch-compatible forward() method
    that accepts torch tensors and returns torch tensors.
    """

    def __init__(self, engine_path: str, device: int = 0):
        """
        Initialize TensorRT engine wrapper.
        
        Args:
            engine_path: Path to .plan file
            device: CUDA device index (default: 0)
        """
        self.engine_path = Path(engine_path).expanduser().resolve()
        self.device_idx = device
        self.torch_device = torch.device(f"cuda:{device}")
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        # Create CUDA stream for this engine
        with torch.cuda.device(self.torch_device):
            self.stream = torch.cuda.Stream()
        
        # Load TensorRT engine
        self._load_engine()
        
    def _load_engine(self) -> None:
        """Load TensorRT engine from disk."""
        print(f"[TRT] Loading engine from {self.engine_path}")
        
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with self.engine_path.open("rb") as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get input/output names and shapes (TensorRT 10.x API)
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        # In TensorRT 10.x, use num_io_tensors instead of num_bindings
        self.binding_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for binding in self.binding_names:
            shape = tuple(self.context.get_tensor_shape(binding))
            
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
                self.input_shapes[binding] = shape
            else:
                self.output_names.append(binding)
                self.output_shapes[binding] = shape
        
        print(f"[TRT] Inputs: {self.input_shapes}")
        print(f"[TRT] Outputs: {self.output_shapes}")
        
    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference on latent and timesteps.
        
        Args:
            latent: Input latent tensor, shape (B, C, H, W)
            timesteps: Input timestep tensor, shape (B,)
        
        Returns:
            Output tensor, shape (B, 3, H, W) for CIFAR10 UNet
        """
        # Ensure inputs are on the correct device and have the correct dtype
        latent_gpu = latent.to(device=self.torch_device, dtype=torch.float32).contiguous()
        timesteps_gpu = timesteps.to(device=self.torch_device, dtype=torch.int64).contiguous()

        # Create output tensor
        output_name = self.output_names[0]
        
        # The output shape from the engine might have dynamic dimensions (-1).
        # We need to set the batch size of the output shape to match the input batch size.
        output_shape = list(self.output_shapes[output_name])
        output_shape[0] = latent_gpu.shape[0]
        
        output_gpu = torch.empty(
            tuple(output_shape),
            dtype=torch.float32,
            device=self.torch_device
        )
        
        # Use the custom stream for all CUDA operations
        with torch.cuda.stream(self.stream):
            # Set tensor addresses
            self.context.set_input_shape('latent', latent_gpu.shape)
            self.context.set_input_shape('timesteps', timesteps_gpu.shape)
            
            bindings = [None] * self.engine.num_io_tensors
            bindings[self.binding_names.index('latent')] = latent_gpu.data_ptr()
            bindings[self.binding_names.index('timesteps')] = timesteps_gpu.data_ptr()
            bindings[self.binding_names.index(output_name)] = output_gpu.data_ptr()
            
            # Run inference
            self.context.execute_v2(bindings=bindings)
        
        # Synchronize the custom stream to ensure completion
        self.stream.synchronize()
        
        return output_gpu
    
    def __call__(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(latent, timesteps)


def load_trt_engine(engine_path: str, device: int = 0) -> TRTEngineWrapper:
    """
    Convenience function to load a TensorRT engine.
    
    Args:
        engine_path: Path to .plan file
        device: CUDA device index
    
    Returns:
        TRTEngineWrapper instance
    """
    return TRTEngineWrapper(engine_path, device)


if __name__ == "__main__":
    # Example usage
    engine_path = Path(__file__).parent / "export" / "modiff_unet_fp32.plan"
    
    wrapper = TRTEngineWrapper(str(engine_path))
    
    # Create dummy inputs (note: CIFAR10 model takes 3-channel input, not 4-channel latents)
    latent = torch.randn(1, 3, 32, 32, device="cpu")
    timesteps = torch.tensor([100], dtype=torch.long)
    
    print("[Demo] Running inference with TensorRT engine...")
    output = wrapper(latent, timesteps)
    
    print(f"[Demo] Input shape: {latent.shape}")
    print(f"[Demo] Output shape: {output.shape}")
    print(f"[Demo] Output dtype: {output.dtype}")
    print(f"[Demo] Output device: {output.device}")
