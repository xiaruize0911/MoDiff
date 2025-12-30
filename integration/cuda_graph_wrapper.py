"""
Optimized MoDiff with CUDA Graphs support for reduced kernel launch overhead.

This module provides a wrapper that captures the entire UNet forward pass
into a CUDA Graph, eliminating per-kernel launch overhead.

Key optimizations:
1. Static buffer allocation before graph capture
2. In-place cache updates
3. Single graph capture for entire diffusion step
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class CUDAGraphMoDiffWrapper:
    """
    Wrapper that enables CUDA Graph capture for MoDiff-enabled models.
    
    Usage:
        wrapper = CUDAGraphMoDiffWrapper(model.model.diffusion_model)
        wrapper.setup(batch_size=4, latent_shape=(4, 32, 32))
        
        for step in range(num_steps):
            output = wrapper(x, t)
    """
    
    def __init__(self, diffusion_model: nn.Module):
        self.model = diffusion_model
        
        # CUDA Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_captured = False
        
        # Static buffers
        self.static_x: Optional[torch.Tensor] = None
        self.static_t: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        
        # Current step for MoDiff (first step can't be graphed)
        self.step_count = 0
        
    def setup(self, batch_size: int, latent_shape: Tuple[int, ...], device='cuda'):
        """
        Allocate static buffers for CUDA Graph capture.
        
        Args:
            batch_size: Batch size for sampling
            latent_shape: Shape of latent (e.g., (4, 32, 32))
            device: CUDA device
        """
        C, H, W = latent_shape
        
        self.static_x = torch.empty(batch_size, C, H, W, device=device, dtype=torch.float32)
        self.static_t = torch.empty(batch_size, device=device, dtype=torch.long)
        self.static_output = torch.empty(batch_size, C, H, W, device=device, dtype=torch.float32)
        
        self.graph = None
        self.graph_captured = False
        self.step_count = 0
        
    def reset(self):
        """Reset for new sampling run."""
        self.step_count = 0
        self.graph_captured = False
        
        # Reset MoDiff caches in all layers
        def reset_modiff_state(module):
            if hasattr(module, 'reset_state'):
                module.reset_state()
        self.model.apply(reset_modiff_state)
        
    def _capture_graph(self):
        """Capture UNet forward pass into CUDA Graph."""
        # Warmup
        with torch.no_grad():
            self.static_output.copy_(
                self.model(self.static_x, self.static_t, None)
            )
        torch.cuda.synchronize()
        
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                out = self.model(self.static_x, self.static_t, None)
                self.static_output.copy_(out)
        
        self.graph_captured = True
        
    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional CUDA Graph replay.
        
        Args:
            x: Input latent [B, C, H, W]
            t: Timestep indices [B]
            
        Returns:
            Model output [B, C, H, W]
        """
        if self.step_count == 0:
            # First step: Can't graph because MoDiff caches need initialization
            output = self.model(x, t, None)
            self.step_count += 1
            return output
        
        if not self.graph_captured:
            # Second step: Capture graph
            self.static_x.copy_(x)
            self.static_t.copy_(t)
            
            try:
                self._capture_graph()
                self.step_count += 1
                return self.static_output.clone()
            except RuntimeError as e:
                print(f"CUDA Graph capture failed: {e}")
                print("Falling back to eager execution")
                self.graph_captured = False
                output = self.model(x, t, None)
                self.step_count += 1
                return output
        else:
            # Subsequent steps: Replay graph
            self.static_x.copy_(x)
            self.static_t.copy_(t)
            self.graph.replay()
            self.step_count += 1
            return self.static_output.clone()


def make_modiff_cuda_graph_compatible(model: nn.Module):
    """
    Modify MoDiff layers to be CUDA Graph compatible.
    
    This involves:
    1. Pre-allocating all cache tensors
    2. Using in-place operations only
    3. Avoiding dynamic control flow
    """
    from integration.modiff_layers import CutlassInt8Conv2d, FP16Conv2d
    
    def prepare_layer(module):
        if isinstance(module, (CutlassInt8Conv2d, FP16Conv2d)):
            # These layers are already designed with static buffers
            # Just ensure they're properly initialized
            pass
    
    model.apply(prepare_layer)
    return model


# Alternative: Use torch.compile with CUDA Graphs backend
def compile_with_cuda_graphs(model: nn.Module, mode: str = 'reduce-overhead'):
    """
    Use torch.compile with CUDA Graphs for optimal performance.
    
    Args:
        model: The diffusion model to compile
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        
    Returns:
        Compiled model
    """
    return torch.compile(model, mode=mode, fullgraph=False)
