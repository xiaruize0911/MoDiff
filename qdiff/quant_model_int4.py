"""
INT4 Quantized Model Wrapper

This module provides QuantModelINT4, which wraps a diffusion model and recursively replaces
all Conv2d/Linear layers with QuantModuleINT4 (true int4 weight storage) and all special
blocks (ResBlock, TransformerBlock, etc.) with their INT4 quantized versions.

The code structure is identical to quant_model.py, with the key difference being:
- Uses QuantModuleINT4 instead of QuantModule (true int4 weight storage)
- Uses INT4 quantized blocks (QuantResBlockINT4, QuantBasicTransformerBlockINT4, etc.)
- Weights are stored as packed uint8 (2x memory savings) instead of fake-quantized float32

All APIs remain unchanged for drop-in compatibility with the original MoDiff code.
"""

import logging
import torch.nn as nn

# Import INT4 quantized blocks and layers
from qdiff.quant_block_int4 import get_specials_int4, BaseQuantBlockINT4
from qdiff.quant_block_int4 import (
    QuantBasicTransformerBlockINT4,
    QuantResBlockINT4,
    QuantAttnBlockINT4,
    QuantQKMatMulINT4,
    QuantSMVMatMulINT4
)
from qdiff.quant_layer_int4 import QuantModuleINT4, StraightThrough

# Import original model structures for type checking
from ldm.modules.attention import BasicTransformerBlock

logger = logging.getLogger(__name__)


class QuantModelINT4(nn.Module):
    """
    INT4 Quantized Model Wrapper for Diffusion Models
    
    This class wraps a diffusion model (e.g., Stable Diffusion, LDM, DDIM) and:
    1. Replaces all Conv2d/Linear layers with QuantModuleINT4 (true int4 weight storage)
    2. Replaces special blocks (ResBlock, TransformerBlock, etc.) with INT4 versions
    3. Provides unified API for enabling/disabling quantization, modulation, etc.
    
    Key differences from QuantModel:
    - Uses QuantModuleINT4 → weights stored as packed uint8 (true int4)
    - Memory savings: ~2x reduction for weights compared to fake-quantization
    - Same API: drop-in replacement for QuantModel
    
    Args:
        model: Original diffusion model (nn.Module)
        weight_quant_params: Quantization config for weights (n_bits, channel_wise, etc.)
        act_quant_params: Quantization config for activations
        sm_abit: Bit width for attention weights (softmax output), default 8
        modulate: Enable modulation for MoDiff algorithm
    
    Example:
        >>> from ldm.models.diffusion.ddpm import LatentDiffusion
        >>> from qdiff.quant_model_int4 import QuantModelINT4
        >>> 
        >>> model = LatentDiffusion.load_from_checkpoint(...)
        >>> weight_params = {'n_bits': 4, 'channel_wise': True, 'symmetric': False}
        >>> act_params = {'n_bits': 4, 'channel_wise': False, 'symmetric': False}
        >>> 
        >>> # Wrap model with INT4 quantization
        >>> qmodel = QuantModelINT4(model.model, weight_params, act_params)
        >>> 
        >>> # Enable weight quantization (packs weights to int4)
        >>> qmodel.set_quant_state(weight_quant=True, act_quant=False)
        >>> 
        >>> # Run inference (weights automatically unpacked and dequantized)
        >>> output = qmodel(x, timesteps, context)
    """
    
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, 
                 act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)  # Softmax attention bit width
        self.modulate = kwargs.get("modulate", False)  # MoDiff modulation
        
        # Copy model attributes for compatibility
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        
        # Get mapping of special blocks to INT4 versions
        self.specials = get_specials_int4(act_quant_params['leaf_param'])
        
        # Recursively refactor model: Conv2d/Linear → QuantModuleINT4
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        
        # Replace special blocks with INT4 versions (ResBlock, TransformerBlock, etc.)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, 
                             act_quant_params: dict = {}):
        """
        Recursively replace Conv2d/Conv1d/Linear layers with QuantModuleINT4.
        
        This walks the module tree and wraps each convolutional or linear layer
        with QuantModuleINT4, which:
        - Stores original float32 weights (org_weight)
        - Quantizes to packed int4 on first forward pass (when enabled)
        - Unpacks and dequantizes int4 weights during forward pass
        
        Args:
            module: Parent module to refactor
            weight_quant_params: Config for weight quantization
            act_quant_params: Config for activation quantization
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                # Replace with INT4 quantized module
                setattr(module, name, QuantModuleINT4(
                    child_module, weight_quant_params, act_quant_params, modulate=self.modulate))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                # Skip identity layers
                continue

            else:
                # Recursively refactor children
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, 
                            act_quant_params: dict = {}):
        """
        Recursively replace special blocks with INT4 quantized versions.
        
        Special blocks include:
        - ResBlock → QuantResBlockINT4
        - BasicTransformerBlock → QuantBasicTransformerBlockINT4
        - ResnetBlock → QuantResnetBlockINT4
        - AttnBlock → QuantAttnBlockINT4
        - QKMatMul → QuantQKMatMulINT4 (if leaf_param=True)
        - SMVMatMul → QuantSMVMatMulINT4 (if leaf_param=True)
        
        Args:
            module: Parent module to refactor
            weight_quant_params: Config for weight quantization
            act_quant_params: Config for activation quantization
        """
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                # Replace with INT4 quantized version
                if self.specials[type(child_module)] in [QuantBasicTransformerBlockINT4, QuantAttnBlockINT4]:
                    # Transformer and attention blocks need sm_abit parameter
                    setattr(module, name, self.specials[type(child_module)](
                        child_module, act_quant_params, sm_abit=self.sm_abit))
                
                elif self.specials[type(child_module)] == QuantSMVMatMulINT4:
                    # Softmax-value matmul needs sm_abit
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                
                elif self.specials[type(child_module)] == QuantQKMatMulINT4:
                    # Query-key matmul
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                
                else:
                    # Other special blocks (ResBlock, ResnetBlock, etc.)
                    setattr(module, name, self.specials[type(child_module)](
                        child_module, act_quant_params))
            else:
                # Recursively refactor children
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        Enable/disable quantization for all layers in the model.
        
        When weight_quant=True:
        - All QuantModuleINT4 layers will pack their weights to int4 format
        - Weights are stored as packed uint8 (2 values per byte)
        - Forward pass automatically unpacks and dequantizes weights
        
        When act_quant=True:
        - Activations are quantized using fake-quantization (quantize-dequantize)
        - Activation quantization is not stored, only simulated
        
        Args:
            weight_quant: Enable weight quantization (true int4 storage)
            act_quant: Enable activation quantization (fake-quant)
        """
        for m in self.model.modules():
            if isinstance(m, (QuantModuleINT4, BaseQuantBlockINT4)):
                m.set_quant_state(weight_quant, act_quant)
    
    def set_dynamic_state(self, dynamic):
        """
        Enable/disable dynamic activation quantization.
        
        When dynamic=True, activation scales are recomputed on each forward pass
        based on the current input range (useful for varying input distributions).
        
        Args:
            dynamic: Enable dynamic activation quantization
        """
        for m in self.model.modules():
            if isinstance(m, (QuantModuleINT4, QuantAttnBlockINT4)):
                m.set_dynamic_state(dynamic)

    def set_modulation(self, modulation):
        """
        Enable/disable modulation for MoDiff algorithm.
        
        Modulation caches intermediate activations (a_hat) and outputs (o_hat)
        and quantizes the residuals instead of absolute values, which can
        reduce quantization error.
        
        Args:
            modulation: Enable MoDiff modulation
        """
        for m in self.model.modules():
            if isinstance(m, QuantModuleINT4):
                m.set_modualtion(modulation)

    def reset_cache(self):
        """
        Reset cached activations and outputs (for modulation).
        
        This clears the a_hat and o_hat caches in all QuantModuleINT4 layers,
        which is necessary when starting a new inference pass or batch.
        """
        for m in self.model.modules():
            if isinstance(m, QuantModuleINT4):
                m.reset_cache()
                              
    def forward(self, x, timesteps=None, context=None):
        """
        Forward pass through the quantized model.
        
        Args:
            x: Input tensor (e.g., noisy latent)
            timesteps: Timestep embeddings for diffusion
            context: Context for conditional generation (e.g., text embeddings)
        
        Returns:
            Model output (e.g., predicted noise)
        """
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        """
        Enable/disable running statistics for activation quantization.
        
        When running_stat=True, activation quantizers track running min/max
        using exponential moving average, which can improve quantization
        accuracy over time.
        
        Args:
            running_stat: Enable running statistics
            sm_only: If True, only enable for softmax/attention weights (not Q/K/V)
        """
        for m in self.model.modules():
            # Enable running stats for transformer attention blocks
            if isinstance(m, QuantBasicTransformerBlockINT4):
                if sm_only:
                    # Only attention weights (softmax output)
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    # All attention quantizers (Q, K, V, W)
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            
            # Enable running stats for regular quantized modules
            if isinstance(m, QuantModuleINT4) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        """
        Enable/disable gradient checkpointing for transformer blocks.
        
        Gradient checkpointing trades compute for memory by not storing
        intermediate activations during the forward pass, recomputing them
        during backward pass instead.
        
        Args:
            grad_ckpt: Enable gradient checkpointing
        """
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlockINT4, BasicTransformerBlock)):
                m.checkpoint = grad_ckpt
            # Note: ResBlock checkpointing can be enabled similarly if needed
            # elif isinstance(m, QuantResBlockINT4):
            #     m.use_checkpoint = grad_ckpt


# ============================================================================
# Utility functions for working with QuantModelINT4
# ============================================================================

def count_quantized_params(qmodel: QuantModelINT4):
    """
    Count the number of quantized parameters and compute memory savings.
    
    Args:
        qmodel: QuantModelINT4 instance
    
    Returns:
        Dictionary with parameter counts and memory usage statistics
    """
    total_params = 0
    quantized_params = 0
    fp32_memory = 0
    int4_memory = 0
    
    for m in qmodel.modules():
        if isinstance(m, QuantModuleINT4):
            if m.weight_int4_packed is not None:
                # Weight is quantized
                num_params = m.org_weight.numel()
                quantized_params += num_params
                fp32_memory += num_params * 4  # 4 bytes per float32
                int4_memory += m.weight_int4_packed.numel() * 1  # 1 byte per uint8 (2 int4 values)
            total_params += m.org_weight.numel()
            
            # Add bias (not quantized)
            if m.bias is not None:
                total_params += m.bias.numel()
                fp32_memory += m.bias.numel() * 4
                int4_memory += m.bias.numel() * 4
    
    return {
        'total_params': total_params,
        'quantized_params': quantized_params,
        'quantization_ratio': quantized_params / total_params if total_params > 0 else 0,
        'fp32_memory_bytes': fp32_memory,
        'int4_memory_bytes': int4_memory,
        'memory_savings_bytes': fp32_memory - int4_memory,
        'compression_ratio': fp32_memory / int4_memory if int4_memory > 0 else 0,
    }


def print_quantization_summary(qmodel: QuantModelINT4):
    """
    Print a summary of the quantized model's memory usage and compression.
    
    Args:
        qmodel: QuantModelINT4 instance
    """
    stats = count_quantized_params(qmodel)
    
    print("\n" + "="*70)
    print("INT4 Quantization Summary")
    print("="*70)
    print(f"Total parameters:     {stats['total_params']:,}")
    print(f"Quantized parameters: {stats['quantized_params']:,} "
          f"({stats['quantization_ratio']*100:.1f}%)")
    print(f"\nMemory Usage:")
    print(f"  FP32:  {stats['fp32_memory_bytes']:,} bytes "
          f"({stats['fp32_memory_bytes']/1024**2:.2f} MB)")
    print(f"  INT4:  {stats['int4_memory_bytes']:,} bytes "
          f"({stats['int4_memory_bytes']/1024**2:.2f} MB)")
    print(f"  Saved: {stats['memory_savings_bytes']:,} bytes "
          f"({stats['memory_savings_bytes']/1024**2:.2f} MB)")
    print(f"\nCompression Ratio: {stats['compression_ratio']:.2f}x")
    print("="*70 + "\n")
