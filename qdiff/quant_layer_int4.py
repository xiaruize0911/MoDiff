"""
INT4 Quantized Layer Implementation with True Integer Storage

This module provides quantization layers that store weights in packed int4 format
(two 4-bit values per uint8 byte) for memory efficiency, while maintaining the same
API and code structure as the original fake-quantization implementation.

Key differences from original quant_layer.py:
1. Weights are stored as packed uint8 tensors (2x memory reduction)
2. Dequantization happens on-the-fly during forward pass
3. CUDA kernels are used for efficient pack/unpack operations
4. Quantization parameters (scale, zero_point) are stored separately

This is a "true int4" implementation where weights are stored as integers,
unlike the original fake-quantization which kept everything in float32.
"""

import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# Import int4 CUDA kernels for packing/unpacking
from qdiff.int4_cuda_kernels import (
    pack_int4_to_uint8,
    unpack_uint8_to_int4,
    dequantize_int4_cuda,
    quantize_to_int4_cuda
)

logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    """
    Identity module that passes input through unchanged.
    Used to temporarily disable activation functions during reconstruction.
    """
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Straight-Through Estimator for rounding operation.
    Forward: returns rounded values
    Backward: gradients pass through unchanged (as if no rounding occurred)
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    L_p norm loss function for reconstruction.
    Used in layer-wise reconstruction to minimize output differences.
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizerINT4(nn.Module):
    """
    INT4 Uniform Affine Quantizer with true integer storage.
    
    This quantizer stores weights in packed int4 format (uint8 storage with 2 values per byte)
    instead of the fake-quantization approach that keeps dequantized float32 tensors.
    
    Key behaviors:
    - Quantization: float32 → int4 [0, 15] → packed uint8 storage
    - Dequantization: unpacked int4 [0, 15] → float32 using (x - zero_point) * scale
    - Memory: 2x reduction compared to fake-quant (0.5 bytes per weight vs 4 bytes)
    
    Args:
        n_bits: Number of bits (must be 4 for this implementation)
        symmetric: If True, use symmetric quantization around zero
        channel_wise: If True, use per-channel (per-output-feature) scales
        scale_method: Method for computing quantization scale ('max' or 'mse')
        leaf_param: If True, scale/zero_point are learnable nn.Parameters
        always_zero: If True, force zero_point to be 0
        dynamic: If True, recompute scale/zero_point on each forward pass
    """
    def __init__(self, n_bits: int = 4, symmetric: bool = False, channel_wise: bool = False, 
                 scale_method: str = 'max', leaf_param: bool = False, always_zero: bool = False, 
                 dynamic: bool = False):
        super(UniformAffineQuantizerINT4, self).__init__()
        
        # Quantization configuration
        assert n_bits == 4, 'INT4 quantizer only supports 4-bit quantization'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits  # 16 levels for int4
        
        # Quantization parameters (computed during init or made learnable)
        self.scale = None
        self.zero_point = None
        self.packed_weight = None  # Stores packed int4 weights as uint8
        
        # Configuration flags
        self.inited = False
        self.leaf_param = leaf_param  # If True, scale is a learnable parameter
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.running_stat = False  # For activation quantization with running statistics
        self.always_zero = always_zero
        self.dynamic = dynamic
        
        # For running statistics (activation quantization)
        if self.leaf_param:
            self.x_min, self.x_max = None, None

    def forward(self, x: torch.Tensor):
        """
        Forward pass: quantize input to int4 and dequantize back to float32.
        
        For weight quantization:
            - x is a float32 weight tensor
            - We pack it to int4 (if not already packed)
            - Unpack and dequantize for computation
        
        For activation quantization:
            - x is a float32 activation tensor
            - We quantize-dequantize on the fly (don't store packed version)
        """
        
        # Initialize quantization parameters on first forward pass
        if self.inited is False:
            if self.leaf_param:
                # Learnable scale (for activation quantization during training)
                scale, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.scale = torch.nn.Parameter(scale)
            else:
                # Fixed scale (for weight quantization)
                self.scale, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # Dynamic quantization: recompute scale/zero_point on each forward
        if self.dynamic and (self.inited is True):
            scale, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.scale = torch.nn.Parameter(scale)
            if isinstance(self.zero_point, torch.nn.Parameter):
                self.zero_point = torch.nn.Parameter(zero_point)
            else:
                self.zero_point = zero_point

        # Update running statistics (for activation quantization with momentum)
        if self.running_stat:
            self.act_momentum_update(x)

        # Quantize to int4 and dequantize back to float32
        # x_int = round((x / scale)) + zero_point, clamped to [0, 15]
        x_int = round_ste(x / self.scale) + self.zero_point
        
        # Clamp to valid int4 range
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        
        # Dequantize: x_float = (x_int - zero_point) * scale
        x_dequant = (x_quant - self.zero_point) * self.scale
        
        return x_dequant
    
    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        """
        Update running min/max statistics for activation quantization.
        Uses exponential moving average to track activation ranges.
        """
        assert(self.inited)
        assert(self.leaf_param)

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            scale = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        else:
            scale = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max / (self.n_levels - 1)
        
        scale = torch.clamp(scale, min=1e-8)
        if not self.sym:
            self.zero_point = (-self.x_min / scale).round() if not (self.sym or self.always_zero) else 0
        self.scale = torch.nn.Parameter(scale)

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        """
        Compute quantization scale and zero_point from input tensor.
        
        For per-tensor quantization:
            - Compute global min/max
            - scale = (max - min) / (n_levels - 1)
            - zero_point = round(-min / scale)
        
        For per-channel quantization:
            - Compute min/max per output channel (first dimension)
            - Produces scale and zero_point tensors with shape [out_channels, 1, ...]
        """
        scale, zero_point = None, None
        
        if channel_wise:
            # Per-channel quantization (per output feature)
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            
            # Compute min/max per channel
            if len(x.shape) == 4:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0]
            else:
                x_max = x_clone.max(dim=-1)[0]
                x_min = x_clone.min(dim=-1)[0]
            
            # Compute scale and zero_point per channel
            scale = x_max.clone()
            zero_point = x_max.clone()
            x_absmax = torch.maximum((x_min).abs(), x_max)
            
            if self.sym:
                scale = x_absmax / self.n_levels
            else:
                scale = (x_max - x_min) / (self.n_levels - 1)
            
            scale[scale < 1e-8] = 1e-8  # Avoid division by zero
            zero_point = (-x_min / scale).round() if not (self.sym or self.always_zero) else torch.zeros_like(scale)
            
            # Reshape for broadcasting
            if len(x.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                scale = scale.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                scale = scale.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            # Per-tensor quantization
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    scale = x_absmax / self.n_levels
                else:
                    scale = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                
                if scale < 1e-8:
                    scale = 1e-8

                zero_point = round(-x.min().item() / scale) if not (self.sym or self.always_zero) else 0
                scale = torch.tensor(scale).type_as(x)

            elif self.scale_method == 'mse':
                # MSE-based scale search (more accurate but slower)
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    score = lp_loss(x, x_q, 2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        scale = (new_max - new_min) / (self.n_levels - 1) \
                            if not self.always_zero else new_max / (self.n_levels - 1)
                        zero_point = (- new_min / scale).round() if not self.always_zero else 0
            else:
                raise NotImplementedError

        return scale, zero_point

    def quantize(self, x, max_val, min_val):
        """Helper function for MSE-based scale search."""
        scale = (max_val - min_val) / (self.n_levels - 1) if not self.always_zero else max_val / (self.n_levels - 1)
        zero_point = (- min_val / scale).round() if not self.always_zero else 0
        x_int = torch.round(x / scale)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * scale
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        """Change bit width (not recommended after initialization)."""
        assert refactored_bit == 4, 'INT4 quantizer only supports 4-bit'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class QuantModuleINT4(nn.Module):
    """
    INT4 Quantized Module that stores weights in packed int4 format.
    
    This module wraps Conv2d/Linear/Conv1d layers and replaces their float32 weights
    with packed int4 storage. During forward pass, weights are unpacked and dequantized
    on-the-fly for computation.
    
    Key differences from original QuantModule:
    1. Weights are stored as packed uint8 (self.weight_int4_packed)
    2. Scale and zero_point are stored separately (self.weight_scale, self.weight_zp)
    3. Forward pass unpacks int4 → dequantizes → performs conv/linear
    4. Memory footprint: ~2x smaller than fake-quant for weights
    
    Note: Activations still use fake-quantization (quantize-dequantize in float32)
    because we don't store activations - they're computed on-the-fly.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], 
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {}, 
                 disable_act_quant: bool = False, 
                 act_quant_mode: str = 'qdiff',
                 modulate: bool = False):
        super(QuantModuleINT4, self).__init__()
        
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        
        # Set up forward function based on original module type
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        
        # Store original float32 weights (for reconstruction and fallback)
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        
        # INT4 storage: packed weights + scale + zero_point
        # These are initialized when quantization is enabled
        self.weight_int4_packed = None  # uint8 tensor, shape [out, in//2, ...]
        self.weight_scale = None        # float32 scale
        self.weight_zp = None           # float32 zero_point
        
        # Quantization state flags
        self.use_weight_quant = False
        self.use_act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant
        
        # Initialize quantizers
        # Note: weight_quantizer is used during quantization process (layer reconstruction)
        # but actual weights are stored in packed int4 format
        self.weight_quantizer = UniformAffineQuantizerINT4(**self.weight_quant_params)
        self.act_quantizer = UniformAffineQuantizerINT4(**self.act_quant_params)
        self.split = 0  # For split quantization (not commonly used with int4)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.extra_repr = org_module.extra_repr

        # Modulation support (for MoDiff algorithm)
        self.modulate = modulate
        self.a_hat = None  # Cached activation for modulation
        self.o_hat = None  # Cached output for modulation

    def quantize_weights_to_int4(self):
        """
        Quantize the original float32 weights to packed int4 format.
        
        This method:
        1. Takes self.org_weight (float32)
        2. Quantizes to int4 using quantize_to_int4_cuda
        3. Stores packed uint8 weights in self.weight_int4_packed
        4. Stores scale and zero_point in self.weight_scale and self.weight_zp
        
        This is called when quantization is enabled (set_quant_state(True, ...))
        """
        if self.weight_int4_packed is not None:
            # Already quantized
            return
        
        # Quantize float32 weights to packed int4
        packed, scale, zero_point = quantize_to_int4_cuda(
            self.org_weight,
            n_bits=4,
            symmetric=self.weight_quant_params.get('symmetric', False),
            channel_wise=self.weight_quant_params.get('channel_wise', False)
        )
        
        # Store packed weights and quantization parameters
        # Note: We use nn.Parameter for packed weights so they're saved with model state
        self.weight_int4_packed = nn.Parameter(packed, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        self.weight_zp = nn.Parameter(zero_point, requires_grad=False)
        
        logger.info(f"Quantized weights to INT4: original shape {self.org_weight.shape} "
                   f"→ packed shape {packed.shape} "
                   f"(memory: {self.org_weight.numel()*4} bytes → {packed.numel()} bytes)")

    def dequantize_weights_from_int4(self):
        """
        Dequantize packed int4 weights back to float32 for computation.
        
        Returns:
            weight_fp32: Float32 weight tensor with original shape
        """
        if self.weight_int4_packed is None:
            raise RuntimeError("Weights not quantized yet. Call quantize_weights_to_int4() first.")
        
        # Unpack and dequantize using CUDA kernel
        weight_fp32 = dequantize_int4_cuda(
            self.weight_int4_packed,
            self.weight_scale,
            self.weight_zp
        )
        
        # Remove padding if it was added during quantization
        if weight_fp32.shape[1] != self.org_weight.shape[1]:
            weight_fp32 = weight_fp32[:, :self.org_weight.shape[1], ...]
        
        return weight_fp32

    def forward(self, input: torch.Tensor, split: int = 0):
        """
        Forward pass with int4 weight dequantization.
        
        Flow:
        1. Quantize activations if enabled (fake-quant: float32 → int4 → float32)
        2. Get weights: either dequantized int4 or original float32
        3. Perform conv/linear operation
        4. Apply activation function
        5. Handle modulation if enabled (for MoDiff)
        """
        # Handle split quantization (rare, mostly for experimentation)
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        # === Activation quantization (fake-quant, not stored) ===
        if not self.disable_act_quant and self.use_act_quant:
            if self.modulate:
                # MoDiff modulation: quantize residual activations
                if self.a_hat is None:
                    # First pass: cache activation without quantizing
                    self.a_hat = input.clone().detach()
                else:
                    # Subsequent passes: quantize residual (input - cached)
                    input = input - self.a_hat
                    if self.split != 0:
                        input_0 = self.act_quantizer(input[:, :self.split, :, :])
                        input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                        input = torch.cat([input_0, input_1], dim=1)
                    else:
                        input = self.act_quantizer(input)
                    self.a_hat = (self.a_hat + input).clone().detach()
            else:
                # Standard activation quantization
                if self.split != 0:
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                    input = torch.cat([input_0, input_1], dim=1)
                else:
                    input = self.act_quantizer(input)
        
        # === Weight quantization (true int4 storage) ===
        if self.use_weight_quant:
            # Quantize weights to int4 if not already done
            if self.weight_int4_packed is None:
                self.quantize_weights_to_int4()
            
            # Dequantize int4 weights to float32 for computation
            weight = self.dequantize_weights_from_int4()
            bias = self.bias
        else:
            # Use original float32 weights
            weight = self.org_weight
            bias = self.org_bias
        
        # === Perform convolution or linear operation ===
        if self.modulate and self.use_act_quant:
            # MoDiff modulation for outputs
            if self.o_hat is None:
                out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            else:
                out = self.fwd_func(input, weight, None, **self.fwd_kwargs)
                out = self.o_hat + out
            self.o_hat = out.clone().detach()
        else:
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        # Apply activation function (e.g., ReLU, or StraightThrough during reconstruction)
        out = self.activation_function(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        Enable/disable quantization for weights and activations.
        
        When weight_quant is enabled, weights are quantized to int4 if not already done.
        """
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
        # Quantize weights when enabling weight quantization
        if weight_quant and self.weight_int4_packed is None:
            self.quantize_weights_to_int4()

    def set_dynamic_state(self, dynamic):
        """Enable/disable dynamic activation quantization (recompute scale each forward)."""
        self.act_quantizer.dynamic = dynamic
        if self.split != 0:
            self.act_quantizer_0.dynamic = dynamic

    def set_modualtion(self, modulation):
        """Enable/disable modulation (for MoDiff algorithm)."""
        self.modulate = modulation

    def reset_cache(self):
        """Reset cached activations and outputs (for modulation)."""
        self.a_hat = None
        self.o_hat = None

    def set_split(self):
        """Initialize split quantizers (for experimental split-channel quantization)."""
        self.weight_quantizer_0 = UniformAffineQuantizerINT4(**self.weight_quant_params)
        self.act_quantizer_0 = UniformAffineQuantizerINT4(**self.act_quant_params)

    def set_running_stat(self, running_stat: bool):
        """Enable/disable running statistics for activation quantization."""
        self.act_quantizer.running_stat = running_stat
        if self.split != 0:
            self.act_quantizer_0.running_stat = running_stat


# ============================================================================
# Additional utility functions for int4 quantization
# ============================================================================

def convert_model_to_int4(model, weight_quant_params: dict = None, act_quant_params: dict = None):
    """
    Convert all Conv2d/Linear layers in a model to INT4 quantized layers.
    
    This recursively replaces nn.Conv2d, nn.Linear, nn.Conv1d modules with
    QuantModuleINT4 instances that store weights in packed int4 format.
    
    Args:
        model: PyTorch model to convert
        weight_quant_params: Dict of parameters for weight quantization
        act_quant_params: Dict of parameters for activation quantization
    
    Returns:
        model: Modified model with INT4 quantized layers
    
    Example:
        >>> model = torchvision.models.resnet18()
        >>> weight_params = {'n_bits': 4, 'channel_wise': True, 'symmetric': False}
        >>> act_params = {'n_bits': 4, 'channel_wise': False, 'symmetric': False}
        >>> model = convert_model_to_int4(model, weight_params, act_params)
        >>> model.eval()
        >>> # Now model uses int4 weights with 2x memory reduction
    """
    if weight_quant_params is None:
        weight_quant_params = {'n_bits': 4, 'channel_wise': False, 'symmetric': False}
    if act_quant_params is None:
        act_quant_params = {'n_bits': 4, 'channel_wise': False, 'symmetric': False}
    
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            # Replace with INT4 quantized module
            quant_module = QuantModuleINT4(
                module, 
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params
            )
            setattr(model, name, quant_module)
            logger.info(f"Converted {name} to INT4 quantized layer")
        elif len(list(module.children())) > 0:
            # Recursively convert child modules
            convert_model_to_int4(module, weight_quant_params, act_quant_params)
    
    return model
