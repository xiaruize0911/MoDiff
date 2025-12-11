"""
Quantized Layer with MoDiff Error-Compensated Modulation

This module implements the core MoDiff algorithm from the paper at the layer level.
Each layer maintains state (a_hat, o_hat) for error-compensated modulation.

Paper Reference:
- Section 3.2: Modulated Quantization (Equations 5-8)
- Section 3.3: Error-Compensated Modulation (Equations 13-20)

Key Insight:
    Instead of quantizing raw activations a_t, we quantize the RESIDUAL:
    Q(a_t - â_{t+1})
    
    This residual has ~10x smaller range, resulting in ~100x lower quantization error.
"""

import logging
from typing import Union, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_int8_modiff.quant_int8_native import NativeINT8Quantizer, SimulatedINT8Quantizer

logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    """Identity activation (pass-through)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class QuantLayerMoDiff(nn.Module):
    """
    Quantized Layer with MoDiff Error-Compensated Modulation.
    
    This is the core building block that implements:
    1. Weight quantization (static, per-channel)
    2. Activation quantization on RESIDUALS (dynamic, per-tensor)
    3. Error-compensated modulation (cache â_hat and ô_hat)
    
    Paper Algorithm (Equations 13-20):
    
    First timestep (t=T):
        â_T = Q(a_T)
        ô_T = A(â_T)
        
    Subsequent timesteps (t=T-1, ..., 1):
        â_t = Q(a_t - â_{t+1}) + â_{t+1}    # Quantize residual, add back
        ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}  # Incremental output update
    
    where:
        - a_t: current activation (input to this layer)
        - â_t: quantized activation (stored for next timestep)
        - ô_t: output with error compensation
        - A(): linear operator (Conv2d, Linear)
        - Q(): quantization function
    """
    
    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
        weight_quant_params: Dict[str, Any] = None,
        act_quant_params: Dict[str, Any] = None,
        disable_act_quant: bool = False,
        modulate: bool = True,
        use_native_int8: bool = True,
    ):
        super().__init__()
        
        # Default quantization parameters
        if weight_quant_params is None:
            weight_quant_params = {
                'n_bits': 8,
                'symmetric': True,
                'channel_wise': True,
                'scale_method': 'mse',
            }
        if act_quant_params is None:
            act_quant_params = {
                'n_bits': 8,
                'symmetric': True,
                'channel_wise': False,
                'scale_method': 'mse',
                'dynamic': True,  # Dynamic for activations
            }
        
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        
        # Store original module properties
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv2d
            self.layer_type = 'conv2d'
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv1d
            self.layer_type = 'conv1d'
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = {}
            self.fwd_func = F.linear
            self.layer_type = 'linear'
        else:
            raise ValueError(f"Unsupported module type: {type(org_module)}")
        
        # Store weights
        self.weight = nn.Parameter(org_module.weight.data.clone())
        self.org_weight = org_module.weight.data.clone()
        
        if org_module.bias is not None:
            self.bias = nn.Parameter(org_module.bias.data.clone())
            self.org_bias = org_module.bias.data.clone()
        else:
            self.register_parameter('bias', None)
            self.org_bias = None
        
        # Quantizers
        QuantizerClass = NativeINT8Quantizer if use_native_int8 else SimulatedINT8Quantizer
        
        self.weight_quantizer = QuantizerClass(
            n_bits=weight_quant_params.get('n_bits', 8),
            symmetric=weight_quant_params.get('symmetric', True),
            channel_wise=weight_quant_params.get('channel_wise', True),
            scale_method=weight_quant_params.get('scale_method', 'mse'),
            dynamic=False,  # Static for weights
        )
        
        self.act_quantizer = QuantizerClass(
            n_bits=act_quant_params.get('n_bits', 8),
            symmetric=act_quant_params.get('symmetric', True),
            channel_wise=act_quant_params.get('channel_wise', False),
            scale_method=act_quant_params.get('scale_method', 'mse'),
            dynamic=act_quant_params.get('dynamic', True),
        )
        
        # State flags
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        self.modulate = modulate
        
        # MoDiff state: cached quantized input and output
        # These persist across timesteps within one sample generation
        self.a_hat: Optional[torch.Tensor] = None  # Cached quantized input
        self.o_hat: Optional[torch.Tensor] = None  # Cached output
        
        # Activation function (usually identity for diffusion models)
        self.activation_function = StraightThrough()
        
        # For reconstruction
        self.ignore_reconstruction = False
        
        # Cache quantized weight to avoid re-quantizing every forward
        self._cached_qweight: Optional[torch.Tensor] = None
        self._weight_dirty = True
    
    def reset_cache(self) -> None:
        """
        Reset MoDiff cache at the start of each sample generation.
        
        Must be called before generating each new sample!
        """
        self.a_hat = None
        self.o_hat = None
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """Enable or disable quantization."""
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        self._weight_dirty = True  # Invalidate weight cache
    
    def set_modulation(self, modulate: bool) -> None:
        """Enable or disable MoDiff modulation."""
        self.modulate = modulate
    
    def calibrate_weight(self) -> None:
        """Calibrate weight quantizer using current weights."""
        self.weight_quantizer.calibrate(self.weight.data)
        self._weight_dirty = True
    
    def calibrate_activation(self, x: torch.Tensor) -> None:
        """Calibrate activation quantizer using sample activations."""
        self.act_quantizer.calibrate(x)
    
    def _get_quantized_weight(self) -> torch.Tensor:
        """Get quantized weight, using cache when possible."""
        if self._weight_dirty or self._cached_qweight is None:
            if self.use_weight_quant:
                self._cached_qweight = self.weight_quantizer(self.weight)
            else:
                self._cached_qweight = self.org_weight
            self._weight_dirty = False
        return self._cached_qweight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization and MoDiff modulation.
        
        MoDiff Algorithm (when modulate=True and use_act_quant=True):
        
        First timestep (a_hat is None):
            1. Quantize input directly: â = Q(a)
            2. Compute output: ô = Conv(â, W)
            3. Store â and ô for next timestep
            
        Subsequent timesteps:
            1. Compute residual: r = a - â_prev
            2. Quantize residual: Q(r)  # Much smaller range!
            3. Update cached input: â = â_prev + Q(r)
            4. Compute delta output: Δo = Conv(Q(r), W)
            5. Update output: ô = ô_prev + Δo
            6. Store â and ô for next timestep
        
        Args:
            x: Input activation tensor
            
        Returns:
            Output tensor
        """
        # Fast path: no quantization
        if not self.use_weight_quant and not self.use_act_quant and not self.modulate:
            out = self.fwd_func(x, self.org_weight, self.org_bias, **self.fwd_kwargs)
            return self.activation_function(out)
        
        # Get (possibly quantized) weight
        weight = self._get_quantized_weight()
        
        # === MoDiff Error-Compensated Modulation ===
        if self.modulate and self.use_act_quant and not self.disable_act_quant:
            if self.a_hat is None:
                # First timestep: direct quantization
                # â_T = Q(a_T)
                q_input = self.act_quantizer(x)
                self.a_hat = q_input.detach().clone()
                
                # ô_T = A(â_T)
                out = self.fwd_func(q_input, weight, self.bias, **self.fwd_kwargs)
                self.o_hat = out.detach().clone()
            else:
                # Subsequent timesteps: quantize RESIDUAL
                # This is the key insight of MoDiff!
                # Residual has ~10x smaller range → ~100x lower quant error
                
                # r = a_t - â_{t+1}
                residual = x - self.a_hat
                
                # Q(r) - quantize the small residual
                q_residual = self.act_quantizer(residual)
                
                # â_t = â_{t+1} + Q(r)
                # Update cached input for next timestep
                self.a_hat = (self.a_hat + q_residual).detach().clone()
                
                # Δo = A(Q(r), W) - no bias for delta!
                delta_out = self.fwd_func(q_residual, weight, None, **self.fwd_kwargs)
                
                # ô_t = ô_{t+1} + Δo
                out = self.o_hat + delta_out
                self.o_hat = out.detach().clone()
        
        # Standard quantization (no modulation)
        elif self.use_act_quant and not self.disable_act_quant:
            q_input = self.act_quantizer(x)
            out = self.fwd_func(q_input, weight, self.bias, **self.fwd_kwargs)
        
        # No activation quantization
        else:
            out = self.fwd_func(x, weight, self.bias, **self.fwd_kwargs)
        
        return self.activation_function(out)
    
    def extra_repr(self) -> str:
        return (
            f"layer_type={self.layer_type}, "
            f"weight_quant={self.use_weight_quant}, "
            f"act_quant={self.use_act_quant}, "
            f"modulate={self.modulate}"
        )


def convert_to_modiff_layer(
    module: nn.Module,
    weight_quant_params: Dict[str, Any] = None,
    act_quant_params: Dict[str, Any] = None,
    modulate: bool = True,
    use_native_int8: bool = True,
) -> nn.Module:
    """
    Convert a single Conv2d/Linear module to QuantLayerMoDiff.
    
    Args:
        module: Original nn.Conv2d or nn.Linear
        weight_quant_params: Weight quantization parameters
        act_quant_params: Activation quantization parameters
        modulate: Enable MoDiff modulation
        use_native_int8: Use native INT8 ops
        
    Returns:
        QuantLayerMoDiff instance
    """
    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        return QuantLayerMoDiff(
            module,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            modulate=modulate,
            use_native_int8=use_native_int8,
        )
    else:
        return module
