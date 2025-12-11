"""
Quantized Model Wrapper with MoDiff Support

This module wraps a full diffusion model (UNet) for quantized inference
with MoDiff error-compensated modulation.

Features:
1. Automatic conversion of Conv2d/Linear to QuantLayerMoDiff
2. Global reset_cache() for all MoDiff layers
3. Unified set_quant_state() control
4. Calibration helpers

Usage:
    model = load_pretrained_model()
    qmodel = QuantModelMoDiff(model, weight_bits=8, act_bits=8, modulate=True)
    
    # Calibrate
    qmodel.calibrate_weights()
    qmodel.calibrate_activations(calib_data)
    
    # Enable quantization
    qmodel.set_quant_state(True, True)
    
    # Generate samples
    for i in range(num_samples):
        qmodel.reset_cache()  # Reset MoDiff state!
        output = sample_with_ddim(qmodel, ...)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn

from demo_int8_modiff.quant_layer_modiff import QuantLayerMoDiff, convert_to_modiff_layer

logger = logging.getLogger(__name__)


class QuantModelMoDiff(nn.Module):
    """
    Quantized Model Wrapper with MoDiff Error-Compensated Modulation.
    
    This wraps a diffusion model (UNet) and converts all Conv2d/Linear
    layers to QuantLayerMoDiff for INT8 quantization with MoDiff.
    
    Args:
        model: Original PyTorch model (e.g., DDIM UNet)
        weight_bits: Weight quantization bits (default: 8)
        act_bits: Activation quantization bits (default: 8)
        modulate: Enable MoDiff modulation (default: True)
        use_native_int8: Use native INT8 ops (default: True)
        symmetric_act: Use symmetric activation quantization
        
    Paper Reference:
        Section 4.1: "For CIFAR-10, we use DDIM models with 100 denoising steps"
    """
    
    def __init__(
        self,
        model: nn.Module,
        weight_bits: int = 8,
        act_bits: int = 8,
        modulate: bool = True,
        use_native_int8: bool = True,
        symmetric_act: bool = True,
        sm_abit: int = 8,  # Softmax activation bits (for attention)
    ):
        super().__init__()
        
        self.model = model
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.modulate = modulate
        self.use_native_int8 = use_native_int8
        self.sm_abit = sm_abit
        
        # Quantization parameters
        self.weight_quant_params = {
            'n_bits': weight_bits,
            'symmetric': True,
            'channel_wise': True,
            'scale_method': 'mse',
        }
        
        self.act_quant_params = {
            'n_bits': act_bits,
            'symmetric': symmetric_act,
            'channel_wise': False,
            'scale_method': 'mse',
            'dynamic': True,
        }
        
        # Copy model attributes
        if hasattr(model, 'in_channels'):
            self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        
        # Convert layers to quantized versions
        self._replace_modules(self.model)
        
        logger.info(f"Created QuantModelMoDiff: W{weight_bits}A{act_bits}, "
                   f"modulate={modulate}, native_int8={use_native_int8}")
    
    def _replace_modules(self, module: nn.Module, prefix: str = "") -> None:
        """
        Recursively replace Conv2d/Linear with QuantLayerMoDiff.
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                # Replace with quantized layer
                quant_layer = QuantLayerMoDiff(
                    child,
                    weight_quant_params=self.weight_quant_params,
                    act_quant_params=self.act_quant_params,
                    modulate=self.modulate,
                    use_native_int8=self.use_native_int8,
                )
                setattr(module, name, quant_layer)
                logger.debug(f"Converted {full_name} to QuantLayerMoDiff")
            else:
                # Recurse into children
                self._replace_modules(child, full_name)
    
    def reset_cache(self) -> None:
        """
        Reset MoDiff cache for all layers.
        
        MUST be called at the start of each sample generation!
        """
        for module in self.model.modules():
            if isinstance(module, QuantLayerMoDiff):
                module.reset_cache()
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """
        Enable or disable quantization for all layers.
        
        Args:
            weight_quant: Enable weight quantization
            act_quant: Enable activation quantization
        """
        for module in self.model.modules():
            if isinstance(module, QuantLayerMoDiff):
                module.set_quant_state(weight_quant, act_quant)
    
    def set_modulation(self, modulate: bool) -> None:
        """Enable or disable MoDiff modulation for all layers."""
        self.modulate = modulate
        for module in self.model.modules():
            if isinstance(module, QuantLayerMoDiff):
                module.set_modulation(modulate)
    
    def calibrate_weights(self) -> None:
        """
        Calibrate weight quantizers for all layers.
        
        This uses MSE-based scale search on the weight tensors.
        Should be called once before inference.
        """
        logger.info("Calibrating weight quantizers...")
        for name, module in self.model.named_modules():
            if isinstance(module, QuantLayerMoDiff):
                module.calibrate_weight()
        logger.info("Weight calibration complete")
    
    def calibrate_activations(
        self, 
        calib_data: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int = 32,
    ) -> None:
        """
        Calibrate activation quantizers using calibration data.
        
        For MoDiff, calibration should ideally use RESIDUALS between
        adjacent timesteps, not raw activations.
        
        Args:
            calib_data: Tuple of (inputs, timesteps)
            batch_size: Batch size for calibration forward passes
        """
        logger.info("Calibrating activation quantizers...")
        
        xs, ts = calib_data
        device = next(self.model.parameters()).device
        
        # Run a few forward passes to calibrate
        self.set_quant_state(True, False)  # Weight quant only first
        
        with torch.no_grad():
            for i in range(0, min(len(xs), batch_size * 4), batch_size):
                batch_x = xs[i:i+batch_size].to(device)
                batch_t = ts[i:i+batch_size].to(device)
                
                self.reset_cache()
                _ = self.model(batch_x, batch_t)
        
        logger.info("Activation calibration complete")
    
    def set_running_stat(self, running_stat: bool) -> None:
        """Enable/disable running statistics for activation quantizers."""
        for module in self.model.modules():
            if isinstance(module, QuantLayerMoDiff):
                module.act_quantizer.running_stat = running_stat
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the quantized model.
        
        Args:
            x: Input tensor (noisy image)
            timesteps: Diffusion timesteps
            context: Optional conditioning context
            
        Returns:
            Model output (predicted noise)
        """
        if context is not None:
            return self.model(x, timesteps, context)
        else:
            return self.model(x, timesteps)
    
    def get_quant_layers(self) -> List[QuantLayerMoDiff]:
        """Get list of all quantized layers."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, QuantLayerMoDiff):
                layers.append(module)
        return layers
    
    def num_quant_layers(self) -> int:
        """Get number of quantized layers."""
        return len(self.get_quant_layers())
    
    def print_quant_summary(self) -> None:
        """Print summary of quantized layers."""
        layers = self.get_quant_layers()
        print(f"\n{'='*60}")
        print(f"Quantized Model Summary")
        print(f"{'='*60}")
        print(f"Total quantized layers: {len(layers)}")
        print(f"Weight bits: {self.weight_bits}")
        print(f"Activation bits: {self.act_bits}")
        print(f"MoDiff modulation: {self.modulate}")
        print(f"Native INT8: {self.use_native_int8}")
        print(f"{'='*60}\n")
        
        # Count by layer type
        conv_count = sum(1 for l in layers if l.layer_type == 'conv2d')
        linear_count = sum(1 for l in layers if l.layer_type == 'linear')
        print(f"Conv2d layers: {conv_count}")
        print(f"Linear layers: {linear_count}")


def create_quant_model(
    model: nn.Module,
    weight_bits: int = 8,
    act_bits: int = 8,
    modulate: bool = True,
    use_native_int8: bool = True,
) -> QuantModelMoDiff:
    """
    Convenience function to create a quantized model with MoDiff.
    
    Args:
        model: Original model
        weight_bits: Weight quantization bits
        act_bits: Activation quantization bits
        modulate: Enable MoDiff
        use_native_int8: Use native INT8 ops
        
    Returns:
        QuantModelMoDiff instance
    """
    return QuantModelMoDiff(
        model,
        weight_bits=weight_bits,
        act_bits=act_bits,
        modulate=modulate,
        use_native_int8=use_native_int8,
    )
