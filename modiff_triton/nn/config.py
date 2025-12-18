"""
MoDiff Configuration

This module contains configuration classes for MoDiff quantized layers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoDiffConfig:
    """
    Configuration for MoDiff quantized layers.
    
    Attributes:
        weight_bits: Number of bits for weight quantization (4 or 8)
        act_bits: Number of bits for activation quantization (4 or 8)
        symmetric: Whether to use symmetric quantization
        weight_channel_wise: Whether to use per-channel quantization for weights
        act_dynamic: Whether to use dynamic quantization for activations
        modulation_enabled: Whether to enable MoDiff error-compensated modulation
        use_accumulation: Whether to use MoDiff output accumulation (Eq. ec6)
        store_cache_fp16: Whether to store caches in FP16 to save memory
        eps: Small constant for numerical stability
    """
    weight_bits: int = 8
    act_bits: int = 8
    symmetric: bool = True
    weight_channel_wise: bool = True
    act_dynamic: bool = True
    modulation_enabled: bool = True
    use_accumulation: bool = True
    store_cache_fp16: bool = True
    eps: float = 1e-8
    
    def __post_init__(self):
        assert self.weight_bits in [4, 8], f"weight_bits must be 4 or 8, got {self.weight_bits}"
        assert self.act_bits in [4, 8], f"act_bits must be 4 or 8, got {self.act_bits}"


@dataclass  
class W8A8Config(MoDiffConfig):
    """Configuration for W8A8 (INT8 weights, INT8 activations) quantization."""
    weight_bits: int = 8
    act_bits: int = 8


@dataclass
class W4A4Config(MoDiffConfig):
    """Configuration for W4A4 (INT4 weights, INT4 activations) quantization."""
    weight_bits: int = 4
    act_bits: int = 4


# Default configurations
DEFAULT_W8A8_CONFIG = W8A8Config()
DEFAULT_W4A4_CONFIG = W4A4Config()
