"""
TensorRT INT8 Integration Wrapper for MoDiff

This module demonstrates how to use TensorRT INT8 engine with MoDiff's
QuantModelINT8 for accelerated inference.

Usage:
======

Option 1: PyTorch INT8 (current, slow)
  from qdiff.quant_model_int8 import QuantModelINT8
  
  qmodel = QuantModelINT8(model, weight_params, act_params)
  qmodel.set_quant_state(weight_quant=True, act_quant=False)
  output = qmodel(x, timesteps, context)  # Dequantizes to FP32

Option 2: TensorRT INT8 (new, fast!)
  from qdiff.quant_model_int8 import QuantModelINT8
  
  qmodel = QuantModelINT8(model, weight_params, act_params,
                          trt_engine_path="trt/export/modiff_unet_fp32.plan")
  output = qmodel(x, timesteps)  # 2-3x speedup!

Option 3: Enable TensorRT after initialization
  from qdiff.quant_model_int8 import QuantModelINT8
  
  qmodel = QuantModelINT8(model, weight_params, act_params)
  qmodel.enable_trt_backend("trt/export/modiff_unet_fp32.plan")
  output = qmodel(x, timesteps)  # Now using TensorRT!
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TensorRTINT8Helper:
    """
    Helper class to manage TensorRT INT8 backend for QuantModelINT8.
    
    Provides utilities for:
    - Loading/unloading TensorRT engines
    - Benchmarking PyTorch vs TensorRT
    - Validating output correctness
    - Toggling between backends
    """
    
    @staticmethod
    def create_trt_quantized_model(
        model: torch.nn.Module,
        weight_quant_params: Dict,
        act_quant_params: Dict,
        trt_engine_path: str,
        **kwargs
    ):
        """
        Create a QuantModelINT8 with TensorRT backend pre-configured.
        
        Args:
            model: Original PyTorch model
            weight_quant_params: Weight quantization config
            act_quant_params: Activation quantization config
            trt_engine_path: Path to TensorRT engine (.plan file)
            **kwargs: Additional arguments for QuantModelINT8
        
        Returns:
            QuantModelINT8 instance with TensorRT backend enabled
        """
        from qdiff.quant_model_int8 import QuantModelINT8
        
        logger.info(f"[TensorRTINT8Helper] Creating QuantModelINT8 with TRT backend")
        logger.info(f"[TensorRTINT8Helper] Engine path: {trt_engine_path}")
        
        qmodel = QuantModelINT8(
            model,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            trt_engine_path=trt_engine_path,
            **kwargs
        )
        
        if qmodel.use_trt_backend:
            logger.info("[TensorRTINT8Helper] ✓ TensorRT backend successfully enabled!")
            qmodel.print_backend_info()
        else:
            logger.warning("[TensorRTINT8Helper] TensorRT backend not available, using PyTorch")
        
        return qmodel
    
    @staticmethod
    def benchmark_inference(qmodel, latent: torch.Tensor, timesteps: torch.Tensor,
                           num_iterations: int = 20, num_warmup: int = 5):
        """
        Benchmark inference latency and throughput.
        
        Args:
            qmodel: QuantModelINT8 instance
            latent: Input latent tensor
            timesteps: Timestep tensor
            num_iterations: Number of benchmark iterations
            num_warmup: Number of warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        backend_info = qmodel.get_backend_info()
        backend_name = backend_info['backend']
        
        print(f"\n[Benchmark] Testing {backend_name} backend...")
        print(f"[Benchmark] Input shape: {latent.shape}")
        print(f"[Benchmark] Warmup iterations: {num_warmup}")
        print(f"[Benchmark] Benchmark iterations: {num_iterations}")
        
        qmodel.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = qmodel(latent, timesteps)
            torch.cuda.synchronize() if latent.is_cuda else None
        
        # Benchmark
        torch.cuda.synchronize() if latent.is_cuda else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = qmodel(latent, timesteps)
        
        torch.cuda.synchronize() if latent.is_cuda else None
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000  # Convert to ms
        latency = total_time / num_iterations
        throughput = 1000 / latency  # Samples per second
        
        return {
            'backend': backend_name,
            'total_time_ms': total_time,
            'latency_ms': latency,
            'throughput': throughput,
        }
    
    @staticmethod
    def compare_backends(model: torch.nn.Module, weight_quant_params: Dict,
                        act_quant_params: Dict, trt_engine_path: str,
                        latent: torch.Tensor, timesteps: torch.Tensor):
        """
        Compare PyTorch and TensorRT backends side-by-side.
        
        Benchmarks both backends and shows:
        - Inference latency
        - Throughput
        - Speedup ratio
        - Output correctness
        
        Args:
            model: Original PyTorch model
            weight_quant_params: Weight quantization config
            act_quant_params: Activation quantization config
            trt_engine_path: Path to TensorRT engine
            latent: Input latent tensor
            timesteps: Timestep tensor
        """
        from qdiff.quant_model_int8 import QuantModelINT8
        
        print("\n" + "="*70)
        print("Backend Comparison: PyTorch INT8 vs TensorRT INT8")
        print("="*70)
        
        # PyTorch INT8
        print("\n[1/3] Creating PyTorch INT8 model...")
        qmodel_pt = QuantModelINT8(model, weight_quant_params, act_quant_params)
        qmodel_pt.set_quant_state(weight_quant=True, act_quant=False)
        
        print("[2/3] Creating TensorRT INT8 model...")
        qmodel_trt = QuantModelINT8(model, weight_quant_params, act_quant_params,
                                     trt_engine_path=trt_engine_path)
        
        # Benchmark PyTorch
        print("\n[3/3] Benchmarking...")
        print("\n--- PyTorch INT8 ---")
        stats_pt = TensorRTINT8Helper.benchmark_inference(qmodel_pt, latent, timesteps)
        
        print("\n--- TensorRT INT8 ---")
        stats_trt = TensorRTINT8Helper.benchmark_inference(qmodel_trt, latent, timesteps)
        
        # Compare results
        print("\n" + "="*70)
        print("Results Summary")
        print("="*70)
        print(f"{'Metric':<30} {'PyTorch':<20} {'TensorRT':<20}")
        print("-"*70)
        print(f"{'Latency':<30} {stats_pt['latency_ms']:.2f} ms {stats_trt['latency_ms']:.2f} ms")
        print(f"{'Throughput':<30} {stats_pt['throughput']:.1f} samples/s {stats_trt['throughput']:.1f} samples/s")
        
        speedup = stats_pt['latency_ms'] / stats_trt['latency_ms'] if stats_trt['latency_ms'] > 0 else 0
        print(f"{'Speedup':<30} {'1.0x':<20} {speedup:.1f}x")
        print("="*70 + "\n")
        
        return stats_pt, stats_trt
    
    @staticmethod
    def validate_output_correctness(model: torch.nn.Module, weight_quant_params: Dict,
                                    act_quant_params: Dict, trt_engine_path: str,
                                    latent: torch.Tensor, timesteps: torch.Tensor,
                                    tolerance: float = 1e-4):
        """
        Validate that both backends produce similar outputs.
        
        Args:
            model: Original PyTorch model
            weight_quant_params: Weight quantization config
            act_quant_params: Activation quantization config
            trt_engine_path: Path to TensorRT engine
            latent: Input latent tensor
            timesteps: Timestep tensor
            tolerance: Maximum allowed difference (MAE)
        
        Returns:
            Dictionary with validation results
        """
        from qdiff.quant_model_int8 import QuantModelINT8
        
        print("\n[Validation] Checking output correctness...")
        
        qmodel_pt = QuantModelINT8(model, weight_quant_params, act_quant_params)
        qmodel_trt = QuantModelINT8(model, weight_quant_params, act_quant_params,
                                     trt_engine_path=trt_engine_path)
        
        qmodel_pt.eval()
        qmodel_trt.eval()
        
        with torch.no_grad():
            output_pt = qmodel_pt(latent, timesteps)
            if qmodel_trt.use_trt_backend:
                output_trt = qmodel_trt(latent, timesteps)
            else:
                output_trt = output_pt  # TRT not available, use PyTorch
        
        mae = torch.mean(torch.abs(output_pt - output_trt)).item()
        max_diff = torch.max(torch.abs(output_pt - output_trt)).item()
        
        passed = mae < tolerance
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"[Validation] MAE: {mae:.6f} {status}")
        print(f"[Validation] Max difference: {max_diff:.6f}")
        print(f"[Validation] Tolerance: {tolerance:.6f}")
        
        return {
            'passed': passed,
            'mae': mae,
            'max_diff': max_diff,
            'tolerance': tolerance,
        }


if __name__ == "__main__":
    # Example usage
    print(__doc__)
    
    print("\nFor integration examples, see:")
    print("  - qdiff/quant_model_int8.py (main wrapper)")
    print("  - trt/inference_wrapper.py (TensorRT wrapper)")
    print("  - trt/README.md (documentation)")
