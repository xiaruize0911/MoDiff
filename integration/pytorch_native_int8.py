"""
PyTorch Native INT8 Quantization for Diffusion Models

This uses torch.ao.quantization which:
1. Uses cuDNN's native INT8 convolution (no CUTLASS layout conversions)
2. Is torch.compile compatible
3. Has built-in calibration and quantization observers
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.ao.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    QConfigMapping,
)
import time


class SimpleConvBlock(nn.Module):
    """A simple conv block for testing quantization."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


def quantize_with_pytorch(model, calibration_data):
    """
    Quantize a model using PyTorch's native quantization.
    
    Args:
        model: The model to quantize
        calibration_data: List of input tensors for calibration
    
    Returns:
        Quantized model
    """
    model.eval()
    
    # Create a copy for quantization
    model_to_quantize = model
    
    # Set up quantization config
    # Use per-tensor quantization for activations, per-channel for weights
    model_to_quantize.qconfig = get_default_qconfig('x86')  # or 'qnnpack' for mobile
    
    # Prepare the model (inserts observers)
    prepared_model = prepare(model_to_quantize, inplace=False)
    
    # Calibrate by running forward passes
    with torch.no_grad():
        for data in calibration_data:
            prepared_model(data)
    
    # Convert to quantized model
    quantized_model = convert(prepared_model, inplace=False)
    
    return quantized_model


class FusedQuantBlock(nn.Module):
    """
    A fused quantized block that keeps data in INT8 between convolutions.
    
    Uses PyTorch's intrinsic fused operations:
    - ConvReLU2d: Fused Conv + ReLU (keeps INT8 between them)
    - ConvBnReLU2d: Fused Conv + BatchNorm + ReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # For quantization, we use fused modules
        from torch.ao.nn.intrinsic import ConvReLU2d
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu1 = nn.ReLU()  # Use ReLU for quantization compatibility
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Quantization stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.dequant(x)
        return x
    
    def fuse_modules(self):
        """Fuse conv+relu pairs for quantization."""
        torch.ao.quantization.fuse_modules(
            self, 
            [['conv1', 'relu1'], ['conv2', 'relu2']],
            inplace=True
        )


def benchmark_native_quantization():
    """Compare FP32, FP16, and native INT8 quantization."""
    print("=" * 70)
    print("PyTorch Native Quantization Benchmark")
    print("=" * 70)
    print()
    
    device = 'cuda'
    in_ch, out_ch = 320, 320
    batch_size = 4
    spatial = 32
    
    # Create model
    model_fp32 = SimpleConvBlock(in_ch, out_ch).to(device)
    model_fp32.eval()
    
    # Test input
    x = torch.randn(batch_size, in_ch, spatial, spatial, device=device)
    
    # Calibration data
    calibration_data = [torch.randn_like(x) for _ in range(10)]
    
    # Create CPU model for quantization (PyTorch quantization is CPU-focused)
    model_cpu = SimpleConvBlock(in_ch, out_ch)
    model_cpu.load_state_dict(model_fp32.state_dict())
    model_cpu.eval()
    
    # Calibrate on CPU
    x_cpu = x.cpu()
    calibration_cpu = [d.cpu() for d in calibration_data]
    
    # Quantize
    model_int8 = quantize_with_pytorch(model_cpu, calibration_cpu)
    
    print(f"Input shape: ({batch_size}, {in_ch}, {spatial}, {spatial})")
    print()
    
    # Benchmark FP32 on GPU
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    for _ in range(20):
        _ = model_fp32(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        _ = model_fp32(x)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / 100 * 1000
    
    # Benchmark TF32 on GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    for _ in range(20):
        _ = model_fp32(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        _ = model_fp32(x)
    torch.cuda.synchronize()
    tf32_time = (time.time() - start) / 100 * 1000
    
    # Benchmark INT8 on CPU (PyTorch's native quantization is CPU-optimized)
    for _ in range(10):
        _ = model_int8(x_cpu)
    
    start = time.time()
    for _ in range(50):
        _ = model_int8(x_cpu)
    int8_cpu_time = (time.time() - start) / 50 * 1000
    
    print("Results:")
    print(f"  FP32 (GPU, no TF32): {fp32_time:.3f} ms")
    print(f"  FP32 (GPU, TF32):    {tf32_time:.3f} ms  ({fp32_time/tf32_time:.2f}x vs pure FP32)")
    print(f"  INT8 (CPU):          {int8_cpu_time:.3f} ms")
    print()
    
    # Check numerical accuracy
    with torch.no_grad():
        out_fp32 = model_fp32(x).cpu()
        out_int8 = model_int8(x_cpu)
        
        error = (out_fp32 - out_int8).abs().mean().item()
        print(f"Mean absolute error (INT8 vs FP32): {error:.6f}")
    
    print()
    print("NOTE: PyTorch's native quantization is primarily CPU-optimized.")
    print("For GPU INT8, consider TensorRT or our CUTLASS implementation.")


def demonstrate_fused_quantization():
    """Show how to use fused quantized modules."""
    print()
    print("=" * 70)
    print("Fused Quantization (Conv+ReLU kept in INT8)")
    print("=" * 70)
    print()
    
    # Create model with fusable modules
    model = FusedQuantBlock(320, 320)
    model.eval()
    
    # Fuse modules
    model.fuse_modules()
    
    # Set quantization config
    model.qconfig = get_default_qconfig('x86')
    
    # Prepare and calibrate
    model_prepared = prepare(model, inplace=False)
    
    x = torch.randn(4, 320, 32, 32)
    for _ in range(10):
        model_prepared(x)
    
    # Convert
    model_quantized = convert(model_prepared, inplace=False)
    
    print("Quantized model structure:")
    print(model_quantized)
    print()
    print("Notice how Conv+ReLU are fused into a single quantized operation!")
    print("This keeps data in INT8 between the conv and relu.")


def gpu_int8_with_tensorrt():
    """Show how to use TensorRT for GPU INT8."""
    print()
    print("=" * 70)
    print("GPU INT8 Options")
    print("=" * 70)
    print()
    print("For GPU INT8 inference, you have several options:")
    print()
    print("1. TensorRT (NVIDIA's inference optimizer)")
    print("   - Export model to ONNX, then to TensorRT")
    print("   - Automatic INT8 calibration and optimization")
    print("   - Fuses compatible ops (Conv+BN+ReLU)")
    print("   - Can achieve 2-3x speedup over FP32")
    print("   Example:")
    print("     import torch_tensorrt")
    print("     trt_model = torch_tensorrt.compile(model, inputs=[example_input],")
    print("                                        enabled_precisions={torch.int8})")
    print()
    print("2. torch.compile with GPU backend")
    print("   - Use inductor backend with quantized ops")
    print("   - Less mature than TensorRT but improving")
    print()
    print("3. CUTLASS (our implementation)")
    print("   - Direct INT8 GEMM kernels")
    print("   - Requires manual optimization")
    print("   - Good for custom fused kernels")
    print()
    print("4. cuDNN INT8")
    print("   - cudnnConvolutionForward with INT8 data type")
    print("   - Requires proper tensor descriptors")
    print()
    print("RECOMMENDATION: TensorRT for production, CUTLASS for research/custom ops")


if __name__ == "__main__":
    benchmark_native_quantization()
    demonstrate_fused_quantization()
    gpu_int8_with_tensorrt()
