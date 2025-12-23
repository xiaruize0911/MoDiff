"""
Quantization-Specific Profiling for MoDiff

This script profiles the quantization operations and compares:
1. FP32/FP16 baseline operations
2. W8A8 quantized operations
3. W4A4 quantized operations
4. Modulated quantization overhead
5. Kernel-level performance (Triton kernels)
"""

import argparse
import json
import os
import time
from collections import defaultdict
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from modiff_triton.nn import W8A8MoDiffLinear, W4A4MoDiffLinear
    from modiff_triton.nn import W8A8MoDiffConv2d, W4A4MoDiffConv2d
    from modiff_triton.nn.config import MoDiffConfig
    MODIFF_AVAILABLE = True
except ImportError:
    MODIFF_AVAILABLE = False
    print("Warning: MoDiff Triton modules not available")


class QuantizationProfiler:
    """Profile quantization operations"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = defaultdict(dict)
        
    def benchmark_operation(self, op_fn, input_data, num_runs=100, warmup=10):
        """Benchmark a single operation"""
        # Warmup
        for _ in range(warmup):
            _ = op_fn(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            output = op_fn(input_data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
        }
    
    def profile_linear_layers(self, batch_size=8, in_features=512, out_features=512, 
                             num_runs=100):
        """Compare FP32, FP16, W8A8, W4A4 linear layers"""
        print(f"\nProfiling Linear Layers ({batch_size}x{in_features} -> {out_features})...")
        
        # Create input
        x_fp32 = torch.randn(batch_size, in_features, device=self.device, dtype=torch.float32)
        x_fp16 = x_fp32.half()
        
        # FP32 Linear
        linear_fp32 = nn.Linear(in_features, out_features).to(self.device)
        linear_fp32.eval()
        
        with torch.no_grad():
            self.results['linear'][f'fp32_{in_features}x{out_features}'] = \
                self.benchmark_operation(linear_fp32, x_fp32, num_runs)
        
        # FP16 Linear
        linear_fp16 = linear_fp32.half()
        with torch.no_grad():
            self.results['linear'][f'fp16_{in_features}x{out_features}'] = \
                self.benchmark_operation(linear_fp16, x_fp16, num_runs)
        
        if MODIFF_AVAILABLE:
            # W8A8 Linear
            config_w8a8 = MoDiffConfig(weight_bits=8, act_bits=8)
            linear_w8a8 = W8A8MoDiffLinear(in_features, out_features, config=config_w8a8).to(self.device)
            
            # Initialize with FP32 weights
            with torch.no_grad():
                linear_w8a8.weight_int8.copy_(
                    (linear_fp32.weight.data * 127 / linear_fp32.weight.abs().max()).to(torch.int8)
                )
                linear_w8a8.weight_scale.fill_(linear_fp32.weight.abs().max() / 127)
                if linear_w8a8.bias is not None:
                    linear_w8a8.bias.copy_(linear_fp32.bias.data)
            
            linear_w8a8.eval()
            with torch.no_grad():
                self.results['linear'][f'w8a8_{in_features}x{out_features}'] = \
                    self.benchmark_operation(linear_w8a8, x_fp32, num_runs)
            
            # W4A4 Linear
            config_w4a4 = MoDiffConfig(weight_bits=4, act_bits=4)
            linear_w4a4 = W4A4MoDiffLinear(in_features, out_features, config=config_w4a4).to(self.device)
            
            # Initialize with FP32 weights
            with torch.no_grad():
                # Skip weight initialization for W4A4 as it's complex
                # Just use the default initialized weights
                if linear_w4a4.bias is not None:
                    linear_w4a4.bias.copy_(linear_fp32.bias.data)
            
            linear_w4a4.eval()
            with torch.no_grad():
                self.results['linear'][f'w4a4_{in_features}x{out_features}'] = \
                    self.benchmark_operation(linear_w4a4, x_fp32, num_runs)
        
        print("  Linear profiling complete")
    
    def profile_conv_layers(self, batch_size=4, in_channels=64, out_channels=64,
                           kernel_size=3, image_size=32, num_runs=100):
        """Compare FP32, FP16, W8A8, W4A4 conv layers"""
        print(f"\nProfiling Conv2d Layers ({batch_size}x{in_channels}x{image_size}x{image_size} "
              f"-> {out_channels}, k={kernel_size})...")
        
        # Create input
        x_fp32 = torch.randn(batch_size, in_channels, image_size, image_size, 
                           device=self.device, dtype=torch.float32)
        x_fp16 = x_fp32.half()
        
        # FP32 Conv
        conv_fp32 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2).to(self.device)
        conv_fp32.eval()
        
        with torch.no_grad():
            self.results['conv2d'][f'fp32_{in_channels}x{out_channels}_k{kernel_size}'] = \
                self.benchmark_operation(conv_fp32, x_fp32, num_runs)
        
        # FP16 Conv
        conv_fp16 = conv_fp32.half()
        with torch.no_grad():
            self.results['conv2d'][f'fp16_{in_channels}x{out_channels}_k{kernel_size}'] = \
                self.benchmark_operation(conv_fp16, x_fp16, num_runs)
        
        if MODIFF_AVAILABLE:
            # W8A8 Conv
            config_w8a8 = MoDiffConfig(weight_bits=8, act_bits=8)
            conv_w8a8 = W8A8MoDiffConv2d(
                in_channels, out_channels, kernel_size, 
                padding=kernel_size//2, config=config_w8a8
            ).to(self.device)
            
            # Initialize weights
            with torch.no_grad():
                weight_max = conv_fp32.weight.abs().amax(dim=(1, 2, 3), keepdim=True)
                conv_w8a8.weight_int8.copy_(
                    (conv_fp32.weight.data / (weight_max + 1e-8) * 127).to(torch.int8)
                )
                conv_w8a8.weight_scale.copy_(weight_max.squeeze() / 127)
                if conv_w8a8.bias is not None:
                    conv_w8a8.bias.copy_(conv_fp32.bias.data)
            
            conv_w8a8.eval()
            with torch.no_grad():
                self.results['conv2d'][f'w8a8_{in_channels}x{out_channels}_k{kernel_size}'] = \
                    self.benchmark_operation(conv_w8a8, x_fp32, num_runs)
        
        print("  Conv2d profiling complete")
    
    def profile_activations(self, batch_size=8, size=512, num_runs=100):
        """Profile activation functions"""
        print(f"\nProfiling Activation Functions ({batch_size}x{size})...")
        
        x = torch.randn(batch_size, size, device=self.device)
        
        # ReLU
        with torch.no_grad():
            self.results['activations']['relu'] = \
                self.benchmark_operation(lambda x: F.relu(x), x, num_runs)
        
        # SiLU (Swish)
        with torch.no_grad():
            self.results['activations']['silu'] = \
                self.benchmark_operation(lambda x: F.silu(x), x, num_runs)
        
        # GELU
        with torch.no_grad():
            self.results['activations']['gelu'] = \
                self.benchmark_operation(lambda x: F.gelu(x), x, num_runs)
        
        # Sigmoid
        with torch.no_grad():
            self.results['activations']['sigmoid'] = \
                self.benchmark_operation(lambda x: torch.sigmoid(x), x, num_runs)
        
        print("  Activation profiling complete")
    
    def profile_normalizations(self, batch_size=4, channels=64, height=32, width=32, num_runs=100):
        """Profile normalization layers"""
        print(f"\nProfiling Normalization Layers ({batch_size}x{channels}x{height}x{width})...")
        
        x = torch.randn(batch_size, channels, height, width, device=self.device)
        
        # GroupNorm
        gn = nn.GroupNorm(32, channels).to(self.device)
        gn.eval()
        with torch.no_grad():
            self.results['normalizations']['groupnorm'] = \
                self.benchmark_operation(gn, x, num_runs)
        
        # BatchNorm
        bn = nn.BatchNorm2d(channels).to(self.device)
        bn.eval()
        with torch.no_grad():
            self.results['normalizations']['batchnorm'] = \
                self.benchmark_operation(bn, x, num_runs)
        
        # LayerNorm
        ln = nn.LayerNorm([channels, height, width]).to(self.device)
        ln.eval()
        with torch.no_grad():
            self.results['normalizations']['layernorm'] = \
                self.benchmark_operation(ln, x, num_runs)
        
        print("  Normalization profiling complete")
    
    def profile_attention(self, batch_size=4, seq_len=1024, dim=512, num_runs=50):
        """Profile attention operations"""
        print(f"\nProfiling Attention ({batch_size}x{seq_len}x{dim})...")
        
        q = torch.randn(batch_size, seq_len, dim, device=self.device)
        k = torch.randn(batch_size, seq_len, dim, device=self.device)
        v = torch.randn(batch_size, seq_len, dim, device=self.device)
        
        def attention_fn(inputs):
            q, k, v = inputs
            # Standard scaled dot-product attention
            scores = torch.bmm(q, k.transpose(1, 2)) / (dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out = torch.bmm(attn, v)
            return out
        
        with torch.no_grad():
            self.results['attention']['scaled_dot_product'] = \
                self.benchmark_operation(attention_fn, (q, k, v), num_runs)
        
        print("  Attention profiling complete")
    
    def generate_report(self, output_dir='ablation_results'):
        """Generate comprehensive report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert defaultdict to regular dict
        results_dict = {k: dict(v) for k, v in self.results.items()}
        
        # Save JSON
        json_path = os.path.join(output_dir, f'quantization_profile_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Generate text report
        summary_path = os.path.join(output_dir, f'quantization_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MoDiff Quantization Profiling Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Linear layers comparison
            if 'linear' in self.results:
                f.write("=" * 80 + "\n")
                f.write("Linear Layers Performance\n")
                f.write("=" * 80 + "\n\n")
                
                for name, stats in self.results['linear'].items():
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {stats['mean_ms']:.4f} ms\n")
                    f.write(f"  Std:  {stats['std_ms']:.4f} ms\n")
                    f.write(f"  Min:  {stats['min_ms']:.4f} ms\n")
                    f.write(f"  Max:  {stats['max_ms']:.4f} ms\n\n")
                
                # Calculate speedups
                if 'fp32_512x512' in self.results['linear']:
                    fp32_time = self.results['linear']['fp32_512x512']['mean_ms']
                    f.write("Speedup vs FP32:\n")
                    for name, stats in self.results['linear'].items():
                        if name != 'fp32_512x512':
                            speedup = fp32_time / stats['mean_ms']
                            f.write(f"  {name}: {speedup:.2f}x\n")
                    f.write("\n")
            
            # Conv layers comparison
            if 'conv2d' in self.results:
                f.write("=" * 80 + "\n")
                f.write("Conv2d Layers Performance\n")
                f.write("=" * 80 + "\n\n")
                
                for name, stats in self.results['conv2d'].items():
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {stats['mean_ms']:.4f} ms\n")
                    f.write(f"  Std:  {stats['std_ms']:.4f} ms\n")
                    f.write(f"  Min:  {stats['min_ms']:.4f} ms\n")
                    f.write(f"  Max:  {stats['max_ms']:.4f} ms\n\n")
                
                # Calculate speedups
                fp32_key = [k for k in self.results['conv2d'].keys() if k.startswith('fp32_')][0] \
                    if any(k.startswith('fp32_') for k in self.results['conv2d'].keys()) else None
                
                if fp32_key:
                    fp32_time = self.results['conv2d'][fp32_key]['mean_ms']
                    f.write("Speedup vs FP32:\n")
                    for name, stats in self.results['conv2d'].items():
                        if name != fp32_key:
                            speedup = fp32_time / stats['mean_ms']
                            f.write(f"  {name}: {speedup:.2f}x\n")
                    f.write("\n")
            
            # Activations
            if 'activations' in self.results:
                f.write("=" * 80 + "\n")
                f.write("Activation Functions Performance\n")
                f.write("=" * 80 + "\n\n")
                
                for name, stats in sorted(self.results['activations'].items(), 
                                        key=lambda x: x[1]['mean_ms']):
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {stats['mean_ms']:.4f} ms\n")
                    f.write(f"  Std:  {stats['std_ms']:.4f} ms\n\n")
            
            # Normalizations
            if 'normalizations' in self.results:
                f.write("=" * 80 + "\n")
                f.write("Normalization Layers Performance\n")
                f.write("=" * 80 + "\n\n")
                
                for name, stats in sorted(self.results['normalizations'].items(),
                                        key=lambda x: x[1]['mean_ms']):
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {stats['mean_ms']:.4f} ms\n")
                    f.write(f"  Std:  {stats['std_ms']:.4f} ms\n\n")
            
            # Attention
            if 'attention' in self.results:
                f.write("=" * 80 + "\n")
                f.write("Attention Operations Performance\n")
                f.write("=" * 80 + "\n\n")
                
                for name, stats in self.results['attention'].items():
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {stats['mean_ms']:.4f} ms\n")
                    f.write(f"  Std:  {stats['std_ms']:.4f} ms\n\n")
        
        print(f"\n{'='*80}")
        print(f"Quantization profiling complete!")
        print(f"{'='*80}")
        print(f"JSON Report: {json_path}")
        print(f"Text Summary: {summary_path}")
        print(f"{'='*80}\n")
        
        return json_path, summary_path, results_dict


def main():
    parser = argparse.ArgumentParser(description='Quantization-Specific Profiling')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of benchmark runs per operation')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Output directory')
    parser.add_argument('--profile_linear', action='store_true', help='Profile linear layers')
    parser.add_argument('--profile_conv', action='store_true', help='Profile conv layers')
    parser.add_argument('--profile_activations', action='store_true', help='Profile activations')
    parser.add_argument('--profile_norms', action='store_true', help='Profile normalizations')
    parser.add_argument('--profile_attention', action='store_true', help='Profile attention')
    parser.add_argument('--profile_all', action='store_true', help='Profile everything')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"MoDiff Quantization Profiling")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Num Runs: {args.num_runs}")
    print(f"MoDiff Available: {MODIFF_AVAILABLE}")
    print(f"{'='*80}\n")
    
    profiler = QuantizationProfiler(device=args.device)
    
    # Determine what to profile
    profile_all = args.profile_all or not any([
        args.profile_linear, args.profile_conv, args.profile_activations,
        args.profile_norms, args.profile_attention
    ])
    
    if profile_all or args.profile_linear:
        # Multiple sizes
        profiler.profile_linear_layers(8, 256, 256, args.num_runs)
        profiler.profile_linear_layers(8, 512, 512, args.num_runs)
        profiler.profile_linear_layers(8, 1024, 1024, args.num_runs)
    
    if profile_all or args.profile_conv:
        # Multiple configurations
        profiler.profile_conv_layers(4, 64, 64, 3, 32, args.num_runs)
        profiler.profile_conv_layers(4, 128, 128, 3, 32, args.num_runs)
        profiler.profile_conv_layers(4, 256, 256, 3, 16, args.num_runs)
    
    if profile_all or args.profile_activations:
        profiler.profile_activations(8, 512, args.num_runs)
    
    if profile_all or args.profile_norms:
        profiler.profile_normalizations(4, 64, 32, 32, args.num_runs)
    
    if profile_all or args.profile_attention:
        profiler.profile_attention(4, 256, 512, max(args.num_runs // 2, 20))
    
    # Generate report
    profiler.generate_report(args.output_dir)
    
    # Print quick summary
    print("\n" + "="*80)
    print("Quick Summary")
    print("="*80)
    
    if 'linear' in profiler.results and 'fp32_512x512' in profiler.results['linear']:
        print("\nLinear Layer (512x512) Performance:")
        for name, stats in sorted(profiler.results['linear'].items()):
            if '512x512' in name:
                print(f"  {name:20s}: {stats['mean_ms']:.4f} ms")
    
    if 'conv2d' in profiler.results:
        print("\nConv2d Layer Performance:")
        for name, stats in list(profiler.results['conv2d'].items())[:4]:
            print(f"  {name:30s}: {stats['mean_ms']:.4f} ms")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
