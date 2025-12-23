"""
Comprehensive Ablation Study and Profiling for MoDiff

This script performs detailed profiling of:
1. Individual layers (Conv2d, Linear, ResNet blocks, Attention blocks)
2. Model sections (Downsampling, Middle, Upsampling)
3. Operations (Convolutions, Activations, Normalizations, Quantizations)
4. Full forward pass timing
5. Memory usage analysis

Results are saved to a detailed JSON report and visualizations.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import yaml

# For hooking into model layers
from torch.utils.hooks import RemovableHandle


class ProfilerHook:
    """Hook to profile individual module execution time and memory"""
    
    def __init__(self, module_name: str, profile_data: dict):
        self.module_name = module_name
        self.profile_data = profile_data
        self.start_time = None
        self.start_memory = None
        
    def pre_hook(self, module, input):
        """Called before module forward pass"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        self.start_time = time.perf_counter()
        
    def post_hook(self, module, input, output):
        """Called after module forward pass"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        elapsed = (end_time - self.start_time) * 1000  # Convert to ms
        
        # Record timing
        if self.module_name not in self.profile_data['timings']:
            self.profile_data['timings'][self.module_name] = []
        self.profile_data['timings'][self.module_name].append(elapsed)
        
        # Record memory if CUDA
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_delta = (end_memory - self.start_memory) / 1024 / 1024  # MB
            if self.module_name not in self.profile_data['memory']:
                self.profile_data['memory'][self.module_name] = []
            self.profile_data['memory'][self.module_name].append(memory_delta)


class ModelProfiler:
    """Main profiling class for diffusion models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.hooks: List[RemovableHandle] = []
        self.profile_data = {
            'timings': {},
            'memory': {},
            'layer_types': {},
            'layer_params': {},
        }
        
    def register_hooks(self, granularity='all'):
        """
        Register profiling hooks on model modules
        
        Args:
            granularity: 'all', 'blocks', 'operations'
                - all: Profile every leaf module
                - blocks: Profile ResNet blocks, Attention blocks, etc.
                - operations: Profile only specific operation types
        """
        self.remove_hooks()
        
        for name, module in self.model.named_modules():
            should_profile = False
            
            if granularity == 'all':
                # Profile all leaf modules (those without children)
                if len(list(module.children())) == 0:
                    should_profile = True
            elif granularity == 'blocks':
                # Profile high-level blocks
                if any(block_type in name for block_type in 
                       ['ResnetBlock', 'AttnBlock', 'Downsample', 'Upsample', 
                        'block_1', 'block_2', 'attn_1']):
                    should_profile = True
            elif granularity == 'operations':
                # Profile specific operation types
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.GroupNorm, 
                                     nn.MultiheadAttention)):
                    should_profile = True
            
            if should_profile:
                hook = ProfilerHook(name, self.profile_data)
                handle_pre = module.register_forward_pre_hook(hook.pre_hook)
                handle_post = module.register_forward_hook(hook.post_hook)
                self.hooks.extend([handle_pre, handle_post])
                
                # Record layer type and parameters
                self.profile_data['layer_types'][name] = type(module).__name__
                if isinstance(module, nn.Conv2d):
                    params = sum(p.numel() for p in module.parameters())
                    self.profile_data['layer_params'][name] = {
                        'type': 'Conv2d',
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'params': params
                    }
                elif isinstance(module, nn.Linear):
                    params = sum(p.numel() for p in module.parameters())
                    self.profile_data['layer_params'][name] = {
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'params': params
                    }
                    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def profile_forward_pass(self, x: torch.Tensor, t: torch.Tensor, 
                           num_runs: int = 10, warmup: int = 3):
        """
        Profile a complete forward pass with multiple runs
        
        Args:
            x: Input tensor
            t: Timestep tensor
            num_runs: Number of profiling runs
            warmup: Number of warmup runs (not profiled)
        """
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(x, t)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # Reset profile data
        self.profile_data['timings'] = {}
        self.profile_data['memory'] = {}
        
        # Profile runs
        total_times = []
        with torch.no_grad():
            for run in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                output = self.model(x, t)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                total_times.append((end - start) * 1000)  # ms
        
        self.profile_data['total_forward_time'] = {
            'mean': np.mean(total_times),
            'std': np.std(total_times),
            'min': np.min(total_times),
            'max': np.max(total_times),
            'all_runs': total_times
        }
        
    def analyze_results(self) -> Dict:
        """Analyze profiling results and compute statistics"""
        analysis = {
            'layer_statistics': {},
            'operation_type_summary': defaultdict(lambda: {'time': 0, 'count': 0, 'params': 0}),
            'top_10_slowest_layers': [],
            'top_10_memory_layers': [],
            'section_summary': defaultdict(lambda: {'time': 0, 'count': 0}),
        }
        
        # Compute statistics for each layer
        for layer_name, times in self.profile_data['timings'].items():
            mean_time = np.mean(times)
            std_time = np.std(times)
            total_time = np.sum(times)
            
            analysis['layer_statistics'][layer_name] = {
                'mean_time_ms': float(mean_time),
                'std_time_ms': float(std_time),
                'total_time_ms': float(total_time),
                'num_calls': len(times),
                'layer_type': self.profile_data['layer_types'].get(layer_name, 'unknown')
            }
            
            # Add memory stats if available
            if layer_name in self.profile_data['memory']:
                memory = self.profile_data['memory'][layer_name]
                analysis['layer_statistics'][layer_name]['mean_memory_mb'] = float(np.mean(memory))
                analysis['layer_statistics'][layer_name]['total_memory_mb'] = float(np.sum(memory))
            
            # Aggregate by operation type
            layer_type = self.profile_data['layer_types'].get(layer_name, 'unknown')
            analysis['operation_type_summary'][layer_type]['time'] += total_time
            analysis['operation_type_summary'][layer_type]['count'] += 1
            
            if layer_name in self.profile_data['layer_params']:
                params = self.profile_data['layer_params'][layer_name].get('params', 0)
                analysis['operation_type_summary'][layer_type]['params'] += params
            
            # Categorize by section (down, mid, up)
            if 'down' in layer_name:
                section = 'downsampling'
            elif 'mid' in layer_name:
                section = 'middle'
            elif 'up' in layer_name:
                section = 'upsampling'
            elif 'conv_in' in layer_name:
                section = 'input'
            elif 'conv_out' in layer_name or 'norm_out' in layer_name:
                section = 'output'
            elif 'temb' in layer_name:
                section = 'timestep_embedding'
            else:
                section = 'other'
                
            analysis['section_summary'][section]['time'] += total_time
            analysis['section_summary'][section]['count'] += 1
        
        # Convert defaultdicts to regular dicts
        analysis['operation_type_summary'] = dict(analysis['operation_type_summary'])
        analysis['section_summary'] = dict(analysis['section_summary'])
        
        # Find top 10 slowest layers
        sorted_layers = sorted(
            analysis['layer_statistics'].items(),
            key=lambda x: x[1]['total_time_ms'],
            reverse=True
        )
        analysis['top_10_slowest_layers'] = [
            {'name': name, **stats} 
            for name, stats in sorted_layers[:10]
        ]
        
        # Find top 10 memory consuming layers
        if self.profile_data['memory']:
            sorted_memory = sorted(
                [(name, stats) for name, stats in analysis['layer_statistics'].items() 
                 if 'total_memory_mb' in stats],
                key=lambda x: abs(x[1]['total_memory_mb']),
                reverse=True
            )
            analysis['top_10_memory_layers'] = [
                {'name': name, **stats}
                for name, stats in sorted_memory[:10]
            ]
        
        return analysis
    
    def generate_report(self, output_dir: str = 'ablation_results'):
        """Generate comprehensive report with JSON and text summary"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Add total forward time if available
        if 'total_forward_time' in self.profile_data:
            analysis['total_forward_time'] = self.profile_data['total_forward_time']
        
        # Save JSON report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f'profile_report_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate text summary
        summary_path = os.path.join(output_dir, f'profile_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MoDiff Model Profiling - Ablation Study Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Total forward time
            if 'total_forward_time' in analysis:
                ft = analysis['total_forward_time']
                f.write(f"Total Forward Pass Time:\n")
                f.write(f"  Mean: {ft['mean']:.3f} ms\n")
                f.write(f"  Std:  {ft['std']:.3f} ms\n")
                f.write(f"  Min:  {ft['min']:.3f} ms\n")
                f.write(f"  Max:  {ft['max']:.3f} ms\n\n")
            
            # Section summary
            f.write("=" * 80 + "\n")
            f.write("Model Section Summary (Total Time)\n")
            f.write("=" * 80 + "\n")
            section_summary = sorted(
                analysis['section_summary'].items(),
                key=lambda x: x[1]['time'],
                reverse=True
            )
            for section, stats in section_summary:
                f.write(f"\n{section.upper()}:\n")
                f.write(f"  Total Time: {stats['time']:.3f} ms\n")
                f.write(f"  Num Layers: {stats['count']}\n")
                if stats['time'] > 0 and 'total_forward_time' in analysis:
                    percentage = (stats['time'] / sum(s['time'] for s in analysis['section_summary'].values())) * 100
                    f.write(f"  Percentage: {percentage:.2f}%\n")
            
            # Operation type summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("Operation Type Summary\n")
            f.write("=" * 80 + "\n")
            op_summary = sorted(
                analysis['operation_type_summary'].items(),
                key=lambda x: x[1]['time'],
                reverse=True
            )
            for op_type, stats in op_summary:
                f.write(f"\n{op_type}:\n")
                f.write(f"  Total Time: {stats['time']:.3f} ms\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Total Params: {stats['params']:,}\n")
                if stats['count'] > 0:
                    f.write(f"  Avg Time per Call: {stats['time']/stats['count']:.3f} ms\n")
            
            # Top 10 slowest layers
            f.write("\n" + "=" * 80 + "\n")
            f.write("Top 10 Slowest Layers (by total time)\n")
            f.write("=" * 80 + "\n")
            for i, layer in enumerate(analysis['top_10_slowest_layers'], 1):
                f.write(f"\n{i}. {layer['name']}\n")
                f.write(f"   Type: {layer['layer_type']}\n")
                f.write(f"   Total Time: {layer['total_time_ms']:.3f} ms\n")
                f.write(f"   Mean Time: {layer['mean_time_ms']:.3f} ms\n")
                f.write(f"   Num Calls: {layer['num_calls']}\n")
            
            # Top 10 memory layers
            if analysis['top_10_memory_layers']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("Top 10 Memory-Intensive Layers (by total memory change)\n")
                f.write("=" * 80 + "\n")
                for i, layer in enumerate(analysis['top_10_memory_layers'], 1):
                    f.write(f"\n{i}. {layer['name']}\n")
                    f.write(f"   Type: {layer['layer_type']}\n")
                    f.write(f"   Total Memory: {layer['total_memory_mb']:.3f} MB\n")
                    f.write(f"   Mean Memory: {layer['mean_memory_mb']:.3f} MB\n")
        
        print(f"\n{'='*80}")
        print(f"Profiling complete!")
        print(f"{'='*80}")
        print(f"JSON Report: {json_path}")
        print(f"Text Summary: {summary_path}")
        print(f"{'='*80}\n")
        
        return json_path, summary_path, analysis


def load_ddim_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    """Load DDIM model from checkpoint"""
    from ddim.models.diffusion import Model
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert dict to object-like structure
    class DictToObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, DictToObj(v))
                else:
                    setattr(self, k, v)
    
    config = DictToObj(config)
    
    # Add missing attributes
    if not hasattr(config, 'split_shortcut'):
        config.split_shortcut = False
    
    # Load model
    model = Model(config)
    
    # Load checkpoint
    if os.path.exists(ckpt_path):
        states = torch.load(ckpt_path, map_location=device)
        if 'state_dict' in states:
            model.load_state_dict(states['state_dict'], strict=False)
        elif 'model_state_dict' in states:
            model.load_state_dict(states['model_state_dict'], strict=False)
        else:
            model.load_state_dict(states, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}, using random weights")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Profiling')
    parser.add_argument('--config', type=str, default='configs/cifar10.yml',
                       help='Path to model config')
    parser.add_argument('--ckpt', type=str, default='',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for profiling')
    parser.add_argument('--num_runs', type=int, default=20,
                       help='Number of profiling runs')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup runs')
    parser.add_argument('--granularity', type=str, default='all',
                       choices=['all', 'blocks', 'operations'],
                       help='Profiling granularity')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Output directory for results')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Input image size')
    parser.add_argument('--channels', type=int, default=3,
                       help='Number of input channels')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"MoDiff Model Profiling - Ablation Study")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt if args.ckpt else 'None (random weights)'}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Runs: {args.num_runs}")
    print(f"Granularity: {args.granularity}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model, config = load_ddim_model(args.config, args.ckpt, args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create profiler
    profiler = ModelProfiler(model, device=args.device)
    profiler.register_hooks(granularity=args.granularity)
    
    print(f"\nRegistered hooks on {len(profiler.hooks) // 2} modules")
    
    # Create dummy input
    x = torch.randn(args.batch_size, args.channels, args.image_size, args.image_size).to(device)
    t = torch.randint(0, 1000, (args.batch_size,)).to(device)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    
    # Profile
    print(f"\nRunning profiling with {args.warmup} warmup + {args.num_runs} measured runs...")
    profiler.profile_forward_pass(x, t, num_runs=args.num_runs, warmup=args.warmup)
    
    # Generate report
    print("\nGenerating report...")
    json_path, summary_path, analysis = profiler.generate_report(args.output_dir)
    
    # Print summary to console
    print("\n" + "="*80)
    print("Quick Summary")
    print("="*80)
    if 'total_forward_time' in analysis:
        ft = analysis['total_forward_time']
        print(f"Total Forward Time: {ft['mean']:.3f} ± {ft['std']:.3f} ms")
    
    print("\nTop 5 Slowest Layers:")
    for i, layer in enumerate(analysis['top_10_slowest_layers'][:5], 1):
        print(f"  {i}. {layer['name'][:60]}")
        print(f"     {layer['total_time_ms']:.3f} ms ({layer['layer_type']})")
    
    print("\nSection Breakdown:")
    section_summary = sorted(
        analysis['section_summary'].items(),
        key=lambda x: x[1]['time'],
        reverse=True
    )
    total_time = sum(s['time'] for s in analysis['section_summary'].values())
    for section, stats in section_summary[:5]:
        percentage = (stats['time'] / total_time) * 100 if total_time > 0 else 0
        print(f"  {section}: {stats['time']:.3f} ms ({percentage:.1f}%)")
    
    # Cleanup
    profiler.remove_hooks()
    
    print("\n" + "="*80)
    print("Profiling Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
