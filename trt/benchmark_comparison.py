#!/usr/bin/env python3
"""
Benchmark Comparison: TensorRT FP32 vs INT4 Engines

This script compares TensorRT engines:
1. Inference time (latency)
2. Throughput (samples/sec)
3. Prediction quality vs a TensorRT FP32 baseline (if provided)

Results are logged to CSV and plotted.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("WARNING: TensorRT not available")

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("WARNING: pandas/matplotlib not available for plotting")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TRTEngineRunner:
    """TensorRT engine runner for benchmarking."""
    
    def __init__(self, engine_path: str, name: str = "engine"):
        self.name = name
        self.engine_path = Path(engine_path)
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        # Load engine
        logger.info(f"Loading {name} engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Setup buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        self._setup_buffers()
        
        logger.info(f"  Engine size: {len(engine_data) / (1024*1024):.1f} MB")
        
    def _setup_buffers(self):
        """Allocate input/output buffers."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Replace -1 with 1 for dynamic dims
            shape = tuple(1 if s == -1 else s for s in shape)
            size = int(np.prod(shape))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            tensor_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem,
            }
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
            else:
                self.outputs.append(tensor_info)
    
    def infer(self, latent: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        """Run inference and return output."""
        # Set input shapes for dynamic dimensions (TensorRT 10.x)
        for inp in self.inputs:
            if inp['name'] == 'latent':
                self.context.set_input_shape('latent', latent.shape)
            elif inp['name'] == 'timesteps':
                self.context.set_input_shape('timesteps', timesteps.shape)
        
        # Copy inputs to host buffers
        for inp in self.inputs:
            if inp['name'] == 'latent':
                np.copyto(inp['host'][:latent.size], latent.flatten())
            elif inp['name'] == 'timesteps':
                np.copyto(inp['host'][:timesteps.size], timesteps.flatten())
        
        # Copy to device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # Set tensor addresses for TensorRT 10
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        # Execute (TensorRT 10 API)
        self.context.execute_async_v3(self.stream.handle)
        
        # Copy output back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # Return output
        return self.outputs[0]['host'].copy()
    
    def benchmark_latency(self, latent: np.ndarray, timesteps: np.ndarray, 
                          num_warmup: int = 10, num_iters: int = 100) -> Dict:
        """Benchmark inference latency."""
        # Warmup
        for _ in range(num_warmup):
            self.infer(latent, timesteps)
        
        # Benchmark
        latencies = []
        for _ in range(num_iters):
            start = time.perf_counter()
            self.infer(latent, timesteps)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput': 1000 / np.mean(latencies),  # samples/sec
        }


def compute_quality_metrics(baseline: np.ndarray, quantized: np.ndarray) -> Dict:
    """
    Compute prediction quality metrics comparing an output to a baseline.
    
    Returns:
        Dict with MSE, PSNR, MAE, and cosine similarity
    """
    baseline = baseline.flatten().astype(np.float32)
    quantized = quantized.flatten().astype(np.float32)
    
    # Ensure same size
    min_len = min(len(baseline), len(quantized))
    baseline = baseline[:min_len]
    quantized = quantized[:min_len]
    
    # MSE
    mse = np.mean((baseline - quantized) ** 2)
    
    # PSNR
    max_val = max(np.abs(baseline).max(), np.abs(quantized).max())
    if mse > 0 and max_val > 0:
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # MAE
    mae = np.mean(np.abs(baseline - quantized))
    
    # Cosine similarity
    norm_base = np.linalg.norm(baseline)
    norm_quant = np.linalg.norm(quantized)
    if norm_base > 0 and norm_quant > 0:
        cosine_sim = np.dot(baseline, quantized) / (norm_base * norm_quant)
    else:
        cosine_sim = 0.0
    
    # Max absolute error
    max_error = np.max(np.abs(baseline - quantized))
    
    return {
        'mse': float(mse),
        'psnr_db': float(psnr),
        'mae': float(mae),
        'cosine_similarity': float(cosine_sim),
        'max_error': float(max_error),
    }


def run_benchmark(args):
    """Run the full benchmark comparison."""
    results = []
    
    # Create test input
    np.random.seed(42)
    batch_size = args.batch_size
    latent = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
    timesteps = np.array([500] * batch_size, dtype=np.int64)
    
    logger.info(f"Test input: latent={latent.shape}, timesteps={timesteps.shape}")
    
    # Store outputs for quality comparison
    outputs = {}
    baseline_key = None
    
    # 1. Benchmark FP32 TensorRT engine (ordinary baseline)
    fp32_trt_engine = Path(args.fp32_engine)
    if fp32_trt_engine.exists():
        try:
            logger.info("\n" + "="*60)
            logger.info("Benchmarking FP32 TensorRT Engine (Ordinary)")
            logger.info("="*60)

            fp32_trt_runner = TRTEngineRunner(str(fp32_trt_engine), "FP32-TRT")

            outputs['FP32-TRT'] = fp32_trt_runner.infer(latent, timesteps)
            baseline_key = 'FP32-TRT'

            latency = fp32_trt_runner.benchmark_latency(latent, timesteps,
                                                        args.warmup, args.iterations)

            quality = {
                'mse': 0.0,
                'psnr_db': float('inf'),
                'mae': 0.0,
                'cosine_similarity': 1.0,
                'max_error': 0.0,
            }

            engine_size = fp32_trt_engine.stat().st_size / (1024 * 1024)

            results.append({
                'model': 'FP32 (TensorRT)',
                'precision': 'FP32-TRT',
                'engine_size_mb': f'{engine_size:.1f}',
                **latency,
                **quality,
            })

            logger.info(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
            logger.info(f"  Throughput: {latency['throughput']:.1f} samples/sec")
            logger.info("  Quality baseline established (self-baseline)")

        except Exception as e:
            logger.warning(f"FP32 TensorRT benchmark failed: {e}")
    else:
        logger.warning(f"FP32 TensorRT engine not found: {fp32_trt_engine}")
    
    # 2. Benchmark INT4 TensorRT engine
    int4_engine = Path(args.int4_engine)
    if int4_engine.exists():
        try:
            logger.info("\n" + "="*60)
            logger.info("Benchmarking INT4 TensorRT Engine")
            logger.info("="*60)
            
            int4_runner = TRTEngineRunner(str(int4_engine), "INT4")
            
            # Get output
            outputs['INT4'] = int4_runner.infer(latent, timesteps)
            
            # Benchmark latency
            latency = int4_runner.benchmark_latency(latent, timesteps,
                                                     args.warmup, args.iterations)
            
            # Compute quality vs FP32 TensorRT baseline if available
            if baseline_key in outputs:
                quality = compute_quality_metrics(outputs[baseline_key], outputs['INT4'])
            else:
                quality = {'mse': 0, 'psnr_db': 0, 'mae': 0, 'cosine_similarity': 0, 'max_error': 0}
            
            engine_size = int4_engine.stat().st_size / (1024 * 1024)
            
            results.append({
                'model': 'INT4 (TensorRT)',
                'precision': 'INT4',
                'engine_size_mb': f'{engine_size:.1f}',
                **latency,
                **quality,
            })
            
            logger.info(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
            logger.info(f"  Throughput: {latency['throughput']:.1f} samples/sec")
            if baseline_key in outputs:
                logger.info(f"  Quality vs {baseline_key}: MSE={quality['mse']:.6f}, PSNR={quality['psnr_db']:.2f}dB")
            
        except Exception as e:
            logger.warning(f"INT4 benchmark failed: {e}")
    else:
        logger.warning(f"INT4 engine not found: {int4_engine}")
    
    return results, outputs


def save_results_csv(results: List[Dict], output_path: str):
    """Save benchmark results to CSV."""
    if not HAS_PLOTTING:
        # Manual CSV writing
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            if results:
                headers = list(results[0].keys())
                f.write(','.join(headers) + '\n')
                for row in results:
                    values = [str(row.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
        logger.info(f"Saved results to {output_path}")
    else:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        print("\n" + "="*60)
        print("Benchmark Results Summary")
        print("="*60)
        print(df.to_string(index=False))


def plot_results(results: List[Dict], output_dir: str):
    """Generate comparison plots."""
    if not HAS_PLOTTING:
        logger.warning("matplotlib/pandas not available, skipping plots")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    models = df['model'].tolist()
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f1c40f']  # Extend palette for extra models
    
    # 1. Latency comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Latency
    ax1 = axes[0]
    bars = ax1.bar(models, df['mean_ms'], yerr=df['std_ms'], capsize=5, color=colors[:len(models)])
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Inference Latency Comparison')
    ax1.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df['mean_ms']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    # Throughput
    ax2 = axes[1]
    bars = ax2.bar(models, df['throughput'], color=colors[:len(models)])
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Throughput Comparison')
    ax2.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df['throughput']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Quality (PSNR)
    ax3 = axes[2]
    psnr_values = df['psnr_db'].replace([np.inf, -np.inf], np.nan).fillna(100)  # Cap inf at 100
    bars = ax3.bar(models, psnr_values, color=colors[:len(models)])
    ax3.set_ylabel('PSNR (dB)')
    ax3.set_title('Prediction Quality (PSNR vs FP32)')
    ax3.tick_params(axis='x', rotation=15)
    for bar, val, orig in zip(bars, psnr_values, df['psnr_db']):
        label = '∞' if np.isinf(orig) else f'{val:.1f}'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_dir / 'benchmark_comparison.png'}")
    plt.close()
    
    # 2. Detailed quality metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE
    ax1 = axes[0]
    bars = ax1.bar(models, df['mse'], color=colors[:len(models)])
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error vs FP32 Baseline')
    ax1.tick_params(axis='x', rotation=15)
    ax1.set_yscale('log' if df['mse'].max() > 0 and df['mse'].min() > 0 else 'linear')
    
    # Cosine Similarity
    ax2 = axes[1]
    bars = ax2.bar(models, df['cosine_similarity'], color=colors[:len(models)])
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Output Similarity to FP32 Baseline')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df['cosine_similarity']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_dir / 'quality_comparison.png'}")
    plt.close()
    
    # 3. Speedup chart (vs FP32-TRT baseline if present)
    if len(df) > 1 and 'FP32-TRT' in df['precision'].values:
        fp32_latency = df[df['precision'] == 'FP32-TRT']['mean_ms'].values[0]
        speedups = fp32_latency / df['mean_ms']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, speedups, color=colors[:len(models)])
        ax.set_ylabel('Speedup vs FP32')
        ax.set_title('Inference Speedup Comparison')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'speedup_comparison.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_dir / 'speedup_comparison.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT FP32 vs INT4")
    parser.add_argument("--fp32-engine", default="export/modiff_unet_fp32.plan", help="FP32 TensorRT engine path")
    parser.add_argument("--int4-engine", default="int4_output/modiff_unet_int4.plan", help="INT4 engine path")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()
    
    print("="*60)
    print("MoDiff TensorRT Benchmark")
    print("FP32 (TensorRT) vs INT4 Comparison")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results, outputs = run_benchmark(args)
    
    if not results:
        logger.error("No benchmark results collected!")
        return 1
    
    # Save results
    csv_path = output_dir / "benchmark_results.csv"
    save_results_csv(results, str(csv_path))
    
    # Save raw data as JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved JSON to {json_path}")
    
    # Generate plots
    plot_results(results, str(output_dir))
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {json_path.name}")
    print(f"  - benchmark_comparison.png")
    print(f"  - quality_comparison.png")
    print(f"  - speedup_comparison.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
