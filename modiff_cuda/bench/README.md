# MoDiff CUDA Kernel Benchmarks

This directory contains comprehensive benchmarking scripts for MoDiff custom CUDA kernels, following the same pattern as ViDiT-Q benchmarks.

## Available Benchmarks

### 1. INT8 Convolution (`bench_int8_conv.py`)

Benchmarks INT8 convolution kernels against PyTorch FP32/FP16 baselines.

**Features:**
- Correctness testing with ground truth comparison
- Performance comparison: FP32 vs FP16 vs INT8 (dynamic) vs INT8 (static)
- Speedup calculations
- Multiple common configurations

**Usage:**
```bash
# Run with default settings
python bench_int8_conv.py

# Test correctness only
python bench_int8_conv.py --test_correctness

# Custom configuration
python bench_int8_conv.py --batch_size 8 --in_channels 512 --out_channels 512 \
    --height 32 --width 32 --kernel_size 3 --stride 1 --padding 1
```

**Arguments:**
- `--batch_size`: Batch size (default: 4)
- `--in_channels`: Input channels (default: 256)
- `--out_channels`: Output channels (default: 256)
- `--height`: Input height (default: 64)
- `--width`: Input width (default: 64)
- `--kernel_size`: Kernel size (default: 3)
- `--stride`: Stride (default: 1)
- `--padding`: Padding (default: 1)
- `--num_iter`: Number of iterations (default: 100)
- `--num_warmup_iter`: Number of warmup iterations (default: 20)
- `--test_correctness`: Test correctness only

### 2. INT4 Convolution (`bench_int4_conv.py`)

Benchmarks INT4 convolution kernels against PyTorch baselines and INT8 kernels.

**Features:**
- INT4 quantization correctness testing
- Performance comparison: FP32 vs FP16 vs INT8 vs INT4 (dynamic) vs INT4 (static)
- Memory efficiency analysis
- Accuracy vs performance tradeoff

**Usage:**
```bash
# Run with default settings
python bench_int4_conv.py

# Test correctness only
python bench_int4_conv.py --test_correctness

# Custom configuration
python bench_int4_conv.py --batch_size 8 --in_channels 512 --out_channels 512 \
    --height 32 --width 32 --kernel_size 3
```

**Arguments:** Same as `bench_int8_conv.py`

### 3. Fused Kernels (`bench_fused_kernels.py`)

Benchmarks fused operations (GroupNorm + SiLU, Conv + GroupNorm + SiLU, etc.)

**Features:**
- Fused GroupNorm + SiLU
- Fused Conv + GroupNorm + SiLU (two-pass)
- Kernel fusion speedup analysis
- Memory bandwidth savings

**Usage:**
```bash
# Benchmark all fused kernels
python bench_fused_kernels.py

# Benchmark specific kernel
python bench_fused_kernels.py --kernel groupnorm_silu
python bench_fused_kernels.py --kernel conv_groupnorm_silu

# Test correctness only
python bench_fused_kernels.py --test_correctness

# Custom configuration for GroupNorm + SiLU
python bench_fused_kernels.py --kernel groupnorm_silu --batch_size 4 \
    --channels 512 --height 32 --width 32 --num_groups 32

# Custom configuration for Conv + GroupNorm + SiLU
python bench_fused_kernels.py --kernel conv_groupnorm_silu --batch_size 4 \
    --in_channels 256 --out_channels 512 --height 64 --width 64 \
    --kernel_size 3 --stride 2 --padding 1 --num_groups 32
```

**Arguments:**
- `--kernel`: Which kernel to benchmark: `all`, `groupnorm_silu`, `conv_groupnorm_silu` (default: all)
- `--batch_size`: Batch size (default: 4)
- `--channels`: Number of channels for groupnorm_silu (default: 256)
- `--in_channels`: Input channels for conv_groupnorm_silu (default: 256)
- `--out_channels`: Output channels for conv_groupnorm_silu (default: 256)
- `--height`: Input height (default: 64)
- `--width`: Input width (default: 64)
- `--num_groups`: Number of groups for GroupNorm (default: 32)
- `--kernel_size`: Kernel size for conv (default: 3)
- `--stride`: Stride for conv (default: 1)
- `--padding`: Padding for conv (default: 1)
- `--num_iter`: Number of iterations (default: 100)
- `--num_warmup_iter`: Number of warmup iterations (default: 20)
- `--test_correctness`: Test correctness only

### 4. Quantization Operations (`bench_quant_ops.py`)

Benchmarks quantization/dequantization and fused residual operations.

**Features:**
- Basic quantization operations
- Fused residual + quantize
- Fused dequantize + accumulate
- Fast quantization kernels

**Usage:**
```bash
# Benchmark all quantization operations
python bench_quant_ops.py

# Benchmark specific operation
python bench_quant_ops.py --operation quantize
python bench_quant_ops.py --operation fused_residual_quantize
python bench_quant_ops.py --operation fused_dequantize_accumulate

# Test correctness only
python bench_quant_ops.py --test_correctness

# Custom configuration
python bench_quant_ops.py --operation quantize --batch_size 4 \
    --channels 512 --height 32 --width 32
```

**Arguments:**
- `--operation`: Which operation to benchmark: `all`, `quantize`, `fused_residual_quantize`, `fused_dequantize_accumulate` (default: all)
- `--batch_size`: Batch size (default: 4)
- `--channels`: Number of channels (default: 256)
- `--height`: Input height (default: 64)
- `--width`: Input width (default: 64)
- `--num_iter`: Number of iterations (default: 100)
- `--num_warmup_iter`: Number of warmup iterations (default: 20)
- `--test_correctness`: Test correctness only

## Benchmark Methodology

All benchmarks follow a consistent methodology:

1. **Correctness Testing**: Compare custom kernel output against PyTorch ground truth
   - Maximum absolute error
   - Mean absolute error
   - Relative error (where applicable)

2. **Performance Measurement**:
   - Warmup iterations to stabilize GPU state
   - CUDA events for precise timing
   - Multiple iterations for averaging
   - Synchronization to ensure accurate measurements

3. **Metrics Reported**:
   - Average execution time (ms)
   - Speedup compared to baselines
   - TFLOPS (for compute-bound operations)

## Expected Results

Typical speedups you should see (GPU-dependent):

- **INT8 Conv**: 2-4x faster than FP32, 1.5-2.5x faster than FP16
- **INT4 Conv**: 1.5-2x faster than INT8
- **Fused GroupNorm+SiLU**: 1.5-2.5x faster than sequential
- **Fused Conv+GroupNorm+SiLU**: 1.3-2x faster than sequential
- **Quantization ops**: 2-5x faster than PyTorch baseline

## Notes

- Ensure CUDA kernels are properly compiled before running benchmarks
- Results vary based on GPU architecture, input sizes, and system configuration
- Use `--test_correctness` first to verify kernel correctness
- For production use, benchmark with your actual model configurations
