# MoDiff Triton Kernels

Custom Triton kernel implementation of **MoDiff (Modulated Diffusion)** for accelerating diffusion models with low-bit quantization.

## Overview

This implementation follows the MoDiff paper's layer-level error-compensated modulation framework:

```
At timestep T (first step):
    â_T = Q(a_T)                              -- Eq. (ec1)
    ô_T = A(â_T) + bias                       -- Eq. (ec2)

At timestep t < T:
    â_t = Q(a_t - â_{t+1}) + â_{t+1}          -- Eq. (ec5)
    ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}       -- Eq. (ec6)
```

**Key insight**: The residual `(a_t - â_{t+1})` has a much smaller range than the original activation, enabling lower-bit quantization (3-4 bits) with comparable accuracy to 8-bit.

## Features

- **W8A8 MoDiff**: INT8 weights + INT8 activations
- **W4A4 MoDiff**: INT4 weights + INT4 activations (packed)
- **Error-Compensated Modulation**: Prevents error accumulation across diffusion timesteps
- **Triton Kernels**: High-performance GPU implementations
- **Layer Support**: Linear and Conv2d layers

## Directory Structure

```
triton/
├── __init__.py
├── kernels/
│   ├── quantize.py           # INT8/INT4 quantization kernels
│   ├── modulated_quantize.py # MoDiff modulated quantization (Eq. ec5)
│   ├── gemm_w8a8.py          # W8A8 GEMM with accumulation (Eq. ec6)
│   └── gemm_w4a4.py          # W4A4 GEMM with INT4 packing
├── nn/
│   ├── config.py             # MoDiffConfig
│   ├── linear.py             # W8A8/W4A4MoDiffLinear
│   └── conv.py               # W8A8/W4A4MoDiffConv2d
├── utils.py                  # Model conversion utilities
└── tests/
    ├── test_modiff.py        # Correctness tests
    └── benchmark.py          # Performance benchmarks
```

## Usage

### Basic Usage

```python
import torch
from triton.nn import W8A8MoDiffLinear, MoDiffConfig

# Create from existing linear layer
linear = torch.nn.Linear(512, 256).cuda()
config = MoDiffConfig(weight_bits=8, act_bits=8)
q_linear = W8A8MoDiffLinear.from_linear(linear, config)

# Diffusion loop
q_linear.reset_cache()  # Start new sequence
for t in reversed(range(T)):
    x_t = ...  # Get activation at timestep t
    output = q_linear(x_t)  # MoDiff forward
```

### Model Conversion

```python
from triton.utils import convert_to_modiff, MoDiffModelWrapper
from triton.nn import MoDiffConfig

# Convert entire model
config = MoDiffConfig(weight_bits=4, act_bits=4)  # W4A4
model_q = convert_to_modiff(model, config)

# Wrap for convenient inference
wrapper = MoDiffModelWrapper(model_q)

# Inference
for batch in dataloader:
    wrapper.start_new_sequence()  # Reset caches
    for t in timesteps:
        output = wrapper(x_t, t)
```

### Running Tests

```bash
cd MoDiff/triton
python -m tests.test_modiff
python -m tests.benchmark
```

## Algorithm Details

### Modulated Quantization (Section 3.2)

Standard quantization applies `Q(a_t)` directly to activations. MoDiff instead:

1. Computes the residual: `residual = a_t - â_{t+1}`
2. Quantizes the residual: `Q(residual)` 
3. Updates cache: `â_t = Q(residual) + â_{t+1}`

Since the residual has ~10x smaller range, we can use 3-4 fewer bits with the same quantization error.

### Error-Compensated Modulation (Section 3.3)

Without error compensation, quantization errors accumulate across timesteps:
- Standard modulation: Error grows as O(2^(T-t))
- Error-compensated: Error grows as O((2c)^(T-t)) where c < 1/2

The key is computing residuals with respect to `â_{t+1}` (the quantized cache) rather than `a_{t+1}` (the original activation).

### Memory Overhead (Section 3.4)

MoDiff requires storing two caches per layer:
- `â_t`: Quantized activation cache [batch, input_dim]
- `ô_t`: Output cache [batch, output_dim]

This overhead is negligible compared to model weights at small batch sizes.

## Performance

Expected speedups (vs FP16 baseline):
- **W8A8 MoDiff**: 1.5-2x speedup, 2x memory reduction
- **W4A4 MoDiff**: 2-3x speedup, 4x memory reduction

Accuracy (from paper):
- W8A8: Near-lossless (<1% FID degradation)
- W4A4: Minimal loss (<3% FID degradation) thanks to error compensation

## Citation

```bibtex
@inproceedings{modiff2025,
  title={Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization},
  author={Gao, Weizhi and Hou, Zhichao and Yin, Junqi and Wang, Feiyi and Peng, Linyu and Liu, Xiaorui},
  booktitle={ICML},
  year={2025}
}
```
