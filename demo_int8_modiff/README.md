# INT8 MoDiff Demo

This demo implements **INT8 quantization with MoDiff (Error-Compensated Modulation)** for diffusion models following the MODiff paper methodology.

## Overview

Unlike simulated quantization (fake quant) that is commonly used for accuracy measurement only, this implementation uses **native PyTorch INT8 operations** for actual inference speedup while preserving quality through MoDiff's error compensation.

### Key Features

1. **Native INT8 Quantization**: Uses `torch.quantize_per_tensor` and quantized ops for real speedup
2. **MoDiff Error Compensation**: Implements paper equations 13-20 for error modulation
3. **Q-Diff Calibration**: MSE-based scale search for optimal quantization parameters
4. **Complete Pipeline**: End-to-end sampling with DDIM integration

## Installation

```bash
# From the MoDiff root directory
pip install -e .

# Or just ensure dependencies are installed
pip install torch torchvision pyyaml numpy scipy pillow tqdm
```

## Quick Start

### 1. Generate Calibration Data

```bash
python -m demo_int8_modiff.generate_calib_data \
    --config configs/cifar10.yml \
    --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --num_samples 1024 \
    --output calibration/cifar10_calib_data.pt
```

### 2. Generate Samples with INT8 + MoDiff

```bash
python -m demo_int8_modiff.sample_cifar10 \
    --config configs/cifar10.yml \
    --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --num_samples 1000 \
### 2. Generate Samples with INT8 + MoDiff

```bash
python -m demo_int8_modiff.sample_cifar10 \
    --config configs/cifar10.yml \
    --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --num_samples 1000 \
    --ddim_steps 100 \
    --modulate \
    --output_dir output/cifar10_int8_modiff \
    --save_samples
```

### 3. Evaluate FID

```bash
python -m demo_int8_modiff.fid_evaluation \
    --gen_dir output/cifar10_int8_modiff \
    --ref_dataset cifar10 \
    --batch_size 128
```

### 4. Run Full Benchmark

```bash
python -m demo_int8_modiff.benchmark \
    --config configs/cifar10.yml \
    --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --num_samples 20 \
    --num_images 1000 \
    --compute_fid
```

## Architecture

```
demo_int8_modiff/
├── __init__.py              # Package exports
├── quant_int8_native.py     # Native INT8 quantizer (torch.qint8)
├── quant_layer_modiff.py    # Layer-level MoDiff implementation
├── quant_model_modiff.py    # Full model wrapper
├── ddim_sampler.py          # DDIM sampling with MoDiff state management
├── calibration.py           # Q-Diff calibration utilities
├── sample_cifar10.py        # Main sampling script
├── fid_evaluation.py        # FID computation
├── benchmark.py             # Performance benchmarking
├── generate_calib_data.py   # Calibration data generation
├── utils.py                 # Common utilities
├── configs/
│   └── cifar10_int8.yml     # INT8 configuration
└── README.md                # This file
```

## MoDiff Algorithm

The MoDiff algorithm compensates for quantization errors across timesteps:

### Forward Pass (Paper Equation 13-20)

For timestep $t$ (reverse order: $T \to 1$):

1. **Compute activation delta**: $\Delta a_t = a_t - \hat{a}_{t+1}$
2. **Quantize delta**: $Q(\Delta a_t)$
3. **Apply quantization**: $\hat{a}_t = Q(\Delta a_t) + \hat{a}_{t+1}$
4. **Compute error-compensated output**: $\hat{o}_t = A(Q(\Delta a_t)) + \hat{o}_{t+1}$

Where:
- $a_t$ is the current activation
- $\hat{a}_t$ is the quantized activation with error compensation
- $A(\cdot)$ is the layer operation (Conv2d, Linear)
- $Q(\cdot)$ is the quantization function

### Key Insight

The error from quantization at timestep $t$ propagates and accumulates through timesteps. MoDiff maintains "caches" ($\hat{a}$, $\hat{o}$) that carry forward the accumulated error, allowing each step to compensate for previous errors.

## Comparison: TRT vs PyTorch+MoDiff

| Aspect | TRT Implementation | This Implementation |
|--------|-------------------|---------------------|
| Quantization | Static INT8 | Native PyTorch INT8 |
| MoDiff Support | ❌ No (stateless) | ✅ Yes (stateful caches) |
| Error Compensation | ❌ None | ✅ Full (Eq 13-20) |
| Expected FID | ~35-40 | ~4-5 (paper target) |
| Speed | Fast (TRT optimized) | Moderate (PyTorch) |

## Expected Results

Based on the MODiff paper:

| Model | Bits | FID (CIFAR-10) |
|-------|------|----------------|
| FP32 Baseline | 32/32 | 4.24 |
| Q-Diff W8/A8 | 8/8 | ~6-8 |
| Q-Diff + MoDiff W8/A8 | 8/8 | **3.75-4.10** |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python -m demo_int8_modiff.sample_cifar10 --batch_size 32
```

### Model Not Found

Ensure the checkpoint exists:
```bash
ls -la models/ema_diffusion_cifar10_model/model-790000.ckpt
```

### Poor FID Score

1. Ensure MoDiff is enabled: `--use_modiff`
2. Use sufficient calibration data (1024+ samples)
3. Check that the model reset occurs between samples

## Technical Details

### Quantization Implementation

```python
# Native INT8 quantization (not simulated)
q_tensor = torch.quantize_per_tensor(
    input, scale=scale, zero_point=0, dtype=torch.qint8
)
output = torch.ops.quantized.conv2d(q_input, q_weight, ...)
```

### MoDiff State Management

```python
class QuantModuleMoDiff(nn.Module):
    def __init__(self, ...):
        # Cache for error compensation
        self.register_buffer('a_hat', None)  # Previous quantized activation
        self.register_buffer('o_hat', None)  # Previous output with compensation
    
    def forward(self, x, ...):
        if self.use_modulation and self.a_hat is not None:
            # Error-compensated modulation (Eq 14-15)
            delta = x - self.a_hat
            q_delta = self.quantize(delta)
            x_quant = q_delta + self.a_hat
            output = self.layer(q_delta) + self.o_hat
        else:
            # Standard quantization
            x_quant = self.quantize(x)
            output = self.layer(x_quant)
        
        # Update caches for next timestep
        self.a_hat = x_quant
        self.o_hat = output
        
        return output
    
    def reset_cache(self):
        """Reset caches between different images"""
        self.a_hat = None
        self.o_hat = None
```

## Citation

If you use this code, please cite the original MoDiff paper:

```bibtex
@article{modiff2024,
  title={MoDiff: Boosting Diffusion Models for Quantization via Error-compensated Modulation},
  author={...},
  journal={ICML},
  year={2025}
}
```

## License

This code follows the same license as the main MoDiff repository.
