# MoDiff INT8 Demo - Implementation Plan

## Overview

This demo implements **Custom INT8 Kernels with MoDiff** for the Q-Diff baseline, providing:
1. **True INT8 computation** (not simulated quantization)
2. **MoDiff error-compensated modulation** (paper's key innovation)
3. **FID evaluation** matching paper methodology (50k samples, 100 steps)
4. **Benchmarking** for speed comparison

## Directory Structure

```
demo_int8_modiff/
├── __init__.py                    # Package initialization
├── README.md                      # Usage documentation
│
├── # Core Quantization (INT8 with true speedup)
├── quant_int8_native.py           # Native INT8 quantizer using PyTorch quantized ops
├── quant_layer_modiff.py          # QuantModule with MoDiff + INT8
├── quant_model_modiff.py          # Full model wrapper
│
├── # Sampling & Generation
├── ddim_sampler.py                # DDIM sampler with MoDiff support
├── sample_cifar10.py              # Main sampling script for CIFAR-10
│
├── # Calibration
├── calibration.py                 # Q-Diff calibration with residual support
├── generate_calib_data.py         # Generate calibration data
│
├── # Evaluation
├── fid_evaluation.py              # FID computation (50k samples)
├── benchmark.py                   # Speed benchmarking
│
├── # Utilities
├── utils.py                       # Helper functions
└── configs/
    └── cifar10_int8.yml           # Configuration file
```

## Implementation Plan

### Phase 1: Core INT8 Quantization with True Speedup
**Files:** `quant_int8_native.py`, `quant_layer_modiff.py`

#### 1.1 Native INT8 Quantizer (`quant_int8_native.py`)
- Use `torch.quantize_per_tensor` / `torch.quantize_per_channel` for true INT8
- MSE-based scale search (paper methodology)
- Support both symmetric and asymmetric quantization
- Keep data in INT8 format as long as possible

```python
class NativeINT8Quantizer:
    """True INT8 using PyTorch quantized tensors"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize to qint8 (true INT8 storage)
        x_int8 = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
        return x_int8  # Return quantized tensor, NOT dequantized!
```

#### 1.2 MoDiff Layer (`quant_layer_modiff.py`)
- Implement error-compensated modulation from paper (Equations 13-20)
- Cache `â_hat` and `ô_hat` per layer
- Use quantized convolution/linear for actual speedup
- Residual computation: `Q(a_t - â_{t+1})`

```python
class QuantLayerMoDiff:
    def forward(self, x):
        if self.a_hat is None:  # First timestep
            self.a_hat = self.quantize(x)
            out = self.quantized_op(self.a_hat)
            self.o_hat = out
        else:  # Subsequent timesteps
            residual = x - self.dequantize(self.a_hat)
            q_residual = self.quantize(residual)
            self.a_hat = self.a_hat + q_residual
            delta_out = self.quantized_op(q_residual)
            out = self.o_hat + delta_out
            self.o_hat = out
        return out
```

### Phase 2: Model Wrapper and Sampling
**Files:** `quant_model_modiff.py`, `ddim_sampler.py`, `sample_cifar10.py`

#### 2.1 Quantized Model (`quant_model_modiff.py`)
- Replace Conv2d/Linear with QuantLayerMoDiff
- `reset_cache()` method for new sample generation
- `set_quant_state()` to enable/disable quantization

#### 2.2 DDIM Sampler (`ddim_sampler.py`)
- Standard DDIM with 100 steps (paper setting for CIFAR-10)
- `eta=0` for deterministic sampling
- Integrate with MoDiff model (call `reset_cache()` per sample)

#### 2.3 Sampling Script (`sample_cifar10.py`)
```bash
python demo_int8_modiff/sample_cifar10.py \
    --config configs/cifar10.yml \
    --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --weight_bit 8 \
    --act_bit 8 \
    --modulate \          # Enable MoDiff
    --timesteps 100 \     # Paper setting
    --num_samples 50000 \ # For FID
    --output_dir output/int8_modiff_samples
```

### Phase 3: Calibration (Q-Diff Methodology)
**Files:** `calibration.py`, `generate_calib_data.py`

#### 3.1 Calibration Data Generation
- Generate intermediate activations at multiple timesteps
- For MoDiff: also store previous timestep activations
- Save as `.pt` files for reuse

#### 3.2 Q-Diff Calibration
- Weight calibration: AdaRound for optimal rounding
- Activation calibration: MSE-based scale search on residuals (key for MoDiff)
- Layer-by-layer reconstruction

### Phase 4: Evaluation
**Files:** `fid_evaluation.py`, `benchmark.py`

#### 4.1 FID Evaluation (`fid_evaluation.py`)
- Generate 50,000 samples (paper requirement)
- Compute FID against CIFAR-10 test set
- Use `pytorch-fid` or custom Inception-based computation
- Report IS (Inception Score) and sFID as well

#### 4.2 Benchmarking (`benchmark.py`)
- Measure inference time per sample
- Compare: FP32 vs Simulated INT8 vs Native INT8 + MoDiff
- Report throughput (samples/second)
- Memory usage comparison

## Expected Results (Based on Paper)

### FID Scores (CIFAR-10, 100 DDIM steps, 50k samples)

| Method | W/A Bits | FID (↓) | sFID (↓) | IS (↑) |
|--------|----------|---------|----------|--------|
| Full Precision | 32/32 | 4.24 | 4.41 | 9.00 |
| Q-Diff | 8/8 | 3.75 | 4.49 | 9.48 |
| **Q-Diff + MoDiff** | **8/8** | **4.10** | **4.39** | **9.10** |
| LCQ | 8/8 | 4.21 | 4.41 | 9.01 |
| **LCQ + MoDiff** | **8/8** | **4.10** | **4.39** | **9.10** |

### Speed Comparison (Expected)

| Method | Time/Sample | Speedup | Notes |
|--------|-------------|---------|-------|
| FP32 Baseline | 100ms | 1.0x | No quantization |
| Simulated INT8 | 100ms | 1.0x | Fake quant (current) |
| **Native INT8 + MoDiff** | **40-50ms** | **2-2.5x** | This demo |
| TensorRT INT8 | 20-30ms | 3-5x | No MoDiff (broken FID) |

## Implementation Order

### Day 1: Core INT8 + MoDiff Layer
1. [ ] Create `demo_int8_modiff/` directory structure
2. [ ] Implement `quant_int8_native.py` - Native INT8 quantizer
3. [ ] Implement `quant_layer_modiff.py` - MoDiff layer with INT8
4. [ ] Unit tests for quantizer accuracy

### Day 2: Model & Sampling
5. [ ] Implement `quant_model_modiff.py` - Full model wrapper
6. [ ] Implement `ddim_sampler.py` - DDIM with MoDiff support
7. [ ] Implement `sample_cifar10.py` - Main sampling script
8. [ ] Verify model loads and runs

### Day 3: Calibration
9. [ ] Implement `generate_calib_data.py` - Calibration data
10. [ ] Implement `calibration.py` - Q-Diff calibration
11. [ ] Run calibration on CIFAR-10 model

### Day 4: Evaluation
12. [ ] Implement `fid_evaluation.py` - FID computation
13. [ ] Implement `benchmark.py` - Speed measurement
14. [ ] Generate 50k samples and compute FID
15. [ ] Document results in README

## Key Technical Decisions

### 1. Native INT8 vs Simulated
**Choice:** Native INT8 using `torch.quantize_per_tensor`
**Reason:** True speedup from INT8 matmul, not just accuracy simulation

### 2. Quantization Granularity
**Choice:** Per-tensor for activations, per-channel for weights
**Reason:** Matches paper methodology, good accuracy-speed tradeoff

### 3. Scale Calibration
**Choice:** MSE-based search (not entropy)
**Reason:** Paper explicitly uses MSE for low-bit quantization

### 4. MoDiff State Management
**Choice:** Store `a_hat` and `o_hat` as class attributes
**Reason:** Simple, matches original implementation style

## Dependencies

```
torch>=2.0.0          # For torch.compile and native quantization
torchvision>=0.15.0
pytorch-fid>=0.3.0    # FID computation
tqdm
numpy
pyyaml
pytorch-lightning     # Seed everything
```

## Usage Examples

### Quick Test (100 samples)
```bash
cd /teamspace/studios/this_studio/MoDiff
python -m demo_int8_modiff.sample_cifar10 \
    --num_samples 100 \
    --weight_bit 8 --act_bit 8 \
    --modulate
```

### Full FID Evaluation (50k samples)
```bash
python -m demo_int8_modiff.fid_evaluation \
    --num_samples 50000 \
    --output_dir results/int8_modiff
```

### Benchmark Speed
```bash
python -m demo_int8_modiff.benchmark \
    --num_warmup 10 \
    --num_runs 100
```
