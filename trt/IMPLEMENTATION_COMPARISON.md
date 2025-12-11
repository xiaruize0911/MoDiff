# Implementation Comparison: TRT vs PyTorch MoDiff

## Executive Summary

The current TensorRT (TRT) implementation **does NOT implement the MoDiff methodology** from the paper. This explains the poor FID results (36-37) compared to the paper's results (~4-5 FID).

## Key Differences

### 1. Modulated Quantization (Critical Missing Feature)

**Paper's MoDiff Equation (Error-Compensated Modulation):**
```
â_T = Q(a_T)                           # First timestep: quantize directly
ô_T = A(â_T)                           # Compute output

â_{t} = Q(a_t - â_{t+1}) + â_{t+1}     # Later: quantize RESIDUAL + add back
ô_{t} = A(Q(a_t - â_{t+1})) + ô_{t+1}  # Output = incremental update
```

**PyTorch MoDiff Implementation (`qdiff/quant_layer.py` lines 320-340):**
```python
if self.modulate:
    if self.a_hat is None:
        # First timestep: store the input
        self.a_hat = input.clone().detach()
    else:
        # Subsequent timesteps: compute and quantize RESIDUAL
        input = input - self.a_hat                    # residual = current - previous
        input = self.act_quantizer(input)             # quantize the residual
        self.a_hat = (self.a_hat + input).clone().detach()  # update cache

# For output:
if self.modulate and self.use_act_quant:
    if self.o_hat is None:
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
    else:
        out = self.fwd_func(input, weight, None, **self.fwd_kwargs)
        out = self.o_hat + out  # Add to previous output
    self.o_hat = out.clone().detach()
```

**Current TRT Implementation (`trt/fid_comparison.py` lines 145-200):**
```python
def ddim_sample(engine, num_samples, num_steps=50, eta=0.0, seed=None):
    for idx, t in enumerate(timesteps):
        t_batch = np.array([t], dtype=np.int64)
        
        # PROBLEM: Direct inference without any modulation!
        noise_pred = engine.infer(x, t_batch)  # ← No residual computation
        
        # Standard DDIM update (no error compensation)
        pred_x0 = (x - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
        # ... rest of DDIM
```

### 2. Calibration Methodology

**Paper's MoDiff Calibration:**
- Uses **residual data** for calibration: `(a_t - a_{t+1})`
- Calibrates quantizer scales on residual distributions (smaller, more concentrated)
- Requires storing previous timestep activations

**PyTorch Implementation (`qdiff/layer_recon.py` lines 117-210):**
```python
def layer_reconstruction_modiff(...):
    # Get both current and previous timestep data
    cached_inps, cached_inps_prev, cached_outs = save_inp_oup_data_modiff(...)
    
    # Initialize scale based on RESIDUAL range (key!)
    with torch.no_grad():
        delta = ((cached_inps - cached_inps_prev).max() - 
                 (cached_inps - cached_inps_prev).min()) / (n_levels - 1)
        layer.act_quantizer.delta.copy_(delta)
    
    # Calibration loop uses residual
    for i in range(iters):
        model.reset_cache()
        with torch.no_grad():
            layer(prev_inp)      # Process previous to populate cache
        out_quant = layer(cur_inp)  # Now current computes residual internally
```

**Current TRT Calibration (`trt/int4_calibrator.py`):**
```python
# PROBLEM: Calibrates on raw activations, NOT residuals
def compute_scale(self, x: np.ndarray):
    # MSE search on raw values, missing the residual insight
    x_absmax = np.maximum(np.abs(x.min()), np.abs(x.max()))
    base_scale = x_absmax / self.q_max
```

### 3. DDIM Steps

| Setting | Paper | Current TRT |
|---------|-------|-------------|
| CIFAR-10 DDIM Steps | 100 | 50 |

### 4. State Management

**PyTorch MoDiff:**
- Maintains per-layer state: `a_hat` (cached input), `o_hat` (cached output)
- `model.reset_cache()` called at start of each sample generation
- State updates throughout diffusion timesteps

**TRT Implementation:**
- Stateless inference
- No caching mechanism
- No residual tracking

## Why This Matters

### Quantization Error Analysis

From the paper (Section 3, Theorem 1):
```
||x - Q(x)||² ≤ (max(x) - min(x))² × d / (2^b - 1)²
```

Key insight: **Residuals have ~10x smaller range** than raw activations.

| Data Type | Typical Range | INT4 Quant Error |
|-----------|---------------|------------------|
| Raw activation | [-10, 10] | High |
| Residual (a_t - a_{t+1}) | [-1, 1] | ~100x lower |

### Error Accumulation

Without error compensation (current TRT):
```
||o_t - õ_t||² ≤ Σ 2^(T-k-1) × c × ||A||² × ||a_k - a_{k+1}||²
              (grows exponentially!)
```

With MoDiff error compensation:
```
||o_t - ô_t||² ≤ Σ (2c)^(T-k-1) × ||A||² × ||a_k - a_{k+1}||²
              (shrinks exponentially when c < 0.5!)
```

## FID Impact

| Method | Expected FID | Actual FID | Gap |
|--------|--------------|------------|-----|
| FP32 | 4.24 | 6.61 | +2.4 |
| W8/A8 + MoDiff | 4.10 | 36.22 (no MoDiff) | +32 |
| W4/A4 + MoDiff | 5.10 | 37.12 (no MoDiff) | +32 |

## Solution: Use PyTorch MoDiff for FID Evaluation

To get paper-level results, use the PyTorch implementation with proper MoDiff:

```bash
# Generate samples with MoDiff quantization
python scripts/sample_diffusion_ddim.py \
    --config configs/cifar10.yml \
    --timesteps 100 \
    --eta 0.0 \
    --ptq \
    --weight_bit 4 \
    --act_bit 4 \
    --modulate \              # ← KEY: Enable MoDiff
    --quant_act \
    --max_images 50000 \
    --image_folder output/modiff_samples
```

## Code Flow Comparison

### PyTorch MoDiff Flow
```
1. Load model
2. Create QuantModel with modulate=True
3. Generate calibration data with residuals (xs, ts, xs_prev, ts_prev)
4. Calibrate on residual distributions
5. For each sample:
   a. model.reset_cache()
   b. For each timestep t:
      - Layer internally computes: residual = input - a_hat
      - Quantizes residual (small range, low error)
      - Updates: a_hat = a_hat + Q(residual)
      - Output: o_hat = o_hat + A(Q(residual))
```

### Current TRT Flow (Broken)
```
1. Load TRT engine
2. For each sample:
   a. For each timestep t:
      - Direct inference: output = engine.infer(x, t)  # No modulation!
      - Standard DDIM update
```

## Files Reference

| Component | PyTorch (Correct) | TRT (Missing MoDiff) |
|-----------|-------------------|---------------------|
| Quantized Layer | `qdiff/quant_layer.py` | N/A (uses TRT native) |
| Modulation Logic | `QuantModule.forward()` | Missing |
| Calibration | `layer_recon.py` → `layer_reconstruction_modiff()` | `int4_calibrator.py` (no residual) |
| State Management | `model.reset_cache()` | None |
| Sampling | `sample_diffusion_ddim.py` | `fid_comparison.py` |

---

## Can MoDiff Be Implemented in TensorRT?

**YES**, but it requires architectural changes. Here are the options:

### Option 1: Pure PyTorch MoDiff (Recommended for Accuracy)
- Use the existing PyTorch implementation
- Get paper-level FID results (~4-5)
- No TRT acceleration
- **Best for: Validating paper results**

### Option 2: Hybrid PyTorch + TRT (Best Balance)
```
┌─────────────────────────────────────────────────────────┐
│                    PyTorch Orchestrator                  │
│  - State management (a_hat, o_hat per layer)            │
│  - Residual computation: residual = input - a_hat       │
│  - Quantization: Q(residual)                            │
│  - Cache updates: a_hat = a_hat + Q(residual)           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    TensorRT Engine                       │
│  - Heavy compute: Conv2D, Linear, Attention             │
│  - Accepts pre-quantized residual as input              │
│  - Returns layer output (added to o_hat in PyTorch)     │
└─────────────────────────────────────────────────────────┘
```

**Implementation approach:**
1. Export ONLY the compute-heavy operations to TRT (Conv2D, Linear)
2. Keep quantization and state management in PyTorch
3. Call TRT for the matrix multiplications
4. PyTorch handles residual logic around TRT calls

**Pros:** 
- Gets most TRT speedup (matmul is 90%+ of compute)
- Maintains MoDiff accuracy
- Easier to implement

**Cons:**
- Some CPU-GPU data transfer overhead
- More complex inference loop

### Option 3: Custom TRT Plugins (Maximum Speed)
Create custom TensorRT plugins that implement MoDiff internally:

```cpp
// Pseudocode for MoDiff TRT Plugin
class MoDiffConvPlugin : public IPluginV2 {
    // Persistent state
    float* a_hat;  // Cached input
    float* o_hat;  // Cached output
    
    int enqueue(...) {
        if (first_timestep) {
            a_hat = quantize(input);
            output = conv(a_hat, weight);
            o_hat = output;
        } else {
            residual = input - a_hat;
            q_residual = quantize(residual);
            a_hat = a_hat + q_residual;
            delta_out = conv(q_residual, weight);
            output = o_hat + delta_out;
            o_hat = output;
        }
        return 0;
    }
};
```

**Pros:**
- Maximum speed (all on GPU)
- True INT4 operations possible

**Cons:**
- Requires CUDA/TRT plugin development
- Complex to debug
- TRT version dependent

### Option 4: Modified TRT Inference with External State (Practical Compromise)

Modify the TRT inference loop to handle state externally:

```python
class MoDiffTRTInference:
    def __init__(self, engine):
        self.engine = engine
        self.layer_states = {}  # {layer_name: {'a_hat': tensor, 'o_hat': tensor}}
    
    def reset_cache(self):
        for name in self.layer_states:
            self.layer_states[name] = {'a_hat': None, 'o_hat': None}
    
    def infer_with_modulation(self, x, t, is_first_step):
        # For each layer in the network:
        # 1. If not first step: compute residual externally
        # 2. Quantize residual (PyTorch or custom kernel)
        # 3. Run TRT for the matrix multiply
        # 4. Add to cached output
        # 5. Update cache
        pass
```

This requires exposing intermediate layer outputs from TRT, which needs model restructuring.

---

## Recommendation

For your goal of **matching paper FID results**:

| Priority | Approach | FID Expected | Speedup | Effort |
|----------|----------|--------------|---------|--------|
| 1st | Pure PyTorch MoDiff | ~4-5 | 1x | Low |
| 2nd | Hybrid PyTorch+TRT | ~4-5 | 3-5x | Medium |
| 3rd | TRT Plugins | ~4-5 | 8-10x | High |
| Current | TRT without MoDiff | ~35-37 | 1.7x | Done (broken) |

**Immediate action:** Run PyTorch MoDiff to validate paper results first, then consider hybrid approach for speedup.

---

## Why Original PyTorch Has No Speed Optimization

### The Paper's Goal vs Implementation Reality

From the paper (Section 4, Remark):
> "The primary objective of this paper is to demonstrate the effectiveness of our method. 
> We do not report the real acceleration metrics, such as running time. Following existing 
> works, we evaluate efficiency by measuring the number of binary operations (Bops) per 
> denoising step... **Implementing acceleration on specialized hardware is beyond the scope 
> of this work, but will be a promising future direction.**"

### Current PyTorch Implementation: Simulated Quantization

The current code does **fake/simulated quantization** - NOT true INT8/INT4:

```python
# From qdiff/quant_layer.py - UniformAffineQuantizer.forward()
def forward(self, x: torch.Tensor):
    # Quantize to integers
    x_int = round_ste(x / self.delta) + self.zero_point
    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
    
    # IMMEDIATELY dequantize back to FP32!
    x_dequant = (x_quant - self.zero_point) * self.delta  # ← Still FP32!
    return x_dequant
```

**Problem:** The data stays in FP32 throughout. This:
- ✅ Simulates quantization error accurately (for FID measurement)
- ❌ Provides ZERO speed benefit (still FP32 compute)
- ❌ Uses same memory as FP32

### Why No Speedup Happens

| Operation | Current PyTorch | True INT8 |
|-----------|-----------------|-----------|
| Storage | FP32 (32 bits) | INT8 (8 bits) |
| Compute | FP32 matmul | INT8 matmul |
| Memory BW | Full FP32 | 4x less |
| Speedup | 1x | 2-4x |

---

## How to Add Real Speed Optimization to PyTorch MoDiff

### Solution 1: PyTorch Native Quantization (Easiest)

Use `torch.quantization` for true INT8 operations:

```python
import torch.quantization as quant

# Convert model to use true INT8 operations
model_int8 = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

**Limitation:** Doesn't directly support MoDiff's residual logic.

### Solution 2: Custom INT8 Kernels with MoDiff (Recommended)

Create optimized kernels that:
1. Keep data in INT8 format
2. Implement MoDiff residual computation in INT8
3. Only dequantize at layer boundaries if needed

```python
# Pseudocode for optimized MoDiff layer
class OptimizedMoDiffConv2d(nn.Module):
    def forward(self, x_int8, a_hat_int8):
        if a_hat_int8 is None:
            # First timestep: quantize and store
            self.a_hat_int8 = x_int8
            out_int8 = torch.ops.quantized.conv2d(x_int8, self.weight_int8)
            self.o_hat_int8 = out_int8
        else:
            # Compute residual in INT8
            residual_int8 = x_int8 - self.a_hat_int8  # INT8 subtraction
            self.a_hat_int8 = self.a_hat_int8 + residual_int8
            
            # INT8 convolution on residual
            delta_out_int8 = torch.ops.quantized.conv2d(residual_int8, self.weight_int8)
            out_int8 = self.o_hat_int8 + delta_out_int8
            self.o_hat_int8 = out_int8
        
        return out_int8
```

### Solution 3: Use torch.compile (PyTorch 2.0+)

```python
import torch

# Compile with optimizations
model = torch.compile(model, mode="reduce-overhead")

# Enable TF32 for faster FP32 on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Solution 4: FBGEMM / OneDNN Backend

PyTorch can use optimized INT8 backends:

```python
# Force FBGEMM backend for x86 or OneDNN
torch.backends.quantized.engine = 'fbgemm'  # or 'qnnpack' for ARM
```

---

## Implementing Fast MoDiff: Step-by-Step Plan

### Phase 1: Validate Accuracy (Current State)
```bash
# Run original PyTorch MoDiff - slow but accurate
python scripts/sample_diffusion_ddim.py \
    --config configs/cifar10.yml \
    --timesteps 100 --eta 0 \
    --ptq --weight_bit 8 --act_bit 8 \
    --modulate --quant_act \
    --max_images 50000
```
**Expected:** FID ~4-5, Speed: ~1x baseline

### Phase 2: Add torch.compile Optimization
```python
# In sample_diffusion_ddim.py, after loading model:
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
```
**Expected:** FID ~4-5, Speed: ~1.5-2x

### Phase 3: True INT8 with MoDiff Logic
Replace `UniformAffineQuantizer` with real quantized operations:

```python
class TrueINT8MoDiffQuantizer(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.scale = None
        self.zero_point = None
        
    def forward(self, x_fp32):
        # Quantize to true INT8
        x_int8 = torch.quantize_per_tensor(
            x_fp32, self.scale, self.zero_point, torch.qint8
        )
        return x_int8  # Return INT8, not dequantized!
```
**Expected:** FID ~4-5, Speed: ~2-4x

### Phase 4: Fused Kernels (Advanced)
Write custom CUDA kernels that fuse:
- Residual computation
- Quantization  
- Convolution
- Cache update

**Expected:** FID ~4-5, Speed: ~4-8x

---

## Quick Optimization Script

Here's a drop-in optimization you can add NOW:

**New file created: `qdiff/quant_layer_fast.py`**

This provides:
- `FastINT8Quantizer` - Uses native PyTorch INT8 ops
- `FastQuantModule` - Optimized layer with MoDiff + caching
- `FastQuantModel` - Easy wrapper for whole model
- `optimize_for_inference()` - Enable all speed optimizations
- `benchmark_quantized_inference()` - Measure actual speedup

### Usage Example

```python
from qdiff.quant_layer_fast import FastQuantModel, optimize_for_inference

# Load your model
model = load_pretrained_ddim_model()

# Convert to fast quantized model with MoDiff
fast_model = FastQuantModel(
    model, 
    weight_bits=8,    # W8 quantization
    act_bits=8,       # A8 quantization  
    modulate=True,    # Enable MoDiff!
)

# Apply additional optimizations (torch.compile, TF32, etc.)
fast_model = optimize_for_inference(fast_model)

# Enable quantization
fast_model.set_quant_state(weight_quant=True, act_quant=True)

# Generate samples
for i in range(num_samples):
    fast_model.reset_cache()  # Reset MoDiff state
    x = torch.randn(1, 3, 32, 32, device='cuda')
    
    for t in timesteps:
        x = ddim_step(fast_model, x, t)
    
    save_image(x, f"sample_{i}.png")
```

### Expected Speedups

| Optimization | Speedup | Notes |
|--------------|---------|-------|
| torch.compile | 1.3-2x | PyTorch 2.0+ |
| Native INT8 | 1.5-2x | Real INT8 matmul |
| TF32 | 1.2x | Ampere+ GPUs |
| Weight caching | 1.1x | Avoid re-quantization |
| **Combined** | **2-4x** | All together |
