
import os
import torch
import time
from ldm.util import instantiate_from_config
from integration.int8_optimized import convert_model_to_optimized_int8, enable_modiff_mode as enable_int8_modiff
from integration.int4_optimized import convert_model_to_optimized_int4, enable_modiff_mode as enable_int4_modiff

# Global settings for fair benchmarking
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def benchmark_ldm_baseline(mode="fp16", batch_size=32):
    # Setup model
    config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
    from omegaconf import OmegaConf
    conf = OmegaConf.load(config)
    model = instantiate_from_config(conf.model)
    model.cuda().eval()
    
    # Correct device conversion
    if mode == "fp16":
        model.half()
    elif mode == "int8_std":
        # Keep backbone in FP16 for speed, but layers in INT8
        model.half()
        model.model.diffusion_model = convert_model_to_optimized_int8(model.model.diffusion_model)
        enable_int8_modiff(model.model.diffusion_model, False) # Disable MoDiff caching
        
    # Dummy input (Batch Size 32 to saturate GPU)
    shape = (batch_size, 4, 32, 32)
    x = torch.randn(shape).cuda().half() if mode != "fp32" else torch.randn(shape).cuda()
    t = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Warmup
    print(f"Warming up {mode}...")
    for _ in range(15):
        with torch.no_grad():
            model.model.diffusion_model(x, t)
                
    # Timing
    print(f"Benchmarking {mode}...")
    torch.cuda.synchronize()
    start = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            model.model.diffusion_model(x, t)
    torch.cuda.synchronize()
    
    avg_ms = (time.time() - start) * 1000 / num_runs
    return avg_ms

if __name__ == "__main__":
    modes = ["fp32", "fp16", "int8_std"]
    results = {}
    
    # We use FP16 as the REAL-WORLD BASELINE for diffusion
    for m in modes:
        results[m] = benchmark_ldm_baseline(m, batch_size=32)
        
    print("\n" + "="*50)
    print(f"{'Mode':15s} | {'Latency (ms)':12s} | {'Speedup (vs FP32)':15s}")
    print("-" * 50)
    for m in modes:
        latency = results[m]
        speedup = results["fp32"] / latency
        print(f"{m:15s} | {latency:12.2f} ms | {speedup:8.2f}x")
    print("="*50)

