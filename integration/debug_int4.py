"""
Diagnostic script: Compare FP32 vs INT4 per-layer and per-timestep
to identify where signal degrades.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.getcwd())
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model():
    config_path = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
    ckpt_path = "models/ldm/lsun_churches256/model.ckpt"
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()

# ============================================================
# TEST 0: Check dtype under autocast
# ============================================================
def test_dtype_under_autocast():
    print("="*60)
    print("TEST 0: dtype behavior under autocast")
    print("="*60)
    
    from integration.int4_optimized import OptimizedInt4Conv2d
    
    conv = nn.Conv2d(64, 64, 3, padding=1).cuda()
    int4_conv = OptimizedInt4Conv2d(conv, layer_name="test_dtype")
    
    x_fp32 = torch.randn(1, 64, 8, 8, device='cuda')
    x_fp16 = x_fp32.half()
    
    # Test 1: Without autocast, FP32 input
    out1 = int4_conv(x_fp32)
    print(f"  FP32 input -> output dtype: {out1.dtype}, shape: {out1.shape}")
    
    # Test 2: Without autocast, FP16 input
    try:
        out2 = int4_conv(x_fp16)
        print(f"  FP16 input -> output dtype: {out2.dtype}, shape: {out2.shape}")
        print(f"  WARNING: FP16 input did NOT crash! Output may be garbage.")
        print(f"  FP16 out mean={out2.mean():.4f} vs FP32 out mean={out1.mean():.4f}")
    except Exception as e:
        print(f"  FP16 input -> CRASHED: {e}")
    
    # Test 3: With autocast (FP16), FP32 input
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        # Check what dtype x becomes after channels_last conversion
        x_cl = x_fp32.contiguous(memory_format=torch.channels_last)
        print(f"  Under autocast: channels_last dtype={x_cl.dtype}")
        x_scaled = x_cl * 3.14
        print(f"  Under autocast: x*scalar dtype={x_scaled.dtype}")
        
        try:
            out3 = int4_conv(x_fp32)
            print(f"  Autocast FP32 input -> output dtype: {out3.dtype}")
        except Exception as e:
            print(f"  Autocast FP32 input -> CRASHED: {e}")
    
    # Test 4: With autocast, what a typical layer output looks like
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        gn = nn.GroupNorm(32, 64).cuda()
        silu = nn.SiLU()
        gn_out = gn(x_fp32)
        silu_out = silu(gn_out)
        print(f"  Under autocast: GroupNorm output dtype={gn_out.dtype}")
        print(f"  Under autocast: SiLU output dtype={silu_out.dtype}")
        
        try:
            out4 = int4_conv(silu_out)
            print(f"  Autocast SiLU->INT4Conv -> output dtype: {out4.dtype}")
        except Exception as e:
            print(f"  Autocast SiLU->INT4Conv -> CRASHED: {e}")
            print(f"  *** BUG CONFIRMED: FP16 activations fed to INT4 kernel that expects FP32 ***")

# ============================================================  
# TEST 1: Weight distribution analysis
# ============================================================
def test_weight_distributions(model):
    print("\n" + "="*60)
    print("TEST 1: Weight distributions")
    print("="*60)
    unet = model.model.diffusion_model
    for name, m in unet.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels >= 32:
            w = m.weight.data
            w_abs = w.abs()
            print(f"  {name}: shape={tuple(w.shape)}, "
                  f"max={w_abs.max():.4f}, mean={w_abs.mean():.4f}, "
                  f"std={w_abs.std():.4f}, "
                  f"3sigma={w_abs.mean()+3*w_abs.std():.4f}, "
                  f"pct_gt1={100*(w_abs>1).float().mean():.1f}%")
            # Only show first 10
            break  # Remove this to see all

# ============================================================
# TEST 2: Single-layer comparison with real weights
# ============================================================
def test_single_layer(model):
    print("\n" + "="*60)
    print("TEST 2: Single-layer FP32 vs INT4 (real model weights)")
    print("="*60)
    
    from integration.int4_optimized import OptimizedInt4Conv2d
    
    unet = model.model.diffusion_model
    
    # Gather a few conv layers from different parts of the model
    test_layers = []
    for name, m in unet.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels >= 32:
            test_layers.append((name, m))
            if len(test_layers) >= 5:
                break
    
    for name, conv_fp32 in test_layers:
        # Create INT4 version from same weights
        int4_conv = OptimizedInt4Conv2d(conv_fp32, layer_name=name)
        
        # Create a realistic input
        x = torch.randn(1, conv_fp32.in_channels, 32, 32, device='cuda') * 2.0  # range ~[-6,6]
        
        with torch.no_grad():
            out_fp32 = conv_fp32(x)
            out_int4 = int4_conv(x)
        
        # Compare
        diff = (out_fp32 - out_int4).abs()
        fp32_mag = out_fp32.abs().mean() + 1e-8
        rel_err = diff.mean() / fp32_mag
        cos_sim = torch.nn.functional.cosine_similarity(
            out_fp32.flatten().unsqueeze(0), 
            out_int4.flatten().unsqueeze(0)
        ).item()
        
        print(f"  Layer: {name}")
        print(f"    FP32: mean={out_fp32.mean():.4f}, std={out_fp32.std():.4f}, mag={fp32_mag:.4f}")
        print(f"    INT4: mean={out_int4.mean():.4f}, std={out_int4.std():.4f}")
        print(f"    Rel Error: {rel_err:.4f} ({rel_err*100:.1f}%)")
        print(f"    Cosine Sim: {cos_sim:.6f}")
        print(f"    Max Diff: {diff.max():.4f}")
        print()

# ============================================================
# TEST 3: Full UNet single-timestep comparison
# ============================================================
def test_unet_single_step(model_fp32):
    print("\n" + "="*60)
    print("TEST 3: Full UNet single-timestep FP32 vs INT4")
    print("="*60)
    
    from integration.int4_optimized import convert_model_to_optimized_int4
    from integration.fused_resblock import fuse_resblocks_in_module
    import copy
    
    # Create INT4 model (deep copy)
    model_int4 = load_model()
    fuse_resblocks_in_module(model_int4.model.diffusion_model, inplace=True)
    convert_model_to_optimized_int4(model_int4.model.diffusion_model)
    
    # Also fuse the FP32 model to match
    fuse_resblocks_in_module(model_fp32.model.diffusion_model, inplace=True)
    model_fp32 = model_fp32.to(memory_format=torch.channels_last)
    
    # Same input noise and timestep
    torch.manual_seed(42)
    x = torch.randn(1, 4, 32, 32, device='cuda')
    t = torch.tensor([999], device='cuda')  # High noise timestep
    
    with torch.no_grad():
        # FP32 UNet output
        out_fp32 = model_fp32.model.diffusion_model(x, t)
        
        # INT4 UNet output (NO autocast - pure FP32 surrounding ops)
        out_int4_fp32 = model_int4.model.diffusion_model(x, t)
    
    diff = (out_fp32 - out_int4_fp32).abs()
    fp32_mag = out_fp32.abs().mean() + 1e-8
    rel_err = diff.mean() / fp32_mag
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fp32.flatten().unsqueeze(0),
        out_int4_fp32.flatten().unsqueeze(0)
    ).item()
    
    print(f"  WITHOUT autocast (pure FP32 surrounding ops):")
    print(f"    FP32: mean={out_fp32.mean():.4f}, std={out_fp32.std():.4f}")
    print(f"    INT4: mean={out_int4_fp32.mean():.4f}, std={out_int4_fp32.std():.4f}")
    print(f"    Rel Error: {rel_err:.4f} ({rel_err*100:.1f}%)")
    print(f"    Cosine Sim: {cos_sim:.6f}")
    
    # Now test WITH autocast (as the benchmark does)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        try:
            out_int4_autocast = model_int4.model.diffusion_model(x, t)
            diff2 = (out_fp32 - out_int4_autocast.float()).abs()
            rel_err2 = diff2.mean() / fp32_mag
            cos_sim2 = torch.nn.functional.cosine_similarity(
                out_fp32.flatten().unsqueeze(0),
                out_int4_autocast.flatten().unsqueeze(0).float()
            ).item()
            print(f"\n  WITH autocast (FP16 surrounding ops, as benchmark does):")
            print(f"    INT4+AC: mean={out_int4_autocast.float().mean():.4f}, std={out_int4_autocast.float().std():.4f}")
            print(f"    Rel Error: {rel_err2:.4f} ({rel_err2*100:.1f}%)")
            print(f"    Cosine Sim: {cos_sim2:.6f}")
            
            if cos_sim2 < cos_sim * 0.9:
                print(f"\n  *** AUTOCAST DEGRADES QUALITY: cos_sim {cos_sim:.4f} -> {cos_sim2:.4f} ***")
        except Exception as e:
            print(f"\n  WITH autocast: CRASHED: {e}")
            print(f"  *** AUTOCAST IS THE BUG: FP16 data fed to FP32-only kernel ***")
    
    # Test multiple timesteps to check drift
    print(f"\n  Per-timestep drift analysis (5 steps):")
    timesteps = [999, 800, 600, 400, 200]
    for t_val in timesteps:
        t = torch.tensor([t_val], device='cuda')
        with torch.no_grad():
            o_fp32 = model_fp32.model.diffusion_model(x, t)
            o_int4 = model_int4.model.diffusion_model(x, t)
        cs = torch.nn.functional.cosine_similarity(
            o_fp32.flatten().unsqueeze(0), o_int4.flatten().unsqueeze(0)
        ).item()
        re = (o_fp32 - o_int4).abs().mean() / (o_fp32.abs().mean() + 1e-8)
        print(f"    t={t_val}: cos_sim={cs:.6f}, rel_err={re:.4f}")

# ============================================================
# TEST 4: INT8 sanity check  
# ============================================================
def test_int8_sanity(model_fp32):
    print("\n" + "="*60)
    print("TEST 4: INT8 sanity check")
    print("="*60)
    
    from integration.int8_optimized import convert_model_to_optimized_int8
    from integration.fused_resblock import fuse_resblocks_in_module
    
    model_int8 = load_model()
    fuse_resblocks_in_module(model_int8.model.diffusion_model, inplace=True)
    convert_model_to_optimized_int8(model_int8.model.diffusion_model)
    
    torch.manual_seed(42)
    x = torch.randn(1, 4, 32, 32, device='cuda')
    t = torch.tensor([999], device='cuda')
    
    with torch.no_grad():
        out_fp32 = model_fp32.model.diffusion_model(x, t)
        out_int8 = model_int8.model.diffusion_model(x, t)
    
    diff = (out_fp32 - out_int8).abs()
    fp32_mag = out_fp32.abs().mean() + 1e-8
    rel_err = diff.mean() / fp32_mag
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fp32.flatten().unsqueeze(0),
        out_int8.flatten().unsqueeze(0)
    ).item()
    
    print(f"  FP32: mean={out_fp32.mean():.4f}, std={out_fp32.std():.4f}")
    print(f"  INT8: mean={out_int8.mean():.4f}, std={out_int8.std():.4f}")
    print(f"  Rel Error: {rel_err:.4f} ({rel_err*100:.1f}%)")
    print(f"  Cosine Sim: {cos_sim:.6f}")
    
    if cos_sim < 0.5:
        print(f"  *** INT8 IS ALSO BROKEN (cos_sim < 0.5) ***")
    elif cos_sim < 0.9:
        print(f"  *** INT8 has significant error (cos_sim < 0.9) ***")
    else:
        print(f"  INT8 looks reasonable")

# ============================================================
# TEST 5: Quantization roundtrip check
# ============================================================
def test_quant_roundtrip():
    print("\n" + "="*60)
    print("TEST 5: Quantization roundtrip (pack → unpack check)")
    print("="*60)
    
    import modiff_cutlass
    
    # Create known input  
    # Values that should survive INT4 quantization: integers in [-7, 7]
    # Shape must be (N, C, H, W) with C >= 2 for packing
    vals = [1., 2., -3., 4., 5., -6., 7., -7.,
            0., 1., -1., 2., -2., 3., -3., 4.,
            5., 6., -5., -4., 3., 2., 1., 0.,
            -1., -2., -3., -4., -5., -6., -7., 7.]
    x_clean = torch.tensor(vals, device='cuda').reshape(1, 32, 1, 1)  # (N=1, C=32, H=1, W=1)
    x_clean = x_clean.contiguous(memory_format=torch.channels_last)
    
    packed = modiff_cutlass.quantize_and_pack(x_clean)
    
    # Manually unpack to verify
    packed_flat = packed.flatten()
    unpacked = []
    for byte_val in packed_flat.cpu().tolist():
        byte_val = byte_val & 0xFF  # unsigned
        low = byte_val & 0x0F
        high = (byte_val >> 4) & 0x0F
        # Sign extend
        if low >= 8: low -= 16
        if high >= 8: high -= 16
        unpacked.extend([low, high])
    
    original = x_clean.flatten().cpu().tolist()
    print(f"  Original: {original[:16]}")
    print(f"  Unpacked: {unpacked[:16]}")
    
    match = all(int(o) == u for o, u in zip(original, unpacked))
    print(f"  Match: {match}")
    if not match:
        diffs = [(i, int(o), u) for i, (o, u) in enumerate(zip(original, unpacked)) if int(o) != u]
        print(f"  Mismatches: {diffs[:10]}")

if __name__ == '__main__':
    print("INT4 Diagnostic Suite")
    print("="*60)
    
    # Test 0: dtype under autocast  
    test_dtype_under_autocast()
    
    # Test 5: quantization roundtrip
    test_quant_roundtrip()
    
    # Load model for remaining tests
    print("\nLoading model...")
    model = load_model()
    
    # Test 1
    test_weight_distributions(model)
    
    # Test 2
    test_single_layer(model)
    
    # Test 3
    test_unet_single_step(model)
    
    # Test 4
    test_int8_sanity(model)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
