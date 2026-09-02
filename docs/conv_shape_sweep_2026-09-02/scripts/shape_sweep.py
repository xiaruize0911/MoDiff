"""fp16 vs int8 vs int8-blockwise conv, swept over (B, N, H, W, C) and over the 20 real UNet shapes.

WHY. docs/conv_blockk_e2e_2026-09-02 found the two main W8A8 conv kernels take about the same time
as the two fp16 ones (24.64 vs 23.99 ms/step) -- int8's 2x tensor-core peak does not show up at
this model's shapes. That is a claim about shapes, so it needs a shape sweep.

Four variants per shape, all 3x3 stride 1 pad 1, CUDA events, median of REPS after WARMUP:
  fp16       F.conv2d on channels_last fp16 (cuDNN / CUTLASS picks the kernel)
  int8 EVT   conv2d_int8_evt_bias_residual_fp16 -- the shipped baseline conv, one fused kernel
  blockk ctrl  our tile with a scalar alpha  (isolates the tile from the blockwise dequant)
  blockk B=64  our tile with a blockwise-along-C activation scale

Source setup_cuda_env.sh && python docs/conv_shape_sweep_2026-09-02/scripts/shape_sweep.py
"""
import json, os, statistics, sys
ROOT = "/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0] = [ROOT]
import torch, torch.nn.functional as F
import modiff_cutlass as mc

DEV, CL = "cuda", torch.channels_last
WARMUP, REPS = 8, 25
DEFAULT = dict(B=128, H=16, W=16, N=384, C=384)
SWEEPS = {"B": [8, 16, 32, 64, 128, 256],
          "H": [4, 8, 16, 32, 64],
          "W": [4, 8, 16, 32, 64],
          "N": [128, 192, 256, 384, 512, 768, 1152, 1536],
          "C": [64, 128, 192, 256, 384, 512, 768, 1152, 1536]}
UNET = [(768,768,2,2,12),(384,384,8,8,8),(192,192,32,32,7),(384,384,16,16,7),
        (768,768,4,4,7),(1536,768,2,2,3),(1536,768,4,4,2),(768,384,8,8,2),
        (768,384,16,16,2),(384,192,32,32,2),(192,192,16,16,1),(192,384,16,16,1),
        (384,384,4,4,1),(384,768,4,4,1),(1152,768,4,4,1),(768,768,8,8,1),
        (1152,384,8,8,1),(576,384,16,16,1),(384,384,32,32,1),(576,192,32,32,1)]


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def measure(B, C, H, W, N):
    r = {"B": B, "C": C, "H": H, "W": W, "N": N}
    try:
        xh = torch.randn(B, C, H, W, device=DEV, dtype=torch.float16).contiguous(memory_format=CL)
        wh = torch.randn(N, C, 3, 3, device=DEV, dtype=torch.float16).contiguous(memory_format=CL)
        r["fp16"] = bench(lambda: F.conv2d(xh, wh, None, 1, 1))
        del xh, wh

        x = torch.randint(-8, 8, (B, C, H, W), device=DEV, dtype=torch.int8).contiguous(memory_format=CL)
        w = torch.randint(-8, 8, (N, 3, 3, C), device=DEV, dtype=torch.int8).contiguous()
        inv = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
        ws = torch.full((N,), 0.02, device=DEV, dtype=torch.float32)
        bias = torch.zeros(N, device=DEV, dtype=torch.float32)
        eh = torch.empty(0, device=DEV, dtype=torch.float16)
        ef = torch.empty(0, device=DEV, dtype=torch.float32)
        out = torch.empty(B, N, H, W, device=DEV, dtype=torch.float16).contiguous(memory_format=CL)
        r["int8_evt"] = bench(lambda: mc.conv2d_int8_evt_bias_residual_fp16(
            x, w, inv, ws, bias, eh, out, 1, 1, 1, 1, 1, 1))
        if C % 64 == 0 and N % 2 == 0:
            r["blockk_ctrl"] = bench(lambda: mc.conv2d_int8_blockk(
                x, w, ws, ef, 0.0625, 64, 1, 1, None, None, None))
            sb = (torch.rand(B, H, W, C // 64, device=DEV) * 0.02 + 0.005).float()
            r["blockk_b64"] = bench(lambda: mc.conv2d_int8_blockk(
                x, w, ws, sb, 0.0, 64, 1, 1, None, None, None))
    except RuntimeError as ex:
        r["error"] = str(ex).split("\n")[0][:120]
    torch.cuda.empty_cache()
    return r


out = {"gpu": torch.cuda.get_device_name(0), "default": DEFAULT,
       "method": f"CUDA events, median of {REPS} after {WARMUP} warmup; 3x3 s1 p1",
       "sweeps": {}, "unet": []}
for axis, vals in SWEEPS.items():
    rows = []
    for v in vals:
        cfg = dict(DEFAULT); cfg[axis] = v
        rows.append({**measure(**cfg), "axis": axis, "value": v})
        print(f"  {axis}={v:5d}  " + "  ".join(
            f"{k}={rows[-1][k]:.3f}" for k in ("fp16","int8_evt","blockk_ctrl","blockk_b64")
            if k in rows[-1]), flush=True)
    out["sweeps"][axis] = rows
print("--- UNet shapes ---", flush=True)
for (C, N, H, W, freq) in UNET:
    rr = {**measure(128, C, H, W, N), "freq": freq}
    out["unet"].append(rr)
    print(f"  C{C}->N{N} {H}x{W} f{freq}  " + "  ".join(
        f"{k}={rr[k]:.3f}" for k in ("fp16","int8_evt","blockk_ctrl","blockk_b64") if k in rr), flush=True)
json.dump(out, open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json", "w"), indent=1)
print("wrote data/shape_sweep.json")
