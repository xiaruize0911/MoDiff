import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import modiff_cutlass

torch.manual_seed(0)
EMPTY_BIAS = torch.empty(0, device='cuda', dtype=torch.float32)
EMPTY_RES = torch.empty(0, device='cuda', dtype=torch.float16)

# Top real conv shapes by call frequency (from the 20-shape enumeration earlier this session).
SHAPES = [
    (128, 768, 2, 2, 768, 12),
    (128, 384, 8, 8, 384, 8),
    (128, 192, 32, 32, 192, 7),
    (128, 384, 16, 16, 384, 7),
    (128, 768, 4, 4, 768, 7),
]


def make_layer(N, Cin, H, W, Cout):
    x = torch.randn(N, Cin, H, W, device='cuda').to(memory_format=torch.channels_last)
    w_conv = nn.Conv2d(Cin, Cout, 3, padding=1, bias=False).cuda()
    w_data = w_conv.weight.data
    w_flat = w_data.reshape(Cout, -1)
    ch_scale = torch.clamp(w_flat.abs().max(dim=1).values / 127.0, min=1e-8)
    w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    w_quant = w_quant.reshape_as(w_data).permute(0, 2, 3, 1).contiguous()
    weight_scale = ch_scale.contiguous()
    x_int8 = (x / 8.0).round().clamp(-127, 127).to(torch.int8).contiguous(memory_format=torch.channels_last)
    alpha = torch.tensor([1.0 / 16.0], device='cuda')
    o_hat = torch.randn(N, Cout, H, W, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last)
    return dict(x=x_int8, w=w_quant, alpha=alpha, ws=weight_scale, o_hat=o_hat, N=N, Cin=Cin, H=H, W=W, Cout=Cout)


def call_ohat(L):
    modiff_cutlass.conv2d_int8_evt_o_hat(L['x'], L['w'], L['alpha'], L['ws'], L['o_hat'], 1, 1, 1, 1, 1, 1)


layer_sets = [[make_layer(*s[:5]) for _ in range(4)] for s in SHAPES]

for layers in layer_sets:
    for _ in range(5):
        for L in layers:
            call_ohat(L)
torch.cuda.synchronize()

for i, (shape, layers) in enumerate(zip(SHAPES, layer_sets)):
    N, Cin, H, W, Cout, freq = shape
    nvtx.range_push(f"SHAPE_{i}_{Cin}x{Cout}_{H}x{W}")
    for _ in range(30):
        for L in layers:
            call_ohat(L)
    torch.cuda.synchronize()
    nvtx.range_pop()

print("done")
