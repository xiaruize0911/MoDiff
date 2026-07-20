"""Enumerate EVERY attention block the churches UNet runs, capture the exact
(num_heads, head_dim, T) each one sees, and count how many blocks share each
shape. Instantiates the UNet from config only (no checkpoint / no VAE needed) and
runs one forward at a chosen batch, hooking the real AttentionBlock modules.

Writes docs/flash_attention_2026-07-19/data/attn_shapes.csv
Usage: python capture_attn_shapes.py [batch]   (default batch=128)
"""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, yaml
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, UNetModel

b = int(sys.argv[1]) if len(sys.argv) > 1 else 128
with open("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml") as f:
    conf = yaml.safe_load(f)
mp = conf["model"]["params"]
up = mp["unet_config"]["params"]
unet = UNetModel(**up).cuda().eval()  # fp32: shapes are dtype-independent

records = []  # (name, C, nh, hd, T)
hooks = []

def mk_hook(name):
    def hook(mod, inp):
        x = inp[0]
        b_, C, *spatial = x.shape
        T = 1
        for s in spatial:
            T *= s
        nh = mod.num_heads
        hd = C // nh
        records.append((name, C, nh, hd, T, b_))
    return hook

for name, m in unet.named_modules():
    if isinstance(m, AttentionBlock):
        hooks.append(m.register_forward_pre_hook(mk_hook(name)))

img = mp["image_size"]        # latent spatial (32)
ch = up["in_channels"]         # 4
x = torch.randn(b, ch, img, img, device="cuda")
t = torch.randint(0, 1000, (b,), device="cuda")
with torch.no_grad():
    unet(x, t)
for h in hooks:
    h.remove()

# Aggregate by (C, nh, hd, T)
from collections import Counter, defaultdict
counts = Counter()
examples = defaultdict(list)
for name, C, nh, hd, T, b_ in records:
    key = (C, nh, hd, T)
    counts[key] += 1
    examples[key].append(name)

print(f"UNet churches — {len(records)} AttentionBlocks total at batch={b}\n")
print(f"{'C':>5} {'nh':>3} {'hd':>4} {'T':>6} {'BH=b*nh':>8} {'count':>6}  eligible(flash-quant)")
rows = []
for key in sorted(counts, key=lambda k: -k[3]):
    C, nh, hd, T = key
    BH = b * nh
    elig = (T % 64 == 0) and (hd <= 48)
    print(f"{C:5d} {nh:3d} {hd:4d} {T:6d} {BH:8d} {counts[key]:6d}  {'YES' if elig else 'no'}")
    rows.append(dict(C=C, nh=nh, hd=hd, T=T, batch=b, BH=BH, count=counts[key],
                     flash_quant_eligible=int(elig)))

os.makedirs("docs/flash_attention_2026-07-19/data", exist_ok=True)
with open("docs/flash_attention_2026-07-19/data/attn_shapes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nTotal blocks = {sum(counts.values())}  |  unique shapes = {len(counts)}")
print("WROTE data/attn_shapes.csv")
