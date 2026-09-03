"""Just the FIRST kernel of the blockwise conv block: the fused GN(+SiLU) -> quantize.

The blockwise conv block is two kernels -- (1) fused GN+blockwise-quantize, (2) blockk conv.
This benchmarks (1) alone against the shipped per-tensor equivalent, so the two are compared at
identical work: same GN, same SiLU, same output element count, differing only in whether the
scale is per-tensor or per (pixel, C-block).

  int8   shipped: group_norm_silu_quantize_nhwc   vs  gn_silu_blockk_quantize_b32 (B=32 / B=64)
  int4   shipped: group_norm_silu_quantize_pack_nhwc vs gn_silu_blockk_quantize_pack_int4 (B=64)

Frequency-weighted over the 20 churches-UNet conv input shapes, batch 128, CUDA events,
median of 25 after 8 warmup.
"""
import json, os, statistics, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
import modiff_cutlass as mc
DEV, CL = "cuda", torch.channels_last
WARMUP, REPS, BATCH = 8, 25, 128
# (C, H, W, freq) -- the conv INPUT shapes; the norm kernel does not see the output channels
SHAPES=[(768,2,2,12),(384,8,8,8),(192,32,32,7),(384,16,16,7),(768,4,4,7),(1536,2,2,3),
        (1536,4,4,2),(768,8,8,2),(768,16,16,2),(384,32,32,2),(192,16,16,1),(192,16,16,1),
        (384,4,4,1),(384,4,4,1),(1152,4,4,1),(768,8,8,1),(1152,8,8,1),(576,16,16,1),
        (384,32,32,1),(576,32,32,1)]
G=32
def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(REPS):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

out={"int8":{}, "int4":{}}
for C,H,W,freq in SHAPES:
    x=torch.randn(BATCH,C,H,W,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
    w=torch.randn(C,device=DEV,dtype=torch.float16); b=torch.randn(C,device=DEV,dtype=torch.float16)
    sc=torch.full((1,),16.0,device=DEV,dtype=torch.float32)
    ef=torch.empty(0,device=DEV,dtype=torch.float32); eh=torch.empty(0,device=DEV,dtype=torch.float16)
    res={}
    # ---- int8 ----
    res[("int8","shipped")]=bench(lambda: mc.group_norm_silu_quantize_nhwc(x,w,b,G,1e-6,True,sc,ef,eh,eh))
    for blk in (32,64):
        if C%blk: continue
        res[("int8",f"blockwise B={blk}")]=bench(
            lambda blk=blk: mc.gn_silu_blockk_quantize_b32(x,w,b,eh,G,1e-6,True,ef,eh,eh,blk))
    # ---- int4 ----
    try:
        res[("int4","shipped")]=bench(lambda: mc.group_norm_silu_quantize_pack_nhwc(x,w,b,G,1e-6,True,sc,ef,eh,eh,0))
    except Exception: pass
    if C%64==0:
        try:
            res[("int4","blockwise B=64")]=bench(
                lambda: mc.gn_silu_blockk_quantize_pack_int4(x,w,b,G,1e-6,True,ef,eh,eh,64))
        except Exception: pass
    for (tag,name),v in res.items():
        d=out[tag].setdefault(name,{"t":0.0,"f":0})
        d["t"]+=v*freq; d["f"]+=freq
    print(f"  C{C:5d} {H:2d}x{W:<2d} f{freq:<3d} " + "  ".join(
        f"{n}={res[(t,n)]:.4f}" for t,n in res if t=="int8") , flush=True)
    del x,w,b; torch.cuda.empty_cache()
json.dump(out, open("docs/conv_int4_blockk_2026-09-02/data/norm_kernel_bench.json","w"), indent=1)
tf=sum(f for _,_,_,f in SHAPES)
for tag in ("int8","int4"):
    print(f"\n=== {tag}: fused GN+quantize kernel ONLY, freq-weighted (batch 128) ===")
    base=out[tag].get("shipped")
    for n,d in out[tag].items():
        cov=100*d["f"]/tf
        rel=f"{base['t']/d['t']:.3f}x vs shipped" if base and d["f"]==base["f"] else \
            (f"(cov {cov:.0f}%, not directly comparable)" if base else "")
        print(f"  {n:20s} {d['t']:8.3f} ms   cov {cov:5.0f}%   {rel}")
