"""int4 analogue of docs/conv_shape_sweep_2026-09-02: fp16 vs CUTLASS int4 vs our tile
(scalar) vs our tile (blockwise B=64), over the 20 real churches-UNet conv shapes.

Conv kernels only, matching the int8 sweep's convention. The standalone blockwise
quantize+pack kernel is timed separately -- the shipped path fuses its quantize into GN, so
mixing them into one number would compare different amounts of work.
"""
import json, os, statistics, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, torch.nn.functional as F
import modiff_cutlass as mc
DEV, CL = "cuda", torch.channels_last
WARMUP, REPS = 8, 25
UNET=[(768,768,2,2,12),(384,384,8,8,8),(192,192,32,32,7),(384,384,16,16,7),
      (768,768,4,4,7),(1536,768,2,2,3),(1536,768,4,4,2),(768,384,8,8,2),
      (768,384,16,16,2),(384,192,32,32,2),(192,192,16,16,1),(192,384,16,16,1),
      (384,384,4,4,1),(384,768,4,4,1),(1152,768,4,4,1),(768,768,8,8,1),
      (1152,384,8,8,1),(576,384,16,16,1),(384,384,32,32,1),(576,192,32,32,1)]

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(REPS):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

def pack(q):
    lo=q[...,0::2].to(torch.int16)&0xF; hi=q[...,1::2].to(torch.int16)&0xF
    return ((hi<<4)|lo).to(torch.uint8).view(torch.int8).contiguous()

out={"gpu":torch.cuda.get_device_name(0),"method":f"CUDA events, median of {REPS}","unet":[]}
B=128
for (C,N,H,W,freq) in UNET:
    r={"C":C,"N":N,"H":H,"W":W,"freq":freq}
    try:
        xh=torch.randn(B,C,H,W,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
        wh=torch.randn(N,C,3,3,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
        r["fp16"]=bench(lambda: F.conv2d(xh,wh,None,1,1))
        a=torch.randint(-7,8,(B,H,W,C),device=DEV,dtype=torch.int8); xp=pack(a)
        w=torch.randint(-7,8,(N,3,3,C),device=DEV,dtype=torch.int8); wp=pack(w)
        ws=torch.full((N,),0.02,device=DEV,dtype=torch.float32)
        sc=torch.full((1,),0.02,device=DEV,dtype=torch.float32)
        eb=torch.empty(0,device=DEV,dtype=torch.float16)
        ef=torch.empty(0,device=DEV,dtype=torch.float32)
        r["cutlass_int4"]=bench(lambda: mc.conv2d_int4_fprop(xp,wp,sc,eb,1,1,1,1,1,1))
        if C%128==0 and N%2==0:
            r["blockk_ctrl"]=bench(lambda: mc.conv2d_int4_blockk(xp,wp,ws,ef,0.02,64,1,1,None,None,None))
            sb=(torch.rand(B,H,W,C//64,device=DEV)*0.02+0.005).float()
            r["blockk_b64"]=bench(lambda: mc.conv2d_int4_blockk(xp,wp,ws,sb,0.0,64,1,1,None,None,None))
            xq=torch.randn(B,C,H,W,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
            r["quant_pack"]=bench(lambda: mc.conv_quantize_block_pack_int4(xq,64))
        else:
            r["skip"]="C%128!=0"
    except RuntimeError as ex:
        r["error"]=str(ex).split("\n")[0][:110]
    out["unet"].append(r)
    print("  C%-5d->N%-5d %2dx%-2d f%-3d " % (C,N,H,W,freq) + "  ".join(
        f"{k}={r[k]:.3f}" for k in ("fp16","cutlass_int4","blockk_ctrl","blockk_b64","quant_pack")
        if k in r) + ("   " + r.get("skip","") if "skip" in r else ""), flush=True)
    torch.cuda.empty_cache()
json.dump(out, open("docs/conv_int4_blockk_2026-09-02/data/int4_shape_sweep.json","w"), indent=1)
print("wrote data/int4_shape_sweep.json")
