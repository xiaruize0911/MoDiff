"""Benchmark every config in conv2d_blockk_tune's table, int8 and int4, frequency-weighted over
the churches-UNet conv shapes.

GOAL: blockwise conv within 1.25x of the SHIPPED per-tensor conv (>= 80% of its speed).
Each config is also run with a scalar alpha -- same kernel, same tile, dequant off -- so the
report separates "our tile" from "blockwise". Coverage is reported per config, because a config
with a large TK or BLK is eligible on fewer shapes and its weighted total is then not comparable
to a config that ran everywhere; the fp16 and shipped references are recomputed over exactly the
shapes each config covered.
"""
import json, os, statistics, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, torch.nn.functional as F
import modiff_cutlass as mc
DEV, CL = "cuda", torch.channels_last
WARMUP, REPS = 6, 20
UNET=[(768,768,2,2,12),(384,384,8,8,8),(192,192,32,32,7),(384,384,16,16,7),
      (768,768,4,4,7),(1536,768,2,2,3),(1536,768,4,4,2),(768,384,8,8,2),
      (768,384,16,16,2),(384,192,32,32,2),(192,192,16,16,1),(192,384,16,16,1),
      (384,384,4,4,1),(384,768,4,4,1),(1152,768,4,4,1),(768,768,8,8,1),
      (1152,384,8,8,1),(576,384,16,16,1),(384,384,32,32,1),(576,192,32,32,1)]
BATCH=128
NC=mc.blockk_tune_num_cfgs()
NAMES=[mc.blockk_tune_cfg_name(i) for i in range(NC)]
def parse(nm):
    tk=int(nm.split("K")[1].split("_")[0]); blk=int(nm.split("_B")[1])
    return tk, blk
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

out={"names":NAMES,"int8":{},"int4":{}}
for i4 in (False,True):
    tag="int4" if i4 else "int8"
    EPB=2 if i4 else 1
    print(f"\n===== {tag} =====",flush=True)
    for (C,N,H,W,freq) in UNET:
        xh=torch.randn(BATCH,C,H,W,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
        wh=torch.randn(N,C,3,3,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
        t_fp16=bench(lambda: F.conv2d(xh,wh,None,1,1)); del xh,wh
        if i4:
            a=torch.randint(-7,8,(BATCH,H,W,C),device=DEV,dtype=torch.int8); xp=pack(a)
            w=torch.randint(-7,8,(N,3,3,C),device=DEV,dtype=torch.int8); wp=pack(w)
            sc=torch.full((1,),0.02,device=DEV,dtype=torch.float32)
            eb=torch.empty(0,device=DEV,dtype=torch.float16)
            t_ship=bench(lambda: mc.conv2d_int4_fprop(xp,wp,sc,eb,1,1,1,1,1,1))
        else:
            xp=torch.randint(-127,128,(BATCH,C,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
            wp=torch.randint(-127,128,(N,3,3,C),device=DEV,dtype=torch.int8).contiguous()
            inv=torch.tensor([1/16.],device=DEV,dtype=torch.float32)
            wsf=torch.full((N,),0.02,device=DEV,dtype=torch.float32)
            bi=torch.zeros(N,device=DEV,dtype=torch.float32)
            eb=torch.empty(0,device=DEV,dtype=torch.float16)
            o=torch.empty(BATCH,N,H,W,device=DEV,dtype=torch.float16).contiguous(memory_format=CL)
            t_ship=bench(lambda: mc.conv2d_int8_evt_bias_residual_fp16(xp,wp,inv,wsf,bi,eb,o,1,1,1,1,1,1))
        ws=torch.full((N,),0.02,device=DEV,dtype=torch.float32)
        ef=torch.empty(0,device=DEV,dtype=torch.float32)
        for cfg in range(NC):
            tk,blk=parse(NAMES[cfg])
            if C % (tk*EPB) or C % blk: continue
            if i4 and blk < 64: continue
            try:
                sb=(torch.rand(BATCH,H,W,C//blk,device=DEV)*0.02+0.005).float()
                tb=bench(lambda: mc.conv2d_blockk_tune(xp,wp,ws,sb,0.0,cfg,i4,1,1))
                tc=bench(lambda: mc.conv2d_blockk_tune(xp,wp,ws,ef,0.02,cfg,i4,1,1))
                d=out[tag].setdefault(str(cfg),{"bw":0.,"ctrl":0.,"fp16":0.,"ship":0.,"f":0})
                d["bw"]+=tb*freq; d["ctrl"]+=tc*freq
                d["fp16"]+=t_fp16*freq; d["ship"]+=t_ship*freq; d["f"]+=freq
                del sb
            except Exception:
                pass
        del xp,wp; torch.cuda.empty_cache()
        print(f"  C{C}->N{N} {H}x{W} done",flush=True)
    out[tag]["_total_freq"]=sum(s[4] for s in UNET)
json.dump(out, open("docs/conv_int4_blockk_2026-09-02/data/tune_sweep.json","w"), indent=1)
print("\nwrote data/tune_sweep.json")
