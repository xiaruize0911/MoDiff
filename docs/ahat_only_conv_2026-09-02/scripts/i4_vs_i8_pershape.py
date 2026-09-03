"""Per-shape i8 B=32 vs i4 B=32 on kernel 1, split by whether the int8 arm gets blk32_vec4
(CPG%4==0).  If i4's deficit vanishes on the CPG%4!=0 shapes, the cause is vectorization;
if it survives, the cause is the nibble datapath."""
import os,sys,json,statistics
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
W,R,G,EPS,BLK=8,25,32,1e-6,32
SHAPES=[(s["C"],s["H"],s["W"],s["B"],s["freq"]) for s in
        json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))["unet"]]
ef=torch.empty(0,device=DEV,dtype=torch.float32); eh=torch.empty(0,device=DEV,dtype=torch.float16)
ei=torch.empty(0,device=DEV,dtype=torch.int32)
def bench(fn):
    for _ in range(W): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(R):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
def one(B,C,H,W_,ahat):
    x=torch.empty(B,C,H,W_,device=DEV,dtype=torch.float16,memory_format=CL).normal_()
    gw=torch.randn(C,device=DEV,dtype=torch.float16); gb=torch.randn(C,device=DEV,dtype=torch.float16)
    sc=torch.full((1,),16.0,device=DEV,dtype=torch.float32)
    chan=C//2 if ahat=="i4" else C
    A=torch.empty(B,chan,H,W_,device=DEV,dtype=torch.int8,memory_format=CL).zero_()
    As=torch.ones(B,H,W_,C//BLK,device=DEV,dtype=torch.float32)
    fn=lambda: mc.group_norm_silu_delta_quantize_nhwc(
        x,gw,gb,A,G,EPS,True,sc,ef,eh,eh,ef,ef,ef,ei,127.0,False,1.0,False,True,As)
    fn(); torch.cuda.synchronize(); t=bench(fn)
    del x,A,As,fn; torch.cuda.empty_cache(); return t
print(f"{'C':>5} {'HxW':>7} {'CPG':>4} {'vec4?':>6} {'i8 B=32':>9} {'i4 B=32':>9} {'i4/i8':>7} {'freq':>5}")
rows=[]
for C,H,W_,B,f in SHAPES:
    cpg=C//G; v4=(cpg%4==0)
    t8=one(B,C,H,W_,"i8"); t4=one(B,C,H,W_,"i4")
    rows.append((C,H,W_,cpg,v4,t8,t4,f))
    print(f"{C:>5} {H}x{W_:<5} {cpg:>4} {'yes' if v4 else 'NO':>6} {t8:9.4f} {t4:9.4f} {t4/t8:7.3f} {f:>5}")
for lab,sel in (("CPG%4==0 (i8 has vec4)",True),("CPG%4!=0 (both vec2)",False)):
    s=[r for r in rows if r[4]==sel]
    a=sum(r[5]*r[7] for r in s); b=sum(r[6]*r[7] for r in s); fr=sum(r[7] for r in s)
    print(f"\n{lab}: freq-weighted i8 {a:.4f}  i4 {b:.4f}  i4/i8 = {b/a:.3f}   (freq {fr})")
json.dump([{"C":r[0],"H":r[1],"W":r[2],"cpg":r[3],"vec4":r[4],"i8":r[5],"i4":r[6],"freq":r[7]} for r in rows],
          open("docs/ahat_only_conv_2026-09-02/data/i4_vs_i8_pershape.json","w"),indent=1)
