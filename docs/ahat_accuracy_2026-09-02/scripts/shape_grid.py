"""Does the (bits x block) behaviour depend on SHAPE, and along which axis?

The 5 captured layers differ in shape AND in content, so their spread cannot separate the two.
This sweeps N/C/H/W one at a time on synthetic input (walk trajectory), holding everything else
fixed, and evaluates a 3x3 (bits x B) grid at each shape point.

Predictions being tested:
  - eta_cum is a RELATIVE norm, so adding independent samples (N, H, W) should not move its mean,
    only its scatter => flat in N, H, W.
  - C could matter structurally: GN has 32 groups, so channels-per-group CPG = C/32. A block with
    B > CPG spans several GN groups (whose means/scales differ); B < CPG sits inside one. That
    boundary is a real candidate for a shape x block interaction.
"""
import os, sys, json, math, statistics as st
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
DEV="cuda"; G=32; DLIM=127.0; T=49; S_DELTA=8.0
DEFAULT={"N":4,"C":384,"H":16,"W":16}
SWEEP={"N":[1,2,4,8,16],"C":[128,192,256,384,512,768,1024,1536],
       "H":[2,4,8,16,32,64],"W":[2,4,8,16,32,64]}
BITS=[4,6,8]; BLKS=[16,32,64]

def gn_silu(x,gw,gb,N,C,H,W):
    xg=x.float().view(N,G,C//G,H,W)
    mu=xg.mean(dim=(2,3,4),keepdim=True); var=xg.var(dim=(2,3,4),unbiased=False,keepdim=True)
    n=((xg-mu)*(var+1e-6).rsqrt()).view(N,C,H,W)
    n=(n*gw.float().view(1,C,1,1)+gb.float().view(1,C,1,1)).half().float()
    return n*torch.sigmoid(n)
def blk_q(v,B,bits,N,C,H,W):
    lim=float(2**(bits-1)-1); x=v.permute(0,2,3,1).reshape(N,H,W,C//B,B)
    s=x.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
    return ((x/s).round().clamp_(-lim,lim)*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous()

def run(N,C,H,W):
    torch.manual_seed(11)
    gw=torch.randn(C,device=DEV,dtype=torch.float16); gb=torch.randn(C,device=DEV,dtype=torch.float16)
    x=torch.randn(N,C,H,W,device=DEV,dtype=torch.float16)
    xs=[]
    for t in range(T):
        x=(math.sqrt(0.98)*x.float()+math.sqrt(0.02)*torch.randn(N,C,H,W,device=DEV)).half()
        xs.append(x.clone())
    o=[gn_silu(v,gw,gb,N,C,H,W) for v in xs]
    ref=torch.zeros(N,C,H,W,device=DEV); nrm=None
    for t in range(T):
        q=torch.clamp(torch.round((o[t]-ref)*S_DELTA),-DLIM,DLIM); ref=ref+q/S_DELTA
    nrm=ref.norm()
    out={}
    for bits in BITS:
        for B in BLKS:
            if C%B: continue
            a=torch.zeros(N,C,H,W,device=DEV); eta=torch.zeros(N,C,H,W,device=DEV)
            for t in range(T):
                q=torch.clamp(torch.round((o[t]-a)*S_DELTA),-DLIM,DLIM); cons=a+q/S_DELTA
                new=blk_q(cons,B,bits,N,C,H,W); eta+=new-cons; a=new
            out[f"{bits}|{B}"]=(eta.norm()/nrm).item()
    del o,xs; torch.cuda.empty_cache()
    return out

res={}
for axis,vals in SWEEP.items():
    for v in vals:
        cfg=dict(DEFAULT); cfg[axis]=v
        key=f"{axis}={v}"
        if key in res: continue
        res[key]={"axis":axis,"value":v,**cfg,"eta":run(cfg["N"],cfg["C"],cfg["H"],cfg["W"])}
        print(f"  {key:<8} " + "  ".join(
            f"{k}={x:.4f}" for k,x in res[key]["eta"].items() if k in ("4|32","6|32","8|32")),
            flush=True)
json.dump(res,open("docs/ahat_accuracy_2026-09-02/data/shape_grid.json","w"),indent=1)

print("\n=== eta_cum，按轴展开（8bit B=32 / 6bit B=32 / 4bit B=32）===")
for axis in SWEEP:
    rows=[r for r in res.values() if r["axis"]==axis]
    rows.sort(key=lambda r:r["value"])
    print(f"\n{axis}: " + "  ".join(f"{r['value']}" for r in rows))
    for bits in BITS:
        vals=[r["eta"].get(f"{bits}|32") for r in rows]
        v=[x for x in vals if x is not None]
        print(f"  {bits}bit B=32 " + "  ".join(f"{x:.4f}" if x else "  --  " for x in vals) +
              f"   | 极差 {max(v)/min(v):.2f}x")
print("\n=== B 的相对效应是否随 shape 变化？（eta(B=64)/eta(B=16)，8bit）===")
for axis in SWEEP:
    rows=sorted([r for r in res.values() if r["axis"]==axis],key=lambda r:r["value"])
    rr=[(r["value"],r["eta"].get("8|64"),r["eta"].get("8|16")) for r in rows]
    print(f"  {axis:<2}: " + "  ".join(f"{v}:{a/b:.3f}" if a and b else f"{v}:--" for v,a,b in rr))
print("\n=== bits 的相对效应是否随 shape 变化？（eta(4bit)/eta(8bit)，B=32）===")
for axis in SWEEP:
    rows=sorted([r for r in res.values() if r["axis"]==axis],key=lambda r:r["value"])
    rr=[(r["value"],r["eta"].get("4|32"),r["eta"].get("8|32")) for r in rows]
    print(f"  {axis:<2}: " + "  ".join(f"{v}:{a/b:.1f}" if a and b else f"{v}:--" for v,a,b in rr))
