"""Fine-grained (bits x block) grid on the REAL captured inputs, to turn the metrics into a
calibrated threshold instead of a 1-vs-10 bracket.

Anchors: 8-bit at B=16/32/64 all produce correct samples; 4-bit collapses. So the decision
boundary is somewhere between, and a 2-point bracket cannot locate it. This sweeps 4..8 bits so
the error curve is continuous; e2e_sim.sh then finds which bit widths actually still sample, and
the two are crossed to read off a threshold on eta_cum.
"""
import os, sys, json, statistics as st
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
CAP = torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV = "cuda"; DLIM = 127.0
BITS = [4, 5, 6, 7, 8]; BLKS = [16, 32, 64]

def gn_silu(x, meta, mod, N, C, H, W):
    G = meta["num_groups"]
    xg = x.float().view(N, G, C//G, H, W)
    mu = xg.mean(dim=(2,3,4), keepdim=True)
    var = xg.var(dim=(2,3,4), unbiased=False, keepdim=True)
    n = ((xg-mu)*(var+meta["eps"]).rsqrt()).view(N, C, H, W)
    n = n*meta["weight"].to(DEV).view(1,C,1,1) + meta["bias"].to(DEV).view(1,C,1,1)
    if mod is not None:
        n = n*(1.0+mod[0].to(DEV).view(N,C,1,1)) + mod[1].to(DEV).view(N,C,1,1)
    n = n.half().float()
    o = n*torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None:
        o = o*meta["smooth_inv"].to(DEV).view(1,C,1,1)
    return o

def blk_q(v, B, bits, N, C, H, W):
    lim = float(2**(bits-1)-1)
    x = v.permute(0,2,3,1).reshape(N,H,W,C//B,B)
    s = x.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
    q = (x/s).round().clamp_(-lim,lim)
    return (q*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous(), (q.abs()==lim).float().mean().item()

res = {}
for name, L in CAP["layers"].items():
    C,H,W,N = L["C"],L["H"],L["W"],L["batch"]; T = L["x"].shape[0]
    o_cache = [gn_silu(L["x"][t].to(DEV), L["meta"], L["mod"][t], N,C,H,W) for t in range(T)]
    # reference: exact fp32 a_hat
    ref, cbar = torch.zeros(N,C,H,W,device=DEV), []
    qbar = []
    for t in range(T):
        q = torch.clamp(torch.round((o_cache[t]-ref)*L["scale"][t]), -DLIM, DLIM)
        ref = ref + q/L["scale"][t]; cbar.append(ref.clone()); qbar.append(q)
    for bits in BITS:
        for B in BLKS:
            if C % B: continue
            a = torch.zeros(N,C,H,W,device=DEV); eta = torch.zeros(N,C,H,W,device=DEV)
            m = {k: [] for k in ("eta_cum","eta_step","consumed","state","codes","sat")}
            for t in range(T):
                s = L["scale"][t]
                q = torch.clamp(torch.round((o_cache[t]-a)*s), -DLIM, DLIM)
                cons = a + q/s
                new, sat = blk_q(cons, B, bits, N,C,H,W)
                eta += new-cons
                d = cbar[t].norm()
                m["eta_cum"].append((eta.norm()/d).item())
                m["eta_step"].append(((new-cons).norm()/d).item())
                m["consumed"].append(((cons-cbar[t])/d.clamp_min(1e-12)).norm().item()*1.0
                                     if False else ((cons-cbar[t]).norm()/d).item())
                m["state"].append(((new-cbar[t]).norm()/d).item())
                m["codes"].append((q!=qbar[t]).float().mean().item())
                m["sat"].append(sat)
                a = new
            res.setdefault(f"{bits}bit B={B}", {})[name] = {
                "eta_cum_final": m["eta_cum"][-1], "eta_cum_max": max(m["eta_cum"]),
                **{k: st.median(v[5:]) for k, v in m.items() if k != "eta_cum"}}
    del o_cache, cbar, qbar; torch.cuda.empty_cache()
json.dump(res, open("docs/ahat_accuracy_2026-09-02/data/bits_grid.json","w"), indent=1)
print(f"{'config':>12} | {'B/elem':>7} | {'eta_cum final':>25} | {'eta_step':>9} | {'codes':>6} | {'sat':>6}")
print(f"{'':>12} | {'':>7} | {'min      median      max':>25} |")
print("-"*90)
for k, per in res.items():
    bits=int(k.split("bit")[0]); B=int(k.split("B=")[1])
    e=[v["eta_cum_final"] for v in per.values()]
    print(f"{k:>12} | {bits/8+4/B:7.3f} | {min(e):8.4f} {st.median(e):9.4f} {max(e):9.4f} | "
          f"{st.median([v['eta_step'] for v in per.values()]):9.4f} | "
          f"{st.median([v['codes'] for v in per.values()]):6.3f} | "
          f"{st.median([v['sat'] for v in per.values()]):6.3f}")
