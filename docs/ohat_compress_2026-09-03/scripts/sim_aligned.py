"""SIM: window-frozen a_hat block scales + feeding the conv (a_hat_t - a_hat_{t-1}) instead of the
delta codes.  Question: does the accumulated term drop from T eta's to T/K eta's, and where is the
optimum K once scale staleness is paid for?

Arms, all at 8-bit a_hat B=32 unless swept:
  current        block scale re-derived every step; conv is fed q_t/s_t (the codes).
                 accumulation mismatch = sum_t eta_t   (this is REPORT.md's eta_cum)
  aligned K      block scale FROZEN for K steps; conv is fed (a_hat_t - a_hat_{t-1}), which inside
                 a window is an exact integer multiple of the frozen scale.  Increment clamped to
                 the int8 code range and DROPPED on overflow.
  aligned K +c   same, but the clamped remainder is CARRIED into the next step (error feedback in
                 exact code units -- free, the residual is an integer).

Metrics (both normalised by the exact-a_hat reference at t=T):
  acc_err   || sum_t inc_t  -  consumed_T(own) ||   the accumulation mismatch, i.e. exactly what
            the conv output carries.  Apples-to-apples across arms.
  cons_err  || consumed_T(own) - ref_T ||          the delta quantizer's own floor; should be flat.
  clip_a    fraction of a_hat codes clamped (scale staleness)
  clip_i    fraction of increments clamped (window drift)

This is the PyTorch model of the recurrence, validated against the real CUDA kernel to 1e-4 in
REPORT.md section 3; the real kernel does not implement the aligned recursion, so a model is the
only way to price it before building it.
"""
import os, sys, json, math
ROOT = "/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0] = [ROOT]
import torch

PREC = "int8"
CAP = torch.load(f"docs/ahat_accuracy_2026-09-02/data/capture_{PREC}.pt", weights_only=False)
DEV = "cuda"
DLIM = 127.0            # delta / increment code limit
ALIM = 127.0            # a_hat code limit at 8 bit
KS = [1, 2, 5, 7, 10, 17, 25, 49]
BITS = [8]
BLK = 32

def gn_silu(x, meta, mod):
    G = meta["num_groups"]; N, C, H, W = x.shape
    xg = x.view(N, G, C // G, H, W)
    mu = xg.mean(dim=(2, 3, 4), keepdim=True)
    var = xg.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
    n = ((xg - mu) * (var + meta["eps"]).rsqrt()).view(N, C, H, W)
    n = n * meta["weight"].to(DEV).view(1, C, 1, 1) + meta["bias"].to(DEV).view(1, C, 1, 1)
    if mod is not None:
        ms, sh = mod
        n = n * (1.0 + ms.to(DEV).view(N, C, 1, 1)) + sh.to(DEV).view(N, C, 1, 1)
    n = n.half().float()
    o = n * torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None:
        o = o * meta["smooth_inv"].to(DEV).view(1, C, 1, 1)
    return o

def blk_scale(v, blk, lim):
    """Per-(n,h,w,block-of-C) symmetric amax scale, the kernel's scheme."""
    N, C, H, W = v.shape
    vb = v.permute(0, 2, 3, 1).reshape(N, H, W, C // blk, blk)
    amax = vb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return (amax / lim)                                 # [N,H,W,C/blk,1]

def blk_q(v, S, blk, lim):
    """Quantize to codes on the given per-block scale, and return (codes, dequantized)."""
    N, C, H, W = v.shape
    vb = v.permute(0, 2, 3, 1).reshape(N, H, W, C // blk, blk)
    code = torch.clamp(torch.round(vb / S), -lim, lim)
    deq = (code * S).reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
    return code, deq

out = {}
for name, L in CAP["layers"].items():
    C, H, W, N = L["C"], L["H"], L["W"], L["batch"]
    T = L["x"].shape[0]; meta, scales, mods = L["meta"], L["scale"], L["mod"]
    if C % BLK: continue
    O = []                                              # the true GN+SiLU output per step
    for t in range(T):
        O.append(gn_silu(L["x"][t].to(DEV).float(), meta, mods[t]))
    # ---- reference: exact fp32 a_hat, conv fed the codes ----
    ra = torch.zeros(N, C, H, W, device=DEV); ref_cons = []
    for t in range(T):
        s = scales[t]
        q = torch.clamp(torch.round((O[t] - ra) * s), -DLIM, DLIM)
        ra = ra + q / s; ref_cons.append(ra.clone())
    refn = ref_cons[-1].norm()

    rows = {}
    # ---------------- arm: current (per-step scale, conv fed the codes) ----------------
    a_deq = torch.zeros(N, C, H, W, device=DEV)
    inc_sum = torch.zeros(N, C, H, W, device=DEV)
    eta_sum = torch.zeros(N, C, H, W, device=DEV)
    cur = []
    for t in range(T):
        s = scales[t]
        q = torch.clamp(torch.round((O[t] - a_deq) * s), -DLIM, DLIM)
        cons = a_deq + q / s
        inc_sum += q / s
        S = blk_scale(cons, BLK, ALIM)
        _, a_deq = blk_q(cons, S, BLK, ALIM)
        eta_sum += (a_deq - cons)
        cur.append({"t": t, "acc_err": ((inc_sum - cons).norm() / refn).item(),
                    "eta_cum": (eta_sum.norm() / refn).item(),
                    "cons_err": ((cons - ref_cons[t]).norm() / refn).item()})
    rows["current"] = cur

    # ---------------- arm: aligned, window K (+/- carry) ----------------
    for K in KS:
        for carry_on in (False, True):
            a_deq = torch.zeros(N, C, H, W, device=DEV)
            a_code = torch.zeros(N, H, W, C // BLK, BLK, device=DEV)
            S = None; inc_sum = torch.zeros(N, C, H, W, device=DEV)
            carry = torch.zeros(N, H, W, C // BLK, BLK, device=DEV)
            ca = ci = 0.0; per = []
            for t in range(T):
                s = scales[t]
                q = torch.clamp(torch.round((O[t] - a_deq) * s), -DLIM, DLIM)
                cons = a_deq + q / s
                if t % K == 0:                          # window boundary: re-derive the scale
                    Snew = blk_scale(cons, BLK, ALIM)
                    if S is not None:                   # re-express the old state on the new grid
                        a_code = torch.round(a_code * S / Snew)
                    S = Snew
                code = torch.clamp(torch.round(cons.permute(0, 2, 3, 1)
                                   .reshape(N, H, W, C // BLK, BLK) / S), -ALIM, ALIM)
                ca += (code.abs() == ALIM).float().mean().item()
                inc = code - a_code + (carry if carry_on else 0.0)
                inc_c = torch.clamp(inc, -DLIM, DLIM)
                ci += (inc_c != inc).float().mean().item()
                if carry_on: carry = inc - inc_c
                a_code = a_code + inc_c                 # what the conv actually accumulated
                inc_sum += (inc_c * S).reshape(N, H, W, C).permute(0, 3, 1, 2)
                a_deq = (a_code * S).reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
                per.append({"t": t, "acc_err": ((inc_sum - cons).norm() / refn).item(),
                            "cons_err": ((cons - ref_cons[t]).norm() / refn).item()})
            rows[f"aligned K={K}" + ("+c" if carry_on else "")] = per
            rows[f"aligned K={K}" + ("+c" if carry_on else "")][-1]["clip_a"] = ca / T
            rows[f"aligned K={K}" + ("+c" if carry_on else "")][-1]["clip_i"] = ci / T
    out[name] = {"C": C, "H": H, "W": W, "batch": N, "steps": T, "arms": rows}
    b = rows["current"][-1]
    print(f"{name}: current acc_err={b['acc_err']:.4f} (eta_cum={b['eta_cum']:.4f})  " +
          "  ".join(f"K{K}={rows[f'aligned K={K}'][-1]['acc_err']:.4f}"
                    f"/{rows[f'aligned K={K}+c'][-1]['acc_err']:.4f}" for K in KS), flush=True)
    del O, ref_cons; torch.cuda.empty_cache()
json.dump(out, open("docs/ohat_compress_2026-09-03/data/sim_aligned.json", "w"))
print("\nwrote data/sim_aligned.json")
