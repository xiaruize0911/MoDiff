"""Spectrum, accumulator-pair recursion, residual bit-budget, SVD cost.

Reads data/capture.pt from capture.py. No extra generation.

    python docs/ahat_svd_residual_2026-09-01/scripts/analyze.py
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
KS = (4, 8, 16, 32)
QMAX = 127.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_as_mat(x: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) -> (N,C,HW)."""
    n, c, h, w = x.shape
    return x.reshape(n, c, h * w)


def svd_factors(x: torch.Tensor, k: int):
    """Batched SVD of C×HW. Zeros → identity-like (recon=x, empty factors)."""
    n, c, h, w = x.shape
    m = _stack_as_mat(x)
    k_use = min(k, c, h * w)
    if float(x.abs().max()) < 1e-12:
        return x, None, None, None, k_use
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    uk = u[:, :, :k_use].contiguous()
    sk = s[:, :k_use].contiguous()
    vhk = vh[:, :k_use, :].contiguous()
    recon = (uk * sk.unsqueeze(1)) @ vhk
    return recon.reshape(n, c, h, w), uk, sk, vhk, k_use


def project_uv(x: torch.Tensor, uk: torch.Tensor, vhk: torch.Tensor) -> torch.Tensor:
    """U U^T X V V^T onto the frozen rank-k subspace."""
    n, c, h, w = x.shape
    m = _stack_as_mat(x)
    # coords = U^T X V  -> (N,k,k) but V = Vh^T so X V = M @ Vh^T
    xv = m @ vhk.transpose(-1, -2)
    coords = uk.transpose(-1, -2) @ xv
    recon = uk @ (coords @ vhk)
    return recon.reshape(n, c, h, w)


def energy_ratios(x: torch.Tensor, ks=KS):
    """Per-sample cumulative energy; returns median over N for each k."""
    m = _stack_as_mat(x.float())
    # skip degenerate
    if float(x.abs().max()) < 1e-12:
        return {k: 0.0 for k in ks}
    _, s, _ = torch.linalg.svd(m, full_matrices=False)
    tot = (s * s).sum(dim=-1).clamp_min(1e-30)
    out = {}
    for k in ks:
        kk = min(k, s.shape[-1])
        part = (s[:, :kk] * s[:, :kk]).sum(dim=-1)
        frac = (part / tot).median().item()
        out[k] = frac
    return out


def quantize(delta: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.clamp(torch.round(delta * scale), -QMAX, QMAX)
    return q / scale


def pack_int8(r: torch.Tensor) -> torch.Tensor:
    amax = r.abs().amax().clamp_min(1e-12)
    s = amax / QMAX
    q = torch.clamp(torch.round(r / s), -QMAX, QMAX)
    return q * s


def pack_sparse(r: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """Keep the largest |R| fraction (per tensor), zero the rest."""
    flat = r.reshape(-1)
    nkeep = max(1, int(round(keep_frac * flat.numel())))
    if nkeep >= flat.numel():
        return r
    thresh = torch.topk(flat.abs(), nkeep, largest=True).values[-1]
    return torch.where(r.abs() >= thresh, r, torch.zeros_like(r))


def replay_prod(tgts, scales):
    a = torch.zeros_like(tgts[0])
    acc = a.clone()
    rows = []
    for t, (tgt, sc) in enumerate(zip(tgts, scales)):
        d = tgt - a
        deq = quantize(d, sc)
        acc = acc + deq
        a = (a + deq).half().float()
        rows.append({
            "step": t + 1,
            "d_absmax": d.abs().max().item(),
            "resid": (a - acc).abs().max().item(),
            "tgt_acc": (tgt - acc).abs().max().item(),
        })
    return rows, a


def replay_s1(tgts, scales, k):
    """Naive SVD cache: ref = SVD_k(stored); o_hat follows codes."""
    a = torch.zeros_like(tgts[0])
    acc = a.clone()
    rows = []
    for t, (tgt, sc) in enumerate(zip(tgts, scales)):
        ref, *_ = svd_factors(a, k)
        d = tgt - ref
        deq = quantize(d, sc)
        acc = acc + deq
        a = (ref + deq).half().float()
        rows.append({
            "step": t + 1,
            "d_absmax": d.abs().max().item(),
            "resid": (a - acc).abs().max().item(),
            "tgt_acc": (tgt - acc).abs().max().item(),
        })
    return rows


def replay_s2(tgts, scales, k):
    """Subspace MoDiff: freeze U,V after step 1; encode projected delta; pair synced."""
    a = torch.zeros_like(tgts[0])
    acc = a.clone()
    uk = vhk = None
    rows = []
    for t, (tgt, sc) in enumerate(zip(tgts, scales)):
        if uk is None:
            d = tgt - a
            deq = quantize(d, sc)
            a_full = (a + deq).half().float()
            a, uk, _, vhk, _ = svd_factors(a_full, k)
            acc = a.clone()  # drop orthogonal tail of the first increment
            d_used = d
        else:
            d = tgt - a
            d_k = project_uv(d, uk, vhk)
            deq = quantize(d_k, sc)
            a = (a + deq).half().float()
            acc = acc + deq
            d_used = d_k
        rows.append({
            "step": t + 1,
            "d_absmax": d_used.abs().max().item(),
            "d_full_absmax": d.abs().max().item(),
            "resid": (a - acc).abs().max().item(),
            "tgt_acc": (tgt - acc).abs().max().item(),
        })
    return rows


def replay_s3(tgts, scales, k, r_mode: str, ohat: str):
    """A_k + R. r_mode: fp16 | int8 | drop | sparse10. ohat: codes | drecon."""
    recon = torch.zeros_like(tgts[0])
    acc = recon.clone()
    rows = []
    for t, (tgt, sc) in enumerate(zip(tgts, scales)):
        d = tgt - recon
        deq = quantize(d, sc)
        full_new = (recon + deq).half().float()
        ak, *_ = svd_factors(full_new, k)
        r_exact = full_new - ak
        if r_mode == "fp16":
            r_store = r_exact
        elif r_mode == "int8":
            r_store = pack_int8(r_exact)
        elif r_mode == "drop":
            r_store = torch.zeros_like(r_exact)
        elif r_mode == "sparse10":
            r_store = pack_sparse(r_exact, 0.10)
        else:
            raise ValueError(r_mode)
        recon_new = (ak + r_store).half().float()
        if ohat == "codes":
            acc = acc + deq
        elif ohat == "drecon":
            acc = acc + (recon_new - recon)
        else:
            raise ValueError(ohat)
        recon = recon_new
        rows.append({
            "step": t + 1,
            "d_absmax": d.abs().max().item(),
            "resid": (recon - acc).abs().max().item(),
            "tgt_acc": (tgt - acc).abs().max().item(),
            "r_absmax": r_exact.abs().max().item(),
            "r_rms": r_exact.pow(2).mean().sqrt().item(),
        })
    return rows


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def summarize(rows):
    return {
        "final_resid": rows[-1]["resid"],
        "final_tgt_acc": rows[-1]["tgt_acc"],
        "median_d_absmax": median([r["d_absmax"] for r in rows]),
        "resid_step1": rows[0]["resid"],
        "resid_step2": rows[1]["resid"] if len(rows) > 1 else None,
        "resid_mid": rows[len(rows) // 2]["resid"],
        "curve": [{"step": r["step"], "resid": r["resid"],
                   "tgt_acc": r["tgt_acc"], "d_absmax": r["d_absmax"]}
                  for r in rows],
    }


def bits_needed(rng, quantum):
    if quantum <= 0 or rng <= 0:
        return 0.0
    levels = 2.0 * rng / quantum
    return math.log2(max(levels, 1.0))


def layer_spectrum_and_budget(name, bundle, k_list=KS):
    tgts = bundle["tgt"].to(DEVICE).float()
    ahats = bundle["ahat"].to(DEVICE).float()
    scales = bundle["scales"]
    t_steps, n, c, h, w = tgts.shape

    spec_ahat, spec_delta, spec_pca = [], [], []
    budget_rows = []
    for ti in range(t_steps):
        tgt = tgts[ti]
        ah = ahats[ti]
        d = tgt - ah
        spec_ahat.append(energy_ratios(ah, k_list))
        spec_delta.append(energy_ratios(d, k_list))
        # channel PCA: covariance over (N*HW) of C dims
        x = ah.permute(0, 2, 3, 1).reshape(-1, c)
        x = x - x.mean(0, keepdim=True)
        # eigendecomposition of CxC via SVD of the data
        # economy SVD of (N*HW)×C
        if x.shape[0] >= 2 and float(x.abs().max()) > 1e-12:
            _, s, _ = torch.linalg.svd(x, full_matrices=False)
            tot = (s * s).sum().clamp_min(1e-30)
            pca = {}
            for k in k_list:
                kk = min(k, s.numel())
                pca[k] = float((s[:kk] * s[:kk]).sum() / tot)
            spec_pca.append(pca)
        quantum = 1.0 / max(scales[ti], 1e-12)
        a_range = ah.abs().max().item()
        row_b = {
            "step": int(bundle["steps"][ti]),
            "scale": scales[ti],
            "quantum": quantum,
            "ahat_range": a_range,
            "ahat_bits": bits_needed(a_range, quantum),
            "k": {},
        }
        for k in k_list:
            ak, *_ = svd_factors(ah, k)
            r = ah - ak
            r_range = r.abs().max().item()
            row_b["k"][str(k)] = {
                "r_range": r_range,
                "r_rms": r.pow(2).mean().sqrt().item(),
                "r_bits": bits_needed(r_range, quantum),
                "energy": spec_ahat[-1][k],
            }
        budget_rows.append(row_b)

    def med_k(series, k):
        return median([s[k] for s in series])

    spec = {
        "n_steps": t_steps,
        "ahat_energy_median": {str(k): med_k(spec_ahat, k) for k in k_list},
        "delta_energy_median": {str(k): med_k(spec_delta, k) for k in k_list},
        "channel_pca_energy_median": {str(k): med_k(spec_pca, k) for k in k_list} if spec_pca else {},
        "per_step_ahat": [{str(k): s[k] for k in k_list} for s in spec_ahat],
        "per_step_delta": [{str(k): s[k] for k in k_list} for s in spec_delta],
    }
    # tail bits: last third of schedule
    tail = budget_rows[max(1, 2 * t_steps // 3):]
    budget = {
        "ahat_bits_median": median([r["ahat_bits"] for r in budget_rows]),
        "ahat_bits_tail_median": median([r["ahat_bits"] for r in tail]),
        "r_bits_median": {
            str(k): median([r["k"][str(k)]["r_bits"] for r in budget_rows]) for k in k_list
        },
        "r_bits_tail_median": {
            str(k): median([r["k"][str(k)]["r_bits"] for r in tail]) for k in k_list
        },
        "r_range_median": {
            str(k): median([r["k"][str(k)]["r_range"] for r in budget_rows]) for k in k_list
        },
        "per_step": budget_rows,
    }
    return spec, budget, tgts, scales


def svd_cost_bench():
    """A40 timing: dense SVD / svd_lowrank vs 2 ms a_hat-write ceiling."""
    if DEVICE.type != "cuda":
        return {"skipped": True}
    shapes = [(128, 192, 32, 32), (128, 192, 16, 16), (128, 384, 16, 16)]
    out = []
    for shape in shapes:
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        m = x.reshape(shape[0], shape[1], shape[2] * shape[3])
        # warmup
        for _ in range(3):
            torch.linalg.svd(m, full_matrices=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nrep = 5
        for _ in range(nrep):
            torch.linalg.svd(m, full_matrices=False)
        torch.cuda.synchronize()
        ms_full = (time.perf_counter() - t0) * 1000 / nrep
        row = {"shape": f"{shape[1]},{shape[2]}x{shape[3]}", "batch": shape[0],
               "svd_full_ms": round(ms_full, 3)}
        for k in (8, 16):
            q = k + 2
            for _ in range(3):
                torch.svd_lowrank(m.reshape(shape[0] * shape[1], -1) if False else m[0], q=k)
            # batched lowrank: loop samples would be worst; try as (C, HW) per sample mean
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(nrep):
                # one representative sample (lower bound) and a batched-for loop of 8
                torch.svd_lowrank(m[0], q=k)
            torch.cuda.synchronize()
            ms_one = (time.perf_counter() - t0) * 1000 / nrep
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(nrep):
                for i in range(min(8, shape[0])):
                    torch.svd_lowrank(m[i], q=k)
            torch.cuda.synchronize()
            ms_8 = (time.perf_counter() - t0) * 1000 / nrep
            row[f"lowrank_k{k}_1sample_ms"] = round(ms_one, 3)
            row[f"lowrank_k{k}_8sample_ms"] = round(ms_8, 3)
            row[f"lowrank_k{k}_128sample_extrap_ms"] = round(ms_8 / 8 * shape[0], 3)
        # channel PCA: C×C eig on reduced cov
        c = shape[1]
        cov = torch.randn(c, c, device=DEVICE)
        cov = cov @ cov.T
        for _ in range(3):
            torch.linalg.eigh(cov)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            torch.linalg.eigh(cov)
        torch.cuda.synchronize()
        row["channel_eigh_ms"] = round((time.perf_counter() - t0) * 1000 / 20, 3)
        out.append(row)
        del x, m
        torch.cuda.empty_cache()
    return {"ceiling_ahat_write_ms": 2.024, "shapes": out}


def main():
    cap_path = os.path.join(DATA, "capture.pt")
    cap = torch.load(cap_path, map_location="cpu", weights_only=False)
    spectrum, recursion, budget = {}, {}, {}

    for name, bundle in cap["layers"].items():
        shape = tuple(bundle["shape"])
        print(f"\n=== {name} {shape} steps={bundle['n_steps']} ===")
        spec, bud, tgts, scales = layer_spectrum_and_budget(name, bundle)
        spectrum[name] = {"shape": shape, **spec}
        budget[name] = {"shape": shape, **{k: v for k, v in bud.items() if k != "per_step"}}
        # keep a compact per-step bits for plots (k=8 and k=16)
        budget[name]["per_step_compact"] = [
            {"step": r["step"], "ahat_bits": r["ahat_bits"],
             "r8_bits": r["k"]["8"]["r_bits"], "r16_bits": r["k"]["16"]["r_bits"]}
            for r in bud["per_step"]
        ]
        print("  a_hat energy k=4/8/16/32:",
              [f"{spec['ahat_energy_median'][str(k)]:.3f}" for k in KS])
        print("  delta energy k=4/8/16/32:",
              [f"{spec['delta_energy_median'][str(k)]:.3f}" for k in KS])
        print("  PCA  energy k=4/8/16/32:",
              [f"{spec['channel_pca_energy_median'].get(str(k), float('nan')):.3f}" for k in KS])

        rec = {}
        rows_p, _ = replay_prod(tgts, scales)
        rec["prod"] = summarize(rows_p)
        print(f"  prod final resid {rec['prod']['final_resid']:.4g} |tgt-acc| {rec['prod']['final_tgt_acc']:.4g}")

        for k in KS:
            rec[f"s1_k{k}"] = summarize(replay_s1(tgts, scales, k))
            rec[f"s2_k{k}"] = summarize(replay_s2(tgts, scales, k))
            rec[f"s3_fp16_drecon_k{k}"] = summarize(replay_s3(tgts, scales, k, "fp16", "drecon"))
            rec[f"s3_int8_codes_k{k}"] = summarize(replay_s3(tgts, scales, k, "int8", "codes"))
            rec[f"s3_int8_drecon_k{k}"] = summarize(replay_s3(tgts, scales, k, "int8", "drecon"))
            rec[f"s3_drop_codes_k{k}"] = summarize(replay_s3(tgts, scales, k, "drop", "codes"))
            rec[f"s3_sparse10_codes_k{k}"] = summarize(replay_s3(tgts, scales, k, "sparse10", "codes"))
            rec[f"s3_sparse10_drecon_k{k}"] = summarize(replay_s3(tgts, scales, k, "sparse10", "drecon"))
            print(f"  k={k:2d} S1 resid {rec[f's1_k{k}']['final_resid']:.4g}  "
                  f"S2 resid {rec[f's2_k{k}']['final_resid']:.4g} |tgt-acc| {rec[f's2_k{k}']['final_tgt_acc']:.4g}  "
                  f"S3 int8-codes {rec[f's3_int8_codes_k{k}']['final_resid']:.4g}  "
                  f"S3 int8-drecon {rec[f's3_int8_drecon_k{k}']['final_resid']:.4g}  "
                  f"R bits {bud['r_bits_median'][str(k)]:.2f}")
        recursion[name] = {"shape": shape, **rec}

        # free
        del tgts
        torch.cuda.empty_cache()

    print("\n=== SVD cost (batch 128) ===")
    cost = svd_cost_bench()
    for row in cost.get("shapes", []):
        print(row)

    meta = {
        "steps": cap["steps"],
        "batch_capture": cap["batch"],
        "bound": cap["bound"],
        "ks": list(KS),
        "device": str(DEVICE),
    }
    for fname, obj in (
        ("spectrum.json", {"meta": meta, "layers": spectrum}),
        ("recursion.json", {"meta": meta, "layers": recursion}),
        ("bit_budget.json", {"meta": meta, "layers": budget}),
        ("svd_cost.json", {"meta": meta, **cost}),
    ):
        path = os.path.join(DATA, fname)
        with open(path, "w") as f:
            json.dump(obj, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
