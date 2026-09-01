"""Per skip-step change: ||Δo_hat|| vs ||Δskip|| vs ||Δout||.

A computes every step: o_hat += conv(Q(a-a_hat)), out = o_hat + skip(x).
B skip: o_hat frozen, out = o_hat + skip(x)  =>  Δout = Δskip, Δo_hat = 0.
C skip: out = skip(x)                       =>  Δout = Δskip, and the level is missing o_hat.
"""
import json, math, os, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [
    ROOT, os.path.join(ROOT, "src/taming-transformers"),
    os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts"),
]
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
os.environ["MODIFF_QUANT_LINEAR"] = "1"
os.environ["MODIFF_QUANT_ATTN"] = "1"
os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
os.environ["MODIFF_QATTN_FLASH"] = "1"
os.environ["MODIFF_FLASH_GATE"] = "on"
os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
os.environ["MODIFF_REPLAY_DROP_OHAT"] = "0"
os.environ["MODIFF_WARMUP_STEPS"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"

import torch
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

NQ, SEED, STEPS = 4, 20260805, 50
SHAPE = (4, 32, 32)
OUT = "docs/bc_identity_halfstep_2026-09-01/data/skip_step_delta.json"

ROWS = []
PREV = {}
_ORIG = {}


def nrm(t):
    return float(torch.linalg.vector_norm(t.detach().float()))


def _will_replay(self):
    if self.calibrating or not self.modiff_enabled:
        return False
    try:
        k = int(os.environ.get("MODIFF_REPLAY_K", "1"))
    except (TypeError, ValueError):
        k = 1
    nxt = self.step_count + 1
    return (k > 1 and nxt > 0 and (nxt % k) != 0
            and self.o_hat_cache is not None and self.a_hat_cache is not None)


def record(self, skip, out, oh_before, kind, arm):
    if skip is None or self.o_hat_cache is None:
        return
    name = getattr(self, "layer_name", "") or "?"
    oh = self.o_hat_cache.detach()
    skip_f = skip.detach()
    out_f = out.detach()
    key = name
    prev = PREV.get(key)
    rec = {
        "arm": arm, "kind": kind, "name": name,
        "step": int(self.step_count),
        "n_ohat": nrm(oh), "n_skip": nrm(skip_f), "n_out": nrm(out_f),
    }
    if oh_before is not None:
        rec["d_ohat"] = nrm(oh.float() - oh_before.float())
    else:
        rec["d_ohat"] = None
    if prev is not None and prev["skip"].shape == skip_f.shape:
        rec["d_skip"] = nrm(skip_f.float() - prev["skip"].float())
        rec["d_out"] = nrm(out_f.float() - prev["out"].float())
        rec["d_ohat_from_prev"] = nrm(oh.float() - prev["ohat"].float())
    PREV[key] = {
        "ohat": oh.float().clone(),
        "skip": skip_f.float().clone(),
        "out": out_f.float().clone(),
    }
    ROWS.append(rec)


def install(arm):
    PREV.clear()
    _ORIG["replay"] = OptimizedInt8Conv2d._replay_out
    _ORIG["gn"] = OptimizedInt8Conv2d.forward_gn_fused_modiff
    _ORIG["silu"] = OptimizedInt8Conv2d.forward_modiff_fused_silu_residual

    def replay(self, residual=None):
        out = _ORIG["replay"](self, residual)
        if residual is not None:
            record(self, residual, out, self.o_hat_cache, "skip", arm)
        return out

    def gn(self, x, gn_weight, gn_bias, num_groups, eps, mod_scale2d, mod_shift2d,
           residual=None):
        will = _will_replay(self)
        oh_b = (self.o_hat_cache.detach().clone()
                if self.o_hat_cache is not None else None)
        out = _ORIG["gn"](self, x, gn_weight, gn_bias, num_groups, eps,
                          mod_scale2d, mod_shift2d, residual=residual)
        if residual is not None and not will:
            record(self, residual, out, oh_b, "commit", arm)
        return out

    def silu(self, x, residual):
        will = _will_replay(self)
        oh_b = (self.o_hat_cache.detach().clone()
                if self.o_hat_cache is not None else None)
        out = _ORIG["silu"](self, x, residual)
        if residual is not None and not will:
            record(self, residual, out, oh_b, "commit", arm)
        return out

    OptimizedInt8Conv2d._replay_out = replay
    OptimizedInt8Conv2d.forward_gn_fused_modiff = gn
    OptimizedInt8Conv2d.forward_modiff_fused_silu_residual = silu


def uninstall():
    if "replay" in _ORIG:
        OptimizedInt8Conv2d._replay_out = _ORIG["replay"]
        OptimizedInt8Conv2d.forward_gn_fused_modiff = _ORIG["gn"]
        OptimizedInt8Conv2d.forward_modiff_fused_silu_residual = _ORIG["silu"]
        _ORIG.clear()


def summarize(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {}
    xs = sorted(xs)
    n = len(xs)
    def pct(p):
        return xs[min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))]
    return {"n": n, "mean": sum(xs) / n, "median": pct(50),
            "p10": pct(10), "p90": pct(90)}


def ratios(sub):
    d_oh, d_sk, d_out, oh_over_d, d_oh_over_sk, d_oh_over_out = [], [], [], [], [], []
    d_sk_over_out, d_out_over_out = [], []
    for r in sub:
        if r.get("d_ohat_from_prev") is None:
            continue
        d_oh.append(r["d_ohat_from_prev"])
        d_sk.append(r["d_skip"])
        d_out.append(r["d_out"])
        if r["n_ohat"] > 0:
            oh_over_d.append(r["d_ohat_from_prev"] / r["n_ohat"])
        if r["d_skip"] > 0:
            d_oh_over_sk.append(r["d_ohat_from_prev"] / r["d_skip"])
        if r["d_out"] > 0:
            d_oh_over_out.append(r["d_ohat_from_prev"] / r["d_out"])
            d_sk_over_out.append(r["d_skip"] / r["d_out"])
        if r["n_out"] > 0:
            d_out_over_out.append(r["d_out"] / r["n_out"])
    return {
        "d_ohat": summarize(d_oh),
        "d_skip": summarize(d_sk),
        "d_out": summarize(d_out),
        "d_ohat_over_n_ohat": summarize(oh_over_d),
        "d_ohat_over_d_skip": summarize(d_oh_over_sk),
        "d_ohat_over_d_out": summarize(d_oh_over_out),
        "d_skip_over_d_out": summarize(d_sk_over_out),
        "d_out_over_n_out": summarize(d_out_over_out),
    }


def setup():
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/bc_identity_halfstep_2026-09-01/tmp",
        batch_size=NQ, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model("int8")


def gen(model, sampler, replay_k, drop, arm):
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_REPLAY_DROP_OHAT"] = "1" if drop else "0"
    PREV.clear()
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    install(arm)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=NQ, shape=SHAPE, eta=0.0, verbose=False)
    uninstall()


def main():
    print("===== load int8 =====", flush=True)
    model, sampler = setup()
    for arm, rk, drop in (
        ("A", 1, False),
        ("B_k2", 2, False),
        ("C_k2", 2, True),
        ("B_k3", 3, False),
    ):
        print(f"===== probe {arm} =====", flush=True)
        n0 = len(ROWS)
        gen(model, sampler, rk, drop, arm)
        print(f"  +{len(ROWS)-n0} records", flush=True)

    by = {}
    for arm in ("A", "B_k2", "C_k2", "B_k3"):
        by[arm] = {}
        for kind in ("commit", "skip"):
            sub = [r for r in ROWS if r["arm"] == arm and r["kind"] == kind]
            by[arm][kind] = {"n_calls": len(sub), **ratios(sub)}
            d = by[arm][kind]
            print(f"  {arm} {kind} n={d['n_calls']}  "
                  f"d_ohat={d.get('d_ohat', {}).get('median')}  "
                  f"d_skip={d.get('d_skip', {}).get('median')}  "
                  f"d_out={d.get('d_out', {}).get('median')}  "
                  f"d_ohat/d_skip={d.get('d_ohat_over_d_skip', {}).get('median')}",
                  flush=True)

    json.dump({
        "gpu": torch.cuda.get_device_name(0),
        "n": NQ, "steps": STEPS, "seed": SEED,
        "identity": "skip-step B: Δout=Δskip, Δo_hat=0; C: same Δ, missing o_hat level",
        "by_arm_kind": by,
    }, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
