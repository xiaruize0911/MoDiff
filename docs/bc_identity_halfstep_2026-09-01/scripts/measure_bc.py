"""B vs C: same-input layer identity, e2e paired error, half-step DDIM control.

On a skip step of a ResBlock out_conv:

    out_B = o_hat_frozen + skip(x)
    out_C = skip(x)
    out_B - out_C = o_hat_frozen     (exact, same x)

So ||B-C|| / ||B|| = ||o_hat|| / ||o_hat + skip||. If this is ≪ 1, dropping
o_hat is a small perturbation and B ≈ C. If it is O(1), they are different
functions even before error accumulates across steps.

Half-step arms: if skip-K is "doing half the convs", the control is DDIM S=25
with full compute (A or fp16), and "skip the residual after t=T" (C with K=∞).
"""
import json, math, os, statistics, sys
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
from PIL import Image, ImageDraw, ImageFont
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

BATCH, NQ, SEED = 128, 4, 20260805
SHAPE = (4, 32, 32)
OUT_DIR = "docs/bc_identity_halfstep_2026-09-01"
OUT_JSON = f"{OUT_DIR}/data/bc_identity.json"
OUT_PNG = f"{OUT_DIR}/plots/bc_halfstep_grid.png"


def reset_all(model, quantized=True):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)


def set_scheme(replay_k=1, drop=False):
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_REPLAY_DROP_OHAT"] = "1" if drop else "0"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"


def time_n(model, sampler, steps, n=2, warm=1, quantized=True):
    def once():
        reset_all(model, quantized=quantized)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=steps, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    for _ in range(warm):
        once()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); once(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / steps)
    return statistics.median(xs), xs


def setup(mode):
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{OUT_DIR}/tmp",
        batch_size=BATCH, steps=50, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def gen_lat(model, sampler, n, seed, steps, quantized=True):
    reset_all(model, quantized=quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def relL2(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    denom = float(b.norm())
    return float((a - b).norm() / denom) if denom > 0 else float("nan")


def summarize(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return {}
    xs = sorted(xs)
    n = len(xs)
    def pct(p):
        i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return xs[i]
    return {
        "n": n, "mean": sum(xs) / n, "median": pct(50),
        "p10": pct(10), "p90": pct(90), "min": xs[0], "max": xs[-1],
    }


# ---------------------------------------------------------------------------
# Layer probe: wrap out_conv residual path. Skip steps hit _replay_out;
# commit steps hit the fused residual forwards.
# ---------------------------------------------------------------------------
_PROBE = []
_ORIG = {}


def _record(self, residual, kind):
    oh = self.o_hat_cache
    if residual is None or oh is None:
        return
    oh_f = oh.detach().float()
    sk_f = residual.detach().float()
    n_oh = torch.linalg.vector_norm(oh_f)
    n_sk = torch.linalg.vector_norm(sk_f)
    n_b = torch.linalg.vector_norm(oh_f + sk_f)
    # cosine in fp32; clamp empty
    den = (n_oh * n_sk).clamp_min(1e-12)
    cos = (oh_f * sk_f).sum() / den
    _PROBE.append({
        "name": getattr(self, "layer_name", "") or type(self).__name__,
        "C": int(oh.shape[1]), "H": int(oh.shape[2]), "W": int(oh.shape[3]),
        "step_count": int(self.step_count),
        "kind": kind,
        "n_ohat": float(n_oh),
        "n_skip": float(n_sk),
        "n_B": float(n_b),
        "rel_BC_over_B": float(n_oh / n_b.clamp_min(1e-12)),
        "ohat_over_skip": float(n_oh / n_sk.clamp_min(1e-12)),
        "cos": float(cos),
    })


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


def install_probe():
    _PROBE.clear()
    _ORIG["replay"] = OptimizedInt8Conv2d._replay_out
    _ORIG["gn"] = OptimizedInt8Conv2d.forward_gn_fused_modiff
    _ORIG["silu"] = OptimizedInt8Conv2d.forward_modiff_fused_silu_residual

    def replay(self, residual=None):
        if residual is not None:
            _record(self, residual, "skip")
        return _ORIG["replay"](self, residual)

    def gn(self, x, gn_weight, gn_bias, num_groups, eps, mod_scale2d, mod_shift2d,
           residual=None):
        will = _will_replay(self)
        out = _ORIG["gn"](self, x, gn_weight, gn_bias, num_groups, eps,
                          mod_scale2d, mod_shift2d, residual=residual)
        if residual is not None and not will:
            _record(self, residual, "commit")
        return out

    def silu(self, x, residual):
        will = _will_replay(self)
        out = _ORIG["silu"](self, x, residual)
        if residual is not None and not will:
            _record(self, residual, "commit")
        return out

    OptimizedInt8Conv2d._replay_out = replay
    OptimizedInt8Conv2d.forward_gn_fused_modiff = gn
    OptimizedInt8Conv2d.forward_modiff_fused_silu_residual = silu


def uninstall_probe():
    if "replay" in _ORIG:
        OptimizedInt8Conv2d._replay_out = _ORIG["replay"]
        OptimizedInt8Conv2d.forward_gn_fused_modiff = _ORIG["gn"]
        OptimizedInt8Conv2d.forward_modiff_fused_silu_residual = _ORIG["silu"]
        _ORIG.clear()


def aggregate_probe(rows):
    by_kind = {}
    for kind in ("skip", "commit"):
        sub = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "n_calls": len(sub),
            "rel_BC_over_B": summarize([r["rel_BC_over_B"] for r in sub]),
            "ohat_over_skip": summarize([r["ohat_over_skip"] for r in sub]),
            "cos": summarize([r["cos"] for r in sub]),
        }
    by_layer = {}
    for r in rows:
        if r["kind"] != "skip":
            continue
        key = f'{r["name"]}|C{r["C"]}_{r["H"]}x{r["W"]}'
        by_layer.setdefault(key, []).append(r["rel_BC_over_B"])
    layer_med = sorted(
        [{"layer": k, "n": len(v), **summarize(v)} for k, v in by_layer.items()],
        key=lambda d: -d.get("median", 0))
    return {"by_kind": by_kind, "skip_by_layer": layer_med[:20]}


def grid(path, images, nq=NQ):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = font_sm = ImageFont.load_default()
    cell, pad, lab = 256, 6, 48
    W = pad + nq * (cell + pad)
    H = pad + len(images) * (cell + lab + pad)
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for title, sub, arr in images:
        dr.text((pad, y + 4), title, fill=(11, 11, 11), font=font)
        dr.text((pad, y + 24), sub, fill=(70, 70, 70), font=font_sm)
        y += lab
        for i in range(min(nq, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path, "PNG")
    print("wrote", path, flush=True)


# (id, label, mode, replay_k, drop, steps)
# K=10**9 ≈ skip residual after t=T (every later step replays).
ARMS = [
    ("fp16_s50", "fp16  S=50", "fp16", 1, False, 50),
    ("A_s50", "A  Full MoDiff  S=50", "int8", 1, False, 50),
    ("B_k2_s50", "B  Frozen Residual  K=2  S=50", "int8", 2, False, 50),
    ("C_k2_s50", "C  Layer Skip  K=2  S=50", "int8", 2, True, 50),
    ("B_k3_s50", "B  Frozen Residual  K=3  S=50", "int8", 3, False, 50),
    ("C_k3_s50", "C  Layer Skip  K=3  S=50", "int8", 3, True, 50),
    ("fp16_s25", "fp16  S=25  (half steps)", "fp16", 1, False, 25),
    ("A_s25", "A  Full MoDiff  S=25", "int8", 1, False, 25),
    ("B_inf_s50", "B  freeze after t=T  S=50", "int8", 10**9, False, 50),
    ("C_inf_s50", "C  skip residual after t=T  S=50", "int8", 10**9, True, 50),
    ("C_inf_s25", "C  skip residual + S=25", "int8", 10**9, True, 25),
]


def main():
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

    lats, images, rows = {}, [], []
    fp16_ms_s50 = None

    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    for aid, label, mode, rk, drop, steps in ARMS:
        if mode != "fp16":
            continue
        print(f"----- {aid} -----", flush=True)
        ms, trials = time_n(model, sampler, steps, quantized=False)
        lat = gen_lat(model, sampler, NQ, SEED, steps, quantized=False)
        lats[aid] = lat
        if aid == "fp16_s50":
            fp16_ms_s50 = ms
        rec = {
            "arm": aid, "label": label, "ms_step": ms, "trials": trials,
            "steps": steps, "ms_sample": ms * steps,
            "vs_fp16_s50_sample": (fp16_ms_s50 * 50) / (ms * steps),
            "replay_k": rk, "drop_ohat": drop,
        }
        rows.append(rec)
        print(f"  {ms:.2f} ms/step  {ms*steps:.1f} ms/sample  "
              f"{rec['vs_fp16_s50_sample']:.3f}x vs fp16-S50", flush=True)
        images.append((label, f"{ms:.1f} ms/step   {ms*steps:.0f} ms/sample",
                       G.decode(model, lat)))
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 MoDiff =====", flush=True)
    model, sampler = setup("int8")

    print("===== layer probe (B K=2, n=4) =====", flush=True)
    set_scheme(2, drop=False)
    install_probe()
    _ = gen_lat(model, sampler, NQ, SEED, 50, quantized=True)
    uninstall_probe()
    probe_rows = list(_PROBE)
    probe = aggregate_probe(probe_rows)
    sk = probe["by_kind"]["skip"].get("rel_BC_over_B", {})
    print(f"  skip calls={probe['by_kind']['skip']['n_calls']}  "
          f"median ||B-C||/||B||={sk.get('median')}  "
          f"p10={sk.get('p10')}  p90={sk.get('p90')}", flush=True)
    ck = probe["by_kind"]["commit"].get("rel_BC_over_B", {})
    print(f"  commit calls={probe['by_kind']['commit']['n_calls']}  "
          f"median ||o_hat||/||out_B||={ck.get('median')}", flush=True)

    for aid, label, mode, rk, drop, steps in ARMS:
        if mode != "int8":
            continue
        print(f"----- {aid} -----", flush=True)
        set_scheme(rk, drop=drop)
        ms, trials = time_n(model, sampler, steps, quantized=True)
        lat = gen_lat(model, sampler, NQ, SEED, steps, quantized=True)
        lats[aid] = lat
        rec = {
            "arm": aid, "label": label, "ms_step": ms, "trials": trials,
            "steps": steps, "ms_sample": ms * steps,
            "vs_fp16_s50_sample": (fp16_ms_s50 * 50) / (ms * steps),
            "replay_k": rk if rk < 10**6 else "inf",
            "drop_ohat": drop,
        }
        rows.append(rec)
        print(f"  {ms:.2f} ms/step  {ms*steps:.1f} ms/sample  "
              f"{rec['vs_fp16_s50_sample']:.3f}x vs fp16-S50", flush=True)
        images.append((label, f"{ms:.1f} ms/step   {ms*steps:.0f} ms/sample",
                       G.decode(model, lat)))

    del model, sampler
    torch.cuda.empty_cache()

    ref = lats["fp16_s50"]
    pairwise = {}
    keys = [a[0] for a in ARMS]
    for k in keys:
        if k not in lats:
            continue
        pairwise[k] = {"vs_fp16_s50": relL2(lats[k], ref)}
        for k2 in keys:
            if k2 not in lats:
                continue
            pairwise[k][f"vs_{k2}"] = relL2(lats[k], lats[k2])
        for rec in rows:
            if rec["arm"] == k:
                rec["relL2_vs_fp16_s50"] = pairwise[k]["vs_fp16_s50"]

    # Highlighted comparisons
    highlight = {
        "B_k2_vs_C_k2": pairwise.get("B_k2_s50", {}).get("vs_C_k2_s50"),
        "B_k3_vs_C_k3": pairwise.get("B_k3_s50", {}).get("vs_C_k3_s50"),
        "A_s50_vs_B_k2": pairwise.get("A_s50", {}).get("vs_B_k2_s50"),
        "A_s50_vs_C_k2": pairwise.get("A_s50", {}).get("vs_C_k2_s50"),
        "A_s25_vs_B_k2": pairwise.get("A_s25", {}).get("vs_B_k2_s50"),
        "A_s25_vs_C_k2": pairwise.get("A_s25", {}).get("vs_C_k2_s50"),
        "fp16_s25_vs_fp16_s50": pairwise.get("fp16_s25", {}).get("vs_fp16_s50"),
        "C_inf_s50_vs_fp16": pairwise.get("C_inf_s50", {}).get("vs_fp16_s50"),
        "C_inf_s25_vs_fp16": pairwise.get("C_inf_s25", {}).get("vs_fp16_s50"),
        "B_inf_s50_vs_fp16": pairwise.get("B_inf_s50", {}).get("vs_fp16_s50"),
    }
    print("highlights:", json.dumps(highlight, indent=2), flush=True)

    for rec in rows:
        rec["relL2_vs_fp16_s50"] = pairwise.get(rec["arm"], {}).get("vs_fp16_s50")

    img_by_arm = {}
    seq = [a for a in ARMS if a[2] == "fp16"] + [a for a in ARMS if a[2] == "int8"]
    for (aid, _label, *_rest), (_t, _s, arr) in zip(seq, images):
        img_by_arm[aid] = arr
    labeled = []
    for aid, label, mode, rk, drop, steps in ARMS:
        rec = next(r for r in rows if r["arm"] == aid)
        labeled.append((
            label,
            f"{rec['ms_step']:.1f} ms/step  {rec['ms_sample']:.0f} ms/sample  "
            f"{rec['vs_fp16_s50_sample']:.2f}x  relL2 {rec['relL2_vs_fp16_s50']:.3f}",
            img_by_arm[aid],
        ))
    grid(OUT_PNG, labeled)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "protocol": {
            "model": "LDM-8 LSUN-Churches", "batch_time": BATCH, "quality_n": NQ,
            "seed": SEED, "identity": "out_B-out_C = o_hat on skip steps, same x",
        },
        "fp16_s50_ms_step": fp16_ms_s50,
        "arms": rows,
        "highlight": highlight,
        "pairwise_relL2": {k: {kk: vv for kk, vv in v.items()
                               if kk in ("vs_fp16_s50", "vs_B_k2_s50", "vs_C_k2_s50",
                                         "vs_B_k3_s50", "vs_C_k3_s50", "vs_A_s50",
                                         "vs_A_s25", "vs_fp16_s25", "vs_C_inf_s50",
                                         "vs_C_inf_s25", "vs_B_inf_s50")}
                           for k, v in pairwise.items()},
        "probe": probe,
        "grid": OUT_PNG,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
