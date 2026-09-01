"""Why not skip the DDIM step? 1.5% is one ResBlock, not ε or x.

Records ||Δε||/||ε|| and ||Δx||/||x|| per DDIM step for A and B K=2.
Arm `eps_replay`: skip the UNet on odd steps, reuse last ε, still take the DDIM
integrator step (the literal reading of "skip this step's compute").
A S=25 is "skip the DDIM step entirely" (coarser schedule).
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

NQ, SEED, STEPS = 4, 20260805, 50
BATCH = 128
SHAPE = (4, 32, 32)
OUT_DIR = "docs/bc_identity_halfstep_2026-09-01"
OUT_JSON = f"{OUT_DIR}/data/skip_the_step.json"
OUT_PNG = f"{OUT_DIR}/plots/skip_the_step.png"

LOG = []
PREV = {}


def nrm(t):
    return float(torch.linalg.vector_norm(t.detach().float()))


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


def reset_all(model):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def setup():
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{OUT_DIR}/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model("int8")


def install_trace(model, sampler, arm, replay_k, drop, eps_reuse_k=1):
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_REPLAY_DROP_OHAT"] = "1" if drop else "0"
    PREV.clear()
    orig_apply = model.apply_model
    orig_step = sampler.p_sample_ddim
    state = {"i": 0, "last_eps": None, "n_apply": 0, "n_reuse": 0}

    def apply_model(x, t, c):
        i = state["i"]
        reuse = (eps_reuse_k > 1 and i > 0 and (i % eps_reuse_k) != 0
                 and state["last_eps"] is not None)
        if reuse:
            state["n_reuse"] += 1
            return state["last_eps"]
        state["n_apply"] += 1
        y = orig_apply(x, t, c)
        state["last_eps"] = y
        return y

    def p_sample(x, c, t, index, **kw):
        i = state["i"]
        eps_before = state["last_eps"]
        x_prev, pred_x0 = orig_step(x, c, t, index, **kw)
        # last_eps is the ε used this step
        eps = state["last_eps"]
        rec = {
            "arm": arm, "i": i, "t": int(t[0].item()) if t.numel() else None,
            "n_eps": nrm(eps) if eps is not None else None,
            "n_x": nrm(x), "n_x_prev": nrm(x_prev),
            "d_x_over_x": nrm(x_prev - x) / max(nrm(x), 1e-12),
        }
        if eps_before is not None and eps is not None:
            rec["d_eps_over_eps"] = nrm(eps - eps_before) / max(nrm(eps), 1e-12)
            rec["reused"] = bool(eps_reuse_k > 1 and i > 0 and (i % eps_reuse_k) != 0)
        else:
            rec["d_eps_over_eps"] = None
            rec["reused"] = False
        # B conv skip: after this UNet, out_conv step_count % K != 0
        sc = None
        for m in model.model.diffusion_model.modules():
            if isinstance(m, OptimizedInt8Conv2d) and getattr(m, "step_count", 0) > 0:
                sc = int(m.step_count)
                break
        rec["conv_step_count"] = sc
        rec["conv_skip"] = bool(replay_k > 1 and sc is not None and sc > 0
                                and (sc % replay_k) != 0)
        LOG.append(rec)
        state["i"] += 1
        return x_prev, pred_x0

    model.apply_model = apply_model
    sampler.p_sample_ddim = p_sample
    return orig_apply, orig_step, state


def uninstall(model, sampler, orig_apply, orig_step):
    model.apply_model = orig_apply
    sampler.p_sample_ddim = orig_step


def gen(model, sampler, n, seed, steps):
    reset_all(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def time_n(model, sampler, steps, n=2, warm=1):
    def once():
        reset_all(model)
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


def relL2(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    return float((a - b).norm() / b.norm())


def grid(path, images):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = font_sm = ImageFont.load_default()
    cell, pad, lab = 256, 6, 48
    W = pad + NQ * (cell + pad)
    H = pad + len(images) * (cell + lab + pad)
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for title, sub, arr in images:
        dr.text((pad, y + 4), title, fill=(11, 11, 11), font=font)
        dr.text((pad, y + 24), sub, fill=(70, 70, 70), font=font_sm)
        y += lab
        for i in range(min(NQ, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path, "PNG")


def arm_stats(arm, pred):
    rows = [r for r in LOG if r["arm"] == arm and pred(r)]
    return {
        "n": len(rows),
        "d_eps_over_eps": summarize([r["d_eps_over_eps"] for r in rows]),
        "d_x_over_x": summarize([r["d_x_over_x"] for r in rows]),
    }


def main():
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    print("===== load =====", flush=True)
    model, sampler = setup()
    lats, images, timed = {}, [], {}

    specs = [
        ("A_s50", 1, False, 1, 50),
        ("B_k2", 2, False, 1, 50),
        ("eps_replay_k2", 1, False, 2, 50),  # full MoDiff convs, skip UNet every other DDIM step
        ("A_s25", 1, False, 1, 25),
    ]
    for aid, rk, drop, ek, steps in specs:
        print(f"===== {aid} =====", flush=True)
        orig_a, orig_s, st = install_trace(model, sampler, aid, rk, drop, ek)
        lat = gen(model, sampler, NQ, SEED, steps)
        uninstall(model, sampler, orig_a, orig_s)
        lats[aid] = lat
        print(f"  apply={st['n_apply']} reuse={st['n_reuse']} i={st['i']}", flush=True)
        # timing without the quality wrap
        os.environ["MODIFF_REPLAY_K"] = str(rk)
        os.environ["MODIFF_REPLAY_DROP_OHAT"] = "1" if drop else "0"
        if ek > 1:
            oa, os_, st2 = install_trace(model, sampler, aid + "_time", rk, drop, ek)
            ms, trials = time_n(model, sampler, steps)
            uninstall(model, sampler, oa, os_)
            # drop timer logs
            while LOG and LOG[-1]["arm"] == aid + "_time":
                LOG.pop()
        else:
            ms, trials = time_n(model, sampler, steps)
        timed[aid] = {"ms_step": ms, "ms_sample": ms * steps, "trials": trials,
                      "n_apply": st["n_apply"], "n_reuse": st["n_reuse"]}
        print(f"  {ms:.2f} ms/step  {ms*steps:.1f} ms/sample", flush=True)

    ref = lats["A_s50"]
    quality = {k: {"relL2_vs_A_s50": relL2(v, ref)} for k, v in lats.items()}
    for aid, rec in timed.items():
        rec["relL2_vs_A_s50"] = quality[aid]["relL2_vs_A_s50"]
        images.append((
            aid,
            f"{rec['ms_step']:.1f} ms/step  {rec['ms_sample']:.0f} ms/sample  "
            f"relL2 vs A {rec['relL2_vs_A_s50']:.3f}",
            G.decode(model, lats[aid]),
        ))
        print(f"  {aid} relL2 vs A {rec['relL2_vs_A_s50']:.4f}", flush=True)

    grid(OUT_PNG, images)

    stats = {
        "A_all": arm_stats("A_s50", lambda r: r["d_eps_over_eps"] is not None),
        "B_conv_skip": arm_stats("B_k2", lambda r: r["conv_skip"] and r["d_eps_over_eps"] is not None),
        "B_conv_commit": arm_stats("B_k2", lambda r: (not r["conv_skip"]) and r["d_eps_over_eps"] is not None),
        "eps_replay_reuse": arm_stats("eps_replay_k2", lambda r: r.get("reused")),
        "eps_replay_fresh": arm_stats("eps_replay_k2", lambda r: r["d_eps_over_eps"] is not None and not r.get("reused")),
        "A_s25": arm_stats("A_s25", lambda r: r["d_eps_over_eps"] is not None),
    }
    for k, v in stats.items():
        print(f"  {k} n={v['n']} d_eps={v['d_eps_over_eps'].get('median')} "
              f"d_x={v['d_x_over_x'].get('median')}", flush=True)

    json.dump({
        "gpu": torch.cuda.get_device_name(0),
        "timed": timed, "quality": quality, "step_stats": stats,
        "grid": OUT_PNG,
    }, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
