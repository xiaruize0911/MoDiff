"""Whole-ResBlock replay (MODIFF_REPLAY_BLOCK=full) vs out-only (default).

W8A8, batch 128, 50 DDIM, same process. Skip-K stays 1. Never skip+replay.

  out  (env 1, default): skip emb + out-GN + out_conv; in_conv still runs
  full (env full/in/2):  also skip in-GN + in_conv; live skip_connection still runs
  `in` and `in+emb` are the same path (out-GN has no input without in_conv)

Timing: K=1; replay-K=2/4 BLOCK=out; replay-K=2/4 BLOCK=full.
relL2: n=6 seed 20260805, Frobenius, same as replay_gen.py / BRIEF.
FID: generate ONLY replay2_full / replay4_full; reuse fp16, w8a8_full, replay2, replay4.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/replay_block_full_fid.py
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ.setdefault("MODIFF_REPLAY_BLOCK", "1")

from integration.utils.preflight import preflight, MODEL, FID  # noqa: E402
preflight(*MODEL, *FID, what="replay_block_full_fid.py")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.fused_ops.fused_resblock import _replay_block_mode  # noqa: E402

SHAPE = (4, 32, 32)
CALIB8 = "integration/calibration/int8_calibration_realckpt.pt"
STEPS = 50
BATCH = 128
SEED0 = 20260805
N_FID = 2048
N_RELL2 = 6
DECODE_CHUNK = 32
OUT_SAMPLES = "docs/cache_schemes_report_2026-08-28/fid_samples"
JSON_OUT = "docs/cache_schemes_report_2026-08-28/data/replay_block_full.json"
PRIOR_FID = "docs/cache_schemes_report_2026-08-28/data/fid_cache_schemes.json"

# Existing BLOCK=out FID from fid_cache_schemes.py (N=2048). Not recomputed.
EXISTING_OUT_FID = {
    "replay2": {"fid_vs_fp16": 5.396568097849752, "fid_vs_w8a8_full": 4.793709096611906,
                "ms_step": 74.8, "relL2": 0.19, "block": "out"},
    "replay4": {"fid_vs_fp16": 16.346218773385715, "fid_vs_w8a8_full": 15.222052804398146,
                "ms_step": 66.0, "relL2": 0.29, "block": "out"},
    "w8a8_full": {"fid_vs_fp16": 0.9203298739598154, "fid_vs_w8a8_full": 0.0,
                  "ms_step": 93.4, "relL2": 0.12, "block": "n/a"},
}


def _apply(replay_k, block):
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_REPLAY_BLOCK"] = str(block)
    os.environ["MODIFF_AHAT_BITS"] = "16"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"
    os.environ["MODIFF_LINEAR"] = "0"


def _reset(model, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def rel_l2(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    den = b.norm().clamp_min(1e-12)
    return float((a - b).norm() / den)


def gen_latent(runner, model, sampler, n, seed, steps, quantized):
    _reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def time_once(runner, model, sampler, seed, steps, batch, quantized):
    _reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=steps, batch_size=batch, shape=SHAPE, eta=0.0,
                       verbose=False, **cond)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / steps


def sample_batch(runner, model, sampler, n, seed, steps, decode_chunk, quantized):
    _reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=runner.shape, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
        lat = lat.to("cuda", torch.float16)
        chunks = []
        for i in range(0, lat.shape[0], decode_chunk):
            d = model.decode_first_stage(lat[i:i + decode_chunk])
            chunks.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0)
                          .permute(0, 2, 3, 1).cpu())
            del d
        img = torch.cat(chunks, 0)
    return (img.numpy() * 255).round().astype("uint8")


def generate_folder(runner, model, sampler, folder, n, batch, seed0, steps,
                    decode_chunk, quantized):
    os.makedirs(folder, exist_ok=True)
    have = len([f for f in os.listdir(folder) if f.endswith(".png")])
    if have >= n:
        print(f"  {folder}: already {have} pngs, skip", flush=True)
        return
    if have:
        print(f"  {folder}: resume from {have}/{n}", flush=True)
    _reset(model, quantized)
    sample_batch(runner, model, sampler, min(batch, 16), seed0 - 1, steps,
                 decode_chunk, quantized)
    written, bi, t0 = have, have // batch, time.time()
    if have % batch:
        raise SystemExit(f"{folder} has {have} pngs, not a multiple of batch={batch}")
    while written < n:
        k = min(batch, n - written)
        arr = sample_batch(runner, model, sampler, k, seed0 + bi, steps,
                           decode_chunk, quantized)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i]).save(os.path.join(folder, f"{written + i:06d}.png"), "PNG")
        written += arr.shape[0]
        bi += 1
        el = time.time() - t0
        print(f"  {os.path.basename(folder)}  {written}/{n}  {el:.0f}s  "
              f"~{el / max(written - have, 1) * (n - written):.0f}s left", flush=True)


def compute_fid(out_root, folders, batch, n, steps, seed0):
    from pytorch_fid.fid_score import calculate_frechet_distance, compute_statistics_of_path
    from pytorch_fid.inception import InceptionV3

    dev = torch.device("cuda")
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc = InceptionV3([block]).to(dev)

    def stats(name):
        path = os.path.join(out_root, name)
        n_img = len([f for f in os.listdir(path) if f.endswith(".png")])
        print(f"  features {name} n={n_img}", flush=True)
        mu, sigma = compute_statistics_of_path(path, inc, batch, 2048, dev, num_workers=4)
        return mu, sigma, n_img

    cache = {name: stats(name) for name in folders}

    def fid(a, b):
        mu_a, s_a, _ = cache[a]
        mu_b, s_b, _ = cache[b]
        return float(calculate_frechet_distance(mu_a, s_a, mu_b, s_b))

    rows = {}
    for name in folders:
        rec = {
            "n": cache[name][2],
            "fid_vs_fp16": 0.0 if name == "fp16" else fid("fp16", name),
            "fid_vs_w8a8_full": (None if "w8a8_full" not in cache else
                                 (0.0 if name == "w8a8_full" else fid("w8a8_full", name))),
        }
        extra = ""
        if rec["fid_vs_w8a8_full"] is not None:
            extra = f"  vs W8A8-full {rec['fid_vs_w8a8_full']:.3f}"
        print(f"  {name:20s}  FID vs fp16 {rec['fid_vs_fp16']:.3f}{extra}", flush=True)
        rows[name] = rec
    return rows


def confirm_modes():
    orig = os.environ.get("MODIFF_REPLAY_BLOCK", "1")
    mapping = {}
    for v in ("0", "perconv", "1", "out", "in", "in+emb", "in_emb", "full", "2"):
        os.environ["MODIFF_REPLAY_BLOCK"] = v
        mapping[v] = _replay_block_mode()
    os.environ["MODIFF_REPLAY_BLOCK"] = orig
    assert mapping["0"] == mapping["perconv"] == "perconv"
    assert mapping["1"] == mapping["out"] == "out"
    assert mapping["in"] == mapping["in+emb"] == mapping["in_emb"] == mapping["full"] == mapping["2"] == "full"
    print("mode map:", mapping, flush=True)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_FID)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--n-rell2", type=int, default=N_RELL2)
    ap.add_argument("--decode-chunk", type=int, default=DECODE_CHUNK)
    ap.add_argument("--out", default=OUT_SAMPLES)
    ap.add_argument("--json", default=JSON_OUT)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--skip-rell2", action="store_true")
    ap.add_argument("--skip-fid-gen", action="store_true")
    ap.add_argument("--skip-fid-compute", action="store_true")
    a = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}  n_fid={a.n} steps={a.steps} "
          f"batch={a.batch} n_rell2={a.n_rell2}", flush=True)
    mode_map = confirm_modes()

    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(a.out, "_tmp_full"),
        batch_size=a.batch, steps=a.steps, shape=SHAPE,
        calibration_path=CALIB8, auto_delta_table=True)

    payload = {
        "gpu": torch.cuda.get_device_name(),
        "batch": a.batch, "steps": a.steps, "seed0": a.seed0,
        "n_fid": a.n, "n_rell2": a.n_rell2, "decode_chunk": a.decode_chunk,
        "calib": CALIB8,
        "note": ("BLOCK=out skips emb+out-GN+out_conv; BLOCK=full also skips in-GN+in_conv. "
                 "in == in+emb. Skip-K=1. MODIFF_LINEAR=0. realckpt + static delta."),
        "mode_map": mode_map,
        "bugs_fixed": [],
        "timing": [],
        "rell2": {},
        "fid": {},
        "existing_out_fid": EXISTING_OUT_FID,
    }

    timing_arms = [
        ("K=1 full compute", 1, "out"),
        ("K=2 BLOCK=out", 2, "out"),
        ("K=2 BLOCK=full", 2, "full"),
        ("K=4 BLOCK=out", 4, "out"),
        ("K=4 BLOCK=full", 4, "full"),
    ]

    z_fp16 = None
    if not a.skip_rell2:
        print("===== fp16 latents for relL2 =====", flush=True)
        _apply(1, "out")
        model, sampler = runner._setup_model("fp16")
        print("  fp16 warmup", flush=True)
        gen_latent(runner, model, sampler, a.n_rell2, a.seed0, a.steps, quantized=False)
        z_fp16 = gen_latent(runner, model, sampler, a.n_rell2, a.seed0, a.steps, quantized=False)
        del model, sampler
        torch.cuda.empty_cache()

    need_int8 = (not a.skip_timing) or (not a.skip_rell2) or (not a.skip_fid_gen)
    model = sampler = None
    if need_int8:
        print("===== int8 MoDiff =====", flush=True)
        runner.calibration_path = CALIB8
        _apply(1, "out")
        model, sampler = runner._setup_model("int8")

    if not a.skip_timing:
        print("===== timing (warmup then timed, batch 128) =====", flush=True)
        k1_ms = None
        for label, k, block in timing_arms:
            _apply(k, block)
            print(f"  warmup {label}  REPLAY_K={k} BLOCK={block} "
                  f"(resolved {_replay_block_mode()})", flush=True)
            time_once(runner, model, sampler, a.seed0, a.steps, a.batch, quantized=True)
            ms = time_once(runner, model, sampler, a.seed0, a.steps, a.batch, quantized=True)
            if k1_ms is None:
                k1_ms = ms
            rec = {
                "label": label, "k": k, "block": block,
                "ms_step": ms,
                "speedup_vs_k1": (k1_ms / ms) if ms else None,
                "vs_k1_ms": (k1_ms - ms) if k1_ms is not None else None,
            }
            payload["timing"].append(rec)
            print(f"    {label:22s} {ms:.3f} ms/step  vs K=1 {k1_ms/ms:.3f}x  "
                  f"({k1_ms - ms:+.2f} ms)", flush=True)
        _apply(1, "out")

    if not a.skip_rell2:
        print("===== relL2 n=%d seed %d =====" % (a.n_rell2, a.seed0), flush=True)
        rell2_arms = [
            ("w8a8_k1", 1, "out"),
            ("out_k2", 2, "out"),
            ("full_k2", 2, "full"),
            ("out_k4", 4, "out"),
            ("full_k4", 4, "full"),
        ]
        zs = {}
        for name, k, block in rell2_arms:
            _apply(k, block)
            print(f"  warmup {name}", flush=True)
            gen_latent(runner, model, sampler, a.n_rell2, a.seed0, a.steps, quantized=True)
            zs[name] = gen_latent(runner, model, sampler, a.n_rell2, a.seed0, a.steps,
                                  quantized=True)
            print(f"    got {name}", flush=True)
        _apply(1, "out")

        pairs = [
            ("full_k2_vs_fp16", "full_k2", "fp16"),
            ("full_k4_vs_fp16", "full_k4", "fp16"),
            ("out_k2_vs_fp16", "out_k2", "fp16"),
            ("out_k4_vs_fp16", "out_k4", "fp16"),
            ("full_k2_vs_out_k2", "full_k2", "out_k2"),
            ("full_k4_vs_out_k4", "full_k4", "out_k4"),
            ("w8a8_k1_vs_fp16", "w8a8_k1", "fp16"),
            ("full_k2_vs_w8a8_k1", "full_k2", "w8a8_k1"),
            ("full_k4_vs_w8a8_k1", "full_k4", "w8a8_k1"),
            ("out_k2_vs_w8a8_k1", "out_k2", "w8a8_k1"),
            ("out_k4_vs_w8a8_k1", "out_k4", "w8a8_k1"),
        ]
        zs["fp16"] = z_fp16
        for key, left, right in pairs:
            if zs.get(left) is None or zs.get(right) is None:
                continue
            val = rel_l2(zs[left], zs[right])
            payload["rell2"][key] = val
            print(f"  {key:28s} {val:.6f}", flush=True)

    if not a.skip_fid_gen:
        print("===== FID generate replay2_full / replay4_full =====", flush=True)
        for name, k in (("replay2_full", 2), ("replay4_full", 4)):
            _apply(k, "full")
            print(f"===== {name}  replay={k} BLOCK=full "
                  f"(resolved {_replay_block_mode()}) =====", flush=True)
            generate_folder(runner, model, sampler, os.path.join(a.out, name),
                            a.n, a.batch, a.seed0, a.steps, a.decode_chunk,
                            quantized=True)
        _apply(1, "out")

    if model is not None:
        del model, sampler
        torch.cuda.empty_cache()

    if not a.skip_fid_compute:
        print("===== FID compute =====", flush=True)
        folders = ["fp16", "w8a8_full", "replay2_full", "replay4_full"]
        for name in folders:
            path = os.path.join(a.out, name)
            n_img = len([f for f in os.listdir(path) if f.endswith(".png")]) if os.path.isdir(path) else 0
            if n_img < a.n:
                print(f"  WARN {name}: {n_img} pngs < n={a.n}", flush=True)
        payload["fid"] = compute_fid(a.out, folders, batch=64, n=a.n,
                                     steps=a.steps, seed0=a.seed0)
        payload["fid_n_note"] = (
            f"N={a.n}" + ("" if a.n == 2048 else " (NOT the committed N=2048 protocol)")
        )

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _apply(1, "out")
        os.environ["MODIFF_CACHE_SKIP_K"] = "1"
        os.environ["MODIFF_REPLAY_K"] = "1"
        os.environ["MODIFF_REPLAY_BLOCK"] = "1"
