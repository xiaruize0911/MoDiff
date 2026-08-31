"""Inception-v3 FID for the skip / replay / a_hat-quant cache schemes.

WHY FID. Latent relL2 ordered the arms and matches the n=6 contact sheets, but it is not a
generation metric and it has already disagreed with FID on this tree (OPEN_ITEMS B3: MSE weight
scale −7.5% relL2, +5.7% FID). The paper tables and this project's ranking instrument are
Inception-v3 pool3 FID (dims=2048). This run is that instrument for the cache schemes.

WHAT IS COMPARED. Same noise sequence across arms (seed0 + batch index). FID vs the fp16
model is the quantization/scheme question; FID vs W8A8-full (fp16 a_hat) is the scheme-induced
shift on top of MoDiff. vs-real LSUN is omitted: /workspace/fid/real is not on this machine.

N=2048, 50 DDIM, batch 128. Absolute FID is biased at this N and is NOT comparable to the
committed 10k-vs-real numbers (fp16 7.803). Ranking between arms at the same N is the point.

Protocol matches generate_fid_samples.py: realckpt calibration, one discarded warm-up run,
MODIFF_LINEAR=0, static delta table, decode in chunks. Skip and replay are never on together.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/fid_cache_schemes.py
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

from integration.utils.preflight import preflight, MODEL, FID  # noqa: E402
preflight(*MODEL, *FID, what="fid_cache_schemes.py")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE = (4, 32, 32)
CALIB8 = "integration/calibration/int8_calibration_realckpt.pt"

# Prior e2e cost / n=6 relL2 from BRIEF.md — carried so the FID table sits next to them.
PRIOR = {
    "fp16":           {"ms_step": None, "speedup": None, "relL2": 0.0},
    "w8a8_full":      {"ms_step": 93.4, "speedup": 1.00, "relL2": 0.12},
    "skip4":          {"ms_step": 92.2, "speedup": 1.01, "relL2": 0.16},
    "skip8":          {"ms_step": None, "speedup": None, "relL2": 0.33},
    "replay2":        {"ms_step": 74.8, "speedup": 1.25, "relL2": 0.19},
    "replay4":        {"ms_step": 66.0, "speedup": 1.42, "relL2": 0.29},
    "replay8":        {"ms_step": 61.5, "speedup": 1.52, "relL2": 0.40},
    "int8held":       {"ms_step": 94.4, "speedup": 0.99, "relL2": 0.69},
    "skip4_int8held": {"ms_step": 93.9, "speedup": 1.00, "relL2": 0.26},
    "replay4_int8held": {"ms_step": 67.8, "speedup": 1.38, "relL2": 0.34},
}

# (folder, skip_k, replay_k, ahat_bits, refresh) — W8A8 MoDiff only. Never skip+replay.
INT8_ARMS = (
    ("w8a8_full",       1, 1, 16, 0),
    ("skip4",           4, 1, 16, 0),
    ("skip8",           8, 1, 16, 0),
    ("replay2",         1, 2, 16, 0),
    ("replay4",         1, 4, 16, 0),
    ("replay8",         1, 8, 16, 0),
    ("int8held",        1, 1,  8, 0),
    ("skip4_int8held",  4, 1,  8, 0),
    ("replay4_int8held", 1, 4,  8, 0),
)


def _apply(skip_k, replay_k, bits, refresh):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(skip_k)
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_AHAT_REFRESH"] = str(refresh)


def _reset(model, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


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


def compute_fid(out_root, folders, batch, json_out, n, steps, seed0):
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

    cache = {}
    for name in folders:
        cache[name] = stats(name)

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
        rec.update(PRIOR.get(name, {}))
        rows[name] = rec
        extra = ""
        if rec["fid_vs_w8a8_full"] is not None:
            extra = f"  vs W8A8-full {rec['fid_vs_w8a8_full']:.3f}"
        print(f"  {name:20s}  FID vs fp16 {rec['fid_vs_fp16']:.3f}{extra}", flush=True)

    payload = {
        "metric": "inception_v3_fid_dims2048",
        "n": n, "steps": steps, "seed0": seed0,
        "note": ("N=2048 ranking FID, not 10k-vs-real. Absolute values are biased low-N; "
                 "compare arms at this N. Historical 10k-vs-real: fp16 7.803, W8A8+MoDiff 7.802, "
                 "W8A8+MoDiff vs fp16 0.175."),
        "arms": rows,
    }
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_out}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=20260805)
    ap.add_argument("--decode-chunk", type=int, default=32)
    ap.add_argument("--out", default="docs/cache_schemes_report_2026-08-28/fid_samples")
    ap.add_argument("--json", default="docs/cache_schemes_report_2026-08-28/data/fid_cache_schemes.json")
    ap.add_argument("--compute-only", action="store_true")
    a = ap.parse_args()

    folders = ["fp16"] + [arm[0] for arm in INT8_ARMS]
    print(f"GPU: {torch.cuda.get_device_name()}  n={a.n} steps={a.steps} batch={a.batch}",
          flush=True)

    if not a.compute_only:
        runner = B.BenchmarkRunner(
            config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
            ckpt_path="models/ldm/lsun_churches256/model.ckpt",
            output_dir=os.path.join(a.out, "_tmp"),
            batch_size=a.batch, steps=a.steps, shape=SHAPE,
            calibration_path=None, auto_delta_table=True)

        fp16_dir = os.path.join(a.out, "fp16")
        if len([f for f in os.listdir(fp16_dir) if f.endswith(".png")]) < a.n \
                if os.path.isdir(fp16_dir) else True:
            print("===== fp16 =====", flush=True)
            _apply(1, 1, 16, 0)
            model, sampler = runner._setup_model("fp16")
            generate_folder(runner, model, sampler, fp16_dir, a.n, a.batch, a.seed0,
                            a.steps, a.decode_chunk, quantized=False)
            del model, sampler
            torch.cuda.empty_cache()

        need_int8 = False
        for name, *_ in INT8_ARMS:
            d = os.path.join(a.out, name)
            have = len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
            if have < a.n:
                need_int8 = True
                break
        if need_int8:
            print("===== int8 MoDiff =====", flush=True)
            runner.calibration_path = CALIB8
            _apply(1, 1, 16, 0)
            model, sampler = runner._setup_model("int8")
            for name, skip_k, replay_k, bits, refresh in INT8_ARMS:
                print(f"===== {name}  skip={skip_k} replay={replay_k} bits={bits} "
                      f"refresh={refresh} =====", flush=True)
                _apply(skip_k, replay_k, bits, refresh)
                generate_folder(runner, model, sampler, os.path.join(a.out, name),
                                a.n, a.batch, a.seed0, a.steps, a.decode_chunk,
                                quantized=True)
            _apply(1, 1, 16, 0)
            del model, sampler
            torch.cuda.empty_cache()

    print("===== FID =====", flush=True)
    compute_fid(a.out, folders, batch=64, json_out=a.json,
                n=a.n, steps=a.steps, seed0=a.seed0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _apply(1, 1, 16, 0)
