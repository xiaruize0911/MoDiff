"""FID N=2048 for I-MoDiff arms vs existing fp16 / w8a8_full samples.

Reuses fid_cache_schemes.py generate/compute. Writes into the same fid_samples/
tree under frozen_s, imode16, imode8, imode4.

Run: source setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/imode_fid.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_DELTA_FREEZE"] = "0"

import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
import fid_cache_schemes as F  # noqa: E402

ARMS = (
    # name, imode, bits, freeze
    ("frozen_s", False, 16, True),
    ("imode16", True, 16, False),
    ("imode8", True, 8, False),
    ("imode4", True, 4, False),
)


def _apply(imode, bits, freeze):
    os.environ["MODIFF_IMODE"] = "1" if imode else "0"
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_DELTA_FREEZE"] = "1" if freeze else "0"
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"


def main():
    n, batch, steps, seed0 = 2048, 128, 50, 20260805
    out = "docs/cache_schemes_report_2026-08-28/fid_samples"
    json_out = "docs/cache_schemes_report_2026-08-28/data/imode.json"
    print(f"GPU: {torch.cuda.get_device_name()}  n={n} steps={steps}", flush=True)

    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(out, "_tmp_imode"),
        batch_size=batch, steps=steps, shape=F.SHAPE,
        calibration_path=F.CALIB8, auto_delta_table=True)

    need = False
    for name, *_ in ARMS:
        d = os.path.join(out, name)
        have = len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
        if have < n:
            need = True
            break
    if need:
        print("===== int8 I-MoDiff FID samples =====", flush=True)
        model, sampler = runner._setup_model("int8")
        for name, imode, bits, freeze in ARMS:
            print(f"===== {name} imode={imode} bits={bits} freeze={freeze} =====",
                  flush=True)
            _apply(imode, bits, freeze)
            F.generate_folder(runner, model, sampler, os.path.join(out, name),
                              n, batch, seed0, steps, 32, quantized=True)
        _apply(False, 16, False)
        del model, sampler
        torch.cuda.empty_cache()

    folders = ["fp16", "w8a8_full"] + [a[0] for a in ARMS]
    print("===== FID =====", flush=True)
    payload = F.compute_fid(out, folders, batch=64, json_out="/tmp/imode_fid.json",
                            n=n, steps=steps, seed0=seed0)
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    prev = {}
    if os.path.exists(json_out):
        prev = __import__("json").load(open(json_out))
    prev["fid"] = payload["arms"]
    __import__("json").dump(prev, open(json_out, "w"), indent=1)
    print(f"wrote {json_out}")
    _apply(False, 16, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
