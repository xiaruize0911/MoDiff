"""MODIFF_CAT2_FOLD must be a no-op for every mode the fold cannot serve.

The fold only engages on int4 MoDiff non-updown ResBlocks. Every other mode still receives the
(h, skip) tuple from the decoder loop and must materialize the concatenation itself and produce exactly
what it produced before. This checks that, per mode, with the flag off and on.

HOW THIS IS TESTED, after two wrong attempts:

  ATTEMPT 1 compared flag=0 against flag=1 per mode and called fp16 BROKEN. Wrong: fp16 sampling is
  NONDETERMINISTIC ACROSS PROCESSES here (relL2 ~4-6e-3 between two runs with identical settings), and
  the flag cannot even reach fp16 -- with no int4 convs the fold is never eligible.

  ATTEMPT 2 added a same-flag control and required flag-diff <= control-diff. Also wrong, and wrong in
  a way worth remembering: with ONE sample of each, both are draws from the same ~4-6e-3 distribution,
  so the comparison is a coin flip. It duly flipped -- 3.9e-3 vs 5.2e-3 on one run (pass) and 5.8e-3
  vs 3.9e-3 on the next (fail). A criterion that reverses on re-run is not measuring anything.

  THIS VERSION asks the question that has a deterministic answer: DID THE FOLD KERNEL RUN? Transparency
  for a mode the fold cannot serve is not "the outputs happen to match" -- which is unmeasurable when
  the mode is nondeterministic -- it is "the code path was never taken". That is counted exactly, by
  wrapping the pybind entry. Only modes where the fold DID run must then match bit-for-bit, and those
  are the deterministic ones.

The fp16 nondeterminism is worth naming on its own: it is the "unidentified second source" that made
docs/attn_modiff_2026-08-13's A/B unable to reproduce a committed reference, and the reason
scripts/fp16_refs.py pins the fp16 references to disk. Quantized modes are deterministic here (fixed
CUTLASS kernels); fp16 is not (cuDNN selects its convolution algorithm per process).

Run: python integration/tests/test_cat2_fold_fallback.py    # ~6 min, needs the GPU
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

SCRATCH = os.environ.get("MODIFF_SCRATCH", "/tmp/modiff_cat2_fallback")
MODES = ["fp16", "int8", "int4_baseline", "int4"]


def child(mode, tag):
    import torch
    import modiff_cutlass as mc
    import dynamic_delta_ab as H
    import integration.benchmarks.benchmark_ldm as B
    calls = [0]
    real = mc.group_norm_silu_delta_quantize_pack_cat2_nhwc

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    mc.group_norm_silu_delta_quantize_pack_cat2_nhwc = counted
    H.STEPS, H.BATCH = 10, 2
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    r, m, s = H.build(mode, B._default_calibration_path(mode), "static")
    H.SEED = 1234
    H.latent(r, m, s)                      # discard: attention self-calibration
    lat, _ = H.latent(r, m, s)
    os.makedirs(SCRATCH, exist_ok=True)
    torch.save({"lat": lat, "calls": calls[0]}, f"{SCRATCH}/{mode}_{tag}.pt")
    return 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3])
    import torch
    os.makedirs(SCRATCH, exist_ok=True)
    for mode in MODES:
        for tag, flag in (("off", "0"), ("on", "1")):
            env = dict(os.environ, MODIFF_CAT2_FOLD=flag,
                       PYTHONPATH=f"{ROOT}:{ROOT}/src/taming-transformers")
            p = subprocess.run([sys.executable, __file__, "--child", mode, tag],
                               env=env, capture_output=True, text=True)
            if p.returncode != 0:
                print(f"FAIL: {mode} flag={flag} exited {p.returncode}")
                print(p.stderr[-2500:])
                return 1
        print(f"  ran {mode}", flush=True)

    def rel(x, y):
        return float((x.float() - y.float()).norm() / y.float().norm())

    print(f"\n{'mode':16}{'fold calls ON':>15}{'fold calls OFF':>16}{'latents':>22}{'verdict':>10}")
    bad = 0
    for mode in MODES:
        off = torch.load(f"{SCRATCH}/{mode}_off.pt", weights_only=False)
        on = torch.load(f"{SCRATCH}/{mode}_on.pt", weights_only=False)
        n_on, n_off = on["calls"], off["calls"]
        same = torch.equal(off["lat"], on["lat"])
        if n_off != 0:
            note, ok = "flag OFF still folded!", False
        elif n_on == 0:
            # The fold provably never ran, so the flag cannot have changed this mode. No latent
            # comparison is made, because for fp16 none is possible.
            note, ok = "n/a - fold never ran", True
        else:
            note = "bit-identical" if same else f"DIFFER relL2 {rel(on['lat'], off['lat']):.2e}"
            ok = same
        bad += 0 if ok else 1
        print(f"{mode:16}{n_on:>15}{n_off:>16}{note:>22}{'ok' if ok else 'BROKEN':>10}")
    print()
    if bad:
        print(f"{bad} mode(s) BROKEN -- do not enable by default.")
        return 1
    print("ALL MODES CLEAN: the fold ran only where it is eligible, and where it ran the latent is "
          "bit-identical. Modes it declined are untouched by construction, not by comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
