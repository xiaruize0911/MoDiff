"""End-to-end gate for the decoder skip-concat fold: same latents, and the fold actually ran.

MODIFF_CAT2_FOLD=1 makes the decoder hand the two halves to the fused ResBlock instead of their
concatenation. The kernel is already gated to bit-exactness at the op level (test_cat2_gn_fold.py) and
the wired entry reproduces cat2 + the ordinary prologue bit-for-bit including the in-place a_hat
update. This checks the whole model.

TWO ASSERTIONS, and the second is the one that makes the first mean anything:

  1. The sampled latent with the fold ON is BIT-IDENTICAL to the latent with it OFF. Not close --
     identical. Every kernel in the chain is bit-exact, so anything else is a bug.

  2. The fold KERNEL WAS ACTUALLY CALLED, counted by wrapping the pybind entry. Without this the test
     passes trivially when the fold silently declines every block -- which is exactly the failure this
     session hit three times (a flag that swapped 0 parameters, a gate comparing a function against
     itself, a threshold that could not fail). Identical latents from a fold that never ran is not
     evidence of correctness, it is evidence of nothing.

`_CAT2_FOLD` is read at import, so the two arms must be separate PROCESSES; this script runs itself as
a subprocess twice and compares what each wrote.

Run: python integration/tests/verify_cat2_fold_e2e.py    # ~4 min, needs the GPU
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

SCRATCH = "/tmp/claude-0/-workspace/3c7113f0-c033-4c7a-8d00-7fa278467591/scratchpad"


def child(tag):
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

    H.STEPS, H.BATCH = 20, 4
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    r, m, s = H.build("int4", B._default_calibration_path("int4"), "static")
    H.SEED = 1234
    H.latent(r, m, s)                    # discard: attention self-calibration
    lat, _ = H.latent(r, m, s)
    torch.save({"lat": lat, "calls": calls[0]}, f"{SCRATCH}/cat2_fold_{tag}.pt")
    print(f"  [{tag}] fold kernel calls: {calls[0]}", flush=True)
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2])

    import torch
    for tag, flag in (("off", "0"), ("on", "1")):
        env = dict(os.environ, MODIFF_CAT2_FOLD=flag,
                   PYTHONPATH=f"{ROOT}:{ROOT}/src/taming-transformers")
        print(f"running MODIFF_CAT2_FOLD={flag} ...", flush=True)
        p = subprocess.run([sys.executable, __file__, "--child", tag], env=env,
                           capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if "fold kernel calls" in line:
                print(line)
        if p.returncode != 0:
            print(f"FAIL: the {tag} arm exited {p.returncode}")
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            return 1

    off = torch.load(f"{SCRATCH}/cat2_fold_off.pt", weights_only=False)
    on = torch.load(f"{SCRATCH}/cat2_fold_on.pt", weights_only=False)
    print(f"\nfold kernel calls   off={off['calls']}   on={on['calls']}")
    if on["calls"] == 0:
        print("FAIL (VACUOUS): the fold never ran with the flag ON, so identical latents prove "
              "nothing. Find out why every block declined before reading anything into this.")
        return 1
    if off["calls"] != 0:
        print("FAIL: the fold ran with the flag OFF -- the switch does not gate what it claims to.")
        return 1
    same = torch.equal(off["lat"], on["lat"])
    rel = float((on["lat"].float() - off["lat"].float()).norm()
                / off["lat"].float().norm()) if not same else 0.0
    print(f"latents bit-identical: {same}" + ("" if same else f"   relL2 {rel:.3e}"))
    print()
    if same:
        print(f"PASS: the fold ran on {on['calls']} blocks and the sampled latent is unchanged, "
              f"bit for bit.")
        return 0
    print("FAIL: the fold changed the latent. Every kernel in this chain is bit-exact at the op "
          "level, so this is a wiring bug -- most likely the concatenation the fold returns is not "
          "what the skip conv and the residual then consume.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
