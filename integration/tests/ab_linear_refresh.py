"""Paired A/B of MODIFF_LINEAR_DELTA_REFRESH, both arms on ONE model object.

The 42 attention projections recompute their dynamic delta scale on EVERY modulated step:
MODIFF_DELTA_REFRESH is read in the conv wrappers' __init__ only and never reached
QuantLinearWxAx. Measured premise (docs/profile_kernels_layers_2026-08-11): the delta_quantize
kernel bucket moves +4.7 ms/step with K at conv-only AND +4.7 at conv+proj, i.e. the projections'
share is K-independent. Their own recomputation is ~3.7 ms/step and a K=4 schedule should remove
about three quarters, the ratio the conv path gets.

`delta_refresh` is set in __init__ from the env, but it is a plain attribute, so both arms can run on
the same model by walking the 42 modules and flipping it -- which makes this paired rather than
cross-session. That mattered before: the cross-session comparison of the updown fusion had a drift
control moving as much as the effect.

Each arm also counts how many times the absmax pass actually ran, so an arm proves it is the arm it
claims rather than being trusted to be.

Run: python integration/tests/ab_linear_refresh.py [--k 4] [--batch 128] [--steps 200]
"""
import argparse, os, statistics, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "integration/benchmarks/report")]
import torch, modiff_cutlass as mc                                          # noqa: E402

ENV = {"MODIFF_QUANT_LINEAR": "1", "MODIFF_QUANT_ATTN": "1", "MODIFF_QUANT_ATTN_STATIC": "1",
       "MODIFF_QATTN_FLASH": "1", "MODIFF_FLASH_GATE": "on", "MODIFF_QUANT_ATTN_ALLT": "0",
       "MODIFF_LINEAR_OUT_I8": "0", "MODIFF_FUSE_PROJ_QUANT": "1", "MODIFF_LINEAR": "1",
       "MODIFF_ACT_BITS": "8", "MODIFF_DELTA_MODE": "dynamic", "MODIFF_WARMUP_STEPS": "5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--conv-k", type=int, default=4, help="MODIFF_DELTA_REFRESH for the convs")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--warmups", type=int, default=1)
    a = ap.parse_args()
    os.environ.update(ENV)
    os.environ["MODIFF_DELTA_REFRESH"] = str(a.conv_k)
    os.environ["MODIFF_LINEAR_DELTA_REFRESH"] = "1"

    import integration.benchmarks.benchmark_ldm as B
    from kernel_suites_bench import CALIB
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="docs/updown_refresh_fusion_2026-08-10/tmp_out",
                          batch_size=a.batch, steps=a.steps, shape=(4, 32, 32),
                          calibration_path=CALIB.get("int8"), linear_backend="fp16")
    model, sampler = r._setup_model("int8")
    cond = r._cond_kwargs(model, a.batch)
    from integration.kernels.wxax_linear import QuantLinearWxAx
    projs = list({id(m): m for m in model.model.diffusion_model.modules()
                  if isinstance(m, QuantLinearWxAx)}.values())
    print(f"{len(projs)} wxax projections, conv K={a.conv_k}, proj K under test = {a.k}")

    # Count the absmax pass so each arm is self-verifying.
    hits = {"n": 0}
    orig = mc.delta_absmax_fp16
    def counting(*x, **kw):
        hits["n"] += 1
        return orig(*x, **kw)
    mc.delta_absmax_fp16 = counting

    def one(k):
        for m in projs:
            m.delta_refresh = k
            m._step = 0
        hits["n"] = 0
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=a.steps, batch_size=a.batch, shape=r.shape, eta=0.0,
                           verbose=False, **cond)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / a.steps, hits["n"] / a.steps

    for _ in range(a.warmups):
        one(1); one(a.k)
    on_ms, off_ms, on_c, off_c = [], [], [], []
    for _ in range(a.repeats):
        for k, ms_l, c_l in ((a.k, on_ms, on_c), (1, off_ms, off_c)):
            ms, c = one(k); ms_l.append(ms); c_l.append(c)
    pairs = [o - n for n, o in zip(on_ms, off_ms)]
    print(f"\nbatch {a.batch}, {a.steps} steps, {a.repeats} paired repeats, "
          f"{torch.cuda.get_device_name(0)}\n")
    print("| arm | ms/step | absmax calls/step |")
    print("|---|---:|---:|")
    print(f"| proj K={a.k} | {statistics.median(on_ms):.2f} | {statistics.median(on_c):.2f} |")
    print(f"| proj K=1    | {statistics.median(off_ms):.2f} | {statistics.median(off_c):.2f} |")
    print(f"\npaired (K=1 - K={a.k}): " + ", ".join(f"{p:+.2f}" for p in pairs))
    print(f"median {statistics.median(pairs):+.2f} ms/step recovered")
    if len(pairs) > 1:
        sd = statistics.stdev(pairs); sem = sd / len(pairs) ** 0.5
        print(f"stdev {sd:.3f}, SEM {sem:.3f} -> "
              f"{'RESOLVED' if abs(statistics.median(pairs)) > 2 * sem else 'not resolved'}")


if __name__ == "__main__":
    main()
