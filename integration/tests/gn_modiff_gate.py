"""Deterministic replacement for gn_modiff_verify_realinput.py: capture the inputs once, replay forever.

WHAT IT CHECKS (unchanged): MoDiff's fused GN->delta-quantize kernel against the two-kernel reference
(`group_norm_silu_nhwc` + `step1_static_quantize[_pack]_fprop_silu`) on REAL activations, requiring
bit-identical int codes and a_hat. csrc/modiff/norm/group_norm_silu.cu cites this differential twice as
the reason a previous reduction change was reverted, at max_code_diff = 1.

WHY A REWRITE. The original took a **max** over the first 40 fused calls of a **live sample**, and fp16
sampling here varies ~4-6e-3 between processes (which is why cat2_fold_2026-08-13's gate counts kernel
calls instead of comparing latents). So every run instrumented DIFFERENT data. Measured, five runs at one
fixed configuration: max_code_diff 35, 38, 34, 27, 36 -- and at another, 81, 42, 30, 23, 35. A
zero-tolerance gate whose own statistic ranges 23-81 without any change cannot gate anything, and cannot
discriminate two candidate implementations: a first n=1 reading of that data (35 vs 81) looked decisive
and was refuted at n=5.

So: --capture writes the exact inputs of the first N eligible calls to a file; --replay reads that file
and compares. Same file in, same numbers out, every time and on any machine with the same build.

AND IT REPORTS PER CASE, not just a max. A max says "something differs by 27"; the per-case table says
WHICH layers differ and by how much, which is what any diagnosis of a non-zero result needs. The original
threw that away.

    python integration/tests/gn_modiff_gate.py --capture /tmp/gn_cases_int8.pt --mode int8
    python integration/tests/gn_modiff_gate.py --replay  /tmp/gn_cases_int8.pt --mode int8

The capture file is NOT committed -- it is tens of MB of activations. Capture is the slow step (a real
20-step sample); replay is seconds, which is what makes this usable as a gate rather than an errand.
"""
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src/taming-transformers"))
os.environ.setdefault("MODIFF_QUANT_LINEAR", "1")
os.environ.setdefault("MODIFF_QUANT_ATTN", "1")
os.environ.setdefault("MODIFF_LINEAR_OUT_I8", "0")

import torch                                                              # noqa: E402
import modiff_cutlass as M                                                # noqa: E402
import integration.fused_ops.fused_resblock as FR                         # noqa: E402

BATCH, STEPS = 8, 20


def _cl(t):
    """Restore channels_last, which torch.save/load does not preserve for a 4-D tensor."""
    return t.contiguous(memory_format=torch.channels_last) if t.dim() == 4 else t.contiguous()


def capture(path, mode, n_cases):
    import integration.benchmarks.benchmark_ldm as B
    calib = f"integration/calibration/{mode}_calibration.pt"
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="integration/results/gn_gate",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
                          calibration_path=calib, linear_backend="int_gemm")
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, BATCH)
    is_int4 = mode == "int4"

    cases = []
    orig = FR._prequant_gn_conv_modiff

    def wrap(x, gn, conv, mod_scale=None, mod_shift=None, residual=None, x2=None, **kw):
        # x2 -> the cat2-folded path, which takes a different fused kernel and is not comparable to the
        # two-kernel reference; forwarded uninstrumented rather than counted.
        if (x2 is None and len(cases) < n_cases and conv is not None
                and getattr(conv, "modiff_enabled", False)
                and hasattr(conv, "can_gn_fuse_modiff") and conv.can_gn_fuse_modiff(x)
                and x.is_contiguous(memory_format=torch.channels_last)):
            w, b = gn._cast_params(x.dtype)
            N, C = x.size(0), x.size(1)
            if mod_scale is not None:
                ms = mod_scale.reshape(N, C).contiguous()
                sh = mod_shift.reshape(N, C).contiguous()
            else:
                ms = sh = x.new_empty(0)
            smooth = (conv._smooth_inv_flat if hasattr(conv, "_smooth_inv_flat")
                      else x.new_empty(0, dtype=torch.float32))
            dyn = FR._delta_gn_dynamic_args_any(conv, x.device, is_int4)
            cases.append({
                "x": x.detach().cpu().clone(), "w": w.detach().cpu().clone(),
                "b": b.detach().cpu().clone(), "ms": ms.detach().cpu().clone(),
                "sh": sh.detach().cpu().clone(),
                "a_hat": conv.a_hat_cache.detach().cpu().clone(),
                "scale": conv.static_input_scale.view(1).detach().cpu().clone(),
                "smooth": smooth.detach().cpu().clone(),
                # only the tensors in dyn need moving; the trailing scalars are plain Python
                "dyn_t": [d.detach().cpu().clone() if torch.is_tensor(d) else d for d in dyn],
                "ng": gn.num_groups, "eps": gn.eps, "shape": tuple(x.shape),
                "layer": f"C{C}/{x.size(2)}x{x.size(3)}",
            })
        return orig(x, gn, conv, mod_scale, mod_shift, residual, x2=x2, **kw)

    FR._prequant_gn_conv_modiff = wrap
    torch.manual_seed(1234)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    FR._prequant_gn_conv_modiff = orig

    torch.save({"mode": mode, "cases": cases}, path)
    print(f"captured {len(cases)} cases -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


def replay(path, mode, verbose):
    blob = torch.load(path, map_location="cuda")
    assert blob["mode"] == mode, f"capture is {blob['mode']}, asked for {mode}"
    is_int4 = mode == "int4"
    cases = blob["cases"]

    rows = []
    for i, c in enumerate(cases):
        x = _cl(c["x"].cuda())
        w, b = c["w"].cuda(), c["b"].cuda()
        ms, sh = c["ms"].cuda(), c["sh"].cuda()
        scale, smooth = c["scale"].cuda(), c["smooth"].cuda()
        dyn = [d.cuda() if torch.is_tensor(d) else d for d in c["dyn_t"]]
        ng, eps = c["ng"], c["eps"]

        with torch.inference_mode():
            a_ref = _cl(c["a_hat"].cuda())
            normed = M.group_norm_silu_nhwc(x, w, b, ng, eps, False, ms, sh)
            if is_int4:
                q_ref = M.step1_static_quantize_pack_int4_fprop_silu(normed, a_ref, scale, smooth)
                q_fus = M.group_norm_silu_delta_quantize_pack_nhwc(
                    x, w, b, _cl(c["a_hat"].cuda()), ng, eps, True, scale, smooth, ms, sh, *dyn[:-1])
            else:
                q_ref = M.step1_static_quantize_fprop_silu(normed, a_ref, scale, smooth)
                q_fus = M.group_norm_silu_delta_quantize_nhwc(
                    x, w, b, _cl(c["a_hat"].cuda()), ng, eps, True, scale, smooth, ms, sh, *dyn)
            torch.cuda.synchronize()
            d = (q_ref.int() - q_fus.int()).abs()
            cd = int(d.max().item())
            nz = int((d != 0).sum().item())
            frac = nz / d.numel()

            a_fus = _cl(c["a_hat"].cuda())
            if is_int4:
                M.group_norm_silu_delta_quantize_pack_nhwc(
                    x, w, b, a_fus, ng, eps, True, scale, smooth, ms, sh, *dyn[:-1])
            else:
                M.group_norm_silu_delta_quantize_nhwc(
                    x, w, b, a_fus, ng, eps, True, scale, smooth, ms, sh, *dyn)
            torch.cuda.synchronize()
            ad = float((a_ref.float() - a_fus.float()).abs().max().item())
        rows.append({"i": i, "layer": c["layer"], "shape": c["shape"],
                     "code_diff": cd, "n_diff": nz, "frac_diff": frac, "ahat_diff": ad})

    bad = [r for r in rows if r["code_diff"] != 0]
    print(f"\nmode={mode}  cases={len(rows)}  nonzero={len(bad)}  "
          f"max_code_diff={max(r['code_diff'] for r in rows)}  "
          f"max_ahat_diff={max(r['ahat_diff'] for r in rows):.3e}")
    if verbose or bad:
        print(f"\n{'#':>3} {'layer':>14} {'code':>6} {'n_diff':>10} {'frac':>10} {'ahat':>11}")
        for r in (rows if verbose else bad):
            print(f"{r['i']:>3} {r['layer']:>14} {r['code_diff']:>6} {r['n_diff']:>10} "
                  f"{r['frac_diff']:>10.2e} {r['ahat_diff']:>11.3e}")
    #: grouped by layer, because a per-layer pattern is the difference between "one kernel is wrong"
    #: and "everything drifts a little"
    if bad:
        import collections
        by = collections.Counter(r["layer"] for r in bad)
        print(f"\nnonzero cases by layer: {dict(by)}")
        clean = collections.Counter(r["layer"] for r in rows if r["code_diff"] == 0)
        print(f"clean cases by layer:   {dict(clean)}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", metavar="PATH")
    ap.add_argument("--replay", metavar="PATH")
    ap.add_argument("--mode", default="int8", choices=["int8", "int4"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.capture:
        return capture(a.capture, a.mode, a.n)
    if a.replay:
        return replay(a.replay, a.mode, a.verbose)
    ap.error("need --capture or --replay")


if __name__ == "__main__":
    sys.exit(main())
