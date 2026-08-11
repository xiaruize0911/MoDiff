"""Method A: attribute the K=1 full-MoDiff step time by DIFFERENTIAL wall clock, no profiler.

`component_profile.py` (docs/delta_clip_2026-08-06) tried to answer "how long did each component
take" by wrapping scopes in `record_function`, and produced two invalid runs: forward hooks missed
the 62 convs the ResBlock dispatches directly, and `ProfilerActivity.CPU` plus double-counted scope
entries put the total at 235.74 ms/step against a measured 106.30. Neither defect is fixable by more
repetitions -- they are bias, not variance.

This asks the other question instead: **how much does the whole change when one component changes.**
Every arm is a full sampling run timed with CUDA events and NO profiler attached, which is the
measurement this project already trusts (CV <= 0.2% at batch 128 / DDIM 200). The cost is that a
marginal is not an absolute and components interact -- but the interaction is the finding worth
having here, since projection MoDiff's own kernels are ~9 ms while turning it on costs +25.8.

Arms are grouped by the reference point they are a delta from, because a marginal only means
something against a stated base:

  anchors        fp16, int8 PTQ
  ladder         int8 PTQ -> +conv MoDiff -> +K=1 -> +projection MoDiff   (the shipped -> paper path)
  knockouts      one component reverted from the K=1 full-MoDiff base
  epilogue       MODIFF_FUSE_PROJ_QUANT=0 against BOTH bases; against int8 PTQ this is what the
                 flash qout epilogue is worth, i.e. the ceiling on what an a_hat-aware flash qout
                 (Part 3 in docs/delta_clip_2026-08-06) could recover

Every arm records a route check (attention block class, wxax modiff flags, conv modiff flags, qout
eligibility after warm-up). Those guards exist because this codebase has twice measured a different
configuration than the one it labelled -- see e2e_three_mode_bench's route-check comment.

Writes data/differential_timing.json.

The checkpoint is the 856-byte stub with an empty state_dict, so weights are random. Timing is
unaffected (no data-dependent control flow); nothing here is a quality statement.
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch                                                                    # noqa: E402
import integration.benchmarks.benchmark_ldm as B                                # noqa: E402
from ck_bench_stats import summarize, stability_verdict                         # noqa: E402

CALIB8 = "integration/calibration/int8_calibration.pt"

#: The env every arm starts from, so an arm's dict states its whole difference from the others.
#: Same values as e2e_three_mode_bench.QUANT_ENV -- the harness whose numbers these must line up
#: with -- plus the MoDiff knobs left at their shipped defaults, written out rather than implied.
BASE_ENV = {
    "MODIFF_QUANT_LINEAR": "1",
    "MODIFF_QUANT_ATTN": "1",
    "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1",
    "MODIFF_FLASH_GATE": "on",
    "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0",
    "MODIFF_FUSE_PROJ_QUANT": "1",
    "MODIFF_LINEAR": "1",
    "MODIFF_DELTA_REFRESH": "1",
    "MODIFF_DELTA_CLIP": "1.0",
    "MODIFF_ACT_Q": "127",
    "MODIFF_DELTA_MODE": "dynamic",
    "MODIFF_WARMUP_STEPS": "5",
}
#: Cleared before every arm: benchmark_ldm and the kernels both write into os.environ
#: (`if quant_lin: os.environ["MODIFF_FUSE_GN_QKV"] = "0"`), so a previous arm can leak into the
#: next one inside a single process. Anything MODIFF_* not in BASE_ENV is removed.
_STICKY = ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND", "MODIFF_FUSE_GN_QKV")


#: Facts a post-setup hook wants in the arm's route check. Cleared per arm.
HOOK_FACTS = {}


def _no_gn_qkv_modiff_fusion(model):
    """Knock out the 2026-08-06 GN+delta-quantize+a_hat qkv fusion (it has no env gate).

    Returns the undo thunk. Patching the class, not the instances: the method is looked up on the
    class at call time, and every block in this model is the same class.

    The stub counts its calls, because a route check cannot see this one: a knockout that never
    fired and a knockout worth 0 ms both read as "no change", and this arm exists to distinguish
    "the fusion is worth nothing" from "I did not turn it off".
    """
    from integration.fused_ops.quantized_std_attention import QuantizedStandardAttentionBlock as Q
    orig = Q._qkv_from_gn_modiff_fused
    HOOK_FACTS["gn_qkv_fusion_calls_suppressed"] = 0

    def stub(self, *a, **k):
        HOOK_FACTS["gn_qkv_fusion_calls_suppressed"] += 1
        return None

    Q._qkv_from_gn_modiff_fused = stub
    return lambda: setattr(Q, "_qkv_from_gn_modiff_fused", orig)


def _no_conv_modiff(model):
    """Turn the temporal delta path off on the 70 eligible convs, leaving them int8-quantized."""
    from integration.kernels.int8_optimized import enable_modiff_mode
    enable_modiff_mode(model.model.diffusion_model, False)
    return lambda: None


#: The canonical dataset: every arm, read by committed figures in several reports. A subset run may
#: not write here -- see the guard in main().
DEFAULT_OUTPUT = "docs/component_attribution_2026-08-07/data/differential_timing.json"

#: (label, base it is a delta from, mode, env overrides, post-setup hook, why)
ARMS = [
    # The three explicit zeros are load-bearing and the first run of this script got them wrong.
    # BASE_ENV carries MODIFF_QUANT_LINEAR=1, and benchmark_ldm's quant_lin block is NOT gated on
    # the mode: with it set, the fp16 arm converted 79 nn.Linear to W8A8 and switched attention to
    # token-major, i.e. measured a partly quantized model under the label "fp16". The route check
    # is what caught it ('wxax': 79 where fp16 must be 0), which is why _assert_route exists below.
    ("fp16", None, "fp16",
     {"MODIFF_QUANT_LINEAR": "0", "MODIFF_QUANT_ATTN": "0", "MODIFF_LINEAR": "0"}, None,
     "reference"),
    ("int8_ptq", None, "int8_baseline", {"MODIFF_LINEAR": "0"}, None,
     "quantized, no MoDiff anywhere; the qout epilogue is live on all 21 blocks"),

    # ---- ladder: int8 PTQ -> the paper's configuration, one component at a time ----
    ("modiff_conv_k4", "int8_ptq", "int8", {"MODIFF_LINEAR": "0", "MODIFF_DELTA_REFRESH": "4"}, None,
     "+ conv MoDiff at the shipped refresh; the pre-2026-08-06 default"),
    ("modiff_conv_k1", "modiff_conv_k4", "int8", {"MODIFF_LINEAR": "0"}, None,
     "+ per-step delta scale (K=4 -> K=1)"),
    ("modiff_full_k1", "modiff_conv_k1", "int8", {}, None,
     "+ MoDiff on the 42 attention projections -- the paper's full datapath, and the BASE below"),
    ("modiff_full_k4", "modiff_full_k1", "int8", {"MODIFF_DELTA_REFRESH": "4"}, None,
     "the same full datapath at K=4, so the projections' cost can be read at both refreshes"),
    # Added 2026-08-11. A NAMED ARM rather than an env var a reader has to know to set: without one,
    # the report's figures and a fresh clone's behaviour drift apart, which is what happened when the
    # +2.81 ms lived only in prose. MODIFF_LINEAR_DELTA_REFRESH gives the 42 projections the schedule
    # the convs have had; it defaults to 1 (off) because it changes numerics on reuse steps, so this
    # arm documents a deliberately non-default configuration and its label says so.
    ("modiff_full_k4_projk4", "modiff_full_k4", "int8",
     {"MODIFF_DELTA_REFRESH": "4", "MODIFF_LINEAR_DELTA_REFRESH": "4"}, None,
     "K=4 on the convs AND on the 42 projections; the projections' unconditional absmax was the "
     "last K-independent term (docs/profile_kernels_layers_2026-08-11)"),

    # Added 2026-08-12. Route (b): the dual-output qkv GEMM emits int8 straight into flash's gather
    # path, so the three aq_* re-quantize kernels disappear on the 10 hd=48 blocks. NOT the whole
    # 4.60 ms attn_quantize bucket -- the 5 hd=24 blocks are ineligible (24 B/token vs the int8
    # gather's 16 B cp.async), and the packed kernel is 1.8x/1.5x the mma kernel's time, so it pays
    # part of the saving back. Kernel microbenchmark predicted +0.79 ms/step and the paired A/B
    # measured +0.79 (integration/tests/bench_flash_packed_vs_unpacked.py, ab_route_b_qkv_i8.py).
    # Opt-in, like the projection refresh above, and for the same reason: it changes the score kernel.
    ("modiff_full_k4_projk4_qkvi8", "modiff_full_k4_projk4", "int8",
     {"MODIFF_DELTA_REFRESH": "4", "MODIFF_LINEAR_DELTA_REFRESH": "4",
      "MODIFF_FUSE_QKV_I8": "1"}, None,
     "route (b) on top of both refresh schedules: int8 qkv -> flash gather on the 10 hd=48 blocks"),

    # ---- knockouts from modiff_full_k1 ----
    ("base_no_qattn", "modiff_full_k1", "int8", {"MODIFF_QUANT_ATTN": "0"}, None,
     "QK^T/AV reverts to fp16 SDPA; projections stay int8 + MoDiff"),
    ("base_no_qlinear", "modiff_full_k1", "int8", {"MODIFF_QUANT_LINEAR": "0"}, None,
     "qkv/proj revert to fp16 nn.Linear; QK^T/AV stays int8"),
    ("base_no_conv_modiff", "modiff_full_k1", "int8", {}, _no_conv_modiff,
     "convs stay int8 but drop the temporal delta path"),
    ("base_no_gnqkv_fusion", "modiff_full_k1", "int8", {}, _no_gn_qkv_modiff_fusion,
     "the qkv GN+delta-quantize+a_hat fusion landed 2026-08-06, reverted to three passes"),

    # ---- the attention output epilogue, measured where it is live and where it is not ----
    ("ptq_no_projquant", "int8_ptq", "int8_baseline",
     {"MODIFF_LINEAR": "0", "MODIFF_FUSE_PROJ_QUANT": "0"}, None,
     "kills the flash qout epilogue where it IS engaged -> what Part 3 could recover"),
    ("base_no_projquant", "modiff_full_k1", "int8", {"MODIFF_FUSE_PROJ_QUANT": "0"}, None,
     "control: at the base the epilogue is already off (0/21), so this must be a no-op"),
]


def apply_env(overrides):
    for k in list(os.environ):
        if k.startswith("MODIFF_") and k not in BASE_ENV:
            del os.environ[k]
    for k in _STICKY:
        os.environ.pop(k, None)
    os.environ.update(BASE_ENV)
    os.environ.update(overrides)


def route_check(model, mode):
    """Config facts, read from the MODULES after warm-up. No profiler, no timing impact."""
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    unet = model.model.diffusion_model
    r = {}
    blks = [m for m in unet.modules()
            if type(m).__name__ in ("QuantizedStandardAttentionBlock", "TokenMajorAttentionBlock")]
    r["attn_blocks"] = len(blks)
    r["attn_class"] = type(blks[0]).__name__ if blks else None
    if blks and hasattr(blks[0], "_qout_eligible"):
        r["qout_eligible"] = sum(bool(b._qout_eligible()) for b in blks)
    # Route (b) eligibility. `_qkv_i8_ok` also takes T, which is a per-forward quantity no module
    # stores, so this passes the T of each block's own resolution: hd<=48 means T in {256, 64} here,
    # both T%64==0, so the shape half of the gate is decided by head_dim alone in THIS model. Stated
    # rather than assumed, because the previous version of that gate was wrong in exactly this area.
    if blks and hasattr(blks[0], "_qkv_i8_ok"):
        r["attn_qkv_i8"] = sum(bool(b._qkv_i8_ok(256)) for b in blks)
    try:
        from integration.kernels.wxax_linear import QuantLinearWxAx
        wx = [m for m in unet.modules() if isinstance(m, QuantLinearWxAx)]
        r["wxax"] = len(wx)
        r["wxax_modiff"] = sum(bool(getattr(m, "modiff", False)) for m in wx)
        r["wxax_out_i8"] = sum(bool(getattr(m, "_out_i8", False)) for m in wx)
        r["wxax_bias_res"] = sum(bool(getattr(m, "_use_bias_res", False)) for m in wx)
    except Exception as e:
        r["wxax_error"] = str(e)
    convs = [m for m in unet.modules() if isinstance(m, OptimizedInt8Conv2d)]
    r["int8_convs"] = len(convs)
    r["conv_modiff_on"] = sum(bool(getattr(m, "modiff_enabled", getattr(m, "use_modiff", False)))
                              for m in convs)
    r["conv_delta_refresh"] = sorted({int(getattr(m, "delta_refresh", -1)) for m in convs})
    r.update(HOOK_FACTS)
    return r


def _assert_route(label, mode, over, rc):
    """Fail loudly when an arm is not the configuration its label claims.

    Every entry here is a mistake this harness has actually made or that its ancestors document.
    A wrong arm does not look wrong -- it completes and reports a believable number.
    """
    def want(cond, msg):
        assert cond, f"{label}: {msg} -- route was {rc}"

    if mode == "fp16":
        want(rc.get("wxax", 0) == 0, "fp16 arm has quantized Linears")
        want(rc.get("int8_convs", 0) == 0, "fp16 arm has quantized convs")
        want(rc.get("conv_modiff_on", 0) == 0, "fp16 arm has conv MoDiff on")
        return
    want(rc.get("attn_blocks") == 21, "expected 21 attention blocks")
    quant_lin = over.get("MODIFF_QUANT_LINEAR", "1") == "1"
    # `is_modiff` in benchmark_ldm is the flag AND a mode whitelist, so a baseline mode never
    # receives projection MoDiff however MODIFF_LINEAR is set. Mirror that, do not re-derive it.
    modiff_lin = quant_lin and over.get("MODIFF_LINEAR", "1") == "1" and mode == "int8"
    if quant_lin:
        want(rc.get("wxax") == 42, "expected the 42 attention qkv/proj as wxax Linears")
        want(bool(rc.get("wxax_modiff")) == modiff_lin,
             f"projection MoDiff should be {modiff_lin}")
    else:
        want(not rc.get("wxax"), "MODIFF_QUANT_LINEAR=0 arm still has wxax Linears")
    # The fused int8-output epilogue and projection MoDiff are mutually exclusive by construction
    # (_use_bias_res is `... and not modiff`), and that exclusion is the whole cost story for the
    # projections -- so check it rather than assume it.
    if "qout_eligible" in rc:
        if modiff_lin or over.get("MODIFF_FUSE_PROJ_QUANT") == "0":
            want(rc["qout_eligible"] == 0, "qout epilogue should be disabled")
        elif quant_lin:
            want(rc["qout_eligible"] == 21, "qout epilogue should be live on all 21 blocks")
    # Route (b) must be live on exactly the 10 hd=48 blocks when asked for and nowhere otherwise.
    # An arm that silently declined would time production twice and report a believable ~0.
    if "attn_qkv_i8" in rc:
        want(rc["attn_qkv_i8"] == (10 if over.get("MODIFF_FUSE_QKV_I8") == "1" else 0),
             "route (b) should be live on 10 of 21 blocks iff MODIFF_FUSE_QKV_I8=1")


def run_arm(arm, a):
    label, base, mode, over, hook, why = arm
    print(f"\n{'='*72}\n{label}   (mode={mode}, delta-from={base})\n  {why}\n{'='*72}", flush=True)
    apply_env(over)
    HOOK_FACTS.clear()
    print("  env:", {k: os.environ[k] for k in sorted(BASE_ENV)}, flush=True)

    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="integration/results/component_attribution",
        batch_size=a.batch, steps=a.steps, shape=(4, 32, 32),
        calibration_path=None if mode == "fp16" else CALIB8,
        linear_backend="fp16" if mode == "fp16" else "int_gemm",
    )
    model, sampler = runner._setup_model(mode)
    cond = runner._cond_kwargs(model, a.batch)
    undo = hook(model) if hook is not None else (lambda: None)

    def sample():
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=a.steps, batch_size=a.batch, shape=runner.shape, eta=0.0,
                           verbose=False, **cond)

    for _ in range(a.warmups):
        sample()
    torch.cuda.synchronize()

    # After warm-up: the flash static scales need MODIFF_ATTN_CALIB_STEPS forwards before
    # _qout_eligible() can report anything but 0, and conv MoDiff needs its warm-up rounds.
    rc = route_check(model, mode)
    print(f"  route: {rc}", flush=True)
    if hook is None:                 # the hooks deliberately break the invariants below
        _assert_route(label, mode, over, rc)

    times = []
    for _ in range(a.repeats):
        s, e = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        s.record()
        sample()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)          # us for the whole batch

    med, mean = statistics.median(times), statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    st = summarize(times)
    out = {
        "mode": mode, "delta_from": base, "why": why,
        "env": {k: os.environ.get(k) for k in sorted(BASE_ENV)},
        "hook": getattr(hook, "__name__", None),
        "route_check": rc,
        "stats": st, "stability": stability_verdict(st),
        "wall_us_per_batch": med, "wall_mean_us": mean, "wall_stdev_us": sd,
        "wall_cv_pct": sd / mean * 100 if mean else 0.0,
        "wall_all_us": [round(t, 1) for t in times],
        "wall_spread_pct": (max(times) - min(times)) / min(times) * 100,
        "ms_per_step": med / 1e3 / a.steps,
        "ms_per_sample": med / 1e3 / a.batch,
    }
    print(f"  {out['ms_per_step']:8.3f} ms/step   CV {out['wall_cv_pct']:.3f}%   "
          f"spread {out['wall_spread_pct']:.3f}%", flush=True)

    undo()
    del model, sampler, runner
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--arms", default="", help="comma-separated subset of labels")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    a = ap.parse_args()

    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    arms = [x for x in ARMS if not want or x[0] in want]

    # A subset run writing to the default path REPLACES the canonical dataset with two arms, which
    # is a committed file several reports' figures read. It happened on 2026-08-12 and the only
    # symptom was `make_plots.py` losing bars. Refuse instead: the caller has to name the file, and
    # a full run (no --arms) keeps working exactly as before.
    if want and a.output == DEFAULT_OUTPUT:
        sys.exit(f"--arms is a SUBSET run and would overwrite the canonical {DEFAULT_OUTPUT}, "
                 f"which holds all {len(ARMS)} arms and is read by committed figures.\n"
                 f"Pass an explicit --output, e.g.\n"
                 f"  --output docs/<your-report>/data/differential_timing_"
                 f"{want[0].replace('/', '_')}.json")

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "batch": a.batch, "steps": a.steps, "repeats": a.repeats, "warmups": a.warmups,
           "base_env": BASE_ENV, "arms": {}}
    t0 = time.time()
    for i, arm in enumerate(arms):
        out["arms"][arm[0]] = run_arm(arm, a)
        el = time.time() - t0
        print(f"  [{i+1}/{len(arms)}] {el/60:.1f} min elapsed, "
              f"~{el/(i+1)*(len(arms)-i-1)/60:.1f} min left", flush=True)
        with open(a.output, "w") as f:                 # checkpoint after every arm
            json.dump(out, f, indent=1)

    # Marginals. Each is (arm - its stated base); a positive number is what the arm's change COSTS.
    for lab, d in out["arms"].items():
        b = d["delta_from"]
        if b and b in out["arms"]:
            d["delta_ms_per_step"] = d["ms_per_step"] - out["arms"][b]["ms_per_step"]
            d["ratio_vs_base"] = d["ms_per_step"] / out["arms"][b]["ms_per_step"]
    fp = (out["arms"].get("fp16") or {}).get("ms_per_step")
    for d in out["arms"].values():
        d["speedup_vs_fp16"] = fp / d["ms_per_step"] if fp else None

    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWROTE {a.output}\n")

    print(f"{'arm':<24}{'ms/step':>10}{'vs fp16':>9}{'delta':>10}  {'from':<18}{'CV':>7}")
    for lab, d in out["arms"].items():
        sp = f"{d['speedup_vs_fp16']:.3f}x" if d["speedup_vs_fp16"] else "—"
        dl = f"{d.get('delta_ms_per_step'):+.2f}" if d.get("delta_ms_per_step") is not None else "—"
        print(f"{lab:<24}{d['ms_per_step']:>10.2f}{sp:>9}{dl:>10}  "
              f"{str(d['delta_from'] or ''):<18}{d['wall_cv_pct']:>6.2f}%")


if __name__ == "__main__":
    main()
