"""End-to-end: everything blockwise, swept over block size. Quality only.

Fake-quant simulation on the LSUN-Churches LDM (NOT the kernel path -- see
FINDINGS.md section 3 for why the real conv cannot take a reduction-axis scale).
A forward_pre_hook on each live ResBlock conv replaces the input with the
MoDiff-reconstructed a_hat, using act_fake_quant's linearity trick.

The one thing this script is careful about: only GRANULARITY changes between arms.
Warmup rounds (5) and the delta-scale refresh cadence (4) are held at the shipped
values in every arm, including the baseline. The committed group-quant run
(docs/cache_schemes_report_2026-08-28/data/imode_group_quant.json) refreshed s*
every step, which moved granularity and cadence together and so cannot say which
one paid.

Arms, per bit-width:
  shipped   W per-out-channel (absmax@8 / mse-clip@4), act+delta per-tensor
  token     W per-out-channel, act+delta one scale per (n,h,w)  [all of C, one block]
  bw-G      W per-(out-channel, G-block along Cin*kH*kW), act+delta per (n,h,w,G-block along C)
  w-G       W blockwise only, act+delta per-tensor          [attribution]
  a-G       W per-out-channel, act+delta blockwise          [attribution]

Metric: relative L2 of the sampled latent against the fp16 arm, same seed and
same noise per column, which is what every other quality doc in this tree reports.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/blockwise_e2e.py --groups 256 128 64 32
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_QUANT_ATTN"] = "0"
os.environ["MODIFF_QUANT_LINEAR"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402

preflight(*MODEL, what="blockwise_e2e.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
import act_fake_quant as A  # noqa: E402
import ahat_fake_quant_grid as G  # noqa: E402

JSON_OUT = "docs/blockwise_2026-08-31/data/blockwise_e2e.json"
SHAPE = (4, 32, 32)
WARMUP = G.WARMUP_ROUNDS       # 5
REFRESH = G.DELTA_REFRESH      # 4
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)


# ---------- channel-block helpers (NCHW in, block along C) ----------

def _group_nchw(x, gsize):
    """[N,C,H,W] -> [N,nG,G,H,W], zero-padded on C. Padding cannot change a block absmax."""
    n, c, h, w = x.shape
    if gsize is None or gsize >= c:
        return x.reshape(n, 1, c, h, w), c
    pad = (gsize - c % gsize) % gsize
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
    return x.reshape(n, x.shape[1] // gsize, gsize, h, w), c


def _ungroup_nchw(xg, c):
    n, ng, gs, h, w = xg.shape
    return xg.reshape(n, ng * gs, h, w)[:, :c]


def _scale_block(x, qmax, gsize):
    """Per-(n, block, h, w) scale. gsize None/>=C collapses to one block = per-token."""
    xg, c = _group_nchw(x, gsize)
    return qmax / xg.abs().amax(dim=2, keepdim=True).clamp_min(1e-6), c


def _apply_block(x, s, c, qmax, gsize):
    xg, _ = _group_nchw(x, gsize)
    return _ungroup_nchw((xg * s).round().clamp(-qmax, qmax) / s, c)


# ---------- the delta recursion, one hook covering every granularity ----------

class DeltaHook:
    """MoDiff delta recursion. `gsize=None` and `per_tensor=True` is the shipped path.

    per_tensor: one scalar scale for the whole tensor (shipped)
    else:       one scale per (n, C-block, h, w); gsize>=C means per-token
    """

    def __init__(self, qmax, per_tensor: bool, gsize=None, refresh: int = REFRESH):
        self.qmax = float(qmax)
        self.per_tensor = per_tensor
        self.gsize = gsize
        self.refresh = int(refresh)
        self.a_hat = None
        self.step = 0
        self._held = None
        #: clipped delta codes, counted only on held-scale steps (the warmup rounds
        #: always use a fresh scale and cannot clip). This is the mechanism a
        #: tight per-block scale is suspected of: it cannot absorb delta growth
        #: inside the refresh window.
        self.n_clip = 0
        self.n_elem = 0

    def reset(self):
        self.a_hat, self.step, self._held = None, 0, None
        self.n_clip = self.n_elem = 0

    def _q_dynamic(self, v):
        """Fresh-scale quantize, used for the t=T warmup rounds in every arm."""
        if self.per_tensor:
            s = self.qmax / v.abs().max().clamp_min(1e-6)
            return torch.clamp(torch.round(v * s), -self.qmax, self.qmax) / s
        s, c = _scale_block(v, self.qmax, self.gsize)
        return _apply_block(v, s, c, self.qmax, self.gsize)

    def _q_held(self, v):
        """Quantize with a scale refreshed every `refresh` steps (shipped cadence)."""
        if self._held is None or (self.step - 1) % self.refresh == 0:
            if self.per_tensor:
                self._held = self.qmax / v.abs().max().clamp_min(1e-6)
            else:
                self._held = _scale_block(v, self.qmax, self.gsize)[0]
        if self.per_tensor:
            raw = torch.round(v * self._held)
        else:
            vg, _c = _group_nchw(v, self.gsize)
            raw = torch.round(vg * self._held)
        self.n_clip += int((raw.abs() > self.qmax).sum())
        self.n_elem += raw.numel()
        if self.per_tensor:
            return torch.clamp(raw, -self.qmax, self.qmax) / self._held
        _, c = _group_nchw(v, self.gsize)
        return _apply_block(v, self._held, c, self.qmax, self.gsize)

    def __call__(self, mod, args):
        x = args[0].float()
        if self.a_hat is None or self.a_hat.shape != x.shape:
            a = self._q_dynamic(x)
            for _ in range(WARMUP - 1):
                a = a + self._q_dynamic(x - a)
            self.a_hat = a
            self.step = 1
            self._held = None
        else:
            self.a_hat = self.a_hat + self._q_held(x - self.a_hat)
            self.step += 1
        return (self.a_hat.to(args[0].dtype),) + args[1:]


# ---------- weight fake-quant ----------

def _wq_blocks(w, gchan):
    """[Cout,Cin,R,S] -> [Cout, nG, G*R*S]: C-ALIGNED blocks of `gchan` input channels.

    G counts CHANNELS, the same unit as the activation blocks, so weights and
    activations share the C-block boundaries. That alignment is what makes a
    channel-block split-K exact (see blockwise_cost.py) and is what a fused
    blockwise mainloop would need. gchan None or >= Cin collapses to
    per-output-channel, i.e. the shipped rule.
    """
    cout, cin = w.shape[0], w.shape[1]
    wf = w.reshape(cout, cin, -1).float()
    if gchan is None or gchan >= cin:
        return wf.reshape(cout, 1, -1), cin
    pad = (gchan - cin % gchan) % gchan
    if pad:
        wf = F.pad(wf, (0, 0, 0, pad))
    return wf.reshape(cout, wf.shape[1] // gchan, -1), cin


def quantize_weights_(convs, bits, gsize, rule):
    """In-place weight fake-quant. gsize None -> per-output-channel (shipped)."""
    q = 127.0 if bits == 8 else 7.0
    saved = {}
    for name, m in convs.items():
        w = m.weight.data
        saved[name] = w.clone()
        wf = w.reshape(w.shape[0], -1).float()
        wg, ncin = _wq_blocks(w, gsize)
        keep = ncin * w.shape[2] * w.shape[3]
        am = wg.abs().amax(dim=-1, keepdim=True)
        if rule == "mse":
            best_err = sc = None
            for r in CLIPS:
                c = (am * r / q).clamp_min(1e-8)
                e = (((wg / c).round().clamp(-q, q) * c - wg) ** 2).sum(dim=-1, keepdim=True)
                if best_err is None:
                    best_err, sc = e, c
                else:
                    msk = e < best_err
                    best_err = torch.where(msk, e, best_err)
                    sc = torch.where(msk, c, sc)
        else:
            sc = (am / q).clamp_min(1e-8)
        wq = (wg / sc).round().clamp(-q, q) * sc
        m.weight.data = wq.reshape(wf.shape[0], -1)[:, :keep].reshape_as(w).to(w.dtype)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, nargs="+", default=[20260805],
                    help="one or more seeds; relL2 is measured against that seed's own fp16 arm")
    ap.add_argument("--groups", type=int, nargs="+", default=[256, 128, 64, 32])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--refresh", type=int, nargs="+", default=[REFRESH],
                    help="delta-scale refresh cadence(s). Shipped is 4; 1 = every step.")
    ap.add_argument("--attrib", type=int, default=0,
                    help="also run W-only / A-only attribution arms at this G (0 = skip)")
    ap.add_argument("--no-token", action="store_true")
    ap.add_argument("--w-only", action="store_true",
                    help="hold activations at the shipped per-tensor scale in the bw arms, "
                         "so the sweep isolates WEIGHT granularity")
    ap.add_argument("--act-only", action="store_true",
                    help="hold weights at the shipped per-output-channel rule in the bw arms, "
                         "so the sweep isolates ACTIVATION granularity")
    ap.add_argument("--w-rule", default="absmax", choices=("absmax", "mse"),
                    help="scale rule for blockwise weight blocks (ignored with --act-only)")
    ap.add_argument("--warmup-pass", action="store_true",
                    help="run one throwaway sampling pass per arm (GPU clock parity)")
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps} "
          f"warmup={WARMUP} refresh={REFRESH} groups={a.groups}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="/tmp/claude-0/-workspace/blockwise_e2e",
        batch_size=a.n, steps=a.steps, shape=SHAPE, calibration_path=None,
        auto_delta_table=False)
    model, sampler = runner._setup_model("fp16")
    convs = A.target_convs(model.model.diffusion_model)
    print(f"  {len(convs)} FusedResBlock conv hook targets", flush=True)
    if len(convs) < 50:
        print(f"FAIL: expected ~70 live convs, got {len(convs)}")
        return 1

    # (label, bits, w_gsize, w_rule, act_per_tensor, act_gsize, refresh)
    arms = [("fp16", 0, None, None, None, None, 0)]
    for bits in a.bits:
        tag = f"W{bits}A{bits}"
        wrule = "absmax" if bits == 8 else "mse"
        for rf in a.refresh:
            rt = "" if rf == REFRESH else f" r={rf}"
            arms.append((f"{tag} shipped{rt}", bits, None, wrule, True, None, rf))
            if not a.no_token:
                arms.append((f"{tag} token act{rt}", bits, None, wrule, False, 10 ** 9, rf))
            for g in a.groups:
                wg = None if a.act_only else g
                wr = wrule if a.act_only else a.w_rule
                nm = "A-only" if a.act_only else ("W-only" if a.w_only else "bw")
                # W-only keeps the per-tensor activation scale; act_block is unused then.
                arms.append((f"{tag} {nm} G={g}{rt}", bits, wg, wr,
                             bool(a.w_only), None if a.w_only else g, rf))
            if a.attrib:
                g = a.attrib
                arms.append((f"{tag} bw G={g} W-only{rt}", bits, g, "absmax", True, None, rf))
                arms.append((f"{tag} bw G={g} A-only{rt}", bits, None, wrule, False, g, rf))

    per_seed = {}
    for seed in a.seed:
        print(f"\n=== seed {seed} ===", flush=True)
        quality, ref = {}, None
        for label, bits, wg, wrule, pt, ag, rf in arms:
            saved = quantize_weights_(convs, bits, wg, wrule) if bits else None
            hooks, handles = [], []
            if bits:
                qmax = 127.0 if bits == 8 else 7.0
                for _key, mod in convs.items():
                    h = DeltaHook(qmax=qmax, per_tensor=bool(pt), gsize=ag, refresh=rf)
                    hooks.append(h)
                    handles.append(mod.register_forward_pre_hook(h))
            if a.warmup_pass:
                G.sample_latent(model, sampler, a.steps, a.n, seed)
                for h in hooks:
                    h.reset()
            lat = G.sample_latent(model, sampler, a.steps, a.n, seed)
            for hd in handles:
                hd.remove()
            if saved is not None:
                G.restore_weights_(convs, saved)

            if ref is None:
                ref = lat.float().clone()
                rel = 0.0
            else:
                rel = float((lat.float() - ref).norm() / (ref.norm() + 1e-12))
            clip = (sum(h.n_clip for h in hooks) / max(1, sum(h.n_elem for h in hooks))
                    if hooks else 0.0)
            quality[label] = {"relL2_vs_fp16": rel, "bits": bits, "w_block": wg,
                              "w_rule": wrule, "act_per_tensor": pt, "act_block": ag,
                              "refresh": rf, "clip_frac": clip}
            print(f"  {label:26s} relL2 {rel:.4f}  clip {clip * 100:6.3f}%", flush=True)
        per_seed[str(seed)] = quality

    # aggregate: mean / min / max across seeds, so a G recommendation can be checked
    # against the spread rather than read off a single run.
    agg = {}
    for label in per_seed[str(a.seed[0])]:
        vals = [per_seed[str(s)][label]["relL2_vs_fp16"] for s in a.seed]
        base = dict(per_seed[str(a.seed[0])][label])
        base.pop("relL2_vs_fp16", None)
        agg[label] = {**base, "relL2_mean": sum(vals) / len(vals),
                      "relL2_min": min(vals), "relL2_max": max(vals),
                      "relL2_all": vals, "n_seeds": len(vals)}
    if len(a.seed) > 1:
        print(f"\n=== mean over {len(a.seed)} seeds (spread = max-min) ===", flush=True)
        for label, r in agg.items():
            print(f"  {label:26s} relL2 {r['relL2_mean']:.4f} "
                  f"+-{(r['relL2_max'] - r['relL2_min']) / 2:.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"seeds": a.seed, "steps": a.steps, "n": a.n, "groups": a.groups,
               "warmup_rounds": WARMUP, "delta_refresh": REFRESH,
               "note": "all-blockwise fake-quant; weights along Cin*kH*kW per out-channel, "
                       "act/delta along C per (n,h,w). Granularity is the ONLY thing that "
                       "varies between arms; warmup and refresh cadence held at shipped values "
                       "unless the label says r=. o_hat not quantized.",
               "agg": agg, "per_seed": per_seed}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
