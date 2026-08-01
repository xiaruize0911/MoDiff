"""Trace a real DDIM run with Nsight Systems and attribute GPU time to each layer IN CONTEXT.

Why this exists. The layer suite times each layer in isolation from a Python loop, which exposes
the module's CPU dispatch cost: at batch 128 that cost is roughly constant per call (76-240 us
depending on mode) while GPU time scales with the shape, so the small shapes starve the GPU and
their wall-clock ratio partly measures Python path length rather than kernel speed. The claim has
been that a full run hides this, because one layer's dispatch overlaps the previous layer's GPU
work -- but that was an inference from "56 layers x ~200 us << 103 ms per step", not a measurement.
This measures it: NVTX ranges per layer, a real sample under nsys, and then

  1. GPU time attributed to each layer as it actually runs, and
  2. the GPU idle gaps on the timeline -- where the device is doing nothing and for how long.

nsys is used rather than torch.profiler because only nsys gives the timeline; torch.profiler gives
aggregates, which cannot answer "is the GPU ever idle and where". nsys works in this container
while ncu does not: it traces through CUPTI's Activity API, which is not gated by the GPU
performance-counter permission that RmProfilingAdminOnly=1 withholds. CPU sampling IS gated
(perf_event_paranoid=4, no CAP_SYS_ADMIN), so it is disabled.

Run it directly; it re-executes itself under nsys:
  python3 nsys_layer_trace.py --mode int4_baseline --batch 128 --steps 20
"""
import argparse
import collections
import csv
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
INNER_ENV = "MODIFF_NSYS_INNER"


# ----------------------------------------------------------------- inner: the annotated workload
def run_inner(mode, batch, steps, only=None):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src/taming-transformers"))
    sys.path.insert(0, HERE)
    import torch
    from layer_pipeline_bench import collect_layers, ATTN_CLASSES  # noqa: F401
    from integration.fused_ops.fused_resblock import FusedResBlock

    model, sampler, layers = collect_layers(mode)
    by_id = {id(L["module"]): L for L in layers}

    handles = []

    def mk_pre(name):
        def pre(m, inp):
            torch.cuda.nvtx.range_push(name)
        return pre

    def post(m, inp, out):
        torch.cuda.nvtx.range_pop()

    unet = model.model.diffusion_model
    for nm, m in unet.named_modules():
        L = by_id.get(id(m))
        if L is None:
            continue
        if only and L["kind"] != only:
            continue
        xs = L["x_shape"]
        label = "L|%s|C%d|T%d" % (L["kind"], xs[1], xs[2] * xs[3])
        handles.append(m.register_forward_pre_hook(mk_pre(label)))
        handles.append(m.register_forward_hook(post))

    r_batch = batch
    cond = None
    import integration.benchmarks.benchmark_ldm as B  # noqa: F401
    # collect_layers already built the runner's model; re-derive the conditioning the same way
    from layer_pipeline_bench import B as _B  # noqa: N811
    runner = _B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/final_report_2026-07-28/tmp_out", batch_size=r_batch, steps=steps,
        shape=(4, 32, 32), calibration_path=None, linear_backend="fp16")
    cond = runner._cond_kwargs(model, r_batch)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        # warm up outside the region of interest, then profile the steady state
        sampler.sample(S=steps, batch_size=r_batch, shape=(4, 32, 32), eta=0.0,
                       verbose=False, **cond)
        torch.cuda.synchronize()
        torch.cuda.profiler.cudart().cudaProfilerStart()
        sampler.sample(S=steps, batch_size=r_batch, shape=(4, 32, 32), eta=0.0,
                       verbose=False, **cond)
        torch.cuda.synchronize()
        torch.cuda.profiler.cudart().cudaProfilerStop()
    for h in handles:
        h.remove()
    print("inner done: %s batch=%d steps=%d" % (mode, batch, steps))


# ----------------------------------------------------------------- outer: drive nsys and analyse
def find_nsys():
    p = subprocess.run(["bash", os.path.join(HERE, "setup_nsys.sh")],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("setup_nsys.sh failed:\n" + p.stderr[-2000:])
    return p.stdout.strip().splitlines()[-1]


def nsys_stats(nsys, rep, report):
    # --force-export is not optional: nsys reuses a stale .sqlite next to the report if one
    # exists and silently returns an empty table, which is how the first run of this script
    # concluded there were no NVTX ranges when the report in fact held 1120 of them.
    p = subprocess.run([nsys, "stats", "--force-export=true", "--report", report,
                        "--format", "csv", rep],
                       capture_output=True, text=True, cwd=os.path.dirname(rep) or ".")
    if p.returncode != 0:
        raise SystemExit("nsys stats %s failed:\n%s" % (report, p.stderr[-2000:]))
    # the CSV is preceded by progress lines; the table starts at the first line with commas
    lines = p.stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if l.count(",") >= 3), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def analyse(nsys, rep, steps):
    print("\n" + "=" * 92)
    print("GPU time attributed to each layer, as it runs inside a real sample")
    print("=" * 92)
    rows = nsys_stats(nsys, rep, "nvtx_gpu_proj_sum")
    key = lambda r: next((k for k in r if "Range" in k or "Name" in k), None)  # noqa: E731
    lay = []
    for r in rows:
        kk = key(r)
        nm = (r.get(kk) or "").strip().lstrip(":")   # nsys prefixes range names with ':'
        if not nm.startswith("L|"):
            continue
        # nvtx_gpu_proj_sum's header is
        #   Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,...
        # "Total Proj Time" is the GPU time projected onto the range; "Total Range Time" is the
        # CPU-side wall of the range and is NOT what we want. Matching loosely on "Total Time"
        # hit neither and silently produced zeros.
        tot = next((num(r[c]) for c in r if "Total Proj Time" in c), 0.0)
        wall = next((num(r[c]) for c in r if "Total Range Time" in c), 0.0)
        inst = next((num(r[c]) for c in r if "Range Instances" in c), 0.0)
        lay.append((tot, nm, inst, wall))
    lay.sort(reverse=True)
    if lay:
        total = sum(t for t, _, _, _ in lay)
        # The NVTX range is CPU-side: it closes when the CPU has finished ISSUING the layer's
        # kernels, not when the GPU has run them. So range time is the issue cost and the ratio
        # GPU/issue is how far the CPU runs ahead -- above 1 the layer feeds the queue, below 1
        # it drains it. Calling that column "busy" (as a first version did) is wrong; busy_frac
        # in the layer suite is a different quantity measured a different way.
        print("| layer | GPU ms | share | calls | GPU µs/call | CPU issue µs/call | GPU/issue |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for t, nm, inst, wall in lay:
            _, kind, c, T = nm.split("|")
            g = (t / inst / 1e3) if inst else 0.0
            w = (wall / inst / 1e3) if inst else 0.0
            print("| %-16s %-6s %-6s | %.2f | %.1f%% | %d | %.1f | %.1f | %.3f |"
                  % (kind, c, T, t / 1e6, t / total * 100 if total else 0, inst, g, w,
                     (g / w) if w else 0))
        print("| **all annotated layers** | **%.2f** | 100%% | | | | |" % (total / 1e6))
    else:
        print("no NVTX layer ranges found in the report")

    print("\n" + "=" * 92)
    print("GPU idle on the timeline (is the device ever waiting for the CPU?)")
    print("=" * 92)
    tr = nsys_stats(nsys, rep, "cuda_gpu_trace")
    spans = []
    for r in tr:
        s = next((num(r[c]) for c in r if c.startswith("Start")), None)
        d = next((num(r[c]) for c in r if c.startswith("Duration")), None)
        if s is None or d is None:
            continue
        spans.append((s, s + d))
    if not spans:
        print("no GPU spans in the report")
        return
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    wall = merged[-1][1] - merged[0][0]
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    gaps.sort(reverse=True)
    idle = sum(gaps)
    print("kernels traced        %d" % len(spans))
    print("timeline span         %.2f ms" % (wall / 1e6))
    print("GPU busy              %.2f ms  (%.1f%%)" % (busy / 1e6, busy / wall * 100))
    print("GPU idle              %.2f ms  (%.1f%%) in %d gaps"
          % (idle / 1e6, idle / wall * 100, len(gaps)))
    if gaps:
        print("gap sizes: max %.1f µs, p99 %.1f µs, median %.2f µs"
              % (gaps[0] / 1e3, gaps[max(0, int(0.01 * len(gaps)))] / 1e3,
                 gaps[len(gaps) // 2] / 1e3))
        big = [g for g in gaps if g > 50_000]
        print("gaps > 50 µs: %d, totalling %.2f ms (%.1f%% of the span)"
              % (len(big), sum(big) / 1e6, sum(big) / wall * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int4_baseline")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20, help="must divide 1000")
    ap.add_argument("--only", default=None,
                    help="annotate only this layer kind (e.g. attention)")
    ap.add_argument("--run", type=int, default=None, help="run index, used in the output name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", metavar="REPORT",
                    help="skip tracing and analyse an existing .nsys-rep")
    a = ap.parse_args()

    if a.analyze:
        analyse(find_nsys(), os.path.abspath(a.analyze), a.steps)
        return

    if os.environ.get(INNER_ENV):
        run_inner(a.mode, a.batch, a.steps, a.only)
        return

    nsys = find_nsys()
    out = a.out or os.path.join(
        ROOT, "docs/final_report_2026-07-28/data/nsys",
        "nsys_%s_%s_b%d%s" % (a.only or "all", a.mode, a.batch,
                              "" if a.run is None else "_run%d" % a.run))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ, **{INNER_ENV: "1", "LBENCH_BATCH": str(a.batch)})
    cmd = [nsys, "profile", "--trace=cuda,nvtx",
           "--sample=none", "--cpuctxsw=none",          # perf_event is not permitted here
           "--capture-range=cudaProfilerApi",           # skip the warm-up sample
           "--capture-range-end=stop",
           "--force-overwrite=true", "-o", out,
           sys.executable, os.path.abspath(__file__),
           "--mode", a.mode, "--batch", str(a.batch), "--steps", str(a.steps)]
    if a.only:
        cmd += ["--only", a.only]
    print("running: %s" % " ".join(cmd[:8]))
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    print("\n".join(tail))
    rep = out + ".nsys-rep"
    if not os.path.exists(rep):
        raise SystemExit("no report produced")
    analyse(nsys, rep, a.steps)
    print("\nreport: %s" % rep)


if __name__ == "__main__":
    main()
