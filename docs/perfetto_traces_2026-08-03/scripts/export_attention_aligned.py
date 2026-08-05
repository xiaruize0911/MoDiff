"""One Perfetto trace holding every attention-layer profile, time-aligned across FP16/INT8/INT4.

Problem with separate trace files: to compare what the attention layer does in three modes you have
to open three tabs and eyeball independent timelines. Chrome Trace Event JSON has no notion of
"these three recordings belong together", but it does have processes, and timestamps are just
numbers -- so they can be merged and shifted.

What this produces
------------------
One `.json`. Six process tracks: `FP16 CPU`, `FP16 GPU`, `INT8 CPU`, `INT8 GPU`, `INT4 CPU`,
`INT4 GPU`, plus a `shape` ruler track on top. The timeline is divided into one slot per attention
shape (T=1024, 256, 64, 16, 4), and inside a slot **all three modes start at exactly the same
timestamp**, so the three GPU rows under one ruler slice are the same work in three precisions and
can be read straight down the page.

How the alignment works
-----------------------
Each (mode, shape) is profiled in its OWN profiler session, so every event in it belongs
unambiguously to that one attention forward -- that is what makes a per-slot time shift safe. Per
session: anchor = the `attn` `record_function` slice start on the CPU track; span = anchor to the
last GPU event end. Slot width = max span over the three modes, plus padding. Every event in the
session (including the `ac2g` flow arrows that link launch to kernel) is shifted by
`slot_start - anchor`. Flow ids are namespaced per session so arrows cannot cross-link between modes.

One forward per (mode, shape), after 12+ warmups so the quantized blocks' static scales have frozen
and the trace shows the production route. One forward, not many: this is a structural comparison --
which kernels fire, in what order, at what relative cost. For statistics use
docs/MEASUREMENT_REPORT_2026-08-01.md, which times without a profiler attached.
"""

import argparse
import collections
import copy
import gzip
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import layer_pipeline_bench as lb

OUT = "docs/perfetto_traces_2026-08-03/traces"
MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int4_baseline", "INT4")]
ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
WARM_FORWARDS = 12
#: Iterations run inside the profiler but before the annotation opens, so the profiler's
#: own first-call overhead does not land in the measured window.
PROF_WARM = 3
ANNOTATION = "attn"
#: Blank time between slots, as a fraction of the widest span in the slot.
SLOT_PAD = 0.18
#: Pseudo-processes torch adds for its own viewer; they carry one event each and would collide.
DROP_PIDS = ("Spans", "Traces", "")


def profile_attention(mode_key, iters):
    """{x_shape: (chrome_trace_dict, n_instances)} for every attention shape in `mode_key`."""
    model, sampler, layers = lb.collect_layers(mode_key)
    del sampler
    groups = collections.OrderedDict()
    for L in layers:
        if L["kind"] == "attention":
            groups.setdefault(tuple(L["x_shape"]), []).append(L)

    out = {}
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for shape, insts in groups.items():
            m = insts[0]["module"]
            x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous(
                memory_format=torch.channels_last)
            for _ in range(WARM_FORWARDS):
                m(x)
            torch.cuda.synchronize()
            with profile(activities=ACTIVITIES, record_shapes=True) as prof:
                # Absorb the profiler's own first-call cost INSIDE the session but OUTSIDE the
                # annotation. CUPTI instrumentation and per-op symbol resolution make the first
                # profiled iteration wildly unrepresentative on the CPU side -- measured 2.8 ms of
                # CPU before the first kernel even launched at T=64, against 411 us of actual GPU
                # work. Those iterations are then dropped by session_events().
                for _ in range(PROF_WARM):
                    m(x)
                torch.cuda.synchronize()
                with record_function(ANNOTATION):
                    for _ in range(iters):
                        m(x)
                torch.cuda.synchronize()
            path = os.path.join(OUT, f".tmp_{mode_key}_{shape[1]}_{shape[2]}.json")
            prof.export_chrome_trace(path)
            with open(path) as f:
                out[shape] = (json.load(f), len(insts))
            os.remove(path)
            print(f"    {shape} x{len(insts)}: "
                  f"{sum(1 for e in out[shape][0]['traceEvents'] if e.get('cat') == 'kernel')} "
                  f"kernel events")
            del x
            torch.cuda.empty_cache()
    del model, layers
    torch.cuda.empty_cache()
    return out


#: Harness artifacts, not layer work: the explicit torch.cuda.synchronize() after the annotation
#: closes, plus profiler teardown. Left in, these dominate the span (5.6 ms against 3.0 ms of
#: actual kernels at T=1024) and every slot would be mostly dead space.
SYNC_NAMES = ("cudaDeviceSynchronize", "cudaStreamSynchronize")


def session_events(trace):
    """(anchor_ts, end_ts, events) with the post-annotation sync tail removed.

    anchor is the annotation start on the CPU track -- the alignment point. end is the last
    remaining event end, which is what the slot has to be wide enough to hold.
    """
    ann_start = ann_end = None
    for e in trace["traceEvents"]:
        if (e.get("ph") == "X" and e.get("cat") == "user_annotation"
                and e.get("name") == ANNOTATION and e.get("ts") is not None):
            stop = e["ts"] + e.get("dur", 0)
            ann_start = e["ts"] if ann_start is None else min(ann_start, e["ts"])
            ann_end = stop if ann_end is None else max(ann_end, stop)

    keep = []
    for e in trace["traceEvents"]:
        ts = e.get("ts")
        # Drop the profiler-warmup iterations that ran before the annotation opened, and the
        # explicit synchronize() that closes the session after it. What is left is the annotation
        # window plus the GPU tail that drains inside it.
        if ann_start is not None and ts is not None and ts < ann_start - 1e-6:
            continue
        if (ann_end is not None and e.get("cat") == "cuda_runtime"
                and e.get("name") in SYNC_NAMES and (ts or 0) >= ann_end):
            continue
        keep.append(e)

    if ann_start is None:                  # no annotation: fall back to the first event
        ann_start = min((e["ts"] for e in keep
                         if e.get("ph") == "X" and e.get("ts") is not None), default=0.0)
    end = max((e["ts"] + e.get("dur", 0) for e in keep
               if e.get("ph") == "X" and e.get("ts") is not None), default=ann_start)
    return ann_start, end, keep


def merge(sessions, shapes, out_path):
    """sessions: {(mode_key, shape): (trace, n_inst)}. Writes one aligned trace."""
    # Slot layout: width per shape is the widest span across modes, so no mode overflows its slot.
    slots, cursor = {}, 0.0
    for shape in shapes:
        spans = []
        for mode_key, _ in MODES:
            tr = sessions.get((mode_key, shape))
            if tr is None:
                continue
            a, e, _ = session_events(tr[0])
            spans.append(e - a)
        width = max(spans) if spans else 1.0
        slots[shape] = (cursor, width)
        cursor += width * (1.0 + SLOT_PAD)

    events, meta_done = [], set()
    template = None

    for m, (mode_key, label) in enumerate(MODES):
        cpu_pid, gpu_pid = 10 + m * 2, 11 + m * 2
        for s, shape in enumerate(shapes):
            entry = sessions.get((mode_key, shape))
            if entry is None:
                continue
            trace, n_inst = entry
            if template is None:
                template = trace
            anchor, _, kept = session_events(trace)
            slot_start, _ = slots[shape]
            shift = slot_start - anchor
            # Namespace the flow ids: without this, an arrow emitted by FP16 could be paired with
            # one from INT4 and Perfetto would draw a launch->kernel link between two modes.
            id_base = (m * len(shapes) + s + 1) * 10 ** 8

            for e in kept:
                pid = e.get("pid")
                if pid in DROP_PIDS:
                    continue
                if e.get("ph") == "M":
                    continue                      # regenerated below, once per track
                new = copy.copy(e)
                # GPU pids are device indices (0..7); everything else is the python process.
                new["pid"] = gpu_pid if isinstance(pid, int) and pid < 10 else cpu_pid
                if new.get("ts") is not None:
                    new["ts"] = e["ts"] + shift
                if "id" in new and isinstance(new["id"], int):
                    new["id"] = new["id"] + id_base
                if "id2" in new:
                    new.pop("id2")
                # Make the annotation slice say which shape and mode it is, so a slice tooltip is
                # self-describing once six tracks are stacked.
                if e.get("name") == ANNOTATION and e.get("cat") in (
                        "user_annotation", "gpu_user_annotation"):
                    new["name"] = (f"{label} attn C{shape[1]} {shape[2]}x{shape[3]} "
                                   f"T={shape[2] * shape[3]} x{n_inst}")
                events.append(new)

            for pid, name in ((cpu_pid, f"{label} CPU"), (gpu_pid, f"{label} GPU")):
                if pid in meta_done:
                    continue
                meta_done.add(pid)
                events += [
                    {"name": "process_name", "ph": "M", "ts": 0, "pid": pid, "tid": 0,
                     "args": {"name": name}},
                    {"name": "process_labels", "ph": "M", "ts": 0, "pid": pid, "tid": 0,
                     "args": {"labels": name}},
                    {"name": "process_sort_index", "ph": "M", "ts": 0, "pid": pid, "tid": 0,
                     "args": {"sort_index": pid}},
                ]

    # Ruler track: one slice per shape spanning its slot, so the slot boundaries are visible and
    # you can see at a glance that the three modes below start together.
    events += [
        {"name": "process_name", "ph": "M", "ts": 0, "pid": 1, "tid": 0,
         "args": {"name": "shape"}},
        {"name": "process_sort_index", "ph": "M", "ts": 0, "pid": 1, "tid": 0,
         "args": {"sort_index": 0}},
        {"name": "thread_name", "ph": "M", "ts": 0, "pid": 1, "tid": 1,
         "args": {"name": "attention shape"}},
    ]
    for shape in shapes:
        start, width = slots[shape]
        events.append({"ph": "X", "cat": "shape", "pid": 1, "tid": 1, "ts": start, "dur": width,
                       "name": f"C{shape[1]} {shape[2]}x{shape[3]} T={shape[2] * shape[3]}",
                       "args": {"batch": shape[0], "channels": shape[1],
                                "tokens": shape[2] * shape[3],
                                "slot_us": round(width, 1)}})

    merged = {k: v for k, v in (template or {}).items() if k != "traceEvents"}
    merged["traceEvents"] = events
    merged["traceName"] = "MoDiff attention layer, FP16/INT8/INT4 aligned per shape"
    merged["displayTimeUnit"] = "ns"
    with open(out_path, "w") as f:
        json.dump(merged, f)
    with open(out_path, "rb") as f_in, gzip.open(out_path + ".gz", "wb", 9) as f_out:
        f_out.writelines(f_in)
    return slots, len(events)


def summarize(out_path, shapes):
    """Per (shape, mode): kernel count and summed GPU kernel time, read back from the merged file."""
    ev = json.load(open(out_path))["traceEvents"]
    pid_name = {e["pid"]: e["args"]["name"] for e in ev
                if e.get("ph") == "M" and e["name"] == "process_name"}
    slot = {}
    for e in ev:
        if e.get("cat") == "shape":
            slot[e["name"]] = (e["ts"], e["ts"] + e["dur"])
    rows = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for e in ev:
        if e.get("cat") != "kernel":
            continue
        name = pid_name.get(e["pid"], "")
        for sname, (a, b) in slot.items():
            if a <= e["ts"] < b:
                cell = rows[sname][name.split()[0]]
                cell[0] += 1
                cell[1] += e.get("dur", 0)
                break
    print("\n  shape                         " + "".join(f"{lb_:>22}" for _, lb_ in MODES))
    for shape in shapes:
        sname = f"C{shape[1]} {shape[2]}x{shape[3]} T={shape[2] * shape[3]}"
        line = f"  {sname:<28}"
        for _, label in MODES:
            n, us = rows[sname][label]
            line += f"{n:>6} k {us:>10.1f} us"
        print(line)
    return {s: {m: rows[f'C{s[1]} {s[2]}x{s[3]} T={s[2]*s[3]}'][lbl]
                for _, lbl in MODES for m in [lbl]} for s in shapes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1,
                    help="forwards inside the profiled region; 1 keeps each slot unambiguous")
    ap.add_argument("--out", default=os.path.join(OUT, "attention_aligned_fp16_int8_int4.json"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    sessions, shapes = {}, None
    for mode_key, label in MODES:
        print(f"\n=== {label} ({mode_key}) ===", flush=True)
        per_shape = profile_attention(mode_key, a.iters)
        if shapes is None:
            shapes = sorted(per_shape, key=lambda s: -(s[2] * s[3]))
        for shape, v in per_shape.items():
            sessions[(mode_key, shape)] = v

    slots, n_events = merge(sessions, shapes, a.out)
    print(f"\nwrote {a.out}")
    print(f"  {n_events} events, {os.path.getsize(a.out) / 2**20:.2f} MiB "
          f"({os.path.getsize(a.out + '.gz') / 2**20:.2f} MiB gz)")
    print("  slots (us):", {f"T={s[2]*s[3]}": (round(v[0], 1), round(v[1], 1))
                            for s, v in slots.items()})
    stats = summarize(a.out, shapes)
    with open(os.path.join(OUT, "attention_aligned_summary.json"), "w") as f:
        json.dump({f"C{s[1]}_T{s[2]*s[3]}": v for s, v in stats.items()}, f, indent=2)


if __name__ == "__main__":
    main()
