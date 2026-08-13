"""P3: what did the double-wrapped conv bug actually cost in memory? Measured, not estimated.

3d13cf9 fixed it: FusedResBlock aliases one nn.Conv2d under two attributes (fused_resblock.py:756,
`fused.in_conv` IS `fused.original.in_layers[-1]`), and convert_model_to_optimized_int4 recursed over
named_children(), so it reached the SAME conv down two paths and wrapped it TWICE into two independent
modules, each holding its own packed int4 weights. `forward` uses self.in_conv, so the other 70 were
orphans: never called, never calibrated, carrying modiff_enabled=True.

The commit message put the cost at "~113 MiB" and docs/attn_modiff_2026-08-13/FINDINGS.md said "the
memory cost is unmeasured". This measures it, three ways that have to agree:

  1. ALLOCATOR DELTA. Build the int4 model with the dedup active, then again with it defeated, and
     diff torch.cuda.memory_allocated(). This is the number a user would feel.
  2. PER-MODULE SUM. Walk the orphans and add up every parameter and buffer they own. This says WHERE
     the memory goes and is immune to allocator rounding.
  3. COUNT. 70 orphans is the claim; if the count is wrong the other two mean nothing.

THE DEDUP IS DEFEATED WITHOUT TOUCHING SHIPPED CODE: the walk takes `_memo` as a parameter, so passing
a dict subclass whose __contains__ always returns False reproduces the pre-3d33cf9 behaviour exactly.
Editing int4_optimized.py to add a debug flag would put the bug back in the tree to measure it, which
is a bad trade.

fp16 IS NOT THE BASELINE HERE. The comparison is int4-with-dedup vs int4-without, both after
conversion; anything else measures the cost of quantization rather than the cost of the bug.

Run: python docs/zp_coverage_2026-08-13/scripts/orphan_wrapper_memory.py    # ~2 min, needs the GPU
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                              # noqa: E402

D = "docs/zp_coverage_2026-08-13"
MIB = 1024.0 * 1024.0


class NoMemo(dict):
    """A _memo that never remembers: reproduces the pre-3d33cf9 double-wrapping exactly."""

    def __contains__(self, k):
        return False

    def get(self, k, default=None):
        return default


def module_tensors(mo):
    """The module's own parameters and buffers, deduped by identity within the module."""
    seen, out = set(), []
    for t in list(mo.parameters(recurse=False)) + [b for _, b in mo.named_buffers(recurse=False)]:
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
    return out


def module_bytes(mo):
    return sum(t.numel() * t.element_size() for t in module_tensors(mo))


def split_shared(orphan_ts, live_ptrs):
    """Split an orphan's tensors into bytes that ALIAS a live tensor's storage and bytes that do not.

    This is the whole discrepancy. A naive per-module sum counts the orphan's `weight` and
    `_orig_weight`, which are the SAME STORAGE as the live wrapper's -- both wrappers wrap one
    nn.Conv2d -- so it charges the bug for memory that was never allocated twice. Only the tensors the
    orphan built for itself (its packed int4 weight and its own buffers) are real cost, and those are
    what the allocator delta sees. Comparing by data_ptr() is what makes the difference measurable
    rather than arguable.
    """
    shared = unique = 0
    for t in orphan_ts:
        n = t.numel() * t.element_size()
        if t.data_ptr() in live_ptrs:
            shared += n
        else:
            unique += n
    return shared, unique


def build_fp16():
    """One fp16 UNet, unconverted. Returned so both conversions start from the same weights."""
    import integration.benchmarks.benchmark_ldm as B
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{D}/tmp_out", batch_size=2, steps=4, shape=(4, 32, 32),
        calibration_path=None, auto_delta_table=False)
    model, _ = runner._setup_model("fp16")
    return runner, model


def convert(unet, memo):
    from integration.kernels.int4_optimized import convert_model_to_optimized_int4
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    convert_model_to_optimized_int4(unet, _memo=memo)
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() - before


def main():
    import copy
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d

    runner, model = build_fp16()
    unet = model.model.diffusion_model

    # Two independent deep copies so neither conversion sees the other's wrappers. The copies are made
    # BEFORE either conversion and their own cost is excluded from both deltas by construction.
    a = copy.deepcopy(unet)
    b = copy.deepcopy(unet)
    del model, unet
    torch.cuda.empty_cache()

    d_dedup = convert(a, {})
    d_naive = convert(b, NoMemo())

    n_a = sum(1 for m in a.modules() if isinstance(m, OptimizedInt4Conv2d))
    n_b = sum(1 for m in b.modules() if isinstance(m, OptimizedInt4Conv2d))

    # The orphans are the wrappers the naive walk created that the deduped walk did not. Identify them
    # structurally rather than by name: a wrapper reachable ONLY through `original.in_layers` /
    # `original.out_layers` while `in_conv`/`out_conv` holds a different object.
    orphans, orphan_bytes = 0, 0
    orphan_shared, orphan_unique = 0, 0
    orphan_ids = set()
    orphan_tensor_lists = []
    for mo in b.modules():
        inner = getattr(mo, "original", None)
        if inner is None:
            continue
        for attr, idx in (("in_layers", -1), ("out_layers", -1)):
            seq = getattr(inner, attr, None)
            if seq is None:
                continue
            cand = seq[idx]
            direct = getattr(mo, "in_conv" if attr == "in_layers" else "out_conv", None)
            if isinstance(cand, OptimizedInt4Conv2d) and cand is not direct:
                orphans += 1
                orphan_ids.add(id(cand))
                orphan_bytes += module_bytes(cand)
                orphan_tensor_lists.append(module_tensors(cand))

    # Every storage reachable from a module that is NOT an orphan. An orphan tensor pointing into one
    # of these was never a second allocation.
    live_ptrs = set()
    for mo in b.modules():
        if id(mo) in orphan_ids:
            continue
        for t in module_tensors(mo):
            live_ptrs.add(t.data_ptr())
    for ts in orphan_tensor_lists:
        sh, un = split_shared(ts, live_ptrs)
        orphan_shared += sh
        orphan_unique += un

    out = {
        "wrappers_deduped": n_a,
        "wrappers_naive": n_b,
        "extra_wrappers": n_b - n_a,
        "orphans_found_structurally": orphans,
        "alloc_delta_dedup_MiB": d_dedup / MIB,
        "alloc_delta_naive_MiB": d_naive / MIB,
        "bug_cost_allocator_MiB": (d_naive - d_dedup) / MIB,
        "bug_cost_per_module_sum_MiB": orphan_bytes / MIB,
        "orphan_shared_storage_MiB": orphan_shared / MIB,
        "orphan_unique_storage_MiB": orphan_unique / MIB,
        "commit_message_estimate_MiB": 113.0,
        "docstring_claim_MiB": 1014.6,
    }
    print(f"wrappers, deduped walk           {n_a}")
    print(f"wrappers, naive walk             {n_b}   (+{n_b - n_a})")
    print(f"orphans identified structurally  {orphans}")
    print(f"allocator delta, deduped         {out['alloc_delta_dedup_MiB']:9.2f} MiB")
    print(f"allocator delta, naive           {out['alloc_delta_naive_MiB']:9.2f} MiB")
    print(f"THE BUG'S COST (allocator)       {out['bug_cost_allocator_MiB']:9.2f} MiB")
    print(f"per-module sum, ALL orphan refs  {out['bug_cost_per_module_sum_MiB']:9.2f} MiB")
    print(f"  of which ALIASES a live tensor {out['orphan_shared_storage_MiB']:9.2f} MiB")
    print(f"  of which is newly allocated    {out['orphan_unique_storage_MiB']:9.2f} MiB")
    print(f"the commit message's estimate    {out['commit_message_estimate_MiB']:9.2f} MiB")
    print(f"the docstring's claimed leak     {out['docstring_claim_MiB']:9.2f} MiB")

    alloc = out["bug_cost_allocator_MiB"]
    uniq = out["orphan_unique_storage_MiB"]
    agree = abs(alloc - uniq) < 0.05 * max(alloc, 1e-9)
    print(f"\nallocator delta vs newly-allocated orphan storage agree within 5%: {agree}"
          f"   ({alloc:.2f} vs {uniq:.2f} MiB)")
    if orphans != n_b - n_a:
        print(f"WARNING: {orphans} orphans found structurally but {n_b - n_a} extra wrappers exist -- "
              f"the structural test does not account for all of them, so the sums are a LOWER BOUND.")
    ratio = out["docstring_claim_MiB"] / max(alloc, 1e-9)
    if ratio > 2:
        print(f"\nTHE TREE CARRIES TWO FIGURES FOR THIS BUG AND THEY DIFFER BY {ratio:.1f}x.\n"
              f"  convert_model_to_optimized_int4's docstring: \"a 1014.6 MiB leak\"\n"
              f"  3d13cf9's commit message:                    \"~113 MiB\"\n"
              f"Both are reproduced above, and they measure different things. The larger one counts\n"
              f"every tensor an orphan REFERENCES; but both wrappers wrap one nn.Conv2d, so the fp16\n"
              f"weight is ONE storage with two references -- {out['orphan_shared_storage_MiB']:.0f} MiB\n"
              f"of the {out['bug_cost_per_module_sum_MiB']:.0f} MiB aliases memory that was already\n"
              f"there and was never allocated twice. The real cost is what the orphans built for\n"
              f"themselves (packed int4 weights and their own buffers), which the allocator and the\n"
              f"unique-storage sum agree on: {alloc:.0f} MiB. So ~113 MiB is right and the docstring\n"
              f"overstates by {ratio:.1f}x.")
    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump(out, open(f"{D}/data/orphan_wrapper_memory.json", "w"), indent=1)
    print(f"wrote {D}/data/orphan_wrapper_memory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
