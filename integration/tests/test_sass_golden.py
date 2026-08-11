"""Did the compiled device code change? Per-kernel SASS hashes, for gating pure refactors.

The csrc/ split into baseline/ and modiff/ trees claims to move code without changing it. This is how
that claim is checked: `cuobjdump --dump-sass` the built extension, hash each kernel's instruction
stream, and diff against a committed list. A pure move leaves all 289 hashes identical. Anything that
recompiles differently -- a different template instantiation, a lost compile flag, a header resolved to
the wrong copy -- moves a hash and names the kernel.

WHY THIS RATHER THAN CALLING THE KERNELS. The alternative gate was a golden output per export: fixed
seeded inputs, compare tensors. That needs a bespoke argument recipe for each of 130 exports, and it
would still miss a kernel nobody wrote a recipe for. This needs no recipes, covers every kernel in the
binary including the ones with no test, and is deterministic in a way the end-to-end gate is not
(see the note in test_export_manifest.py: e2e_output_check false-fails about one run in three).

WHAT IT CANNOT SEE: host-side changes. A wrong argument, a swapped scale, a dropped launch, an export
registered against the wrong function -- all leave the SASS untouched. Pair it with
test_export_manifest.py (which exports exist) and the kernel-level tests (what the host wrapper does).

A duplicate mangled name is meaningful here, not noise: two translation units compiling the same
non-template `__global__` from a duplicated header would collide at link time, so the count per name is
part of the record.

  python integration/tests/test_sass_golden.py                    # gate
  UPDATE_SASS=1 python integration/tests/test_sass_golden.py      # after an intended change
"""
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)

SO = os.path.join(ROOT, "modiff_cutlass.cpython-311-x86_64-linux-gnu.so")
GOLDEN = os.path.join(HERE, "golden", "sass_kernels.txt")
UPDATE = os.environ.get("UPDATE_SASS", "0") == "1"


def kernel_hashes(so_path):
    """{mangled name: (hash, count)}. Absolute paths: cuobjdump is run from an arbitrary cwd."""
    out = subprocess.run(["cuobjdump", "--dump-sass", so_path], capture_output=True, text=True)
    if out.returncode != 0 or "Function : " not in out.stdout:
        raise SystemExit(f"cuobjdump produced no SASS for {so_path}\n{out.stderr[:400]}")
    hashes, counts, name, acc = {}, {}, None, []

    def flush():
        if name is None:
            return
        h = hashlib.sha256("\n".join(acc).encode()).hexdigest()[:16]
        counts[name] = counts.get(name, 0) + 1
        # Same name compiled into two TUs must hash the same, or the two copies have diverged.
        if name in hashes and hashes[name] != h:
            hashes[name] = "DIVERGENT"
        else:
            hashes.setdefault(name, h)

    for line in out.stdout.splitlines():
        m = re.match(r"\s*Function : (.+)$", line)
        if m:
            flush()
            name, acc = m.group(1).strip(), []
            continue
        if name is None:
            continue
        # Strip the /*addr*/ and /*hex*/ comment columns: they encode position, not semantics, and
        # shift when an unrelated kernel earlier in the same TU changes size.
        body = re.sub(r"/\*[0-9a-fx]*\*/", "", line).strip()
        # Also normalise long hex literals. SASS prints BRANCH TARGETS as absolute addresses in the
        # operand text, not in the comment columns, so they move when the fatbin is laid out
        # differently -- which happens when the object ORDER changes even if no code did. Family 1
        # exposed this: moving two quantize files flagged fp16_ncw_to_fp32_cl_kernel, which lives in
        # layout_transform.cu, a translation unit that was neither edited nor recompiled. Four or more
        # hex digits is an address; short immediates stay in the hash. The cost is that a genuine
        # change to a >=4-digit immediate constant would be masked.
        body = re.sub(r"0x[0-9a-f]{4,}", "0xADDR", body)
        if body:
            acc.append(body)
    flush()
    return {n: (h, counts[n]) for n, h in hashes.items()}


def load():
    if not os.path.exists(GOLDEN):
        return None
    rec = {}
    for ln in open(GOLDEN):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        h, c, n = ln.split(None, 2)
        rec[n] = (h, int(c))
    return rec


def save(rec):
    with open(GOLDEN, "w") as f:
        f.write(f"# per-kernel SASS hashes, {len(rec)} kernels. sha256[:16]  count  mangled_name\n")
        for n in sorted(rec):
            h, c = rec[n]
            f.write(f"{h} {c} {n}\n")


def main():
    now = kernel_hashes(SO)
    div = [n for n, (h, _) in now.items() if h == "DIVERGENT"]
    want = load()

    if want is None:
        if not UPDATE:
            print(f"no golden at {GOLDEN}; create it with UPDATE_SASS=1")
            return 1
        save(now)
        print(f"wrote {GOLDEN} with {len(now)} kernels")
        return 0

    added = sorted(set(now) - set(want))
    removed = sorted(set(want) - set(now))
    changed = sorted(n for n in set(now) & set(want) if now[n][0] != want[n][0])
    recount = sorted(n for n in set(now) & set(want) if now[n][1] != want[n][1])

    print(f"golden {len(want)} kernels, binary {len(now)}")
    for n in added:
        print(f"  + {n}")
    for n in removed:
        print(f"  - {n}")
    for n in changed:
        print(f"  ~ {n}   {want[n][0]} -> {now[n][0]}")
    for n in recount:
        print(f"  # {n}   count {want[n][1]} -> {now[n][1]}")
    # Divergence is only a failure if it is NEW. One case is pre-existing and recorded in the golden:
    # conv_epilogue.cuh's 10 template kernels are instantiated by both conv2d_int8.cu and
    # conv2d_int4.cu (weak symbols, link-deduped), and 9 of the 10 hash identically while
    # bias_residual_store_half_from_half_kernel<__half> does not. That predates the csrc/ split and is
    # captured as the literal value DIVERGENT, so the gate holds it steady instead of chasing it.
    div_new = [n for n in div if want.get(n, ("", 0))[0] != "DIVERGENT"]
    for n in div_new:
        print(f"  !! {n} NEWLY compiles to different code in two translation units")
    for n in div:
        if n not in div_new:
            print(f"  (known-divergent, unchanged: {n})")

    bad = added or removed or changed or recount or div_new
    if not bad:
        print("\nPASS -- every kernel's device code is unchanged")
        return 0
    if UPDATE:
        save(now)
        print(f"\nUPDATED {GOLDEN}. Check the list above: for a pure code move it should be EMPTY, "
              f"and any '~' means that kernel now compiles differently.")
        return 0
    print("\nFAILED -- device code moved. For a pure refactor this list should be empty.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
