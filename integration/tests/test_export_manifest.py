"""Does `modiff_cutlass` still export exactly the kernels it is supposed to?

This exists for one failure mode, and it is the characteristic failure of the csrc/ split into
baseline/ and modiff/ trees: **leaving a host function out of the new .cu compiles cleanly.** The
kernel simply vanishes from the module, nothing warns, and the first symptom is an AttributeError in
whichever script calls it next -- possibly weeks later, in a report script nobody has run since July.

`test_kernel_correctness.py` cannot catch it: its goldens exercise MODULES (conv, linear, GroupNorm),
which reach a handful of the 130 exports. Its goldens are also `.pt` files, and `.gitignore:6` ignores
`*.pt` -- `git ls-files integration/tests/golden/` returns only README.md, so every one of them is a
local artifact that a container reset wipes. This manifest is deliberately a committable .txt for that
reason.

The manifest is the committed list in golden/exports.txt. A migration commit is expected to change it
-- that is the point of renaming the dual-purpose kernels to modiff_* / baseline_* -- so the workflow
is: make the change, run with UPDATE_MANIFEST=1, and READ THE PRINTED DIFF. It must be exactly the
renames that commit intended and nothing else. An unexplained removal is a dropped kernel.

  python integration/tests/test_export_manifest.py                  # gate
  UPDATE_MANIFEST=1 python integration/tests/test_export_manifest.py  # after an intended change
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import modiff_cutlass as mc                                          # noqa: E402

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "exports.txt")
UPDATE = os.environ.get("UPDATE_MANIFEST", "0") == "1"


def current():
    """Every name Python can call on the module. dir() and not pybind's source, deliberately: this
    asks what the BUILT extension offers, which is the thing call sites depend on."""
    return sorted(n for n in dir(mc) if not n.startswith("_"))


def main():
    now = current()
    if not os.path.exists(MANIFEST):
        if not UPDATE:
            print(f"no manifest at {MANIFEST}; create it with UPDATE_MANIFEST=1")
            return 1
        with open(MANIFEST, "w") as f:
            f.write("\n".join(now) + "\n")
        print(f"wrote {MANIFEST} with {len(now)} exports")
        return 0

    with open(MANIFEST) as f:
        want = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    added = [n for n in now if n not in set(want)]
    removed = [n for n in want if n not in set(now)]

    print(f"manifest {len(want)} exports, module {len(now)}")
    for n in added:
        print(f"  + {n}")
    for n in removed:
        print(f"  - {n}")

    if not added and not removed:
        print("\nPASS -- exports unchanged")
        return 0
    if UPDATE:
        with open(MANIFEST, "w") as f:
            f.write("\n".join(now) + "\n")
        print(f"\nUPDATED {MANIFEST}: +{len(added)} -{len(removed)}")
        print("Check the diff above against what this change intended. A removal you cannot name a "
              "reason for is a kernel that got dropped during a file split.")
        return 0
    print(f"\nFAILED -- {len(added)} added, {len(removed)} removed. If this is intended, re-run with "
          f"UPDATE_MANIFEST=1 and check the diff.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
