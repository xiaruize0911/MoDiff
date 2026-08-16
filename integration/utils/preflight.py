"""Assert an entry point's dependencies BEFORE it spends its first GPU second.

WHY THIS EXISTS. This container lost its Python environment at least once since 2026-08-13, and on
2026-08-16 six separate dependencies were rediscovered one at a time, each *inside* a job:

    matplotlib                                     regenerating report plots
    markdown, weasyprint + the libpango system libs  rendering a PDF
    omegaconf, einops, pytorch-lightning, torchmetrics  first model build of a paired A/B
    pytorch-fid                                    the LAST step of a 25-minute pipeline
    ninja                                          silently -- see below
    lmdb                                           starting a 50k reference export

`docs/aq_fusion_2026-08-12`'s provisioning note has gone stale three times, which is the argument for
putting the check in code rather than in prose.

NINJA IS WHY THIS ASSERTS RATHER THAN WARNS. Its absence did not raise: without it `torch.utils
.cpp_extension` falls back to distutils, which does not track header dependencies, so a header-only
change rebuilt NOTHING and produced a `.so` silently missing the change. A dependency that errors costs
minutes; one that silently yields a stale artifact costs a wrong conclusion.

TWO PROPERTIES THAT MATTER, both learned from how the six were found:

  * It reports EVERY missing item at once. Six sequential discoveries is what "fail on the first one"
    produces, and each cost a job restart.
  * It runs before the heavy imports and before any CUDA context, so the cost of being wrong is seconds.

Usage, at the top of an entry point:

    from integration.utils.preflight import preflight
    preflight("omegaconf", "einops", "pytorch_lightning", "torchmetrics", libs=["pango-1.0"])

Install suggestions come out under the torch pin, because an unconstrained install can swap torch out
from under the built extension -- `modiff_cutlass` is compiled against a specific ABI.
"""
import ctypes.util
import importlib.util
import shutil
import sys

#: torch is pinned because the CUDA extension is built against it; an unconstrained `pip install`
#: resolving a different torch is how a working tree becomes an unimportable one.
TORCH_PIN = "torch==2.4.1"

#: import name -> pip name, where they differ
PIP_NAME = {
    "pytorch_lightning": "pytorch-lightning==1.4.2",
    "torchmetrics": "torchmetrics==0.6.0",
    "pytorch_fid": "pytorch-fid",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
}


def _have_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def preflight(*modules, libs=(), tools=(), what=None, exit_code=2):
    """Check imports, shared libraries and executables. Raise SystemExit listing ALL that are missing.

    modules  import names, e.g. "omegaconf"
    libs     shared-library stems for ctypes.util.find_library, e.g. "pango-1.0" (no lib prefix/suffix)
    tools    executables that must be on PATH, e.g. "ninja". Present here rather than as a module check
             because ninja is consumed by torch's build backend, not imported.
    what     a label for the message, defaults to the calling script's name
    """
    missing_mod = [m for m in modules if not _have_module(m)]
    missing_lib = [l for l in libs if ctypes.util.find_library(l) is None]
    missing_tool = [t for t in tools if shutil.which(t) is None]
    if not (missing_mod or missing_lib or missing_tool):
        return

    label = what or sys.argv[0]
    lines = [f"preflight FAILED for {label} -- stopping before any GPU work rather than partway through."]
    if missing_mod:
        pips = " ".join(PIP_NAME.get(m, m) for m in missing_mod)
        lines.append(f"  missing Python modules: {', '.join(missing_mod)}")
        lines.append(f"    pip install {pips} -c <(echo {TORCH_PIN})")
    if missing_lib:
        lines.append(f"  missing shared libraries: {', '.join(missing_lib)}")
        lines.append(f"    apt-get install -y lib{missing_lib[0]}-0   # names vary; check the distro")
    if missing_tool:
        lines.append(f"  missing executables on PATH: {', '.join(missing_tool)}")
        lines.append(f"    pip install {' '.join(missing_tool)} -c <(echo {TORCH_PIN})")
        if "ninja" in missing_tool:
            lines.append("    NOTE: without ninja, distutils does not track header dependencies -- a")
            lines.append("    header-only change will rebuild NOTHING and leave a stale .so. This is")
            lines.append("    asserted, not warned, for exactly that reason.")
    raise SystemExit("\n".join(lines))


#: the sets the repo's entry points actually need, named so a script asks for a role rather than a list
MODEL = ("omegaconf", "einops", "pytorch_lightning", "torchmetrics", "tqdm", "PIL")
PLOTS = ("matplotlib",)
FID = ("pytorch_fid", "scipy", "PIL")
LMDB = ("lmdb", "PIL")
PDF = ("markdown", "weasyprint")
BUILD_TOOLS = ("ninja",)
PDF_LIBS = ("pango-1.0", "harfbuzz")
