#!/bin/bash
# Measured per-kernel / per-shape DRAM read+write bytes via NVBit (tools/mem_bytes) — NO perf
# counters needed (works on this counter-locked box). One process per (family,mode,shape) config:
# the whole op runs inside cudaProfilerStart/Stop, so MEMBYTES_TOTAL = that config's DRAM IO and the
# per-kernel MEMBYTES lines are its kernel breakdown (incl. CUTLASS conv / cuDNN).
set -e
cd /workspace/MoDiff
source setup_cuda_env.sh >/dev/null 2>&1
TOOL=/workspace/nvbit_release_x86_64/tools/mem_bytes/mem_bytes.so
DRV=docs/benchmark_5mode_2026-07-20/scripts/nvbit_io_driver.py
OUT=docs/benchmark_5mode_2026-07-20/data/nvbit_io_raw.txt
if [ ! -f "$TOOL" ]; then echo "mem_bytes.so missing — build it: (cd /workspace/nvbit_release_x86_64/tools/mem_bytes && make)"; exit 1; fi
: > "$OUT"
n=0
for tag in $(python "$DRV" --list); do
  n=$((n+1))
  echo "### TAG $tag" >> "$OUT"
  ACTIVE_FROM_START=0 MANGLED_NAMES=0 LD_PRELOAD="$TOOL" \
    python "$DRV" --one "$tag" 2>&1 | grep -aE "^MEMBYTES" >> "$OUT" || true
  echo "  [$n] $tag"
done
echo "ALL_NVBIT_DONE ($n configs) -> $OUT"
python docs/benchmark_5mode_2026-07-20/scripts/parse_nvbit_io.py
