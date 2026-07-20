#!/bin/bash
# Measured per-kernel, per-shape DRAM read/write bytes via Nsight Compute.
# REQUIRES UNLOCKED GPU PERF COUNTERS (see NCU_IO_README.md). On a locked box ncu returns
# ERR_NVGPUCTRPERM and this produces no data.
#
# Usage: bash run_ncu_io.sh [conv|linear|attn|all]   (default all)
set -e
cd /workspace/MoDiff
source setup_cuda_env.sh >/dev/null 2>&1
NCU=/usr/local/cuda/bin/ncu
OUT=docs/benchmark_5mode_2026-07-20/data
FAM=${1:-all}
export PYTHONPATH=/workspace/MoDiff/src/taming-transformers:/workspace/MoDiff
export CUTLASS_PATH=/workspace/cutlass
# read + write split (+ total, duration, %peak-BW). NVTX so each kernel maps to family|mode|shape.
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed

echo "running ncu (needs unlocked counters) on family=$FAM ..."
"$NCU" --target-processes all --nvtx \
    --metrics "$METRICS" --csv --page raw \
    python docs/benchmark_5mode_2026-07-20/scripts/ncu_io_driver.py "$FAM" \
    > "$OUT/ncu_io_raw_${FAM}.csv" 2> "$OUT/ncu_io_${FAM}.log" || true

# ncu writes ERR_NVGPUCTRPERM to stdout (the raw csv) AND/OR stderr (the log) — check both.
if grep -qs "ERR_NVGPUCTRPERM" "$OUT/ncu_io_raw_${FAM}.csv" "$OUT/ncu_io_${FAM}.log"; then
    echo "BLOCKED: GPU perf counters are locked (ERR_NVGPUCTRPERM). Unlock first — see NCU_IO_README.md."
    exit 2
fi
echo "wrote $OUT/ncu_io_raw_${FAM}.csv ; parsing ..."
python docs/benchmark_5mode_2026-07-20/scripts/parse_ncu_io.py "$FAM"
