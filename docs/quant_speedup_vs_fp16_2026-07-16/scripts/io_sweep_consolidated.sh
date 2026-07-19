#!/usr/bin/env bash
# Total-IO / memory-traffic sweep after consolidating on the AWQ-tiling ports.
# Runs each production mode separately (fp16 unquantized; int8/int4 with MODIFF_QUANT_LINEAR=1 -> ports)
# through run_nsys_memory_redo.sh, so the fp16 baseline is NOT accidentally quantized.
# Output: integration/results/awqtile_io_consolidated/{fp16,int8,int4}/nsys_memory_summary.json
set -u
cd /workspace/MoDiff
export PATH="/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64:$PATH"
export PYTHONPATH="src/taming-transformers:${PYTHONPATH:-}"
STEPS=${STEPS:-15}; BATCH=${BATCH:-16}; SAMPLES=${SAMPLES:-16}
BASE=integration/results/awqtile_io_consolidated
run () {
  local mode="$1"; local quant="$2"
  echo "########## IO mode=$mode (MODIFF_QUANT_LINEAR=$quant) ##########"
  env MODIFF_QUANT_LINEAR="$quant" \
    OUT_DIR="$BASE/$mode" STEPS="$STEPS" BATCH_SIZE="$BATCH" NUM_SAMPLES="$SAMPLES" \
    LINEAR_BACKEND=int_gemm MODES="$mode" ANALYZE_MODES="$mode" \
    bash integration/benchmarks/run_nsys_memory_redo.sh
}
run fp16 0
run int8 1
run int4 1
echo "WROTE $BASE/{fp16,int8,int4}/nsys_memory_summary.json"
