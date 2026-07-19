#!/usr/bin/env bash
# E2E pipeline sweep after consolidating on the AWQ-tiling ports (the sole int8/int4 Linear backend).
# Compares fp16 vs int8(ports) vs int4(ports) on the churches UNet. 3 reps/mode, reports min ms/step.
# Output: docs/quant_speedup_vs_fp16_2026-07-16/data/e2e_sweep_consolidated.txt
set -u
cd /workspace/MoDiff
export PYTHONPATH="src/taming-transformers:${PYTHONPATH:-}"
OUT=docs/quant_speedup_vs_fp16_2026-07-16/data/e2e_sweep_consolidated.txt
STEPS=${STEPS:-30}; BATCH=${BATCH:-16}; SAMPLES=${SAMPLES:-16}; REPS=${REPS:-3}
: > "$OUT"
run_mode () {
  local label="$1"; shift
  local best=""
  echo "=== $label ===" | tee -a "$OUT"
  for r in $(seq 1 "$REPS"); do
    ms=$("$@" 2>/dev/null | grep -oP 'Per-step:\s*\K[0-9.]+' | tail -1)
    echo "  rep$r: ${ms:-NA} ms/step" | tee -a "$OUT"
    if [ -n "${ms:-}" ]; then
      if [ -z "$best" ] || python3 -c "import sys; sys.exit(0 if float('$ms')<float('$best') else 1)"; then best="$ms"; fi
    fi
  done
  echo "  MIN: ${best:-NA} ms/step" | tee -a "$OUT"
}

COMMON="--linear_backend int_gemm --steps $STEPS --batch_size $BATCH --num_samples $SAMPLES"

run_mode "fp16 (baseline)" \
  env python3.11 integration/benchmarks/benchmark_ldm.py --mode fp16 --steps "$STEPS" --batch_size "$BATCH" --num_samples "$SAMPLES"
run_mode "int8 (AWQ-tiling port)" \
  env MODIFF_QUANT_LINEAR=1 python3.11 integration/benchmarks/benchmark_ldm.py --mode int8 $COMMON
run_mode "int4 (AWQ-tiling port)" \
  env MODIFF_QUANT_LINEAR=1 python3.11 integration/benchmarks/benchmark_ldm.py --mode int4 $COMMON

echo "WROTE $OUT"
