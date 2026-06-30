#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-integration/results/nsys_memory_redo}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-168}"
NUM_SAMPLES="${NUM_SAMPLES:-168}"
LINEAR_BACKEND="${LINEAR_BACKEND:-int_gemm}"
BENCHMARK="${BENCHMARK:-integration/benchmarks/benchmark_ldm.py}"
MODES="${MODES:-fp16 int8 int8_baseline int4 int4_baseline}"
ANALYZE_MODES="${ANALYZE_MODES:-$MODES}"

mkdir -p "$OUT_DIR/profiles" "$OUT_DIR/benchmarks"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys is not installed or not on PATH." >&2
  echo "Install NVIDIA Nsight Systems on the profiling machine, then rerun this script." >&2
  exit 127
fi

run_profile() {
  local mode="$1"
  shift
  local base="$OUT_DIR/profiles/${mode}_s${STEPS}_b${BATCH_SIZE}"
  local bench_out="$OUT_DIR/benchmarks/${mode}"

  mkdir -p "$bench_out"
  echo "==> Profiling $mode"
  nsys profile \
    --force-overwrite=true \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    -o "$base" \
    python "$BENCHMARK" \
      --mode "$mode" \
      --steps "$STEPS" \
      --batch_size "$BATCH_SIZE" \
      --num_samples "$NUM_SAMPLES" \
      --output_dir "$bench_out" \
      "$@"

  if command -v nsys >/dev/null 2>&1; then
    nsys export --force-overwrite=true --type sqlite --output "${base}.sqlite" "${base}.nsys-rep" || true
  fi
}

for mode in $MODES; do
  case "$mode" in
    int8|int8_baseline|int4|int4_baseline)
      run_profile "$mode" --linear_backend "$LINEAR_BACKEND"
      ;;
    *)
      run_profile "$mode"
      ;;
  esac
done

python integration/benchmarks/analyze_nsys_memory.py \
  --profile-dir "$OUT_DIR/profiles" \
  --benchmark-dir "$OUT_DIR/benchmarks" \
  --output-json "$OUT_DIR/nsys_memory_summary.json" \
  --output-md "$OUT_DIR/NSYS_MEMORY_REDO_REPORT.md" \
  --modes $ANALYZE_MODES

echo "Done. Report: $OUT_DIR/NSYS_MEMORY_REDO_REPORT.md"
