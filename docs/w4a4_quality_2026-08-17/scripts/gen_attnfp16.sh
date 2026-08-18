#!/bin/bash
# Two arms toward the reference standard, isolating then combining the two surviving axes:
#   A) ATTN_FP16=1                -- full-precision score path, as qdiff's QuantAttnBlock has
#   B) ATTN_FP16=1 --adaround 1   -- plus AdaRound W4 weights, i.e. the paper's own two ingredients
#
# NON-VACUITY for ATTN_FP16: the build log must NOT contain "QUANTIZED standard attention" and MUST
# contain "token-major (MATH SDPA)". A flag that silently failed to apply would produce images
# identical to the baseline and read as "attention quantization does not matter".
set -o pipefail
cd /workspace/MoDiff || { echo "GEN FAILED: cannot cd"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
OUT=/workspace/fid_warmup5
echo "cwd=$(pwd)  $(date -u +%H:%M:%S)"

run() {   # $1=tag  $2=extra args  $3=expected folder
  echo "=== $1"
  ATTN_FP16=1 MODIFF_WARMUP_STEPS=5 python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
    --n 16 --batch 16 --steps 50 --modes int4_l0 --out "$OUT" $2 > "$L/gen_$1.log" 2>&1
  rc=$?
  n=$(ls "$OUT/$3"/*.png 2>/dev/null | wc -l)
  q=$(grep -c "QUANTIZED standard attention" "$L/gen_$1.log")
  m=$(grep -c "token-major (MATH SDPA)" "$L/gen_$1.log")
  echo "  rc=$rc  images=$n  quantized_attn_lines=$q  math_sdpa_lines=$m"
  [ "$rc" -eq 0 ] || { echo "  $1 FAILED: rc"; return 1; }
  [ "$n" -ge 16 ] || { echo "  $1 FAILED: only $n images"; return 1; }
  [ "$q" -eq 0 ] || { echo "  $1 FAILED: attention was still quantized -- ATTN_FP16 did not apply"; return 1; }
  [ "$m" -ge 1 ] || { echo "  $1 FAILED: no MATH SDPA line -- attention route unconfirmed"; return 1; }
  echo "  $1 OK"
}

bad=0
run attnfp16 "" int4_modiff_l0_attnfp16 || bad=1
run attnfp16_adaround "--adaround 1" int4_modiff_l0_adaround_attnfp16 || bad=1
[ "$bad" -eq 0 ] || { echo "GEN FAILED"; exit 1; }
echo "GEN OK $(date -u +%H:%M:%S)"
