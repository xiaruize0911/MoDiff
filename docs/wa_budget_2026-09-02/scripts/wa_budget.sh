#!/bin/bash
# W/A precision x granularity budget. One process per arm (fusion switches are import-time).
set -uo pipefail
cd /workspace/MoDiff
A=docs/wa_budget_2026-09-02/scripts/wa_budget_arm.py
OUT=docs/wa_budget_2026-09-02/data/arms.jsonl
: > "$OUT"
run() { echo "### $1" >&2
  python3 "$A" "$@" 2>&1 | tee /tmp/claude-0/-workspace/8ac24b7c-6ed9-4e73-97a7-dda1fd9a380b/scratchpad/wa_$1.log \
    | grep '^ARMJSON:' | sed 's/^ARMJSON://' >> "$OUT" || echo "  $1 FAILED" >&2; }
#                    arm              ablk aqmax wbits wblock
run fp16              -3   127   -1  0
run floor             -3   127   -1  0
# --- 8 bit ---
run w8_perchan        -3   127    8  0
run w8_block64        -3   127    8  64
run a8_pertensor      -2   127   -1  0
run a8_block64        64   127   -1  0
run both8_coarse      -2   127    8  0
run both8_block64     64   127    8  64
# --- 4 bit ---
run w4_perchan        -3     7    4  0
run w4_block64        -3     7    4  64
run w4_block32        -3     7    4  32
run a4_pertensor      -2     7   -1  0
run a4_block64        64     7   -1  0
run both4_coarse      -2     7    4  0
run both4_block64     64     7    4  64
echo DONE >&2
