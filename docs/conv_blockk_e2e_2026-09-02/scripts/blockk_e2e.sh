#!/bin/bash
# Driver: one process per arm (fusion kill switches are import-time).
#   source /workspace/MoDiff/setup_cuda_env.sh
#   bash docs/conv_blockk_e2e_2026-09-02/scripts/blockk_e2e.sh
set -uo pipefail
cd /workspace/MoDiff
A=docs/conv_blockk_e2e_2026-09-02/scripts/blockk_e2e_arm.py
OUT=docs/conv_blockk_e2e_2026-09-02/data/arms.jsonl
: > "$OUT"
run() {  # name mode fusions blockk ctrl
    echo "### $1" >&2
    python3 "$A" "$@" 2>&1 | tee /tmp/claude-0/-workspace/8ac24b7c-6ed9-4e73-97a7-dda1fd9a380b/scratchpad/arm_$1.log \
        | grep '^ARMJSON:' | sed 's/^ARMJSON://' >> "$OUT" \
        || echo "  ARM $1 FAILED (see scratchpad/arm_$1.log)" >&2
}
run fp16              fp16          on  0  0
for m in int8_baseline int8; do
    run ${m}_shipped   "$m" on  0  0
    run ${m}_unfused   "$m" off 0  0
    run ${m}_ctrl      "$m" off 64 1
    run ${m}_b64       "$m" off 64 0
    run ${m}_b32       "$m" off 32 0
done
echo "wrote $OUT" >&2
