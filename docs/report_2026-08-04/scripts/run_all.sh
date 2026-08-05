#!/bin/bash
# Serial driver: everything shares one GPU, so nothing runs concurrently. Order is
# longest-first so a partial run still yields the headline e2e table.
set -x
cd /workspace/MoDiff
D=docs/report_2026-08-04
python -u docs/modiff_correctness_2026-08-03/scripts/e2e_wallclock.py 2>&1 | tail -40 > $D/data/e2e_raw.txt
cp docs/modiff_correctness_2026-08-03/data/e2e_wallclock.json $D/data/ 2>/dev/null
python -u docs/modiff_correctness_2026-08-03/scripts/modiff_bucket_breakdown.py > $D/data/profile_raw.txt 2>&1
cp docs/modiff_correctness_2026-08-03/data/bucket_breakdown.json $D/data/ 2>/dev/null
python -u $D/scripts/conv_kernel.py > $D/data/conv_raw.txt 2>&1
python -u $D/scripts/attn_kernel_fair.py > $D/data/attn_raw.txt 2>&1
echo ALL_DONE
