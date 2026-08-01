"""Benchmark + profiling suite behind the FP16 / INT8 / INT4 checkpoint reports.

Moved here from docs/final_report_2026-07-28/scripts/ so the code lives with the rest of the
package instead of inside a dated report directory. The reports and their measured JSON/CSV
stay under docs/ -- only the code moved.

  ck_bench_stats.py               timing with a reported distribution (t-CI, CV, spread)
  ck_stages.py                    kernel-name -> pipeline-stage attribution
  ck_report_numbers.py            emits every report table from the measured JSON
  ck_verify_report.py             fails if a figure quoted in a report is not in the data
  make_checkpoint_report_plots.py the four report figures
  e2e_three_mode_bench.py         end-to-end DDIM, three modes in one process
  layer_pipeline_bench.py         per-layer-type kernel pipeline, three modes in one process
  profile_tree.py                 whole-model kernel tree + the shared role classifier
"""
