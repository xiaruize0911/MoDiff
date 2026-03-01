#!/usr/bin/env python3
"""
Re-run only Experiment 1 (full pipeline speedup) with benchmark_ldm.py defaults:
  steps=200, num_samples=128, batch_size=32
Then patch experiment_results.json and regenerate report + plot 1.
"""
import sys, os
sys.path.insert(0, '/workspace/MoDiff')
os.chdir('/workspace/MoDiff')

import json
import subprocess

# Import experiment 1 directly from run_all_experiments
from analysis_int4_vs_int8.run_all_experiments import experiment_1_pipeline_speedup

RESULTS_PATH = 'analysis_int4_vs_int8/experiment_results.json'

print("=" * 60)
print("Re-running Experiment 1: steps=200, num_samples=128, batch_size=32")
print("=" * 60)

result = experiment_1_pipeline_speedup(steps=200, num_samples=128, batch_size=32)

# Patch the JSON
with open(RESULTS_PATH) as f:
    data = json.load(f)

data['exp1_pipeline'] = result

with open(RESULTS_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nPatched {RESULTS_PATH}")
print("\nRebuilding report and plots...")

subprocess.run([sys.executable, 'analysis_int4_vs_int8/generate_report.py'], check=True)

print("\nDone.")
