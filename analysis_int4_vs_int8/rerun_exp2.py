#!/usr/bin/env python3
"""Re-run only experiment 2 and update experiment_results.json in place."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_all_experiments import experiment_2_breakdown
import json, time

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results.json')

with open(RESULTS_PATH) as f:
    all_data = json.load(f)

print("Re-running Experiment 2 (with calibration + autocast, batch_size=32)...")
all_data['exp2_breakdown'] = experiment_2_breakdown(steps=50, num_batches=2, batch_size=32)
all_data['metadata']['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

with open(RESULTS_PATH, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f"\nSaved updated results to {RESULTS_PATH}")
