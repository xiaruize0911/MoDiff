#!/usr/bin/env python3
"""
Evaluate FID score between Real LSUN-Church dataset (or stats) and 
generated images from FP32, INT8, and INT4 modes.

Usage:
    python integration/eval_fid_lsun.py --real_path /path/to/lsun/val/images_or_stats.npz

Requirements:
    pip install pytorch-fid
"""

import argparse
import os
import sys
import subprocess
import torch
from tabulate import tabulate

def check_install():
    try:
        import pytorch_fid
        print(f"Found pytorch-fid version: {pytorch_fid.__version__}")
        return True
    except ImportError:
        print("Error: pytorch-fid is not installed.")
        print("Please run: pip install pytorch-fid")
        return False

def compute_fid(real_path, fake_path, batch_size=50, device='cuda', dims=2048):
    """
    Computes FID using pytorch-fid module via command line interface.
    This ensures compatibility and avoids internal API changes issues.
    """
    if not os.path.exists(fake_path):
        print(f"Warning: Fake path does not exist: {fake_path}")
        return None

    # Check if folder is empty
    if os.path.isdir(fake_path):
        files = [f for f in os.listdir(fake_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"Warning: Fake path is empty: {fake_path}")
            return None
        print(f"  Found {len(files)} images in {fake_path}")
        if len(files) < 100:
            print("  Warning: Low sample count (<100). FID may be unstable. 50k recommended.")
    else:
        # It might be an npz file
        pass

    print(f"Computing FID between:\n  Real: {real_path}\n  Fake: {fake_path}")
    
    cmd = [
        sys.executable, '-m', 'pytorch_fid',
        real_path,
        fake_path,
        '--device', device,
        '--batch-size', str(batch_size),
        '--dims', str(dims)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Output format is usually "FID:  12.3456789"
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if "FID:" in line:
                fid_score = float(line.split(':')[-1].strip())
                return fid_score
        
        print("Could not parse FID from output:")
        print(result.stdout)
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error computing FID:")
        print(e.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Compute FID for LSUN-Church comparison")
    parser.add_argument('--real_path', type=str, required=True,
                        help='Path to real images folder or precomputed .npz stats file')
    parser.add_argument('--results_dir', type=str, default='integration/results_ldm_benchmark',
                        help='Directory containing generated results')
    parser.add_argument('--modes', type=str, nargs='+', default=['fp32', 'int8', 'int4'],
                        help='Modes to evaluate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for FID')
    
    args = parser.parse_args()
    
    if not check_install():
        sys.exit(1)
        
    if not os.path.exists(args.real_path):
        print(f"Error: Real path not found: {args.real_path}")
        sys.exit(1)
        
    results = []
    
    print("\n" + "="*60)
    print("FID Evaluation - LSUN Churches")
    print("="*60)
    
    for mode in args.modes:
        fake_path = os.path.join(args.results_dir, mode)
        
        print(f"\nEvaluating Mode: {mode.upper()}")
        fid = compute_fid(args.real_path, fake_path, args.batch_size, args.device)
        
        res = {
            'Mode': mode.upper(),
            'FID': fid if fid is not None else "N/A",
            'Samples Path': fake_path
        }
        results.append(res)
        
        if fid is not None:
            print(f"-> FID: {fid:.2f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Save simple report
    with open('fid_report.txt', 'w') as f:
        f.write(tabulate(results, headers="keys", tablefmt="grid"))
    print("\nReport saved to fid_report.txt")

if __name__ == "__main__":
    main()
