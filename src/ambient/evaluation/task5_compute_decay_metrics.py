#!/usr/bin/env python3
# src/ambient/evaluation/task5_compute_decay_metrics.py
"""
Computes scalar metrics for the Superposition Decay experiment, tracking how 
models respond to initial prompt bias (Prisoner of the Prior vs. Escaping the Prior).
"""

import json
import argparse
import numpy as np
from pathlib import Path

def compute_metrics(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        data = raw_data.get("results", raw_data)
        
    start_entropies, end_entropies, max_entropies, auc_entropies = [], [], [], []
    collapse_count = 0
    total_count = 0
    
    for idx, trajectory in data.items():
        if not trajectory:
            continue
            
        entropies = [t['entropy'] for t in trajectory]
        steps = [t['step'] for t in trajectory]
        max_step = max(steps)
        if max_step == 0: continue
        
        # Normalize steps to 0-1 for Area Under Curve calculation
        norm_steps = [s / max_step for s in steps]
        
        start_entropies.append(entropies[0])
        end_entropies.append(entropies[-1])
        max_entropies.append(max(entropies))
        auc_entropies.append(np.trapz(entropies, norm_steps))
        
        if entropies[-1] < 0.05:
            collapse_count += 1
            
        total_count += 1
        
    return {
        "Mean Start Entropy (H_0)": np.mean(start_entropies),
        "Mean End Entropy (H_100)": np.mean(end_entropies),
        "Mean Peak Entropy (H_max)": np.mean(max_entropies),
        "Total Superposition (AUC)": np.mean(auc_entropies),
        "Total Collapse Rate (End H < 0.05)": collapse_count / total_count if total_count > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description="Compute Superposition Decay Metrics")
    parser.add_argument("--ar-file", type=Path, required=True, help="Path to AR JSON")
    parser.add_argument("--diff-file", type=Path, required=True, help="Path to Diffusion JSON")
    args = parser.parse_args()

    print("="*50)
    print("=== TEMPORAL SEMANTIC COMMITMENT METRICS ===")
    print("="*50)

    for name, filepath in [("Autoregressive (LLaMA-8B)", args.ar_file), ("Discrete Diffusion (LLaDA-8B)", args.diff_file)]:
        metrics = compute_metrics(filepath)
        print(f"\n>>> {name}")
        for k, v in metrics.items():
            if "Rate" in k:
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v:.4f} bits")

if __name__ == "__main__":
    main()