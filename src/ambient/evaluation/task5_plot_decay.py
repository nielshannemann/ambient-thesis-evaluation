#!/usr/bin/env python3
# src/ambient/evaluation/task5_plot_decay.py
"""
=============================================================================
TASK 5: PLOT TEMPORAL SEMANTIC COMMITMENT
=============================================================================
This script parses the trajectory JSONs from Task 5 and plots the normalized 
Shannon Entropy over the generation/unmasking process. 

It maps Autoregressive tokens and Diffusion steps onto a shared X-axis 
(0% to 100% Complete) to visualize the exact moment of mode collapse.
=============================================================================
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-ready plot styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

def load_and_interpolate(file_path: Path):
    """
    Loads a trajectory JSON and interpolates the discrete steps onto a 
    standardized 0.0 to 1.0 (0% to 100%) continuous timeline.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        data = raw_data.get("results", raw_data)
        
    common_x = np.linspace(0, 1, 100)
    all_interpolated_entropies = []
    
    for inst_id, trajectory in data.items():
        if not trajectory:
            continue
            
        steps = [t["step"] for t in trajectory]
        entropies = [t["entropy"] for t in trajectory]
        
        max_step = max(steps)
        if max_step == 0: 
            continue
            
        # Normalize the discrete steps (e.g., Token 5 of 20 = 0.25)
        norm_steps = [s / max_step for s in steps]
        
        # Interpolate the entropies onto the 100-point common X-axis
        interp_y = np.interp(common_x, norm_steps, entropies)
        all_interpolated_entropies.append(interp_y)
        
    return common_x, np.array(all_interpolated_entropies)


def main():
    parser = argparse.ArgumentParser(description="Plot Superposition Decay (Entropy)")
    parser.add_argument("--ar-file", type=Path, help="Path to AR trajectories JSON")
    parser.add_argument("--diff-file", type=Path, help="Path to Diffusion trajectories JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("results/task5/"), help="Directory to save plots")
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.ar_file and not args.diff_file:
        print("[ERROR] You must provide at least one trajectory file (--ar-file or --diff-file).")
        return
        
    plt.figure(figsize=(10, 6))
    
    # --- Plot Autoregressive Track ---
    if args.ar_file and args.ar_file.exists():
        print(f"[INFO] Processing AR Data: {args.ar_file.name}")
        x_ar, y_ar_matrix = load_and_interpolate(args.ar_file)
        
        mean_ar = np.mean(y_ar_matrix, axis=0)
        std_ar = np.std(y_ar_matrix, axis=0)
        
        plt.plot(x_ar * 100, mean_ar, label="Autoregressive (LLaMA)", color="#e74c3c", linewidth=3)
        plt.fill_between(x_ar * 100, mean_ar - std_ar, mean_ar + std_ar, color="#e74c3c", alpha=0.2)

    # --- Plot Diffusion Track ---
    if args.diff_file and args.diff_file.exists():
        print(f"[INFO] Processing Diffusion Data: {args.diff_file.name}")
        x_diff, y_diff_matrix = load_and_interpolate(args.diff_file)
        
        mean_diff = np.mean(y_diff_matrix, axis=0)
        std_diff = np.std(y_diff_matrix, axis=0)
        
        plt.plot(x_diff * 100, mean_diff, label="Discrete Diffusion (LLaDA)", color="#3498db", linewidth=3)
        plt.fill_between(x_diff * 100, mean_diff - std_diff, mean_diff + std_diff, color="#3498db", alpha=0.2)

    # --- Formatting the Graph ---
    plt.title("Temporal Semantic Commitment (Superposition Decay)", pad=15, fontweight="bold")
    plt.xlabel("Generation Progress (%)", fontweight="bold")
    plt.ylabel("Semantic Entropy (Bits)", fontweight="bold")
    
    # 1.0 Entropy = Perfect ambiguity (50/50 probability)
    # 0.0 Entropy = Complete mode collapse (100/0 probability)
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 100)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    
    # Add a horizontal line representing total mode collapse
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    out_file = args.out_dir / "superposition_decay_comparison.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Success! Plot saved to: {out_file}")

if __name__ == "__main__":
    main()