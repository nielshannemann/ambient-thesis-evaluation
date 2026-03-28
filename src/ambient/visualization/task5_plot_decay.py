#!/usr/bin/env python3
# src/ambient/evaluation/task5_plot_decay.py
"""
Plot Task 5: Temporal Semantic Commitment trajectories.

This script parses Task-5 trajectory JSON files and plots normalized Shannon
entropy over a shared 0%--100% progress axis for autoregressive and diffusion
models.

Compatibility:
- supports updated trajectory format:
    {"metadata": ..., "results": {id: {"trajectory": [...]}}}
- also supports older direct trajectory maps:
    {"results": {id: [...]}} or {id: [...]}

Thesis references:
- Section 4.7: Task 5: Temporal Semantic Commitment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Publication-oriented styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)


def _extract_trajectory_records(raw_data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Extract trajectory lists from supported Task-5 JSON formats.
    """
    data = raw_data.get("results", raw_data)
    extracted: List[List[Dict[str, Any]]] = []

    if not isinstance(data, dict):
        return extracted

    for _, value in data.items():
        if not value:
            continue

        # New format: instance dict containing a "trajectory" field
        if isinstance(value, dict) and "trajectory" in value:
            trajectory = value.get("trajectory")
            if isinstance(trajectory, list) and trajectory:
                extracted.append(trajectory)
            continue

        # Older format: value itself is the trajectory list
        if isinstance(value, list) and value:
            extracted.append(value)

    return extracted


def load_and_interpolate(file_path: Path):
    """
    Load a trajectory JSON and interpolate discrete entropy traces onto a
    standardized 0.0--1.0 progress axis.
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    trajectories = _extract_trajectory_records(raw_data)

    common_x = np.linspace(0, 1, 100)
    all_interpolated_entropies = []

    for trajectory in trajectories:
        try:
            steps = [float(t["step"]) for t in trajectory if "step" in t]
            entropies = [float(t["entropy"]) for t in trajectory if "entropy" in t]
        except Exception:
            continue

        if not steps or not entropies or len(steps) != len(entropies):
            continue

        max_step = max(steps)
        if max_step == 0:
            continue

        norm_steps = [s / max_step for s in steps]
        interp_y = np.interp(common_x, norm_steps, entropies)
        all_interpolated_entropies.append(interp_y)

    return common_x, np.array(all_interpolated_entropies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Task-5 temporal semantic commitment")
    parser.add_argument("--ar-file", type=Path, help="Path to AR trajectories JSON")
    parser.add_argument("--diff-file", type=Path, help="Path to diffusion trajectories JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("results/task5/"), help="Directory to save plots")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.ar_file and not args.diff_file:
        print("[ERROR] You must provide at least one trajectory file (--ar-file or --diff-file).")
        return

    plt.figure(figsize=(10, 6))

    # --- Plot Autoregressive Track ---
    if args.ar_file and args.ar_file.exists():
        print(f"[INFO] Processing AR data: {args.ar_file.name}")
        x_ar, y_ar_matrix = load_and_interpolate(args.ar_file)

        if len(y_ar_matrix) > 0:
            mean_ar = np.mean(y_ar_matrix, axis=0)
            std_ar = np.std(y_ar_matrix, axis=0)

            plt.plot(
                x_ar * 100,
                mean_ar,
                label="Autoregressive (LLaMA)",
                color="#e74c3c",
                linewidth=3,
            )
            plt.fill_between(
                x_ar * 100,
                mean_ar - std_ar,
                mean_ar + std_ar,
                color="#e74c3c",
                alpha=0.2,
            )
        else:
            print("[WARN] No valid AR trajectories found for plotting.")

    # --- Plot Diffusion Track ---
    if args.diff_file and args.diff_file.exists():
        print(f"[INFO] Processing diffusion data: {args.diff_file.name}")
        x_diff, y_diff_matrix = load_and_interpolate(args.diff_file)

        if len(y_diff_matrix) > 0:
            mean_diff = np.mean(y_diff_matrix, axis=0)
            std_diff = np.std(y_diff_matrix, axis=0)

            plt.plot(
                x_diff * 100,
                mean_diff,
                label="Discrete Diffusion (LLaDA)",
                color="#3498db",
                linewidth=3,
            )
            plt.fill_between(
                x_diff * 100,
                mean_diff - std_diff,
                mean_diff + std_diff,
                color="#3498db",
                alpha=0.2,
            )
        else:
            print("[WARN] No valid diffusion trajectories found for plotting.")

    # --- Formatting ---
    plt.title("Temporal Semantic Commitment", pad=15, fontweight="bold")
    plt.xlabel("Generation Progress (%)", fontweight="bold")
    plt.ylabel("Semantic Entropy (Bits)", fontweight="bold")

    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 100)
    plt.legend(loc="upper right", frameon=True, shadow=True)

    # Entropy = 0 corresponds to full commitment to one reading
    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    out_file = args.out_dir / "temporal_semantic_commitment_comparison.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Success! Plot saved to: {out_file}")


if __name__ == "__main__":
    main()