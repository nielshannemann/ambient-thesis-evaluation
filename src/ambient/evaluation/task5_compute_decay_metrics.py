#!/usr/bin/env python3
# src/ambient/evaluation/task5_compute_decay_metrics.py
"""
Compute scalar summary metrics for Task 5: Temporal Semantic Commitment.

This script aggregates per-instance entropy trajectories into the summary
statistics reported in the thesis:
- Mean Start Entropy (H_0)
- Mean End Entropy (H_100)
- Mean Peak Entropy (H_max)
- Area Under the Curve (AUC)
- Collapse Rate

It additionally computes percentile bootstrap confidence intervals by
resampling whole trajectories (instances) with replacement.

New in this version:
- reports paired model differences (LLaDA - LLaMA) with paired bootstrap CIs
  over the intersection of shared instance IDs.

Compatibility:
- supports the updated trajectory format:
    {"metadata": ..., "results": {id: {"trajectory": [...]}}}
- also supports older direct trajectory maps:
    {"results": {id: [...]}} or {id: [...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

METRIC_ORDER = [
    "Mean Start Entropy (H_0)",
    "Mean End Entropy (H_100)",
    "Mean Peak Entropy (H_max)",
    "Area Under Entropy Curve (AUC)",
    "Collapse Rate (End H < 0.05)",
]


def _extract_trajectory_map(raw_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract trajectory lists keyed by instance id from supported Task-5 JSON formats.

    Supported forms:
    1. {"metadata": ..., "results": {id: {"trajectory": [...]}}}
    2. {"results": {id: [...]}}
    3. {id: [...]}
    """
    data = raw_data.get("results", raw_data)
    extracted: Dict[str, List[Dict[str, Any]]] = {}

    if not isinstance(data, dict):
        return extracted

    for key, value in data.items():
        if not value:
            continue

        if isinstance(value, dict) and "trajectory" in value:
            trajectory = value.get("trajectory")
            if isinstance(trajectory, list) and trajectory:
                extracted[str(key)] = trajectory
            continue

        if isinstance(value, list) and value:
            extracted[str(key)] = value

    return extracted


def _trajectory_to_instance_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, float] | None:
    """Reduce a single entropy trajectory to scalar per-instance metrics."""
    if not trajectory:
        return None

    try:
        entropies = [float(step["entropy"]) for step in trajectory if "entropy" in step]
        steps = [float(step["step"]) for step in trajectory if "step" in step]
    except Exception:
        return None

    if not entropies or not steps or len(entropies) != len(steps):
        return None

    max_step = max(steps)
    if max_step == 0:
        return None

    norm_steps = np.asarray(steps, dtype=float) / float(max_step)
    entropies_arr = np.asarray(entropies, dtype=float)

    return {
        "Mean Start Entropy (H_0)": float(entropies_arr[0]),
        "Mean End Entropy (H_100)": float(entropies_arr[-1]),
        "Mean Peak Entropy (H_max)": float(np.max(entropies_arr)),
        "Area Under Entropy Curve (AUC)": float(np.trapz(entropies_arr, norm_steps)),
        "Collapse Rate (End H < 0.05)": float(entropies_arr[-1] < 0.05),
    }


def load_instance_metrics_map(filepath: Path) -> Dict[str, Dict[str, float]]:
    """Load a saved trajectory JSON and compute per-instance scalar metrics keyed by id."""
    with open(filepath, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    trajectories = _extract_trajectory_map(raw_data)

    instance_metrics: Dict[str, Dict[str, float]] = {}
    for instance_id, trajectory in trajectories.items():
        metrics = _trajectory_to_instance_metrics(trajectory)
        if metrics is not None:
            instance_metrics[instance_id] = metrics

    return instance_metrics


def summarize_instance_metrics(instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-instance metrics into dataset-level summary statistics."""
    if not instance_metrics:
        return {key: float("nan") for key in METRIC_ORDER}

    summary: Dict[str, float] = {}
    for key in METRIC_ORDER:
        values = np.asarray([row[key] for row in instance_metrics], dtype=float)
        summary[key] = float(np.mean(values))
    return summary


def bootstrap_confidence_intervals(
    instance_metrics: List[Dict[str, float]],
    n_bootstrap: int = 5000,
    ci_level: float = 95.0,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute percentile bootstrap confidence intervals for all Task-5 metrics.

    The bootstrap resamples whole instances (trajectories) with replacement.
    This preserves within-trajectory dependence across timepoints.
    """
    if not instance_metrics:
        return {key: {"lower": float("nan"), "upper": float("nan")} for key in METRIC_ORDER}

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci_level < 100.0):
        raise ValueError("ci_level must be between 0 and 100")

    rng = np.random.default_rng(seed)
    n = len(instance_metrics)

    metric_matrix = np.asarray(
        [[row[key] for key in METRIC_ORDER] for row in instance_metrics],
        dtype=float,
    )

    bootstrap_means = np.empty((n_bootstrap, len(METRIC_ORDER)), dtype=float)

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = metric_matrix[sample_idx].mean(axis=0)

    alpha = (100.0 - ci_level) / 2.0
    lower = np.percentile(bootstrap_means, alpha, axis=0)
    upper = np.percentile(bootstrap_means, 100.0 - alpha, axis=0)

    return {
        key: {"lower": float(lower[i]), "upper": float(upper[i])}
        for i, key in enumerate(METRIC_ORDER)
    }


def paired_bootstrap_differences(
    ar_metrics_map: Dict[str, Dict[str, float]],
    diff_metrics_map: Dict[str, Dict[str, float]],
    n_bootstrap: int = 5000,
    ci_level: float = 95.0,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], int]:
    """
    Compute paired mean differences (LLaDA - LLaMA) with paired bootstrap CIs.

    Only instance IDs present in both maps are used. Resampling is performed over
    shared instances, preserving the pairing between models.
    """
    shared_ids = sorted(set(ar_metrics_map.keys()) & set(diff_metrics_map.keys()))
    if not shared_ids:
        nan_metrics = {key: float("nan") for key in METRIC_ORDER}
        nan_cis = {key: {"lower": float("nan"), "upper": float("nan")} for key in METRIC_ORDER}
        return nan_metrics, nan_cis, 0

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci_level < 100.0):
        raise ValueError("ci_level must be between 0 and 100")

    diff_matrix = np.asarray(
        [
            [diff_metrics_map[idx][key] - ar_metrics_map[idx][key] for key in METRIC_ORDER]
            for idx in shared_ids
        ],
        dtype=float,
    )

    point_estimates = {
        key: float(diff_matrix[:, i].mean())
        for i, key in enumerate(METRIC_ORDER)
    }

    rng = np.random.default_rng(seed)
    n = diff_matrix.shape[0]
    bootstrap_means = np.empty((n_bootstrap, len(METRIC_ORDER)), dtype=float)

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = diff_matrix[sample_idx].mean(axis=0)

    alpha = (100.0 - ci_level) / 2.0
    lower = np.percentile(bootstrap_means, alpha, axis=0)
    upper = np.percentile(bootstrap_means, 100.0 - alpha, axis=0)

    cis = {
        key: {"lower": float(lower[i]), "upper": float(upper[i])}
        for i, key in enumerate(METRIC_ORDER)
    }
    return point_estimates, cis, n


def compute_metrics(
    filepath: Path,
    n_bootstrap: int = 5000,
    ci_level: float = 95.0,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], int, Dict[str, Dict[str, float]]]:
    """Compute scalar Task-5 summary metrics and bootstrap confidence intervals."""
    metrics_map = load_instance_metrics_map(filepath)
    instance_metrics = list(metrics_map.values())
    summary = summarize_instance_metrics(instance_metrics)
    cis = bootstrap_confidence_intervals(
        instance_metrics=instance_metrics,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    return summary, cis, len(instance_metrics), metrics_map


def _format_metric_line(key: str, value: float, ci: Dict[str, float], ci_level: float) -> str:
    if "Rate" in key:
        return (
            f"  {key}: {value * 100:.2f}% "
            f"[{ci_level:.0f}% bootstrap CI: {ci['lower'] * 100:.2f}%, {ci['upper'] * 100:.2f}%]"
        )
    return (
        f"  {key}: {value:.4f} "
        f"[{ci_level:.0f}% bootstrap CI: {ci['lower']:.4f}, {ci['upper']:.4f}]"
    )


def _format_delta_line(key: str, value: float, ci: Dict[str, float], ci_level: float) -> str:
    if "Rate" in key:
        return (
            f"  Δ {key} (LLaDA - LLaMA): {value * 100:+.2f} pp "
            f"[{ci_level:.0f}% paired bootstrap CI: {ci['lower'] * 100:+.2f}, {ci['upper'] * 100:+.2f} pp]"
        )
    return (
        f"  Δ {key} (LLaDA - LLaMA): {value:+.4f} "
        f"[{ci_level:.0f}% paired bootstrap CI: {ci['lower']:+.4f}, {ci['upper']:+.4f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Task-5 temporal semantic commitment metrics")
    parser.add_argument("--ar-file", type=Path, required=True, help="Path to AR trajectory JSON")
    parser.add_argument("--diff-file", type=Path, required=True, help="Path to diffusion trajectory JSON")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for confidence intervals (default: 5000)",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=95.0,
        help="Confidence level for percentile bootstrap intervals (default: 95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42)",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Optional JSON output path for summary metrics and confidence intervals",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("=== TEMPORAL SEMANTIC COMMITMENT METRICS ===")
    print("=" * 50)
    print(f"Bootstrap reps: {args.bootstrap_reps}")
    print(f"Confidence level: {args.ci_level:.1f}%")

    output_payload: Dict[str, Any] = {
        "metadata": {
            "bootstrap_reps": args.bootstrap_reps,
            "ci_level": args.ci_level,
            "seed": args.seed,
        },
        "models": {},
        "paired_differences": {},
    }

    results_cache: Dict[str, Dict[str, Any]] = {}

    for short_name, display_name, filepath in [
        ("llama", "Autoregressive (LLaMA-8B)", args.ar_file),
        ("llada", "Discrete Diffusion (LLaDA-8B)", args.diff_file),
    ]:
        metrics, cis, valid_count, metrics_map = compute_metrics(
            filepath=filepath,
            n_bootstrap=args.bootstrap_reps,
            ci_level=args.ci_level,
            seed=args.seed,
        )

        print(f"\n>>> {display_name}")
        print(f"  Valid trajectories aggregated: {valid_count}")

        for key in METRIC_ORDER:
            print(_format_metric_line(key, metrics[key], cis[key], args.ci_level))

        results_cache[short_name] = {
            "display_name": display_name,
            "metrics": metrics,
            "confidence_intervals": cis,
            "valid_trajectories": valid_count,
            "metrics_map": metrics_map,
            "input_file": str(filepath),
        }

        output_payload["models"][display_name] = {
            "input_file": str(filepath),
            "valid_trajectories": valid_count,
            "metrics": metrics,
            "confidence_intervals": cis,
        }

    delta_metrics, delta_cis, paired_n = paired_bootstrap_differences(
        ar_metrics_map=results_cache["llama"]["metrics_map"],
        diff_metrics_map=results_cache["llada"]["metrics_map"],
        n_bootstrap=args.bootstrap_reps,
        ci_level=args.ci_level,
        seed=args.seed,
    )

    print("\n>>> Paired model differences")
    print(f"  Shared trajectories compared: {paired_n}")
    for key in METRIC_ORDER:
        print(_format_delta_line(key, delta_metrics[key], delta_cis[key], args.ci_level))

    output_payload["paired_differences"] = {
        "definition": "LLaDA - LLaMA",
        "shared_trajectories": paired_n,
        "metrics": delta_metrics,
        "confidence_intervals": delta_cis,
    }

    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Saved metric summary with bootstrap CIs to: {args.out_file}")


if __name__ == "__main__":
    main()
