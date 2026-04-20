#!/usr/bin/env python3
"""
Plot Task 5: Temporal Semantic Commitment trajectories.

This module supports two paper-oriented outputs:
1. a main gold-condition comparison between autoregressive and diffusion models
2. optional appendix-style control panels when control files are provided

Supported input format:
    {"metadata": ..., "results": {id: {"trajectory": [...]}}}

Backward-compatible input formats:
    {"results": {id: [...]}} or {id: [...]}
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


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

        if isinstance(value, dict) and "trajectory" in value:
            trajectory = value.get("trajectory")
            if isinstance(trajectory, list) and trajectory:
                extracted.append(trajectory)
            continue

        if isinstance(value, list) and value:
            extracted.append(value)

    return extracted


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, png_path: Path) -> None:
    ensure_dir(png_path.parent)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[INFO] Saved figure: {png_path}")
    print(f"[INFO] Saved figure: {png_path.with_suffix('.pdf')}")


def load_and_interpolate(file_path: Path, num_points: int = 101) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a trajectory JSON and interpolate discrete entropy traces onto a
    standardized 0.0--1.0 progress axis.
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    trajectories = _extract_trajectory_records(raw_data)
    common_x = np.linspace(0.0, 1.0, num_points)
    all_interpolated_entropies: List[np.ndarray] = []

    for trajectory in trajectories:
        steps: List[float] = []
        entropies: List[float] = []
        for point in trajectory:
            if not isinstance(point, dict):
                continue
            if "step" not in point or "entropy" not in point:
                continue
            try:
                step = float(point["step"])
                entropy = float(point["entropy"])
            except Exception:
                continue
            if not math.isfinite(step) or not math.isfinite(entropy):
                continue
            steps.append(step)
            entropies.append(entropy)

        if not steps or len(steps) != len(entropies):
            continue

        max_step = max(steps)
        if max_step <= 0:
            continue

        norm_steps = np.array([step / max_step for step in steps], dtype=float)
        interp_y = np.interp(common_x, norm_steps, np.array(entropies, dtype=float))
        all_interpolated_entropies.append(interp_y)

    if not all_interpolated_entropies:
        return common_x, np.empty((0, num_points), dtype=float)

    return common_x, np.vstack(all_interpolated_entropies)


def compute_mean_and_band(matrix: np.ndarray, band_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    if band_mode == "std":
        band = std
    elif band_mode == "sem":
        band = std / math.sqrt(matrix.shape[0])
    else:
        band = np.zeros_like(std)

    return mean, std, band


def write_summary_csv(
    rows: Sequence[Tuple[str, str, float, float, float, float, int]],
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "panel",
            "series",
            "progress_pct",
            "mean_entropy",
            "std_entropy",
            "band_entropy",
            "num_examples",
        ])
        writer.writerows(rows)
    print(f"[INFO] Saved summary table: {out_path}")


def add_series(
    ax: plt.Axes,
    summary_rows: List[Tuple[str, str, float, float, float, float, int]],
    *,
    panel_name: str,
    series_name: str,
    progress: np.ndarray,
    matrix: np.ndarray,
    color: str,
    label: str,
    band_mode: str,
    linestyle: str = "-",
) -> None:
    if matrix.size == 0:
        return

    mean, std, band = compute_mean_and_band(matrix, band_mode)
    progress_pct = progress * 100.0

    ax.plot(
        progress_pct,
        mean,
        label=label,
        color=color,
        linewidth=2.2,
        linestyle=linestyle,
    )
    if band_mode != "none":
        lower = np.maximum(mean - band, 0.0)
        upper = np.minimum(mean + band, 1.0)
        ax.fill_between(progress_pct, lower, upper, color=color, alpha=0.15)

    for progress_value, mean_value, std_value, band_value in zip(progress_pct, mean, std, band):
        summary_rows.append((
            panel_name,
            series_name,
            float(progress_value),
            float(mean_value),
            float(std_value),
            float(band_value),
            int(matrix.shape[0]),
        ))


def finalize_axis(ax: plt.Axes, *, show_ylabel: bool) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Generation progress (%)")
    if show_ylabel:
        ax.set_ylabel("Semantic entropy (bits)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)


def plot_gold_comparison(args) -> None:
    series_rows: List[Tuple[str, str, float, float, float, float, int]] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if args.llama_file and args.llama_file.exists():
        progress, matrix = load_and_interpolate(args.llama_file, num_points=args.num_points)
        if matrix.size > 0:
            add_series(
                ax,
                series_rows,
                panel_name="gold",
                series_name="llama_gold",
                progress=progress,
                matrix=matrix,
                color="#c44e52",
                label="Autoregressive (LLaMA)",
                band_mode=args.band,
            )
        else:
            print(f"[WARN] No valid trajectories found in {args.llama_file}.")

    if args.llada_file and args.llada_file.exists():
        progress, matrix = load_and_interpolate(args.llada_file, num_points=args.num_points)
        if matrix.size > 0:
            add_series(
                ax,
                series_rows,
                panel_name="gold",
                series_name="llada_gold",
                progress=progress,
                matrix=matrix,
                color="#4c78a8",
                label="Discrete diffusion (LLaDA)",
                band_mode=args.band,
            )
        else:
            print(f"[WARN] No valid trajectories found in {args.llada_file}.")

    finalize_axis(ax, show_ylabel=True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out_file = args.output_dir / "temporal_semantic_commitment_comparison.png"
    save_figure(fig, out_file)
    plt.close(fig)

    if series_rows:
        write_summary_csv(
            series_rows,
            args.output_dir / "temporal_semantic_commitment_comparison.csv",
        )


def load_optional_series(file_path: Path | None, num_points: int) -> Tuple[np.ndarray | None, np.ndarray]:
    if file_path is None or not file_path.exists():
        return None, np.empty((0, num_points), dtype=float)
    progress, matrix = load_and_interpolate(file_path, num_points=num_points)
    return progress, matrix


def plot_control_panels(args) -> None:
    panel_specs = [
        (
            "Autoregressive (LLaMA)",
            [
                ("gold", args.llama_file, "#c44e52", "Gold", "-"),
                ("distractor_rewrite", args.llama_distractor_file, "#dd8452", "Distractor rewrite", "--"),
                ("random_matched_rewrite", args.llama_random_file, "#7f7f7f", "Random matched rewrite", ":"),
            ],
        ),
        (
            "Discrete diffusion (LLaDA)",
            [
                ("gold", args.llada_file, "#4c78a8", "Gold", "-"),
                ("distractor_rewrite", args.llada_distractor_file, "#72b7b2", "Distractor rewrite", "--"),
                ("random_matched_rewrite", args.llada_random_file, "#999999", "Random matched rewrite", ":"),
            ],
        ),
    ]

    has_any_control = any(
        file_path is not None and file_path.exists()
        for _, specs in panel_specs
        for series_name, file_path, _, _, _ in specs
        if series_name != "gold"
    )
    if not has_any_control:
        return

    available_panels: List[Tuple[str, List[Tuple[str, Path, str, str, str]]]] = []
    for title, specs in panel_specs:
        if any(
            file_path is not None and file_path.exists()
            for series_name, file_path, _, _, _ in specs
            if series_name != "gold"
        ):
            available_panels.append((title, specs))

    if not available_panels:
        return

    fig, axes = plt.subplots(1, len(available_panels), figsize=(6.2 * len(available_panels), 4.6), sharey=True)
    if len(available_panels) == 1:
        axes = [axes]

    series_rows: List[Tuple[str, str, float, float, float, float, int]] = []

    for ax, (panel_title, specs) in zip(axes, available_panels):
        for series_name, file_path, color, label, linestyle in specs:
            progress, matrix = load_optional_series(file_path, num_points=args.num_points)
            if progress is None or matrix.size == 0:
                continue
            add_series(
                ax,
                series_rows,
                panel_name=panel_title,
                series_name=series_name,
                progress=progress,
                matrix=matrix,
                color=color,
                label=label,
                band_mode=args.band,
                linestyle=linestyle,
            )

        ax.set_title(panel_title)
        finalize_axis(ax, show_ylabel=ax is axes[0])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    out_file = args.output_dir / "temporal_semantic_commitment_controls.png"
    save_figure(fig, out_file)
    plt.close(fig)

    if series_rows:
        write_summary_csv(
            series_rows,
            args.output_dir / "temporal_semantic_commitment_controls.csv",
        )


def run(args) -> int:
    ensure_dir(args.output_dir)

    if not args.llama_file and not args.llada_file:
        print("[ERROR] You must provide at least one trajectory file (--llama-file or --llada-file).")
        return 1

    plot_gold_comparison(args)
    plot_control_panels(args)
    return 0
