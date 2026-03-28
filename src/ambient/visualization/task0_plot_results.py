#!/usr/bin/env python3
"""
Visualize Task-0 and Task-2 results across LLaDA diffusion steps and Monte-Carlo
sample counts, with optional LLaMA baseline overlays.

Expected directory structure:
results/
  llada8b-n10-d2/
    metrics_summary_mc2.json
    metrics_summary_mc4.json
    ...
    task2_semantic_metrics.json
  llada8b-n10-d4/
    ...
  ...
  llama8b-n100/
    metrics_summary.json
    task2_semantic_metrics.json

Outputs:
results/plots/
  task0/by_diffusion_steps/*.png
  task0/by_mc/*.png
  task0/overviews/*.png
  task2/*.png
  tables/*.csv

The script intentionally avoids seaborn so it stays lightweight and consistent
with environments where only matplotlib/pandas are available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------
# Plot style / config
# ---------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

TASK0_SECTION_NAME_MAP = {
    "Unnormalized_Unfiltered (Strict Math)": "Unnormalized_Unfiltered",
    "Unnormalized_Cleaned (Math + Heuristic Filter)": "Unnormalized_Cleaned",
    "Normalized_Unfiltered (Length-Penalized)": "Normalized_Unfiltered",
    "Normalized_Cleaned (Original Script Baseline)": "Normalized_Cleaned",
}

TASK0_METRIC_NAME_MAP = {
    "Total Instances (Deduped)": "total_instances",
    "Total Evaluated (No generation failures)": "total_evaluated",
    "Artifact Rate (Garbage Generations)": "artifact_rate",
    "Ranking Accuracy (All valid < distractor)": "rank_acc_all",
    "Ranking Accuracy (Any valid < distractor)": "rank_acc_any",
    "Mean KL Divergence (Valid Options)": "mean_kl_valid",
    "Mean KL Divergence (Distractor)": "mean_kl_distractor",
}

TASK0_METRIC_LABELS = {
    "total_instances": "Total instances",
    "total_evaluated": "Total evaluated",
    "artifact_rate": "Artifact rate",
    "rank_acc_all": "KL rank accuracy (all valid < distractor)",
    "rank_acc_any": "KL rank accuracy (any valid < distractor)",
    "mean_kl_valid": "Mean KL divergence (valid options)",
    "mean_kl_distractor": "Mean KL divergence (distractor)",
}

TASK2_METRIC_LABELS = {
    "diversity_mean_cosine_dist": "Mean cosine distance (diversity)",
    "perplexity_median": "Median perplexity",
    "perplexity_mean": "Mean perplexity",
    "overlap_mean": "Mean lexical overlap",
    "num_evaluated_instances": "Evaluated instances",
}

SUMMARY_SECTIONS_TO_PLOT = [
    "Normalized_Cleaned",
    "Normalized_Unfiltered",
    "Unnormalized_Cleaned",
    "Unnormalized_Unfiltered",
]

SUMMARY_METRICS_TO_PLOT = [
    "rank_acc_all",
    "rank_acc_any",
    "artifact_rate",
    "mean_kl_valid",
    "mean_kl_distractor",
]

TASK2_METRICS_TO_PLOT = [
    "diversity_mean_cosine_dist",
    "perplexity_median",
    "perplexity_mean",
    "overlap_mean",
    "num_evaluated_instances",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Task0/Task2 result overviews")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Root results directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output plot directory (default: <results-dir>/plots)")
    parser.add_argument("--llada-pattern", type=str, default=r"llada8b-n10-d(\d+)", help="Regex for llada result dirs")
    parser.add_argument("--llama-dir", type=str, default="llama8b-n100", help="Baseline LLaMA directory name")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def discover_llada_dirs(results_dir: Path, pattern: str) -> List[Tuple[int, Path]]:
    regex = re.compile(pattern)
    found: List[Tuple[int, Path]] = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        match = regex.fullmatch(child.name)
        if match:
            found.append((int(match.group(1)), child))
    return sorted(found, key=lambda x: x[0])


def discover_mc_files(model_dir: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for path in model_dir.glob("metrics_summary_mc*.json"):
        match = re.search(r"_mc(\d+)\.json$", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda x: x[0])


def normalize_task0_summary(raw: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for raw_section, section_values in raw.items():
        section_name = TASK0_SECTION_NAME_MAP.get(raw_section, raw_section)
        if not isinstance(section_values, dict):
            continue
        normalized_metrics: Dict[str, float] = {}
        for raw_metric, value in section_values.items():
            metric_name = TASK0_METRIC_NAME_MAP.get(raw_metric, raw_metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized_metrics[metric_name] = float(value)
        if normalized_metrics:
            out[section_name] = normalized_metrics
    return out


def extract_task2_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    if not raw:
        return {}
    if len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        payload = next(iter(raw.values()))
    else:
        payload = raw
    out: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    return out


def build_task0_dataframe(llada_dirs: List[Tuple[int, Path]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for d_step, model_dir in llada_dirs:
        for mc, path in discover_mc_files(model_dir):
            raw = load_json(path)
            if not raw:
                continue
            norm = normalize_task0_summary(raw)
            for section, metrics in norm.items():
                row: Dict[str, Any] = {
                    "model": model_dir.name,
                    "diffusion_steps": d_step,
                    "mc": mc,
                    "section": section,
                }
                row.update(metrics)
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["section", "mc", "diffusion_steps"]).reset_index(drop=True)
    return df


def build_llama_task0_baseline(results_dir: Path, llama_dir_name: str) -> pd.DataFrame:
    path = results_dir / llama_dir_name / "metrics_summary.json"
    raw = load_json(path)
    if not raw:
        return pd.DataFrame()
    norm = normalize_task0_summary(raw)
    rows: List[Dict[str, Any]] = []
    for section, metrics in norm.items():
        row = {"model": llama_dir_name, "section": section}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def build_task2_dataframe(llada_dirs: List[Tuple[int, Path]], results_dir: Path, llama_dir_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    llada_rows: List[Dict[str, Any]] = []
    for d_step, model_dir in llada_dirs:
        raw = load_json(model_dir / "task2_semantic_metrics.json")
        if not raw:
            continue
        row = {"model": model_dir.name, "diffusion_steps": d_step}
        row.update(extract_task2_metrics(raw))
        llada_rows.append(row)

    baseline_raw = load_json(results_dir / llama_dir_name / "task2_semantic_metrics.json")
    baseline_df = pd.DataFrame()
    if baseline_raw:
        base_row = {"model": llama_dir_name}
        base_row.update(extract_task2_metrics(baseline_raw))
        baseline_df = pd.DataFrame([base_row])

    llada_df = pd.DataFrame(llada_rows)
    if not llada_df.empty:
        llada_df = llada_df.sort_values("diffusion_steps").reset_index(drop=True)
    return llada_df, baseline_df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved table: {path}")


def finite_series(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            x = float(v)
            if math.isfinite(x):
                out.append(x)
        except Exception:
            continue
    return out


def set_x_ticks(ax: plt.Axes, values: List[int], label: str) -> None:
    ax.set_xlabel(label)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])


def maybe_add_baseline_line(ax: plt.Axes, baseline_df: pd.DataFrame, section: str, metric: str, label: str = "LLaMA n=100") -> None:
    if baseline_df.empty:
        return
    subset = baseline_df[baseline_df["section"] == section]
    if subset.empty or metric not in subset.columns:
        return
    val = subset.iloc[0][metric]
    if pd.isna(val):
        return
    ax.axhline(float(val), linestyle="--", linewidth=1.5, alpha=0.8, label=label)


def plot_task0_by_diffusion_steps(df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if df.empty:
        print("[WARN] No Task-0 data found for diffusion-step plots.")
        return

    diffusion_values = sorted(df["diffusion_steps"].dropna().astype(int).unique().tolist())
    mc_values = sorted(df["mc"].dropna().astype(int).unique().tolist())

    for section in SUMMARY_SECTIONS_TO_PLOT:
        section_df = df[df["section"] == section]
        if section_df.empty:
            continue

        for metric in SUMMARY_METRICS_TO_PLOT:
            if metric not in section_df.columns:
                continue
            values = finite_series(section_df[metric].tolist())
            if not values:
                continue

            fig, ax = plt.subplots(figsize=(9, 5.2))
            for mc in mc_values:
                subset = section_df[section_df["mc"] == mc].sort_values("diffusion_steps")
                if subset.empty or subset[metric].isna().all():
                    continue
                ax.plot(
                    subset["diffusion_steps"],
                    subset[metric],
                    marker="o",
                    linewidth=2,
                    label=f"mc={mc}",
                )

            maybe_add_baseline_line(ax, baseline_df, section, metric)
            ax.set_title(f"{TASK0_METRIC_LABELS[metric]} across diffusion steps\n[{section}]")
            ax.set_ylabel(TASK0_METRIC_LABELS[metric])
            set_x_ticks(ax, diffusion_values, "Diffusion steps")
            ax.legend(ncol=3, frameon=True)
            fig.tight_layout()
            out_path = out_dir / f"{section}__{metric}__by_diffusion_steps.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved plot: {out_path}")


def plot_task0_by_mc(df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if df.empty:
        print("[WARN] No Task-0 data found for mc plots.")
        return

    mc_values = sorted(df["mc"].dropna().astype(int).unique().tolist())
    diffusion_values = sorted(df["diffusion_steps"].dropna().astype(int).unique().tolist())

    for section in SUMMARY_SECTIONS_TO_PLOT:
        section_df = df[df["section"] == section]
        if section_df.empty:
            continue

        for metric in SUMMARY_METRICS_TO_PLOT:
            if metric not in section_df.columns:
                continue
            values = finite_series(section_df[metric].tolist())
            if not values:
                continue

            fig, ax = plt.subplots(figsize=(9, 5.2))
            for d_step in diffusion_values:
                subset = section_df[section_df["diffusion_steps"] == d_step].sort_values("mc")
                if subset.empty or subset[metric].isna().all():
                    continue
                ax.plot(
                    subset["mc"],
                    subset[metric],
                    marker="o",
                    linewidth=2,
                    label=f"d={d_step}",
                )

            maybe_add_baseline_line(ax, baseline_df, section, metric)
            ax.set_title(f"{TASK0_METRIC_LABELS[metric]} across mc values\n[{section}]")
            ax.set_ylabel(TASK0_METRIC_LABELS[metric])
            set_x_ticks(ax, mc_values, "Monte-Carlo samples (mc)")
            ax.legend(ncol=3, frameon=True)
            fig.tight_layout()
            out_path = out_dir / f"{section}__{metric}__by_mc.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved plot: {out_path}")


def plot_task0_overview_grids(df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if df.empty:
        return

    diffusion_values = sorted(df["diffusion_steps"].dropna().astype(int).unique().tolist())
    mc_values = sorted(df["mc"].dropna().astype(int).unique().tolist())

    # One 2x3 grid per section over diffusion steps
    for section in SUMMARY_SECTIONS_TO_PLOT:
        section_df = df[df["section"] == section]
        if section_df.empty:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        for idx, metric in enumerate(SUMMARY_METRICS_TO_PLOT):
            ax = axes[idx]
            if metric not in section_df.columns:
                ax.axis("off")
                continue
            for mc in mc_values:
                subset = section_df[section_df["mc"] == mc].sort_values("diffusion_steps")
                if subset.empty or subset[metric].isna().all():
                    continue
                ax.plot(subset["diffusion_steps"], subset[metric], marker="o", linewidth=1.8, label=f"mc={mc}")
            maybe_add_baseline_line(ax, baseline_df, section, metric)
            ax.set_title(TASK0_METRIC_LABELS[metric])
            set_x_ticks(ax, diffusion_values, "Diffusion steps")

        axes[-1].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=True)
        fig.suptitle(f"Task-0 overview across diffusion steps [{section}]", y=0.98)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        out_path = out_dir / f"{section}__overview_by_diffusion_steps.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_path}")


def plot_task2(llada_df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if llada_df.empty:
        print("[WARN] No Task-2 data found.")
        return

    diffusion_values = sorted(llada_df["diffusion_steps"].dropna().astype(int).unique().tolist())

    for metric in TASK2_METRICS_TO_PLOT:
        if metric not in llada_df.columns:
            continue
        values = finite_series(llada_df[metric].tolist())
        if not values:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(llada_df["diffusion_steps"], llada_df[metric], marker="o", linewidth=2.2, label="LLaDA n=10")

        if not baseline_df.empty and metric in baseline_df.columns and not pd.isna(baseline_df.iloc[0][metric]):
            base_val = float(baseline_df.iloc[0][metric])
            ax.axhline(base_val, linestyle="--", linewidth=1.5, alpha=0.85, label="LLaMA n=100")

        ax.set_title(f"Task-2: {TASK2_METRIC_LABELS[metric]} across diffusion steps")
        ax.set_ylabel(TASK2_METRIC_LABELS[metric])
        set_x_ticks(ax, diffusion_values, "Diffusion steps")
        ax.legend(frameon=True)
        fig.tight_layout()
        out_path = out_dir / f"task2__{metric}__by_diffusion_steps.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_path}")

    # Combined grid for Task-2
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, metric in enumerate(TASK2_METRICS_TO_PLOT):
        ax = axes[idx]
        if metric not in llada_df.columns:
            ax.axis("off")
            continue
        values = finite_series(llada_df[metric].tolist())
        if not values:
            ax.axis("off")
            continue
        ax.plot(llada_df["diffusion_steps"], llada_df[metric], marker="o", linewidth=2.0, label="LLaDA n=10")
        if not baseline_df.empty and metric in baseline_df.columns and not pd.isna(baseline_df.iloc[0][metric]):
            base_val = float(baseline_df.iloc[0][metric])
            ax.axhline(base_val, linestyle="--", linewidth=1.5, alpha=0.85, label="LLaMA n=100")
        ax.set_title(TASK2_METRIC_LABELS[metric])
        set_x_ticks(ax, diffusion_values, "Diffusion steps")

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=True)
    fig.suptitle("Task-2 overview across diffusion steps", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_path = out_dir / "task2__overview_by_diffusion_steps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_path}")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "plots")

    ensure_dir(out_dir)
    ensure_dir(out_dir / "task0" / "by_diffusion_steps")
    ensure_dir(out_dir / "task0" / "by_mc")
    ensure_dir(out_dir / "task0" / "overviews")
    ensure_dir(out_dir / "task2")
    ensure_dir(out_dir / "tables")

    llada_dirs = discover_llada_dirs(results_dir, args.llada_pattern)
    if not llada_dirs:
        print(f"[ERROR] No LLaDA directories found in {results_dir} with pattern {args.llada_pattern!r}")
        return

    print("[INFO] Found LLaDA dirs:")
    for d_step, path in llada_dirs:
        print(f"  - d={d_step}: {path}")

    task0_df = build_task0_dataframe(llada_dirs)
    llama_task0_df = build_llama_task0_baseline(results_dir, args.llama_dir)
    task2_llada_df, task2_llama_df = build_task2_dataframe(llada_dirs, results_dir, args.llama_dir)

    save_dataframe(task0_df, out_dir / "tables" / "task0_llada_long.csv")
    if not llama_task0_df.empty:
        save_dataframe(llama_task0_df, out_dir / "tables" / "task0_llama_baseline.csv")
    save_dataframe(task2_llada_df, out_dir / "tables" / "task2_llada.csv")
    if not task2_llama_df.empty:
        save_dataframe(task2_llama_df, out_dir / "tables" / "task2_llama_baseline.csv")

    plot_task0_by_diffusion_steps(task0_df, llama_task0_df, out_dir / "task0" / "by_diffusion_steps")
    plot_task0_by_mc(task0_df, llama_task0_df, out_dir / "task0" / "by_mc")
    plot_task0_overview_grids(task0_df, llama_task0_df, out_dir / "task0" / "overviews")
    plot_task2(task2_llada_df, task2_llama_df, out_dir / "task2")

    print(f"\n[INFO] Done. All plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
