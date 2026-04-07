#!/usr/bin/env python3
"""
Plot layerwise Task-4 probing results from the upgraded JSON output.

Expected input structure:
- config
- datasets
- weight_entropy
- results
    - <dataset_mode>
        - llama
        - llada
            - <layer_idx>
                - mean_accuracy
                - std_accuracy
                - mean_probe_entropy_bits
                - std_probe_entropy_bits
                - ...

This script creates:
- one accuracy plot per dataset mode
- one probe-entropy plot per dataset mode
- one model-difference plot per dataset mode (LLaDA - LLaMA)
- one optional bar plot for global weight entropy
- one textual summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATASET_MODE_LABELS = {
    "side_reconstructed": "Side-reconstructed disambiguations",
    "fully_disambiguated": "Fully disambiguated pairs",
}

MODEL_LABELS = {
    "llama": "LLaMA-3.1-8B",
    "llada": "LLaDA-8B",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def extract_metric_series(model_results: Dict[str, dict], metric_key: str) -> Tuple[List[int], np.ndarray]:
    layers = sorted(int(k) for k in model_results.keys())
    values = np.array([model_results[str(layer)][metric_key] for layer in layers], dtype=np.float64)
    return layers, values



def safe_std_series(model_results: Dict[str, dict], metric_key: str) -> np.ndarray | None:
    # accuracy has std_accuracy; entropy has std_probe_entropy_bits
    if metric_key == "mean_accuracy":
        std_key = "std_accuracy"
    elif metric_key == "mean_probe_entropy_bits":
        std_key = "std_probe_entropy_bits"
    else:
        return None

    try:
        layers = sorted(int(k) for k in model_results.keys())
        return np.array([model_results[str(layer)][std_key] for layer in layers], dtype=np.float64)
    except KeyError:
        return None



def plot_two_model_curves(
    out_path: Path,
    title: str,
    ylabel: str,
    llama_results: Dict[str, dict],
    llada_results: Dict[str, dict],
    metric_key: str,
    show_error_band: bool,
) -> None:
    llama_layers, llama_values = extract_metric_series(llama_results, metric_key)
    llada_layers, llada_values = extract_metric_series(llada_results, metric_key)

    plt.figure(figsize=(10, 6))
    plt.plot(llama_layers, llama_values, marker="o", label=MODEL_LABELS["llama"])
    plt.plot(llada_layers, llada_values, marker="o", label=MODEL_LABELS["llada"])

    if show_error_band:
        llama_std = safe_std_series(llama_results, metric_key)
        llada_std = safe_std_series(llada_results, metric_key)
        if llama_std is not None:
            plt.fill_between(llama_layers, llama_values - llama_std, llama_values + llama_std, alpha=0.18)
        if llada_std is not None:
            plt.fill_between(llada_layers, llada_values - llada_std, llada_values + llada_std, alpha=0.18)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_difference_curve(
    out_path: Path,
    title: str,
    ylabel: str,
    llama_results: Dict[str, dict],
    llada_results: Dict[str, dict],
    metric_key: str,
) -> None:
    llama_layers, llama_values = extract_metric_series(llama_results, metric_key)
    llada_layers, llada_values = extract_metric_series(llada_results, metric_key)

    if llama_layers != llada_layers:
        raise ValueError("Layer sets differ between llama and llada; cannot plot difference.")

    diff = llada_values - llama_values

    plt.figure(figsize=(10, 6))
    plt.axhline(0.0, linewidth=1)
    plt.plot(llama_layers, diff, marker="o")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_weight_entropy(out_path: Path, weight_entropy: dict) -> None:
    models = []
    values = []
    for model_key in ("llama", "llada"):
        if model_key in weight_entropy and "histogram_entropy_bits" in weight_entropy[model_key]:
            models.append(MODEL_LABELS[model_key])
            values.append(weight_entropy[model_key]["histogram_entropy_bits"])

    if not models:
        return

    plt.figure(figsize=(7, 5))
    plt.bar(models, values)
    plt.ylabel("Approx. histogram entropy (bits)")
    plt.title("Approximate global weight entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def summarize_mode(mode: str, mode_results: dict) -> List[str]:
    lines: List[str] = []
    lines.append(f"Dataset mode: {mode} ({DATASET_MODE_LABELS.get(mode, mode)})")

    for model_key in ("llama", "llada"):
        model_results = mode_results[model_key]
        layers = sorted(int(k) for k in model_results.keys())
        best_acc_layer = max(layers, key=lambda x: model_results[str(x)]["mean_accuracy"])
        lowest_entropy_layer = min(layers, key=lambda x: model_results[str(x)]["mean_probe_entropy_bits"])
        lines.append(
            f"  {MODEL_LABELS[model_key]}: best accuracy at layer {best_acc_layer} "
            f"({model_results[str(best_acc_layer)]['mean_accuracy']:.4f}), "
            f"lowest probe entropy at layer {lowest_entropy_layer} "
            f"({model_results[str(lowest_entropy_layer)]['mean_probe_entropy_bits']:.4f} bits)"
        )

    llama_layers, llama_acc = extract_metric_series(mode_results["llama"], "mean_accuracy")
    _, llada_acc = extract_metric_series(mode_results["llada"], "mean_accuracy")
    _, llama_ent = extract_metric_series(mode_results["llama"], "mean_probe_entropy_bits")
    _, llada_ent = extract_metric_series(mode_results["llada"], "mean_probe_entropy_bits")

    diff_acc = llada_acc - llama_acc
    diff_ent = llada_ent - llama_ent

    best_llada_acc_layer = llama_layers[int(np.argmax(diff_acc))]
    best_llama_acc_layer = llama_layers[int(np.argmin(diff_acc))]
    best_llada_ent_layer = llama_layers[int(np.argmax(diff_ent))]
    best_llama_ent_layer = llama_layers[int(np.argmin(diff_ent))]

    lines.append(
        f"  Largest accuracy advantage for LLaDA: layer {best_llada_acc_layer} ({diff_acc.max():+.4f})"
    )
    lines.append(
        f"  Largest accuracy advantage for LLaMA: layer {best_llama_acc_layer} ({diff_acc.min():+.4f})"
    )
    lines.append(
        f"  Largest probe-entropy advantage for LLaDA: layer {best_llada_ent_layer} ({diff_ent.max():+.4f} bits)"
    )
    lines.append(
        f"  Largest probe-entropy advantage for LLaMA: layer {best_llama_ent_layer} ({diff_ent.min():+.4f} bits)"
    )

    early_slice = slice(0, min(8, len(llama_layers)))
    mid_start = len(llama_layers) // 3
    mid_end = (2 * len(llama_layers)) // 3
    mid_slice = slice(mid_start, mid_end)
    late_slice = slice(max(len(llama_layers) - 8, 0), len(llama_layers))

    for name, slc in (("early", early_slice), ("middle", mid_slice), ("late", late_slice)):
        lines.append(
            f"  Mean accuracy Δ (LLaDA-LLaMA), {name}: {np.mean(diff_acc[slc]):+.4f}"
        )
        lines.append(
            f"  Mean probe entropy Δ (LLaDA-LLaMA), {name}: {np.mean(diff_ent[slc]):+.4f} bits"
        )

    lines.append("")
    return lines



def main() -> None:
    parser = argparse.ArgumentParser(description="Plot upgraded Task-4 layerwise probing results.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the upgraded JSON results file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summary.")
    parser.add_argument("--no-error-band", action="store_true", help="Disable ±1 std shaded bands.")
    args = parser.parse_args()

    data = load_json(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: List[str] = []
    summary_lines.append("Task 4 upgraded layerwise probing summary")
    summary_lines.append("")

    if "config" in data:
        summary_lines.append("Config:")
        for k, v in data["config"].items():
            summary_lines.append(f"  {k}: {v}")
        summary_lines.append("")

    datasets = data.get("datasets", {})
    results = data.get("results", {})

    for mode, mode_results in results.items():
        mode_label = DATASET_MODE_LABELS.get(mode, mode)
        llama_results = mode_results["llama"]
        llada_results = mode_results["llada"]

        plot_two_model_curves(
            out_path=args.output_dir / f"{mode}_accuracy.png",
            title=f"Layerwise probe accuracy — {mode_label}",
            ylabel="Accuracy",
            llama_results=llama_results,
            llada_results=llada_results,
            metric_key="mean_accuracy",
            show_error_band=not args.no_error_band,
        )
        plot_two_model_curves(
            out_path=args.output_dir / f"{mode}_probe_entropy.png",
            title=f"Layerwise probe entropy — {mode_label}",
            ylabel="Probe entropy (bits)",
            llama_results=llama_results,
            llada_results=llada_results,
            metric_key="mean_probe_entropy_bits",
            show_error_band=not args.no_error_band,
        )
        plot_difference_curve(
            out_path=args.output_dir / f"{mode}_accuracy_difference.png",
            title=f"Accuracy difference (LLaDA - LLaMA) — {mode_label}",
            ylabel="Accuracy difference",
            llama_results=llama_results,
            llada_results=llada_results,
            metric_key="mean_accuracy",
        )
        plot_difference_curve(
            out_path=args.output_dir / f"{mode}_probe_entropy_difference.png",
            title=f"Probe entropy difference (LLaDA - LLaMA) — {mode_label}",
            ylabel="Entropy difference (bits)",
            llama_results=llama_results,
            llada_results=llada_results,
            metric_key="mean_probe_entropy_bits",
        )

        if mode in datasets:
            summary_lines.append(f"Dataset metadata for {mode}:")
            for k, v in datasets[mode].items():
                summary_lines.append(f"  {k}: {v}")
            summary_lines.append("")

        summary_lines.extend(summarize_mode(mode, mode_results))

    weight_entropy = data.get("weight_entropy", {})
    if weight_entropy:
        plot_weight_entropy(args.output_dir / "weight_entropy.png", weight_entropy)
        summary_lines.append("Global approximate weight entropy:")
        for model_key in ("llama", "llada"):
            if model_key in weight_entropy:
                entry = weight_entropy[model_key]
                summary_lines.append(
                    f"  {MODEL_LABELS[model_key]}: {entry.get('histogram_entropy_bits', float('nan')):.4f} bits "
                    f"(sampled {entry.get('sampled_weights', 0):,} / {entry.get('total_weights', 0):,} weights, "
                    f"bins={entry.get('num_bins', 'NA')}, stride={entry.get('global_stride', 'NA')})"
                )
        summary_lines.append("")

    summary_path = args.output_dir / "upgraded_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).rstrip() + "\n")

    print(f"Saved plots and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
