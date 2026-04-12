#!/usr/bin/env python3
"""
Plot combined Task-4 probing results with additional von Neumann entropy plots.

Supported input structure:
- config
- datasets
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
- von_neumann_entropy (optional)
    - llama
    - llada
        - <layer_idx>
            - ambiguous_raw_entropy_bits_mean
            - ambiguous_normalized_entropy_mean
            - disambiguated_raw_entropy_bits_mean
            - disambiguated_normalized_entropy_mean
            - delta_raw_entropy_bits_mean
            - delta_normalized_entropy_mean
            - ...

This script creates:
- one accuracy plot per dataset mode
- one probe-entropy plot per dataset mode
- one model-difference plot per dataset mode (LLaDA - LLaMA)
- VNE plots for ambiguous vs. disambiguated inputs
- VNE delta plots comparing models
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
    "von_neumann_entropy": "Von Neumann entropy comparison dataset",
}

MODEL_LABELS = {
    "llama": "LLaMA-3.1-8B",
    "llada": "LLaDA-8B",
}

VNE_VARIANT_LABELS = {
    "ambiguous": "Ambiguous",
    "disambiguated": "Disambiguated",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Generic probing helpers
# -----------------------------------------------------------------------------

def extract_metric_series(model_results: Dict[str, dict], metric_key: str) -> Tuple[List[int], np.ndarray]:
    layers = sorted(int(k) for k in model_results.keys())
    values = np.array([model_results[str(layer)][metric_key] for layer in layers], dtype=np.float64)
    return layers, values


def safe_std_series(model_results: Dict[str, dict], metric_key: str) -> np.ndarray | None:
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


# -----------------------------------------------------------------------------
# Probing plots
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# VNE helpers and plots
# -----------------------------------------------------------------------------

def extract_vne_series(model_results: Dict[str, dict], metric_key: str) -> Tuple[List[int], np.ndarray]:
    layers = sorted(int(k) for k in model_results.keys())
    values = np.array([model_results[str(layer)][metric_key] for layer in layers], dtype=np.float64)
    return layers, values


def plot_vne_within_model(
    out_path: Path,
    title: str,
    ylabel: str,
    model_label: str,
    model_results: Dict[str, dict],
    ambiguous_metric_key: str,
    disambiguated_metric_key: str,
    ambiguous_std_key: str | None = None,
    disambiguated_std_key: str | None = None,
    show_error_band: bool = True,
) -> None:
    layers, ambiguous_values = extract_vne_series(model_results, ambiguous_metric_key)
    layers2, disambiguated_values = extract_vne_series(model_results, disambiguated_metric_key)
    if layers != layers2:
        raise ValueError("Layer sets differ within VNE model plot.")

    plt.figure(figsize=(10, 6))
    plt.plot(layers, ambiguous_values, marker="o", label=f"{model_label} — {VNE_VARIANT_LABELS['ambiguous']}")
    plt.plot(layers, disambiguated_values, marker="o", label=f"{model_label} — {VNE_VARIANT_LABELS['disambiguated']}")

    if show_error_band and ambiguous_std_key is not None and disambiguated_std_key is not None:
        amb_std = np.array([model_results[str(layer)][ambiguous_std_key] for layer in layers], dtype=np.float64)
        dis_std = np.array([model_results[str(layer)][disambiguated_std_key] for layer in layers], dtype=np.float64)
        plt.fill_between(layers, ambiguous_values - amb_std, ambiguous_values + amb_std, alpha=0.18)
        plt.fill_between(layers, disambiguated_values - dis_std, disambiguated_values + dis_std, alpha=0.18)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_vne_delta_two_models(
    out_path: Path,
    title: str,
    ylabel: str,
    llama_results: Dict[str, dict],
    llada_results: Dict[str, dict],
    metric_key: str,
    std_key: str | None = None,
    show_error_band: bool = True,
) -> None:
    layers, llama_values = extract_vne_series(llama_results, metric_key)
    layers2, llada_values = extract_vne_series(llada_results, metric_key)
    if layers != layers2:
        raise ValueError("Layer sets differ between llama and llada VNE results.")

    diff = llada_values - llama_values

    plt.figure(figsize=(10, 6))
    plt.axhline(0.0, linewidth=1)
    plt.plot(layers, diff, marker="o")

    if show_error_band and std_key is not None:
        llama_std = np.array([llama_results[str(layer)][std_key] for layer in layers], dtype=np.float64)
        llada_std = np.array([llada_results[str(layer)][std_key] for layer in layers], dtype=np.float64)
        combined_std = np.sqrt(llama_std ** 2 + llada_std ** 2)
        plt.fill_between(layers, diff - combined_std, diff + combined_std, alpha=0.18)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

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

    lines.append(f"  Largest accuracy advantage for LLaDA: layer {best_llada_acc_layer} ({diff_acc.max():+.4f})")
    lines.append(f"  Largest accuracy advantage for LLaMA: layer {best_llama_acc_layer} ({diff_acc.min():+.4f})")
    lines.append(f"  Largest probe-entropy advantage for LLaDA: layer {best_llada_ent_layer} ({diff_ent.max():+.4f} bits)")
    lines.append(f"  Largest probe-entropy advantage for LLaMA: layer {best_llama_ent_layer} ({diff_ent.min():+.4f} bits)")

    early_slice = slice(0, min(8, len(llama_layers)))
    mid_start = len(llama_layers) // 3
    mid_end = (2 * len(llama_layers)) // 3
    mid_slice = slice(mid_start, mid_end)
    late_slice = slice(max(len(llama_layers) - 8, 0), len(llama_layers))

    for name, slc in (("early", early_slice), ("middle", mid_slice), ("late", late_slice)):
        lines.append(f"  Mean accuracy Δ (LLaDA-LLaMA), {name}: {np.mean(diff_acc[slc]):+.4f}")
        lines.append(f"  Mean probe entropy Δ (LLaDA-LLaMA), {name}: {np.mean(diff_ent[slc]):+.4f} bits")

    lines.append("")
    return lines


def summarize_vne(vne_results: dict) -> List[str]:
    lines: List[str] = []
    lines.append("Von Neumann entropy summary:")

    if not vne_results:
        lines.append("  No VNE results available.")
        lines.append("")
        return lines

    for model_key in ("llama", "llada"):
        if model_key not in vne_results or not vne_results[model_key]:
            continue

        model_results = vne_results[model_key]
        layers = sorted(int(k) for k in model_results.keys())

        max_abs_delta_layer = max(
            layers,
            key=lambda x: abs(model_results[str(x)]["delta_normalized_entropy_mean"]),
        )
        final_layer = layers[-1]
        middle_layer = layers[len(layers) // 2]

        for layer_name, layer_idx in (("middle", middle_layer), ("final", final_layer), ("max |Δ|", max_abs_delta_layer)):
            entry = model_results[str(layer_idx)]
            lines.append(
                f"  {MODEL_LABELS[model_key]} {layer_name} layer {layer_idx}: "
                f"H_amb={entry['ambiguous_normalized_entropy_mean']:.4f}, "
                f"H_dis={entry['disambiguated_normalized_entropy_mean']:.4f}, "
                f"Δ={entry['delta_normalized_entropy_mean']:+.4f}"
            )

    if "llama" in vne_results and "llada" in vne_results and vne_results["llama"] and vne_results["llada"]:
        layers, llama_delta = extract_vne_series(vne_results["llama"], "delta_normalized_entropy_mean")
        layers2, llada_delta = extract_vne_series(vne_results["llada"], "delta_normalized_entropy_mean")
        if layers == layers2:
            diff = llada_delta - llama_delta
            lines.append(
                f"  Largest model gap in normalized VNE delta (LLaDA-LLaMA): "
                f"layer {layers[int(np.argmax(np.abs(diff)))]} ({diff[np.argmax(np.abs(diff))]:+.4f})"
            )

    lines.append("")
    return lines


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot combined Task-4 probing + VNE results.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the combined JSON results file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summary.")
    parser.add_argument("--no-error-band", action="store_true", help="Disable shaded error bands.")
    args = parser.parse_args()

    data = load_json(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: List[str] = []
    summary_lines.append("Task 4 combined layerwise probing + von Neumann entropy summary")
    summary_lines.append("")

    if "config" in data:
        summary_lines.append("Config:")
        for k, v in data["config"].items():
            summary_lines.append(f"  {k}: {v}")
        summary_lines.append("")

    datasets = data.get("datasets", {})
    results = data.get("results", {})

    # Probing plots
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

    # VNE plots
    vne_results = data.get("von_neumann_entropy", {})
    if vne_results:
        if "von_neumann_entropy" in datasets:
            summary_lines.append("Dataset metadata for von_neumann_entropy:")
            for k, v in datasets["von_neumann_entropy"].items():
                summary_lines.append(f"  {k}: {v}")
            summary_lines.append("")

        for model_key in ("llama", "llada"):
            if model_key not in vne_results or not vne_results[model_key]:
                continue

            plot_vne_within_model(
                out_path=args.output_dir / f"vne_{model_key}_raw_entropy.png",
                title=f"Von Neumann entropy (raw) — {MODEL_LABELS[model_key]}",
                ylabel="Von Neumann entropy (bits)",
                model_label=MODEL_LABELS[model_key],
                model_results=vne_results[model_key],
                ambiguous_metric_key="ambiguous_raw_entropy_bits_mean",
                disambiguated_metric_key="disambiguated_raw_entropy_bits_mean",
                ambiguous_std_key="ambiguous_raw_entropy_bits_std",
                disambiguated_std_key="disambiguated_raw_entropy_bits_std",
                show_error_band=not args.no_error_band,
            )
            plot_vne_within_model(
                out_path=args.output_dir / f"vne_{model_key}_normalized_entropy.png",
                title=f"Von Neumann entropy (normalized) — {MODEL_LABELS[model_key]}",
                ylabel="Normalized von Neumann entropy",
                model_label=MODEL_LABELS[model_key],
                model_results=vne_results[model_key],
                ambiguous_metric_key="ambiguous_normalized_entropy_mean",
                disambiguated_metric_key="disambiguated_normalized_entropy_mean",
                ambiguous_std_key="ambiguous_normalized_entropy_std",
                disambiguated_std_key="disambiguated_normalized_entropy_std",
                show_error_band=not args.no_error_band,
            )

        if "llama" in vne_results and "llada" in vne_results and vne_results["llama"] and vne_results["llada"]:
            plot_vne_delta_two_models(
                out_path=args.output_dir / "vne_delta_raw_entropy_difference.png",
                title="Raw VNE delta difference (LLaDA - LLaMA)",
                ylabel="Δ raw entropy difference (bits)",
                llama_results=vne_results["llama"],
                llada_results=vne_results["llada"],
                metric_key="delta_raw_entropy_bits_mean",
                std_key="delta_raw_entropy_bits_std",
                show_error_band=not args.no_error_band,
            )
            plot_vne_delta_two_models(
                out_path=args.output_dir / "vne_delta_normalized_entropy_difference.png",
                title="Normalized VNE delta difference (LLaDA - LLaMA)",
                ylabel="Δ normalized entropy difference",
                llama_results=vne_results["llama"],
                llada_results=vne_results["llada"],
                metric_key="delta_normalized_entropy_mean",
                std_key="delta_normalized_entropy_std",
                show_error_band=not args.no_error_band,
            )

        summary_lines.extend(summarize_vne(vne_results))

    summary_path = args.output_dir / "combined_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).rstrip() + "\n")

    print(f"Saved plots and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
