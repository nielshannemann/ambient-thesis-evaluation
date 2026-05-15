#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_GRAPHICS = ROOT / "paper" / "graphics"


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def save_figure(fig: plt.Figure, stem: str) -> None:
    PAPER_GRAPHICS.mkdir(parents=True, exist_ok=True)
    pdf_path = PAPER_GRAPHICS / f"{stem}.pdf"
    png_path = PAPER_GRAPHICS / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.015)
    print(f"[INFO] wrote {pdf_path}")
    print(f"[INFO] wrote {png_path}")


def external_pythia_ppl() -> dict[str, float]:
    path = RESULTS / "robustness" / "task2_external_ppl" / "pythia410m_ppl_summary.json"
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    out: dict[str, float] = {}
    for model, payload in raw.items():
        ppl = as_float(payload.get("perplexity_median"))
        if math.isfinite(ppl):
            out[model] = ppl
    return out


def plot_task1_task2_tradeoff() -> None:
    task1_llada = read_csv_rows(RESULTS / "plots" / "tables" / "task1_llada_long.csv")
    task1_llama = read_csv_rows(RESULTS / "plots" / "tables" / "task1_llama_baseline.csv")
    task2_llada = read_csv_rows(RESULTS / "plots" / "tables" / "task2_llada.csv")
    task2_llama = read_csv_rows(RESULTS / "plots" / "tables" / "task2_llama_baseline.csv")
    pythia = external_pythia_ppl()

    ppl_by_model = {}
    for row in [*task2_llada, *task2_llama]:
        model = row["model"]
        main_ppl = as_float(row["perplexity_median"])
        ext_ppl = pythia.get(model, main_ppl)
        ppl_by_model[model] = (main_ppl + ext_ppl) / 2.0

    llada_points: list[tuple[int, float, float]] = []
    for row in task1_llada:
        if row["section"] != "Normalized_Cleaned" or int(float(row["mc"])) != 256:
            continue
        model = row["model"]
        step = int(float(row["diffusion_steps"]))
        ppl = ppl_by_model.get(model)
        acc = as_float(row["rank_acc_all"])
        if ppl is not None and math.isfinite(acc):
            llada_points.append((step, ppl, acc))
    llada_points.sort(key=lambda item: item[1])

    llama_points: list[tuple[str, float, float]] = []
    for row in task1_llama:
        if row["section"] != "Normalized_Cleaned":
            continue
        model = row["model"]
        ppl = ppl_by_model.get(model)
        acc = as_float(row["rank_acc_all"])
        if ppl is not None and math.isfinite(acc):
            label = "LLaMA temp=1.0" if model == "llama8b-n100" else "LLaMA temp=2.0"
            llama_points.append((label, ppl, acc))

    fig, ax = plt.subplots(figsize=(3.35, 2.22), constrained_layout=True)

    xs = [point[1] for point in llada_points]
    ys = [point[2] for point in llada_points]
    ax.plot(xs, ys, color="#2f6f9f", linewidth=1.55, alpha=0.95)
    ax.scatter(
        xs,
        ys,
        s=28,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
        label=r"LLaDA ($K=256$)",
    )

    label_offsets = {
        128: (5, -5),
        64: (5, 5),
        32: (5, 0),
        16: (5, 0),
        8: (-15, 5),
        2: (5, 6),
        4: (5, -10),
    }
    for step, x_val, y_val in llada_points:
        dx, dy = label_offsets.get(step, (4, 4))
        ax.annotate(
            f"T={step}",
            xy=(x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.8,
            color="#1f4f73",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
            zorder=4,
        )

    llama_styles = {
        "LLaMA temp=1.0": ("#d6852a", "^", 54),
        "LLaMA temp=2.0": ("#b73b3c", "s", 38),
    }
    for label, ppl, acc in llama_points:
        color, marker, size = llama_styles[label]
        ax.scatter(
            [ppl],
            [acc],
            s=size,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlim(15, 140000)
    ax.set_ylim(0.56, 0.88)
    ax.set_xlabel("Task 2 median PPL (2-scorer mean)")
    ax.set_ylabel("Task 1 rank accuracy")
    ax.set_xticks([20, 50, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["20", "50", "100", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2)

    save_figure(fig, "task1_task2_tradeoff")
    plt.close(fig)


def extract_trajectories(raw_data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    data = raw_data.get("results", raw_data)
    if not isinstance(data, dict):
        return []
    out: list[list[dict[str, Any]]] = []
    for value in data.values():
        if isinstance(value, dict) and isinstance(value.get("trajectory"), list):
            out.append(value["trajectory"])
        elif isinstance(value, list):
            out.append(value)
    return out


def load_entropy_matrix(path: Path, num_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    x_common = np.linspace(0.0, 1.0, num_points)
    rows: list[np.ndarray] = []
    for trajectory in extract_trajectories(raw):
        steps: list[float] = []
        values: list[float] = []
        for point in trajectory:
            if not isinstance(point, dict):
                continue
            step = as_float(point.get("step"))
            entropy = as_float(point.get("entropy"))
            if math.isfinite(step) and math.isfinite(entropy):
                steps.append(step)
                values.append(entropy)
        if len(steps) < 2 or max(steps) <= 0:
            continue
        x = np.array(steps, dtype=float) / max(steps)
        y = np.array(values, dtype=float)
        rows.append(np.interp(x_common, x, y))
    return x_common, np.vstack(rows)


def plot_task5_commitment() -> None:
    x, llama = load_entropy_matrix(RESULTS / "task5" / "llama.json")
    _, llada = load_entropy_matrix(RESULTS / "task5" / "llada.json")

    fig, ax = plt.subplots(figsize=(3.35, 2.18), constrained_layout=True)

    def add_line(matrix: np.ndarray, color: str, label: str) -> None:
        mean = matrix.mean(axis=0)
        ci = 1.96 * matrix.std(axis=0) / math.sqrt(matrix.shape[0])
        progress = x * 100.0
        ax.plot(progress, mean, color=color, linewidth=1.85, label=label)
        ax.fill_between(
            progress,
            np.maximum(mean - ci, 0.0),
            np.minimum(mean + ci, 1.0),
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    add_line(llama, "#b73b3c", "LLaMA")
    add_line(llada, "#2f6f9f", "LLaDA")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("Generation progress (%)")
    ax.set_ylabel("Commitment entropy (bits)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.legend(loc="upper left", frameon=False)

    save_figure(fig, "task5_comparison")
    plt.close(fig)


def main() -> None:
    plot_task1_task2_tradeoff()
    plot_task5_commitment()


if __name__ == "__main__":
    main()
