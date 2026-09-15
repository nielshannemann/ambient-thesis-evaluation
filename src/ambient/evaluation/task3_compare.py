"""Paired item-level comparisons for aligned Task-3 evaluation artifacts."""

from __future__ import annotations

import json
import zlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ambient.utils import write_json_atomic


METRIC_PATHS = {
    "mcd": ("mcd",),
    "silhouette": ("silhouette",),
    "mtc_cos_percent": ("mtc_cos_percent",),
    "mtc_nli_argmax_percent": ("mtc_nli_percent", "argmax"),
    "mtc_nli_0.5_percent": ("mtc_nli_percent", "0.5"),
    "mtc_nli_0.8_percent": ("mtc_nli_percent", "0.8"),
}


def _metric_value(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if value is None:
        return None
    return float(value)


def _derived_seed(seed: int, key: str) -> int:
    return (int(seed) + zlib.crc32(key.encode("utf-8"))) % (2**32)


def bootstrap_summary(
    values: list[float],
    bootstrap_reps: int,
    ci_level: float,
    seed: int,
) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "n": 0,
            "ci_lower": None,
            "ci_upper": None,
        }
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be at least 1")
    if not 0.0 < ci_level < 100.0:
        raise ValueError("ci_level must be between 0 and 100")

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstrap_reps, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (100.0 - ci_level) / 2.0
    lower, upper = np.percentile(means, [alpha, 100.0 - alpha])
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "n": int(array.size),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def _index_item_metrics(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("item_metrics")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"Evaluation '{label}' has no item_metrics; rerun it with the current evaluator."
        )
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id"))
        if identifier in indexed:
            raise ValueError(f"Evaluation '{label}' contains duplicate item ID {identifier}.")
        indexed[identifier] = row
    return indexed


def build_comparison(
    payloads: dict[str, dict[str, Any]],
    bootstrap_reps: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("Provide at least two Task-3 evaluation artifacts.")

    labels = list(payloads)
    policies = {
        label: {
            "artifact_policy": payload.get("artifact_policy", "keep"),
            "duplicate_policy": payload.get("duplicate_policy", "keep"),
        }
        for label, payload in payloads.items()
    }
    if len({tuple(policy.values()) for policy in policies.values()}) != 1:
        raise ValueError(f"Task-3 policy mismatch across evaluation artifacts: {policies}")

    indexed = {
        label: _index_item_metrics(payload, label)
        for label, payload in payloads.items()
    }
    id_sets = {label: set(rows) for label, rows in indexed.items()}
    reference_ids = id_sets[labels[0]]
    mismatched = {
        label: {
            "missing": len(reference_ids - ids),
            "extra": len(ids - reference_ids),
        }
        for label, ids in id_sets.items()
        if ids != reference_ids
    }
    if mismatched:
        raise ValueError(f"Task-3 item-ID mismatch: {mismatched}")

    model_summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        model_summaries[label] = {}
        for metric, path in METRIC_PATHS.items():
            values = [
                value
                for identifier in sorted(reference_ids)
                if (value := _metric_value(indexed[label][identifier], path)) is not None
            ]
            model_summaries[label][metric] = bootstrap_summary(
                values,
                bootstrap_reps,
                ci_level,
                _derived_seed(seed, f"model:{label}:{metric}"),
            )

    paired_differences: dict[str, dict[str, Any]] = {}
    for label_a, label_b in combinations(labels, 2):
        comparison_key = f"{label_b}_minus_{label_a}"
        paired_differences[comparison_key] = {}
        for metric, path in METRIC_PATHS.items():
            differences = []
            for identifier in sorted(reference_ids):
                value_a = _metric_value(indexed[label_a][identifier], path)
                value_b = _metric_value(indexed[label_b][identifier], path)
                if value_a is None or value_b is None:
                    continue
                differences.append(value_b - value_a)
            summary = bootstrap_summary(
                differences,
                bootstrap_reps,
                ci_level,
                _derived_seed(seed, f"pair:{comparison_key}:{metric}"),
            )
            summary["mean_difference"] = summary.pop("mean")
            paired_differences[comparison_key][metric] = summary

    return {
        "artifact_policy": next(iter(policies.values()))["artifact_policy"],
        "duplicate_policy": next(iter(policies.values()))["duplicate_policy"],
        "num_aligned_items": len(reference_ids),
        "bootstrap_reps": bootstrap_reps,
        "ci_level": ci_level,
        "seed": seed,
        "model_summaries": model_summaries,
        "paired_differences": paired_differences,
    }


def parse_evaluation_files(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, received: {value}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in parsed:
            raise ValueError(f"Evaluation labels must be non-empty and unique: {label!r}")
        parsed[label] = Path(raw_path)
    return parsed


def run(args) -> int:
    files = parse_evaluation_files(args.evaluation_file)
    payloads = {}
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            payloads[label] = json.load(handle)

    comparison = build_comparison(
        payloads,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        seed=args.seed,
    )
    comparison["evaluation_files"] = {
        label: str(path) for label, path in files.items()
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, comparison)

    print(
        f"[INFO] Compared {len(files)} models on "
        f"{comparison['num_aligned_items']} aligned Task-3 items."
    )
    print(
        f"[INFO] Policy: artifacts={comparison['artifact_policy']}, "
        f"duplicates={comparison['duplicate_policy']}."
    )
    for label, metrics in comparison["model_summaries"].items():
        values = " | ".join(
            f"{metric}={summary['mean']:.4f}"
            for metric, summary in metrics.items()
            if summary["mean"] is not None
        )
        print(f"[model] {label}: {values}")
    for pair, metrics in comparison["paired_differences"].items():
        values = " | ".join(
            f"{metric}={summary['mean_difference']:+.4f} "
            f"[{summary['ci_lower']:+.4f}, {summary['ci_upper']:+.4f}]"
            for metric, summary in metrics.items()
            if summary["mean_difference"] is not None
        )
        print(f"[pair] {pair}: {values}")
    print(f"[INFO] Wrote paired Task-3 comparison: {args.output_path}")
    return 0
