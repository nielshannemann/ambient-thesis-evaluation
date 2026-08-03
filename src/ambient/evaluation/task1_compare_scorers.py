"""Compare two Task-1 scoring summaries on identical saved continuations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ambient.evaluation.task1_compute_results_metrics import read_jsonl
from ambient.utils import write_json_atomic


def index_rows(path: Path, dedupe: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if dedupe == "row":
        key_fn = lambda row: str(row.get("row_id") or row.get("id"))
    else:
        key_fn = lambda row: str(row.get("instance_id") or row.get("id") or row.get("row_id"))
    return {key_fn(row): row for row in rows}


def summarize_item(row: dict[str, Any], metric_key: str) -> dict[str, Any] | None:
    options = row.get("options") or {}
    y_keys = sorted(key for key in options if key.startswith("y"))
    if "d" not in options or not y_keys:
        return None
    d_value = options["d"].get(metric_key)
    y_values = [options[key].get(metric_key) for key in y_keys]
    if d_value is None or any(value is None for value in y_values):
        return None
    y_array = np.asarray(y_values, dtype=float)
    d_float = float(d_value)
    return {
        "all_correct": bool(np.all(y_array < d_float)),
        "any_correct": bool(np.any(y_array < d_float)),
        "strict_margin": float(d_float - np.max(y_array)),
        "mean_margin": float(d_float - np.mean(y_array)),
    }


def bootstrap_accuracy_difference(
    reference: np.ndarray,
    alternative: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(reference)
    observed = float(np.mean(alternative) - np.mean(reference))
    if n == 0 or reps < 1:
        return {"difference": observed, "ci_low": float("nan"), "ci_high": float("nan")}
    samples = np.empty(reps, dtype=float)
    for index in range(reps):
        draw = rng.integers(0, n, size=n)
        samples[index] = float(np.mean(alternative[draw]) - np.mean(reference[draw]))
    return {
        "difference": observed,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def run(args) -> int:
    reference_rows = index_rows(args.reference_summary, args.dedupe)
    alternative_rows = index_rows(args.alternative_summary, args.dedupe)
    common_ids = sorted(set(reference_rows) & set(alternative_rows))

    details: list[dict[str, Any]] = []
    for instance_id in common_ids:
        reference = summarize_item(reference_rows[instance_id], args.metric_key)
        alternative = summarize_item(alternative_rows[instance_id], args.metric_key)
        if reference is None or alternative is None:
            continue
        details.append(
            {
                "instance_id": instance_id,
                **{f"reference_{key}": value for key, value in reference.items()},
                **{f"alternative_{key}": value for key, value in alternative.items()},
            }
        )

    if not details:
        raise ValueError("The summaries have no commonly evaluable Task-1 items.")

    ref_correct = np.asarray([row["reference_all_correct"] for row in details], dtype=float)
    alt_correct = np.asarray([row["alternative_all_correct"] for row in details], dtype=float)
    ref_margins = np.asarray([row["reference_strict_margin"] for row in details], dtype=float)
    alt_margins = np.asarray([row["alternative_strict_margin"] for row in details], dtype=float)
    correlation = spearmanr(ref_margins, alt_margins)

    both_correct = int(np.sum((ref_correct == 1) & (alt_correct == 1)))
    both_incorrect = int(np.sum((ref_correct == 0) & (alt_correct == 0)))
    reference_only = int(np.sum((ref_correct == 1) & (alt_correct == 0)))
    alternative_only = int(np.sum((ref_correct == 0) & (alt_correct == 1)))

    output = {
        "reference_summary": str(args.reference_summary),
        "alternative_summary": str(args.alternative_summary),
        "metric_key": args.metric_key,
        "dedupe": args.dedupe,
        "num_common_ids": len(common_ids),
        "num_evaluable": len(details),
        "reference_accuracy": float(np.mean(ref_correct)),
        "alternative_accuracy": float(np.mean(alt_correct)),
        "paired_accuracy_difference_alternative_minus_reference": bootstrap_accuracy_difference(
            ref_correct,
            alt_correct,
            args.bootstrap_reps,
            args.seed,
        ),
        "rank_outcome_agreement": float(np.mean(ref_correct == alt_correct)),
        "outcome_counts": {
            "both_correct": both_correct,
            "both_incorrect": both_incorrect,
            "reference_only": reference_only,
            "alternative_only": alternative_only,
        },
        "strict_margin_spearman": {
            "rho": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, output)
    details_path = args.details_path or args.output_path.with_name(f"{args.output_path.stem}_items.csv")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    print(json.dumps(output, indent=2))
    print(f"[INFO] Item-level comparison written to {details_path}")
    return 0
