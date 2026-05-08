#!/usr/bin/env python3
"""Robustness checks for the AMBIENT continuation-ranking experiment.

The module is intentionally read-only: it consumes existing Task-0/Task-1
ranking outputs and writes aggregate CSV/JSON files. It supports both summary
JSONL files and the per-example `example_dirs` written by the evaluation loop.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ambient.evaluation.task0_compute_results_metrics import dedupe_results, read_jsonl
from ambient.utils import is_suspicious


METRIC_VARIANTS = {
    "unnormalized_unfiltered": "empirical_KL_div_all",
    "unnormalized_cleaned": "empirical_KL_div_clean",
    "normalized_unfiltered": "empirical_KL_div_normalized_all",
    "normalized_cleaned": "empirical_KL_div_normalized_clean",
}


@dataclass(frozen=True)
class FilterConfig:
    non_alnum_ratio: float
    max_consec_repeat: int


def parse_csv_numbers(raw: str | None, cast):
    if raw is None or str(raw).strip() == "":
        return []
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def option_sort_key(key: str) -> tuple[int, str]:
    if key == "d":
        return (10_000, key)
    match = re.match(r"y(\d+)$", key)
    if match:
        return (int(match.group(1)), key)
    return (9_000, key)


def extract_k(summary_path: Path) -> int | None:
    match = re.search(r"summary_mc(\d+)\.jsonl$", summary_path.name)
    if match:
        return int(match.group(1))
    return None


def quantile_interval(values: list[float], ci_level: float) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None, None
    alpha = (100.0 - ci_level) / 2.0
    lo = float(np.percentile(clean, alpha))
    hi = float(np.percentile(clean, 100.0 - alpha))
    return lo, hi


def mean_finite(values: Iterable[Any]) -> float | None:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(numeric)
    return float(np.mean(clean)) if clean else None


def outcome_records_from_summary(results: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ex in results:
        opts = ex.get("options", {}) or {}
        y_keys = sorted([key for key in opts if str(key).startswith("y")], key=option_sort_key)
        if "d" not in opts or not y_keys:
            continue

        d_val = opts["d"].get(metric_key)
        y_vals = [opts[key].get(metric_key) for key in y_keys]
        artifact_vals = [float(opts[key].get("artifact_rate", 0.0) or 0.0) for key in [*y_keys, "d"]]

        if d_val is None or any(val is None for val in y_vals):
            records.append(
                {
                    "id": str(ex.get("instance_id") or ex.get("row_id") or ex.get("id")),
                    "evaluated": False,
                    "artifact_rate": float(np.mean(artifact_vals)) if artifact_vals else 0.0,
                }
            )
            continue

        wins = [float(y_val) < float(d_val) for y_val in y_vals]
        records.append(
            {
                "id": str(ex.get("instance_id") or ex.get("row_id") or ex.get("id")),
                "evaluated": True,
                "correct_all": bool(all(wins)),
                "correct_any": bool(any(wins)),
                "artifact_rate": float(np.mean(artifact_vals)) if artifact_vals else 0.0,
                "num_readings": len(y_keys),
            }
        )
    return records


def aggregate_outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("evaluated")]
    artifacts = [float(record.get("artifact_rate", 0.0) or 0.0) for record in records]
    return {
        "total_instances": len(records),
        "evaluated_instances": len(evaluated),
        "ranking_accuracy_all": float(np.mean([record["correct_all"] for record in evaluated])) if evaluated else None,
        "ranking_accuracy_any": float(np.mean([record["correct_any"] for record in evaluated])) if evaluated else None,
        "artifact_rate": float(np.mean(artifacts)) if artifacts else None,
    }


def bootstrap_accuracy(
    records: list[dict[str, Any]],
    reps: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("evaluated")]
    if not evaluated:
        return {
            "bootstrap_reps": reps,
            "all_ci_low": None,
            "all_ci_high": None,
            "any_ci_low": None,
            "any_ci_high": None,
        }

    rng = random.Random(seed)
    all_vals: list[float] = []
    any_vals: list[float] = []
    n = len(evaluated)
    for _ in range(reps):
        sample = [evaluated[rng.randrange(n)] for _ in range(n)]
        all_vals.append(float(np.mean([row["correct_all"] for row in sample])))
        any_vals.append(float(np.mean([row["correct_any"] for row in sample])))

    all_low, all_high = quantile_interval(all_vals, ci_level)
    any_low, any_high = quantile_interval(any_vals, ci_level)
    return {
        "bootstrap_reps": reps,
        "all_ci_low": all_low,
        "all_ci_high": all_high,
        "any_ci_low": any_low,
        "any_ci_high": any_high,
    }


def load_example_dirs(run_dir: Path, example_dir_name: str = "example_dirs") -> dict[str, dict[str, list[dict[str, Any]]]]:
    root = run_dir / example_dir_name
    if not root.exists():
        return {}

    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        options: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(child.glob("*.jsonl")):
            if path.name == "prompts.jsonl":
                continue
            key = path.stem
            if key == "d" or re.match(r"y\d+$", key):
                rows = read_jsonl(path)
                if rows:
                    options[key] = rows
        y_keys = [key for key in options if key.startswith("y")]
        if "d" in options and y_keys:
            examples[child.name] = options
    return examples


def row_metric_value(row: dict[str, Any], metric_key: str) -> float | None:
    if metric_key.endswith("_normalized_clean") or metric_key.endswith("_normalized_all"):
        value = row.get("avg_log_odds")
    else:
        value = row.get("log_odds")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_is_artifact(row: dict[str, Any], filter_config: FilterConfig | None = None) -> bool:
    if filter_config is None:
        return bool(row.get("flagged_artifact", False))
    text = str(row.get("continuation_clean") or "")
    return is_suspicious(
        text,
        max_non_alnum_ratio=filter_config.non_alnum_ratio,
        max_consec_repeat=filter_config.max_consec_repeat,
    )


def option_score_from_rows(
    rows: list[dict[str, Any]],
    metric_key: str,
    rng: random.Random | None = None,
    sample_n: int | None = None,
    sample_with_replacement: bool = False,
    filter_config: FilterConfig | None = None,
) -> tuple[float | None, float]:
    pool = list(rows)
    if sample_n is not None:
        if len(pool) < sample_n and not sample_with_replacement:
            return None, 1.0
        if sample_with_replacement:
            assert rng is not None
            pool = [pool[rng.randrange(len(pool))] for _ in range(sample_n)] if pool else []
        else:
            assert rng is not None
            pool = rng.sample(pool, sample_n)

    scored = [row for row in pool if row_metric_value(row, metric_key) is not None]
    if not scored:
        return None, 1.0

    use_clean = metric_key.endswith("_clean")
    if use_clean:
        kept = [row for row in scored if not row_is_artifact(row, filter_config)]
    else:
        kept = scored

    artifact_rate = 1.0 - (len(kept) / len(scored)) if scored else 1.0
    values = [row_metric_value(row, metric_key) for row in kept]
    values = [value for value in values if value is not None]
    if not values:
        return None, artifact_rate
    return float(np.mean(values)), float(artifact_rate)


def outcome_records_from_example_dirs(
    examples: dict[str, dict[str, list[dict[str, Any]]]],
    metric_key: str,
    seed: int,
    sample_n: int | None = None,
    sample_with_replacement: bool = False,
    filter_config: FilterConfig | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for example_id, options in examples.items():
        y_keys = sorted([key for key in options if key.startswith("y")], key=option_sort_key)
        if "d" not in options or not y_keys:
            continue

        option_scores: dict[str, float | None] = {}
        artifact_rates: list[float] = []
        for key in [*y_keys, "d"]:
            score, artifact_rate = option_score_from_rows(
                options[key],
                metric_key=metric_key,
                rng=rng,
                sample_n=sample_n,
                sample_with_replacement=sample_with_replacement,
                filter_config=filter_config,
            )
            option_scores[key] = score
            artifact_rates.append(artifact_rate)

        d_val = option_scores.get("d")
        y_vals = [option_scores.get(key) for key in y_keys]
        if d_val is None or any(value is None for value in y_vals):
            records.append(
                {
                    "id": example_id,
                    "evaluated": False,
                    "artifact_rate": float(np.mean(artifact_rates)) if artifact_rates else 0.0,
                }
            )
            continue

        wins = [float(y_val) < float(d_val) for y_val in y_vals if y_val is not None]
        records.append(
            {
                "id": example_id,
                "evaluated": True,
                "correct_all": bool(all(wins)),
                "correct_any": bool(any(wins)),
                "artifact_rate": float(np.mean(artifact_rates)) if artifact_rates else 0.0,
                "num_readings": len(y_keys),
            }
        )
    return records


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[info] wrote {path}")


def summarize_summary_files(args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        for summary_path in sorted(run_dir.glob(args.summary_glob)):
            raw = read_jsonl(summary_path)
            if not raw:
                continue
            deduped = dedupe_results(raw, args.dedupe)
            for variant_name, metric_key in METRIC_VARIANTS.items():
                records = outcome_records_from_summary(deduped, metric_key)
                aggregate = aggregate_outcomes(records)
                boot = bootstrap_accuracy(
                    records,
                    reps=args.bootstrap_reps,
                    ci_level=args.ci_level,
                    seed=args.seed,
                )
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "summary_file": summary_path.name,
                        "k": extract_k(summary_path),
                        "metric_variant": variant_name,
                        "metric_key": metric_key,
                        **aggregate,
                        **boot,
                    }
                )
    return rows


def summarize_matched_subsampling(args, metric_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    if args.subsample_n is None or args.subsample_reps <= 0:
        return summary_rows, replicate_rows

    for run_dir in args.run_dirs:
        examples = load_example_dirs(run_dir, args.example_dir_name)
        if not examples:
            print(f"[warn] no example_dirs found for {run_dir}; skipping matched-count subsampling")
            continue

        all_acc: list[float] = []
        any_acc: list[float] = []
        evaluated_counts: list[int] = []
        artifact_rates: list[float] = []
        for rep in range(args.subsample_reps):
            records = outcome_records_from_example_dirs(
                examples,
                metric_key=metric_key,
                seed=args.seed + rep,
                sample_n=args.subsample_n,
                sample_with_replacement=args.subsample_with_replacement,
            )
            aggregate = aggregate_outcomes(records)
            all_acc.append(aggregate["ranking_accuracy_all"])
            any_acc.append(aggregate["ranking_accuracy_any"])
            evaluated_counts.append(aggregate["evaluated_instances"])
            artifact_rates.append(aggregate["artifact_rate"])
            replicate_rows.append(
                {
                    "run_dir": str(run_dir),
                    "replicate": rep,
                    "subsample_n": args.subsample_n,
                    "metric_key": metric_key,
                    **aggregate,
                }
            )

        all_low, all_high = quantile_interval(all_acc, args.ci_level)
        any_low, any_high = quantile_interval(any_acc, args.ci_level)
        summary_rows.append(
            {
                "run_dir": str(run_dir),
                "num_examples_with_dirs": len(examples),
                "subsample_n": args.subsample_n,
                "subsample_reps": args.subsample_reps,
                "sample_with_replacement": args.subsample_with_replacement,
                "metric_key": metric_key,
                "ranking_accuracy_all_mean": mean_finite(all_acc),
                "ranking_accuracy_all_ci_low": all_low,
                "ranking_accuracy_all_ci_high": all_high,
                "ranking_accuracy_any_mean": mean_finite(any_acc),
                "ranking_accuracy_any_ci_low": any_low,
                "ranking_accuracy_any_ci_high": any_high,
                "evaluated_instances_mean": mean_finite(evaluated_counts),
                "artifact_rate_mean": mean_finite(artifact_rates),
            }
        )
    return summary_rows, replicate_rows


def summarize_filter_sensitivity(args, metric_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    non_alnum_values = parse_csv_numbers(args.filter_non_alnum_ratios, float)
    repeat_values = parse_csv_numbers(args.filter_repeat_thresholds, int)
    if not non_alnum_values or not repeat_values:
        return rows

    for run_dir in args.run_dirs:
        examples = load_example_dirs(run_dir, args.example_dir_name)
        if not examples:
            print(f"[warn] no example_dirs found for {run_dir}; skipping threshold-level filter sensitivity")
            continue
        for non_alnum in non_alnum_values:
            for repeat in repeat_values:
                config = FilterConfig(non_alnum_ratio=non_alnum, max_consec_repeat=repeat)
                records = outcome_records_from_example_dirs(
                    examples,
                    metric_key=metric_key,
                    seed=args.seed,
                    filter_config=config,
                )
                aggregate = aggregate_outcomes(records)
                boot = bootstrap_accuracy(records, args.bootstrap_reps, args.ci_level, args.seed)
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "metric_key": metric_key,
                        "non_alnum_ratio": non_alnum,
                        "max_consec_repeat": repeat,
                        **aggregate,
                        **boot,
                    }
                )
    return rows


def run(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_key = args.metric_key
    if metric_key in METRIC_VARIANTS:
        metric_key = METRIC_VARIANTS[metric_key]

    summary_rows = summarize_summary_files(args)
    write_csv(args.output_dir / "task1_summary_and_k_sensitivity.csv", summary_rows)

    subsample_summary, subsample_reps = summarize_matched_subsampling(args, metric_key)
    write_csv(args.output_dir / f"task1_matched_subsample_n{args.subsample_n or 'none'}_summary.csv", subsample_summary)
    if args.write_replicates:
        write_csv(args.output_dir / f"task1_matched_subsample_n{args.subsample_n or 'none'}_replicates.csv", subsample_reps)

    filter_rows = summarize_filter_sensitivity(args, metric_key)
    write_csv(args.output_dir / "task1_filter_threshold_sensitivity.csv", filter_rows)

    config = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "summary_glob": args.summary_glob,
        "dedupe": args.dedupe,
        "metric_key": metric_key,
        "bootstrap_reps": args.bootstrap_reps,
        "ci_level": args.ci_level,
        "seed": args.seed,
        "subsample_n": args.subsample_n,
        "subsample_reps": args.subsample_reps,
        "subsample_with_replacement": args.subsample_with_replacement,
        "filter_non_alnum_ratios": args.filter_non_alnum_ratios,
        "filter_repeat_thresholds": args.filter_repeat_thresholds,
    }
    with (args.output_dir / "task1_robustness_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"[info] wrote {args.output_dir / 'task1_robustness_config.json'}")
    return 0
