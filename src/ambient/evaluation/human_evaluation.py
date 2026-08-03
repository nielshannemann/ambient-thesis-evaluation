"""Prepare and summarize a blinded human audit of Task-3 continuations."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ambient.utils import write_json_atomic


READING_SLOTS = 4
YES_VALUES = {"yes", "y", "1", "true"}
NO_VALUES = {"no", "n", "0", "false"}
UNCERTAIN_VALUES = {"", "uncertain", "unsure", "?", "na", "n/a"}
ANNOTATION_PROTOCOL_VERSION = "1.1"

ANNOTATION_INSTRUCTIONS = """# Human annotation instructions

## Purpose and unit of judgment

This audit tests whether a generated continuation provides human-recognizable
support for the annotated readings of an ambiguous sentence. Read the
ambiguous sentence, all non-empty gold readings, and the continuation before
assigning labels. Judge the continuation **in the context of the ambiguous
sentence**. Do not try to identify the generating model and do not use the
private key.

The gold readings are legitimate alternatives supplied by the benchmark. They
need not be mutually exclusive. The task is not to decide whether the original
ambiguous sentence permits a reading; it is to decide whether the continuation
provides evidence for that reading.

## Reading-support labels

For every non-empty `gold_reading_N`, enter exactly one of `yes`, `no`, or
`uncertain` in the corresponding `supports_reading_N` column.

- `yes`: In context, the continuation gives positive evidence for the reading.
  This may be an explicit restatement, a presupposition, or a consequence that
  makes that interpretation more evident. Exact word overlap is not required.
- `no`: The continuation contradicts the reading or gives no evidence for it.
  Mere grammatical compatibility, topical relatedness, or failure to resolve
  the ambiguity is not sufficient for `yes`.
- `uncertain`: The relation genuinely cannot be decided with reasonable
  confidence. Do not use `uncertain` merely because the continuation supports
  neither reading; use `no` in that case.

More than one reading may receive `yes`, and all readings may receive `no`.
Leave support columns blank only when their gold-reading column is blank.

## Invalidity

Enter `yes` in `invalid_or_uninterpretable` only when no stable interpretation
can be assigned because the continuation is empty, nonsensical, severely
malformed, or truncated before its intended meaning becomes recoverable.
Otherwise enter `no`.

A fluent but irrelevant continuation is not automatically invalid. Mark its
reading-support labels `no` and rate its surface fluency independently. For an
invalid continuation, still complete every non-empty support field; normally
these will be `no`, unless some support remains recoverable.

## Surface fluency

Rate `fluency_1_to_5` independently of relevance and reading support:

1. Unreadable or effectively word salad.
2. Major grammatical or structural problems impede understanding.
3. Understandable, but clearly awkward, fragmentary, or errorful.
4. Fluent with only minor awkwardness or errors.
5. Fully natural and well formed.

Do not lower fluency merely because a continuation is off topic. Minor spacing
or punctuation artifacts matter only when they impair readability.

## Confidence and procedure

Rate `confidence_1_to_3`: 1 means low confidence or a close judgment, 2 means
moderate confidence, and 3 means the labels are clear. Use `notes` only for
short explanations of difficult or exceptional cases.

Annotate every row independently, without consulting the other annotator,
external resources, automatic labels, or the private key. Do not discuss main
evaluation rows until both annotation sheets are complete. Pilot rows are
separate and may be discussed before the main annotation begins.
"""


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_model_files(values: list[str]) -> dict[str, Path]:
    model_files: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH for --model-file, got: {value}")
        label, path_text = value.split("=", 1)
        label = label.strip()
        if not label or label in model_files:
            raise ValueError(f"Model labels must be non-empty and unique: {label!r}")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        model_files[label] = path
    if len(model_files) < 2:
        raise ValueError("Human comparison requires at least two --model-file entries.")
    return model_files


def load_task3(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload.get("results", [])):
        item_id = item.get("id")
        if item_id is None:
            item_id = item.get("row_id")
        item_id = str(index if item_id is None else item_id)
        rows[item_id] = item
    return rows


def extract_readings(item: dict[str, Any]) -> list[str]:
    side = item.get("ambiguity_side")
    readings: list[str] = []
    for reading in item.get("gold_disambiguations", []):
        text = reading.get(side, "") if side in {"premise", "hypothesis"} else ""
        text = text or reading.get("premise") or reading.get("hypothesis") or ""
        normalized = normalize_text(text)
        if normalized and normalized not in readings:
            readings.append(normalized)
    return readings[:READING_SLOTS]


def choose_continuations(
    item: dict[str, Any],
    count: int,
    rng: random.Random,
) -> list[tuple[int, str]]:
    candidates = [
        (index, normalize_text(text))
        for index, text in enumerate(item.get("continuations", []))
    ]
    if len(candidates) < count:
        raise ValueError(
            f"Instance {item.get('id')} has {len(candidates)} continuation slots, "
            f"but {count} were requested."
        )
    return rng.sample(candidates, count)


def load_requested_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return {str(value) for value in json.loads(raw)}
    return {
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args) -> int:
    output_paths = [
        args.output_dir / "instructions.md",
        args.output_dir / "manifest.json",
        args.output_dir / "private_key.csv",
        *[
            args.output_dir / f"annotation_annotator_{index}.csv"
            for index in range(1, args.num_annotators + 1)
        ],
    ]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not getattr(args, "overwrite", False):
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Refusing to overwrite an existing annotation package ({names}). "
            "Use --overwrite only before annotation has begun."
        )

    model_files = parse_model_files(args.model_file)
    model_rows = {label: load_task3(path) for label, path in model_files.items()}
    common_ids = set.intersection(*(set(rows) for rows in model_rows.values()))
    requested_ids = load_requested_ids(args.id_file)
    if requested_ids is not None:
        missing = requested_ids - common_ids
        if missing:
            raise ValueError(f"The requested set contains {len(missing)} IDs not shared by all models.")
        common_ids = requested_ids

    ordered_ids = sorted(common_ids)
    if args.num_instances > len(ordered_ids):
        raise ValueError(
            f"Requested {args.num_instances} instances but only {len(ordered_ids)} are shared."
        )
    rng = random.Random(args.seed)
    sampled_ids = rng.sample(ordered_ids, args.num_instances)

    annotation_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    row_number = 1
    reference_label = next(iter(model_files))

    for instance_id in sampled_ids:
        reference_item = model_rows[reference_label][instance_id]
        readings = extract_readings(reference_item)
        if len(readings) < 2:
            continue

        for model_label, path in model_files.items():
            item = model_rows[model_label][instance_id]
            selected = choose_continuations(item, args.continuations_per_model, rng)
            for continuation_index, continuation in selected:
                blind_id = f"H-{row_number:05d}"
                row_number += 1
                row: dict[str, Any] = {
                    "blind_id": blind_id,
                    "instance_id": instance_id,
                    "ambiguous_sentence": normalize_text(reference_item.get("ambiguous_sentence")),
                    "ambiguity_side": normalize_text(reference_item.get("ambiguity_side")),
                }
                for slot in range(READING_SLOTS):
                    row[f"gold_reading_{slot + 1}"] = readings[slot] if slot < len(readings) else ""
                row["continuation"] = continuation
                for slot in range(READING_SLOTS):
                    row[f"supports_reading_{slot + 1}"] = ""
                row.update(
                    {
                        "invalid_or_uninterpretable": "",
                        "fluency_1_to_5": "",
                        "confidence_1_to_3": "",
                        "notes": "",
                    }
                )
                annotation_rows.append(row)
                key_rows.append(
                    {
                        "blind_id": blind_id,
                        "instance_id": instance_id,
                        "actual_model": model_label,
                        "source_file": str(path),
                        "continuation_index": continuation_index,
                        "sampling_stratum": args.stratum_label,
                    }
                )

    included_ids = sorted({row["instance_id"] for row in annotation_rows})
    if len(included_ids) != args.num_instances:
        raise ValueError(
            f"Prepared rows for {len(included_ids)} of {args.num_instances} sampled instances."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotator_order_seeds: dict[str, int] = {}
    for annotator_index in range(1, args.num_annotators + 1):
        order_seed = args.seed + 100_003 * annotator_index
        annotator_order_seeds[str(annotator_index)] = order_seed
        annotator_rows = list(annotation_rows)
        random.Random(order_seed).shuffle(annotator_rows)
        write_csv(
            args.output_dir / f"annotation_annotator_{annotator_index}.csv",
            annotator_rows,
        )
    write_csv(args.output_dir / "private_key.csv", key_rows)

    (args.output_dir / "instructions.md").write_text(
        ANNOTATION_INSTRUCTIONS,
        encoding="utf-8",
    )
    manifest = {
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "model_files": {label: str(path) for label, path in model_files.items()},
        "num_instances_requested": args.num_instances,
        "num_instances_included": len(included_ids),
        "sampled_instance_ids": included_ids,
        "num_rows": len(annotation_rows),
        "continuations_per_model": args.continuations_per_model,
        "num_annotators": args.num_annotators,
        "annotator_order_seeds": annotator_order_seeds,
        "seed": args.seed,
        "id_file": str(args.id_file) if args.id_file else None,
        "stratum_label": args.stratum_label,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(f"[INFO] Prepared {len(annotation_rows)} blind rows in {args.output_dir}")
    return 0


def read_csv_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["blind_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate blind_id values in {path}")
    return indexed


def binary_value(value: Any, allow_uncertain: bool = True) -> int | None:
    normalized = normalize_text(value).lower()
    if normalized in YES_VALUES:
        return 1
    if normalized in NO_VALUES:
        return 0
    if allow_uncertain and normalized in UNCERTAIN_VALUES:
        return None
    raise ValueError(f"Expected yes/no/uncertain, got {value!r}")


def binary_kappa(a: list[int], b: list[int]) -> float | None:
    if not a:
        return None
    array_a = np.asarray(a, dtype=int)
    array_b = np.asarray(b, dtype=int)
    observed = float(np.mean(array_a == array_b))
    p_a = float(np.mean(array_a))
    p_b = float(np.mean(array_b))
    expected = p_a * p_b + (1.0 - p_a) * (1.0 - p_b)
    if np.isclose(expected, 1.0):
        return 1.0 if np.isclose(observed, 1.0) else None
    return float((observed - expected) / (1.0 - expected))


def agreement_for_column(
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
    ids: list[str],
    column: str,
) -> dict[str, Any]:
    a_values: list[int] = []
    b_values: list[int] = []
    uncertain = 0
    for blind_id in ids:
        a = binary_value(first[blind_id].get(column))
        b = binary_value(second[blind_id].get(column))
        if a is None or b is None:
            uncertain += 1
            continue
        a_values.append(a)
        b_values.append(b)
    return {
        "n_binary_pairs": len(a_values),
        "n_with_uncertain_or_missing": uncertain,
        "percent_agreement": float(np.mean(np.asarray(a_values) == np.asarray(b_values))) if a_values else None,
        "cohen_kappa": binary_kappa(a_values, b_values),
    }


def parse_rating(value: Any, minimum: int, maximum: int) -> int | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    rating = int(normalized)
    if not minimum <= rating <= maximum:
        raise ValueError(f"Rating {rating} is outside [{minimum}, {maximum}]")
    return rating


def bootstrap_mean_ci(values: list[float], reps: int, seed: int) -> dict[str, Any]:
    """Return an item-level percentile bootstrap interval for a mean."""
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if reps < 1:
        raise ValueError("bootstrap_reps must be at least 1")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(reps, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n": int(len(array)),
    }


def paired_bootstrap_difference(
    values_a: list[float],
    values_b: list[float],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the paired mean difference B minus A over shared items."""
    if not values_a:
        return {"difference_b_minus_a": None, "ci_low": None, "ci_high": None, "n": 0}
    if reps < 1:
        raise ValueError("bootstrap_reps must be at least 1")
    differences = np.asarray(values_b, dtype=float) - np.asarray(values_a, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(reps, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "difference_b_minus_a": float(np.mean(differences)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n": int(len(differences)),
    }


def evaluate(args) -> int:
    annotations = [read_csv_index(path) for path in args.annotations]
    key = read_csv_index(args.key_file)
    common_ids = sorted(set(key).intersection(*(set(rows) for rows in annotations)))
    if not common_ids:
        raise ValueError("No shared blind IDs across annotations and private key.")

    agreement: dict[str, Any] = {}
    if len(annotations) >= 2:
        for slot in range(READING_SLOTS):
            column = f"supports_reading_{slot + 1}"
            reading_ids = [
                blind_id
                for blind_id in common_ids
                if normalize_text(
                    annotations[0][blind_id].get(f"gold_reading_{slot + 1}")
                )
            ]
            agreement[column] = agreement_for_column(
                annotations[0], annotations[1], reading_ids, column
            )
        agreement["invalid_or_uninterpretable"] = agreement_for_column(
            annotations[0], annotations[1], common_ids, "invalid_or_uninterpretable"
        )

        fluency_pairs = []
        for blind_id in common_ids:
            first = parse_rating(annotations[0][blind_id].get("fluency_1_to_5"), 1, 5)
            second = parse_rating(annotations[1][blind_id].get("fluency_1_to_5"), 1, 5)
            if first is not None and second is not None:
                fluency_pairs.append((first, second))
        if fluency_pairs:
            first_values, second_values = map(np.asarray, zip(*fluency_pairs))
            corr = spearmanr(first_values, second_values)
            agreement["fluency_1_to_5"] = {
                "n_pairs": len(fluency_pairs),
                "mean_absolute_difference": float(np.mean(np.abs(first_values - second_values))),
                "spearman_rho": float(corr.statistic),
            }

    consensus_rows: list[dict[str, Any]] = []
    for blind_id in common_ids:
        source = annotations[0][blind_id]
        row: dict[str, Any] = {
            "blind_id": blind_id,
            "instance_id": key[blind_id]["instance_id"],
            "actual_model": key[blind_id]["actual_model"],
            "sampling_stratum": key[blind_id].get("sampling_stratum", "random"),
            "ambiguous_sentence": source.get("ambiguous_sentence", ""),
            "ambiguity_side": source.get("ambiguity_side", ""),
            "continuation": source.get("continuation", ""),
            "num_gold_readings": sum(
                bool(normalize_text(source.get(f"gold_reading_{slot + 1}")))
                for slot in range(READING_SLOTS)
            ),
        }
        for slot in range(READING_SLOTS):
            row[f"gold_reading_{slot + 1}"] = source.get(f"gold_reading_{slot + 1}", "")
        invalid_votes = [
            binary_value(annotation[blind_id].get("invalid_or_uninterpretable"))
            for annotation in annotations
        ]
        if any(value is None for value in invalid_votes):
            row["invalid_consensus"] = None
        elif all(value == invalid_votes[0] for value in invalid_votes):
            row["invalid_consensus"] = invalid_votes[0]
        else:
            row["invalid_consensus"] = None
        for slot in range(READING_SLOTS):
            reading_present = bool(normalize_text(source.get(f"gold_reading_{slot + 1}")))
            votes = [
                binary_value(annotation[blind_id].get(f"supports_reading_{slot + 1}"))
                for annotation in annotations
            ]
            if not reading_present:
                consensus = None
            elif any(value is None for value in votes):
                consensus = None
            elif all(value == 1 for value in votes):
                consensus = 1
            elif all(value == 0 for value in votes):
                consensus = 0
            else:
                consensus = None
            row[f"supports_reading_{slot + 1}_consensus"] = consensus

        ratings = [
            parse_rating(annotation[blind_id].get("fluency_1_to_5"), 1, 5)
            for annotation in annotations
        ]
        valid_ratings = [value for value in ratings if value is not None]
        row["fluency_mean"] = float(np.mean(valid_ratings)) if valid_ratings else None
        consensus_rows.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        grouped[(row["actual_model"], row["instance_id"])].append(row)

    all_models = sorted({row["actual_model"] for row in consensus_rows})
    model_item_scores: dict[str, list[float]] = defaultdict(list)
    model_valid_rates: dict[str, list[float]] = defaultdict(list)
    model_fluency: dict[str, list[float]] = defaultdict(list)
    item_metrics: list[dict[str, Any]] = []
    for (model, instance_id), rows in grouped.items():
        num_gold_readings = max(int(row["num_gold_readings"]) for row in rows)
        reading_support: list[float] = []
        complete_reading_coverage = True
        for slot in range(num_gold_readings):
            values = [
                row[f"supports_reading_{slot + 1}_consensus"]
                for row in rows
                if row[f"supports_reading_{slot + 1}_consensus"] is not None
            ]
            if values:
                reading_support.append(float(np.mean(values)))
            else:
                complete_reading_coverage = False
        least_support = (
            float(min(reading_support))
            if complete_reading_coverage and len(reading_support) == num_gold_readings
            else None
        )
        if least_support is not None:
            model_item_scores[model].append(least_support)
        invalid_values = [row["invalid_consensus"] for row in rows if row["invalid_consensus"] is not None]
        valid_rate = None
        if invalid_values:
            valid_rate = 1.0 - float(np.mean(invalid_values))
            model_valid_rates[model].append(valid_rate)
        fluency_values = [
            float(row["fluency_mean"])
            for row in rows
            if row["fluency_mean"] is not None
        ]
        item_fluency = float(np.mean(fluency_values)) if fluency_values else None
        if item_fluency is not None:
            model_fluency[model].append(item_fluency)
        item_metrics.append(
            {
                "actual_model": model,
                "instance_id": instance_id,
                "num_gold_readings": num_gold_readings,
                "num_continuations": len(rows),
                "complete_reading_coverage_judgments": complete_reading_coverage,
                "least_reading_support": least_support,
                "valid_continuation_rate": valid_rate,
                "mean_fluency": item_fluency,
            }
        )

    model_summary = {
        model: {
            "num_items_total": sum(row["actual_model"] == model for row in item_metrics),
            "least_reading_support": bootstrap_mean_ci(
                model_item_scores[model], args.bootstrap_reps, args.seed
            ),
            "valid_continuation_rate": bootstrap_mean_ci(
                model_valid_rates[model], args.bootstrap_reps, args.seed + 1
            ),
            "fluency": bootstrap_mean_ci(
                model_fluency[model], args.bootstrap_reps, args.seed + 2
            ),
        }
        for model in all_models
    }

    item_lookup = {
        (row["actual_model"], row["instance_id"]): row
        for row in item_metrics
    }
    pairwise_model_differences: dict[str, Any] = {}
    for first_index, model_a in enumerate(all_models):
        for model_b in all_models[first_index + 1 :]:
            shared_ids = sorted(
                {
                    row["instance_id"]
                    for row in item_metrics
                    if row["actual_model"] == model_a
                    and row["least_reading_support"] is not None
                }
                & {
                    row["instance_id"]
                    for row in item_metrics
                    if row["actual_model"] == model_b
                    and row["least_reading_support"] is not None
                }
            )
            values_a = [
                float(item_lookup[(model_a, instance_id)]["least_reading_support"])
                for instance_id in shared_ids
            ]
            values_b = [
                float(item_lookup[(model_b, instance_id)]["least_reading_support"])
                for instance_id in shared_ids
            ]
            pairwise_model_differences[f"{model_b}_minus_{model_a}"] = paired_bootstrap_difference(
                values_a,
                values_b,
                args.bootstrap_reps,
                args.seed,
            )

    nli_human_agreement = None
    if args.nli_labels is not None:
        nli_rows = read_csv_index(args.nli_labels)
        human_values: list[int] = []
        nli_values: list[int] = []
        for row in consensus_rows:
            nli_row = nli_rows.get(row["blind_id"])
            if nli_row is None:
                continue
            for slot in range(READING_SLOTS):
                human_value = row[f"supports_reading_{slot + 1}_consensus"]
                nli_value = binary_value(nli_row.get(f"nli_supports_reading_{slot + 1}"))
                if human_value is None or nli_value is None:
                    continue
                human_values.append(int(human_value))
                nli_values.append(int(nli_value))

        if human_values:
            human_array = np.asarray(human_values, dtype=int)
            nli_array = np.asarray(nli_values, dtype=int)
            true_positive = int(np.sum((human_array == 1) & (nli_array == 1)))
            false_positive = int(np.sum((human_array == 0) & (nli_array == 1)))
            false_negative = int(np.sum((human_array == 1) & (nli_array == 0)))
            true_negative = int(np.sum((human_array == 0) & (nli_array == 0)))
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            )
            nli_human_agreement = {
                "n_binary_pairs": len(human_values),
                "accuracy": float(np.mean(human_array == nli_array)),
                "cohen_kappa": binary_kappa(human_values, nli_values),
                "precision_for_human_support": precision,
                "recall_for_human_support": recall,
                "f1_for_human_support": f1,
                "confusion": {
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_negative": true_negative,
                },
            }

    output = {
        "annotation_files": [str(path) for path in args.annotations],
        "key_file": str(args.key_file),
        "num_shared_rows": len(common_ids),
        "agreement": agreement,
        "nli_labels": str(args.nli_labels) if args.nli_labels else None,
        "nli_human_agreement": nli_human_agreement,
        "model_summary_consensus_only": model_summary,
        "pairwise_least_reading_support": pairwise_model_differences,
        "item_metrics": item_metrics,
        "consensus_rule": "all annotators yes => yes; all no => no; disagreements/uncertain => missing",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, output)
    consensus_path = args.consensus_path or args.output_path.with_name(
        f"{args.output_path.stem}_consensus.csv"
    )
    write_csv(consensus_path, consensus_rows)
    print(json.dumps(output, indent=2))
    print(f"[INFO] Consensus rows written to {consensus_path}")
    return 0


def nli_label(args) -> int:
    """Apply the paper's NLI judge to the exact blind human-evaluation rows."""
    from transformers import pipeline

    from ambient.evaluation.task3_silhouette_evaluate import (
        entailment_score_from_scores,
        is_argmax_entailment,
    )

    with args.annotation_sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No annotation rows found in {args.annotation_sheet}")

    pairs: list[dict[str, str]] = []
    pair_locations: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        continuation = normalize_text(row.get("continuation"))
        for slot in range(READING_SLOTS):
            reading = normalize_text(row.get(f"gold_reading_{slot + 1}"))
            if continuation and reading:
                pairs.append({"text": continuation, "text_pair": reading})
                pair_locations.append((row_index, slot))

    device = 0 if __import__("torch").cuda.is_available() else -1
    print(f"[INFO] Loading NLI model {args.nli_model} on device {device}.")
    judge = pipeline("text-classification", model=args.nli_model, device=device)
    print(f"[INFO] Scoring {len(pairs)} continuation-reading pairs.")
    predictions = judge(
        pairs,
        truncation=True,
        max_length=512,
        batch_size=args.batch_size,
        top_k=None,
    )

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        output = {"blind_id": row["blind_id"], "instance_id": row.get("instance_id", "")}
        for slot in range(READING_SLOTS):
            output[f"nli_supports_reading_{slot + 1}"] = ""
            output[f"nli_entailment_score_{slot + 1}"] = ""
        output_rows.append(output)

    for (row_index, slot), prediction in zip(pair_locations, predictions):
        entailment_score = entailment_score_from_scores(prediction)
        if args.threshold == "argmax":
            entails = is_argmax_entailment(prediction)
        else:
            entails = entailment_score >= float(args.threshold)
        output_rows[row_index][f"nli_supports_reading_{slot + 1}"] = "yes" if entails else "no"
        output_rows[row_index][f"nli_entailment_score_{slot + 1}"] = entailment_score

    write_csv(args.output_path, output_rows)
    metadata = {
        "annotation_sheet": str(args.annotation_sheet),
        "nli_model": args.nli_model,
        "threshold": args.threshold,
        "num_rows": len(rows),
        "num_pairs": len(pairs),
    }
    write_json_atomic(args.output_path.with_suffix(".meta.json"), metadata)
    print(f"[INFO] NLI labels written to {args.output_path}")
    return 0
