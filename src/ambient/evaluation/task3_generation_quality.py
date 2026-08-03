"""Lightweight generation-quality diagnostics for existing Task-3 artifacts."""

from __future__ import annotations

import json
import statistics
import unicodedata
from pathlib import Path
from typing import Any

from ambient.utils import is_suspicious, write_json_atomic


def _canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _percentage(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def summarize_task3_quality(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize validity and exact repetition without semantic quality claims."""
    results = list(payload.get("results") or [])
    metadata = payload.get("metadata") or {}
    requested_per_item = (metadata.get("hyperparameters") or {}).get("num_continuations")
    if not isinstance(requested_per_item, int) or requested_per_item < 1:
        requested_per_item = None

    returned_slots = 0
    nonempty_count = 0
    suspicious_count = 0
    duplicate_excess_count = 0
    word_counts: list[int] = []
    items_with_empty = 0
    items_with_empty_or_missing = 0
    items_with_suspicious = 0
    items_with_duplicates = 0
    flagged_items: list[dict[str, Any]] = []

    for index, item in enumerate(results):
        continuations = list(item.get("continuations") or [])
        returned_slots += len(continuations)
        nonempty = [str(text).strip() for text in continuations if str(text or "").strip()]
        empty_count = len(continuations) - len(nonempty)
        missing_count = (
            max(0, requested_per_item - len(continuations))
            if requested_per_item is not None
            else 0
        )
        suspicious = [text for text in nonempty if is_suspicious(text)]
        canonical = [_canonical_text(text) for text in nonempty]
        duplicate_count = len(canonical) - len(set(canonical))

        nonempty_count += len(nonempty)
        suspicious_count += len(suspicious)
        duplicate_excess_count += duplicate_count
        word_counts.extend(len(text.split()) for text in nonempty)
        items_with_empty += int(empty_count > 0)
        items_with_empty_or_missing += int(empty_count > 0 or missing_count > 0)
        items_with_suspicious += int(bool(suspicious))
        items_with_duplicates += int(duplicate_count > 0)

        if empty_count or missing_count or suspicious or duplicate_count:
            identifier = item.get("id")
            if identifier is None:
                identifier = item.get("row_id", index)
            flagged_items.append(
                {
                    "id": str(identifier),
                    "empty": empty_count,
                    "missing": missing_count,
                    "heuristic_artifacts": len(suspicious),
                    "exact_duplicate_excess": duplicate_count,
                }
            )

    expected_slots = (
        requested_per_item * len(results)
        if requested_per_item is not None
        else returned_slots
    )
    missing_slots = max(0, expected_slots - returned_slots)
    empty_or_missing = missing_slots + (returned_slots - nonempty_count)
    nonempty_nonartifact_count = max(0, nonempty_count - suspicious_count)

    return {
        "num_items": len(results),
        "requested_continuations_per_item": requested_per_item,
        "expected_slots": expected_slots,
        "returned_slots": returned_slots,
        "missing_slots": missing_slots,
        "nonempty_count": nonempty_count,
        "empty_or_missing_count": empty_or_missing,
        "empty_or_missing_rate_percent": _percentage(empty_or_missing, expected_slots),
        "heuristic_artifact_count": suspicious_count,
        "heuristic_artifact_rate_nonempty_percent": _percentage(
            suspicious_count, nonempty_count
        ),
        "nonempty_nonartifact_count": nonempty_nonartifact_count,
        "nonempty_nonartifact_rate_expected_percent": _percentage(
            nonempty_nonartifact_count, expected_slots
        ),
        "exact_duplicate_excess_count": duplicate_excess_count,
        "exact_duplicate_excess_rate_nonempty_percent": _percentage(
            duplicate_excess_count, nonempty_count
        ),
        "items_with_empty": items_with_empty,
        "items_with_empty_or_missing": items_with_empty_or_missing,
        "items_with_heuristic_artifacts": items_with_suspicious,
        "items_with_exact_duplicates": items_with_duplicates,
        "word_count_nonempty": {
            "mean": statistics.fmean(word_counts) if word_counts else None,
            "median": statistics.median(word_counts) if word_counts else None,
        },
        "flagged_items": flagged_items,
        "interpretation": (
            "These diagnostics detect empty outputs, heuristic text artifacts, and exact "
            "within-item repetition; they do not measure topicality, fluency, or reading support."
        ),
    }


def run(args) -> int:
    with args.results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = summarize_task3_quality(payload)
    summary["results_path"] = str(args.results_path)

    output_path: Path = args.output_path or args.results_path.with_name(
        f"{args.results_path.stem}__generation_quality.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, summary)

    printable = {key: value for key, value in summary.items() if key != "flagged_items"}
    print(json.dumps(printable, indent=2))
    print(f"[INFO] Wrote Task-3 generation-quality summary to {output_path}")
    return 0
