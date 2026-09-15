#!/usr/bin/env python3
"""Label a Task 3 manual-audit sheet with an NLI model.

This is intentionally standalone and removable. It does not create human
annotations; it creates model-assisted labels that can be compared against or
used to prioritize a manual audit.

Default input:
  results/task3/manual_audit_full/task3_manual_audit_annotation.csv

Default output:
  results/task3/manual_audit_full/task3_manual_audit_nli_labels.csv

Labels:
  reading_1, reading_2, both_or_ambiguous, neither_or_invalid
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("results/task3/manual_audit_full/task3_manual_audit_annotation.csv")
DEFAULT_OUTPUT = Path("results/task3/manual_audit_full/task3_manual_audit_nli_labels.csv")
DEFAULT_CACHE_DIR = Path("models/huggingface")
DEFAULT_MODEL = "roberta-large-mnli"
VALID_LABELS = {"reading_1", "reading_2", "both_or_ambiguous", "neither_or_invalid"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_nli_scores(raw: Any) -> dict[str, float]:
    """Return label -> score for one NLI pipeline result."""
    if isinstance(raw, dict):
        raw_items = [raw]
    else:
        raw_items = list(raw or [])

    scores: dict[str, float] = {}
    for item in raw_items:
        label = str(item.get("label", "")).lower()
        if label == "label_0":
            label = "contradiction"
        elif label == "label_1":
            label = "neutral"
        elif label == "label_2":
            label = "entailment"
        scores[label] = float(item.get("score", 0.0))
    return scores


def run_nli(pipe: Any, pairs: list[dict[str, str]], batch_size: int) -> list[dict[str, float]]:
    """Run the NLI pipeline while supporting old and new transformers APIs."""
    try:
        outputs = pipe(
            pairs,
            truncation=True,
            max_length=512,
            batch_size=batch_size,
            top_k=None,
        )
    except TypeError:
        outputs = pipe(
            pairs,
            truncation=True,
            max_length=512,
            batch_size=batch_size,
            return_all_scores=True,
        )
    return [normalize_nli_scores(item) for item in outputs]


def choose_label(score_1: float, score_2: float, threshold: float, margin: float) -> str:
    r1 = score_1 >= threshold
    r2 = score_2 >= threshold
    if r1 and r2:
        if abs(score_1 - score_2) <= margin:
            return "both_or_ambiguous"
        return "reading_1" if score_1 > score_2 else "reading_2"
    if r1:
        return "reading_1"
    if r2:
        return "reading_2"
    return "neither_or_invalid"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blind_id",
        "human_label",
        "notes",
        "reading_1_entailment_score",
        "reading_2_entailment_score",
        "reading_1_nli_label",
        "reading_2_nli_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--both-margin",
        type=float,
        default=0.10,
        help="If both readings pass threshold and scores differ by at most this margin, label as both_or_ambiguous.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Transformers device index. Defaults to CUDA 0 if available, otherwise CPU.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for NLI labeling. Install the Task 3 evaluation "
            "stack first, e.g. transformers and torch, then rerun this script.\n"
            f"Original import error: {exc}"
        )

    rows = read_rows(args.input_file)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_file}")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NLI model: {args.model}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, cache_dir=args.cache_dir)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    pairs: list[dict[str, str]] = []
    pair_meta: list[tuple[str, str]] = []
    for row in rows:
        continuation = row.get("continuation", "")
        for reading_key in ("gold_reading_1", "gold_reading_2"):
            pairs.append({"text": continuation, "text_pair": row.get(reading_key, "")})
            pair_meta.append((row.get("blind_id", ""), reading_key))

    scores = run_nli(pipe, pairs, args.batch_size)
    by_blind_id: dict[str, dict[str, dict[str, float]]] = {}
    for (blind_id, reading_key), score_map in zip(pair_meta, scores):
        by_blind_id.setdefault(blind_id, {})[reading_key] = score_map

    output_rows: list[dict[str, str]] = []
    for row in rows:
        blind_id = row["blind_id"]
        r1_scores = by_blind_id.get(blind_id, {}).get("gold_reading_1", {})
        r2_scores = by_blind_id.get(blind_id, {}).get("gold_reading_2", {})
        score_1 = r1_scores.get("entailment", 0.0)
        score_2 = r2_scores.get("entailment", 0.0)
        label = choose_label(score_1, score_2, args.threshold, args.both_margin)
        if label not in VALID_LABELS:
            raise AssertionError(f"Invalid label produced: {label}")

        output_rows.append(
            {
                "blind_id": blind_id,
                "human_label": label,
                "notes": f"auto_nli_{args.model}; threshold={args.threshold}; not_human_annotation",
                "reading_1_entailment_score": f"{score_1:.6f}",
                "reading_2_entailment_score": f"{score_2:.6f}",
                "reading_1_nli_label": max(r1_scores, key=r1_scores.get, default=""),
                "reading_2_nli_label": max(r2_scores, key=r2_scores.get, default=""),
            }
        )

    write_rows(args.output_file, output_rows)
    print(f"Wrote NLI-assisted labels: {args.output_file}")
    print(f"Rows labeled: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
