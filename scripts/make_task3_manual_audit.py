#!/usr/bin/env python3
"""Create a blind manual-audit sheet for Task 3 continuations.

This is intentionally standalone and not wired into the project CLI so it can
be removed after the paper audit. It reads the canonical Task 3 JSON outputs
and writes:

1. task3_manual_audit_annotation.csv
   A blind sheet for manual labels.
2. task3_manual_audit_key.csv
   A private key mapping blind rows back to the source model and continuation.

Suggested label set for `human_label`:
  reading_1, reading_2, both_or_ambiguous, neither_or_invalid
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_LLAMA_FILE = Path("results/task3/no_trailing_quotes/llama8b_ambiguous.json")
DEFAULT_LLADA_FILE = Path("results/task3/no_trailing_quotes/llada8b_ambiguous.json")
DEFAULT_OUTPUT_DIR = Path("results/task3/manual_audit")


def normalize_text(text: Any) -> str:
    """Collapse whitespace so CSV rows stay readable in spreadsheet tools."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(payload.get("results", [])):
        item_id = str(item.get("id") or item.get("row_id") or idx)
        out[item_id] = item
    return out


def gold_readings(item: dict[str, Any]) -> list[str]:
    side = item.get("ambiguity_side")
    readings: list[str] = []

    for disambig in item.get("gold_disambiguations", []):
        text = ""
        if side in {"premise", "hypothesis"}:
            text = disambig.get(side) or ""
        if not text:
            text = disambig.get("premise") or disambig.get("hypothesis") or ""
        text = normalize_text(text)
        if text and text not in readings:
            readings.append(text)

    return readings


def choose_continuations(item: dict[str, Any], count: int, rng: random.Random) -> list[tuple[int, str]]:
    continuations = [
        (idx, normalize_text(text))
        for idx, text in enumerate(item.get("continuations", []))
        if normalize_text(text)
    ]
    if len(continuations) <= count:
        return continuations
    return rng.sample(continuations, count)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(args.seed)

    llama = load_results(args.llama_file)
    llada = load_results(args.llada_file)
    common_ids = sorted(set(llama) & set(llada))
    if not common_ids:
        raise ValueError("No shared instance ids found between the two Task 3 files.")

    sample_size = min(args.num_instances, len(common_ids))
    sampled_ids = rng.sample(common_ids, sample_size)

    blind_labels = ["Model A", "Model B"]
    rng.shuffle(blind_labels)
    model_blind = {
        "llama": blind_labels[0],
        "llada": blind_labels[1],
    }

    annotation_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, str]] = []
    row_counter = 1

    for instance_id in sampled_ids:
        # Use LLaMA metadata for prompt/gold fields; they are shared by design.
        base_item = llama[instance_id]
        readings = gold_readings(base_item)
        reading_1 = readings[0] if len(readings) > 0 else ""
        reading_2 = readings[1] if len(readings) > 1 else ""
        extra_readings = " || ".join(readings[2:]) if len(readings) > 2 else ""

        for actual_model, source_file, item in [
            ("llama", str(args.llama_file), llama[instance_id]),
            ("llada", str(args.llada_file), llada[instance_id]),
        ]:
            for continuation_idx, continuation in choose_continuations(
                item,
                args.continuations_per_model,
                rng,
            ):
                blind_id = f"T3A-{row_counter:04d}"
                row_counter += 1

                annotation_rows.append(
                    {
                        "blind_id": blind_id,
                        "instance_id": instance_id,
                        "model_blind": model_blind[actual_model],
                        "ambiguous_sentence": normalize_text(base_item.get("ambiguous_sentence")),
                        "ambiguity_side": normalize_text(base_item.get("ambiguity_side")),
                        "gold_reading_1": reading_1,
                        "gold_reading_2": reading_2,
                        "gold_readings_extra": extra_readings,
                        "continuation": continuation,
                        "human_label": "",
                        "notes": "",
                    }
                )
                key_rows.append(
                    {
                        "blind_id": blind_id,
                        "instance_id": instance_id,
                        "model_blind": model_blind[actual_model],
                        "actual_model": actual_model,
                        "source_file": source_file,
                        "original_continuation_index": str(continuation_idx),
                    }
                )

    rng.shuffle(annotation_rows)
    return annotation_rows, key_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    annotation_rows, key_rows = build_rows(args)

    annotation_path = args.output_dir / "task3_manual_audit_annotation.csv"
    key_path = args.output_dir / "task3_manual_audit_key.csv"

    write_csv(annotation_path, annotation_rows)
    write_csv(key_path, key_rows)

    print(f"Wrote blind annotation sheet: {annotation_path}")
    print(f"Wrote private key:            {key_path}")
    print(f"Rows to annotate:            {len(annotation_rows)}")
    print("Suggested labels: reading_1, reading_2, both_or_ambiguous, neither_or_invalid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-file", type=Path, default=DEFAULT_LLAMA_FILE)
    parser.add_argument("--llada-file", type=Path, default=DEFAULT_LLADA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-instances", type=int, default=25)
    parser.add_argument("--continuations-per-model", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
