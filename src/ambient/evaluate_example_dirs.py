#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze continuation lengths in AMBIENT-style result folders and verify that the
saved `n_tokens` value matches the tokenizer-based recomputation.

Expected folder structure:
./<model-name>/example_dirs/<id>/

Each <id> directory may contain:
- prompts.jsonl
- y0.jsonl / y1.jsonl / y2.jsonl / y3.jsonl / d.jsonl / ...

What this script does:
1. Detects the correct tokenizer from the model folder name:
   - contains "llama" -> Meta-Llama tokenizer
   - contains "llada" -> LLaDA tokenizer
2. Reads every continuation row from every continuation jsonl file.
3. Computes:
   - chars
   - regex word count
   - whitespace split count
   - saved n_tokens
   - recomputed tokenizer token count
   - whether saved n_tokens matches recomputed token count
4. Prints aggregate statistics per model for:
   - all rows
   - clean_only (flagged_artifact == false)
   - flagged_only (flagged_artifact == true)
5. Writes:
   - details CSV with one row per continuation
   - summary CSV with aggregate stats
   - mismatches CSV with only token mismatches

Usage:
python analyze_continuation_lengths.py \
    --roots results/llama8b-n100 results/llada8b-n100 \
    --out-summary continuation_length_summary.csv \
    --out-details continuation_length_details.csv \
    --out-mismatches continuation_token_mismatches.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer


LLAMA_TOKENIZER_ID = "meta-llama/Meta-Llama-3.1-8B"
LLADA_TOKENIZER_ID = "GSAI-ML/LLaDA-8B-Base"

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass
class RowResult:
    model_name: str
    example_id: str
    continuation_file: str
    row_index: int
    flagged_artifact: Optional[bool]
    text: str
    char_count: int
    word_count: int
    whitespace_count: int
    stored_n_tokens: Optional[int]
    recomputed_tokens: Optional[int]
    token_match: Optional[bool]


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except Exception:
        return None


def iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = safe_json_loads(line)
            if obj is not None:
                yield obj


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def count_whitespace_splits(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.split())


def count_tokens_hf(text: str, tokenizer) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        out = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return int(out["input_ids"].shape[1])
    except Exception:
        return None


def avg(vals: List[Optional[int]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return float(mean(clean)) if clean else None


def med(vals: List[Optional[int]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return float(median(clean)) if clean else None


def choose_tokenizer_name(model_name: str) -> Optional[str]:
    lower = model_name.lower()
    if "llama" in lower:
        return "llama"
    if "llada" in lower:
        return "llada"
    return None


def load_needed_tokenizers(model_names: List[str]) -> Dict[str, Any]:
    needed = {choose_tokenizer_name(m) for m in model_names}
    needed.discard(None)

    tokenizers: Dict[str, Any] = {}

    if "llama" in needed:
        print(f"[INFO] Loading LLaMA tokenizer: {LLAMA_TOKENIZER_ID}")
        tokenizers["llama"] = AutoTokenizer.from_pretrained(
            LLAMA_TOKENIZER_ID,
            trust_remote_code=True,
        )

    if "llada" in needed:
        print(f"[INFO] Loading LLaDA tokenizer: {LLADA_TOKENIZER_ID}")
        tokenizers["llada"] = AutoTokenizer.from_pretrained(
            LLADA_TOKENIZER_ID,
            trust_remote_code=True,
        )

    return tokenizers


def find_continuation_files(example_dir: Path) -> List[Path]:
    preferred = ["y0.jsonl", "y1.jsonl", "y2.jsonl", "y3.jsonl", "d.jsonl"]
    out: List[Path] = []

    for name in preferred:
        p = example_dir / name
        if p.exists():
            out.append(p)

    existing = {p.name for p in out}
    for p in sorted(example_dir.glob("*.jsonl")):
        if p.name.startswith("prompts"):
            continue
        if p.name not in existing:
            out.append(p)

    return out


def analyze_root(root: Path, tokenizer_map: Dict[str, Any]) -> List[RowResult]:
    if root.name == "example_dirs":
        model_name = root.parent.name
        example_dirs_root = root
    else:
        model_name = root.name
        example_dirs_root = root / "example_dirs"

    if not example_dirs_root.exists():
        print(f"[WARN] Skipping {root}: no example_dirs found")
        return []

    tokenizer_key = choose_tokenizer_name(model_name)
    tokenizer = tokenizer_map.get(tokenizer_key)

    if tokenizer_key is None:
        print(f"[WARN] Could not infer tokenizer from model name '{model_name}'.")
    else:
        print(f"[INFO] Using tokenizer '{tokenizer_key}' for model '{model_name}'")

    results: List[RowResult] = []

    for example_dir in sorted(p for p in example_dirs_root.iterdir() if p.is_dir()):
        example_id = example_dir.name

        for cont_file in find_continuation_files(example_dir):
            for row_idx, obj in enumerate(iter_jsonl_rows(cont_file)):
                text = (obj.get("continuation_clean") or "").strip()
                flagged = obj.get("flagged_artifact")

                stored_n_tokens = obj.get("n_tokens")
                if isinstance(stored_n_tokens, int):
                    stored_n_tokens_int: Optional[int] = stored_n_tokens
                else:
                    stored_n_tokens_int = None

                recomputed_tokens = count_tokens_hf(text, tokenizer)

                token_match: Optional[bool] = None
                if stored_n_tokens_int is not None and recomputed_tokens is not None:
                    token_match = (stored_n_tokens_int == recomputed_tokens)

                results.append(
                    RowResult(
                        model_name=model_name,
                        example_id=example_id,
                        continuation_file=cont_file.name,
                        row_index=row_idx,
                        flagged_artifact=flagged if isinstance(flagged, bool) else None,
                        text=text,
                        char_count=len(text),
                        word_count=count_words(text),
                        whitespace_count=count_whitespace_splits(text),
                        stored_n_tokens=stored_n_tokens_int,
                        recomputed_tokens=recomputed_tokens,
                        token_match=token_match,
                    )
                )

    return results


def summarize_subset(rows: List[RowResult]) -> Dict[str, Any]:
    comparable = [r for r in rows if r.token_match is not None]
    mismatches = [r for r in comparable if r.token_match is False]

    return {
        "num_rows": len(rows),
        "num_token_comparable": len(comparable),
        "num_token_mismatches": len(mismatches),
        "share_token_mismatches": (len(mismatches) / len(comparable)) if comparable else None,
        "avg_chars_per_continuation": avg([r.char_count for r in rows]),
        "median_chars_per_continuation": med([r.char_count for r in rows]),
        "avg_words_per_continuation": avg([r.word_count for r in rows]),
        "median_words_per_continuation": med([r.word_count for r in rows]),
        "avg_whitespace_splits_per_continuation": avg([r.whitespace_count for r in rows]),
        "median_whitespace_splits_per_continuation": med([r.whitespace_count for r in rows]),
        "avg_stored_tokens_per_continuation": avg([r.stored_n_tokens for r in rows]),
        "median_stored_tokens_per_continuation": med([r.stored_n_tokens for r in rows]),
        "avg_recomputed_tokens_per_continuation": avg([r.recomputed_tokens for r in rows]),
        "median_recomputed_tokens_per_continuation": med([r.recomputed_tokens for r in rows]),
    }


def build_summary_rows(all_rows: List[RowResult]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    model_names = sorted(set(r.model_name for r in all_rows))

    for model_name in model_names:
        model_rows = [r for r in all_rows if r.model_name == model_name]
        clean_rows = [r for r in model_rows if r.flagged_artifact is False]
        flagged_rows = [r for r in model_rows if r.flagged_artifact is True]

        for subset_name, subset_rows in [
            ("all", model_rows),
            ("clean_only", clean_rows),
            ("flagged_only", flagged_rows),
        ]:
            s = summarize_subset(subset_rows)
            s["model_name"] = model_name
            s["subset"] = subset_name
            s["num_examples"] = len(set(r.example_id for r in subset_rows))
            summary_rows.append(s)

    global_all = summarize_subset(all_rows)
    global_all["model_name"] = "__GLOBAL__"
    global_all["subset"] = "all"
    global_all["num_examples"] = len(set((r.model_name, r.example_id) for r in all_rows))
    summary_rows.append(global_all)

    return summary_rows


def fmt_num(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.3f}%"


def print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n=== Continuation Length Summary ===\n")
    header = (
        f"{'model':<20} {'subset':<14} {'rows':>8} {'examples':>10} "
        f"{'avg_words':>12} {'avg_tokens':>12} {'mismatch%':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['model_name']:<20} "
            f"{row['subset']:<14} "
            f"{fmt_num(row['num_rows']):>8} "
            f"{fmt_num(row['num_examples']):>10} "
            f"{fmt_num(row['avg_words_per_continuation']):>12} "
            f"{fmt_num(row['avg_stored_tokens_per_continuation']):>12} "
            f"{fmt_pct(row['share_token_mismatches']):>12}"
        )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[WARN] Not writing empty CSV: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out-summary", default="continuation_length_summary.csv")
    parser.add_argument("--out-details", default="continuation_length_details.csv")
    parser.add_argument("--out-mismatches", default="continuation_token_mismatches.csv")
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]

    model_names: List[str] = []
    for root in roots:
        if root.name == "example_dirs":
            model_names.append(root.parent.name)
        else:
            model_names.append(root.name)

    tokenizer_map = load_needed_tokenizers(model_names)

    all_rows: List[RowResult] = []

    for root in roots:
        print(f"[INFO] Analyzing root: {root}")
        rows = analyze_root(root, tokenizer_map)
        print(f"[INFO]   -> found {len(rows)} continuations")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No continuations found.")
        return

    summary_rows = build_summary_rows(all_rows)
    print_summary(summary_rows)

    detail_rows = [asdict(r) for r in all_rows]
    mismatch_rows = [asdict(r) for r in all_rows if r.token_match is False]

    write_csv(Path(args.out-details if False else args.out_details), detail_rows)
    write_csv(Path(args.out_summary), summary_rows)
    write_csv(Path(args.out_mismatches), mismatch_rows)

    print(f"[INFO] Wrote detailed rows to: {args.out_details}")
    print(f"[INFO] Wrote summary to: {args.out_summary}")
    print(f"[INFO] Wrote mismatch rows to: {args.out_mismatches}")


if __name__ == "__main__":
    main()