"""Rescore saved Task-1 continuations without running generation again."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from ambient.evaluation.get_log_likelihood import (
    get_log_likelihood,
    get_pseudo_log_likelihood,
)
from ambient.modeling import (
    is_masked_diffusion_family,
    load_model_bundle,
    runtime_environment,
)
from ambient.utils import write_json_atomic


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def aggregate_option(sentence: str, stats: list[dict[str, Any]]) -> dict[str, Any]:
    all_scored = [row for row in stats if row.get("avg_log_odds") is not None]
    clean_scored = [row for row in all_scored if not row.get("flagged_artifact", False)]

    raw_all = [float(row["log_odds"]) for row in all_scored]
    normalized_all = [float(row["avg_log_odds"]) for row in all_scored]
    raw_clean = [float(row["log_odds"]) for row in clean_scored]
    normalized_clean = [float(row["avg_log_odds"]) for row in clean_scored]

    return {
        "sentence": sentence,
        "total_continuations": len(stats),
        "valid_continuations_all": len(all_scored),
        "valid_continuations_clean": len(clean_scored),
        "artifact_rate": (
            (len(all_scored) - len(clean_scored)) / len(stats) if stats else 0.0
        ),
        "empirical_KL_div_all": float(np.mean(raw_all)) if raw_all else None,
        "empirical_KL_div_normalized_all": (
            float(np.mean(normalized_all)) if normalized_all else None
        ),
        "empirical_KL_div_clean": float(np.mean(raw_clean)) if raw_clean else None,
        "empirical_KL_div_normalized_clean": (
            float(np.mean(normalized_clean)) if normalized_clean else None
        ),
    }


def build_scorer(args, model, tokenizer) -> tuple[Callable, str]:
    if args.scoring_method == "pll":
        def score(prompts: list[str], continuations: list[str]) -> list[float | None]:
            return get_pseudo_log_likelihood(
                model,
                tokenizer,
                prompts,
                continuations,
                batch_size=args.batch_size,
                cfg_scale=args.cfg_scale,
                progress_every=args.progress_every,
                progress_label="Task-1 PLL rescore",
            )

        return score, "single_token_pseudo_log_likelihood"

    def score(prompts: list[str], continuations: list[str]) -> list[float | None]:
        return get_log_likelihood(
            model,
            tokenizer,
            prompts,
            continuations,
            mc_nums=[args.mc_num],
            batch_size=args.batch_size,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            progress_every=args.progress_every,
            progress_label="Task-1 MC rescore",
        )[0]

    return score, "random_multi_token_mc_reconstruction"


def find_prompt_file(example_dir: Path) -> Path | None:
    for name in ("prompts.jsonl", "prompts.json"):
        path = example_dir / name
        if path.exists():
            return path
    return None


def rescore_example(
    source_dir: Path,
    destination_dir: Path,
    score: Callable,
    scoring_label: str,
) -> dict[str, Any] | None:
    prompt_path = find_prompt_file(source_dir)
    if prompt_path is None:
        return None

    prompt_meta = read_json(prompt_path)
    ambiguous_sentence = str(prompt_meta.get("ambiguous_sentence") or "")
    readings = prompt_meta.get("disambiguations") or {}
    if not ambiguous_sentence or not readings:
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompt_path, destination_dir / prompt_path.name)

    options: dict[str, Any] = {}
    continuation_counts: list[int] = []
    for reading_key, reading_text in readings.items():
        continuation_path = source_dir / f"{reading_key}.jsonl"
        if not continuation_path.exists():
            continue
        source_rows = read_jsonl(continuation_path)
        continuations = [str(row.get("continuation_clean") or "") for row in source_rows]
        spaced = [text if text.startswith(" ") else f" {text}" for text in continuations]

        paired_prompts = [str(reading_text)] * len(spaced) + [ambiguous_sentence] * len(spaced)
        paired_continuations = spaced + spaced
        paired_scores = score(paired_prompts, paired_continuations)
        conditional_scores = paired_scores[: len(spaced)]
        ambiguous_scores = paired_scores[len(spaced) :]

        rescored_rows: list[dict[str, Any]] = []
        for source_row, cond_loss, ambig_loss in zip(
            source_rows,
            conditional_scores,
            ambiguous_scores,
        ):
            n_tokens = int(source_row.get("n_tokens") or 0)
            raw_gap = (
                float(ambig_loss) - float(cond_loss)
                if cond_loss is not None and ambig_loss is not None
                else None
            )
            rescored_rows.append(
                {
                    "continuation_clean": source_row.get("continuation_clean", ""),
                    "flagged_artifact": bool(source_row.get("flagged_artifact", False)),
                    "n_tokens": n_tokens,
                    "nll_cond": cond_loss,
                    "nll_ambig": ambig_loss,
                    "log_odds": raw_gap,
                    "avg_log_odds": (
                        raw_gap / n_tokens if raw_gap is not None and n_tokens > 0 else None
                    ),
                    "scoring_method": scoring_label,
                }
            )

        write_jsonl(destination_dir / f"{reading_key}.jsonl", rescored_rows)
        options[reading_key] = aggregate_option(str(reading_text), rescored_rows)
        continuation_counts.append(len(rescored_rows))

    if not options:
        return None

    return {
        "instance_id": source_dir.name,
        "id": source_dir.name,
        "row_id": source_dir.name.split("_", 1)[0],
        "ambiguous_sentence": ambiguous_sentence,
        "generator_model": "preserved_from_source_run",
        "num_conts": min(continuation_counts) if continuation_counts else 0,
        "mc_num": args_mc_num_placeholder(scoring_label),
        "scoring_summary": {
            "method": scoring_label,
            "source_example_dir": str(source_dir),
        },
        "options": options,
    }


def args_mc_num_placeholder(scoring_label: str) -> int | None:
    """Retain the summary schema while distinguishing deterministic PLL."""
    return None if scoring_label == "single_token_pseudo_log_likelihood" else -1


def load_completed_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(summary_path) if row.get("id") is not None}


def run(args) -> int:
    if not is_masked_diffusion_family(args.model_family):
        raise ValueError("task1 rescore currently supports masked-diffusion families only")

    source_root = args.run_dir / args.example_dir_name
    if not source_root.exists():
        raise FileNotFoundError(
            f"No saved Task-1 example directories found at {source_root}. "
            "Run this command on the workstation copy containing example_dirs."
        )

    output_dir = args.output_dir or args.run_dir / f"rescore_{args.scoring_method}"
    output_examples = output_dir / "example_dirs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_examples.mkdir(parents=True, exist_ok=True)
    summary_name = "summary_pll.jsonl" if args.scoring_method == "pll" else f"summary_mc{args.mc_num}.jsonl"
    summary_path = output_dir / summary_name

    source_meta = {}
    source_meta_path = args.run_dir / "run_meta.json"
    if source_meta_path.exists():
        source_meta = read_json(source_meta_path)
    model_id = args.model_id or source_meta.get("model_id")

    meta = {
        "task": "task1_rescore_existing_continuations",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run_dir": str(args.run_dir),
        "model_family": args.model_family,
        "model_id": model_id,
        "scoring_method": args.scoring_method,
        "mc_num": args.mc_num if args.scoring_method == "mc" else None,
        "batch_size": args.batch_size,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "runtime_environment": runtime_environment(),
        "status": "running",
    }
    write_json_atomic(output_dir / "run_meta.json", meta)

    bundle = load_model_bundle(
        args.model_family,
        model_id=model_id,
        use_4bit=args.use_4bit,
        verbose=True,
    )
    score, scoring_label = build_scorer(args, bundle.model, bundle.tokenizer)
    meta["model_id"] = bundle.model_id
    meta["use_4bit"] = bundle.use_4bit

    example_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if args.max_examples is not None:
        example_dirs = example_dirs[: args.max_examples]

    completed = set() if args.overwrite else load_completed_ids(summary_path)
    pending = [path for path in example_dirs if path.name not in completed]
    mode = "w" if args.overwrite else "a"
    print(
        f"[INFO] Rescoring {len(pending)} examples with {scoring_label}; "
        f"{len(completed)} already complete."
    )

    with summary_path.open(mode, encoding="utf-8") as summary_handle:
        for example_dir in tqdm(pending, desc=f"Task-1 {args.scoring_method} rescore"):
            summary = rescore_example(
                example_dir,
                output_examples / example_dir.name,
                score,
                scoring_label,
            )
            if summary is None:
                print(f"[WARN] Skipping incomplete example directory: {example_dir}")
                continue
            if args.scoring_method == "mc":
                summary["mc_num"] = args.mc_num
            summary_handle.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
            summary_handle.flush()

    meta["status"] = "finished"
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["num_examples"] = len(example_dirs)
    write_json_atomic(output_dir / "run_meta.json", meta)
    print(f"[INFO] Rescored summary written to {summary_path}")
    return 0
