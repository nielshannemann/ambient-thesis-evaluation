#!/usr/bin/env python3
"""
Task 6: LLM-as-a-judge evaluation for explicit disambiguation outputs.

This evaluator preserves the original blind A/B setup and extends it with
optional multi-judge aggregation, agreement statistics, and a saved JSON report.

Compatibility invariants:
- supplying one judge model reproduces the old resolved winner counts
- invalid judge outputs still back off to "Tie" for score aggregation
- Task-6 generation files are read in the current {"metadata", "results"} shape
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from ambient.adapters import ARAdapter
from ambient.constants import TASK6_JUDGE_MODEL_ID, TASK6_SECONDARY_JUDGE_MODEL_ID


WINNER_LABELS = ("LLaDA", "LLaMA-8B", "Tie")


def set_global_determinism(seed: int) -> None:
    """Lock all RNGs used by the judge pipeline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def load_results(path: Path) -> Dict[str, dict]:
    """Load generated Task-6 disambiguations keyed by instance id."""
    data: Dict[str, dict] = {}
    if not path.exists():
        print(f"[error] File not found: {path}")
        return data

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for obj in payload.get("results", []):
        data[str(obj["id"])] = obj
    return data


def get_context_and_claim(instance: dict) -> tuple[str, str]:
    """Read the conversational pair used for Task 6, with legacy fallbacks."""
    context = instance.get("context_text")
    claim = instance.get("claim_text")

    if context is None:
        context = instance.get("premise", "")
    if claim is None:
        claim = instance.get("hypothesis", "")

    return context, claim


def resolve_judge_model_ids(args: Any) -> List[str]:
    """Resolve judge model ids with a backward-compatible single-model alias."""
    if getattr(args, "judge_models", None):
        return list(args.judge_models)
    if getattr(args, "judge_model", None):
        return [args.judge_model]

    default_models = getattr(args, "default_judge_models", None)
    if default_models:
        return list(default_models)

    return [TASK6_JUDGE_MODEL_ID, TASK6_SECONDARY_JUDGE_MODEL_ID]


def parse_judge_response(judge_response: str, llada_is_model_a: bool) -> tuple[str, bool]:
    """
    Map the raw A/B judge output back to architecture labels.

    Invalid parses are counted separately but still fall back to "Tie" so the
    single-judge score totals remain backward compatible.
    """
    cleaned = (judge_response or "").strip()
    if cleaned.startswith("Model A"):
        return ("LLaDA" if llada_is_model_a else "LLaMA-8B"), True
    if cleaned.startswith("Model B"):
        return ("LLaMA-8B" if llada_is_model_a else "LLaDA"), True
    if cleaned.startswith("Tie"):
        return "Tie", True
    return "Tie", False


def build_consensus_label(labels: Sequence[str]) -> str:
    """Return the unanimous label, otherwise the conservative fallback Tie."""
    if not labels:
        return "Tie"
    first = labels[0]
    return first if all(label == first for label in labels[1:]) else "Tie"


def compute_cohens_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    categories: Sequence[str] = WINNER_LABELS,
) -> float:
    """Compute Cohen's kappa over a fixed label set."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires equally sized label sequences.")
    if not labels_a:
        return float("nan")

    index = {label: idx for idx, label in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)

    for label_a, label_b in zip(labels_a, labels_b):
        matrix[index[label_a], index[label_b]] += 1.0

    n = float(matrix.sum())
    if n == 0.0:
        return float("nan")

    observed = float(np.trace(matrix) / n)
    row_marginals = matrix.sum(axis=1) / n
    col_marginals = matrix.sum(axis=0) / n
    expected = float(np.dot(row_marginals, col_marginals))

    if np.isclose(1.0 - expected, 0.0):
        return 1.0 if np.isclose(observed, 1.0) else 0.0
    return float((observed - expected) / (1.0 - expected))


def build_pairwise_agreement(results: Sequence[dict], judge_models: Sequence[str]) -> List[Dict[str, Any]]:
    """Compute raw agreement and Cohen's kappa for each judge pair."""
    comparisons: List[Dict[str, Any]] = []

    for judge_a, judge_b in combinations(judge_models, 2):
        labels_a: List[str] = []
        labels_b: List[str] = []

        for row in results:
            decisions = row.get("judges", {})
            if judge_a not in decisions or judge_b not in decisions:
                continue
            labels_a.append(decisions[judge_a]["winner_model"])
            labels_b.append(decisions[judge_b]["winner_model"])

        raw_agreement = float(np.mean([a == b for a, b in zip(labels_a, labels_b)])) if labels_a else float("nan")
        comparisons.append(
            {
                "judge_a": judge_a,
                "judge_b": judge_b,
                "num_instances": len(labels_a),
                "raw_agreement": raw_agreement,
                "cohen_kappa": compute_cohens_kappa(labels_a, labels_b),
            }
        )

    return comparisons


def serialize_score_counter(counter: Counter[str]) -> Dict[str, int]:
    """Return score counts with a stable label order."""
    return {label: int(counter.get(label, 0)) for label in WINNER_LABELS}


def prepare_evaluation_rows(
    llada_data: Dict[str, dict],
    llama_data: Dict[str, dict],
    common_ids: Iterable[str],
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Prepare deterministic Task-6 comparison rows before judge inference."""
    rows: List[Dict[str, Any]] = []

    for idx_num, idx in enumerate(common_ids):
        llada_instance = llada_data[idx]
        llama_instance = llama_data[idx]

        context_text, claim_text = get_context_and_claim(llada_instance)
        cont_llada_list = llada_instance.get("generated_clean", [])
        cont_llama_list = llama_instance.get("generated_clean", [])

        rows.append(
            {
                "id": idx,
                "context_text": context_text,
                "claim_text": claim_text,
                "llada_continuation": cont_llada_list[0] if cont_llada_list else "",
                "llama_continuation": cont_llama_list[0] if cont_llama_list else "",
                "instance_seed": base_seed + idx_num,
                "judges": {},
            }
        )

    return rows


def build_evaluation_prompt(
    premise: str,
    hypothesis: str,
    cont_llada: str,
    cont_llama: str,
    instance_seed: int,
) -> tuple[str, bool]:
    """Create the blind A/B prompt and deterministic model ordering."""
    rng = random.Random(instance_seed)
    llada_is_model_a = rng.choice([True, False])

    model_a_text = cont_llada if llada_is_model_a else cont_llama
    model_b_text = cont_llama if llada_is_model_a else cont_llada

    user_prompt = f"""--- Example ---
Context: I'm afraid the cat was hit by a car.
Claim: The cat was not hit by a car.

Model A's interpretation:
1. The cat died. Then the claim is false.

Model B's interpretation:
1. I'm worried the cat was hit. Then the claim is inconclusive.
2. I'm sorry to share that the cat was hit. Then the claim is false.

Winner (Model A, Model B, or Tie): Model B
--- End of Example ---

Context: {premise}
Claim: {hypothesis}

Model A's interpretation:
{model_a_text}

Model B's interpretation:
{model_b_text}

Winner (Model A, Model B, or Tie):"""

    return user_prompt, llada_is_model_a


def evaluate_pair(
    judge_adapter: ARAdapter,
    premise: str,
    hypothesis: str,
    cont_llada: str,
    cont_llama: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Query a single judge model for one blind A/B comparison."""
    user_prompt, llada_is_model_a = build_evaluation_prompt(
        premise,
        hypothesis,
        cont_llada,
        cont_llama,
        instance_seed,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial judge evaluating AI language models based on "
                "how well they identify ambiguity. A good interpretation explicitly "
                "states the different ways the context can be understood and how it "
                "affects the claim. You must output STRICTLY 'Model A', 'Model B', "
                "or 'Tie' as your final answer, nothing else."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    tokenizer = judge_adapter.tokenizer
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    raw_responses = judge_adapter.generate(
        prompt=input_text,
        num_return_sequences=1,
        batch_size=1,
        temperature=0.01,
        top_k=1,
        max_new_tokens=10,
        stop_at_sentence=False,
        seed=instance_seed,
    )

    raw_response = raw_responses[0].strip() if raw_responses else ""
    winner_model, valid_parse = parse_judge_response(raw_response, llada_is_model_a)

    return {
        "winner_model": winner_model,
        "raw_response": raw_response,
        "llada_position": "A" if llada_is_model_a else "B",
        "valid_parse": valid_parse,
        "queried": True,
    }


def load_judge_adapter(model_id: str, disable_4bit: bool) -> ARAdapter:
    """Load one judge model at a time to keep GPU memory bounded."""
    load_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "cache_dir": "./models",
    }

    if not disable_4bit:
        print("[info] Activating 4-bit Quantization (BitsAndBytes NF4)...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("[info] 4-bit Quantization DISABLED.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return ARAdapter(model_name=model_id, model=model, tokenizer=tokenizer, ar_score_fn=None)


def summarize_results(
    rows: Sequence[dict],
    judge_models: Sequence[str],
    per_judge_scores: Dict[str, Counter[str]],
    invalid_parse_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Build the saved JSON summary."""
    consensus_counter: Counter[str] = Counter()
    for row in rows:
        consensus_counter[row["consensus"]["agree_else_tie"]] += 1

    pairwise = build_pairwise_agreement(rows, judge_models)
    summary: Dict[str, Any] = {
        "num_instances": len(rows),
        "per_judge": {
            judge_model: {
                "scores": serialize_score_counter(per_judge_scores[judge_model]),
                "invalid_parse_count": int(invalid_parse_counts.get(judge_model, 0)),
            }
            for judge_model in judge_models
        },
        "consensus": {
            "definition": "agree_else_tie",
            "scores": serialize_score_counter(consensus_counter),
        },
        "agreement": {
            "pairwise": pairwise,
        },
    }

    if len(pairwise) == 1:
        summary["agreement"]["raw_agreement"] = pairwise[0]["raw_agreement"]
        summary["agreement"]["cohen_kappa"] = pairwise[0]["cohen_kappa"]

    return summary


def save_payload(payload: Dict[str, Any], output_path: Path) -> None:
    """Persist the Task-6 judge report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run(args) -> int:
    judge_models = resolve_judge_model_ids(args)

    print("=== Starting LLM-as-a-Judge Evaluation Pipeline ===")
    print(f"[info] Judge Models: {judge_models}")
    print(f"[info] Base Seed: {args.seed}")

    set_global_determinism(args.seed)

    llada_data = load_results(args.llada_file)
    llama_data = load_results(args.llama_file)

    common_ids = sorted(set(llada_data.keys()).intersection(set(llama_data.keys())))
    print(f"[info] Found {len(common_ids)} overlapping instances for evaluation.")

    if not common_ids:
        print("[error] No overlapping data found. Exiting.")
        return 1

    rows = prepare_evaluation_rows(llada_data, llama_data, common_ids, base_seed=args.seed)
    per_judge_scores: Dict[str, Counter[str]] = {judge_model: Counter() for judge_model in judge_models}
    invalid_parse_counts: Dict[str, int] = {judge_model: 0 for judge_model in judge_models}

    for judge_model in judge_models:
        print("-" * 72)
        print(f"[info] Initializing judge model: {judge_model}")
        judge_adapter = load_judge_adapter(judge_model, disable_4bit=args.disable_4bit)

        for row in tqdm(rows, desc=f"Judging with {judge_model}"):
            if not row["llada_continuation"] and not row["llama_continuation"]:
                decision = {
                    "winner_model": "Tie",
                    "raw_response": "",
                    "llada_position": None,
                    "valid_parse": None,
                    "queried": False,
                }
            else:
                decision = evaluate_pair(
                    judge_adapter=judge_adapter,
                    premise=row["context_text"],
                    hypothesis=row["claim_text"],
                    cont_llada=row["llada_continuation"],
                    cont_llama=row["llama_continuation"],
                    instance_seed=row["instance_seed"],
                )
                if decision["valid_parse"] is False:
                    invalid_parse_counts[judge_model] += 1

            row["judges"][judge_model] = decision
            per_judge_scores[judge_model][decision["winner_model"]] += 1

        del judge_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for row in rows:
        labels = [row["judges"][judge_model]["winner_model"] for judge_model in judge_models]
        row["consensus"] = {"agree_else_tie": build_consensus_label(labels)}

    summary = summarize_results(rows, judge_models, per_judge_scores, invalid_parse_counts)
    payload = {
        "metadata": {
            "task": "task6_judge_evaluation",
            "llada_file": str(args.llada_file),
            "llama_file": str(args.llama_file),
            "judge_models": judge_models,
            "seed": args.seed,
            "disable_4bit": bool(args.disable_4bit),
        },
        "summary": summary,
        "results": rows,
    }

    save_payload(payload, args.output_path)

    print("\n" + "=" * 50)
    print("--- FINAL LLM-AS-A-JUDGE SCORES ---")
    for judge_model in judge_models:
        print(f"[judge] {judge_model}")
        for label, value in serialize_score_counter(per_judge_scores[judge_model]).items():
            pct = (value / len(rows)) * 100 if rows else 0.0
            print(f"  {label:10s}: {value} ({pct:.2f}%)")
        print(f"  invalid parses: {invalid_parse_counts[judge_model]}")

    if "raw_agreement" in summary["agreement"]:
        print(f"[agreement] raw agreement: {summary['agreement']['raw_agreement']:.4f}")
        print(f"[agreement] Cohen's kappa: {summary['agreement']['cohen_kappa']:.4f}")

    print(f"[info] Saved Task-6 judge report to: {args.output_path}")
    print("=" * 50)
    return 0
