#!/usr/bin/env python3
"""
Task 4: layerwise probing plus von Neumann entropy analyses.

This version preserves the existing ambiguous-input analyses and adds optional
dataset-only negative controls to test whether observed effects are driven by
ambiguity rather than generic rewrite or length effects.

Compatibility invariants:
- existing probe dataset modes stay unchanged
- the gold ambiguous-vs-disambiguated VNE block keeps its current JSON shape
- extra VNE conditions are stored in a new sibling field
"""

from __future__ import annotations

import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from ambient.llada_loader import load_llada_model

warnings.filterwarnings("ignore")

LLADA_MASK_ID = 126336
VALID_BINARY_LABELS = {"entailment", "contradiction"}
DATASET_MODE_CHOICES = ("side_reconstructed", "fully_disambiguated")
PROBE_CONTROL_MODE_CHOICES = ("unambiguous_length_matched",)
VNE_INPUT_MODE_CHOICES = ("sentence_only", "pair_prompt")
VNE_CONTROL_CONDITION_CHOICES = ("distractor_rewrite", "random_matched_rewrite")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


def build_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Question: Does the premise entail or contradict the hypothesis?\n"
        "Answer:"
    )


def approx_token_count(text: str) -> int:
    """Cheap token count proxy used for deterministic control matching."""
    tokens = re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)
    return len(tokens)


def ambiguity_type_name(premise_ambiguous: bool, hypothesis_ambiguous: bool) -> str:
    if premise_ambiguous and hypothesis_ambiguous:
        return "both"
    if premise_ambiguous:
        return "premise"
    if hypothesis_ambiguous:
        return "hypothesis"
    return "none"


def load_examples(path: Path) -> List[dict]:
    examples: List[dict] = []
    if not path.exists():
        print(f"[error] Dataset not found at {path}")
        return examples

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def iter_ambiguous_sides(ex: dict) -> Iterable[str]:
    if ex.get("premise_ambiguous", False):
        yield "premise"
    if ex.get("hypothesis_ambiguous", False):
        yield "hypothesis"


def build_side_text(
    original_premise: str,
    original_hypothesis: str,
    side: str,
    replacement_text: str,
    input_mode: str,
) -> str:
    if input_mode == "sentence_only":
        return replacement_text
    if side == "premise":
        return build_prompt(replacement_text, original_hypothesis)
    return build_prompt(original_premise, replacement_text)


def choose_primary_disambiguation(disambiguations: Sequence[dict], side: str) -> dict | None:
    """Pick one gold rewrite deterministically for control matching."""
    valid: List[dict] = []
    for disambig in disambiguations:
        text = (disambig.get(side) or "").strip()
        if not text:
            continue
        valid.append(
            {
                "text": text,
                "label": (disambig.get("label") or "unknown").lower(),
            }
        )

    if not valid:
        return None

    entailments = [row for row in valid if row["label"] == "entailment"]
    if entailments:
        return sorted(entailments, key=lambda row: (len(row["text"]), row["text"]))[0]

    return sorted(valid, key=lambda row: (row["label"], len(row["text"]), row["text"]))[0]


def choose_disambiguation_closest_to_length(
    disambiguations: Sequence[dict],
    side: str,
    target_length: int,
    length_fn=approx_token_count,
) -> dict | None:
    """Choose the side-specific gold rewrite closest in length to a control text."""
    candidates: List[dict] = []
    for disambig in disambiguations:
        text = (disambig.get(side) or "").strip()
        if not text:
            continue
        candidates.append(
            {
                "text": text,
                "label": (disambig.get("label") or "unknown").lower(),
                "length_tokens": int(length_fn(text)),
            }
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda row: (abs(row["length_tokens"] - target_length), row["length_tokens"], row["label"], row["text"]),
    )


def build_probe_records_from_examples(
    examples: Sequence[dict],
    mode: str,
    max_examples: int = 600,
    length_fn=approx_token_count,
) -> tuple[List[dict], Dict[str, object]]:
    """Parse ambiguous AMBIENT rows into binary probe records."""
    if mode not in DATASET_MODE_CHOICES:
        raise ValueError(f"Unsupported dataset mode: {mode}")

    records: List[dict] = []
    side_counter = Counter()
    pair_counter = Counter()
    used_instance_ids = set()

    for ex in examples:
        disambiguations = ex.get("disambiguations", [])
        if not disambiguations:
            continue

        premise_ambiguous = bool(ex.get("premise_ambiguous", False))
        hypothesis_ambiguous = bool(ex.get("hypothesis_ambiguous", False))
        if not (premise_ambiguous or hypothesis_ambiguous):
            continue

        original_premise = ex.get("premise", "")
        original_hypothesis = ex.get("hypothesis", "")
        instance_id = str(ex.get("id", f"{original_premise} || {original_hypothesis}"))
        ambiguity_type = ambiguity_type_name(premise_ambiguous, hypothesis_ambiguous)

        instance_records = []
        seen_in_instance = set()

        for disambig in disambiguations:
            label = (disambig.get("label") or "").lower()
            if label not in VALID_BINARY_LABELS:
                continue

            if mode == "side_reconstructed":
                if premise_ambiguous:
                    premise = disambig.get("premise", "")
                    hypothesis = original_hypothesis
                    if premise and hypothesis:
                        prompt = build_prompt(premise, hypothesis)
                        key = (prompt, label, instance_id, ambiguity_type)
                        if key not in seen_in_instance:
                            instance_records.append(key)
                            seen_in_instance.add(key)

                if hypothesis_ambiguous:
                    premise = original_premise
                    hypothesis = disambig.get("hypothesis", "")
                    if premise and hypothesis:
                        prompt = build_prompt(premise, hypothesis)
                        key = (prompt, label, instance_id, ambiguity_type)
                        if key not in seen_in_instance:
                            instance_records.append(key)
                            seen_in_instance.add(key)

            else:
                premise = disambig.get("premise", original_premise)
                hypothesis = disambig.get("hypothesis", original_hypothesis)
                if premise and hypothesis:
                    prompt = build_prompt(premise, hypothesis)
                    key = (prompt, label, instance_id, ambiguity_type)
                    if key not in seen_in_instance:
                        instance_records.append(key)
                        seen_in_instance.add(key)

        if not instance_records:
            continue

        if instance_id not in used_instance_ids:
            if len(used_instance_ids) >= max_examples:
                break
            used_instance_ids.add(instance_id)

        side_counter[ambiguity_type] += 1
        for prompt, label, group, source_type in instance_records:
            records.append(
                {
                    "text": prompt,
                    "label": label,
                    "group": group,
                    "source_type": source_type,
                    "prompt_length_tokens": int(length_fn(prompt)),
                }
            )
            pair_counter[(source_type, label)] += 1

    metadata = {
        "mode": mode,
        "num_pairs": len(records),
        "num_unique_instances": len({record["group"] for record in records}),
        "label_distribution": dict(Counter(record["label"] for record in records)),
        "source_ambiguity_distribution_instances": dict(side_counter),
        "source_ambiguity_distribution_pairs": dict(Counter(record["source_type"] for record in records)),
        "pair_distribution_by_source_type_and_label": {
            f"{key[0]}::{key[1]}": value for key, value in pair_counter.items()
        },
    }
    return records, metadata


def probe_records_to_dataset(
    records: Sequence[dict],
    metadata: Dict[str, object],
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, object]]:
    texts = [record["text"] for record in records]
    labels = [record["label"] for record in records]
    groups = [record["group"] for record in records]
    source_types = [record["source_type"] for record in records]
    return texts, labels, groups, source_types, metadata


def build_unambiguous_length_matched_probe_control(
    examples: Sequence[dict],
    reference_records: Sequence[dict],
    length_fn=approx_token_count,
) -> tuple[List[dict], Dict[str, object]]:
    """Match unambiguous binary NLI rows to a reference ambiguous-derived probe set."""
    candidates: Dict[str, List[dict]] = defaultdict(list)
    for ex in examples:
        if bool(ex.get("premise_ambiguous", False)) or bool(ex.get("hypothesis_ambiguous", False)):
            continue

        label = (ex.get("labels") or "").lower()
        if label not in VALID_BINARY_LABELS:
            continue

        premise = ex.get("premise", "")
        hypothesis = ex.get("hypothesis", "")
        if not premise or not hypothesis:
            continue

        prompt = build_prompt(premise, hypothesis)
        instance_id = str(ex.get("id", f"{premise} || {hypothesis}"))
        candidates[label].append(
            {
                "text": prompt,
                "label": label,
                "group": instance_id,
                "source_type": "none",
                "prompt_length_tokens": int(length_fn(prompt)),
            }
        )

    for label in list(candidates.keys()):
        candidates[label] = sorted(
            candidates[label],
            key=lambda row: (row["prompt_length_tokens"], row["group"], row["text"]),
        )

    used_candidates: set[tuple[str, str]] = set()
    matched_records: List[dict] = []
    length_gaps: List[int] = []
    unmatched = 0

    for ref in reference_records:
        label = ref["label"]
        pool = [
            candidate
            for candidate in candidates.get(label, [])
            if (candidate["label"], candidate["group"]) not in used_candidates
        ]
        if not pool:
            unmatched += 1
            continue

        chosen = min(
            pool,
            key=lambda candidate: (
                abs(candidate["prompt_length_tokens"] - ref["prompt_length_tokens"]),
                candidate["prompt_length_tokens"],
                candidate["group"],
                candidate["text"],
            ),
        )
        used_candidates.add((chosen["label"], chosen["group"]))
        matched_records.append(chosen)
        length_gaps.append(abs(chosen["prompt_length_tokens"] - ref["prompt_length_tokens"]))

    metadata = {
        "mode": "unambiguous_length_matched",
        "matched_to_num_reference_pairs": len(reference_records),
        "num_pairs": len(matched_records),
        "num_unique_instances": len({record["group"] for record in matched_records}),
        "label_distribution": dict(Counter(record["label"] for record in matched_records)),
        "source_ambiguity_distribution_instances": {"none": len({record["group"] for record in matched_records})},
        "source_ambiguity_distribution_pairs": {"none": len(matched_records)},
        "pair_distribution_by_source_type_and_label": {
            f"none::{key}": value
            for key, value in Counter(record["label"] for record in matched_records).items()
        },
        "unmatched_reference_pairs": unmatched,
        "mean_length_gap_tokens": float(np.mean(length_gaps)) if length_gaps else None,
        "median_length_gap_tokens": float(np.median(length_gaps)) if length_gaps else None,
        "max_length_gap_tokens": int(max(length_gaps)) if length_gaps else None,
    }
    return matched_records, metadata


def load_probe_dataset(
    path: Path,
    mode: str,
    max_examples: int = 600,
    return_records: bool = False,
):
    """Load an ambiguous-derived probe dataset from disk."""
    examples = load_examples(path)
    records, metadata = build_probe_records_from_examples(examples, mode=mode, max_examples=max_examples)
    dataset = probe_records_to_dataset(records, metadata)
    if return_records:
        return (*dataset, records)
    return dataset


def load_probe_control_dataset(
    path: Path,
    control_mode: str,
    reference_records: Sequence[dict],
):
    """Load a probe-control dataset derived from the existing AMBIENT rows."""
    if control_mode not in PROBE_CONTROL_MODE_CHOICES:
        raise ValueError(f"Unsupported probe control mode: {control_mode}")

    examples = load_examples(path)
    if control_mode == "unambiguous_length_matched":
        records, metadata = build_unambiguous_length_matched_probe_control(examples, reference_records)
        return (*probe_records_to_dataset(records, metadata), records)

    raise ValueError(f"Unsupported probe control mode: {control_mode}")


def build_gold_vne_pairs(
    examples: Sequence[dict],
    input_mode: str,
    max_examples: int = 580,
) -> tuple[List[Dict[str, Any]], Dict[str, object]]:
    """Build the original ambiguous-vs-disambiguated VNE dataset."""
    if input_mode not in VNE_INPUT_MODE_CHOICES:
        raise ValueError(f"Unsupported vne input mode: {input_mode}")

    pairs: List[Dict[str, Any]] = []
    side_counter = Counter()
    label_counter = Counter()
    used_instance_ids = set()

    for ex in examples:
        disambiguations = ex.get("disambiguations", [])
        if not disambiguations:
            continue

        original_premise = ex.get("premise", "")
        original_hypothesis = ex.get("hypothesis", "")
        instance_id = str(ex.get("id", f"{original_premise} || {original_hypothesis}"))

        if not any(iter_ambiguous_sides(ex)):
            continue

        if instance_id not in used_instance_ids:
            if len(used_instance_ids) >= max_examples:
                break
            used_instance_ids.add(instance_id)

        seen = set()
        for disambig in disambiguations:
            label = str(disambig.get("label", "unknown"))
            for side in iter_ambiguous_sides(ex):
                ambiguous_text = (ex.get(side) or "").strip()
                disambiguated_side = (disambig.get(side) or "").strip()
                if not ambiguous_text or not disambiguated_side:
                    continue

                key = (instance_id, side, label, ambiguous_text, disambiguated_side)
                if key in seen:
                    continue

                pairs.append(
                    {
                        "instance_id": instance_id,
                        "side": side,
                        "label": label,
                        "ambiguous_text": build_side_text(
                            original_premise,
                            original_hypothesis,
                            side,
                            ambiguous_text,
                            input_mode,
                        ),
                        "disambiguated_text": build_side_text(
                            original_premise,
                            original_hypothesis,
                            side,
                            disambiguated_side,
                            input_mode,
                        ),
                    }
                )
                seen.add(key)
                side_counter[side] += 1
                label_counter[label] += 1

    metadata = {
        "input_mode": input_mode,
        "num_pairs": len(pairs),
        "num_unique_instances": len({pair["instance_id"] for pair in pairs}),
        "side_distribution": dict(side_counter),
        "label_distribution": dict(label_counter),
    }
    return pairs, metadata


def build_vne_control_pairs(
    examples: Sequence[dict],
    input_mode: str,
    condition: str,
    max_examples: int = 580,
    length_fn=approx_token_count,
) -> tuple[List[Dict[str, Any]], Dict[str, object]]:
    """Build dataset-only VNE negative controls."""
    if condition not in VNE_CONTROL_CONDITION_CHOICES:
        raise ValueError(f"Unsupported VNE control condition: {condition}")

    pools: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        for side in iter_ambiguous_sides(ex):
            for disambig in ex.get("disambiguations", []):
                text = (disambig.get(side) or "").strip()
                label = (disambig.get("label") or "unknown").lower()
                if not text:
                    continue
                pools[side][label].append(
                    {
                        "source_instance_id": str(ex.get("id")),
                        "text": text,
                        "label": label,
                        "length_tokens": int(length_fn(text)),
                    }
                )

    for side in list(pools.keys()):
        for label in list(pools[side].keys()):
            pools[side][label] = sorted(
                pools[side][label],
                key=lambda row: (row["length_tokens"], row["source_instance_id"], row["text"]),
            )

    pairs: List[Dict[str, Any]] = []
    side_counter = Counter()
    label_counter = Counter()
    used_instance_ids = set()
    length_gaps: List[int] = []

    for ex in examples:
        instance_id = str(ex.get("id"))
        if not any(iter_ambiguous_sides(ex)):
            continue

        if instance_id not in used_instance_ids:
            if len(used_instance_ids) >= max_examples:
                break
            used_instance_ids.add(instance_id)

        original_premise = ex.get("premise", "")
        original_hypothesis = ex.get("hypothesis", "")

        for side in iter_ambiguous_sides(ex):
            ambiguous_side_text = (ex.get(side) or "").strip()
            if not ambiguous_side_text:
                continue

            if condition == "distractor_rewrite":
                control_side_text = (ex.get(f"distractor_{side}") or "").strip()
                if not control_side_text:
                    continue

                matched_gold = choose_disambiguation_closest_to_length(
                    ex.get("disambiguations", []),
                    side=side,
                    target_length=int(length_fn(control_side_text)),
                    length_fn=length_fn,
                )
                if matched_gold is None:
                    continue

                length_gap = abs(int(length_fn(control_side_text)) - matched_gold["length_tokens"])
                pair = {
                    "instance_id": instance_id,
                    "side": side,
                    "label": matched_gold["label"],
                    "ambiguous_text": build_side_text(
                        original_premise,
                        original_hypothesis,
                        side,
                        ambiguous_side_text,
                        input_mode,
                    ),
                    "disambiguated_text": build_side_text(
                        original_premise,
                        original_hypothesis,
                        side,
                        control_side_text,
                        input_mode,
                    ),
                    "control_source_instance_id": instance_id,
                    "selection_rule": "same_instance_distractor_closest_gold_length_anchor",
                    "length_gap_tokens": length_gap,
                }
            else:
                anchor = choose_primary_disambiguation(ex.get("disambiguations", []), side=side)
                if anchor is None:
                    continue

                anchor_length = int(length_fn(anchor["text"]))
                pool = [
                    candidate
                    for candidate in pools[side].get(anchor["label"], [])
                    if candidate["source_instance_id"] != instance_id
                ]
                if not pool:
                    continue

                chosen = min(
                    pool,
                    key=lambda candidate: (
                        abs(candidate["length_tokens"] - anchor_length),
                        candidate["length_tokens"],
                        candidate["source_instance_id"],
                        candidate["text"],
                    ),
                )
                length_gap = abs(chosen["length_tokens"] - anchor_length)
                pair = {
                    "instance_id": instance_id,
                    "side": side,
                    "label": anchor["label"],
                    "ambiguous_text": build_side_text(
                        original_premise,
                        original_hypothesis,
                        side,
                        ambiguous_side_text,
                        input_mode,
                    ),
                    "disambiguated_text": build_side_text(
                        original_premise,
                        original_hypothesis,
                        side,
                        chosen["text"],
                        input_mode,
                    ),
                    "control_source_instance_id": chosen["source_instance_id"],
                    "selection_rule": "other_instance_same_label_nearest_length",
                    "length_gap_tokens": length_gap,
                }

            pairs.append(pair)
            side_counter[side] += 1
            label_counter[pair["label"]] += 1
            length_gaps.append(pair["length_gap_tokens"])

    metadata = {
        "condition": condition,
        "input_mode": input_mode,
        "num_pairs": len(pairs),
        "num_unique_instances": len({pair["instance_id"] for pair in pairs}),
        "side_distribution": dict(side_counter),
        "label_distribution": dict(label_counter),
        "mean_length_gap_tokens": float(np.mean(length_gaps)) if length_gaps else None,
        "median_length_gap_tokens": float(np.median(length_gaps)) if length_gaps else None,
        "max_length_gap_tokens": int(max(length_gaps)) if length_gaps else None,
    }
    return pairs, metadata


def load_vne_dataset(
    path: Path,
    input_mode: str,
    max_examples: int = 580,
) -> Tuple[List[Dict[str, Any]], Dict[str, object]]:
    examples = load_examples(path)
    return build_gold_vne_pairs(examples, input_mode=input_mode, max_examples=max_examples)


def load_vne_control_dataset(
    path: Path,
    input_mode: str,
    condition: str,
    max_examples: int = 580,
) -> Tuple[List[Dict[str, Any]], Dict[str, object]]:
    examples = load_examples(path)
    return build_vne_control_pairs(
        examples,
        input_mode=input_mode,
        condition=condition,
        max_examples=max_examples,
    )


def load_model_and_tokenizer(model_id: str, is_llada: bool, use_4bit: bool):
    if is_llada:
        model, tokenizer = load_llada_model(
            hf_model=model_id,
            use_4bit=use_4bit,
            verbose=False,
        )
    else:
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir="./models",
        )

    model.eval()
    tokenizer.padding_side = "right"
    return model, tokenizer


def extract_hidden_states(
    model_id: str,
    texts: List[str],
    is_llada: bool,
    batch_size: int,
    use_4bit: bool,
    include_embedding_layer: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Extract per-layer summary vectors for the probing setup.

    AR:
        final valid causal token state
    LLaDA:
        appended [MASK] summary state
    """
    print(f"[info] Initializing feature extraction pipeline for: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, is_llada=is_llada, use_4bit=use_4bit)

    all_embeddings_by_layer: Dict[int, List[np.ndarray]] = {}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            current_batch_size = input_ids.shape[0]
            seq_lengths = attention_mask.sum(dim=1).long()

            if is_llada:
                new_input_ids = torch.full(
                    (current_batch_size, input_ids.shape[1] + 1),
                    tokenizer.pad_token_id,
                    device=model.device,
                )
                new_attention_mask = torch.zeros(
                    (current_batch_size, input_ids.shape[1] + 1),
                    dtype=attention_mask.dtype,
                    device=model.device,
                )

                for j in range(current_batch_size):
                    length = seq_lengths[j]
                    new_input_ids[j, :length] = input_ids[j, :length]
                    new_input_ids[j, length] = LLADA_MASK_ID
                    new_attention_mask[j, : length + 1] = 1

                outputs = model(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    output_hidden_states=True,
                )
                extraction_indices = seq_lengths
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                extraction_indices = seq_lengths - 1

            hidden_states = outputs.hidden_states
            start_idx = 0 if include_embedding_layer else 1

            for layer_idx in range(start_idx, len(hidden_states)):
                target_hidden_states = hidden_states[layer_idx]
                embeddings = target_hidden_states[
                    torch.arange(current_batch_size, device=target_hidden_states.device),
                    extraction_indices,
                ].detach().float().cpu().numpy()
                all_embeddings_by_layer.setdefault(layer_idx, []).append(embeddings)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_embeddings_by_layer = {
        layer_idx: np.concatenate(chunks, axis=0)
        for layer_idx, chunks in all_embeddings_by_layer.items()
    }

    layer_list = sorted(final_embeddings_by_layer.keys())
    print(
        f"[info] Extracted {len(layer_list)} layers "
        f"({'including' if include_embedding_layer else 'excluding'} embedding layer)."
    )
    print(f"[info] Layer indices: {layer_list[0]} .. {layer_list[-1]}")
    return final_embeddings_by_layer


def binary_entropy_bits_from_positive_probs(p_positive: np.ndarray) -> np.ndarray:
    p_positive = np.clip(p_positive, 1e-12, 1.0 - 1e-12)
    return -(
        p_positive * np.log2(p_positive)
        + (1.0 - p_positive) * np.log2(1.0 - p_positive)
    )


def run_layerwise_probe(
    embeddings_by_layer: Dict[int, np.ndarray],
    labels: List[str],
    groups: List[str],
    seed: int,
) -> Dict[int, Dict[str, object]]:
    """Run grouped CV logistic probes on each layer independently."""
    y = np.array([1 if label == "entailment" else 0 for label in labels], dtype=np.int64)
    groups_arr = np.array(groups)
    dummy_x = np.zeros(len(y), dtype=np.int64)

    cv_strategy = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    cv_splits = list(cv_strategy.split(dummy_x, y, groups_arr))

    results: Dict[int, Dict[str, object]] = {}
    for layer_idx in sorted(embeddings_by_layer.keys()):
        X = embeddings_by_layer[layer_idx]
        heldout_positive_probs = np.zeros(len(y), dtype=np.float64)
        fold_accuracies: List[float] = []
        fold_mean_entropies_bits: List[float] = []

        for train_idx, test_idx in cv_splits:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            preds = (probs >= 0.5).astype(np.int64)
            entropies_bits = binary_entropy_bits_from_positive_probs(probs)

            heldout_positive_probs[test_idx] = probs
            fold_accuracies.append(float(np.mean(preds == y[test_idx])))
            fold_mean_entropies_bits.append(float(np.mean(entropies_bits)))

        all_entropies_bits = binary_entropy_bits_from_positive_probs(heldout_positive_probs)
        results[layer_idx] = {
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "std_accuracy": float(np.std(fold_accuracies)),
            "fold_accuracies": [float(x) for x in fold_accuracies],
            "mean_probe_entropy_bits": float(np.mean(all_entropies_bits)),
            "std_probe_entropy_bits": float(np.std(all_entropies_bits)),
            "median_probe_entropy_bits": float(np.median(all_entropies_bits)),
            "fold_mean_entropies_bits": [float(x) for x in fold_mean_entropies_bits],
            "mean_positive_class_probability": float(np.mean(heldout_positive_probs)),
        }

    return results


def summarize_results(model_name: str, results: Dict[int, Dict[str, object]]) -> None:
    ordered_layers = sorted(results.keys())
    middle_layer = ordered_layers[len(ordered_layers) // 2]
    best_layer = max(ordered_layers, key=lambda layer: results[layer]["mean_accuracy"])
    lowest_entropy_layer = min(
        ordered_layers,
        key=lambda layer: results[layer]["mean_probe_entropy_bits"],
    )

    print(f"[summary] {model_name}")
    print(
        f"  best layer:   {best_layer} | "
        f"{results[best_layer]['mean_accuracy'] * 100:.2f}% "
        f"(+- {results[best_layer]['std_accuracy'] * 100:.2f}%)"
    )
    print(
        f"  middle layer: {middle_layer} | "
        f"{results[middle_layer]['mean_accuracy'] * 100:.2f}% "
        f"(+- {results[middle_layer]['std_accuracy'] * 100:.2f}%)"
    )
    print(
        f"  final layer:  {ordered_layers[-1]} | "
        f"{results[ordered_layers[-1]]['mean_accuracy'] * 100:.2f}% "
        f"(+- {results[ordered_layers[-1]]['std_accuracy'] * 100:.2f}%)"
    )
    print(
        f"  lowest probe-entropy layer: {lowest_entropy_layer} | "
        f"{results[lowest_entropy_layer]['mean_probe_entropy_bits']:.4f} bits"
    )


def von_neumann_entropy_from_token_matrix(
    token_matrix: np.ndarray,
    center_tokens: bool = False,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute von Neumann entropy from the token hidden-state matrix H."""
    if token_matrix.ndim != 2:
        raise ValueError(f"Expected 2D token matrix, got shape={token_matrix.shape}")

    H = token_matrix.astype(np.float64, copy=False)
    num_tokens = int(H.shape[0])

    if num_tokens <= 1:
        return {
            "raw_entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "effective_rank": 1.0,
            "num_tokens": num_tokens,
        }

    if center_tokens:
        H = H - H.mean(axis=0, keepdims=True)

    gram = H @ H.T
    gram = 0.5 * (gram + gram.T)
    trace = float(np.trace(gram))

    if trace <= eps:
        return {
            "raw_entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "effective_rank": 1.0,
            "num_tokens": num_tokens,
        }

    rho = gram / trace
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals[eigvals > eps]

    if eigvals.size == 0:
        raw_entropy_bits = 0.0
    else:
        eigvals = eigvals / eigvals.sum()
        raw_entropy_bits = float(-(eigvals * np.log2(eigvals)).sum())

    max_entropy = math.log2(num_tokens) if num_tokens > 1 else 1.0
    normalized_entropy = raw_entropy_bits / max_entropy if max_entropy > 0 else 0.0
    effective_rank = float(2 ** raw_entropy_bits)

    return {
        "raw_entropy_bits": raw_entropy_bits,
        "normalized_entropy": float(normalized_entropy),
        "effective_rank": effective_rank,
        "num_tokens": num_tokens,
    }


def extract_vne_comparison(
    model_id: str,
    pairs: List[Dict[str, Any]],
    is_llada: bool,
    batch_size: int,
    use_4bit: bool,
    include_embedding_layer: bool = False,
    center_tokens: bool = False,
) -> Dict[int, Dict[str, object]]:
    """Compute layerwise VNE summaries for paired inputs."""
    print(f"[info] Initializing von Neumann extraction pipeline for: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, is_llada=is_llada, use_4bit=use_4bit)

    layer_stats: DefaultDict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "ambiguous_raw_entropy_bits": [],
            "ambiguous_normalized_entropy": [],
            "ambiguous_effective_rank": [],
            "ambiguous_num_tokens": [],
            "disambiguated_raw_entropy_bits": [],
            "disambiguated_normalized_entropy": [],
            "disambiguated_effective_rank": [],
            "disambiguated_num_tokens": [],
            "delta_raw_entropy_bits": [],
            "delta_normalized_entropy": [],
            "delta_effective_rank": [],
        }
    )

    all_texts: List[str] = []
    pair_index: List[Tuple[int, str]] = []
    for idx, pair in enumerate(pairs):
        all_texts.append(pair["ambiguous_text"])
        pair_index.append((idx, "ambiguous"))
        all_texts.append(pair["disambiguated_text"])
        pair_index.append((idx, "disambiguated"))

    per_pair_per_layer: DefaultDict[int, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(dict)

    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            batch_meta = pair_index[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            start_idx = 0 if include_embedding_layer else 1

            attention_mask_cpu = attention_mask.detach().cpu().numpy().astype(bool)

            for layer_idx in range(start_idx, len(hidden_states)):
                layer_h = hidden_states[layer_idx].detach().float().cpu().numpy()

                for local_idx, (pair_id, variant) in enumerate(batch_meta):
                    valid_tokens = attention_mask_cpu[local_idx]
                    token_matrix = layer_h[local_idx][valid_tokens]
                    entropy_info = von_neumann_entropy_from_token_matrix(
                        token_matrix,
                        center_tokens=center_tokens,
                    )
                    per_pair_per_layer[pair_id].setdefault(layer_idx, {})[variant] = entropy_info

    for pair_id in sorted(per_pair_per_layer.keys()):
        for layer_idx, variants in per_pair_per_layer[pair_id].items():
            if "ambiguous" not in variants or "disambiguated" not in variants:
                continue

            amb = variants["ambiguous"]
            dis = variants["disambiguated"]
            stats = layer_stats[layer_idx]

            stats["ambiguous_raw_entropy_bits"].append(float(amb["raw_entropy_bits"]))
            stats["ambiguous_normalized_entropy"].append(float(amb["normalized_entropy"]))
            stats["ambiguous_effective_rank"].append(float(amb["effective_rank"]))
            stats["ambiguous_num_tokens"].append(float(amb["num_tokens"]))

            stats["disambiguated_raw_entropy_bits"].append(float(dis["raw_entropy_bits"]))
            stats["disambiguated_normalized_entropy"].append(float(dis["normalized_entropy"]))
            stats["disambiguated_effective_rank"].append(float(dis["effective_rank"]))
            stats["disambiguated_num_tokens"].append(float(dis["num_tokens"]))

            stats["delta_raw_entropy_bits"].append(float(dis["raw_entropy_bits"] - amb["raw_entropy_bits"]))
            stats["delta_normalized_entropy"].append(float(dis["normalized_entropy"] - amb["normalized_entropy"]))
            stats["delta_effective_rank"].append(float(dis["effective_rank"] - amb["effective_rank"]))

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output: Dict[int, Dict[str, object]] = {}
    for layer_idx, stats in sorted(layer_stats.items()):
        output[layer_idx] = {
            "num_pairs": len(stats["delta_normalized_entropy"]),
            "ambiguous_raw_entropy_bits_mean": float(np.mean(stats["ambiguous_raw_entropy_bits"])),
            "ambiguous_raw_entropy_bits_std": float(np.std(stats["ambiguous_raw_entropy_bits"])),
            "ambiguous_normalized_entropy_mean": float(np.mean(stats["ambiguous_normalized_entropy"])),
            "ambiguous_normalized_entropy_std": float(np.std(stats["ambiguous_normalized_entropy"])),
            "ambiguous_effective_rank_mean": float(np.mean(stats["ambiguous_effective_rank"])),
            "ambiguous_num_tokens_mean": float(np.mean(stats["ambiguous_num_tokens"])),
            "disambiguated_raw_entropy_bits_mean": float(np.mean(stats["disambiguated_raw_entropy_bits"])),
            "disambiguated_raw_entropy_bits_std": float(np.std(stats["disambiguated_raw_entropy_bits"])),
            "disambiguated_normalized_entropy_mean": float(np.mean(stats["disambiguated_normalized_entropy"])),
            "disambiguated_normalized_entropy_std": float(np.std(stats["disambiguated_normalized_entropy"])),
            "disambiguated_effective_rank_mean": float(np.mean(stats["disambiguated_effective_rank"])),
            "disambiguated_num_tokens_mean": float(np.mean(stats["disambiguated_num_tokens"])),
            "delta_raw_entropy_bits_mean": float(np.mean(stats["delta_raw_entropy_bits"])),
            "delta_raw_entropy_bits_std": float(np.std(stats["delta_raw_entropy_bits"])),
            "delta_normalized_entropy_mean": float(np.mean(stats["delta_normalized_entropy"])),
            "delta_normalized_entropy_std": float(np.std(stats["delta_normalized_entropy"])),
            "delta_effective_rank_mean": float(np.mean(stats["delta_effective_rank"])),
            "delta_effective_rank_std": float(np.std(stats["delta_effective_rank"])),
        }

    return output


def summarize_vne_results(model_name: str, results: Dict[int, Dict[str, object]]) -> None:
    if not results:
        print(f"[summary] {model_name} | no von Neumann results")
        return

    ordered_layers = sorted(results.keys())
    middle_layer = ordered_layers[len(ordered_layers) // 2]
    max_delta_layer = max(ordered_layers, key=lambda layer: abs(results[layer]["delta_normalized_entropy_mean"]))

    print(f"[summary-vne] {model_name}")
    for label, layer_idx in [
        ("middle layer", middle_layer),
        ("final layer", ordered_layers[-1]),
        ("largest |delta normalized entropy|", max_delta_layer),
    ]:
        res = results[layer_idx]
        print(
            f"  {label}: {layer_idx} | "
            f"H_amb={res['ambiguous_normalized_entropy_mean']:.4f}, "
            f"H_dis={res['disambiguated_normalized_entropy_mean']:.4f}, "
            f"delta={res['delta_normalized_entropy_mean']:+.4f}"
        )


def evaluate_probe_dataset(
    mode_key: str,
    records: Sequence[dict],
    dataset_metadata: Dict[str, object],
    llama_model: str,
    llada_model: str,
    args: Any,
) -> Dict[str, Dict[str, object]] | None:
    texts, labels, groups, source_types, dataset_metadata = probe_records_to_dataset(records, dataset_metadata)
    print("=" * 72)
    print(f"[info] Preparing probing dataset mode: {mode_key}")
    print(
        f"[info] Extracted {len(texts)} binary NLI pairs across "
        f"{len(set(groups))} unique source instances."
    )
    print(f"[info] Label distribution: {dict(Counter(labels))}")
    print(f"[info] Source ambiguity distribution (pairs): {dict(Counter(source_types))}")

    if len(texts) == 0:
        print(f"[warn] No valid NLI pairs found for dataset mode '{mode_key}'. Skipping.")
        return None

    llama_embeddings = extract_hidden_states(
        llama_model,
        texts,
        is_llada=False,
        batch_size=args.batch_size,
        use_4bit=args.use_4bit,
        include_embedding_layer=args.include_embedding_layer,
    )
    llada_embeddings = extract_hidden_states(
        llada_model,
        texts,
        is_llada=True,
        batch_size=args.batch_size,
        use_4bit=args.use_4bit,
        include_embedding_layer=args.include_embedding_layer,
    )

    llama_results = run_layerwise_probe(
        llama_embeddings,
        labels,
        groups,
        seed=args.seed,
    )
    llada_results = run_layerwise_probe(
        llada_embeddings,
        labels,
        groups,
        seed=args.seed,
    )

    print("-" * 72)
    print(f"=== PROBING RESULTS FOR DATASET MODE: {mode_key} ===")
    summarize_results("LLaMA-3.1-8B (AR)", llama_results)
    summarize_results("LLaDA-8B (Diffusion)", llada_results)
    print("-" * 72)

    return {
        "dataset_metadata": dataset_metadata,
        "results": {
            "llama": {str(k): v for k, v in llama_results.items()},
            "llada": {str(k): v for k, v in llada_results.items()},
        },
    }


def run(args) -> int:
    args.vne_control_conditions = [
        {
            "distractor_reading": "distractor_rewrite",
            "random_matched_reading": "random_matched_rewrite",
        }.get(condition, condition)
        for condition in args.vne_control_conditions
    ]
    llama_model = args.llama_model_id
    llada_model = args.llada_model_id

    print("=== Starting Task 4: Layerwise Internal Representation Probing + VNE ===")
    print(f"[info] Global seed: {args.seed}")
    print(f"[info] Batch size: {args.batch_size}")
    print(f"[info] 4-bit quantization: {args.use_4bit}")
    print(f"[info] Include embedding layer: {args.include_embedding_layer}")
    print(f"[info] Probe dataset modes: {args.dataset_modes}")
    print(f"[info] Probe control modes: {args.probe_control_modes}")
    print(f"[info] Compute von Neumann entropy: {not args.skip_vne}")
    if not args.skip_vne:
        print(f"[info] VNE input mode: {args.vne_input_mode}")
        print(f"[info] VNE control conditions: {args.vne_control_conditions}")
        print(f"[info] VNE token centering: {args.vne_center_tokens}")

    set_all_seeds(args.seed)
    examples = load_examples(args.data_path)

    output = {
        "config": {
            "llama_model": llama_model,
            "llada_model": llada_model,
            "data_path": str(args.data_path),
            "max_examples": args.max_examples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "use_4bit": args.use_4bit,
            "include_embedding_layer": args.include_embedding_layer,
            "dataset_modes": args.dataset_modes,
            "probe_control_modes": args.probe_control_modes,
            "vne_input_mode": args.vne_input_mode,
            "vne_control_conditions": args.vne_control_conditions,
            "vne_center_tokens": args.vne_center_tokens,
            "compute_vne": not args.skip_vne,
        },
        "datasets": {
            "von_neumann_entropy_controls": {},
        },
        "results": {},
        "von_neumann_entropy": {},
        "von_neumann_entropy_controls": {},
    }

    reference_probe_records: Dict[str, List[dict]] = {}
    for mode in args.dataset_modes:
        records, metadata = build_probe_records_from_examples(
            examples,
            mode=mode,
            max_examples=args.max_examples,
        )
        reference_probe_records[mode] = records
        bundle = evaluate_probe_dataset(mode, records, metadata, llama_model, llada_model, args)
        if bundle is None:
            continue
        output["datasets"][mode] = bundle["dataset_metadata"]
        output["results"][mode] = bundle["results"]

    for control_mode in args.probe_control_modes:
        if control_mode != "unambiguous_length_matched":
            continue
        for reference_mode, reference_records in reference_probe_records.items():
            control_key = f"{control_mode}__matched_to__{reference_mode}"
            records, metadata = build_unambiguous_length_matched_probe_control(
                examples,
                reference_records=reference_records,
            )
            metadata["matched_to_mode"] = reference_mode
            bundle = evaluate_probe_dataset(control_key, records, metadata, llama_model, llada_model, args)
            if bundle is None:
                continue
            output["datasets"][control_key] = bundle["dataset_metadata"]
            output["results"][control_key] = bundle["results"]

    if not args.skip_vne:
        print("=" * 72)
        print("[info] Preparing ambiguous vs. disambiguated VNE dataset...")
        vne_pairs, vne_metadata = build_gold_vne_pairs(
            examples,
            input_mode=args.vne_input_mode,
            max_examples=args.max_examples,
        )
        print(
            f"[info] Extracted {len(vne_pairs)} VNE pairs across "
            f"{vne_metadata.get('num_unique_instances', 0)} unique source instances."
        )
        print(f"[info] VNE side distribution: {vne_metadata.get('side_distribution', {})}")
        print(f"[info] VNE label distribution: {vne_metadata.get('label_distribution', {})}")

        if vne_pairs:
            llama_vne = extract_vne_comparison(
                llama_model,
                vne_pairs,
                is_llada=False,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit,
                include_embedding_layer=args.include_embedding_layer,
                center_tokens=args.vne_center_tokens,
            )
            llada_vne = extract_vne_comparison(
                llada_model,
                vne_pairs,
                is_llada=True,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit,
                include_embedding_layer=args.include_embedding_layer,
                center_tokens=args.vne_center_tokens,
            )

            print("-" * 72)
            print("=== VON NEUMANN ENTROPY RESULTS (AMBIGUOUS VS. DISAMBIGUATED) ===")
            summarize_vne_results("LLaMA-3.1-8B (AR)", llama_vne)
            summarize_vne_results("LLaDA-8B (Diffusion)", llada_vne)
            print("-" * 72)

            output["datasets"]["von_neumann_entropy"] = vne_metadata
            output["von_neumann_entropy"] = {
                "llama": {str(k): v for k, v in llama_vne.items()},
                "llada": {str(k): v for k, v in llada_vne.items()},
            }
        else:
            print("[warn] No valid ambiguous/disambiguated pairs found for VNE analysis.")

        for condition in args.vne_control_conditions:
            print("=" * 72)
            print(f"[info] Preparing VNE control condition: {condition}")
            control_pairs, control_metadata = build_vne_control_pairs(
                examples,
                input_mode=args.vne_input_mode,
                condition=condition,
                max_examples=args.max_examples,
            )
            print(
                f"[info] Extracted {len(control_pairs)} VNE control pairs across "
                f"{control_metadata.get('num_unique_instances', 0)} unique source instances."
            )
            print(f"[info] Control side distribution: {control_metadata.get('side_distribution', {})}")
            print(f"[info] Control label distribution: {control_metadata.get('label_distribution', {})}")

            if not control_pairs:
                print(f"[warn] No valid pairs found for VNE control condition '{condition}'.")
                continue

            llama_control_vne = extract_vne_comparison(
                llama_model,
                control_pairs,
                is_llada=False,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit,
                include_embedding_layer=args.include_embedding_layer,
                center_tokens=args.vne_center_tokens,
            )
            llada_control_vne = extract_vne_comparison(
                llada_model,
                control_pairs,
                is_llada=True,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit,
                include_embedding_layer=args.include_embedding_layer,
                center_tokens=args.vne_center_tokens,
            )

            print("-" * 72)
            print(f"=== VON NEUMANN ENTROPY RESULTS ({condition}) ===")
            summarize_vne_results("LLaMA-3.1-8B (AR)", llama_control_vne)
            summarize_vne_results("LLaDA-8B (Diffusion)", llada_control_vne)
            print("-" * 72)

            output["datasets"]["von_neumann_entropy_controls"][condition] = control_metadata
            output["von_neumann_entropy_controls"][condition] = {
                "llama": {str(k): v for k, v in llama_control_vne.items()},
                "llada": {str(k): v for k, v in llada_control_vne.items()},
            }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("=" * 72)
    print(f"[info] Saved combined Task-4 results to: {args.output_path}")
    return 0
