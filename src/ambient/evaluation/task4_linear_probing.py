#!/usr/bin/env python3
# src/ambient/evaluation/task4_linear_probing.py
"""
Task 4: Internal Representation Probing (Layerwise Linear Probing + Probe Entropy
+ von Neumann Entropy)

This script keeps the layerwise linear probing setup from the extended Task-4
implementation and additionally computes a representation-level von Neumann
entropy analysis for ambiguous versus disambiguated inputs.

What is preserved from the earlier probing script:
- grouped 5-fold layerwise linear probing with LogisticRegression
- per-layer mean/std accuracy
- per-layer probe entropy derived from held-out classifier probabilities
- support for multiple probe dataset constructions:
    * side_reconstructed
    * fully_disambiguated

What changes relative to the earlier entropy extension:
- the approximate histogram-based global weight entropy is removed
- instead, the script adds von Neumann entropy over token-level hidden-state
  matrices for ambiguous vs. disambiguated inputs

von Neumann entropy construction:
- For one input and one layer, let H in R^{T x D} be the hidden-state matrix for
  the non-padding tokens.
- We form the token Gram matrix G = H H^T.
- We normalize it to rho = G / Tr(G).
- The von Neumann entropy is S(rho) = -Tr(rho log2 rho).

Notes:
- Using H H^T is efficient because T is much smaller than D, and it shares the
  same non-zero spectrum as H^T H after trace-normalization.
- Because disambiguated strings are often longer than ambiguous originals, the
  script reports both raw entropy and a normalized entropy, dividing by log2(T).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from ambient.llada_loader import load_llada_model

warnings.filterwarnings("ignore")

LLADA_MASK_ID = 126336
VALID_BINARY_LABELS = {"entailment", "contradiction"}
DATASET_MODE_CHOICES = ("side_reconstructed", "fully_disambiguated")
VNE_INPUT_MODE_CHOICES = ("sentence_only", "pair_prompt")


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


def ambiguity_type_name(premise_ambiguous: bool, hypothesis_ambiguous: bool) -> str:
    if premise_ambiguous and hypothesis_ambiguous:
        return "both"
    if premise_ambiguous:
        return "premise"
    if hypothesis_ambiguous:
        return "hypothesis"
    return "none"


def load_probe_dataset(
    path: Path,
    mode: str,
    max_examples: int = 600,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, object]]:
    """
    Parse AMBIENT into binary probe datasets.

    Modes:
    - side_reconstructed:
        If premise is ambiguous, use the disambiguated premise with the original
        hypothesis. If hypothesis is ambiguous, use the original premise with the
        disambiguated hypothesis. If both are ambiguous, include both variants.

    - fully_disambiguated:
        Use the fully disambiguated premise-hypothesis pair from each
        disambiguation entry.
    """
    if mode not in DATASET_MODE_CHOICES:
        raise ValueError(f"Unsupported dataset mode: {mode}")

    texts: List[str] = []
    labels: List[str] = []
    groups: List[str] = []
    source_types: List[str] = []

    side_counter = Counter()
    pair_counter = Counter()
    used_instance_ids = set()

    if not path.exists():
        print(f"[error] Dataset not found at {path}")
        return texts, labels, groups, source_types, {}

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            ex = json.loads(line)
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

            instance_pairs = []
            seen_in_instance = set()

            for disambig in disambiguations:
                label = disambig.get("label", "")
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
                                instance_pairs.append(key)
                                seen_in_instance.add(key)

                    if hypothesis_ambiguous:
                        premise = original_premise
                        hypothesis = disambig.get("hypothesis", "")
                        if premise and hypothesis:
                            prompt = build_prompt(premise, hypothesis)
                            key = (prompt, label, instance_id, ambiguity_type)
                            if key not in seen_in_instance:
                                instance_pairs.append(key)
                                seen_in_instance.add(key)

                elif mode == "fully_disambiguated":
                    premise = disambig.get("premise", original_premise)
                    hypothesis = disambig.get("hypothesis", original_hypothesis)
                    if premise and hypothesis:
                        prompt = build_prompt(premise, hypothesis)
                        key = (prompt, label, instance_id, ambiguity_type)
                        if key not in seen_in_instance:
                            instance_pairs.append(key)
                            seen_in_instance.add(key)

            if not instance_pairs:
                continue

            if instance_id not in used_instance_ids:
                if len(used_instance_ids) >= max_examples:
                    break
                used_instance_ids.add(instance_id)

            texts.extend([p[0] for p in instance_pairs])
            labels.extend([p[1] for p in instance_pairs])
            groups.extend([p[2] for p in instance_pairs])
            source_types.extend([p[3] for p in instance_pairs])

            side_counter[ambiguity_type] += 1
            for _, label, _, source_type in instance_pairs:
                pair_counter[(source_type, label)] += 1

    metadata = {
        "mode": mode,
        "num_pairs": len(texts),
        "num_unique_instances": len(set(groups)),
        "label_distribution": dict(Counter(labels)),
        "source_ambiguity_distribution_instances": dict(side_counter),
        "source_ambiguity_distribution_pairs": dict(Counter(source_types)),
        "pair_distribution_by_source_type_and_label": {
            f"{k[0]}::{k[1]}": v for k, v in pair_counter.items()
        },
    }

    print(f"[info] Dataset mode: {mode}")
    print(
        f"[info] Source ambiguity distribution (instances used): "
        f"{metadata['source_ambiguity_distribution_instances']}"
    )
    print(
        "[info] Pair distribution by source-type/label: "
        + json.dumps(metadata["pair_distribution_by_source_type_and_label"], indent=2)
    )
    return texts, labels, groups, source_types, metadata


def load_vne_dataset(
    path: Path,
    input_mode: str,
    max_examples: int = 580,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """
    Build paired ambiguous/disambiguated inputs for the von Neumann analysis.

    sentence_only:
        Compare the ambiguous sentence directly to the corresponding same-side
        disambiguation.

    pair_prompt:
        Keep the full NLI prompt template and replace only the ambiguous side.
    """
    if input_mode not in VNE_INPUT_MODE_CHOICES:
        raise ValueError(f"Unsupported vne input mode: {input_mode}")

    pairs: List[Dict[str, str]] = []
    side_counter = Counter()
    label_counter = Counter()
    used_instance_ids = set()

    if not path.exists():
        print(f"[error] Dataset not found at {path}")
        return pairs, {}

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            ex = json.loads(line)
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

            if instance_id not in used_instance_ids:
                if len(used_instance_ids) >= max_examples:
                    break
                used_instance_ids.add(instance_id)

            seen = set()
            for disambig in disambiguations:
                label = str(disambig.get("label", "unknown"))

                if premise_ambiguous:
                    amb = (original_premise or "").strip()
                    dis = (disambig.get("premise") or "").strip()
                    if amb and dis:
                        if input_mode == "sentence_only":
                            ambiguous_text = amb
                            disambiguated_text = dis
                        else:
                            ambiguous_text = build_prompt(amb, original_hypothesis)
                            disambiguated_text = build_prompt(dis, original_hypothesis)
                        key = (instance_id, "premise", label, ambiguous_text, disambiguated_text)
                        if key not in seen:
                            pairs.append(
                                {
                                    "instance_id": instance_id,
                                    "side": "premise",
                                    "label": label,
                                    "ambiguous_text": ambiguous_text,
                                    "disambiguated_text": disambiguated_text,
                                }
                            )
                            seen.add(key)
                            side_counter["premise"] += 1
                            label_counter[label] += 1

                if hypothesis_ambiguous:
                    amb = (original_hypothesis or "").strip()
                    dis = (disambig.get("hypothesis") or "").strip()
                    if amb and dis:
                        if input_mode == "sentence_only":
                            ambiguous_text = amb
                            disambiguated_text = dis
                        else:
                            ambiguous_text = build_prompt(original_premise, amb)
                            disambiguated_text = build_prompt(original_premise, dis)
                        key = (instance_id, "hypothesis", label, ambiguous_text, disambiguated_text)
                        if key not in seen:
                            pairs.append(
                                {
                                    "instance_id": instance_id,
                                    "side": "hypothesis",
                                    "label": label,
                                    "ambiguous_text": ambiguous_text,
                                    "disambiguated_text": disambiguated_text,
                                }
                            )
                            seen.add(key)
                            side_counter["hypothesis"] += 1
                            label_counter[label] += 1

    metadata = {
        "input_mode": input_mode,
        "num_pairs": len(pairs),
        "num_unique_instances": len({p['instance_id'] for p in pairs}),
        "side_distribution": dict(side_counter),
        "label_distribution": dict(label_counter),
    }
    return pairs, metadata


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
    """
    Run the same grouped CV probe on each layer independently.

    Besides accuracy, this also returns per-layer probe entropy derived from
    held-out logistic-regression class probabilities.
    """
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
        f"(± {results[best_layer]['std_accuracy'] * 100:.2f}%)"
    )
    print(
        f"  middle layer: {middle_layer} | "
        f"{results[middle_layer]['mean_accuracy'] * 100:.2f}% "
        f"(± {results[middle_layer]['std_accuracy'] * 100:.2f}%)"
    )
    print(
        f"  final layer:  {ordered_layers[-1]} | "
        f"{results[ordered_layers[-1]]['mean_accuracy'] * 100:.2f}% "
        f"(± {results[ordered_layers[-1]]['std_accuracy'] * 100:.2f}%)"
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
    """
    Compute von Neumann entropy from the token hidden-state matrix H.
    """
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
    pairs: List[Dict[str, str]],
    is_llada: bool,
    batch_size: int,
    use_4bit: bool,
    include_embedding_layer: bool = False,
    center_tokens: bool = False,
) -> Dict[int, Dict[str, object]]:
    """
    Compute layerwise von Neumann entropy summaries for ambiguous vs.
    disambiguated paired inputs.

    For the VNE analysis, both AR and LLaDA are run on the observed token string
    as-is. Unlike the probing path, LLaDA does not receive an appended [MASK]
    summary token here because we want the token-token representation matrix of
    the actual input.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 4: Layerwise Linear Probing with additional von Neumann entropy analysis"
    )
    parser.add_argument(
        "--llama-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Hugging Face ID for the AR model.",
    )
    parser.add_argument(
        "--llada-model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Base",
        help="Hugging Face ID for the diffusion model.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/test_baked.jsonl"),
        help="Path to the AMBIENT dataset.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=580,
        help="Maximum number of ambiguous source instances to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for hidden-state extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization (NF4) for model loading.",
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help="Also include the embedding layer (index 0). Default: hidden layers only.",
    )
    parser.add_argument(
        "--dataset-modes",
        nargs="+",
        default=list(DATASET_MODE_CHOICES),
        choices=list(DATASET_MODE_CHOICES),
        help=(
            "Which probe dataset constructions to evaluate. "
            "Default: both side_reconstructed and fully_disambiguated."
        ),
    )
    parser.add_argument(
        "--vne-input-mode",
        type=str,
        default="sentence_only",
        choices=list(VNE_INPUT_MODE_CHOICES),
        help=(
            "How the von Neumann entropy comparison should be built. "
            "sentence_only compares just the ambiguous string vs. its same-side rewrite; "
            "pair_prompt keeps the full NLI prompt context."
        ),
    )
    parser.add_argument(
        "--vne-center-tokens",
        action="store_true",
        help="Mean-center token representations before computing the token Gram matrix.",
    )
    parser.add_argument(
        "--skip-vne",
        action="store_true",
        help="Skip the additional von Neumann entropy analysis.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/task4/layerwise_probe_results_with_vne.json"),
        help="Where to save the combined Task-4 results as JSON.",
    )
    args = parser.parse_args()

    print("=== Starting Task 4: Layerwise Internal Representation Probing + VNE ===")
    print(f"[info] Global seed: {args.seed}")
    print(f"[info] Batch size: {args.batch_size}")
    print(f"[info] 4-bit quantization: {args.use_4bit}")
    print(f"[info] Include embedding layer: {args.include_embedding_layer}")
    print(f"[info] Probe dataset modes: {args.dataset_modes}")
    print(f"[info] Compute von Neumann entropy: {not args.skip_vne}")
    if not args.skip_vne:
        print(f"[info] VNE input mode: {args.vne_input_mode}")
        print(f"[info] VNE token centering: {args.vne_center_tokens}")

    set_all_seeds(args.seed)

    output = {
        "config": {
            "llama_model": args.llama_model,
            "llada_model": args.llada_model,
            "data_path": str(args.data_path),
            "max_examples": args.max_examples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "use_4bit": args.use_4bit,
            "include_embedding_layer": args.include_embedding_layer,
            "dataset_modes": args.dataset_modes,
            "vne_input_mode": args.vne_input_mode,
            "vne_center_tokens": args.vne_center_tokens,
            "compute_vne": not args.skip_vne,
        },
        "datasets": {},
        "results": {},
        "von_neumann_entropy": {},
    }

    for mode in args.dataset_modes:
        print("=" * 72)
        print(f"[info] Preparing probing dataset mode: {mode}")
        texts, labels, groups, source_types, dataset_metadata = load_probe_dataset(
            args.data_path,
            mode=mode,
            max_examples=args.max_examples,
        )
        print(
            f"[info] Extracted {len(texts)} binary NLI pairs across "
            f"{len(set(groups))} unique source instances."
        )
        print(f"[info] Label distribution: {dict(Counter(labels))}")
        print(f"[info] Source ambiguity distribution (pairs): {dict(Counter(source_types))}")

        if len(texts) == 0:
            print(f"[warn] No valid NLI pairs found for dataset mode '{mode}'. Skipping.")
            continue

        llama_embeddings = extract_hidden_states(
            args.llama_model,
            texts,
            is_llada=False,
            batch_size=args.batch_size,
            use_4bit=args.use_4bit,
            include_embedding_layer=args.include_embedding_layer,
        )
        llada_embeddings = extract_hidden_states(
            args.llada_model,
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
        print(f"=== PROBING RESULTS FOR DATASET MODE: {mode} ===")
        summarize_results("LLaMA-3.1-8B (AR)", llama_results)
        summarize_results("LLaDA-8B (Diffusion)", llada_results)
        print("-" * 72)

        output["datasets"][mode] = dataset_metadata
        output["results"][mode] = {
            "llama": {str(k): v for k, v in llama_results.items()},
            "llada": {str(k): v for k, v in llada_results.items()},
        }

    if not args.skip_vne:
        print("=" * 72)
        print("[info] Preparing ambiguous vs. disambiguated VNE dataset...")
        vne_pairs, vne_metadata = load_vne_dataset(
            args.data_path,
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
                args.llama_model,
                vne_pairs,
                is_llada=False,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit,
                include_embedding_layer=args.include_embedding_layer,
                center_tokens=args.vne_center_tokens,
            )
            llada_vne = extract_vne_comparison(
                args.llada_model,
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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("=" * 72)
    print(f"[info] Saved combined Task-4 results to: {args.output_path}")


if __name__ == "__main__":
    main()
