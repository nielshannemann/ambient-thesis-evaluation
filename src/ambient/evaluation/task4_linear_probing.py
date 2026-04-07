#!/usr/bin/env python3
# src/ambient/evaluation/task4_linear_probing.py
"""
Task 4: Internal Representation Probing (Layerwise Linear Probing + Probe Entropy)

This script evaluates whether binary NLI entailment states are linearly
accessible in the hidden representations of:
- an autoregressive base model
- a diffusion base model (LLaDA)

Compared with the earlier layerwise version, this implementation additionally:
- supports multiple probe input regimes
- keeps results both for the current side-reconstructed disambiguations and for
  fully disambiguated premise-hypothesis pairs
- computes per-layer probe entropy from held-out logistic-regression probabilities
- computes an approximate global histogram entropy over the model weights

Important note on entropy:
- "probe entropy" is the Shannon entropy of the held-out binary class
  probabilities predicted by the probe for a given layer. This is interpretable
  as task-level uncertainty of a linear readout, not entropy of the raw hidden
  vector itself.
- "weight entropy" is an approximate histogram-based Shannon entropy over a
  large deterministic sample of model parameter values. It is descriptive and
  depends on the chosen sampling budget and number of histogram bins; it should
  not be interpreted as Bayesian uncertainty.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
WEIGHT_ENTROPY_CACHE: Dict[Tuple[object, ...], Dict[str, float]] = {}


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
        Current Task-4 behavior. If premise is ambiguous, use the disambiguated
        premise with the original hypothesis. If hypothesis is ambiguous, use
        the original premise with the disambiguated hypothesis. If both are
        ambiguous, include both variants.

    - fully_disambiguated:
        Use the fully disambiguated premise-hypothesis pair from each
        disambiguation entry. This keeps the rewritten pair together and is the
        most direct comparison condition against side-reconstructed prompts.
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



def _iter_parameter_samples(model: torch.nn.Module, global_stride: int) -> Iterable[np.ndarray]:
    for param in model.parameters():
        if param.numel() == 0:
            continue
        flat = param.detach().view(-1)
        sampled = flat[::global_stride]
        if sampled.numel() == 0:
            continue
        yield sampled.float().cpu().numpy()



def approximate_model_weight_entropy_bits(
    model: torch.nn.Module,
    max_samples: int = 2_000_000,
    num_bins: int = 256,
) -> Dict[str, float]:
    """
    Approximate a global histogram entropy over model parameter values.

    This uses deterministic strided sampling across all parameters to avoid
    materializing billions of weights in memory.
    """
    total_numel = 0
    for param in model.parameters():
        total_numel += int(param.numel())

    if total_numel == 0:
        return {
            "histogram_entropy_bits": float("nan"),
            "sampled_weights": 0,
            "total_weights": 0,
            "num_bins": num_bins,
            "global_stride": 1,
        }

    global_stride = max(1, total_numel // max_samples)
    sample_chunks = list(_iter_parameter_samples(model, global_stride))
    if not sample_chunks:
        return {
            "histogram_entropy_bits": float("nan"),
            "sampled_weights": 0,
            "total_weights": total_numel,
            "num_bins": num_bins,
            "global_stride": global_stride,
        }

    samples = np.concatenate(sample_chunks, axis=0)
    if samples.size > max_samples:
        samples = samples[:max_samples]

    if samples.size == 0:
        entropy = float("nan")
        min_value = float("nan")
        max_value = float("nan")
    else:
        min_value = float(np.min(samples))
        max_value = float(np.max(samples))
        if np.isclose(min_value, max_value):
            entropy = 0.0
        else:
            counts, _ = np.histogram(samples, bins=num_bins, range=(min_value, max_value))
            probs = counts.astype(np.float64)
            probs /= probs.sum()
            probs = probs[probs > 0]
            entropy = float(-(probs * np.log2(probs)).sum())

    return {
        "histogram_entropy_bits": entropy,
        "sampled_weights": int(samples.size),
        "total_weights": int(total_numel),
        "num_bins": int(num_bins),
        "global_stride": int(global_stride),
        "sample_min": min_value,
        "sample_max": max_value,
    }



def extract_hidden_states(
    model_id: str,
    texts: List[str],
    is_llada: bool,
    batch_size: int,
    use_4bit: bool,
    include_embedding_layer: bool = False,
    compute_weight_entropy: bool = True,
    weight_entropy_max_samples: int = 2_000_000,
    weight_entropy_bins: int = 256,
) -> Tuple[Dict[int, np.ndarray], Dict[str, float] | None]:
    print(f"[info] Initializing feature extraction pipeline for: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, is_llada=is_llada, use_4bit=use_4bit)

    weight_entropy_info = None
    entropy_cache_key = (
        model_id,
        is_llada,
        use_4bit,
        weight_entropy_max_samples,
        weight_entropy_bins,
    )
    if compute_weight_entropy:
        if entropy_cache_key in WEIGHT_ENTROPY_CACHE:
            weight_entropy_info = WEIGHT_ENTROPY_CACHE[entropy_cache_key]
            print(
                f"[info] Reusing cached weight entropy for {model_id}: "
                f"{weight_entropy_info['histogram_entropy_bits']:.4f} bits"
            )
        else:
            print(f"[info] Computing approximate global weight entropy for: {model_id}")
            weight_entropy_info = approximate_model_weight_entropy_bits(
                model,
                max_samples=weight_entropy_max_samples,
                num_bins=weight_entropy_bins,
            )
            WEIGHT_ENTROPY_CACHE[entropy_cache_key] = weight_entropy_info
            print(
                f"[info] Approximate global weight entropy for {model_id}: "
                f"{weight_entropy_info['histogram_entropy_bits']:.4f} bits "
                f"(sampled {weight_entropy_info['sampled_weights']:,} / "
                f"{weight_entropy_info['total_weights']:,} weights)"
            )

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
    return final_embeddings_by_layer, weight_entropy_info



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



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 4: Layerwise Linear Probing of Internal Representations"
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
        help="Also probe the embedding layer (index 0). Default: hidden layers only.",
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
        "--skip-weight-entropy",
        action="store_true",
        help="Skip the approximate global histogram entropy over model weights.",
    )
    parser.add_argument(
        "--weight-entropy-max-samples",
        type=int,
        default=2_000_000,
        help="Maximum number of sampled weights for approximate global weight entropy.",
    )
    parser.add_argument(
        "--weight-entropy-bins",
        type=int,
        default=256,
        help="Number of histogram bins for approximate global weight entropy.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/task4/layerwise_probe_results_with_entropy.json"),
        help="Where to save the layerwise probing results as JSON.",
    )
    args = parser.parse_args()

    print("=== Starting Task 4: Layerwise Internal Representation Probing ===")
    print(f"[info] Global seed: {args.seed}")
    print(f"[info] Batch size: {args.batch_size}")
    print(f"[info] 4-bit quantization: {args.use_4bit}")
    print(f"[info] Include embedding layer: {args.include_embedding_layer}")
    print(f"[info] Dataset modes: {args.dataset_modes}")
    print(f"[info] Compute weight entropy: {not args.skip_weight_entropy}")

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
            "compute_weight_entropy": not args.skip_weight_entropy,
            "weight_entropy_max_samples": args.weight_entropy_max_samples,
            "weight_entropy_bins": args.weight_entropy_bins,
        },
        "datasets": {},
        "weight_entropy": {},
        "results": {},
    }

    for mode in args.dataset_modes:
        print("=" * 72)
        print(f"[info] Preparing dataset mode: {mode}")
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

        llama_embeddings, llama_weight_entropy = extract_hidden_states(
            args.llama_model,
            texts,
            is_llada=False,
            batch_size=args.batch_size,
            use_4bit=args.use_4bit,
            include_embedding_layer=args.include_embedding_layer,
            compute_weight_entropy=not args.skip_weight_entropy,
            weight_entropy_max_samples=args.weight_entropy_max_samples,
            weight_entropy_bins=args.weight_entropy_bins,
        )
        llada_embeddings, llada_weight_entropy = extract_hidden_states(
            args.llada_model,
            texts,
            is_llada=True,
            batch_size=args.batch_size,
            use_4bit=args.use_4bit,
            include_embedding_layer=args.include_embedding_layer,
            compute_weight_entropy=not args.skip_weight_entropy,
            weight_entropy_max_samples=args.weight_entropy_max_samples,
            weight_entropy_bins=args.weight_entropy_bins,
        )

        if llama_weight_entropy is not None:
            output["weight_entropy"]["llama"] = llama_weight_entropy
        if llada_weight_entropy is not None:
            output["weight_entropy"]["llada"] = llada_weight_entropy

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
        print(f"=== RESULTS FOR DATASET MODE: {mode} ===")
        summarize_results("LLaMA-3.1-8B (AR)", llama_results)
        summarize_results("LLaDA-8B (Diffusion)", llada_results)
        print("-" * 72)

        output["datasets"][mode] = dataset_metadata
        output["results"][mode] = {
            "llama": {str(k): v for k, v in llama_results.items()},
            "llada": {str(k): v for k, v in llada_results.items()},
        }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("=" * 72)
    print(f"[info] Saved layerwise results to: {args.output_path}")


if __name__ == "__main__":
    main()
