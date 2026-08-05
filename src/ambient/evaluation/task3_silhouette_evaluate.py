#!/usr/bin/env python3
# src/ambient/evaluation/task3_silhouette_evaluate.py
"""
=============================================================================
TASK 3: GENERATIVE SEMANTIC CLUSTERING (PHASE 2 - EVALUATION)
=============================================================================
This script evaluates the unconstrained semantic continuations generated in 
Phase 1. It calculates four core metrics to quantify the latent ambiguity 
retention of the tested architectures:

1. Mean Cosine Distance (MCD): Measures raw intra-cluster dispersion.
2. Silhouette Score: Measures the density and separation of semantic clusters.
3. Target Coverage (Cosine): Embedding-based least-covered reading support.
4. Target Coverage (NLI): Entailment-based least-covered reading support.

Methodological Integration:
Extracts the exact random seed from the generation metadata to guarantee 
100% deterministic k-means clustering and evaluation parity.

[Thesis: Methodology > Study 3 Metrics and Evaluation]
=============================================================================
"""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ.get("AMBIENT_HF_HOME", "hf_cache")

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from transformers import pipeline, set_seed

from ambient.utils import is_suspicious


def set_global_determinism(seed: int):
    """Lock all random number generators and backend heuristics."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def infer_ambiguity_side(result_item: dict) -> str | None:
    """
    Backward-compatible ambiguity-side inference.

    Priority:
      1. Explicit `ambiguity_side` field written by the current generator.
      2. Exact string match against the ambiguous prompt stored in legacy files.
    """
    side = result_item.get("ambiguity_side")
    if side in {"premise", "hypothesis"}:
        return side

    ambiguous_sentence = (result_item.get("ambiguous_sentence") or "").strip()
    gold_data = result_item.get("gold_disambiguations", [])
    if not ambiguous_sentence or not gold_data:
        return None

    first = gold_data[0]
    premise_text = (first.get("premise") or "").strip()
    hypothesis_text = (first.get("hypothesis") or "").strip()

    if ambiguous_sentence == premise_text:
        return "premise"
    if ambiguous_sentence == hypothesis_text:
        return "hypothesis"

    return None


def extract_gold_texts(result_item: dict) -> tuple[list[str], str | None]:
    """Extract the gold disambiguation texts on the correct ambiguity side."""
    side = infer_ambiguity_side(result_item)
    gold_data = result_item.get("gold_disambiguations", [])

    if side is None:
        return [], None

    gold_texts = []
    for disambig in gold_data:
        text = (disambig.get(side) or "").strip()
        if text:
            gold_texts.append(text)

    return gold_texts, side


def parse_nli_thresholds(raw: str) -> list[str | float]:
    """Parse `argmax,0.5,0.7` style threshold specifications."""
    thresholds: list[str | float] = []
    for part in str(raw or "argmax").split(","):
        value = part.strip()
        if not value:
            continue
        if value.lower() == "argmax":
            thresholds.append("argmax")
        else:
            thresholds.append(float(value))
    return thresholds or ["argmax"]


def entailment_score_from_scores(scores: Any) -> float:
    """Extract the entailment probability from a transformers pipeline result."""
    if isinstance(scores, dict):
        label = str(scores.get("label", "")).upper()
        if label == "ENTAILMENT" or label.endswith("_2"):
            return float(scores.get("score", 0.0))
        return 0.0

    best = 0.0
    if isinstance(scores, list):
        for row in scores:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).upper()
            if label == "ENTAILMENT" or label.endswith("_2"):
                best = max(best, float(row.get("score", 0.0)))
    return best


def is_argmax_entailment(result: Any) -> bool:
    """Return whether the top NLI label is entailment for both old and new pipeline formats."""
    if isinstance(result, dict):
        label = str(result.get("label", "")).upper()
        return label == "ENTAILMENT" or label.endswith("_2")
    if isinstance(result, list) and result:
        top = max(
            [row for row in result if isinstance(row, dict)],
            key=lambda row: float(row.get("score", 0.0)),
            default=None,
        )
        if top is None:
            return False
        label = str(top.get("label", "")).upper()
        return label == "ENTAILMENT" or label.endswith("_2")
    return False


def metric_summary(values: list[float], scale: float = 1.0) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=float) * scale
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def log(message: str) -> None:
    print(message, flush=True)


def select_task3_continuations(
    result_item: dict,
    artifact_policy: str,
) -> tuple[list[str], int, int]:
    """Select non-empty continuations while preserving sampled multiplicities."""
    if artifact_policy not in {"keep", "drop"}:
        raise ValueError(f"Unsupported artifact policy: {artifact_policy}")

    nonempty = [
        str(text).strip()
        for text in result_item.get("continuations", [])
        if str(text or "").strip()
    ]
    if artifact_policy == "keep":
        return nonempty, len(nonempty), 0

    retained = [text for text in nonempty if not is_suspicious(text)]
    return retained, len(nonempty), len(nonempty) - len(retained)


def run(args) -> int:
    if not args.results_path.exists():
        print(f"[ERROR] Target file not found: {args.results_path}")
        return 1

    with open(args.results_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    metadata = full_data.get("metadata", {})
    results_list = full_data.get("results", [])
    artifact_policy = str(getattr(args, "artifact_policy", "keep"))
    log(f"[INFO] Loaded {len(results_list)} generated examples from {args.results_path}")
    log(f"[INFO] Artifact policy: {artifact_policy}; exact duplicates are retained.")

    extracted_seed = metadata.get("hyperparameters", {}).get("seed", 42)
    log(f"[INFO] Extracted deterministic seed {extracted_seed} from metadata. Locking environment...")
    set_global_determinism(extracted_seed)

    log(f"[INFO] Loading Semantic Embedding Model: '{args.embed_model}'...")
    embedder = SentenceTransformer(args.embed_model)

    nli_thresholds = parse_nli_thresholds(args.nli_thresholds)
    nli_pipe = None
    if not args.skip_nli:
        log(f"[INFO] Loading Natural Language Inference (NLI) Model: '{args.nli_model}' (This may take a moment)...")
        device = 0 if torch.cuda.is_available() else -1
        nli_pipe = pipeline("text-classification", model=args.nli_model, device=device)
        log(
            f"[INFO] NLI ready. thresholds={','.join(str(t) for t in nli_thresholds)}, "
            f"batch_size={args.nli_batch_size}, full_scores={any(t != 'argmax' for t in nli_thresholds)}"
        )
    needs_full_nli_scores = any(threshold != "argmax" for threshold in nli_thresholds)

    mcd_scores = []
    silhouette_scores = []
    cos_coverage_scores = []
    nli_coverage_scores = {str(threshold): [] for threshold in nli_thresholds}

    valid_examples = 0
    skipped_side_unknown = 0
    side_counter = {"premise": 0, "hypothesis": 0}
    total_nonempty_continuations = 0
    retained_continuations = 0
    filtered_artifacts = 0
    items_with_filtered_artifacts = 0
    items_below_minimum_after_filtering = 0

    log("\n[INFO] Commencing deterministic evaluation pipeline...\n")

    start_time = time.time()
    total_nli_pairs = 0
    progress_every = max(1, int(args.progress_every or 1))

    for input_idx, data in enumerate(results_list, start=1):
        continuations, nonempty_count, filtered_count = select_task3_continuations(
            data, artifact_policy
        )
        total_nonempty_continuations += nonempty_count
        retained_continuations += len(continuations)
        filtered_artifacts += filtered_count
        items_with_filtered_artifacts += int(filtered_count > 0)
        if nonempty_count >= 2 and len(continuations) < 2:
            items_below_minimum_after_filtering += 1
        gold_disambigs, side = extract_gold_texts(data)

        if side is None:
            skipped_side_unknown += 1
            continue

        side_counter[side] += 1

        # Require at least two continuations and two distinct gold meanings for clustering.
        gold_disambigs = list(dict.fromkeys(gold_disambigs))
        if len(continuations) < 2 or len(gold_disambigs) < 2:
            continue

        k = len(gold_disambigs)

        # 1 & 2: Mean Cosine Distance (MCD) & Silhouette Score
        cont_embeddings = embedder.encode(continuations, convert_to_numpy=True, normalize_embeddings=True)
        gold_embeddings = embedder.encode(gold_disambigs, convert_to_numpy=True, normalize_embeddings=True)

        dist_matrix = cosine_distances(cont_embeddings)
        iu1 = np.triu_indices(len(continuations), k=1)
        mcd = np.mean(dist_matrix[iu1]) if len(iu1[0]) > 0 else 0.0
        mcd_scores.append(mcd)

        if len(continuations) > k and mcd > 1e-5:
            kmeans = KMeans(n_clusters=k, random_state=extracted_seed, n_init=10)
            labels = kmeans.fit_predict(cont_embeddings)
            try:
                sil = silhouette_score(cont_embeddings, labels, metric="cosine")
                silhouette_scores.append(sil)
            except ValueError:
                pass

        # 3: Target Coverage (Cosine Similarity Proxy)
        sims = cosine_similarity(cont_embeddings, gold_embeddings)
        closest_gold_idx = np.argmax(sims, axis=1)
        cos_percentages = [np.sum(closest_gold_idx == i) / len(continuations) for i in range(k)]
        cos_coverage_scores.append(min(cos_percentages))

        # 4: NLI Target Coverage (Strict Entailment Filter)
        if nli_pipe is not None:
            nli_pairs = [{"text": cont, "text_pair": gold} for cont in continuations for gold in gold_disambigs]
            total_nli_pairs += len(nli_pairs)
            if valid_examples < 5 or input_idx % progress_every == 0:
                log(
                    f"[NLI] scoring input {input_idx}/{len(results_list)} "
                    f"(valid_so_far={valid_examples}, pairs={len(nli_pairs)}, "
                    f"total_pairs={total_nli_pairs})"
                )
            if needs_full_nli_scores:
                results = nli_pipe(
                    nli_pairs,
                    truncation=True,
                    max_length=512,
                    batch_size=args.nli_batch_size,
                    top_k=None,
                )
            else:
                results = nli_pipe(
                    nli_pairs,
                    truncation=True,
                    max_length=512,
                    batch_size=args.nli_batch_size,
                )

            for threshold in nli_thresholds:
                nli_entail_counts = [0] * k
                for idx, result in enumerate(results):
                    if threshold == "argmax":
                        entails = is_argmax_entailment(result)
                    else:
                        entails = entailment_score_from_scores(result) >= float(threshold)
                    if entails:
                        gold_idx = idx % k
                        nli_entail_counts[gold_idx] += 1

                nli_percentages = [count / len(continuations) for count in nli_entail_counts]
                nli_coverage_scores[str(threshold)].append(min(nli_percentages))

        valid_examples += 1
        if valid_examples <= 5 or valid_examples % progress_every == 0:
            elapsed = time.time() - start_time
            rate = valid_examples / elapsed if elapsed > 0 else 0.0
            log(
                f"[progress] valid={valid_examples}, input={input_idx}/{len(results_list)}, "
                f"elapsed={elapsed/60:.1f}m, rate={rate:.2f} examples/s"
            )

    print("=" * 65)
    print(f"=== EVALUATION RESULTS FOR: {args.results_path.name} ===")
    print(f"Processed Inputs: {valid_examples}")
    print(
        f"Continuation policy: {artifact_policy}; retained "
        f"{retained_continuations}/{total_nonempty_continuations} non-empty outputs; "
        f"filtered={filtered_artifacts}"
    )
    print(f"Ambiguity sides seen: premise={side_counter['premise']}, hypothesis={side_counter['hypothesis']}")
    if skipped_side_unknown:
        print(f"Skipped (side could not be inferred): {skipped_side_unknown}")
    print("-" * 65)
    if mcd_scores:
        print(f"-> Mean Cosine Distance (MCD):         {np.mean(mcd_scores):.4f}")
    if silhouette_scores:
        print(f"-> Average Silhouette Score:           {np.mean(silhouette_scores):.4f}")
    if cos_coverage_scores:
        print(f"-> Minority Target Coverage (Cosine):  {np.mean(cos_coverage_scores) * 100:.1f}%")
    for threshold_key, values in nli_coverage_scores.items():
        if values:
            label = "argmax" if threshold_key == "argmax" else f"threshold>={threshold_key}"
            print(f"-> Minority Target Coverage (NLI, {label}): {np.mean(values) * 100:.1f}%")
    print("=" * 65)

    output = {
        "results_path": str(args.results_path),
        "embed_model": args.embed_model,
        "nli_model": args.nli_model,
        "skip_nli": bool(args.skip_nli),
        "nli_thresholds": [str(threshold) for threshold in nli_thresholds],
        "artifact_policy": artifact_policy,
        "continuation_selection": {
            "total_nonempty": total_nonempty_continuations,
            "retained": retained_continuations,
            "filtered_artifacts": filtered_artifacts,
            "items_with_filtered_artifacts": items_with_filtered_artifacts,
            "items_below_minimum_after_filtering": items_below_minimum_after_filtering,
            "exact_duplicates_retained": True,
        },
        "seed": extracted_seed,
        "processed_inputs": valid_examples,
        "ambiguity_sides_seen": side_counter,
        "skipped_side_unknown": skipped_side_unknown,
        "metrics": {
            "mcd": metric_summary(mcd_scores),
            "silhouette": metric_summary(silhouette_scores),
            "mtc_cos_percent": metric_summary(cos_coverage_scores, scale=100.0),
            "mtc_nli_percent": {
                key: metric_summary(values, scale=100.0)
                for key, values in nli_coverage_scores.items()
            },
        },
    }
    output_path = args.output_path
    if output_path is None:
        safe_embed = args.embed_model.replace("/", "__")
        safe_nli = args.nli_model.replace("/", "__")
        policy_suffix = "" if artifact_policy == "keep" else "__artifacts-drop"
        output_path = args.results_path.with_name(
            f"{args.results_path.stem}__eval__embed-{safe_embed}__nli-{safe_nli}"
            f"{policy_suffix}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"[INFO] Wrote evaluation summary: {output_path}")
    return 0
