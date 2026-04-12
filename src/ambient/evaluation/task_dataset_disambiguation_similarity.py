#!/usr/bin/env python3
"""
Analyze how similar or dissimilar the gold disambiguations in AMBIENT are.

The script computes two complementary views per instance:
1. changed_view: only the ambiguous side(s) rewritten by the disambiguation
2. full_pair_view: the full disambiguated premise-hypothesis pair

Why both?
- changed_view isolates the semantic variation that actually resolves ambiguity
- full_pair_view reflects the full input a model would see

Main metrics:
- pairwise cosine distance over sentence embeddings (semantic distance)
- pairwise lexical Jaccard distance (surface distance)
- optional same-label vs cross-label splits
- optional disambiguation-to-distractor distances when distractor fields exist
- optional disambiguation-to-original-ambiguous distances

Outputs:
- JSON summary
- CSV with per-instance metrics
- CSV with aggregated summaries by source type
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from transformers import set_seed

from ambient.paths import dataset_similarity_default_paths

DEFAULT_DATA_PATH = Path("data/test_baked.jsonl")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUTS = dataset_similarity_default_paths()
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUTS["json"]
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUTS["csv"]
DEFAULT_OUTPUT_AGG_CSV = DEFAULT_OUTPUTS["agg_csv"]
DEFAULT_SEED = 42
CACHE_DIR = "./models"

STOP_WORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "in", "on", "at", "by", "for", "with", "of", "it", "that", "this", "as"}

def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def tokenize_for_jaccard(text: str) -> set[str]:
    import re

    toks = set(re.findall(r"\w+", (text or "").lower()))
    return toks - STOP_WORDS


def jaccard_distance(a: str, b: str) -> float:
    sa = tokenize_for_jaccard(a)
    sb = tokenize_for_jaccard(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - (len(sa & sb) / len(union))


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def std_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.std(values)) if values else None


def median_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def get_source_type(ex: dict) -> str:
    p = bool(ex.get("premise_ambiguous", False))
    h = bool(ex.get("hypothesis_ambiguous", False))
    if p and h:
        return "both"
    if p:
        return "premise"
    if h:
        return "hypothesis"
    return "none"


def build_changed_view(ex: dict, dis: dict) -> str:
    p = bool(ex.get("premise_ambiguous", False))
    h = bool(ex.get("hypothesis_ambiguous", False))
    parts = []
    if p:
        parts.append(f"Premise: {dis.get('premise', '')}")
    if h:
        parts.append(f"Hypothesis: {dis.get('hypothesis', '')}")
    return "\n".join(parts).strip()


def build_original_ambiguous_view(ex: dict, view: str) -> str:
    p = bool(ex.get("premise_ambiguous", False))
    h = bool(ex.get("hypothesis_ambiguous", False))
    premise = ex.get("premise", "")
    hypothesis = ex.get("hypothesis", "")

    if view == "changed_view":
        parts = []
        if p:
            parts.append(f"Premise: {premise}")
        if h:
            parts.append(f"Hypothesis: {hypothesis}")
        return "\n".join(parts).strip()

    if view == "full_pair_view":
        return (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}"
        ).strip()

    raise ValueError(f"Unsupported view: {view}")


def build_full_pair_view(dis: dict) -> str:
    return (
        f"Premise: {dis.get('premise', '')}\n"
        f"Hypothesis: {dis.get('hypothesis', '')}"
    ).strip()


def build_distractor_view(ex: dict, view: str) -> Optional[str]:
    p = bool(ex.get("premise_ambiguous", False))
    h = bool(ex.get("hypothesis_ambiguous", False))

    distractor_premise = ex.get("distractor_premise")
    distractor_hypothesis = ex.get("distractor_hypothesis")
    base_premise = ex.get("premise", "")
    base_hypothesis = ex.get("hypothesis", "")

    if view == "changed_view":
        parts = []
        if p and distractor_premise:
            parts.append(f"Premise: {distractor_premise}")
        if h and distractor_hypothesis:
            parts.append(f"Hypothesis: {distractor_hypothesis}")
        text = "\n".join(parts).strip()
        return text or None

    if view == "full_pair_view":
        premise = distractor_premise if p and distractor_premise else base_premise
        hypothesis = distractor_hypothesis if h and distractor_hypothesis else base_hypothesis
        if not premise and not hypothesis:
            return None
        return f"Premise: {premise}\nHypothesis: {hypothesis}".strip()

    return None


def load_instances(data_path: Path, max_examples: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            disambiguations = ex.get("disambiguations", [])
            if len(disambiguations) < 2:
                continue
            source_type = get_source_type(ex)
            if source_type == "none":
                continue
            rows.append(ex)
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def compute_pairwise_metrics(
    texts: List[str],
    labels: List[str],
    embeddings: np.ndarray,
) -> Dict[str, Optional[float]]:
    n = len(texts)
    if n < 2:
        return {
            "num_interpretations": n,
            "num_pairs": 0,
            "cosine_mean": None,
            "cosine_min": None,
            "cosine_max": None,
            "jaccard_mean": None,
            "jaccard_min": None,
            "jaccard_max": None,
            "same_label_cosine_mean": None,
            "cross_label_cosine_mean": None,
            "same_label_jaccard_mean": None,
            "cross_label_jaccard_mean": None,
        }

    cos_matrix = cosine_distances(embeddings)
    cosine_vals: List[float] = []
    jaccard_vals: List[float] = []
    same_label_cos: List[float] = []
    cross_label_cos: List[float] = []
    same_label_jac: List[float] = []
    cross_label_jac: List[float] = []

    for i, j in itertools.combinations(range(n), 2):
        c = float(cos_matrix[i, j])
        jd = jaccard_distance(texts[i], texts[j])
        cosine_vals.append(c)
        jaccard_vals.append(jd)
        if labels[i] == labels[j]:
            same_label_cos.append(c)
            same_label_jac.append(jd)
        else:
            cross_label_cos.append(c)
            cross_label_jac.append(jd)

    return {
        "num_interpretations": n,
        "num_pairs": len(cosine_vals),
        "cosine_mean": mean_or_none(cosine_vals),
        "cosine_min": float(np.min(cosine_vals)),
        "cosine_max": float(np.max(cosine_vals)),
        "jaccard_mean": mean_or_none(jaccard_vals),
        "jaccard_min": float(np.min(jaccard_vals)),
        "jaccard_max": float(np.max(jaccard_vals)),
        "same_label_cosine_mean": mean_or_none(same_label_cos),
        "cross_label_cosine_mean": mean_or_none(cross_label_cos),
        "same_label_jaccard_mean": mean_or_none(same_label_jac),
        "cross_label_jaccard_mean": mean_or_none(cross_label_jac),
    }


def compute_anchor_metrics(
    dis_embeddings: np.ndarray,
    anchor_embedding: Optional[np.ndarray],
    texts: List[str],
    anchor_text: Optional[str],
    prefix: str,
) -> Dict[str, Optional[float]]:
    if anchor_embedding is None or anchor_text is None:
        return {
            f"{prefix}_available": False,
            f"mean_disambig_to_{prefix}_cosine": None,
            f"mean_disambig_to_{prefix}_jaccard": None,
            f"min_disambig_to_{prefix}_cosine": None,
            f"max_disambig_to_{prefix}_cosine": None,
        }

    dists = cosine_distances(dis_embeddings, anchor_embedding.reshape(1, -1)).reshape(-1)
    jac = [jaccard_distance(t, anchor_text) for t in texts]
    return {
        f"{prefix}_available": True,
        f"mean_disambig_to_{prefix}_cosine": float(np.mean(dists)),
        f"mean_disambig_to_{prefix}_jaccard": float(np.mean(jac)),
        f"min_disambig_to_{prefix}_cosine": float(np.min(dists)),
        f"max_disambig_to_{prefix}_cosine": float(np.max(dists)),
    }


def summarize_numeric(series: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(x) for x in series if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return {
        "mean": mean_or_none(vals),
        "std": std_or_none(vals),
        "median": median_or_none(vals),
        "min": float(np.min(vals)) if vals else None,
        "max": float(np.max(vals)) if vals else None,
        "count": len(vals),
    }


def aggregate_from_dataframe(df: pd.DataFrame, group_name: str) -> List[dict]:
    rows = []
    for value, sub in df.groupby(group_name):
        rows.append({
            "group_by": group_name,
            "group_value": value,
            "num_instances": int(len(sub)),
            "changed_view_cosine_mean": summarize_numeric(sub["changed_view_cosine_mean"].tolist())["mean"],
            "changed_view_jaccard_mean": summarize_numeric(sub["changed_view_jaccard_mean"].tolist())["mean"],
            "changed_view_cross_label_cosine_mean": summarize_numeric(sub["changed_view_cross_label_cosine_mean"].tolist())["mean"],
            "changed_view_same_label_cosine_mean": summarize_numeric(sub["changed_view_same_label_cosine_mean"].tolist())["mean"],
            "changed_view_disambig_to_ambiguous_cosine_mean": summarize_numeric(
                sub["changed_view_mean_disambig_to_ambiguous_cosine"].tolist()
            )["mean"],
            "changed_view_disambig_to_distractor_cosine_mean": summarize_numeric(
                sub["changed_view_mean_disambig_to_distractor_cosine"].tolist()
            )["mean"],
            "full_pair_view_cosine_mean": summarize_numeric(sub["full_pair_view_cosine_mean"].tolist())["mean"],
            "full_pair_view_jaccard_mean": summarize_numeric(sub["full_pair_view_jaccard_mean"].tolist())["mean"],
            "full_pair_view_cross_label_cosine_mean": summarize_numeric(sub["full_pair_view_cross_label_cosine_mean"].tolist())["mean"],
            "full_pair_view_same_label_cosine_mean": summarize_numeric(sub["full_pair_view_same_label_cosine_mean"].tolist())["mean"],
            "full_pair_view_disambig_to_ambiguous_cosine_mean": summarize_numeric(
                sub["full_pair_view_mean_disambig_to_ambiguous_cosine"].tolist()
            )["mean"],
            "full_pair_view_disambig_to_distractor_cosine_mean": summarize_numeric(
                sub["full_pair_view_mean_disambig_to_distractor_cosine"].tolist()
            )["mean"],
        })
    return rows


def run(args) -> int:
    set_global_determinism(args.seed)
    instances = load_instances(args.data_path, args.max_examples)
    if not instances:
        raise SystemExit(f"No usable ambiguous instances with >=2 disambiguations found in {args.data_path}")

    print(f"[info] Loaded {len(instances)} ambiguous instances with >=2 disambiguations")
    print(f"[info] Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model, cache_folder=CACHE_DIR)

    per_instance_rows: List[dict] = []
    source_counter = Counter()
    num_disambigs_counter = Counter()
    global_pair_type_counter = Counter()

    for ex in instances:
        source_type = get_source_type(ex)
        source_counter[source_type] += 1

        disambiguations = ex.get("disambiguations", [])
        labels = [d.get("label", "") for d in disambiguations]
        num_disambigs_counter[len(disambiguations)] += 1

        changed_texts = [build_changed_view(ex, d) for d in disambiguations]
        full_texts = [build_full_pair_view(d) for d in disambiguations]

        changed_emb = embedder.encode(
            changed_texts, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        full_emb = embedder.encode(
            full_texts, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=False
        )

        changed_metrics = compute_pairwise_metrics(changed_texts, labels, changed_emb)
        full_metrics = compute_pairwise_metrics(full_texts, labels, full_emb)

        for i, j in itertools.combinations(range(len(labels)), 2):
            global_pair_type_counter["same_label" if labels[i] == labels[j] else "cross_label"] += 1

        changed_ambiguous_text = build_original_ambiguous_view(ex, "changed_view")
        full_ambiguous_text = build_original_ambiguous_view(ex, "full_pair_view")
        changed_ambiguous_emb = embedder.encode(
            [changed_ambiguous_text], batch_size=1, convert_to_numpy=True, show_progress_bar=False
        )[0] if changed_ambiguous_text else None
        full_ambiguous_emb = embedder.encode(
            [full_ambiguous_text], batch_size=1, convert_to_numpy=True, show_progress_bar=False
        )[0] if full_ambiguous_text else None

        changed_ambiguous_metrics = compute_anchor_metrics(
            changed_emb, changed_ambiguous_emb, changed_texts, changed_ambiguous_text, "ambiguous"
        )
        full_ambiguous_metrics = compute_anchor_metrics(
            full_emb, full_ambiguous_emb, full_texts, full_ambiguous_text, "ambiguous"
        )

        changed_distractor_text = build_distractor_view(ex, "changed_view")
        full_distractor_text = build_distractor_view(ex, "full_pair_view")
        changed_distractor_emb = embedder.encode(
            [changed_distractor_text], batch_size=1, convert_to_numpy=True, show_progress_bar=False
        )[0] if changed_distractor_text else None
        full_distractor_emb = embedder.encode(
            [full_distractor_text], batch_size=1, convert_to_numpy=True, show_progress_bar=False
        )[0] if full_distractor_text else None

        changed_distractor_metrics = compute_anchor_metrics(
            changed_emb, changed_distractor_emb, changed_texts, changed_distractor_text, "distractor"
        )
        full_distractor_metrics = compute_anchor_metrics(
            full_emb, full_distractor_emb, full_texts, full_distractor_text, "distractor"
        )

        row = {
            "id": ex.get("id"),
            "source_type": source_type,
            "num_disambiguations": len(disambiguations),
            "labels": " | ".join(labels),
            "changed_view_num_pairs": changed_metrics["num_pairs"],
            "changed_view_cosine_mean": changed_metrics["cosine_mean"],
            "changed_view_cosine_min": changed_metrics["cosine_min"],
            "changed_view_cosine_max": changed_metrics["cosine_max"],
            "changed_view_jaccard_mean": changed_metrics["jaccard_mean"],
            "changed_view_same_label_cosine_mean": changed_metrics["same_label_cosine_mean"],
            "changed_view_cross_label_cosine_mean": changed_metrics["cross_label_cosine_mean"],
            "changed_view_same_label_jaccard_mean": changed_metrics["same_label_jaccard_mean"],
            "changed_view_cross_label_jaccard_mean": changed_metrics["cross_label_jaccard_mean"],
            "changed_view_mean_disambig_to_ambiguous_cosine": changed_ambiguous_metrics["mean_disambig_to_ambiguous_cosine"],
            "changed_view_mean_disambig_to_ambiguous_jaccard": changed_ambiguous_metrics["mean_disambig_to_ambiguous_jaccard"],
            "changed_view_min_disambig_to_ambiguous_cosine": changed_ambiguous_metrics["min_disambig_to_ambiguous_cosine"],
            "changed_view_max_disambig_to_ambiguous_cosine": changed_ambiguous_metrics["max_disambig_to_ambiguous_cosine"],
            "changed_view_mean_disambig_to_distractor_cosine": changed_distractor_metrics["mean_disambig_to_distractor_cosine"],
            "changed_view_mean_disambig_to_distractor_jaccard": changed_distractor_metrics["mean_disambig_to_distractor_jaccard"],
            "changed_view_min_disambig_to_distractor_cosine": changed_distractor_metrics["min_disambig_to_distractor_cosine"],
            "changed_view_max_disambig_to_distractor_cosine": changed_distractor_metrics["max_disambig_to_distractor_cosine"],
            "full_pair_view_num_pairs": full_metrics["num_pairs"],
            "full_pair_view_cosine_mean": full_metrics["cosine_mean"],
            "full_pair_view_cosine_min": full_metrics["cosine_min"],
            "full_pair_view_cosine_max": full_metrics["cosine_max"],
            "full_pair_view_jaccard_mean": full_metrics["jaccard_mean"],
            "full_pair_view_same_label_cosine_mean": full_metrics["same_label_cosine_mean"],
            "full_pair_view_cross_label_cosine_mean": full_metrics["cross_label_cosine_mean"],
            "full_pair_view_same_label_jaccard_mean": full_metrics["same_label_jaccard_mean"],
            "full_pair_view_cross_label_jaccard_mean": full_metrics["cross_label_jaccard_mean"],
            "full_pair_view_mean_disambig_to_ambiguous_cosine": full_ambiguous_metrics["mean_disambig_to_ambiguous_cosine"],
            "full_pair_view_mean_disambig_to_ambiguous_jaccard": full_ambiguous_metrics["mean_disambig_to_ambiguous_jaccard"],
            "full_pair_view_min_disambig_to_ambiguous_cosine": full_ambiguous_metrics["min_disambig_to_ambiguous_cosine"],
            "full_pair_view_max_disambig_to_ambiguous_cosine": full_ambiguous_metrics["max_disambig_to_ambiguous_cosine"],
            "full_pair_view_mean_disambig_to_distractor_cosine": full_distractor_metrics["mean_disambig_to_distractor_cosine"],
            "full_pair_view_mean_disambig_to_distractor_jaccard": full_distractor_metrics["mean_disambig_to_distractor_jaccard"],
            "full_pair_view_min_disambig_to_distractor_cosine": full_distractor_metrics["min_disambig_to_distractor_cosine"],
            "full_pair_view_max_disambig_to_distractor_cosine": full_distractor_metrics["max_disambig_to_distractor_cosine"],
        }
        per_instance_rows.append(row)

    df = pd.DataFrame(per_instance_rows)
    agg_rows = []
    agg_rows.extend(aggregate_from_dataframe(df, "source_type"))
    agg_rows.extend(aggregate_from_dataframe(df, "num_disambiguations"))
    agg_df = pd.DataFrame(agg_rows)

    overall_summary = {
        "dataset": {
            "data_path": str(args.data_path),
            "num_instances": int(len(df)),
            "source_type_distribution": dict(source_counter),
            "num_disambiguations_distribution": {str(k): int(v) for k, v in num_disambigs_counter.items()},
            "global_pair_type_distribution": dict(global_pair_type_counter),
        },
        "changed_view": {
            "cosine_mean": summarize_numeric(df["changed_view_cosine_mean"].tolist()),
            "jaccard_mean": summarize_numeric(df["changed_view_jaccard_mean"].tolist()),
            "same_label_cosine_mean": summarize_numeric(df["changed_view_same_label_cosine_mean"].tolist()),
            "cross_label_cosine_mean": summarize_numeric(df["changed_view_cross_label_cosine_mean"].tolist()),
            "same_label_jaccard_mean": summarize_numeric(df["changed_view_same_label_jaccard_mean"].tolist()),
            "cross_label_jaccard_mean": summarize_numeric(df["changed_view_cross_label_jaccard_mean"].tolist()),
            "disambig_to_ambiguous_cosine_mean": summarize_numeric(df["changed_view_mean_disambig_to_ambiguous_cosine"].tolist()),
            "disambig_to_ambiguous_jaccard_mean": summarize_numeric(df["changed_view_mean_disambig_to_ambiguous_jaccard"].tolist()),
            "disambig_to_distractor_cosine_mean": summarize_numeric(df["changed_view_mean_disambig_to_distractor_cosine"].tolist()),
            "disambig_to_distractor_jaccard_mean": summarize_numeric(df["changed_view_mean_disambig_to_distractor_jaccard"].tolist()),
        },
        "full_pair_view": {
            "cosine_mean": summarize_numeric(df["full_pair_view_cosine_mean"].tolist()),
            "jaccard_mean": summarize_numeric(df["full_pair_view_jaccard_mean"].tolist()),
            "same_label_cosine_mean": summarize_numeric(df["full_pair_view_same_label_cosine_mean"].tolist()),
            "cross_label_cosine_mean": summarize_numeric(df["full_pair_view_cross_label_cosine_mean"].tolist()),
            "same_label_jaccard_mean": summarize_numeric(df["full_pair_view_same_label_jaccard_mean"].tolist()),
            "cross_label_jaccard_mean": summarize_numeric(df["full_pair_view_cross_label_jaccard_mean"].tolist()),
            "disambig_to_ambiguous_cosine_mean": summarize_numeric(df["full_pair_view_mean_disambig_to_ambiguous_cosine"].tolist()),
            "disambig_to_ambiguous_jaccard_mean": summarize_numeric(df["full_pair_view_mean_disambig_to_ambiguous_jaccard"].tolist()),
            "disambig_to_distractor_cosine_mean": summarize_numeric(df["full_pair_view_mean_disambig_to_distractor_cosine"].tolist()),
            "disambig_to_distractor_jaccard_mean": summarize_numeric(df["full_pair_view_mean_disambig_to_distractor_jaccard"].tolist()),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_agg_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    df.to_csv(args.output_csv, index=False)
    agg_df.to_csv(args.output_agg_csv, index=False)

    print(f"[info] Wrote summary JSON to: {args.output_json}")
    print(f"[info] Wrote per-instance CSV to: {args.output_csv}")
    print(f"[info] Wrote aggregate CSV to: {args.output_agg_csv}")

    def _fmt(x: Optional[float]) -> str:
        return "None" if x is None else f"{x:.4f}"

    print("\n=== DISAMBIGUATION SIMILARITY SUMMARY ===")
    print(f"Instances analyzed: {len(df)}")
    print(f"Source types: {dict(source_counter)}")
    print("Changed view:")
    print(f"  mean cosine distance: {_fmt(overall_summary['changed_view']['cosine_mean']['mean'])}")
    print(f"  same-label mean cosine distance: {_fmt(overall_summary['changed_view']['same_label_cosine_mean']['mean'])}")
    print(f"  cross-label mean cosine distance: {_fmt(overall_summary['changed_view']['cross_label_cosine_mean']['mean'])}")
    print(f"  mean jaccard distance: {_fmt(overall_summary['changed_view']['jaccard_mean']['mean'])}")
    print(f"  mean disambig→ambiguous cosine distance: {_fmt(overall_summary['changed_view']['disambig_to_ambiguous_cosine_mean']['mean'])}")
    print(f"  mean disambig→distractor cosine distance: {_fmt(overall_summary['changed_view']['disambig_to_distractor_cosine_mean']['mean'])}")
    print("Full pair view:")
    print(f"  mean cosine distance: {_fmt(overall_summary['full_pair_view']['cosine_mean']['mean'])}")
    print(f"  same-label mean cosine distance: {_fmt(overall_summary['full_pair_view']['same_label_cosine_mean']['mean'])}")
    print(f"  cross-label mean cosine distance: {_fmt(overall_summary['full_pair_view']['cross_label_cosine_mean']['mean'])}")
    print(f"  mean jaccard distance: {_fmt(overall_summary['full_pair_view']['jaccard_mean']['mean'])}")
    print(f"  mean disambig→ambiguous cosine distance: {_fmt(overall_summary['full_pair_view']['disambig_to_ambiguous_cosine_mean']['mean'])}")
    print(f"  mean disambig→distractor cosine distance: {_fmt(overall_summary['full_pair_view']['disambig_to_distractor_cosine_mean']['mean'])}")


    return 0
