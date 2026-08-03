"""Cross-dataset scope-ambiguity evaluation following Kamath et al. (2024)."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ambient.evaluation.get_log_likelihood import get_log_likelihood, get_pseudo_log_likelihood
from ambient.evaluation.run_ambient_experiments import batched_exact_nll_score, set_seed
from ambient.modeling import (
    is_autoregressive_family,
    load_model_bundle,
    runtime_environment,
)
from ambient.utils import write_json_atomic


REQUIRED_COLUMNS = {
    "idx",
    "sentence",
    "followup",
    "stype",
    "ftype",
    "OP1",
    "OP1_type",
    "OP2",
    "OP2_type",
}


def load_scope_items(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Scope dataset is missing columns: {sorted(missing)}")

    items: list[dict[str, Any]] = []
    for item_id, group in frame.groupby("idx", sort=True):
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for _, row in group.iterrows():
            lookup[(str(row["stype"]), str(row["ftype"]))] = row.to_dict()
        required_pairs = (("S", "F1"), ("S", "F2"), ("Sc", "F1"), ("Sc", "F2"))
        if any(pair not in lookup for pair in required_pairs):
            continue

        sf1 = lookup[("S", "F1")]
        sf2 = lookup[("S", "F2")]
        scf1 = lookup[("Sc", "F1")]
        scf2 = lookup[("Sc", "F2")]
        if str(sf1["sentence"]) != str(sf2["sentence"]):
            raise ValueError(f"Ambiguous sentence mismatch for idx={item_id}")
        if str(scf1["sentence"]) != str(scf2["sentence"]):
            raise ValueError(f"Control sentence mismatch for idx={item_id}")
        if str(sf1["followup"]) != str(scf1["followup"]):
            raise ValueError(f"F1 mismatch between S and Sc for idx={item_id}")
        if str(sf2["followup"]) != str(scf2["followup"]):
            raise ValueError(f"F2 mismatch between S and Sc for idx={item_id}")

        items.append(
            {
                "idx": str(item_id),
                "ambiguous_sentence": str(sf1["sentence"]).strip(),
                "control_sentence": str(scf1["sentence"]).strip(),
                "followup_f1": str(sf1["followup"]).strip(),
                "followup_f2": str(sf2["followup"]).strip(),
                "op1": str(sf1["OP1"]).strip(),
                "op1_type": str(sf1["OP1_type"]).strip(),
                "op2": str(sf1["OP2"]).strip(),
                "op2_type": str(sf1["OP2_type"]).strip(),
            }
        )
        if max_examples is not None and len(items) >= max_examples:
            break
    return items


def score_all_pairs(args, bundle, items: list[dict[str, Any]]) -> tuple[list[float], str]:
    prompts: list[str] = []
    continuations: list[str] = []
    for item in items:
        prompts.extend(
            [
                item["ambiguous_sentence"],
                item["ambiguous_sentence"],
                item["control_sentence"],
                item["control_sentence"],
            ]
        )
        continuations.extend(
            [
                item["followup_f1"],
                item["followup_f2"],
                item["followup_f1"],
                item["followup_f2"],
            ]
        )

    method = args.scoring_method
    is_ar = is_autoregressive_family(args.model_family)
    if method == "auto":
        method = "exact" if is_ar else "mc"
    if is_ar and method != "exact":
        raise ValueError("Autoregressive scope scoring supports --scoring-method exact or auto.")
    if not is_ar and method == "exact":
        raise ValueError("Masked-diffusion models do not expose exact causal sequence likelihoods.")

    if method == "exact":
        losses = batched_exact_nll_score(
            bundle.model,
            bundle.tokenizer,
            prompts,
            continuations,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            progress_label="Scope exact-NLL scoring",
        )
        return losses, "exact_autoregressive_nll"

    spaced = [text if text.startswith(" ") else f" {text}" for text in continuations]
    if method == "pll":
        losses = get_pseudo_log_likelihood(
            bundle.model,
            bundle.tokenizer,
            prompts,
            spaced,
            batch_size=args.batch_size,
            cfg_scale=args.cfg_scale,
            progress_every=args.progress_every,
            progress_label="Scope PLL scoring",
        )
        return losses, "single_token_pseudo_log_likelihood"

    losses = get_log_likelihood(
        bundle.model,
        bundle.tokenizer,
        prompts,
        spaced,
        mc_nums=[args.mc_num],
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        progress_every=args.progress_every,
        progress_label="Scope MC reconstruction scoring",
    )[0]
    return losses, "random_multi_token_mc_reconstruction"


def bootstrap_mean(values: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for index in range(reps):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[index] = float(np.mean(sample))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def compute_human_alphas(path: Path, allowed_ids: set[str]) -> dict[str, float]:
    frame = pd.read_csv(path)
    needed = {"idx", "stype", "ftype", "response"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Human-results file is missing columns: {sorted(missing)}")
    frame["idx"] = frame["idx"].astype(str)
    frame = frame[frame["idx"].isin(allowed_ids)]
    means = frame.groupby(["idx", "stype", "ftype"], sort=True)["response"].mean()

    def normalized_log_rating(value: float, epsilon: float = 0.01) -> float:
        return float(np.log((value - 1.0 + epsilon) / (6.0 + epsilon)))

    human_alphas: dict[str, float] = {}
    for item_id in sorted(allowed_ids):
        keys = [(item_id, "S", "F1"), (item_id, "S", "F2"), (item_id, "Sc", "F1"), (item_id, "Sc", "F2")]
        if any(key not in means.index for key in keys):
            continue
        sf1, sf2, scf1, scf2 = [normalized_log_rating(float(means.loc[key])) for key in keys]
        human_alphas[item_id] = -((sf1 - sf2) - (scf1 - scf2))
    return human_alphas


def build_summary(
    rows: list[dict[str, Any]],
    human_alphas: dict[str, float] | None,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
    s_differences = np.asarray([row["logp_sf1"] - row["logp_sf2"] for row in rows])
    sc_differences = np.asarray([row["logp_scf1"] - row["logp_scf2"] for row in rows])
    paired_test = stats.ttest_rel(s_differences, sc_differences)
    summary: dict[str, Any] = {
        "num_items": len(rows),
        "alpha": bootstrap_mean(alphas, bootstrap_reps, seed),
        "median_alpha": float(np.median(alphas)),
        "std_alpha": float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0,
        "proportion_positive_alpha": float(np.mean(alphas > 0.0)),
        "paired_ttest_s_vs_sc": {
            "statistic": float(paired_test.statistic),
            "pvalue": float(paired_test.pvalue),
        },
    }

    by_operator: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_operator[f"{row['op1_type']}+{row['op2_type']}"].append(float(row["alpha"]))
    summary["by_operator_type"] = {
        key: {
            "n": len(values),
            "mean_alpha": float(np.mean(values)),
            "proportion_positive": float(np.mean(np.asarray(values) > 0.0)),
        }
        for key, values in sorted(by_operator.items())
    }

    if human_alphas:
        paired = [(row["alpha"], human_alphas[row["idx"]]) for row in rows if row["idx"] in human_alphas]
        if len(paired) >= 2:
            model_values, human_values = map(np.asarray, zip(*paired))
            correlation = stats.pearsonr(model_values, human_values)
            summary["human_proxy_comparison"] = {
                "n": len(paired),
                "pearson_r": float(correlation.statistic),
                "pvalue": float(correlation.pvalue),
                "human_mean_alpha": float(np.mean(human_values)),
                "human_proportion_positive": float(np.mean(human_values > 0.0)),
            }
    return summary


def run(args) -> int:
    set_seed(args.seed)
    items = load_scope_items(args.data_path, args.max_examples)
    if not items:
        raise ValueError(f"No complete S/Sc x F1/F2 items found in {args.data_path}")
    print(f"[INFO] Loaded {len(items)} complete scope-ambiguity items.")

    bundle = load_model_bundle(
        args.model_family,
        model_id=args.model_id,
        use_4bit=args.use_4bit,
        verbose=True,
    )
    print("[INFO] Scoring S/F1, S/F2, Sc/F1, and Sc/F2 pairs.")
    losses, scoring_label = score_all_pairs(args, bundle, items)
    if any(value is None for value in losses):
        raise RuntimeError("At least one scope continuation could not be scored.")

    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(tqdm(items, desc="Computing scope alpha")):
        loss_sf1, loss_sf2, loss_scf1, loss_scf2 = map(
            float,
            losses[item_index * 4 : item_index * 4 + 4],
        )
        logp_sf1, logp_sf2 = -loss_sf1, -loss_sf2
        logp_scf1, logp_scf2 = -loss_scf1, -loss_scf2
        alpha = -((logp_sf1 - logp_sf2) - (logp_scf1 - logp_scf2))
        rows.append(
            {
                **item,
                "loss_sf1": loss_sf1,
                "loss_sf2": loss_sf2,
                "loss_scf1": loss_scf1,
                "loss_scf2": loss_scf2,
                "logp_sf1": logp_sf1,
                "logp_sf2": logp_sf2,
                "logp_scf1": logp_scf1,
                "logp_scf2": logp_scf2,
                "alpha": float(alpha),
                "positive_alpha": bool(alpha > 0.0),
            }
        )

    human_alphas = (
        compute_human_alphas(args.human_results, {row["idx"] for row in rows})
        if args.human_results
        else None
    )
    summary = build_summary(rows, human_alphas, args.bootstrap_reps, args.seed)
    output = {
        "metadata": {
            "task": "scope_ambiguity_experiment_2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_path": str(args.data_path),
            "human_results": str(args.human_results) if args.human_results else None,
            "model_name": args.model_name,
            "model_family": args.model_family,
            "model_id": bundle.model_id,
            "architecture": bundle.architecture,
            "scoring_method": scoring_label,
            "mc_num": args.mc_num if scoring_label.startswith("random_multi") else None,
            "batch_size": args.batch_size,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "runtime_environment": runtime_environment(),
            "alpha_definition": "-[(logP(F1|S)-logP(F2|S))-(logP(F1|Sc)-logP(F2|Sc))]",
        },
        "summary": summary,
        "results": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, output)
    csv_path = args.csv_path or args.output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Scope results written to {args.output_path} and {csv_path}")
    return 0


def summarize(args) -> int:
    """Combine per-model outputs without treating proxy magnitudes as calibrated."""
    rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for specification in args.model_result:
        if "=" not in specification:
            raise ValueError(f"Expected LABEL=PATH for --model-result, got {specification!r}")
        label, path_text = specification.split("=", 1)
        label = label.strip()
        if not label or label in seen_labels:
            raise ValueError(f"Scope result labels must be non-empty and unique: {label!r}")
        seen_labels.add(label)
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata") or {}
        summary = payload.get("summary") or {}
        alpha = summary.get("alpha") or {}
        human = summary.get("human_proxy_comparison") or {}
        paired = summary.get("paired_ttest_s_vs_sc") or {}
        rows.append(
            {
                "label": label,
                "model_id": metadata.get("model_id"),
                "model_family": metadata.get("model_family"),
                "scoring_method": metadata.get("scoring_method"),
                "num_items": summary.get("num_items"),
                "mean_alpha": alpha.get("mean"),
                "mean_alpha_ci_low": alpha.get("ci_low"),
                "mean_alpha_ci_high": alpha.get("ci_high"),
                "proportion_positive_alpha": summary.get("proportion_positive_alpha"),
                "paired_ttest_pvalue": paired.get("pvalue"),
                "human_pearson_r": human.get("pearson_r"),
                "human_pearson_pvalue": human.get("pvalue"),
                "source_path": str(path),
            }
        )

    output = {
        "comparison_scope": (
            "Within-model alpha direction, positive-alpha rate, and human correlation. "
            "Raw alpha magnitudes are not calibrated across exact AR and diffusion proxy scorers."
        ),
        "models": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, output)
    csv_path = args.csv_path or args.output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2))
    print(f"[INFO] Scope comparison written to {args.output_path} and {csv_path}")
    return 0
