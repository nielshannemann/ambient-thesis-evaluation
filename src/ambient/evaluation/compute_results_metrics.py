#!/usr/bin/env python3
# src/ambient/compute_results_metrics.py
"""
=============================================================================
AMBIENT METRICS AGGREGATION
=============================================================================
This script aggregates the outputs of the evaluation loop and computes the 
final benchmark metrics (KL Divergence Ranking Accuracy).

Restored Features from Original Codebase:
- Advanced Deduplication (Instance-level vs. Row-level).
- Robust handling of generation failures (None-type scores).

[Thesis Ref: Section 2.4 - Evaluation Metrics & Aggregation]
=============================================================================
"""

import json
import argparse
from pathlib import Path
import numpy as np

def read_jsonl(path: Path):
    """Safely yields parsed JSON objects from a JSONL file."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out

def dedupe_results(results: list, dedupe_strategy: str = "instance"):
    """
    Safely deduplicates results in case of resume-aborts or overlaps.
    [Thesis Ref: Section 2.4.2 - Instance-Level Deduplication]
    """
    if dedupe_strategy == "row":
        key_fn = lambda r: str(r.get("row_id") or r.get("id"))
    else:
        # Default AMBIENT behaviour: aggregate by instance first
        key_fn = lambda r: str(r.get("instance_id") or r.get("row_id") or r.get("id"))

    seen = {}
    order = []
    for r in results:
        k = key_fn(r)
        if k not in order:
            order.append(k)
        seen[k] = r
    return [seen[k] for k in order]

def compute_metrics(results: list, metric_key: str = "empirical_KL_div"):
    """
    Calculates the Ranking Accuracy based on empirical KL divergence (Log-Odds).
    [Thesis Ref: Equation 7 - Ranking Accuracy Condition]
    """
    total_examples = len(results)
    evaluated_examples = 0
    
    kl_rank_correct_all = 0
    kl_rank_correct_any = 0
    
    mean_kl_valid_list = []
    mean_kl_distractor_list = []
    mean_artifact_rate_list = []

    for ex in results:
        opts = ex.get("options", {})
        
        # Identify the true disambiguations (y0, y1...) and the distractor (d)
        y_keys = [k for k in opts.keys() if k.startswith("y")]
        d_key = "d"
        
        if d_key not in opts or not y_keys:
            continue
            
        d_val = opts[d_key].get(metric_key)
        y_vals = [opts[yk].get(metric_key) for yk in y_keys]

        # Track Artifact/Garbage Generation Rate across all options
        d_art = opts[d_key].get("artifact_rate", 0.0)
        y_arts = [opts[yk].get("artifact_rate", 0.0) for yk in y_keys]
        mean_artifact_rate_list.append(d_art)
        mean_artifact_rate_list.extend(y_arts)
        
        # Robustness: Skip example if the model completely failed to generate or score a branch
        if d_val is None or any(v is None for v in y_vals):
            continue
            
        evaluated_examples += 1
        
        # D_KL(valid) < D_KL(distractor) means the model is LESS surprised by the valid meaning
        wins = [y_val < d_val for y_val in y_vals]
        
        if all(wins):
            kl_rank_correct_all += 1
        if any(wins):
            kl_rank_correct_any += 1
            
        mean_kl_distractor_list.append(d_val)
        mean_kl_valid_list.extend(y_vals)

    metrics = {
        "Total Instances (Deduped)": total_examples,
        "Total Evaluated (No generation failures)": evaluated_examples,
        "Artifact Rate (Garbage Generations)": float(np.mean(mean_artifact_rate_list)) if mean_artifact_rate_list else 0.0,
        "Ranking Accuracy (All valid < distractor)": (kl_rank_correct_all / evaluated_examples) if evaluated_examples > 0 else None,
        "Ranking Accuracy (Any valid < distractor)": (kl_rank_correct_any / evaluated_examples) if evaluated_examples > 0 else None,
        "Mean KL Divergence (Valid Options)": float(np.mean(mean_kl_valid_list)) if mean_kl_valid_list else None,
        "Mean KL Divergence (Distractor)": float(np.mean(mean_kl_distractor_list)) if mean_kl_distractor_list else None,
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for AMBIENT.")
    parser.add_argument("results_path", type=Path, help="Path to the summary.jsonl file.")
    parser.add_argument("--dedupe", choices=["instance", "row"], default="instance", help="Deduplication strategy.")
    args = parser.parse_args()
    
    if not args.results_path.exists():
        print(f"[Error] File not found: {args.results_path}")
        return
        
    print(f"\n{'='*50}\nAMBIENT METRICS AGGREGATION\n{'='*50}")
    print(f"File: {args.results_path.name}")
    print(f"Deduplication: {args.dedupe}\n{'-'*50}")
    
    # 1. Load and Clean
    raw_results = read_jsonl(args.results_path)
    deduped_results = dedupe_results(raw_results, dedupe_strategy=args.dedupe)
    
    # 2. Compute
    final_output = {
        "Unnormalized_Unfiltered (Strict Math)": compute_metrics(deduped_results, "empirical_KL_div_all"),
        "Unnormalized_Cleaned (Math + Heuristic Filter)": compute_metrics(deduped_results, "empirical_KL_div_clean"),
        "Normalized_Unfiltered (Length-Penalized)": compute_metrics(deduped_results, "empirical_KL_div_normalized_all"),
        "Normalized_Cleaned (Original Script Baseline)": compute_metrics(deduped_results, "empirical_KL_div_normalized_clean")
    }
    
    # 3. Output
    for metric_type, metrics in final_output.items():
        print(f"\n>>> {metric_type.upper()}")
        for k, v in metrics.items():
            if isinstance(v, float):
                if "Accuracy" in k:
                    print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
                else:
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
            
    # Save the metrics exactly where the summary file is located
    out_metrics_path = args.results_path.parent / f"metrics_{args.results_path.stem}.json"
    with open(out_metrics_path, "w", encoding="utf-8") as fo:
        json.dump(final_output, fo, indent=2, default=str)
    print(f"\n[info] Wrote final_output to {out_metrics_path}")

if __name__ == "__main__":
    main()