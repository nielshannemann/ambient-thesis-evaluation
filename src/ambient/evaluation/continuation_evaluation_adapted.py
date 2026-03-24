#!/usr/bin/env python3
# src/ambient/evaluation/continuation_evaluation_adapted.py
"""
=============================================================================
AMBIENT CONTINUATION EVALUATION (ADAPTED)
=============================================================================
This script orchestrates the core evaluation loop for the AMBIENT benchmark.
It handles the generation of continuations, sanitization, and empirical NLL 
scoring across both Autoregressive and Diffusion models.

Restored Features from Original Codebase:
- Exponential Backoff for transient generation failures.
- Cryptographic Deterministic Sub-sampling (CRC32) for reproducible min_conts.
- Graceful Zero-Continuation fallbacks to maintain dataset alignment.
- Hardware-agnostic generation via Adapter micro-batching.

[Thesis Reference: Section 3.1.1 - Overview of the Experimental Pipeline]
=============================================================================
"""

import json
import random
import traceback
import signal
import time
import zlib
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# [Thesis Reference: Section 3.1.2 - The Adapter Framework]
from ambient.adapters import get_adapter, BaseAdapter, ARAdapter
# [Thesis Reference: Section 3.2.2 - Output Sanitization and Artifact Filtering]
from ambient.utils import (
    read_jsonl, ensure_dir,
    clean_continuation_text, is_suspicious, _num_tokens,
    make_instance_id
)


def save_example_results(ambiguous_sent: str, continuation_stats: dict, out_dir: Path, disambiguations: dict = None):
    """
    Saves per-example generated continuations and plots the KL divergence distributions.
    """
    ensure_dir(out_dir)
    meta = {"ambiguous_sentence": ambiguous_sent, "disambiguations": disambiguations}
    
    # Safe JSON writing fallback mechanism
    try:
        with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as mf:
            mf.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    except Exception:
        with open(out_dir / "prompts.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots()
    plotted = False
    
    for d_key, stats in continuation_stats.items():
        try:
            stat_df = pd.DataFrame(stats)
            if not stat_df.empty:
                stat_df.to_json(out_dir / f"{d_key}.jsonl", lines=True, orient="records")
                
                # Plot the log-odds (KL Divergence estimators)
                for col in ("avg_log_odds",):
                    if col in stat_df.columns:
                        numeric = stat_df[pd.to_numeric(stat_df[col], errors='coerce').notnull()]
                        if not numeric.empty:
                            sns.histplot(data=numeric, x=col, label=f"{d_key}", kde=True, stat="density", ax=ax)
                            plotted = True
        except Exception as e:
            print(f"[WARN] Could not plot/save stats for {d_key}: {e}")

    if plotted:
        ax.legend()
        short_title = (ambiguous_sent[:100] + "...") if len(ambiguous_sent) > 100 else ambiguous_sent
        ax.set_title(short_title)
        # [Thesis Reference: Equation 4 - Unbiased Log-Odds Estimator]
        ax.set_xlabel("log P(c | d_i) - log P(c | a)")
        plt.tight_layout()
        try:
            plt.savefig(out_dir / "hist.png", dpi=300)
        except Exception:
            pass
    plt.close(fig)

def create_test_instances(test_df):
    test_instances = []
    for i, row in test_df.iterrows():
        for sentence_key in ['premise', 'hypothesis']:
            if row[f'{sentence_key}_ambiguous']:
                test_instances.append({
                    'id': row['id'],
                    'ambiguous_sentence_key': sentence_key,
                    'ambiguous_sentence': row[sentence_key],
                    'disambiguations': list(set([l[sentence_key] for l in row['disambiguations']])),
                    'distractor': row.get(f'distractor_{sentence_key}') 
                })
    return test_instances

def canonicalize_continuation(continuation_text: str, adapter: BaseAdapter) -> tuple[str, int, bool]:
    """
    Cleans the generated text and flags artifacts (e.g., CJK characters, repetitions).
    [Thesis Reference: Section 3.2.2 - Output Sanitization and Artifact Filtering]
    """
    cleaned = clean_continuation_text(continuation_text)
    tokenizer = getattr(adapter, "tokenizer", getattr(adapter, "ar_tokenizer", None))
    n_tokens = _num_tokens(cleaned, tokenizer_to_use=tokenizer)
    flagged = is_suspicious(cleaned)
    return cleaned, int(n_tokens), bool(flagged)


def continuation_evaluation(
    test_df: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    mc_nums: List[int] = [128],
    summary_names: List[str] = ["summary_mc128.jsonl"],
    top_p: float = 1.0,
    top_k: int = 0,
    temperature: float = 1.0,
    num_generations: int = 100,
    batch_size: int = 25,
    seed: Optional[int] = None,
    **gen_kwargs
):
    """
    Main evaluation loop for the AMBIENT dataset supporting multiple MC levels.
    [Thesis Reference: Section 3.1.1 - Overview of the Experimental Pipeline]
    """
    if len(mc_nums) != len(summary_names):
        raise ValueError("mc_nums and summary_names must have identical lengths.")

    test_df = test_df[test_df['premise_ambiguous'] | test_df['hypothesis_ambiguous']]
    print(f'[INFO] Number of ambiguous rows in dataset: {len(test_df)}')

    adapter = get_adapter(model_name)
    if adapter is None:
        raise ValueError(f"Adapter for {model_name} not found. Please register it first.")

    # [Paper Reference: Liu et al., 2023]
    stem = '"'
    ensure_dir(out_dir)

    # --- RESUME LOGIC (Checkpointing) ---
    # We base the resume logic on the first summary file assuming they are generated synchronously
    processed_ids = set()
    first_summary_file = out_dir / summary_names[0]
    if first_summary_file.exists():
        with open(first_summary_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        processed_ids.add(str(obj["id"]))
                except Exception:
                    pass

    all_instances = create_test_instances(test_df)
    remaining_instances = []
    for r in all_instances:
        iid = str(r.get('_instance_id') or r.get('id', ''))
        if iid and iid not in processed_ids:
            remaining_instances.append(r)

    if processed_ids:
        print(f"[INFO] Resume mode: {len(processed_ids)} instances already evaluated. Remaining: {len(remaining_instances)}")
    else:
        print(f"[INFO] Starting new run. Remaining instances: {len(remaining_instances)}")

    def _sigint_handler(signum, frame):
        print("\n[INFO] Interrupt signal received. Finishing current example before exiting...")
        globals()['__CONTINUATION_SHOULD_STOP__'] = True
    signal.signal(signal.SIGINT, _sigint_handler)
    globals()['__CONTINUATION_SHOULD_STOP__'] = False

    # Open all summary files simultaneously for parallel writing
    summary_fos = [open(out_dir / name, 'a', encoding='utf-8') for name in summary_names]
    
    # Initialize result dictionary per MC-level
    results = {mc_num: [] for mc_num in mc_nums}

    MAX_RETRIES = 3
    BACKOFF = 2.0

    for row in tqdm(remaining_instances, desc="Evaluating AMBIENT"):
        ambiguous_sentence = row['ambiguous_sentence']
        disambiguations = [stem + d for d in row['disambiguations']]
        distractor = stem + row['distractor']
        ambiguous_sentence_full = stem + ambiguous_sentence
        
        disambiguations_dict = {f'y{i}': d for i, d in enumerate(disambiguations)}
        disambiguations_dict['d'] = distractor

        try:
            row_id_int = int(row.get('id', 0))
        except Exception:
            row_id_int = int(zlib.crc32(str(row.get('id')).encode()) & 0xffffffff)

        generated_continuations = {}
        
        # --- PHASE 1: GENERATION (With Exponential Backoff) ---
        for idx, (d_key, prompt_raw) in enumerate(disambiguations_dict.items()):
            conts = []
            
            per_example_seed = None
            if seed is not None:
                per_example_seed = int((int(seed) & 0xffffffff) + (row_id_int % 1000000) * 100 + idx)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # The adapter internally handles chunking via `batch_size`
                    conts = adapter.generate(
                        prompt=prompt_raw, 
                        num_return_sequences=num_generations, 
                        batch_size=batch_size,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        seed=per_example_seed,
                        **gen_kwargs
                    )
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        print(f"\n[ERROR] Generation failed permanently for '{d_key}': {e}")
                    else:
                        wait = BACKOFF * attempt
                        print(f"\n[WARN] Transient error for '{d_key}' (Attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                        time.sleep(wait)

            generated_continuations[d_key] = [str(c) for c in conts if c and str(c).strip()]

        min_conts = min([len(c) for c in generated_continuations.values()] + [num_generations])
        instance_id = row.get('_instance_id') or str(row.get('id'))
        
        is_ar_model = isinstance(adapter, ARAdapter)

        # --- GRACEFUL ZERO-CONTINUATION HANDLING ---
        if min_conts == 0:
            for m_idx, mc_num in enumerate(mc_nums):
                ex = {
                    'id': instance_id,
                    'row_id': row.get('id'),
                    'ambiguous_sentence': ambiguous_sentence_full,
                    'generator_model': model_name,
                    'mc_num': mc_num,
                    'options': {},
                    'num_conts': 0,
                    'scoring_summary': {
                        "adapter": adapter.__class__.__name__,
                        "notes": "No valid continuations generated; scoring skipped."
                    }
                }
                if not is_ar_model:
                    ex['mc_num'] = mc_num

                summary_fos[m_idx].write(json.dumps(ex, default=str) + '\n')
                summary_fos[m_idx].flush()
                results[mc_num].append(ex)
            continue

        # --- PHASE 1.5: DETERMINISTIC SUB-SAMPLING ---
        for d_key in generated_continuations:
            key_idx = sum(ord(c) for c in str(d_key)) % 100
            seed_for_sampling = None
            if seed is not None:
                seed_for_sampling = int((int(seed) & 0xffffffff) + (row_id_int % 1000000) * 100 + key_idx)
            
            rng = random.Random(seed_for_sampling) if seed_for_sampling is not None else random
            
            if len(set(generated_continuations[d_key])) >= min_conts:
                generated_continuations[d_key] = rng.sample(generated_continuations[d_key], min_conts)
            else:
                generated_continuations[d_key] = [rng.choice(generated_continuations[d_key]) for _ in range(min_conts)]

        # --- PHASE 2: SCORING (Multi MC-Level) ---
        all_mc_stats = {m_idx: {} for m_idx in range(len(mc_nums))}

        for d_key, conts in generated_continuations.items():
            disambiguation_context = disambiguations_dict[d_key]
            
            canonical_entries = [canonicalize_continuation(c, adapter) for c in conts]
            clean_conts = [entry[0] for entry in canonical_entries]

            # Adapter architecture-agnostically returns a list of score arrays
            all_mc_scores_cond = adapter.score_continuations([disambiguation_context] * min_conts, clean_conts, mc_nums=mc_nums)
            all_mc_scores_ambig = adapter.score_continuations([ambiguous_sentence_full] * min_conts, clean_conts, mc_nums=mc_nums)

            # Iterate over the respective MC-level results
            for m_idx, (scores_cond, scores_ambig) in enumerate(zip(all_mc_scores_cond, all_mc_scores_ambig)):
                all_mc_stats[m_idx][d_key] = []
                
                for (clean_c, n_tokens, flagged), loss_cond, loss_ambig in zip(canonical_entries, scores_cond, scores_ambig):
                    raw_log_odds = (loss_ambig - loss_cond) if (loss_cond is not None and loss_ambig is not None) else None
                    avg_log_odds = (raw_log_odds / n_tokens) if (raw_log_odds is not None and n_tokens > 0) else None

                    all_mc_stats[m_idx][d_key].append({
                        'continuation_clean': clean_c,
                        'flagged_artifact': flagged,
                        'n_tokens': n_tokens,
                        'nll_cond': loss_cond,
                        'nll_ambig': loss_ambig,
                        'log_odds': raw_log_odds,
                        'avg_log_odds': avg_log_odds,
                    })

        # --- PHASE 3: AGGREGATION & PROVENANCE ---
        # Store representative plotting data for the highest defined MC level
        save_example_results(
            ambiguous_sentence_full, 
            all_mc_stats[len(mc_nums) - 1], 
            out_dir / f'example_dirs/{instance_id}', 
            disambiguations_dict
        )

        for m_idx, mc_num in enumerate(mc_nums):
            ex = {
                'id': instance_id,
                'row_id': row.get('id'),
                'ambiguous_sentence': ambiguous_sentence_full,
                'generator_model': model_name,
                'num_conts': min_conts,
                'mc_num': mc_num,
                'scoring_summary': {"adapter": adapter.__class__.__name__},
                'options': {}
            }

            for d_key, stats in all_mc_stats[m_idx].items():
                # Separate all scored generations from the heuristically "clean" ones
                all_scored = [s for s in stats if s['avg_log_odds'] is not None]
                clean_scored = [s for s in all_scored if not s['flagged_artifact']]
                
                # Unfiltered values (Mathematically strict distribution)
                raw_vals_all = [s['log_odds'] for s in all_scored]
                avg_vals_all = [s['avg_log_odds'] for s in all_scored]
                
                # Cleaned values (Filtered distribution)
                raw_vals_clean = [s['log_odds'] for s in clean_scored]
                avg_vals_clean = [s['avg_log_odds'] for s in clean_scored]
                
                # Calculate Failure/Artifact Rate
                artifact_count = len(all_scored) - len(clean_scored)
                artifact_rate = artifact_count / len(stats) if len(stats) > 0 else 0.0

                ex['options'][d_key] = {
                    'sentence': disambiguations_dict[d_key],
                    'total_continuations': len(stats),
                    'valid_continuations_all': len(all_scored),
                    'valid_continuations_clean': len(clean_scored),
                    'artifact_rate': artifact_rate,
                    
                    # Unfiltered Metrics (Scientifically strict MC integration)
                    'empirical_KL_div_all': float(np.mean(raw_vals_all)) if raw_vals_all else None,
                    'empirical_KL_div_normalized_all': float(np.mean(avg_vals_all)) if avg_vals_all else None,
                    
                    # Filtered Metrics (Engineering heuristic)
                    'empirical_KL_div_clean': float(np.mean(raw_vals_clean)) if raw_vals_clean else None,
                    'empirical_KL_div_normalized_clean': float(np.mean(avg_vals_clean)) if avg_vals_clean else None,
                }

            summary_fos[m_idx].write(json.dumps(ex, default=str) + '\n')
            summary_fos[m_idx].flush()
            results[mc_num].append(ex)

        if globals().get('__CONTINUATION_SHOULD_STOP__', False):
            print("\n[INFO] Stopping run as requested; flushed outputs to disk.")
            break

    # Final File I/O Cleanup
    for fo in summary_fos:
        fo.close()
        
    return results