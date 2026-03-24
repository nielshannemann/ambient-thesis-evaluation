#!/usr/bin/env python3
# src/ambient/evaluation/task2_semantic_diversity.py
"""
=============================================================================
TASK 2: GENERATIVE QUALITY & DIVERSITY EVALUATION
=============================================================================
This script evaluates the surface-level text quality and semantic diversity 
of the generated continuations. It is highly critical for Ablation Studies 
to ensure that models (especially discrete diffusion architectures) do not 
suffer from mode collapse, repetition, or ungrammatical generation.

Metrics Computed:
1. Oracle Perplexity (PPL): Uses a frozen AR model (e.g., LLaMA-8B) to assess 
   the fluency and grammatical correctness of the generations.
2. Mean Cosine Distance (MCD): Uses SBERT to evaluate the intra-prompt 
   semantic diversity (avoidance of mode collapse).
3. Lexical Overlap: Calculates the Jaccard-like intersection of words between 
   the prompt and the continuation to penalize repetitive copying.

[Thesis Ref: Section X.X - Evaluating Generation Quality and Diversity]
=============================================================================
"""

import os
import re
import math
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION DEFAULTS
# ==========================================
DEFAULT_PPL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_EMBED_MODEL_ID = "all-MiniLM-L6-v2"
DEFAULT_SEED = 42
CACHE_DIR = "./models"

# Methodological Fix: Prevent grammatical overlap from inflating repetition penalties
STOP_WORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "in", "on", "at", "by", "for", "with", "of", "it", "that", "this", "as"}


def set_global_determinism(seed: int):
    """Guarantees exact reproducibility across all random state engines."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def calculate_perplexity(text: str, model, tokenizer) -> float:
    """
    Calculates the exact perplexity of a given text sequence using a causal LM.
    Perplexity = exp(CrossEntropyLoss)
    """
    if not text.strip():
        return None
        
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    # Exclude extremely short or empty generations
    if input_ids.shape[1] < 2:
        return None
        
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    ppl = torch.exp(loss).item()
    return ppl


def calculate_word_overlap(prompt: str, continuation: str) -> float:
    """
    Calculates the proportion of words in the continuation that already 
    appeared in the prompt (Lexical Repetition Penalty), excluding common stop words.
    """
    if not continuation.strip() or not prompt.strip():
        return 0.0
    
    prompt_words = set(re.findall(r'\w+', prompt.lower())) - STOP_WORDS
    cont_words = set(re.findall(r'\w+', continuation.lower())) - STOP_WORDS
    
    if not cont_words:
        return 0.0
        
    overlap = len(prompt_words.intersection(cont_words)) / len(cont_words)
    return float(overlap)


def main():
    parser = argparse.ArgumentParser(description="Task 2: Text Quality & Diversity Metrics")
    parser.add_argument("--model-dirs", nargs="+", type=Path, required=True, 
                        help="List of paths to the 'example_dirs' of the runs to evaluate.")
    parser.add_argument("--ppl-model", type=str, default=DEFAULT_PPL_MODEL_ID, 
                        help="HuggingFace ID for the Oracle Perplexity model.")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_ID, 
                        help="HuggingFace ID for the SBERT diversity model.")
    parser.add_argument("--use-4bit", action="store_true", help="Load PPL model in 4-bit (NF4) to save VRAM.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global deterministic random seed.")
    args = parser.parse_args()

    print(f"=== Starting Task 2: Quality & Diversity Evaluation ===")
    
    # Enforce reproducibility
    set_global_determinism(args.seed)
    
    # 1. LOAD MODELS
    print(f"[info] Loading Embedding Model ({args.embed_model}) for Diversity...")
    embedder = SentenceTransformer(args.embed_model, cache_folder=CACHE_DIR)
    
    print(f"[info] Loading Oracle PPL Model ({args.ppl_model}) for Fluency...")
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "cache_dir": CACHE_DIR}
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model, cache_dir=CACHE_DIR)
    if getattr(ppl_tokenizer, "pad_token_id", None) is None:
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
        
    ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model, **load_kwargs)
    ppl_model.eval()

    all_results = {}
    missing_prompt_warned = False # Flag to avoid spamming the console

    print("\n[info] Commencing Evaluation Loop...")
    for model_dir in tqdm(args.model_dirs, desc="Evaluating Configurations", position=0):
        if not model_dir.exists() or not model_dir.is_dir():
            print(f"\n[warn] Directory not found or invalid: {model_dir}. Skipping.")
            continue
            
        if model_dir.name == "example_dirs":
            model_name = model_dir.parent.name
            model_root_dir = model_dir.parent
        else:
            model_name = model_dir.name
            model_root_dir = model_dir
            
        metrics = {
            "diversity_scores": [],
            "perplexity_scores": [],
            "overlap_scores": []
        }
        
        instance_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        if not instance_dirs:
            print(f"\n[warn] No subdirectories found inside '{model_dir}'. Are you sure this is the 'example_dirs' folder?")
            continue
            
        valid_files_found = 0
        
        for instance_dir in tqdm(instance_dirs, desc=f"Processing {model_name}", position=1, leave=False):
            
            # --- A. Load Prompt Context ---
            ambig_prompt = ""
            prompt_file = instance_dir / "prompts.jsonl"
            if not prompt_file.exists(): 
                prompt_file = instance_dir / "prompts.json"
            
            if prompt_file.exists():
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    meta = {}
                    try:
                        # Try parsing the whole file first (handles pretty-printed JSON)
                        meta = json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback for strict multi-line JSONL formats
                        for line in content.split('\n'):
                            if line.strip():
                                try:
                                    meta = json.loads(line)
                                    break # Got the first object
                                except json.JSONDecodeError:
                                    continue
                    
                    ambig_prompt = (meta.get("ambiguous_sentence") or 
                                    meta.get("prompt") or 
                                    meta.get("premise") or 
                                    "")
            
            if not ambig_prompt and not missing_prompt_warned:
                print(f"\n[warn] Could not find prompt text in {prompt_file}. Lexical overlap will be 0.0%. Check your JSON format.")
                missing_prompt_warned = True

            # --- B. Gather Continuations ---
            target_files = list(instance_dir.glob("y*.jsonl"))
            if not target_files:
                target_files = [f for f in instance_dir.glob("*.jsonl") if "prompts" not in f.name and "d.jsonl" not in f.name]
            
            continuations_for_div = [] 
            
            for cont_file in target_files: 
                valid_files_found += 1
                with open(cont_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            
                            if data.get("flagged_artifact", False):
                                continue
                                
                            text = data.get("continuation_clean", "").strip()
                            if text: 
                                continuations_for_div.append(text)
                        except Exception:
                            pass
            
            # --- C. Compute Metrics PER INSTANCE ---
            if len(continuations_for_div) >= 2:
                embeddings = embedder.encode(continuations_for_div, show_progress_bar=False, convert_to_numpy=True)
                dists = cosine_distances(embeddings)
                upper_triangle_indices = np.triu_indices_from(dists, k=1)
                
                if len(upper_triangle_indices[0]) > 0:
                    mean_pairwise_distance = np.mean(dists[upper_triangle_indices])
                    metrics["diversity_scores"].append(float(mean_pairwise_distance))

            for text in continuations_for_div:
                ppl = calculate_perplexity(text, ppl_model, ppl_tokenizer)
                overlap = calculate_word_overlap(ambig_prompt, text)
                
                if ppl is not None and not math.isnan(ppl) and not math.isinf(ppl): 
                    metrics["perplexity_scores"].append(ppl)
                if overlap is not None: 
                    metrics["overlap_scores"].append(overlap)

        if valid_files_found == 0:
             print(f"\n[warn] Looked in {len(instance_dirs)} folders inside {model_dir}, but found ZERO valid continuation files (y*.jsonl).")

        # --- D. Aggregate and Print Individual Results ---
        individual_json_path = model_root_dir / "task2_semantic_metrics.json"

        model_stats = {
            "diversity_mean_cosine_dist": float(np.mean(metrics["diversity_scores"])) if metrics["diversity_scores"] else None,
            "perplexity_median": float(np.median(metrics["perplexity_scores"])) if metrics["perplexity_scores"] else None,
            "perplexity_mean": float(np.mean(metrics["perplexity_scores"])) if metrics["perplexity_scores"] else None,
            "overlap_mean": float(np.mean(metrics["overlap_scores"])) if metrics["overlap_scores"] else None,
            "num_evaluated_instances": len(instance_dirs),
            "seed_used": args.seed,
            "local_save_path": str(individual_json_path)
        }
        all_results[model_name] = model_stats
        
        # Immediate Printout for Real-Time Feedback
        div_str = f"{model_stats['diversity_mean_cosine_dist']:.4f}" if model_stats['diversity_mean_cosine_dist'] is not None else "N/A"
        ppl_med_str = f"{model_stats['perplexity_median']:.2f}" if model_stats['perplexity_median'] is not None else "N/A"
        ppl_mean_str = f"{model_stats['perplexity_mean']:.2f}" if model_stats['perplexity_mean'] is not None else "N/A"
        ovl_str = f"{model_stats['overlap_mean']*100:.1f}%" if model_stats['overlap_mean'] is not None else "N/A"
        
        print(f"\n[info] Finished processing: {model_name}")
        print(f"  -> MCD (Diversity): {div_str}")
        print(f"  -> PPL (Quality):   {ppl_med_str} (Median) | {ppl_mean_str} (Mean)")
        print(f"  -> Lexical Overlap: {ovl_str}")
        
        # Save individual model stats
        try:
            with open(individual_json_path, "w", encoding="utf-8") as f:
                json.dump({model_name: model_stats}, f, indent=4, ensure_ascii=False)
            print(f"  -> Saved file to:   {individual_json_path}")
        except Exception as e:
            print(f"  -> [error] Could not save individual metrics: {e}")

    # --- PRINT FINAL METRICS AGGREGATION ---
    print("\n" + "="*60)
    print("=== FINAL METRICS SUMMARY ===")
    
    sorted_results = dict(sorted(all_results.items()))
    
    for model, stats in sorted_results.items():
        print(f"\nModel: {model}")
        
        div_str = f"{stats['diversity_mean_cosine_dist']:.4f}" if stats['diversity_mean_cosine_dist'] is not None else "N/A"
        ppl_med_str = f"{stats['perplexity_median']:.2f}" if stats['perplexity_median'] is not None else "N/A"
        ppl_mean_str = f"{stats['perplexity_mean']:.2f}" if stats['perplexity_mean'] is not None else "N/A"
        ovl_str = f"{stats['overlap_mean']*100:.1f}%" if stats['overlap_mean'] is not None else "N/A"
        
        print(f"  -> MCD (Diversity): {div_str}")
        print(f"  -> PPL (Quality):   {ppl_med_str} (Median) | {ppl_mean_str} (Mean)")
        print(f"  -> Lexical Overlap: {ovl_str}")
        if "local_save_path" in stats:
            print(f"  -> Local Summary:   {stats['local_save_path']}")
            
    print("="*60)

if __name__ == "__main__":
    main()