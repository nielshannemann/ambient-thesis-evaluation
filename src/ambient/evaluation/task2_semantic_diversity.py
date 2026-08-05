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
1. External-LM Perplexity (PPL): Uses a frozen AR model (e.g., LLaMA-8B) to assess 
   the fluency and grammatical correctness of the generations.
2. Mean Cosine Distance (MCD): Uses SBERT to evaluate the intra-prompt 
   semantic diversity (avoidance of mode collapse).
3. Lexical Overlap: Calculates the Jaccard-like intersection of words between 
   the prompt and the continuation to penalize repetitive copying.

[Thesis: Methodology > Study 2: Generation-Quality Controls]
=============================================================================
"""

import os
import re
import math
import json
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sklearn.metrics.pairwise import cosine_distances

from ambient.paths import task2_metrics_path

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


def calculate_perplexities_batch(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
    max_length: Optional[int] = None,
) -> List[Optional[float]]:
    """Compute per-text causal-LM perplexities in batches, excluding padding tokens."""
    if not texts:
        return []

    results: List[Optional[float]] = []
    device = model.device
    pad_id = tokenizer.pad_token_id

    for start in tqdm(range(0, len(texts), batch_size), desc="PPL batches", leave=False):
        batch_texts = texts[start : start + batch_size]
        clean_texts = [text if text and text.strip() else "" for text in batch_texts]
        short_mask = [not bool(text.strip()) for text in clean_texts]

        enc_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": max_length is not None,
        }
        if max_length is not None:
            enc_kwargs["max_length"] = max_length

        encodings = tokenizer(clean_texts, **enc_kwargs)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        if input_ids.shape[1] < 2:
            results.extend([None] * len(batch_texts))
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        token_losses = token_losses * shift_mask
        token_counts = shift_mask.sum(dim=1)
        seq_losses = token_losses.sum(dim=1) / token_counts.clamp(min=1)

        for idx, loss in enumerate(seq_losses):
            if short_mask[idx] or token_counts[idx].item() < 1:
                results.append(None)
                continue
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                results.append(None)
                continue
            try:
                results.append(float(math.exp(loss_value)))
            except OverflowError:
                results.append(float("inf"))

    return results


def sanitize_suffix(suffix: Optional[str]) -> Optional[str]:
    if not suffix:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix.strip())
    return safe.strip("._-") or None


def task2_metrics_output_path(model_root_dir: Path, suffix: Optional[str]) -> Path:
    if not suffix:
        return task2_metrics_path(model_root_dir)
    return model_root_dir / f"task2_semantic_metrics__{suffix}.json"


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


def resolve_task2_model_input(model_dir: Path) -> tuple[str, Path, Path]:
    """Resolve either a Task-1 run root or its ``example_dirs`` directory."""
    model_dir = Path(model_dir)
    if model_dir.name == "example_dirs":
        return model_dir.parent.name, model_dir.parent, model_dir

    nested_examples = model_dir / "example_dirs"
    if nested_examples.is_dir():
        return model_dir.name, model_dir, nested_examples

    # Preserve support for historical/custom paths that directly contain the
    # per-instance directories but are not themselves named ``example_dirs``.
    return model_dir.name, model_dir, model_dir


def task2_continuation_files(instance_dir: Path) -> list[Path]:
    """Return the reading-conditioned continuation files used by Task 2."""
    target_files = sorted(instance_dir.glob("y*.jsonl"))
    if target_files:
        return target_files
    return sorted(
        path
        for path in instance_dir.glob("*.jsonl")
        if "prompts" not in path.name and path.name != "d.jsonl"
    )


def run(args) -> int:
    print(f"=== Starting Task 2: Quality & Diversity Evaluation ===")
    
    # Enforce reproducibility
    set_global_determinism(args.seed)
    output_suffix = sanitize_suffix(getattr(args, "output_suffix", None))
    skip_diversity = bool(getattr(args, "skip_diversity", False))
    skip_ppl = bool(getattr(args, "skip_ppl", False))
    ppl_batch_size = int(getattr(args, "ppl_batch_size", 16))
    ppl_max_length = getattr(args, "ppl_max_length", None)

    resolved_inputs = []
    for requested_dir in args.model_dirs:
        if not requested_dir.exists() or not requested_dir.is_dir():
            print(f"[error] Directory not found or invalid: {requested_dir}")
            return 2

        model_name, model_root_dir, examples_root = resolve_task2_model_input(requested_dir)
        instance_dirs = sorted(
            path
            for path in examples_root.iterdir()
            if path.is_dir() and task2_continuation_files(path)
        )
        if not instance_dirs:
            print(
                f"[error] Found no per-instance continuation files below "
                f"'{examples_root}'. Pass either a Task-1 run directory or its "
                "'example_dirs' directory."
            )
            return 2

        print(
            f"[info] Resolved {requested_dir} -> {examples_root} "
            f"({len(instance_dirs):,} instances)."
        )
        resolved_inputs.append(
            (model_name, model_root_dir, examples_root, instance_dirs)
        )
    
    # 1. LOAD MODELS
    embedder = None
    if not skip_diversity:
        print(f"[info] Loading Embedding Model ({args.embed_model}) for Diversity...")
        embedder = SentenceTransformer(args.embed_model, cache_folder=CACHE_DIR)
    else:
        print("[info] Skipping embedding diversity metrics (--skip-diversity).")

    ppl_model = None
    ppl_tokenizer = None
    if not skip_ppl:
        print(f"[info] Loading external-LM PPL model ({args.ppl_model}) for fluency...")
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
    else:
        print("[info] Skipping perplexity metrics (--skip-ppl).")

    all_results = {}
    missing_prompt_warned = False # Flag to avoid spamming the console

    print("\n[info] Commencing Evaluation Loop...")
    for model_name, model_root_dir, examples_root, instance_dirs in tqdm(
        resolved_inputs, desc="Evaluating Configurations", position=0
    ):
        metrics = {
            "diversity_scores": [],
            "perplexity_scores": [],
            "overlap_scores": []
        }
        ppl_texts = []

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
            target_files = task2_continuation_files(instance_dir)
            
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
            if embedder is not None and len(continuations_for_div) >= 2:
                embeddings = embedder.encode(continuations_for_div, show_progress_bar=False, convert_to_numpy=True)
                dists = cosine_distances(embeddings)
                upper_triangle_indices = np.triu_indices_from(dists, k=1)
                
                if len(upper_triangle_indices[0]) > 0:
                    mean_pairwise_distance = np.mean(dists[upper_triangle_indices])
                    metrics["diversity_scores"].append(float(mean_pairwise_distance))

            for text in continuations_for_div:
                overlap = calculate_word_overlap(ambig_prompt, text)
                if ppl_model is not None:
                    ppl_texts.append(text)
                if overlap is not None: 
                    metrics["overlap_scores"].append(overlap)

        if ppl_model is not None and ppl_tokenizer is not None:
            ppl_eval_texts = ppl_texts
            sample_size = getattr(args, "max_ppl_texts_per_dir", None)
            if sample_size is not None and sample_size > 0 and len(ppl_eval_texts) > sample_size:
                rng = random.Random(args.seed)
                sample_indices = sorted(rng.sample(range(len(ppl_eval_texts)), sample_size))
                ppl_eval_texts = [ppl_eval_texts[idx] for idx in sample_indices]
                print(
                    f"\n[info] {model_name}: sampled {len(ppl_eval_texts):,} of "
                    f"{len(ppl_texts):,} continuations for external PPL."
                )
            else:
                print(f"\n[info] {model_name}: scoring {len(ppl_eval_texts):,} continuations for PPL.")

            ppl_values = calculate_perplexities_batch(
                ppl_eval_texts,
                ppl_model,
                ppl_tokenizer,
                batch_size=ppl_batch_size,
                max_length=ppl_max_length,
            )
            for ppl in ppl_values:
                if ppl is not None and not math.isnan(ppl) and not math.isinf(ppl):
                    metrics["perplexity_scores"].append(ppl)

        if valid_files_found == 0:
            print(
                f"\n[error] Looked in {len(instance_dirs)} folders inside "
                f"{examples_root}, but found no valid continuation files."
            )
            return 2

        # --- D. Aggregate and Print Individual Results ---
        individual_json_path = task2_metrics_output_path(model_root_dir, output_suffix)

        model_stats = {
            "diversity_mean_cosine_dist": float(np.mean(metrics["diversity_scores"])) if metrics["diversity_scores"] else None,
            "perplexity_median": float(np.median(metrics["perplexity_scores"])) if metrics["perplexity_scores"] else None,
            "perplexity_mean": float(np.mean(metrics["perplexity_scores"])) if metrics["perplexity_scores"] else None,
            "overlap_mean": float(np.mean(metrics["overlap_scores"])) if metrics["overlap_scores"] else None,
            "num_evaluated_instances": len(instance_dirs),
            "num_ppl_texts_available": len(ppl_texts),
            "num_ppl_texts_scored": len(metrics["perplexity_scores"]),
            "ppl_model": None if skip_ppl else args.ppl_model,
            "ppl_batch_size": None if skip_ppl else ppl_batch_size,
            "ppl_max_length": None if skip_ppl else ppl_max_length,
            "embed_model": None if skip_diversity else args.embed_model,
            "seed_used": args.seed,
            "examples_path": str(examples_root),
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
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("w", encoding="utf-8") as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)
        print(f"[info] Wrote combined Task-2 summary: {args.summary_output}")
    return 0
