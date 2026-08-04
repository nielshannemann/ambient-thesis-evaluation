#!/usr/bin/env python3
# src/ambient/evaluation/run_ambient_experiments.py
"""
=============================================================================
AMBIENT EVALUATION ORCHESTRATOR (SINGLE-MODEL PIPELINE)
=============================================================================
This script orchestrates the generation and scoring of continuations for the 
AMBIENT dataset. It dynamically loads either an Autoregressive (AR) or 
a Diffusion (LLaDA) base model.

Key architectural features for this project:
- Single-Load Policy: The model is loaded exactly once into VRAM and serves 
  as both generator and scorer to prevent memory collisions.
- Batched Exact NLL (for AR) & Batched MC NLL (for Diffusion).
- Strict Determinism for reproducible Ablation Studies.

[Thesis: Methodology > Standardized Evaluation Framework]
=============================================================================
"""

import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import List

import pandas as pd
import torch
import numpy as np
# [Thesis: Methodology > Shared Interface and Inference Controls]
from ambient.adapters import ARAdapter, DreamAdapter, LLaDaAdapter, register_adapter
from ambient.constants import LLADA_BASE_MODEL_ID, LLAMA_BASE_MODEL_ID
from ambient.evaluation.continuation_evaluation_adapted import continuation_evaluation
from ambient.modeling import (
    auto_detect_4bit as shared_auto_detect_4bit,
    canonical_backend,
    default_base_model_id,
    is_masked_diffusion_family,
    load_model_bundle,
    runtime_environment,
)
from ambient.paths import task1_run_dir
from ambient.utils import write_json_atomic

# ==========================================
# CONFIGURATION
# ==========================================
LLADA_MODEL_ID = LLADA_BASE_MODEL_ID
LLAMA_MODEL_ID = LLAMA_BASE_MODEL_ID


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task1_resume_mismatches(
    previous: dict,
    current: dict,
) -> dict[str, tuple[object, object]]:
    """Compare scientific settings while accepting sparse historical metadata."""
    mismatches: dict[str, tuple[object, object]] = {}
    for key in (
        "model_name",
        "model_id",
        "model_type",
        "model_family",
        "backend",
        "reading_order",
    ):
        if key in previous and previous[key] != current.get(key):
            mismatches[key] = (previous[key], current.get(key))

    for section in ("hyperparameters", "data_selection"):
        previous_section = previous.get(section, {})
        current_section = current.get(section, {})
        for key, current_value in current_section.items():
            if key in previous_section and previous_section[key] != current_value:
                mismatches[f"{section}.{key}"] = (
                    previous_section[key],
                    current_value,
                )
    return mismatches


def load_requested_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return {str(value) for value in json.loads(raw)}
    return {
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

def set_seed(seed_val: int):
    """
    Enforces strict reproducibility across CPU, GPU, and NumPy runtimes.
    [Thesis: Implementation and Reproducibility > Reproducibility Boundaries]
    """
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        try:
            # Force deterministic CuDNN algorithms (slightly slower, but exact)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

def fix_tokenizer_pad_token(tokenizer):
    """
    Ensures the tokenizer possesses a valid padding token for batched operations.
    Crucial for LLaMA 3.1 architectures which lack a default structural pad token.
    """
    try:
        if getattr(tokenizer, "pad_token_id", None) is None or getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                if isinstance(tokenizer.eos_token_id, list):
                    tokenizer.pad_token_id = tokenizer.eos_token_id[0]
                else:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    except Exception:
        pass
    return tokenizer

def batched_exact_nll_score(
    model,
    tokenizer,
    prompts: List[str],
    continuations: List[str],
    batch_size: int = 8,
    progress_every: int = 0,
    progress_label: str = "AR exact-NLL scoring",
) -> List[float]:
    """
    Computes the exact Sequence Negative Log-Likelihood for Autoregressive models.
    Utilizes left-aligned manual padding to allow for high-throughput batched inference.
    
    [Thesis: Background > Sequence Scoring and Study 1]
    """
    model.eval()
    results = []
    tokenizer = fix_tokenizer_pad_token(tokenizer)
    started = time.time()
    total_pairs = min(len(prompts), len(continuations))

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_conts = continuations[i:i + batch_size]
            
            batch_input_ids = []
            attention_masks = []
            prompt_lens = []
            
            valid_batch = True
            for p, c in zip(batch_prompts, batch_conts):
                try:
                    if not c.strip():
                        raise ValueError("Empty continuation")
                        
                    # Standardize spacing between prompt and continuation
                    c_spaced = " " + c if not c.startswith(" ") else c
                    full_text = p + c_spaced
                    
                    # Tokenize the full string and the isolated prompt
                    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=False)["input_ids"]
                    p_ids = tokenizer(p, add_special_tokens=True, truncation=False)["input_ids"]
                    
                    # Find exact boundary: First token where full_text diverges from prompt.
                    # This gracefully handles subword merging (e.g., "walk" + "ing" -> "walking").
                    divergence_idx = len(p_ids)
                    for idx, (t_f, t_p) in enumerate(zip(full_ids, p_ids)):
                        if t_f != t_p:
                            divergence_idx = idx
                            break
                            
                    # Failsafe: If the continuation was completely swallowed by a strange tokenization 
                    # artifact, default to the last token.
                    if divergence_idx >= len(full_ids):
                        divergence_idx = max(0, len(full_ids) - 1)

                    batch_input_ids.append(full_ids)
                    prompt_lens.append(divergence_idx)
                except Exception:
                    valid_batch = False
                    break
            
            if not valid_batch:
                results.extend([None] * len(batch_prompts))
                continue
                
            # 2. Manual Right-Padding Initialization
            max_len = max(len(ids) for ids in batch_input_ids)
            pad_token_id = tokenizer.pad_token_id
            
            padded_input_ids = []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                padded_input_ids.append(ids + [pad_token_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
                
            input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(model.device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(model.device)

            # 3. Forward Pass & Log-Softmax computation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            
            # 4. NLL Extraction isolated specifically to the Continuation Subset
            for j in range(len(batch_prompts)):
                try:
                    total_len = int(attention_mask[j].sum().item())
                    len_prompt = prompt_lens[j]

                    if total_len <= len_prompt:
                        results.append(None)
                        continue
                    
                    start_logit_pos = len_prompt - 1
                    end_logit_pos = total_len - 1  

                    relevant_logits = log_probs[j, start_logit_pos : end_logit_pos, :]
                    relevant_ids = input_ids[j, len_prompt : total_len]

                    token_log_probs = relevant_logits.gather(dim=1, index=relevant_ids.unsqueeze(-1)).squeeze(-1)
                    total_nll = -token_log_probs.sum().item()
                    results.append(total_nll)

                except Exception:
                    results.append(None)

            processed = min(i + len(batch_prompts), total_pairs)
            if progress_every and (processed % progress_every < len(batch_prompts) or processed == total_pairs):
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] {progress_label}: {processed}/{total_pairs} pairs "
                    f"({rate:.2f} pairs/s, {elapsed / 60:.1f} min)"
                )

    return results

def auto_detect_4bit(model_id: str) -> bool:
    """
    Dynamically decides whether 4-bit quantization is required based on available VRAM.
    [Thesis: Implementation and Reproducibility > Execution Environment]
    """
    return shared_auto_detect_4bit(model_id)

def run(args) -> int:
    data_path = args.data_path
    model_name = args.model_name
    model_id = args.model_id
    backend = canonical_backend(args.model_family)
    model_type = "diffusion" if is_masked_diffusion_family(args.model_family) else "ar"
    num_generations = args.num_generations
    gen_batch_size = args.batch_size
    diffusion_steps = args.diffusion_steps
    mc_num = args.mc_num
    mc_batch_size = args.mc_batch_size
    cfg_scale = args.cfg_scale
    top_p = args.top_p
    seed = args.seed
    top_k = args.top_k
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    stop_at_sentence = args.stop_at_sentence

    is_diffusion = model_type == "diffusion"
    # Use user-provided model ID if given, else fall back to the instruct defaults
    if model_id is None:
        model_id = default_base_model_id(args.model_family)

    print(f"=== Starting AMBIENT Pipeline ({model_type.upper()}) ===")
    set_seed(seed)
    
    # Dynamic Directory Naming and MC List Parsing
    dir_name = f"{model_name}-n{num_generations}"
    
    if model_type == "diffusion":
        dir_name += f"-d{diffusion_steps}"
        mc_list = sorted([int(x.strip()) for x in str(mc_num).split(",")])
        summary_names = [f"summary_mc{m}.jsonl" for m in mc_list]
    else:
        # Fallback structural enforcement for AR models
        mc_list = [1] 
        summary_names = ["summary.jsonl"]
        
    out_dir = args.output_dir or task1_run_dir(
        model_name=model_name,
        num_generations=num_generations,
        model_family=args.model_family,
        diffusion_steps=diffusion_steps,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_meta.json"
    use_4bit = auto_detect_4bit(model_id) if args.use_4bit is None else args.use_4bit

    # 1. METADATA RECORDING
    hyperparams = {
        "seed": seed,
        "num_generations": num_generations,
        "gen_batch_size": gen_batch_size,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop_at_sentence": stop_at_sentence,
        "use_4bit": use_4bit,
    }

    # Restrict diffusion-specific parameter logging to diffusion models
    if model_type == "diffusion":
        hyperparams.update({
            "diffusion_steps": diffusion_steps,
            "mc_nums": mc_list,
            "mc_batch_size": mc_batch_size,
            "cfg_scale": cfg_scale,
            "diffusion_alg": getattr(args, "diffusion_alg", "entropy"),
            "diffusion_alg_temp": getattr(args, "diffusion_alg_temp", 0.0),
        })

    run_meta = {
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "model_id": model_id,
        "model_type": model_type,
        "model_family": args.model_family,
        "backend": backend,
        "runtime_environment": runtime_environment(),
        "data_selection": {
            "data_path": str(data_path),
            "data_sha256": file_sha256(data_path),
            "id_file": str(args.id_file) if args.id_file else None,
            "id_file_sha256": file_sha256(args.id_file) if args.id_file else None,
            "max_examples": args.max_examples,
        },
        "reading_order": "benchmark_order_with_stable_deduplication",
        "hyperparameters": hyperparams,
        "status": "running"
    }

    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            previous_meta = json.load(handle)
        mismatches = task1_resume_mismatches(previous_meta, run_meta)
        if mismatches:
            raise ValueError(
                "Cannot resume Task-1 output with incompatible metadata: "
                f"{mismatches}"
            )
        run_meta["resume"] = {
            "previous_status": previous_meta.get("status"),
            "previous_timestamp_start": previous_meta.get("timestamp_start"),
        }
    write_json_atomic(meta_path, run_meta)

    test_df = pd.read_json(data_path, lines=True)
    requested_ids = load_requested_ids(args.id_file)
    if requested_ids is not None:
        available_ids = test_df["id"].astype(str)
        test_df = test_df[available_ids.isin(requested_ids)]
        found_ids = set(test_df["id"].astype(str))
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise ValueError(f"ID file contains {len(missing_ids)} IDs absent from {data_path}.")
    if args.max_examples is not None:
        test_df = test_df.head(args.max_examples)
    run_meta["data_selection"]["num_selected_dataset_rows"] = len(test_df)
    write_json_atomic(meta_path, run_meta)
    print(f"[INFO] Task-1 data selection contains {len(test_df)} dataset rows.")

    print(f"[INFO] Loading {model_id} (4-bit: {use_4bit}) ONCE for generation and scoring...")

    # 2. MODEL INITIALIZATION & ADAPTER INJECTION
    bundle = load_model_bundle(
        model_family=args.model_family,
        model_id=model_id,
        use_4bit=use_4bit,
        verbose=True,
    )
    model, tokenizer = bundle.model, bundle.tokenizer
    try:
        run_meta["resolved_model_dtype"] = str(next(model.parameters()).dtype)
    except Exception:
        run_meta["resolved_model_dtype"] = None
    write_json_atomic(meta_path, run_meta)

    if model_type == "diffusion":
        from ambient.evaluation.get_log_likelihood import get_log_likelihood
        
        def diff_score_wrapper(prompts, continuations, mc_nums=None):
            if mc_nums is None:
                mc_nums = mc_list
            spaced_conts = [" " + c if not c.startswith(" ") else c for c in continuations]
            return get_log_likelihood(
                model=model, 
                tokenizer=tokenizer, 
                prompts=prompts, 
                continuations=spaced_conts,
                mc_nums=mc_nums,
                batch_size=mc_batch_size,
                cfg_scale=cfg_scale,
                seed=seed,
                progress_every=getattr(args, "score_progress_every", 20),
                progress_label="Task-1 diffusion scoring",
            )
            
        adapter_class = LLaDaAdapter if backend == "llada" else DreamAdapter
        adapter = adapter_class(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            diff_mc_nll=diff_score_wrapper,
        )
        
    elif model_type == "ar":
        tokenizer = fix_tokenizer_pad_token(tokenizer)

        # Utilize the high-throughput batched AR scorer
        def ar_score_wrapper(prompts, continuations, mc_nums=None):
            scores = batched_exact_nll_score(
                model,
                tokenizer,
                prompts,
                continuations,
                batch_size=mc_batch_size,
                progress_every=getattr(args, "score_progress_every", 20),
            )
            # Wrap in an outer list to strictly mirror the Multi-Level "mc_nums" output structure of Diffusion
            return [scores]

        adapter = ARAdapter(model_name=model_name, model=model, tokenizer=tokenizer, ar_score_fn=ar_score_wrapper)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    register_adapter(model_name, adapter)
    print(f"[INFO] Successfully registered and injected {adapter.__class__.__name__}.")

    # 3. RUN EVALUATION PIPELINE
    exit_code = 0
    try:
        results = continuation_evaluation(
            test_df=test_df, 
            model_name=model_name,
            out_dir=out_dir,
            mc_nums=mc_list,
            summary_names=summary_names,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_generations=num_generations,
            batch_size=gen_batch_size,
            seed=seed,
            max_new_tokens=max_new_tokens,
            stop_at_sentence=stop_at_sentence,
            steps=diffusion_steps,
            cfg_scale=cfg_scale,
            diffusion_alg=getattr(args, "diffusion_alg", "entropy"),
            diffusion_alg_temp=getattr(args, "diffusion_alg_temp", 0.0),
            progress_every_chunks=getattr(args, "progress_every_chunks", 1),
        )
        
        print(f"\n[INFO] Evaluation finished successfully. Results written to: {out_dir}")
        run_meta["status"] = "finished"

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed fatally: {e}")
        traceback.print_exc()
        run_meta["status"] = "failed"
        run_meta["error"] = str(e)
        exit_code = 1
    finally:
        run_meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json_atomic(meta_path, run_meta)
    return exit_code
