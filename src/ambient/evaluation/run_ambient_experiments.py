#!/usr/bin/env python3
# src/ambient/run_ambient_experiments.py
"""
=============================================================================
AMBIENT EVALUATION ORCHESTRATOR (SINGLE-MODEL PIPELINE)
=============================================================================
This script orchestrates the generation and scoring of continuations for the 
AMBIENT dataset. It dynamically loads either an Autoregressive (AR) or 
a Diffusion (LLaDA) base model.

Key architectural features for this thesis:
- Single-Load Policy: The model is loaded exactly once into VRAM and serves 
  as both generator and scorer to prevent memory collisions.
- Batched Exact NLL (for AR) & Batched MC NLL (for Diffusion).
- Strict Determinism for reproducible Ablation Studies.

[Thesis Reference: Section 3.1.1 - Overview of the Experimental Pipeline]
=============================================================================
"""

import os
import time
import json
import random
import traceback
import math
from pathlib import Path
from typing import List
import pandas as pd

import click
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# [Thesis Reference: Section 3.1.2 - The Adapter Framework]
from ambient.adapters import LLaDaAdapter, ARAdapter, register_adapter
from ambient.llada_loader import load_llada_model
from ambient.evaluation.continuation_evaluation_adapted import continuation_evaluation
from ambient.utils import write_json_atomic

# ==========================================
# CONFIGURATION
# ==========================================
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

def set_seed(seed_val: int):
    """
    Enforces strict reproducibility across CPU, GPU, and NumPy runtimes.
    [Thesis Reference: Section 3.3.2 - Experimental Setup and Reproducibility]
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

def batched_exact_nll_score(model, tokenizer, prompts: List[str], continuations: List[str], batch_size: int = 8) -> List[float]:
    """
    Computes the exact Sequence Negative Log-Likelihood for Autoregressive models.
    Utilizes left-aligned manual padding to allow for high-throughput batched inference.
    
    [Thesis Reference: Equation 2 (AR Exact Likelihood)]
    """
    model.eval()
    results = []
    tokenizer = fix_tokenizer_pad_token(tokenizer)

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_conts = continuations[i:i + batch_size]
            
            batch_input_ids = []
            attention_masks = []
            prompt_lens = []
            
            # 1. Independent Tokenization to prevent space-merging artifacts
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

    return results

def auto_detect_4bit(model_id: str) -> bool:
    """
    Dynamically decides whether 4-bit quantization is required based on available VRAM.
    [Thesis Reference: Section 3.1.1 - 4-bit Quantization via BitsAndBytes]
    """
    if not torch.cuda.is_available():
        return False
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    model_id_lower = model_id.lower()
    
    # Large Models (e.g., Llama-70B)
    if "70b" in model_id_lower or "65b" in model_id_lower or "72b" in model_id_lower:
        return vram_gb < 130
    # Medium Models (e.g., LLaDA-8B, Llama-8B)
    if "8b" in model_id_lower or "7b" in model_id_lower or "9b" in model_id_lower:
        return vram_gb < 20
        
    return vram_gb < 16

@click.command()
@click.option("--data-path", type=click.Path(exists=True), default="data/test_baked.jsonl")
@click.option("--model-name", type=str, required=True, help="E.g., 'llama8b', 'llada8b'")
@click.option("--model-id", type=str, required=False, help="Hugging Face Model ID")
@click.option("--model-type", default=None, type=click.Choice(["ar", "diffusion"]), required=True)
# --- Hardware & Reproducibility Parameters ---
@click.option("--num-generations", type=int, default=100, help="Total number of sampled continuations (N).")
@click.option("--gen-batch-size", type=int, default=25, help="Batch size for parallel text generation (chunking).")
@click.option("--seed", type=int, default=42, help="Global deterministic random seed.")
# --- Ablation Study Hyperparameters (Thesis Ref: Section 3.3.2) ---
@click.option("--diffusion-steps", type=int, default=64, help="T: Number of unmasking steps.")
@click.option("--mc-num", type=str, default="128", help="Comma-separated list of MC iterations, e.g. '2,16,128'")
@click.option("--mc-batch-size", type=int, default=16, help="Batch size specifically for Monte Carlo NLL scoring.")
@click.option("--cfg-scale", type=float, default=0.0, help="Classifier-Free Guidance Scale.")
@click.option("--top-p", type=float, default=1.0, help="Top-P sampling for generation.")
@click.option("--top-k", type=int, default=0, help="Top-K sampling for generation.")
@click.option("--temperature", type=float, default=1.0, help="Temperature for generation.")
def main(data_path, model_name, model_id, model_type, num_generations, gen_batch_size, 
         diffusion_steps, mc_num, mc_batch_size, cfg_scale, top_p, seed, top_k, temperature):
    
    is_diffusion = model_type == "diffusion"
    # Use user-provided model ID if given, else fall back to the instruct defaults
    if model_id is None:
        if is_diffusion:
            model_id = LLADA_MODEL_ID
        else:
            model_id = LLAMA_MODEL_ID

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
        
    out_dir = Path(f"results/{dir_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_meta.json"

    # 1. METADATA RECORDING
    hyperparams = {
        "seed": seed,
        "num_generations": num_generations,
        "gen_batch_size": gen_batch_size,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature
    }

    # Restrict diffusion-specific parameter logging to diffusion models
    if model_type == "diffusion":
        hyperparams.update({
            "diffusion_steps": diffusion_steps,
            "mc_nums": mc_list,
            "mc_batch_size": mc_batch_size,
            "cfg_scale": cfg_scale
        })

    run_meta = {
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "model_id": model_id,
        "model_type": model_type,
        "hyperparameters": hyperparams,
        "status": "running"
    }
    write_json_atomic(meta_path, run_meta)

    use_4bit = auto_detect_4bit(model_id)
    print(f"[INFO] Loading {model_id} (Auto 4-bit: {use_4bit}) ONCE for Generate & Score architecture...")

    # 2. MODEL INITIALIZATION & ADAPTER INJECTION
    if model_type == "diffusion":
        model, tokenizer = load_llada_model(hf_model=model_id, use_4bit=use_4bit)
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
                seed=seed
            )
            
        adapter = LLaDaAdapter(model_name=model_name, model=model, tokenizer=tokenizer, diff_mc_nll=diff_score_wrapper)
        
    elif model_type == "ar":
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "cache_dir": "./models"}
        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer = fix_tokenizer_pad_token(tokenizer)

        # Utilize the high-throughput batched AR scorer
        def ar_score_wrapper(prompts, continuations, mc_nums=None):
            scores = batched_exact_nll_score(model, tokenizer, prompts, continuations, batch_size=mc_batch_size)
            # Wrap in an outer list to strictly mirror the Multi-Level "mc_nums" output structure of Diffusion
            return [scores]

        adapter = ARAdapter(model_name=model_name, model=model, tokenizer=tokenizer, ar_score_fn=ar_score_wrapper)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    register_adapter(model_name, adapter)
    print(f"[INFO] Successfully registered and injected {adapter.__class__.__name__}.")

    # 3. RUN EVALUATION PIPELINE
    test_df = pd.read_json(data_path, lines=True)

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
            batch_size=gen_batch_size, # <--- Passes the chunking limit down to the adapter
            seed=seed,
            steps=diffusion_steps,
            cfg_scale=cfg_scale
        )
        
        print(f"\n[INFO] Evaluation finished successfully. Results written to: {out_dir}")
        run_meta["status"] = "finished"

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed fatally: {e}")
        traceback.print_exc()
        run_meta["status"] = "failed"
        run_meta["error"] = str(e)
    finally:
        run_meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json_atomic(meta_path, run_meta)

if __name__ == "__main__":
    main()