#!/usr/bin/env python3
# src/ambient/evaluation/task1_disambiguation.py
"""
=============================================================================
TASK 1: EXPLICIT GENERATIVE DISAMBIGUATION
=============================================================================
Evaluates the capability of instruction-tuned models to explicitly identify 
and verbalize multiple valid semantic interpretations of an ambiguous premise.

Methodological Integration:
- Employs One-Shot In-Context Learning combined with Assistant Prefilling 
  to strictly enforce the output schema.
- Utilizes the unified `ARAdapter` and `LLaDaAdapter` for architectural parity.
- Ensures 100% deterministic generation via isolated instance-level seeding.
- Features dynamic 4-bit quantization detection and comprehensive metadata 
  serialization for strict experimental reproducibility.

[Thesis Reference: Section 3.2.1 - Task 1: Explicit Generative Disambiguation]
=============================================================================
"""

import json
import torch
import argparse
import random
import os
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# Custom AmbiEnt modules
from ambient.llada_loader import load_llada_model
from ambient.adapters import ARAdapter, LLaDaAdapter

# ==========================================
# CONFIGURATION
# ==========================================
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def auto_detect_4bit(hf_model: str) -> bool:
    """
    Dynamically determines whether 4-bit quantization (NF4) is required 
    based on the available GPU memory (VRAM) and the model scale.
    """
    if not torch.cuda.is_available():
        return False
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    hf_lower = hf_model.lower()
    
    if "70b" in hf_lower or "72b" in hf_lower:
        return vram_gb < 130
    if "8b" in hf_lower or "7b" in hf_lower or "9b" in hf_lower:
        return vram_gb < 20
        
    return vram_gb < 16

def load_ambient_data(path: Path, max_examples: int = 50) -> list:
    """
    Parses the AMBIENT dataset and isolates explicitly ambiguous instances.
    """
    data = []
    if not path.exists():
        print(f"[ERROR] Dataset not found at {path}")
        return data
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Ensure we only evaluate instances with confirmed semantic ambiguity
            if obj.get("premise_ambiguous", False) or obj.get("hypothesis_ambiguous", False):
                data.append(obj)
                if len(data) >= max_examples:
                    break
    return data

def clean_generated_interpretations(raw_text: str) -> str:
    """
    Cleans the raw output to strictly extract the enumerated interpretations.
    [Thesis Reference: Section 3.2.2 - Output Sanitization and Artifact Filtering]
    """
    for cutoff_string in ["\nuser:", "user:", "\nContext:", "<|", "We don't know"]:
        if cutoff_string in raw_text:
            raw_text = raw_text.split(cutoff_string)[0]
            
    lines = raw_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and (len(line) > 1 and line[1] in ".)"):
            valid_lines.append(line)
        elif line and not valid_lines:
            valid_lines.append(line)
            
    valid_lines = valid_lines[:2] 
    clean_text = "\n".join(valid_lines).strip()
    
    if not clean_text:
        return raw_text.strip()
        
    if clean_text and clean_text[0].isdigit():
        return clean_text
    else:
        return "1. " + clean_text

def main():
    parser = argparse.ArgumentParser(description="Task 1: Explicit Generative Disambiguation")
    parser.add_argument("--model-name", type=str, required=True, help="E.g., 'llama8b', 'llada8b'")
    parser.add_argument("--model-type", choices=["llama", "llada"], required=True, help="Target instruct-tuned architecture.")
    parser.add_argument("--model-id", type=str, default=None, help="HuggingFace repository ID.")
    parser.add_argument("--data-path", type=Path, default=Path("external/ambient/AmbiEnt/test_baked.jsonl"))
    parser.add_argument("--max-examples", type=int, default=580, help="Number of examples to evaluate.")
    
    # --- HARDWARE & REPRODUCIBILITY ABLATIONS ---
    parser.add_argument("--num-continuations", type=int, default=1, help="Total number of disambiguation attempts generated per premise (N).")
    parser.add_argument("--batch-size", type=int, default=25, help="Maximum sequences generated in parallel (prevents VRAM OOM).")
    parser.add_argument("--seed", type=int, default=42, help="Global deterministic random seed.")
    
    # --- GENERATION HYPERPARAMETERS ---
    parser.add_argument("--temperature", type=float, default=1.0, help="Stochasticity scaling factor for sampling.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cumulative probability threshold.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-K absolute probability truncation (0 = disabled).")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="Classifier-Free Guidance multiplier (LLaDA architecture only).")
    parser.add_argument("--diffusion-steps", type=int, default=128, help="Number of reverse-generation denoising steps (LLaDA only).")
    args = parser.parse_args()

    print(f"=== Starting Task 1: Explicit Disambiguation ===")
    
    # 1. STRICT GLOBAL DETERMINISM
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    is_diffusion = (args.model_type == "llada")
    # Use user-provided model ID if given, else fall back to the instruct defaults
    if args.model_id is None:
        model_id = LLADA_MODEL_ID if is_diffusion else LLAMA_MODEL_ID
    else:
        model_id = args.model_id
    use_4bit = auto_detect_4bit(model_id)
    
    print(f"[INFO] Selected Model: {model_id} (Diffusion: {is_diffusion})")
    print(f"[INFO] Hardware Setting: Auto-detected 4-bit Quantization = {use_4bit}")
    print(f"[INFO] Generation: {args.num_continuations} attempts per premise (Batches of {args.batch_size})")
    print(f"[INFO] Hyperparameters: Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}, CFG={args.cfg_scale}, Steps={args.diffusion_steps}")
    
    # Configure output path
    out_name = f"{args.model_name}_n{args.num_continuations}.json"
    out_path = Path(f"results/task1/{out_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- METADATA RECORDING ---
    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "task1_explicit_disambiguation",
        "model_type": args.model_type,
        "model_id": model_id,
        "hyperparameters": {
            "max_examples": args.max_examples,
            "num_continuations": args.num_continuations,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p
        }
    }
    if is_diffusion:
        run_meta["hyperparameters"]["cfg_scale"] = args.cfg_scale
        run_meta["hyperparameters"]["diffusion_steps"] = args.diffusion_steps

    dataset = load_ambient_data(args.data_path, max_examples=args.max_examples)
    print(f"[INFO] Isolated {len(dataset)} ambiguous instances for evaluation.")

    # --- ARCHITECTURE INITIALIZATION & ADAPTER INJECTION ---
    print("[INFO] Initializing architecture and injecting unified adapters...")
    if is_diffusion:
        model, tokenizer = load_llada_model(hf_model=model_id, use_4bit=use_4bit, verbose=False)
        adapter = LLaDaAdapter(model_name=model_id, model=model, tokenizer=tokenizer, diff_mc_nll=None)
    else:
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "cache_dir": "./models"}
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
            )
            
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
            
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        adapter = ARAdapter(model_name=model_id, model=model, tokenizer=tokenizer, ar_score_fn=None)

    # --- LATENT SAMPLING LOOP ---
    print("[INFO] Commencing generative disambiguation...")
    all_results = []
    
    for prompt_idx, row in enumerate(tqdm(dataset, desc="Generating")):
        row_id = row.get("id") or row.get("_instance_id", "unknown")
        premise = row.get("ambiguous_sentence") or row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        
        # METHODOLOGY ALIGNMENT: One-Shot In-Context Learning + Assistant Prefilling
        messages = [
            {"role": "system", "content": "In each example, you will be given some context and a claim, where the correctness of the claim is affected by some ambiguity in the context. Enumerate two distinct interpretations of the context that lead to different judgments about the claim. Format them strictly as a numbered list."},
            {"role": "user", "content": "Context: I'm afraid the cat was hit by a car.\nClaim: The cat was not hit by a car. Given the context alone, is this claim true, false, or inconclusive?"},
            {"role": "assistant", "content": "We don't know, because the context can be interpreted in many different ways:\n1. I'm worried the cat was hit by a car. Then the claim is inconclusive.\n2. I'm sorry to share that the cat was hit by a car. Then the claim is false."},
            {"role": "user", "content": f"Context: {premise}\nClaim: {hypothesis} Given the context alone, is this claim true, false, or inconclusive?"}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Assistant-Prefilling: Forces the model into the correct enumeration schema
        input_text += "We don't know, because the context can be interpreted in many different ways:\n1."
        
        # 2. STRICT INSTANCE DETERMINISM
        current_seed = args.seed + prompt_idx
        
        # Route generation strictly through the standardized Adapter framework
        raw_responses = adapter.generate(
            prompt=input_text,
            num_return_sequences=args.num_continuations,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            max_new_tokens=160,
            stop_at_sentence=False, 
            seed=current_seed,
            steps=args.diffusion_steps
        )
        
        # Iterate through all generated responses for this premise
        fixed_raw_list = []
        cleaned_list = []
        
        for raw_resp in raw_responses:
            raw_text = raw_resp if raw_resp else ""
            
            # ASSISTANT PREFILL PARSING FIX
            if raw_text and not raw_text.lstrip().startswith("1"):
                raw_text = "1. " + raw_text.lstrip()
                
            clean_text = clean_generated_interpretations(raw_text)
            
            fixed_raw_list.append(raw_text)
            cleaned_list.append(clean_text)
        
        all_results.append({
            "id": row_id,
            "premise": premise,
            "hypothesis": hypothesis,
            "generated_raw": fixed_raw_list,
            "generated_clean": cleaned_list
        })

    # --- FINAL DATA SERIALIZATION (Metadata + Results) ---
    final_output = {
        "metadata": run_meta,
        "results": all_results
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Task 1 complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()