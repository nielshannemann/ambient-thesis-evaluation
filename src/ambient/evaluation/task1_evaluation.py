#!/usr/bin/env python3
# src/ambient/evaluation/task1_evaluation.py
"""
=============================================================================
TASK 1: LLM-AS-A-JUDGE EVALUATION
=============================================================================
This script evaluates the explicit generative disambiguations produced in Task 1.
It utilizes a frozen Instruct model (e.g., Llama-3.1-70B) to act as an impartial 
judge, evaluating which architecture better identified and explained the latent 
ambiguity.

Methodological Integration:
- Employs dynamic HF_HOME routing for cluster-safe model loading.
- Uses the unified `ARAdapter` for the judge model to maintain architectural parity.
- Enforces strict global determinism and instance-level seeding to guarantee 
  reproducible Blind A/B randomization.

[Thesis Ref: Section 2.6.2 - Implementation of the Blind A/B Testing]
=============================================================================
"""

import os

import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# Custom AmbiEnt modules
from ambient.adapters import ARAdapter


def set_global_determinism(seed: int):
    """
    Locks all random number generators and backend heuristics to 
    guarantee 100% exact reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def load_results(path: Path) -> Dict[str, dict]:
    """
    Loads the generated disambiguations mapped by instance ID.
    Parses the structured JSON format containing metadata and results.
    """
    data = {}
    if not path.exists():
        print(f"[error] File not found: {path}")
        return data
        
    with open(path, "r", encoding="utf-8") as f:
        full_json = json.load(f)
        
        # Extract the list of results from the structured JSON
        results_list = full_json.get("results", [])
        
        for obj in results_list:
            data[str(obj["id"])] = obj
            
    return data




def get_context_and_claim(instance: dict):
    """
    Reads the dynamically reconstructed Task-1 conversational pair. Falls back
    to legacy keys for older result files.
    """
    context = instance.get("context_text")
    claim = instance.get("claim_text")

    if context is None:
        context = instance.get("premise", "")
    if claim is None:
        claim = instance.get("hypothesis", "")

    return context, claim

def evaluate_pair(
    judge_adapter: ARAdapter, 
    premise: str,
    hypothesis: str,
    cont_llada: str, 
    cont_llama: str,
    instance_seed: int
) -> Tuple[str, str, str]:
    """
    Constructs the prompt, deterministically randomizes the order of the 
    continuations to prevent position bias, and queries the LLM judge via the Adapter.
    """
    # 1. Randomize assignment to mitigate position bias deterministically
    rng = random.Random(instance_seed)
    llada_is_model_a = rng.choice([True, False])
    
    model_a_text = cont_llada if llada_is_model_a else cont_llama
    model_b_text = cont_llama if llada_is_model_a else cont_llada

    # 2. Construct the exact evaluation prompt WITH ONE-SHOT GRADING DEMONSTRATION
    user_prompt = f"""--- Example ---
Context: I'm afraid the cat was hit by a car.
Claim: The cat was not hit by a car.

Model A's interpretation:
1. The cat died. Then the claim is false.

Model B's interpretation:
1. I'm worried the cat was hit. Then the claim is inconclusive.
2. I'm sorry to share that the cat was hit. Then the claim is false.

Winner (Model A, Model B, or Tie): Model B
--- End of Example ---

Context: {premise}
Claim: {hypothesis}

Model A's interpretation:
{model_a_text}

Model B's interpretation:
{model_b_text}

Winner (Model A, Model B, or Tie):"""

    messages = [
        {"role": "system", "content": "You are an impartial judge evaluating AI language models based on how well they identify ambiguity. A good interpretation explicitly states the different ways the context can be understood and how it affects the claim. You must output STRICTLY 'Model A', 'Model B', or 'Tie' as your final answer, nothing else."},
        {"role": "user", "content": user_prompt}
    ]

    tokenizer = judge_adapter.tokenizer
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. Query the Judge strictly through the Adapter Framework
    # We use temp=0.01 and top_k=1 to approximate greedy decoding within the adapter
    raw_responses = judge_adapter.generate(
        prompt=input_text,
        num_return_sequences=1,
        batch_size=1,
        temperature=0.01, 
        top_k=1,
        max_new_tokens=10,
        stop_at_sentence=False,
        seed=instance_seed
    )
    
    judge_response = raw_responses[0].strip() if raw_responses else ""
    
    # 4. Resolve the randomization back to architecture names
    # Using .startswith to prevent malicious mid-sentence mentions
    if judge_response.startswith("Model A"):
        winner_model = "LLaDA" if llada_is_model_a else "LLaMA-8B"
    elif judge_response.startswith("Model B"):
        winner_model = "LLaMA-8B" if llada_is_model_a else "LLaDA"
    elif judge_response.startswith("Tie"):
        winner_model = "Tie"
    else:
        winner_model = "Tie" # Graceful fallback for unexpected outputs
        
    return winner_model, judge_response, ("A" if llada_is_model_a else "B")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation for Task 1")
    parser.add_argument("--llada-file", type=Path, default=Path("results/task1/llada8b_n100.json"), help="Path to LLaDA generations")
    parser.add_argument("--llama-file", type=Path, default=Path("results/task1/llama8b_n100.json"), help="Path to LLaMA generations")
    parser.add_argument("--judge-model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct", help="HuggingFace ID of the judge model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for blind A/B test assignment [Thesis Ref: Section 2.6.2]")
    parser.add_argument("--disable-4bit", action="store_true", help="Disable 4-bit quantization (BitsAndBytes NF4)")
    args = parser.parse_args()

    print(f"=== Starting LLM-as-a-Judge Evaluation Pipeline ===")
    print(f"[info] Judge Model: {args.judge_model}")
    print(f"[info] Base Seed: {args.seed}")
    
    # Enforce strict global reproducibility
    set_global_determinism(args.seed)
    
    llada_data = load_results(args.llada_file)
    llama_data = load_results(args.llama_file)
    
    # Sort the IDs to ensure deterministic iteration order
    common_ids = sorted(list(set(llada_data.keys()).intersection(set(llama_data.keys()))))
    print(f"[info] Found {len(common_ids)} overlapping instances for evaluation.")
    
    if not common_ids:
        print("[error] No overlapping data found. Exiting.")
        return

    # --- INITIALIZATION ---
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "cache_dir": "./models"}
    
    if not args.disable_4bit:
        print("[info] Activating 4-bit Quantization (BitsAndBytes NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        print("[info] 4-bit Quantization DISABLED.")

    print(f"[INFO] Initializing architecture and injecting ARAdapter for the Judge...")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
        
    model = AutoModelForCausalLM.from_pretrained(args.judge_model, **load_kwargs)
    model.eval()
    
    judge_adapter = ARAdapter(model_name=args.judge_model, model=model, tokenizer=tokenizer, ar_score_fn=None)

    scores = {"LLaDA": 0, "LLaMA-8B": 0, "Tie": 0}

    # --- EVALUATION LOOP ---
    print("[info] Commencing blind A/B evaluation...")
    for idx_num, idx in enumerate(tqdm(common_ids, desc="Judging")):
        llada_instance = llada_data[idx]
        llama_instance = llama_data[idx]
        
        context_text, claim_text = get_context_and_claim(llada_instance)
        
        # Extract the cleaned interpretations generated in Task 1
        cont_llada_list = llada_instance.get("generated_clean", [])
        cont_llama_list = llama_instance.get("generated_clean", [])
        
        # Default to the first continuation for the standard A/B test
        cont_llada = cont_llada_list[0] if cont_llada_list else ""
        cont_llama = cont_llama_list[0] if cont_llama_list else ""
        
        # Skip if both failed to generate anything meaningful
        if not cont_llada and not cont_llama:
            scores["Tie"] += 1
            continue
            
        # Instance level seed for A/B randomization
        instance_seed = args.seed + idx_num
            
        winner_model, raw_response, llada_position = evaluate_pair(
            judge_adapter, context_text, claim_text, cont_llada, cont_llama, instance_seed
        )
        
        scores[winner_model] += 1

    # --- AGGREGATION & REPORTING ---
    print("\n" + "="*50)
    print("--- FINAL LLM-AS-A-JUDGE SCORES ---")
    for k, v in scores.items():
        print(f"{k:10s}: {v} ({(v/len(common_ids))*100:.2f}%)")
    print("="*50)


if __name__ == "__main__":
    main()