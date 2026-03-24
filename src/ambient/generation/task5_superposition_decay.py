#!/usr/bin/env python3
# src/ambient/evaluation/task5_superposition_decay.py
"""
=============================================================================
TASK 5: TEMPORAL SEMANTIC COMMITMENT (SUPERPOSITION DECAY)
=============================================================================
This experiment directly tests the core thesis hypothesis: Autoregressive (AR) 
models suffer from early spatial mode collapse, whereas Discrete Text Diffusion 
models hold multiple semantic interpretations in temporal superposition.

Methodological Innovation:
- AR Track (Spatial): Generates continuations token-by-token. At each step, 
  computes the exact likelihood of the opposing gold hypotheses to pinpoint 
  the exact token index of irreversible commitment.
- Diffusion Track (Temporal): Evaluates the NLL of opposing hypotheses at 
  graduating forward-masking ratios (from 100% masked to 0% masked) to track 
  superposition decay over diffusion timesteps.

[Thesis Ref: Section 3.5 - Measuring Early Commitment and Entropy Decay]
=============================================================================
"""

# ==========================================
# CONFIGURATION
# ==========================================
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# Custom AmbiEnt modules
from ambient.llada_loader import load_llada_model
from ambient.adapters import ARAdapter

# The LLaDA official mask ID
LLADA_MASK_ID = 126336

def set_global_determinism(seed: int):
    """Guarantees 100% exact reproducibility for the ablation study."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def load_ambient_opposing_targets(path: Path, max_examples: int = 50) -> list:
    """
    Parses the AMBIENT dataset and isolates instances containing explicit 
    semantic multiplicity. Dynamically identifies whether the premise or 
    the hypothesis harbors the ambiguity based on dataset flags.
    """
    data = []
    if not path.exists():
        print(f"[ERROR] Dataset not found at {path}")
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            ex = json.loads(line)
            
            disambiguations = ex.get("disambiguations", [])
            if len(disambiguations) < 2:
                continue
            
            if ex.get("premise_ambiguous"):
                prompt_text = ex.get("premise", "")
                target_a = disambiguations[0].get("premise", "")
                target_b = disambiguations[1].get("premise", "")
            elif ex.get("hypothesis_ambiguous"):
                prompt_text = ex.get("hypothesis", "")
                target_a = disambiguations[0].get("hypothesis", "")
                target_b = disambiguations[1].get("hypothesis", "")
            else:
                continue
            
            data.append({
                "id": ex.get("id"),
                "prompt": prompt_text,
                "target_a": target_a,
                "target_b": target_b
            })
            
            if len(data) >= max_examples:
                break
                
    return data


def calculate_normalized_entropy(nll_a: float, nll_b: float) -> float:
    """
    Converts Negative Log-Likelihoods into Softmax probabilities and 
    calculates the Shannon Entropy (0.0 = Total Collapse, 1.0 = Perfect Superposition).
    """
    logits = torch.tensor([-nll_a, -nll_b])
    probs = F.softmax(logits, dim=0).numpy()
    
    p_a, p_b = probs[0], probs[1]
    
    entropy = 0.0
    if p_a > 0: entropy -= p_a * np.log2(p_a)
    if p_b > 0: entropy -= p_b * np.log2(p_b)
        
    return float(entropy), float(p_a), float(p_b)


# ==========================================
# AR TRACK LOGIC (SPATIAL DECAY)
# ==========================================
def compute_exact_ar_nll(model, tokenizer, prefix_str: str, target_str: str) -> float:
    """Computes exact sequence NLL for AR models by applying a logit shift."""
    full_str = prefix_str + " " + target_str
    
    # Tokenize independently to find the boundary
    prefix_ids = tokenizer(prefix_str, return_tensors="pt").input_ids[0].to(model.device)
    full_ids = tokenizer(full_str, return_tensors="pt").input_ids[0].to(model.device)
    
    prefix_len = len(prefix_ids)
    
    with torch.no_grad():
        outputs = model(full_ids.unsqueeze(0))
        logits = outputs.logits[0]
        
    # Shift logits and labels for next-token prediction
    shift_logits = logits[prefix_len-1:-1, :].contiguous()
    shift_labels = full_ids[prefix_len:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    nll = loss_fct(shift_logits, shift_labels).item()
    return nll

def ar_spatial_decay_track(model, tokenizer, prompt: str, target_a: str, target_b: str, max_steps: int = 15):
    """Steps through AR generation one token at a time, checking commitment."""
    trajectory = []
    
    # Extract both input_ids and the attention_mask to silence warnings
    encodings = tokenizer(prompt, return_tensors="pt").to(model.device)
    prefix_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    
    for step in range(max_steps + 1):
        prefix_str = tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
        
        # 1. Score hypotheses
        nll_a = compute_exact_ar_nll(model, tokenizer, prefix_str, target_a)
        nll_b = compute_exact_ar_nll(model, tokenizer, prefix_str, target_b)
            
        entropy, p_a, p_b = calculate_normalized_entropy(nll_a, nll_b)
        
        trajectory.append({
            "step": step,
            "prefix_text": prefix_str,
            "nll_a": nll_a,
            "nll_b": nll_b,
            "prob_a": p_a,
            "prob_b": p_b,
            "entropy": entropy
        })
        
        # 2. Generate next single token
        if step < max_steps:
            with torch.no_grad():
                out = model.generate(
                    input_ids=prefix_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1, 
                    do_sample=False, 
                    temperature=None,  # Silences the generation flag warning
                    top_p=None,        # Silences the generation flag warning
                    pad_token_id=tokenizer.eos_token_id
                )
                prefix_ids = out
                # The sequence grew by 1 token, so the mask must grow by 1 (all 1s for AR generation)
                attention_mask = torch.ones_like(prefix_ids)
                
    return trajectory


# ==========================================
# DIFFUSION TRACK LOGIC (TEMPORAL DECAY)
# ==========================================
def compute_diffusion_nll_at_ratio(model, tokenizer, prompt: str, target: str, mask_ratio: float, num_samples: int = 5) -> float:
    """Computes NLL at a specific mask ratio, averaging out variance over permutations."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(prompt + " " + target, return_tensors="pt").input_ids[0]
    
    prompt_len = len(prompt_ids)
    seq_len = len(full_ids)
    target_len = seq_len - prompt_len
    
    if target_len <= 0: return 0.0
        
    num_to_mask = int(round(mask_ratio * target_len))
    total_nll = 0.0
    
    for _ in range(num_samples):
        perturbed_seq = full_ids.clone().to(model.device)
        
        if num_to_mask > 0:
            # Mask random subset of the continuation
            target_indices = torch.randperm(target_len)[:num_to_mask] + prompt_len
            perturbed_seq[target_indices] = LLADA_MASK_ID
            
        with torch.no_grad():
            outputs = model(perturbed_seq.unsqueeze(0))
            logits = outputs.logits[0]
            
        # LLaDA predicts exactly what is masked, no shifting needed
        target_logits = logits[prompt_len:seq_len]
        target_labels = full_ids[prompt_len:seq_len].to(model.device)
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        nll = loss_fct(target_logits, target_labels).item()
        total_nll += nll
        
    return total_nll / num_samples

def diffusion_temporal_decay_track(model, tokenizer, prompt: str, target_a: str, target_b: str, steps: int = 15):
    """Tracks interpretation divergence as the global sequence unmasks over time."""
    trajectory = []
    
    for step in range(steps + 1):
        mask_ratio = 1.0 - (step / steps) # 1.0 (noise) -> 0.0 (clean)
        
        nll_a = compute_diffusion_nll_at_ratio(model, tokenizer, prompt, target_a, mask_ratio)
        nll_b = compute_diffusion_nll_at_ratio(model, tokenizer, prompt, target_b, mask_ratio)
        
        entropy, p_a, p_b = calculate_normalized_entropy(nll_a, nll_b)
        
        trajectory.append({
            "step": step,
            "mask_ratio": mask_ratio,
            "nll_a": nll_a,
            "nll_b": nll_b,
            "prob_a": p_a,
            "prob_b": p_b,
            "entropy": entropy
        })
    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Task 5: Temporal Semantic Commitment (Superposition Decay)")
    parser.add_argument("--data-path", type=Path, default=Path("external/ambient/AmbiEnt/test_baked.jsonl"), help="Path to AMBIENT dataset")
    parser.add_argument("--model-name", type=str, required=True, help="E.g., 'llama8b', 'llada8b'")
    parser.add_argument("--model-type", choices=["ar", "diffusion"], required=True, help="Target architecture")
    parser.add_argument("--model-id", type=str, required=False, help="HuggingFace repository ID")
    parser.add_argument("--seed", type=int, default=42, help="Global deterministic random seed")
    parser.add_argument("--max-examples", type=int, default=580, help="Number of examples to evaluate")
    parser.add_argument("--max-steps", type=int, default=20, help="Number of steps (AR tokens or Diffusion unmasking blocks)")
    
    args = parser.parse_args()

    is_diffusion = (args.model_type == "diffusion")
    
    # Use user-provided model ID if given, else fall back to the instruct defaults
    if args.model_id is None:
        model_id = LLADA_MODEL_ID if is_diffusion else LLAMA_MODEL_ID
    else:
        model_id = args.model_id

    print(f"=== Starting Task 5: Superposition Decay ({args.model_type.upper()}) ===")
    print(f"[INFO] Using Model: {model_id}")
    
    set_global_determinism(args.seed)

    # --- RECORD METADATA ---
    import time
    run_meta = {
        "task": "task5_superposition_decay",
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_id": model_id,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "max_steps": args.max_steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- LOAD DATA ---
    print(f"[INFO] Parsing dataset to isolate binary opposing targets...")
    test_instances = load_ambient_opposing_targets(args.data_path, max_examples=args.max_examples)
    print(f"[INFO] Successfully loaded {len(test_instances)} ambiguous instances.")

    if not test_instances:
        return

    # --- INITIALIZE MODEL & TRACK DECAY ---
    all_trajectories = {}
    
    if args.model_type == "ar":
        print(f"[INFO] Loading AR Model for Spatial Decay Tracking...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")
        if getattr(tokenizer, "pad_token", None) is None: 
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
            
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, cache_dir="./models")
        model.eval()
        
        for inst in tqdm(test_instances, desc="Tracking Spatial Decay"):
            traj = ar_spatial_decay_track(model, tokenizer, inst["prompt"], inst["target_a"], inst["target_b"], max_steps=args.max_steps)
            all_trajectories[inst["id"]] = traj
            
    elif args.model_type == "diffusion":
        print(f"[INFO] Loading LLaDA Model for Temporal Decay Tracking...")
        model, tokenizer = load_llada_model(hf_model=model_id, use_4bit=False)
        model.eval()
        
        for inst in tqdm(test_instances, desc="Tracking Temporal Decay"):
            traj = diffusion_temporal_decay_track(model, tokenizer, inst["prompt"], inst["target_a"], inst["target_b"], steps=args.max_steps)
            all_trajectories[inst["id"]] = traj

    # --- SAVE RESULTS WITH METADATA ---
    final_output = {
        "metadata": run_meta,
        "results": all_trajectories
    }
    
    out_path = Path(f"results/task5/{args.model_name}_decay_trajectories.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n[INFO] Task 5 Complete. Trajectories (with metadata) saved to {out_path}.")

if __name__ == "__main__":
    main()