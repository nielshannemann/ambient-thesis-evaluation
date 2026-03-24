#!/usr/bin/env python3
# src/ambient/evaluation/task4_linear_probing.py
"""
=============================================================================
TASK 4: INTERNAL REPRESENTATION PROBING (LINEAR PROBING)
=============================================================================
This script investigates the internal, latent representations of the base 
models (AR vs. Diffusion) to determine if NLI entailment states are linearly 
separable within their hidden layers.

Key Methodological Innovation:
- AR Models: Extracts the hidden state of the final causal token.
- Diffusion Models (LLaDA): Artificially appends a [MASK] token to act as a 
  bidirectional "semantic sink" and extracts its corresponding hidden state, 
  ensuring structural parity between causal and non-causal architectures.
- Stratified Group K-Fold: Prevents data leakage by ensuring that all NLI 
  pairs derived from the same ambiguous premise stay strictly within the 
  same validation fold.

[Thesis Ref: Section X.X - Linear Probing of Internal Representations]
=============================================================================
"""

import json
import torch
import random
import warnings
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# Custom loader for the LLaDA architecture
from ambient.llada_loader import load_llada_model

warnings.filterwarnings("ignore")

# The official Token ID for [MASK] in the LLaDA vocabulary
LLADA_MASK_ID = 126336 


def set_all_seeds(seed: int):
    """Enforces strict reproducibility across all compute backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


def load_nli_pairs(path: Path, max_examples: int = 600) -> Tuple[List[str], List[str], List[str]]:
    """
    Parses the AMBIENT dataset to extract binary NLI pairs.
    Crucially, it also returns a 'groups' list (the premise ID) to prevent 
    data leakage across Cross-Validation folds.
    """
    texts, labels, groups = [], [], []
    if not path.exists():
        print(f"[error] Dataset not found at {path}")
        return texts, labels, groups

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= max_examples * 2: 
                break
            ex = json.loads(line)
            amb_sentence = ex.get("premise", ex.get("ambiguous_sentence", ""))
            instance_id = ex.get("id", amb_sentence)  # Fallback to the sentence itself as group ID
            
            for disambig in ex.get("disambiguations", []):
                hyp = disambig.get("hypothesis", "")
                label = disambig.get("label", "")
                
                # Restrict to strictly opposing classes for binary probing
                if label in ["entailment", "contradiction"]:
                    prompt = f"Premise: {amb_sentence}\nHypothesis: {hyp}\nQuestion: Does the premise prove the hypothesis? Answer:"
                    texts.append(prompt)
                    labels.append(label)
                    groups.append(instance_id) # Assign to group
                    
    return texts, labels, groups


def extract_hidden_states(model_id: str, texts: List[str], is_llada: bool, batch_size: int, use_4bit: bool) -> np.ndarray:
    """
    Extracts the contextualized hidden states from the middle layer of the network.
    Implements architecture-specific extraction logic (Causal Last-Token vs. Diffusion MASK-Sink).
    """
    print(f"\n[info] Initializing Feature Extraction Pipeline for: {model_id}")
    
    if is_llada:
        model, tokenizer = load_llada_model(hf_model=model_id, use_4bit=use_4bit, verbose=False)
    else:
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", cache_dir="./models"
        )
        
    model.eval()
    
    # Crucial: Right-padding ensures index math for extraction remains correct
    tokenizer.padding_side = "right"
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            current_batch_size = input_ids.shape[0]
            # Calculate the true length of each sequence excluding padding tokens
            seq_lengths = attention_mask.sum(dim=1).long() 
            
            if is_llada:
                new_input_ids = torch.full((current_batch_size, input_ids.shape[1] + 1), tokenizer.pad_token_id, device=model.device)
                new_attention_mask = torch.zeros((current_batch_size, input_ids.shape[1] + 1), dtype=attention_mask.dtype, device=model.device)
                
                for j in range(current_batch_size):
                    length = seq_lengths[j]
                    new_input_ids[j, :length] = input_ids[j, :length]
                    new_input_ids[j, length] = LLADA_MASK_ID  # Append [MASK]
                    new_attention_mask[j, :length+1] = 1
                    
                outputs = model(input_ids=new_input_ids, attention_mask=new_attention_mask, output_hidden_states=True)
                extraction_indices = seq_lengths 
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                extraction_indices = seq_lengths - 1
            
            # Select the middle layer (empirically holds the richest semantic features)
            middle_layer_idx = len(outputs.hidden_states) // 2
            target_hidden_states = outputs.hidden_states[middle_layer_idx]
            
            embeddings = target_hidden_states[torch.arange(current_batch_size), extraction_indices].cpu().numpy()
            all_embeddings.extend(embeddings)
            
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return np.array(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Task 4: Linear Probing of Internal Representations")
    parser.add_argument("--llama-model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="HuggingFace ID for the AR model")
    parser.add_argument("--llada-model", type=str, default="GSAI-ML/LLaDA-8B-Base", help="HuggingFace ID for the Diffusion model")
    parser.add_argument("--data-path", type=Path, default=Path("data/test_baked.jsonl"), help="Path to the AMBIENT dataset")
    parser.add_argument("--max-examples", type=int, default=580, help="Maximum number of ambiguous premises to process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for hidden state extraction")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--use-4bit", action="store_true", help="Enable 4-bit quantization (NF4) for model loading")
    
    args = parser.parse_args()

    print(f"=== Starting Task 4: Internal Representation Probing ===")
    print(f"[info] Global Seed: {args.seed}")
    print(f"[info] Batch Size: {args.batch_size}")
    print(f"[info] 4-bit Quantization: {args.use_4bit}")
    
    set_all_seeds(args.seed)
    
    print("\n[info] Parsing NLI dataset...")
    texts, labels, groups = load_nli_pairs(args.data_path, max_examples=args.max_examples)
    print(f"[info] Extracted {len(texts)} binary NLI pairs across {len(set(groups))} unique premises.")
    print(f"[info] Label Distribution: {dict(Counter(labels))}")
    
    if len(texts) == 0:
        print("[error] No valid NLI pairs found. Exiting.")
        return

    # 1. Feature Extraction
    llama_embeddings = extract_hidden_states(args.llama_model, texts, is_llada=False, batch_size=args.batch_size, use_4bit=args.use_4bit)
    llada_embeddings = extract_hidden_states(args.llada_model, texts, is_llada=True, batch_size=args.batch_size, use_4bit=args.use_4bit)
    
    # 2. Linear Probing Setup
    print("\n" + "="*60)
    print(f"=== LINEAR PROBING RESULTS (5-Fold Stratified-Group CV, Seed: {args.seed}) ===")
    
    clf = make_pipeline(
        StandardScaler(), 
        LogisticRegression(max_iter=2000, random_state=args.seed)
    )
    
    # CRITICAL FIX: StratifiedGroupKFold prevents Data Leakage!
    cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    # 3. Evaluation
    llama_scores = cross_val_score(clf, llama_embeddings, labels, groups=groups, cv=cv_strategy)
    print(f"-> LLaMA-3.1-8B (AR) Internal Accuracy:       {np.mean(llama_scores)*100:.2f}% (± {np.std(llama_scores)*100:.2f}%)")
    
    llada_scores = cross_val_score(clf, llada_embeddings, labels, groups=groups, cv=cv_strategy)
    print(f"-> LLaDA-8B (Diffusion) Internal Accuracy:    {np.mean(llada_scores)*100:.2f}% (± {np.std(llada_scores)*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()