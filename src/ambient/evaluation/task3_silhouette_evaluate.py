#!/usr/bin/env python3
# src/ambient/evaluation/task3_silhouette_evaluate.py
"""
=============================================================================
TASK 3: GENERATIVE SEMANTIC CLUSTERING (PHASE 2 - EVALUATION)
=============================================================================
This script evaluates the unconstrained semantic continuations generated in 
Phase 1. It calculates four core metrics to quantify the latent ambiguity 
retention of the tested architectures:

1. Mean Cosine Distance (MCD): Measures raw intra-cluster dispersion.
2. Silhouette Score: Measures the density and separation of semantic clusters.
3. Target Coverage (Cosine): Embeddings-based minority intent preservation.
4. Target Coverage (NLI): Strict entailment-based minority intent preservation.

Methodological Integration:
Extracts the exact random seed from the generation metadata to guarantee 
100% deterministic k-means clustering and evaluation parity.

[Thesis Reference: Section 3.4.2 - Semantic Clustering Evaluation]
=============================================================================
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os

# Set HF_HOME dynamically unless already set in the terminal environment
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/mnt/storage2/student_data/nhannemann/hf_cache"

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from transformers import pipeline, set_seed

def set_global_determinism(seed: int):
    """
    Locks all random number generators and backend heuristics to 
    guarantee 100% exact reproducibility based on the metadata seed.
    
    Args:
        seed (int): The deterministic seed extracted from the generation phase.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Forces CuDNN to use deterministic algorithms (prevents floating-point drift)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Task 3: Deterministic Semantic Clustering Evaluation")
    parser.add_argument("--results-path", type=Path, required=True, help="Path to the JSON file generated in Phase 1.")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model ID for semantic projection.")
    parser.add_argument("--nli-model", type=str, default="roberta-large-mnli", help="NLI model ID for strict entailment coverage.")
    args = parser.parse_args()

    if not args.results_path.exists():
        print(f"[ERROR] Target file not found: {args.results_path}")
        return

    # --- DYNAMIC SEED EXTRACTION ---
    with open(args.results_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        
    metadata = full_data.get("metadata", {})
    results_list = full_data.get("results", [])
    
    extracted_seed = metadata.get("hyperparameters", {}).get("seed", 42)
    print(f"[INFO] Extracted deterministic seed {extracted_seed} from metadata. Locking environment...")
    
    # 1. Enforce absolute determinism across the entire script
    set_global_determinism(extracted_seed)

    print(f"[INFO] Loading Semantic Embedding Model: '{args.embed_model}'...")
    embedder = SentenceTransformer(args.embed_model)
    
    print(f"[INFO] Loading Natural Language Inference (NLI) Model: '{args.nli_model}' (This may take a moment)...")
    device = 0 if torch.cuda.is_available() else -1
    nli_pipe = pipeline("text-classification", model=args.nli_model, device=device)
    
    mcd_scores = []
    silhouette_scores = []
    cos_coverage_scores = []
    nli_coverage_scores = []
    
    valid_examples = 0
    
    print(f"\n[INFO] Commencing deterministic evaluation pipeline...\n")
    
    for data in results_list:
        continuations = data.get("continuations", [])
        gold_data = data.get("gold_disambiguations", [])
        
        gold_disambigs = [d["premise"] for d in gold_data if "premise" in d]
        
        # Require at least two continuations and two distinct gold meanings for clustering
        if len(continuations) < 2 or len(gold_disambigs) < 2:
            continue
            
        k = len(gold_disambigs)
        
        # ==========================================
        # 1 & 2: Mean Cosine Distance (MCD) & Silhouette Score
        # ==========================================
        cont_embeddings = embedder.encode(continuations, convert_to_numpy=True, normalize_embeddings=True)
        gold_embeddings = embedder.encode(gold_disambigs, convert_to_numpy=True, normalize_embeddings=True)
        
        dist_matrix = cosine_distances(cont_embeddings)
        iu1 = np.triu_indices(len(continuations), k=1)
        mcd = np.mean(dist_matrix[iu1]) if len(iu1[0]) > 0 else 0.0
        mcd_scores.append(mcd)
        
        if len(continuations) > k and mcd > 1e-5:
            # K-Means receives the strict random seed to guarantee cluster stability across runs
            kmeans = KMeans(n_clusters=k, random_state=extracted_seed, n_init=10)
            labels = kmeans.fit_predict(cont_embeddings)
            try:
                sil = silhouette_score(cont_embeddings, labels, metric='cosine')
                silhouette_scores.append(sil)
            except ValueError:
                pass
        
        # ==========================================
        # 3: TARGET COVERAGE (Cosine Similarity Proxy)
        # ==========================================
        sims = cosine_similarity(cont_embeddings, gold_embeddings)
        closest_gold_idx = np.argmax(sims, axis=1)
        cos_percentages = [np.sum(closest_gold_idx == i) / len(continuations) for i in range(k)]
        # Record the coverage of the least represented valid semantic interpretation
        cos_coverage_scores.append(min(cos_percentages))
        
        # ==========================================
        # 4: NLI TARGET COVERAGE (Strict Academic Gold Standard)
        # ==========================================
        nli_entail_counts = [0] * k
        
        # Flatten all pairs for optimal pipeline batching
        nli_pairs = [{"text": cont, "text_pair": gold} for cont in continuations for gold in gold_disambigs]
        
        # Process the entire list through the NLI pipeline simultaneously
        results = nli_pipe(nli_pairs, truncation=True, max_length=512, batch_size=16)
        
        # Map flat results back to their respective gold disambiguations
        for idx, result in enumerate(results):
            if result['label'].upper() == 'ENTAILMENT':
                # Modulo arithmetic links the flat index back to the specific k gold meaning
                gold_idx = idx % k 
                nli_entail_counts[gold_idx] += 1
                    
        nli_percentages = [count / len(continuations) for count in nli_entail_counts]
        # Record the entailment coverage of the least represented valid semantic interpretation
        nli_coverage_scores.append(min(nli_percentages))
        
        valid_examples += 1

    print("="*65)
    print(f"=== EVALUATION RESULTS FOR: {args.results_path.name} ===")
    print(f"Processed Premises: {valid_examples} (approx. {args.results_path.name.split('_')[-1].replace('.json','')} continuations each)")
    print("-" * 65)
    print(f"-> Mean Cosine Distance (MCD):         {np.mean(mcd_scores):.4f}")
    if silhouette_scores:
        print(f"-> Average Silhouette Score:           {np.mean(silhouette_scores):.4f}")
    if cos_coverage_scores:
        print(f"-> Minority Target Coverage (Cosine):  {np.mean(cos_coverage_scores)*100:.1f}%")
    if nli_coverage_scores:
        print(f"-> Minority Target Coverage (NLI):     {np.mean(nli_coverage_scores)*100:.1f}%")
    print("="*65)

if __name__ == "__main__":
    main()