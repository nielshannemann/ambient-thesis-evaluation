#!/usr/bin/env python3
# src/ambient/generation/task3_silhouette_generate.py
"""
=============================================================================
TASK 3: GENERATIVE SEMANTIC CLUSTERING (PHASE 1 - SAMPLING)
=============================================================================
This script unconditionally samples N=100 continuations from the base models 
to explore the latent semantic distribution of ambiguous inputs.

Methodological Integration:
This script utilizes the unified `ARAdapter` and `LLaDaAdapter` to ensure 
strict architectural consistency. It implements chunk-based micro-batching to 
prevent VRAM exhaustion on consumer hardware while guaranteeing exact 
cryptographic reproducibility via chunk-level deterministic seeding.

[Method Reference: Section 3.4.1 - Unconstrained Continuation Sampling]
=============================================================================
"""

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig

# Custom AmbiEnt modules
from ambient.llada_loader import load_llada_model
from ambient.adapters import ARAdapter, LLaDaAdapter
from ambient.constants import LLADA_BASE_MODEL_ID, LLAMA_BASE_MODEL_ID
from ambient.paths import task3_output_path


def auto_detect_4bit(hf_model: str) -> bool:
    """
    Dynamically determines whether 4-bit quantization (NF4) is required 
    based on the available GPU memory (VRAM) and the model scale.
    """
    if not torch.cuda.is_available():
        return False

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    hf_lower = hf_model.lower()

    # Heuristics based on empirical VRAM consumption for LLM inference
    if "70b" in hf_lower or "72b" in hf_lower:
        return vram_gb < 130
    if "8b" in hf_lower or "7b" in hf_lower or "9b" in hf_lower:
        return vram_gb < 20

    return vram_gb < 16


def load_ambiguous_examples(path: Path, max_examples: int = 600) -> list:
    """
    Parses the AMBIENT dataset and isolates instances containing explicit 
    semantic multiplicity (ambiguity). Dynamically identifies whether the 
    premise or the hypothesis harbors the ambiguity based on dataset flags.

    Returns a list of examples augmented with:
      - ambiguity_side: "premise" or "hypothesis"
      - ambiguous_sentence: the ambiguous input string for the selected side
      - disambiguated_control: one gold control string on the same side
    """
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(data) >= max_examples:
                break

            ex = json.loads(line)
            disambiguations = ex.get("disambiguations", [])

            # Isolate instances with at least two distinct valid disambiguations.
            if len(disambiguations) < 2:
                continue

            if ex.get("premise_ambiguous"):
                side = "premise"
            elif ex.get("hypothesis_ambiguous"):
                side = "hypothesis"
            else:
                continue

            ex["ambiguity_side"] = side
            ex["ambiguous_sentence"] = ex.get(side, "")
            ex["disambiguated_control"] = disambiguations[0].get(side, "")
            data.append(ex)

    return data


def run(args) -> int:
    print("=== Starting Task 3: Generative Sampling ===")

    # 1. STRICT GLOBAL DETERMINISM
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    is_diffusion = args.model_family == "llada"
    model_id = args.model_id or (LLADA_BASE_MODEL_ID if is_diffusion else LLAMA_BASE_MODEL_ID)
    use_4bit = auto_detect_4bit(model_id)

    print(f"[INFO] Architecture: {args.model_family.upper()} | Prompt Construct: {args.prompt_type.upper()}")
    print(f"[INFO] Hardware Setting: Generating {args.num_continuations} total samples in chunks of {args.batch_size}.")
    print(f"[INFO] Hyperparameters: Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}, CFG={args.cfg_scale}, Steps={args.diffusion_steps}")

    out_path = args.output_path or task3_output_path(args.model_name, args.prompt_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "task3_generative_clustering",
        "model_name": args.model_name,
        "model_type": args.model_family,
        "model_id": model_id,
        "prompt_type": args.prompt_type,
        "hyperparameters": {
            "num_continuations": args.num_continuations,
            "batch_size": args.batch_size,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }

    if is_diffusion:
        run_meta["hyperparameters"]["cfg_scale"] = args.cfg_scale
        run_meta["hyperparameters"]["diffusion_steps"] = args.diffusion_steps

    dataset = load_ambiguous_examples(args.data_path, max_examples=args.max_examples)
    print(f"[INFO] Successfully isolated {len(dataset)} ambiguous instances for evaluation.")

    # --- ARCHITECTURE INITIALIZATION & ADAPTER INJECTION ---
    if is_diffusion:
        model, tokenizer = load_llada_model(hf_model=model_id, use_4bit=use_4bit, verbose=False)
        adapter = LLaDaAdapter(model_name=model_id, model=model, tokenizer=tokenizer, diff_mc_nll=None)
    else:
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "cache_dir": "./models"}
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        adapter = ARAdapter(model_name=model_id, model=model, tokenizer=tokenizer, ar_score_fn=None)

    print("[INFO] Commencing unconstrained latent sampling...")
    all_results = []

    for prompt_idx, row in enumerate(tqdm(dataset, desc="Processing Inputs")):
        row_id = row.get("id") or row.get("_instance_id", "unknown")
        ambiguity_side = row.get("ambiguity_side")

        if args.prompt_type == "ambiguous":
            prompt_source = row.get("ambiguous_sentence", "")
        else:
            prompt_source = row.get("disambiguated_control", "")

        prompt = f'{prompt_source} '
        #prompt = f'{prompt_source} "'
        current_seed = args.seed + (prompt_idx * 10000)

        raw_continuations = adapter.generate(
            prompt=prompt,
            num_return_sequences=args.num_continuations,
            batch_size=args.batch_size,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            max_new_tokens=32,
            stop_at_sentence=True,
            seed=current_seed,
            steps=args.diffusion_steps,
        )

        continuations = [c.replace('"', '').replace('”', '').replace('“', '').strip() for c in raw_continuations]

        all_results.append({
            "id": row_id,
            "ambiguity_side": ambiguity_side,
            "prompt_type": args.prompt_type,
            "ambiguous_sentence": row.get("ambiguous_sentence", ""),
            "disambiguated_control": row.get("disambiguated_control", ""),
            "prompt_text": prompt_source,
            "gold_disambiguations": row.get("disambiguations", []),
            "continuations": continuations,
        })

    final_output = {"metadata": run_meta, "results": all_results}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Task 3 Sampling complete. Results serialized to: {out_path}")
    return 0
