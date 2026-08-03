#!/usr/bin/env python3
# src/ambient/generation/task3_silhouette_generate.py
"""
=============================================================================
TASK 3: GENERATIVE SEMANTIC CLUSTERING (PHASE 1 - SAMPLING)
=============================================================================
This script samples continuations from raw base-model prompts or from an
explicit, shared continuation instruction for instruction-tuned models. The
raw mode remains the historical default.

Methodological Integration:
This script utilizes the unified `ARAdapter` and `LLaDaAdapter` to ensure 
strict architectural consistency. It implements chunk-based micro-batching to 
prevent VRAM exhaustion on consumer hardware while guaranteeing exact 
cryptographic reproducibility via chunk-level deterministic seeding.

[Thesis: Methodology > Study 3: Reading Coverage in Free Continuations]
=============================================================================
"""

import hashlib
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import set_seed

# Custom AmbiEnt modules
from ambient.adapters import ARAdapter, DreamAdapter, LLaDaAdapter
from ambient.modeling import (
    auto_detect_4bit as shared_auto_detect_4bit,
    canonical_backend,
    default_base_model_id,
    is_masked_diffusion_family,
    load_model_bundle,
    runtime_environment,
)
from ambient.paths import task3_output_path
from ambient.utils import write_json_atomic


TASK3_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."
TASK3_CHAT_USER_TEMPLATE = (
    "Write exactly one natural sentence that continues the text below. "
    "Return only the continuation.\n\nText:\n{text}"
)


def task3_prompt_template(prompt_mode: str) -> str:
    """Return the exact human-readable prompt template recorded in metadata."""
    if prompt_mode == "raw":
        return "{text} "
    if prompt_mode == "chat_continuation":
        return TASK3_CHAT_USER_TEMPLATE
    raise ValueError(f"Unsupported Task-3 prompt mode: {prompt_mode}")


def build_task3_generation_prompt(tokenizer, prompt_source: str, prompt_mode: str) -> str:
    """Build either the historical raw prompt or a tokenizer-native chat prompt."""
    if prompt_mode == "raw":
        return f"{prompt_source} "
    if prompt_mode != "chat_continuation":
        raise ValueError(f"Unsupported Task-3 prompt mode: {prompt_mode}")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("chat_continuation requires a tokenizer with apply_chat_template().")

    messages = [
        {"role": "system", "content": TASK3_CHAT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TASK3_CHAT_USER_TEMPLATE.format(text=prompt_source.strip()),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_task3_continuation(text: str) -> str:
    """Remove prompt-wrapper quote marks from Study 3 continuations."""
    return text.replace('"', "").replace("\u201d", "").replace("\u201c", "").strip()


def example_id(item: dict, fallback: str = "unknown") -> str:
    """Read an ID without treating the valid integer ID 0 as missing."""
    value = item.get("id")
    if value is None:
        value = item.get("_instance_id")
    return str(fallback if value is None else value)


def auto_detect_4bit(hf_model: str) -> bool:
    """
    Dynamically determines whether 4-bit quantization (NF4) is required 
    based on the available GPU memory (VRAM) and the model scale.
    """
    return shared_auto_detect_4bit(hf_model)


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


def select_examples(
    dataset: list,
    id_file: Path | None,
    sample_size: int | None,
    selection_seed: int,
) -> list:
    """Select a reproducible cross-model subset without modifying the dataset."""
    selected = dataset
    if id_file is not None:
        raw = id_file.read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            requested_ids = {str(value) for value in json.loads(raw)}
        else:
            requested_ids = {
                line.strip()
                for line in raw.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
        selected = [
            item
            for item in selected
            if example_id(item) in requested_ids
        ]
        found_ids = {example_id(item) for item in selected}
        missing = requested_ids - found_ids
        if missing:
            raise ValueError(f"ID file contains {len(missing)} IDs absent from the selected dataset.")

    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if sample_size > len(selected):
            raise ValueError(f"sample_size={sample_size} exceeds {len(selected)} available examples")
        rng = random.Random(selection_seed)
        indices = sorted(rng.sample(range(len(selected)), sample_size))
        selected = [selected[index] for index in indices]
    return selected


def run(args) -> int:
    print("=== Starting Task 3: Generative Sampling ===")

    # 1. STRICT GLOBAL DETERMINISM
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    backend = canonical_backend(args.model_family)
    is_diffusion = is_masked_diffusion_family(args.model_family)
    model_id = args.model_id or default_base_model_id(args.model_family)
    use_4bit = auto_detect_4bit(model_id) if args.use_4bit is None else args.use_4bit
    prompt_mode = getattr(args, "prompt_mode", "raw")

    print(
        f"[INFO] Architecture: {args.model_family.upper()} | "
        f"Prompt Construct: {args.prompt_type.upper()} | Mode: {prompt_mode}"
    )
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
        "runtime_environment": runtime_environment(),
        "prompt_type": args.prompt_type,
        "prompt_mode": prompt_mode,
        "prompt_configuration": {
            "system_prompt": TASK3_CHAT_SYSTEM_PROMPT if prompt_mode == "chat_continuation" else None,
            "user_template": task3_prompt_template(prompt_mode),
            "chat_template_source": "tokenizer.apply_chat_template" if prompt_mode == "chat_continuation" else None,
        },
        "hyperparameters": {
            "num_continuations": args.num_continuations,
            "batch_size": args.batch_size,
            "max_examples": args.max_examples,
            "id_file": str(args.id_file) if args.id_file else None,
            "sample_size": args.sample_size,
            "selection_seed": args.selection_seed,
            "resume_requested": args.resume,
            "checkpoint_every": args.checkpoint_every,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }

    if is_diffusion:
        run_meta["hyperparameters"]["cfg_scale"] = args.cfg_scale
        run_meta["hyperparameters"]["diffusion_steps"] = args.diffusion_steps
        run_meta["hyperparameters"]["diffusion_alg"] = args.diffusion_alg
        run_meta["hyperparameters"]["diffusion_alg_temp"] = args.diffusion_alg_temp

    dataset = load_ambiguous_examples(args.data_path, max_examples=args.max_examples)
    dataset = select_examples(dataset, args.id_file, args.sample_size, args.selection_seed)
    selected_ids = [example_id(item) for item in dataset]
    selected_id_sha256 = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    run_meta["data_selection"] = {
        "num_selected_items": len(selected_ids),
        "selected_id_sha256": selected_id_sha256,
    }
    print(f"[INFO] Successfully isolated {len(dataset)} ambiguous instances for evaluation.")
    indexed_dataset = list(enumerate(dataset))

    if args.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    all_results = []
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        previous_meta = previous.get("metadata") or {}
        previous_prompt_mode = previous_meta.get("prompt_mode", "raw")
        expected_identity = {
            "model_id": model_id,
            "prompt_type": args.prompt_type,
            "prompt_mode": prompt_mode,
        }
        actual_identity = {
            "model_id": previous_meta.get("model_id"),
            "prompt_type": previous_meta.get("prompt_type"),
            "prompt_mode": previous_prompt_mode,
        }
        mismatches = {
            key: (actual_identity.get(key), value)
            for key, value in expected_identity.items()
            if actual_identity.get(key) not in {None, value}
        }

        previous_hyperparameters = previous_meta.get("hyperparameters") or {}
        generation_keys = {
            "seed",
            "num_continuations",
            "batch_size",
            "temperature",
            "top_p",
            "top_k",
            "cfg_scale",
            "diffusion_steps",
            "diffusion_alg",
            "diffusion_alg_temp",
        }
        for key in generation_keys:
            if key not in run_meta["hyperparameters"]:
                continue
            value = run_meta["hyperparameters"][key]
            if key in previous_hyperparameters and previous_hyperparameters[key] != value:
                mismatches[f"hyperparameters.{key}"] = (
                    previous_hyperparameters[key],
                    value,
                )

        previous_selection = previous_meta.get("data_selection") or {}
        previous_selection_hash = previous_selection.get("selected_id_sha256")
        if previous_selection_hash not in {None, selected_id_sha256}:
            mismatches["data_selection.selected_id_sha256"] = (
                previous_selection_hash,
                selected_id_sha256,
            )
        if previous_selection_hash is None:
            for key in {"id_file", "sample_size", "selection_seed", "max_examples"}:
                if key not in previous_hyperparameters:
                    continue
                value = run_meta["hyperparameters"][key]
                if previous_hyperparameters[key] != value:
                    mismatches[f"hyperparameters.{key}"] = (
                        previous_hyperparameters[key],
                        value,
                    )
        if mismatches:
            raise ValueError(
                f"Cannot resume Task-3 output with incompatible metadata: {mismatches}"
            )

        all_results = list(previous.get("results") or [])
        processed_ids = {
            str(item["id"] if item.get("id") is not None else item.get("row_id"))
            for item in all_results
        }
        unexpected_ids = processed_ids - set(selected_ids)
        if unexpected_ids:
            raise ValueError(
                f"Cannot resume Task-3 output: {len(unexpected_ids)} completed IDs "
                "are absent from the current selection."
            )
        indexed_dataset = [
            (index, item)
            for index, item in indexed_dataset
            if example_id(item) not in processed_ids
        ]
        run_meta["resume"] = {
            "existing_items": len(all_results),
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(
            f"[INFO] Resume mode: {len(all_results)} items already complete; "
            f"{len(indexed_dataset)} remain."
        )

    run_meta["status"] = "running"
    run_meta["num_preexisting_results"] = len(all_results)
    write_json_atomic(out_path, {"metadata": run_meta, "results": all_results})

    if not indexed_dataset:
        if all_results:
            run_meta["status"] = "finished"
            run_meta["num_completed_results"] = len(all_results)
            run_meta["timestamp_end"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            write_json_atomic(out_path, {"metadata": run_meta, "results": all_results})
            print(f"[INFO] No pending Task-3 items; output is complete at {out_path}")
            return 0
        raise ValueError("Task-3 selection contains no examples.")

    # --- ARCHITECTURE INITIALIZATION & ADAPTER INJECTION ---
    bundle = load_model_bundle(
        model_family=args.model_family,
        model_id=model_id,
        use_4bit=use_4bit,
        verbose=False,
    )
    model, tokenizer = bundle.model, bundle.tokenizer
    if backend == "llada":
        adapter = LLaDaAdapter(model_name=model_id, model=model, tokenizer=tokenizer, diff_mc_nll=None)
    elif backend == "dream":
        adapter = DreamAdapter(model_name=model_id, model=model, tokenizer=tokenizer, diff_mc_nll=None)
    else:
        adapter = ARAdapter(model_name=model_id, model=model, tokenizer=tokenizer, ar_score_fn=None)

    print("[INFO] Commencing unconstrained latent sampling...")

    for run_index, (prompt_idx, row) in enumerate(
        tqdm(indexed_dataset, desc="Processing Inputs"),
        start=1,
    ):
        row_id = example_id(row)
        ambiguity_side = row.get("ambiguity_side")

        if args.prompt_type == "ambiguous":
            prompt_source = row.get("ambiguous_sentence", "")
        else:
            prompt_source = row.get("disambiguated_control", "")

        prompt = build_task3_generation_prompt(tokenizer, prompt_source, prompt_mode)
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
            diffusion_alg=args.diffusion_alg,
            diffusion_alg_temp=args.diffusion_alg_temp,
            progress_every_chunks=args.progress_every_chunks,
        )

        continuations = [clean_task3_continuation(c) for c in raw_continuations]

        all_results.append({
            "id": row_id,
            "ambiguity_side": ambiguity_side,
            "prompt_type": args.prompt_type,
            "prompt_mode": prompt_mode,
            "ambiguous_sentence": row.get("ambiguous_sentence", ""),
            "disambiguated_control": row.get("disambiguated_control", ""),
            "prompt_text": prompt_source,
            "gold_disambiguations": row.get("disambiguations", []),
            "continuations": continuations,
        })

        if run_index % args.checkpoint_every == 0:
            run_meta["num_completed_results"] = len(all_results)
            write_json_atomic(out_path, {"metadata": run_meta, "results": all_results})

    run_meta["status"] = "finished"
    run_meta["num_completed_results"] = len(all_results)
    run_meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    final_output = {"metadata": run_meta, "results": all_results}
    write_json_atomic(out_path, final_output)

    print(f"\n[INFO] Task 3 Sampling complete. Results serialized to: {out_path}")
    return 0
