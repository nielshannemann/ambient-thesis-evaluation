#!/usr/bin/env python3
# src/ambient/generation/task5_temporal_semantic_commitment.py
"""
Task 5: Temporal Semantic Commitment

This script generates per-instance entropy trajectories for the Task-5 analysis.

Methodological notes:
- deterministic target-pair selection instead of taking the first two
  disambiguations
- diffusion-side scoring aligned more closely with the paper T1 MC estimator
  by scoring masked continuation tokens only and rescaling by the effective
  mask ratio
- explicit metadata on ambiguity side, target labels, and selection rule

Important note:
This remains a controlled proxy analysis rather than a literal traced diffusion
sampling trajectory. The diffusion side is evaluated at fixed mask-ratio
checkpoints rather than by logging internal denoising states.

Method references:
- Methodology: Standardized Evaluation Framework
- Methodology: Study 5: Sequential Commitment Diagnostic
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import set_seed

from ambient.constants import LLADA_BASE_MODEL_ID, LLAMA_BASE_MODEL_ID
from ambient.modeling import (
    canonical_backend,
    default_base_model_id,
    forward_unpadded_logits,
    is_masked_diffusion_family,
    load_model_bundle,
    runtime_environment,
)
from ambient.paths import task5_output_path

LLADA_MODEL_ID = LLADA_BASE_MODEL_ID
LLAMA_MODEL_ID = LLAMA_BASE_MODEL_ID
LLADA_MASK_ID = 126336


def model_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    return forward_unpadded_logits(model, input_ids)


def set_global_determinism(seed: int) -> None:
    """
    Apply deterministic seeding controls across Python, NumPy, and PyTorch.

    This improves reproducibility under fixed software and hardware conditions,
    but should not be interpreted as a guarantee of bit-identical outputs across
    all environments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


def stable_text_key(text: str) -> Tuple[int, str]:
    """Stable deterministic text ordering key."""
    return (len(text or ""), text or "")


def approx_token_count(text: str) -> int:
    """Deterministic token-count proxy for control matching before model loading."""
    return len((text or "").split())


def get_ambiguity_side(ex: dict) -> Optional[str]:
    """Preserve the original Task-5 side priority: premise before hypothesis."""
    if ex.get("premise_ambiguous"):
        return "premise"
    if ex.get("hypothesis_ambiguous"):
        return "hypothesis"
    return None


def choose_target_pair(disambiguations: List[Dict[str, Any]], side: str) -> Optional[Dict[str, Any]]:
    """
    Select two target readings deterministically.

    Preference order:
    1. one entailment + one non-entailment
       (contradiction preferred over neutral if both exist)
    2. otherwise, first two distinct texts after stable sorting by (label, text)
    """
    valid: List[Dict[str, Any]] = []
    for d in disambiguations:
        txt = (d.get(side) or "").strip()
        if not txt:
            continue
        valid.append(
            {
                "text": txt,
                "label": (d.get("label") or "unknown").lower(),
                "raw": d,
            }
        )

    if len(valid) < 2:
        return None

    entailments = [v for v in valid if v["label"] == "entailment"]
    contradictions = [v for v in valid if v["label"] == "contradiction"]
    neutrals = [v for v in valid if v["label"] == "neutral"]
    others = [v for v in valid if v["label"] not in {"entailment", "contradiction", "neutral"}]

    if entailments and (contradictions or neutrals or others):
        a = sorted(entailments, key=lambda x: (x["label"],) + stable_text_key(x["text"]))[0]
        competing_pool = contradictions or neutrals or others
        b = sorted(competing_pool, key=lambda x: (x["label"],) + stable_text_key(x["text"]))[0]
        return {
            "target_a": a["text"],
            "target_b": b["text"],
            "label_a": a["label"],
            "label_b": b["label"],
            "selection_rule": "entailment_vs_nonentailment",
        }

    # Fallback: deterministic distinct pair
    dedup: Dict[str, Dict[str, Any]] = {}
    for v in valid:
        dedup.setdefault(v["text"], v)

    ordered = sorted(dedup.values(), key=lambda x: (x["label"],) + stable_text_key(x["text"]))
    if len(ordered) < 2:
        return None

    return {
        "target_a": ordered[0]["text"],
        "target_b": ordered[1]["text"],
        "label_a": ordered[0]["label"],
        "label_b": ordered[1]["label"],
        "selection_rule": "deterministic_sorted_pair",
    }


def choose_disambiguation_closest_to_length(
    disambiguations: List[Dict[str, Any]],
    side: str,
    target_length: int,
) -> Optional[Dict[str, Any]]:
    """Choose the same-side gold rewrite closest in length to a control target."""
    candidates: List[Dict[str, Any]] = []
    for disambiguation in disambiguations:
        text = (disambiguation.get(side) or "").strip()
        if not text:
            continue
        candidates.append(
            {
                "text": text,
                "label": (disambiguation.get("label") or "unknown").lower(),
                "length_tokens": approx_token_count(text),
            }
        )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda row: (abs(row["length_tokens"] - target_length), row["length_tokens"], row["label"], row["text"]),
    )


def build_task5_instances(
    examples: List[Dict[str, Any]],
    condition: str = "gold_disambiguation",
    max_examples: int = 50,
) -> List[Dict[str, Any]]:
    """Build Task-5 target pairs for the gold or control conditions."""
    random_pool: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if condition == "random_matched_rewrite":
        pool_builder: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for ex in examples:
            side = get_ambiguity_side(ex)
            if side is None:
                continue
            for disambiguation in ex.get("disambiguations", []):
                text = (disambiguation.get(side) or "").strip()
                label = (disambiguation.get("label") or "unknown").lower()
                if not text:
                    continue
                pool_builder.setdefault((side, label), []).append(
                    {
                        "source_instance_id": str(ex.get("id")),
                        "text": text,
                        "label": label,
                        "length_tokens": approx_token_count(text),
                    }
                )

        for key, rows in pool_builder.items():
            random_pool[key] = sorted(
                rows,
                key=lambda row: (row["length_tokens"], row["source_instance_id"], row["text"]),
            )

    data: List[Dict[str, Any]] = []
    for ex in examples:
        disambiguations = ex.get("disambiguations", [])
        if len(disambiguations) < 2:
            continue

        side = get_ambiguity_side(ex)
        if side is None:
            continue

        prompt_text = ex.get(side, "")
        if not prompt_text:
            continue

        instance_id = str(ex.get("id"))
        if condition == "gold_disambiguation":
            pair = choose_target_pair(disambiguations, side)
            if pair is None:
                continue
            data.append(
                {
                    "id": ex.get("id"),
                    "prompt": prompt_text,
                    "ambiguity_side": side,
                    "condition": condition,
                    "target_b_source": "gold_disambiguation",
                    "target_b_source_instance_id": instance_id,
                    "length_gap_tokens": abs(approx_token_count(pair["target_a"]) - approx_token_count(pair["target_b"])),
                    **pair,
                }
            )
        elif condition == "distractor_rewrite":
            distractor_text = (ex.get(f"distractor_{side}") or "").strip()
            if not distractor_text:
                continue
            target_b_length = approx_token_count(distractor_text)
            target_a = choose_disambiguation_closest_to_length(disambiguations, side, target_b_length)
            if target_a is None:
                continue
            data.append(
                {
                    "id": ex.get("id"),
                    "prompt": prompt_text,
                    "ambiguity_side": side,
                    "target_a": target_a["text"],
                    "target_b": distractor_text,
                    "label_a": target_a["label"],
                    "label_b": "control_distractor",
                    "selection_rule": "closest_gold_to_distractor_length",
                    "condition": condition,
                    "target_b_source": "distractor_rewrite",
                    "target_b_source_instance_id": instance_id,
                    "length_gap_tokens": abs(target_a["length_tokens"] - target_b_length),
                }
            )
        elif condition == "random_matched_rewrite":
            pair = choose_target_pair(disambiguations, side)
            if pair is None:
                continue
            target_a_length = approx_token_count(pair["target_a"])
            pool = [
                candidate
                for candidate in random_pool.get((side, pair["label_a"]), [])
                if candidate["source_instance_id"] != instance_id
            ]
            if not pool:
                continue

            chosen = min(
                pool,
                key=lambda candidate: (
                    abs(candidate["length_tokens"] - target_a_length),
                    candidate["length_tokens"],
                    candidate["source_instance_id"],
                    candidate["text"],
                ),
            )
            data.append(
                {
                    "id": ex.get("id"),
                    "prompt": prompt_text,
                    "ambiguity_side": side,
                    "target_a": pair["target_a"],
                    "target_b": chosen["text"],
                    "label_a": pair["label_a"],
                    "label_b": chosen["label"],
                    "selection_rule": f"{pair['selection_rule']}+random_matched_rewrite",
                    "condition": condition,
                    "target_b_source": "random_matched_rewrite",
                    "target_b_source_instance_id": chosen["source_instance_id"],
                    "length_gap_tokens": abs(target_a_length - chosen["length_tokens"]),
                }
            )
        else:
            raise ValueError(f"Unsupported Task-5 condition: {condition}")

        if len(data) >= max_examples:
            break

    return data


def load_ambient_opposing_targets(
    path: Path,
    max_examples: int = 50,
    condition: str = "gold_disambiguation",
) -> List[Dict[str, Any]]:
    """Load ambiguous AMBIENT instances and construct Task-5 target pairs."""
    if not path.exists():
        print(f"[ERROR] Dataset not found at {path}")
        return []

    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))

    return build_task5_instances(
        examples,
        condition=condition,
        max_examples=max_examples,
    )


def calculate_normalized_entropy(nll_a: float, nll_b: float) -> Tuple[float, float, float]:
    """
    Convert two sequence scores into a binary distribution and Shannon entropy.
    """
    logits = torch.tensor([-nll_a, -nll_b], dtype=torch.float32)
    probs = F.softmax(logits, dim=0).cpu().numpy()

    p_a, p_b = float(probs[0]), float(probs[1])

    entropy = 0.0
    if p_a > 0:
        entropy -= p_a * np.log2(p_a)
    if p_b > 0:
        entropy -= p_b * np.log2(p_b)

    return float(entropy), p_a, p_b


def compute_exact_ar_nll(model, tokenizer, prefix_str: str, target_str: str) -> float:
    """
    Compute exact AR continuation NLL for one target reading given the current prefix.
    """
    full_str = prefix_str + " " + target_str
    prefix_ids = tokenizer(prefix_str, return_tensors="pt").input_ids[0].to(model.device)
    full_ids = tokenizer(full_str, return_tensors="pt").input_ids[0].to(model.device)

    prefix_len = len(prefix_ids)

    with torch.no_grad():
        logits = model_logits(model, full_ids.unsqueeze(0))[0]

    shift_logits = logits[prefix_len - 1 : -1, :].contiguous()
    shift_labels = full_ids[prefix_len:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    return float(loss_fct(shift_logits, shift_labels).item())


def ar_spatial_decay_track(
    model,
    tokenizer,
    prompt: str,
    target_a: str,
    target_b: str,
    max_steps: int = 15,
):
    """
    Track AR commitment over greedy token-prefix growth.
    """
    trajectory = []

    encodings = tokenizer(prompt, return_tensors="pt").to(model.device)
    prefix_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    for step in range(max_steps + 1):
        prefix_str = tokenizer.decode(prefix_ids[0], skip_special_tokens=True)

        nll_a = compute_exact_ar_nll(model, tokenizer, prefix_str, target_a)
        nll_b = compute_exact_ar_nll(model, tokenizer, prefix_str, target_b)
        entropy, p_a, p_b = calculate_normalized_entropy(nll_a, nll_b)

        trajectory.append(
            {
                "step": step,
                "prefix_text": prefix_str,
                "nll_a": nll_a,
                "nll_b": nll_b,
                "prob_a": p_a,
                "prob_b": p_b,
                "entropy": entropy,
            }
        )

        if step < max_steps:
            with torch.no_grad():
                out = model.generate(
                    input_ids=prefix_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prefix_ids = out
            attention_mask = torch.ones_like(prefix_ids)

    return trajectory


def _make_local_rng(device: torch.device, seed: int, prompt: str, target: str, ratio: float) -> torch.Generator:
    """
    Build a deterministic local RNG for one diffusion checkpoint.
    """
    text_hash = int(
        hashlib.md5(f"{prompt}|||{target}|||{ratio:.6f}".encode("utf-8")).hexdigest(),
        16,
    )
    local_seed = (seed + text_hash) % (2**31)
    rng = torch.Generator(device=device)
    rng.manual_seed(local_seed)
    return rng


def compute_diffusion_mc_nll_at_ratio(
    model,
    tokenizer,
    prompt: str,
    target: str,
    mask_ratio: float,
    num_samples: int = 8,
    seed: int = 42,
    cfg_scale: float = 0.0,
) -> float:
    """
    Compute a fixed-ratio masked-token Monte Carlo plausibility proxy.

    Alignment with paper T1:
    - prompt tokens are preserved
    - exactly a fixed proportion of continuation tokens is masked per sample
    - loss is computed only on masked tokens and rescaled by 1 / effective_ratio
    """
    device = next(model.parameters()).device
    mask_id = getattr(
        model.config,
        "mask_token_id",
        getattr(tokenizer, "mask_token_id", LLADA_MASK_ID),
    )

    p_ids = torch.tensor(tokenizer(prompt, add_special_tokens=True)["input_ids"], device=device)
    c_ids = torch.tensor(tokenizer(target, add_special_tokens=False)["input_ids"], device=device)

    if c_ids.numel() == 0:
        return 0.0

    seq = torch.cat([p_ids, c_ids])
    prompt_len = len(p_ids)
    seq_len = len(seq)
    target_len = seq_len - prompt_len

    if target_len <= 0:
        return 0.0

    ratio = float(max(0.0, min(1.0, mask_ratio)))
    if ratio == 0.0:
        # Avoid an undefined 1/0 endpoint by using a minimal effective mask ratio.
        ratio = 1.0 / float(target_len)

    num_to_mask = max(1, int(round(ratio * target_len)))
    effective_ratio = num_to_mask / float(target_len)

    rng = _make_local_rng(device, seed, prompt, target, ratio)
    losses: List[float] = []

    for _ in range(num_samples):
        seq_batch = seq.unsqueeze(0)
        perturbed_seq = seq_batch.clone()
        mask_index = torch.zeros_like(seq_batch, dtype=torch.bool)

        target_positions = (
            torch.randperm(target_len, generator=rng, device=device)[:num_to_mask]
            + prompt_len
        )
        perturbed_seq[0, target_positions] = mask_id
        mask_index[0, target_positions] = True

        if cfg_scale > 0.0:
            un_batch = perturbed_seq.clone()
            un_batch[0, :prompt_len] = mask_id
            model_in = torch.cat([perturbed_seq, un_batch], dim=0)

            logits = model_logits(model, model_in)
            logits_cond, logits_uncond = torch.chunk(logits, 2, dim=0)
            logits = logits_uncond + (cfg_scale + 1.0) * (logits_cond - logits_uncond)
        else:
            logits = model_logits(model, perturbed_seq)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            seq_batch.view(-1),
            reduction="none",
        )
        ce_loss = ce_loss.view(1, -1)

        weighted = ce_loss * mask_index.float() / effective_ratio
        weighted = weighted.masked_fill(~mask_index, 0.0)
        losses.append(float(weighted.sum().item()))

    return float(np.mean(losses)) if losses else 0.0


def diffusion_temporal_decay_track(
    model,
    tokenizer,
    prompt: str,
    target_a: str,
    target_b: str,
    steps: int = 15,
    num_samples: int = 8,
    seed: int = 42,
    cfg_scale: float = 0.0,
):
    """
    Track diffusion-side commitment over a normalized fixed-ratio unmasking axis.
    """
    trajectory = []

    for step in range(steps + 1):
        mask_ratio = 1.0 - (step / steps)

        nll_a = compute_diffusion_mc_nll_at_ratio(
            model,
            tokenizer,
            prompt,
            target_a,
            mask_ratio,
            num_samples=num_samples,
            seed=seed,
            cfg_scale=cfg_scale,
        )
        nll_b = compute_diffusion_mc_nll_at_ratio(
            model,
            tokenizer,
            prompt,
            target_b,
            mask_ratio,
            num_samples=num_samples,
            seed=seed,
            cfg_scale=cfg_scale,
        )

        entropy, p_a, p_b = calculate_normalized_entropy(nll_a, nll_b)

        trajectory.append(
            {
                "step": step,
                "mask_ratio": mask_ratio,
                "nll_a": nll_a,
                "nll_b": nll_b,
                "prob_a": p_a,
                "prob_b": p_b,
                "entropy": entropy,
            }
        )

    return trajectory


def run(args) -> int:
    requested_condition = args.condition
    condition = {
        "distractor_reading": "distractor_rewrite",
        "random_matched_reading": "random_matched_rewrite",
    }.get(requested_condition, requested_condition)

    backend = canonical_backend(args.model_family)
    is_diffusion = is_masked_diffusion_family(args.model_family)
    model_id = args.model_id or default_base_model_id(args.model_family)

    print(f"=== Starting Task 5: {args.model_family.upper()} ===")
    print(f"[INFO] Using model: {model_id}")
    print(f"[INFO] Condition: {requested_condition}")

    set_global_determinism(args.seed)

    run_meta = {
        "task": "task5_temporal_semantic_commitment",
        "model_name": args.model_name,
        "model_type": "diffusion" if is_diffusion else "ar",
        "model_family": args.model_family,
        "backend": backend,
        "model_id": model_id,
        "runtime_environment": runtime_environment(),
        "seed": args.seed,
        "max_examples": args.max_examples,
        "max_steps": args.max_steps,
        "mc_num": args.mc_num,
        "cfg_scale": args.cfg_scale,
        "condition": requested_condition,
        "internal_condition": condition,
        "target_selection": (
            "entailment_vs_nonentailment_else_deterministic_sorted_pair"
            if condition == "gold_disambiguation"
            else requested_condition
        ),
        "diffusion_scoring": "fixed_ratio_masked_token_mc_proxy_aligned_to_task1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    test_instances = load_ambient_opposing_targets(
        args.data_path,
        max_examples=args.max_examples,
        condition=condition,
    )
    print(f"[INFO] Loaded {len(test_instances)} instances for the selected Task-5 condition.")
    if not test_instances:
        return 1

    all_trajectories: Dict[str, Any] = {}
    bundle = load_model_bundle(
        args.model_family,
        model_id=model_id,
        use_4bit=args.use_4bit,
        verbose=True,
    )
    model, tokenizer = bundle.model, bundle.tokenizer

    if not is_diffusion:
        for inst in tqdm(test_instances, desc="Tracking AR commitment"):
            traj = ar_spatial_decay_track(
                model,
                tokenizer,
                inst["prompt"],
                inst["target_a"],
                inst["target_b"],
                max_steps=args.max_steps,
            )
            all_trajectories[str(inst["id"])] = {**inst, "trajectory": traj}

    else:
        for inst in tqdm(test_instances, desc="Tracking diffusion commitment"):
            traj = diffusion_temporal_decay_track(
                model,
                tokenizer,
                inst["prompt"],
                inst["target_a"],
                inst["target_b"],
                steps=args.max_steps,
                num_samples=args.mc_num,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
            )
            all_trajectories[str(inst["id"])] = {**inst, "trajectory": traj}

    out = {"metadata": run_meta, "results": all_trajectories}

    out_path = args.output_path or task5_output_path(args.model_name, condition=requested_condition)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved Task-5 trajectories to {out_path}")
    return 0
