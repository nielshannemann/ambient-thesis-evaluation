#!/usr/bin/env python3
# src/ambient/adapters.py
"""
=============================================================================
AMBIENT MODEL ADAPTERS
=============================================================================
This module defines the Adapter Framework, standardizing the interaction 
between the evaluation loop and vastly different model architectures 
(Autoregressive vs. Discrete Text Diffusion).

Restored Features from Original Codebase:
- Attention-Mask Fallbacks for robust AR generation.
- NFKC Unicode Normalization and stray-quote removal for pristine outputs.
- Batched parallel decoding for Diffusion efficiency.
- Chunk-based micro-batching to prevent OOM errors on consumer GPUs 
  while preserving strict cryptographic reproducibility.

[Thesis Reference: Section 3.1.2 - Standardizing the Interface: Adapter Design]
=============================================================================
"""

import re
import inspect
import unicodedata
from typing import Dict, Optional, List, Callable
import torch

# Custom AmbiEnt modules
from ambient.utils import clean_continuation_text
from ambient.llada_loader import run_llada_prompt


class BaseAdapter:
    """
    Abstract base class for all generative model adapters.
    Ensures a standardized API for both sequence generation and 
    Negative Log-Likelihood (NLL) scoring across divergent architectures.
    """
    def generate(self, prompt: str, num_return_sequences: int = 1, batch_size: int = 25, top_p: float = 1.0, max_new_tokens: int = 60, **kwargs) -> List[str]:
        raise NotImplementedError("Each adapter must implement its own generate method.")

    def score_continuations(self, prompts: List[str], continuations: List[str], **kwargs) -> List[Optional[float]]:
        raise NotImplementedError("Each adapter must implement its own scoring method (e.g., Exact NLL or MC NLL).")


# Registry pattern for dynamic model loading in evaluation scripts
_REGISTRY: Dict[str, BaseAdapter] = {}

def register_adapter(name: str, adapter: BaseAdapter):
    _REGISTRY[name] = adapter

def get_adapter(name: str) -> Optional[BaseAdapter]:
    return _REGISTRY.get(name, None)

def _filter_kwargs_for_call(fn: Callable, kwargs: dict) -> dict:
    """Helper to safely filter kwargs to match a specific generation function's exact signature."""
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _post_process_generation(g: str, stop_at_sentence: bool) -> str:
    """
    Applies strict text normalization and formatting rules.
    [Thesis Reference: Section 3.2.2 - Output Sanitization and Artifact Filtering]
    """
    if g is None:
        return ""
    
    g = str(g).strip()
    g = unicodedata.normalize("NFKC", g)

    # Remove a trailing quote if it looks like a formatting artifact without an opening counterpart
    if (g.endswith('"') or g.endswith('”')) and not ('"' in g[:-1] or '“' in g[:-1]):
        g = g[:-1].strip()

    g = clean_continuation_text(g)
    
    # Enforce sentence boundaries by truncating after the first terminal punctuation
    if stop_at_sentence:
        m = re.search(r'([.?!][\"\'”’]?)(?=\s|$)', g)
        if m:
            idx = m.end(1)
            g = g[:idx].strip()
        elif "\n" in g:
            g = g.split("\n")[0].strip()
            
    return g


# ------------------------------------------------------------------------
# Autoregressive (AR) Adapter
# ------------------------------------------------------------------------

class ARAdapter(BaseAdapter):
    """
    Adapter for Autoregressive (AR) Models like LLaMA.
    Executes standard left-to-right causal decoding.
    [Thesis Reference: Section 3.1.2 - The ARAdapter]
    """
    def __init__(self, model_name: str, model, tokenizer, ar_score_fn: Callable):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.ar_score_fn = ar_score_fn

    def generate(self, prompt: str, num_return_sequences: int = 1, batch_size: int = 25, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0, max_new_tokens: int = 64, stop_at_sentence: bool = True, **kwargs) -> List[str]:
        """
        Generates continuations utilizing highly-parallelized batch decoding.
        Implements chunk-based micro-batching to prevent VRAM exhaustion.
        Includes a robust fallback for attention_mask errors common in quantized models.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # --- REPRODUCIBILITY FIX ---
        # Strip Diffusion-specific arguments from the kwargs pool
        kwargs.pop("steps", None)
        kwargs.pop("cfg_scale", None)
        kwargs.pop("mc_num", None)
        kwargs.pop("mc_batch_size", None)
        base_seed = kwargs.pop("seed", 42)
        
        final_gens = []
        remaining_sequences = num_return_sequences
        chunk_index = 0

        while remaining_sequences > 0:
            current_batch_size = min(batch_size, remaining_sequences)
            chunk_seed = base_seed + chunk_index

            if chunk_seed is not None:
                torch.manual_seed(chunk_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(chunk_seed)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "top_k": top_k,
                "num_return_sequences": current_batch_size,
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            }
            gen_kwargs.update(_filter_kwargs_for_call(self.model.generate, kwargs))

            # Robust generation with Attention-Mask Fallback
            try:
                outputs = self.model.generate(**inputs, **gen_kwargs)
            except Exception:
                try:
                    inputs.pop("attention_mask", None)
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                except Exception as e:
                    print(f"[ERROR] AR Generation failed on chunk {chunk_index}: {e}")
                    final_gens.extend([""] * current_batch_size)
                    remaining_sequences -= current_batch_size
                    chunk_index += 1
                    continue
            
            prompt_length = inputs["input_ids"].shape[1]
            generated_sequences = outputs[:, prompt_length:]
            decoded_texts = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            
            chunk_gens = [_post_process_generation(g, stop_at_sentence) for g in decoded_texts]
            final_gens.extend(chunk_gens)

            remaining_sequences -= current_batch_size
            chunk_index += 1

        # Pad to ensure exactly `num_return_sequences` are returned
        if len(final_gens) < num_return_sequences:
            final_gens += [""] * (num_return_sequences - len(final_gens))
        return final_gens[:num_return_sequences]

    def score_continuations(self, prompts: List[str], continuations: List[str], **kwargs) -> List[Optional[float]]:
        """
        Calculates exact Sequence Negative Log-Likelihood (NLL).
        [Thesis Reference: Equation 2 (AR Exact Likelihood)]
        """
        return self.ar_score_fn(prompts, continuations)


# ------------------------------------------------------------------------
# Discrete Text Diffusion (LLaDA) Adapter
# ------------------------------------------------------------------------

class LLaDaAdapter(BaseAdapter):
    """
    Adapter for Discrete Text Diffusion Models (LLaDA).
    Executes iterative mask-based bidirectional refinement.
    [Thesis Reference: Section 3.1.2 - The LLaDaAdapter]
    """
    def __init__(self, model_name: str, model, tokenizer, diff_mc_nll: Callable):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.diff_mc_nll = diff_mc_nll

    def generate(self, prompt: str, num_return_sequences: int = 1, batch_size: int = 25, top_p: float = 1.0, top_k: int = 0, temperature: float = 1.0, max_new_tokens: int = 64, stop_at_sentence: bool = True, **kwargs) -> List[str]:
        """
        Generates continuations using iterative unmasking block-wise diffusion.
        Implements chunked micro-batching to prevent OOM errors on consumer GPUs 
        while preserving exact cryptographic reproducibility via chunk-level seeding.
        
        [Thesis Reference: Section 3.2.1 - Configuring the Generation Process]
        """
        final_gens = []
        base_seed = kwargs.get("seed", 42)
        call_kwargs = dict(kwargs)
        
        remaining_sequences = num_return_sequences
        chunk_index = 0
        
        while remaining_sequences > 0:
            # Determine how many sequences to generate in this specific chunk
            current_batch_size = min(batch_size, remaining_sequences)
            
            # Deterministic seed advancement per chunk guarantees that 
            # N chunks of size B perfectly matches another user running N chunks of size B.
            chunk_seed = base_seed + chunk_index
            
            if chunk_seed is not None:
                torch.manual_seed(chunk_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(chunk_seed)
            
            try:
                # Pass the CURRENT chunk size to the loader for parallel processing
                raw_continuations = run_llada_prompt(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt_text=prompt,
                    num_return_sequences=current_batch_size, 
                    top_p=top_p,
                    temperature=temperature,
                    top_k=top_k,
                    gen_length=max_new_tokens,
                    **_filter_kwargs_for_call(run_llada_prompt, call_kwargs)
                )
                
                # Ensure raw_continuations is iterable (list of strings)
                if isinstance(raw_continuations, str):
                    raw_continuations = [raw_continuations]
                    
            except Exception as e:
                print(f"[ERROR] LLaDA Batched Generation failed on chunk {chunk_index}: {e}")
                raw_continuations = [""] * current_batch_size
            
            # Process and store this chunk's results
            chunk_gens = [_post_process_generation(g, stop_at_sentence) for g in raw_continuations]
            final_gens.extend(chunk_gens)
            
            # Update counters for the next chunk
            remaining_sequences -= current_batch_size
            chunk_index += 1
            
        # Pad to ensure exactly `num_return_sequences` are returned
        if len(final_gens) < num_return_sequences:
            final_gens += [""] * (num_return_sequences - len(final_gens))
            
        return final_gens[:num_return_sequences]

    def score_continuations(self, prompts: List[str], continuations: List[str], **kwargs) -> List[Optional[float]]:
        """
        Approximates Sequence Negative Log-Likelihood via Monte Carlo (MC) estimation.
        [Thesis Reference: Equation 10 (MC NLL Estimator) & Section 3.3.1]
        """
        return self.diff_mc_nll(prompts, continuations)