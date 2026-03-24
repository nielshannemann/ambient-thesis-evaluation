#!/usr/bin/env python3
# src/ambient/llada_loader.py
"""
=============================================================================
ORIGINAL AUTHORSHIP ACKNOWLEDGEMENT & ADAPTATION
=============================================================================
This script is adapted from the official LLaDA repository:
Repository: https://github.com/ML-GSAI/LLaDA
Paper: "LLaDA: A Simple, Scalable and General Purpose Text Diffusion Model" 
(Nie et al., 2024).

Modifications for this thesis:
- Restructured for memory-friendly iterative unmasking.
- Added strict EOS-banning and Chat-Artifact stripping for pristine generation.
- Added academic cross-references mapping the code to the methodology chapter.
- Heavily optimized for batched inference by expanding input tensors prior 
  to the denoising loop, allowing parallel sequence generation.
=============================================================================
"""

import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# [Thesis Reference: Section 3.2.2 - Output Sanitization]
from ambient.utils import clean_continuation_text, _ensure_tokenizer_has_pad

# --- Configuration & Hyperparameters ---
# [Thesis Reference: Section 3.1.1 - Initialization & Quantization]
HF_MODEL = os.environ.get("AMBIENT_HF_MODEL", "GSAI-ML/LLaDA-8B-Base")
USE_4BIT = False

# [Thesis Reference: Section 3.2.1 - Configuring the Generation Process]
DEFAULT_STEPS = 128
DEFAULT_GEN_LENGTH = 64
DEFAULT_BLOCK_LENGTH = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CFG_SCALE = 0.0
DEFAULT_REMASKING = "low_confidence"
DEFAULT_TOP_P = 1.0
TOP_K = 0

MIN_TEMP = 1e-6
FORBID_EOS = True # Prevents premature sequence termination during iterative unmasking
tokenizer = None


def get_embedding_device(model) -> torch.device:
    """Safely extracts the computational device the model is currently residing on."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_llada_model(hf_model: str = HF_MODEL, use_4bit: bool = USE_4BIT, verbose: bool = True):
    """
    Loads the LLaDA architecture.
    [Thesis Reference: Section 3.1.1 - Initialization]
    """
    global tokenizer
    
    if verbose and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"[INFO] GPU Memory: {free_b // (1024**2)} MB free / {total_b // (1024**2)} MB total")

    if use_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(hf_model, quantization_config=bnb, cache_dir="./models", trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto")
            if verbose: print("[INFO] Successfully loaded LLaDA in 4-bit (NF4) mode.")
        except ImportError:
            print("[WARN] bitsandbytes not found. Falling back to float16.")
            model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16, cache_dir="./models", trust_remote_code=True).cuda()
    else:
        # PURE METAL MODE: Direct to VRAM without distributed hooks for maximum throughput
        model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16, cache_dir="./models", trust_remote_code=True).cuda()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    tokenizer = _ensure_tokenizer_has_pad(tokenizer, model=model, prefer_eos=True)

    return model, tokenizer


def _apply_top_k_top_p(probs_flat: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    Applies Top-K and Nucleus (Top-p) truncation to the probability distribution.
    [Thesis Reference: Section 3.2.1 - Top-p / Top-k Sampling]
    """
    if probs_flat.ndim != 2 or (top_k <= 0 and top_p >= 1.0):
        return probs_flat

    V = probs_flat.shape[-1]
    K = min(top_k, V) if top_k > 0 else V

    if top_k > 0:
        topk_vals, topk_idx = torch.topk(probs_flat, k=K, dim=-1)
        cand_sums = torch.clamp(topk_vals.sum(dim=-1, keepdim=True), min=1e-12)
        cand_probs = topk_vals / cand_sums
    else:
        # Fallback if only Top-p is active
        topk_idx = torch.arange(V, device=probs_flat.device).expand(probs_flat.shape[0], -1)
        cand_probs = probs_flat

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(cand_probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        keep_sorted = cumsum <= top_p
        keep_sorted[:, 0] = True 
        
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(1, sorted_idx, keep_sorted)
        
        cand_probs = cand_probs * keep.to(cand_probs.dtype)
        cand_probs = cand_probs / torch.clamp(cand_probs.sum(dim=-1, keepdim=True), min=1e-12)

    out = torch.zeros_like(probs_flat)
    out.scatter_(1, topk_idx, cand_probs)
    return out / torch.clamp(out.sum(dim=-1, keepdim=True), min=1e-12)


def decode_suffix_from_raw_tensor(tokenizer, raw_tensor: torch.LongTensor, prompt_len: int) -> list:
    """
    Extracts the newly generated tokens for an entire batch and strictly removes 
    LLaDA chat artifacts.
    [Thesis Reference: Section 3.2.2 - Output Sanitization and Artifact Filtering]
    """
    # Slice off the prompt and convert the batch to a list of lists
    suffix_ids_batch = raw_tensor[:, prompt_len:].cpu().tolist()
    
    decoded_texts = []
    for suffix_ids in suffix_ids_batch:
        txt = tokenizer.decode(suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Strip hallucinated chat roles (e.g. "assistant", "user") from the tail
        txt = re.sub(r'(?:[\.\s\n]+)?(?:assistant|user|system)(?:[\.\s\n]+)?$', '', txt, flags=re.IGNORECASE)
        txt = re.sub(r'<\|.*?$', '', txt)
        decoded_texts.append(txt.strip())
        
    return decoded_texts


@torch.no_grad()
def generate_memory_friendly_diffusion(
    model,
    prompt_ids: torch.LongTensor,
    steps: int = DEFAULT_STEPS,
    gen_length: int = DEFAULT_GEN_LENGTH,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    temperature: float = DEFAULT_TEMPERATURE,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    remasking: str = DEFAULT_REMASKING,
    top_k: int = TOP_K,
    top_p: float = DEFAULT_TOP_P,
    forbid_eos: bool = FORBID_EOS
):
    """
    Core generation loop for discrete text diffusion.
    Iteratively unmasks tokens in blocks over T timesteps.
    Supports parallel batch processing (B > 1) natively.
    
    [Thesis Reference: Section 1.2.2 - Diffusion Language Modeling (LLaDA)]
    """
    device = get_embedding_device(model)
    mask_id = getattr(model.config, "mask_token_id", getattr(tokenizer, "mask_token_id", 126336))

    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    
    if steps < (gen_length // block_length):
        block_length = gen_length
        
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    # Initialize batch tensor: [batch_size, prompt_len + gen_length]
    x = torch.full((batch_size, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.to(device).clone()
    
    noise_scale = max(float(temperature) if temperature > 0.0 else MIN_TEMP, MIN_TEMP)

    for nb in range(num_blocks):
        block_start = prompt_len + nb * block_length
        block_end = prompt_len + (nb + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        mask_num = block_mask_index.sum(dim=1, keepdim=True)
        base_transfer = mask_num // steps_per_block
        remainder = (mask_num % steps_per_block).view(-1)
        
        num_transfer_tokens = torch.zeros(batch_size, steps_per_block, device=device, dtype=torch.int64) + base_transfer.view(-1, 1)
        for i in range(batch_size):
            num_transfer_tokens[i, :int(remainder[i])] += 1

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[:, :prompt_len] = mask_id 
                x_combined = torch.cat([x, un_x], dim=0)
                
                logits_combined = model(x_combined).logits
                logits_cond, logits_uncond = torch.chunk(logits_combined, 2, dim=0)
                logits = logits_uncond + (cfg_scale + 1.0) * (logits_cond - logits_uncond)
            else:
                logits = model(x).logits

            logits = logits.float()

            if forbid_eos:
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None and 0 <= eos_id < logits.shape[-1]:
                    logits[..., eos_id] = -1e9
            
            logits = logits - logits.max(dim=-1, keepdim=True).values
            logits = logits * 0.95 + (0.05 * logits.mean(dim=-1, keepdim=True))
            probs = F.softmax(logits / noise_scale, dim=-1)

            probs_flat = probs.view(-1, probs.shape[-1])
            probs_flat = _apply_top_k_top_p(probs_flat, top_k=top_k, top_p=top_p)

            # Sample across the entire batch simultaneously
            sampled = torch.multinomial(probs_flat, num_samples=1).view(batch_size, -1)

            del logits, probs

            if remasking == "low_confidence":
                sampled_idx = sampled.view(-1, 1).to(device)
                confidence_scores = torch.gather(probs_flat, dim=1, index=sampled_idx).view(batch_size, -1).to(device)
            elif remasking == "random":
                confidence_scores = torch.rand((batch_size, sampled.shape[1]), device=device)
            else:
                raise NotImplementedError(f"Remasking strategy {remasking} not supported.")

            x0 = torch.where(mask_index, sampled.long(), x)
            
            confidence_scores[:, block_end:] = -float("inf")
            confidence = torch.where(mask_index, confidence_scores, -float("inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(confidence.shape[0]):
                k_tokens = int(num_transfer_tokens[j, i].item())
                if k_tokens > 0:
                    _, select_index = torch.topk(confidence[j], k=k_tokens)
                    transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]

            del probs_flat, sampled

    return x


def run_llada_prompt(model, tokenizer, prompt_text: str, num_return_sequences: int = 1, **kwargs) -> list:
    """
    Entry point for the LLaDaAdapter.
    Expands the input prompt by `num_return_sequences` to enable parallel diffusion batching.
    [Thesis Reference: Section 3.1.2 - The LLaDaAdapter]
    """
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    
    # Expand the prompt tensor to create a batch!
    prompt_ids = enc["input_ids"].long().to(get_embedding_device(model))
    prompt_ids = prompt_ids.repeat(num_return_sequences, 1)

    out_tensor = generate_memory_friendly_diffusion(
        model=model,
        prompt_ids=prompt_ids,
        steps=kwargs.get("steps", DEFAULT_STEPS),
        gen_length=kwargs.get("gen_length", DEFAULT_GEN_LENGTH),
        block_length=kwargs.get("block_length", DEFAULT_BLOCK_LENGTH),
        temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
        cfg_scale=kwargs.get("cfg_scale", DEFAULT_CFG_SCALE),
        top_k=kwargs.get("top_k", TOP_K),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        forbid_eos=kwargs.get("forbid_eos", FORBID_EOS)
    )

    prompt_len = int(prompt_ids.shape[1])
    decoded_texts = decode_suffix_from_raw_tensor(tokenizer, out_tensor, prompt_len)
    
    return [clean_continuation_text(txt) for txt in decoded_texts]