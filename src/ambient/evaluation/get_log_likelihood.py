# src/ambient/evaluation/get_log_likelihood.py
"""
=============================================================================
ORIGINAL AUTHORSHIP ACKNOWLEDGEMENT & ADAPTATION
=============================================================================
This script is directly adapted from the official LLaDA repository:
Repository: https://github.com/ML-GSAI/LLaDA
Paper: "LLaDA: A Simple, Scalable and General Purpose Text Diffusion Model" 
(Nie et al., 2024).

Modifications for this thesis:
- Adapted for AMBIENT dataset scoring (handling prompt + continuation splits).
- Integrated into the Adapter Framework as the MC NLL backend.
- Implemented deterministic local hashing for strictly reproducible MC sampling.
- Added academic cross-references mapping the code to the theoretical 
  mathematical equations established in the methodology chapter.
=============================================================================
"""

import math
import hashlib
import torch
import torch.nn.functional as F

def forward_process(batch, prompt_index, mask_id, rng=None):
    """
    Creates the corrupted sequence \tilde{X}^{(m)} for the diffusion process.
    
    [Thesis Ref: Section 2.3.1 - Variance Reduction via Stratified Sampling]
    Utilizes stratified sampling (via torch.linspace) across the batch dimension 
    to evenly distribute masking ratios. This significantly reduces the variance 
    of the Monte Carlo estimator compared to independent uniform sampling.
    """
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    
    # Generate the base masking count using the provided deterministic RNG
    k = torch.randint(1, target_len + 1, (), generator=rng, device=batch.device)

    # Apply stratified sampling to cover the range of mask ratios within one batch
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    
    # Shuffle mask positions deterministically per batch row
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len, generator=rng, device=batch.device)]

    # Preserve prompt tokens by ensuring they are never masked
    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the corresponding mask ratios (p_mask)
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    """
    Computes model logits, optionally applying Classifier-Free Guidance (CFG).
    [Thesis Ref: Equation 8 - Classifier-Free Guidance for Diffusion]
    """
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index_expanded = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        
        # Construct the unconditional pass by masking the entire prompt
        un_batch = batch.clone()
        un_batch[prompt_index_expanded] = mask_id
        batch = torch.cat([batch, un_batch])

    logits = model(batch).logits

    if cfg_scale > 0.:
        # Extrapolate between conditional and unconditional logits
        logits_cond, logits_uncond = torch.chunk(logits, 2, dim=0)
        logits = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
        
    return logits

@torch.no_grad()
def get_log_likelihood(model, tokenizer, prompts, continuations, mc_nums=[128], batch_size=16, cfg_scale=0.0, seed=42):
    """
    Computes MC NLL for multiple mc_num levels simultaneously.
    Returns a list of lists: [ [scores_for_mc_1], [scores_for_mc_2], ... ]
    """
    mask_id = getattr(model.config, "mask_token_id", getattr(tokenizer, "mask_token_id", 126336))
    device = next(model.parameters()).device
    
    # Sort mc_nums to ensure we can collect intermediate results
    sorted_mc = sorted(list(set(mc_nums)))
    max_mc = sorted_mc[-1]
    
    # Initialize result storage for each mc_level
    results_per_level = {m: [] for m in sorted_mc}

    for prompt_str, cont_str in zip(prompts, continuations):
        if not cont_str.strip():
            for m in sorted_mc: results_per_level[m].append(None)
            continue

        # Deterministic seeding (Hash-based)
        text_hash = int(hashlib.md5((prompt_str + cont_str).encode('utf-8')).hexdigest(), 16)
        local_seed = (seed + text_hash) % (2**31)
        rng = torch.Generator(device=device)
        rng.manual_seed(local_seed)

        p_ids = torch.tensor(tokenizer(prompt_str, add_special_tokens=True)["input_ids"], device=device)
        c_ids = torch.tensor(tokenizer(cont_str, add_special_tokens=False)["input_ids"], device=device)
        seq = torch.cat([p_ids, c_ids])[None, :]
        prompt_index = torch.arange(seq.shape[1], device=device) < len(p_ids)

        total_loss_running = 0.0
        samples_processed = 0
        
        # Batch-Loop bis zum Maximum der MC-Liste
        num_batches = math.ceil(max_mc / batch_size)
        
        mc_idx = 0
        for b in range(num_batches):
            current_batch_size = min(batch_size, max_mc - samples_processed)
            seq_batch = seq.repeat((current_batch_size, 1))
            
            perturbed_seq, p_mask = forward_process(seq_batch, prompt_index, mask_id, rng=rng)
            mask_index_tensor = perturbed_seq == mask_id
            logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)
            
            # 1. Compute Cross Entropy on the full flattened batch (B*L, Vocab) vs (B*L,)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                seq_batch.view(-1), 
                reduction='none'
            )
            
            # 2. Reshape back to (Batch, Sequence_Length)
            ce_loss = ce_loss.view(current_batch_size, -1)
            
            # 3. Apply the p_mask weight only where the tokens are actually masked
            weighted_loss = ce_loss * mask_index_tensor.float() / p_mask
            
            # 4. Zero out the loss for unmasked tokens, then sum across the sequence dimension
            weighted_loss = weighted_loss.masked_fill(~mask_index_tensor, 0.0)
            loss_items = weighted_loss.sum(dim=1)
            
            for item_loss in loss_items:
                total_loss_running += item_loss.item()
                samples_processed += 1
                
                # Wenn wir eine Grenze aus sorted_mc erreichen, speichern wir den aktuellen Durchschnitt
                if mc_idx < len(sorted_mc) and samples_processed == sorted_mc[mc_idx]:
                    results_per_level[sorted_mc[mc_idx]].append(total_loss_running / samples_processed)
                    mc_idx += 1

    return [results_per_level[m] for m in sorted_mc]