"""Dream 7B loading and generation through its official remote-code API."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from ambient.constants import DEFAULT_MODELS_CACHE_DIR, DREAM_BASE_MODEL_ID
from ambient.utils import _ensure_tokenizer_has_pad, clean_continuation_text


def get_model_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _decode_dream_suffix(tokenizer: Any, sequence: torch.Tensor, prompt_len: int) -> str:
    """Decode one Dream suffix and stop at the first generated EOS marker."""
    text = tokenizer.decode(
        sequence[prompt_len:].tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and eos_token in text:
        text = text.split(eos_token, 1)[0]
    return clean_continuation_text(text)


def load_dream_model(
    hf_model: str = DREAM_BASE_MODEL_ID,
    use_4bit: bool = False,
    verbose: bool = True,
) -> tuple[Any, Any]:
    """Load Dream via ``AutoModel`` as required by the upstream checkpoint."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs: dict[str, Any] = {
        "cache_dir": DEFAULT_MODELS_CACHE_DIR,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            }
        )

    if verbose:
        print(f"[INFO] Loading Dream model {hf_model} (4-bit={use_4bit})")
        print("[INFO] Dream uses trust_remote_code=True and its official diffusion_generate API.")

    model = AutoModel.from_pretrained(hf_model, **load_kwargs)
    if torch.cuda.is_available() and not use_4bit:
        model = model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        cache_dir=DEFAULT_MODELS_CACHE_DIR,
        trust_remote_code=True,
    )
    tokenizer = _ensure_tokenizer_has_pad(tokenizer, model=model, prefer_eos=True)
    return model, tokenizer


@torch.no_grad()
def run_dream_prompt(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    num_return_sequences: int = 1,
    steps: int = 64,
    gen_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    return_history: bool = False,
) -> list[str] | tuple[list[str], Any]:
    """Generate a fixed-length suffix with Dream's official sampler."""
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    device = get_model_device(model)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        num_return_sequences=num_return_sequences,
        output_history=return_history,
        return_dict_in_generate=True,
        steps=steps,
        temperature=temperature,
        top_p=None if top_p >= 1.0 else top_p,
        top_k=None if top_k <= 0 else top_k,
        alg=alg,
        alg_temp=alg_temp,
    )

    prompt_len = input_ids.shape[1]
    decoded: list[str] = []
    for sequence in output.sequences:
        decoded.append(_decode_dream_suffix(tokenizer, sequence, prompt_len))

    if return_history:
        return decoded, getattr(output, "history", None)
    return decoded
