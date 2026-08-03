"""Shared model-family resolution and loading for experiment entrypoints.

The historical ``llama`` and ``llada`` family names remain public API. The
generic ``ar`` family accepts any Hugging Face causal LM, while ``dream`` adds
the Dream masked-diffusion backend used by the October extension experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any

import torch

from ambient.constants import (
    DREAM_BASE_MODEL_ID,
    DREAM_INSTRUCT_MODEL_ID,
    LLADA_BASE_MODEL_ID,
    LLADA_INSTRUCT_MODEL_ID,
    LLAMA_BASE_MODEL_ID,
    LLAMA_INSTRUCT_MODEL_ID,
    QWEN_BASE_MODEL_ID,
    QWEN_INSTRUCT_MODEL_ID,
)


MASKED_DIFFUSION_FAMILIES = frozenset({"llada", "dream"})


def runtime_environment() -> dict[str, Any]:
    """Capture software and visible-device versions in new experiment metadata."""
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        environment["visible_cuda_device"] = torch.cuda.get_device_name(0)
    return environment


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model/tokenizer pair plus normalized architecture metadata."""

    model: Any
    tokenizer: Any
    requested_family: str
    backend: str
    architecture: str
    model_id: str
    use_4bit: bool


def canonical_backend(model_family: str) -> str:
    family = model_family.lower()
    if family == "llama":
        return "ar"
    if family in {"ar", "llada", "dream"}:
        return family
    raise ValueError(f"Unsupported model family: {model_family}")


def is_autoregressive_family(model_family: str) -> bool:
    return canonical_backend(model_family) == "ar"


def is_masked_diffusion_family(model_family: str) -> bool:
    return canonical_backend(model_family) in MASKED_DIFFUSION_FAMILIES


def default_base_model_id(model_family: str) -> str:
    defaults = {
        "llama": LLAMA_BASE_MODEL_ID,
        "ar": QWEN_BASE_MODEL_ID,
        "llada": LLADA_BASE_MODEL_ID,
        "dream": DREAM_BASE_MODEL_ID,
    }
    try:
        return defaults[model_family.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model family: {model_family}") from exc


def default_instruct_model_id(model_family: str) -> str:
    defaults = {
        "llama": LLAMA_INSTRUCT_MODEL_ID,
        "ar": QWEN_INSTRUCT_MODEL_ID,
        "llada": LLADA_INSTRUCT_MODEL_ID,
        "dream": DREAM_INSTRUCT_MODEL_ID,
    }
    try:
        return defaults[model_family.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model family: {model_family}") from exc


def auto_detect_4bit(model_id: str) -> bool:
    """Use NF4 only when the visible GPU is unlikely to fit full precision."""
    if not torch.cuda.is_available():
        return False

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    model_id_lower = model_id.lower()
    if any(size in model_id_lower for size in ("70b", "72b", "65b")):
        return vram_gb < 130
    if any(size in model_id_lower for size in ("7b", "8b", "9b")):
        return vram_gb < 20
    return vram_gb < 16


def _load_ar_model(model_id: str, use_4bit: bool, verbose: bool) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "cache_dir": "./models",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    if use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    if verbose:
        print(f"[INFO] Loading autoregressive model {model_id} (4-bit={use_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir="./models",
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_model_bundle(
    model_family: str,
    model_id: str | None = None,
    use_4bit: bool | None = None,
    verbose: bool = True,
) -> ModelBundle:
    """Load one supported backend without changing historical CLI semantics."""
    backend = canonical_backend(model_family)
    resolved_model_id = model_id or default_base_model_id(model_family)
    resolved_4bit = auto_detect_4bit(resolved_model_id) if use_4bit is None else use_4bit

    if backend == "ar":
        model, tokenizer = _load_ar_model(resolved_model_id, resolved_4bit, verbose)
        architecture = "autoregressive"
    elif backend == "llada":
        from ambient.llada_loader import load_llada_model

        model, tokenizer = load_llada_model(
            hf_model=resolved_model_id,
            use_4bit=resolved_4bit,
            verbose=verbose,
        )
        architecture = "masked_diffusion"
    else:
        from ambient.dream_loader import load_dream_model

        model, tokenizer = load_dream_model(
            hf_model=resolved_model_id,
            use_4bit=resolved_4bit,
            verbose=verbose,
        )
        architecture = "masked_diffusion"

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        requested_family=model_family,
        backend=backend,
        architecture=architecture,
        model_id=resolved_model_id,
        use_4bit=resolved_4bit,
    )
