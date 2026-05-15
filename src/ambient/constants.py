"""Shared constants for the public AmbiEnt-style CLI and output contract."""

from pathlib import Path


DEFAULT_DATA_PATH = Path("data/test_baked.jsonl")
DEFAULT_MODELS_CACHE_DIR = "./models"

MODEL_FAMILY_CHOICES = ("llama", "llada")

LLAMA_BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
LLADA_BASE_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

LLAMA_INSTRUCT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLADA_INSTRUCT_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

TASK6_JUDGE_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"
TASK6_SECONDARY_JUDGE_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
TASK2_EMBED_MODEL_ID = "all-MiniLM-L6-v2"
TASK3_NLI_MODEL_ID = "roberta-large-mnli"
