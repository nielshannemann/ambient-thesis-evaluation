# src/ambient/utils.py
"""
=============================================================================
AMBIENT UTILITIES & SANITIZATION PIPELINE
=============================================================================
This module provides the low-level functions for file I/O, token counting, 
and text sanitization. It acts as the primary defense against generation 
artifacts produced by diffusion models.

Restored Features from Original Codebase:
- NFS-safe atomic writing via os.fsync().
- Fault-tolerant JSONL reading (skipping malformed tail-lines).
- Dynamic repetitive-character truncation (salvaging generations).
- Advanced LLaDA artifact filtering (CJK, placeholders, empty content).

[Thesis Ref: Section 2.2.2 - Output Sanitization and Artifact Filtering]
=============================================================================
"""

import os
import re
import json
import zlib
import tempfile
from pathlib import Path
from typing import Optional, List, Any, Dict

# --- Pre-compiled Regular Expressions for Output Sanitization ---
# [Thesis Ref: Section 2.2.2 - Rule 2: Consecutive Repetitions]
_RE_REPEAT = re.compile(r'([^\w\s])\1{3,}')  # e.g., "...." or "????"
_RE_LONG_CHAR = re.compile(r'(.)\1{20,}')    # Extravagant character repetition
_RE_TRAILING_SAME = re.compile(r'([^\w\s])\1+$') # Trailing punctuation spam

# [Thesis Ref: Section 2.2.2 - Rule 4: CJK Character Masking Artifacts]
_RE_CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

# Orphaned or trailing punctuation removal
_RE_TRAILING_PUNCT = re.compile(r'[\)\]\}\s]+$')

# [Thesis Ref: Section 2.2.2 - Rule 3: Stray Multiple Choice Patterns]
_RE_MCQ_PATTERN = re.compile(r'^\s*[A-D][\.\s\)]')


# ------------------------------------------------------------------------
# I/O Helpers
# ------------------------------------------------------------------------

def ensure_dir(d: str | Path):
    """Safely ensures a directory exists."""
    Path(d).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[dict]:
    """
    Fault-tolerant JSONL reader. 
    Skips malformed lines which frequently occur when cluster jobs are killed mid-write.
    """
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # Gracefully skip malformed lines (e.g., EOF cuts)
                    continue
    except FileNotFoundError:
        return []
    return out


def write_json_atomic(path: str | Path, data: dict):
    """
    Safely writes metadata to disk. Uses os.fsync to ensure data clears 
    the OS buffer and hits the physical disk (crucial for NFS cluster setups).
    [Thesis Ref: Section 2.1.1 - Aggregation & Provenance]
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno()) # Force physical write
            except Exception:
                pass
        Path(tmp).replace(path)
    except Exception:
        # Fallback if temp-file creation fails
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def make_instance_id(row: dict) -> str:
    """
    Generates a deterministic hash for deduplication.
    [Thesis Ref: Section 2.4.2 - Instance-Level Deduplication]
    """
    try:
        row_json = json.dumps(row, sort_keys=True, default=str)
        instance_hash = zlib.crc32(row_json.encode("utf-8")) & 0xffffffff
        return f"{row.get('id')}_{instance_hash:x}"
    except Exception:
        return str(row.get('id'))


# ------------------------------------------------------------------------
# Text Sanitization & Validation
# ------------------------------------------------------------------------

def clean_continuation_text(text: str) -> str:
    """
    Standardizes whitespace, caps repeating artifacts, and removes orphaned punctuation.
    Instead of discarding repetitive sequences, it truncates them to a reasonable length.
    [Thesis Ref: Section 2.2.2 - Output Sanitization and Artifact Filtering]
    """
    if not text:
        return ""
    
    # 1. Standardize quotes and whitespace
    t = text.replace('”', '"').replace('“', '"').replace('’', "'").replace('‘', "'")
    t = t.replace('\r', ' ').replace('\t', ' ')
    
    # 2. Dynamic Repetition Capping (Salvages usable strings)
    t = _RE_REPEAT.sub(lambda m: m.group(1) * 3, t)
    t = _RE_LONG_CHAR.sub(lambda m: m.group(1) * 5, t)
    t = _RE_TRAILING_SAME.sub(r'\1', t)
    
    # 3. Strip trailing loose punctuation
    t = _RE_TRAILING_PUNCT.sub('', t).strip()

    # 4. Balance trailing unmatched quotes if the generation cut off abruptly
    if t.count('"') % 2 != 0:
        if t.endswith('"'):
            t = t[:-1].strip()
        elif not t.startswith('"'):
            t = t + '"'
            
    return t.strip()


def is_suspicious(text: str, max_non_alnum_ratio: float = 0.35, max_consec_repeat: int = 12) -> bool:
    """
    Heuristic filter to actively discard degenerated sequences produced by the diffusion process.
    Returns True if the continuation should be flagged and excluded from metric aggregation.
    
    [Thesis Ref: Section 2.2.2 - Artifact Filtering Heuristics]
    """
    if not text or len(text.strip()) < 2:
        return True

    text = text.strip()

    # Rule 1: High Non-Alphanumeric Ratio
    alnum_count = sum(1 for c in text if c.isalnum() and not c.isspace())
    if (1.0 - (alnum_count / max(1, len(text)))) > max_non_alnum_ratio:
        return True

    # Rule 2: Consecutive Repetitions (Fallback if clean_continuation_text didn't catch it)
    if re.search(r'(.)\1{%d,}' % max_consec_repeat, text):
        return True

    # Rule 3: Stray Multiple Choice Patterns (e.g., "A.", "B.")
    if _RE_MCQ_PATTERN.match(text):
        return True

    # Rule 4: CJK Character Masking Artifacts
    if _RE_CJK.search(text):
        return True

    # Rule 5: LLaDA Placeholders / Empty Brackets
    if "( )" in text or "【" in text or "】" in text:
        return True

    # Rule 6: Insufficient real word content (e.g., only a single letter and punctuation)
    if len(re.sub(r'[^\w]', '', text)) < 2:
        return True

    return False


# ------------------------------------------------------------------------
# Tokenizer Utilities
# ------------------------------------------------------------------------

def _num_tokens(text: str, tokenizer_to_use=None) -> int:
    """
    Calculates the exact token length of a sanitized continuation.
    Required to normalize raw log-odds for the empirical KL divergence.
    """
    if not text:
        return 0
    if tokenizer_to_use is not None:
        try:
            # Use Hugging Face tokenizer if available
            out = tokenizer_to_use(text, return_tensors='pt', add_special_tokens=False)
            if "input_ids" in out:
                return int(out["input_ids"].size(1))
            elif isinstance(out, list):
                return len(out)
        except Exception:
            pass
            
    # Fallback to rough whitespace approximation if tokenizer fails
    return max(1, len(text.split()))


def _ensure_tokenizer_has_pad(tokenizer, model=None, prefer_eos: bool = True):
    """
    Ensures the tokenizer has a defined padding token.
    Critical for batched autoregressive generation.
    """
    if getattr(tokenizer, "pad_token", None) is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer
        
    if prefer_eos and getattr(tokenizer, "eos_token", None) is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
        
    try:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
        
    return tokenizer


def _normalize_missing(vals: Any) -> List[Any]:
    if vals is None:
        return []
    if isinstance(vals, (int, float)):
        return [vals]
    if hasattr(vals, "tolist"):
        return vals.tolist()
    return list(vals)


def _ensure_list_of_len(lst: Any, length: int) -> List[Any]:
    if not lst:
        return [None] * length
        
    if hasattr(lst, "tolist"):
        lst = lst.tolist()
    elif not isinstance(lst, list):
        try:
            lst = list(lst)
        except Exception:
            return [None] * length
            
    if len(lst) < length:
        return lst + [None] * (length - len(lst))
    return lst[:length]