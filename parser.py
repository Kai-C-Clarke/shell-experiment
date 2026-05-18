"""
parser.py — Strict numerical output enforcement.
Parse → validate → retry on failure. No text passes through.
"""
import re
import logging

log = logging.getLogger(__name__)

DIM = 32
MAX_RETRIES = 3


def strict_parse_output(raw: str, dim: int = DIM) -> list:
    """
    Extract exactly DIM floats from model output.
    Tries multiple patterns. Raises ValueError if all fail.
    """
    patterns = [
        r"OUT:\s*\[([\d\s\.\-eE,]+)\]",                          # OUT:[1.23 4.56]
        r"OUT:\s*([\d\s\.\-eE,]+)$",                             # OUT: 1.23 4.56 (no brackets)
        r"\{[^}]*[\"']?out[\"']?\s*:\s*\[([\d\s\.\-eE,]+)\]",   # {"out": [...]}
        r"\[([\d\s\.\-eE,]+)\]",                                  # bare array
    ]

    for pat in patterns:
        match = re.search(pat, raw, re.IGNORECASE | re.MULTILINE)
        if match:
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
            if len(numbers) == dim:
                parsed = [float(x) for x in numbers]
                # Clamp to [-1, 1]
                clamped = [max(-1.0, min(1.0, x)) for x in parsed]
                return clamped

    raise ValueError(f"Parse failed (dim={dim}). Raw output: {raw[:300]}")


def make_retry_prompt(round_num: int, dim: int = DIM) -> str:
    """Stronger formatting hint on retry — still minimal language"""
    return (
        f"RND:{round_num} PARSE_FAIL\n"
        f"REQUIRED: OUT:[x1 x2 ... x{dim}]\n"
        f"EXAMPLE: OUT:[0.1234 -0.5678 0.0000 ...] ({dim} floats)\n"
        f"OUT:"
    )


def fallback_vector(dim: int = DIM) -> list:
    """Emergency fallback — silence. Used if all retries fail."""
    log.warning(f"All parse retries failed — using silence fallback")
    return [0.0] * dim


def validate_vector(v: list, dim: int = DIM) -> bool:
    if len(v) != dim:
        return False
    if not all(isinstance(x, float) and -1.0 <= x <= 1.0 for x in v):
        return False
    return True
