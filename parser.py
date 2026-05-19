"""
parser.py — Strict numerical output enforcement.
Parse → validate → retry on failure. No text passes through.

Fix (19 May 2026): Parser was matching the EXAMPLE vector in the retry prompt
rather than the actual OUT: response. Fixed by:
  1. Anchoring OUT: match to the LAST occurrence in the string
  2. Removing numerical example from retry prompt
  3. Adding raw response logging for debugging
  4. Bare array fallback now only triggers if no OUT: line found at all
"""
import re
import logging

log = logging.getLogger(__name__)

DIM = 32
MAX_RETRIES = 3

# Store last raw responses for /portal/raw_last_response endpoint
_last_raw = {}


def record_raw(entity_id: str, raw: str):
    """Store raw LLM response for debugging."""
    _last_raw[entity_id] = raw[-800:]  # last 800 chars


def get_last_raw() -> dict:
    return dict(_last_raw)


def strict_parse_output(raw: str, dim: int = DIM) -> list:
    """
    Extract exactly DIM floats from model output.
    Anchors to the LAST OUT: line to avoid matching prompt examples.
    Raises ValueError if all patterns fail.
    """
    record_raw("_last", raw)

    # Split on OUT: and take the LAST segment — that's the actual response
    # This prevents matching any OUT: example earlier in the prompt echo
    segments = re.split(r"OUT\s*:", raw, flags=re.IGNORECASE)
    
    if len(segments) > 1:
        # Take the last segment after OUT:
        candidate = segments[-1].strip()
        
        # Try bracketed format first: [1.23 4.56 ...]
        bracket_match = re.match(r"\[([^\]]+)\]", candidate)
        if bracket_match:
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", bracket_match.group(1))
            if len(numbers) == dim:
                return [max(-1.0, min(1.0, float(x))) for x in numbers]
            log.warning(f"Bracket match found but got {len(numbers)} numbers, need {dim}")

        # Try bare floats on the line (no brackets)
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate.split("\n")[0])
        if len(numbers) == dim:
            return [max(-1.0, min(1.0, float(x))) for x in numbers]
        log.warning(f"Bare float match got {len(numbers)} numbers, need {dim}")

    # Last resort: JSON-style {"out": [...]}
    json_match = re.search(
        r"\{[^}]*[\"']?out[\"']?\s*:\s*\[([^\]]+)\]",
        raw, re.IGNORECASE
    )
    if json_match:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", json_match.group(1))
        if len(numbers) == dim:
            return [max(-1.0, min(1.0, float(x))) for x in numbers]

    log.warning(f"Parse failed. Raw tail: {raw[-200:]!r}")
    raise ValueError(f"Parse failed (dim={dim}). Raw: {raw[-300:]}")


def make_retry_prompt(round_num: int, dim: int = DIM) -> str:
    """
    Stronger formatting hint on retry.
    No numerical example — avoids parser matching the example instead of output.
    """
    return (
        f"RND:{round_num} PARSE_FAIL\n"
        f"REQUIRED: OUT:[x1 x2 ... x{dim}] ({dim} floats in [-1,1])\n"
        f"OUT:"
    )


def fallback_vector(dim: int = DIM) -> list:
    """Emergency fallback — silence. Used if all retries fail."""
    log.warning("All parse retries failed — using silence fallback")
    return [0.0] * dim


def validate_vector(v: list, dim: int = DIM) -> bool:
    if len(v) != dim:
        return False
    if not all(isinstance(x, float) and -1.0 <= x <= 1.0 for x in v):
        return False
    return True
