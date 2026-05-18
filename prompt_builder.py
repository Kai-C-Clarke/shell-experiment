"""
prompt_builder.py — Symbolic prompt template. Zero natural language in the loop.
Prompts kept SHORT — no few-shot priming blocks, no long history.
System prompt establishes the contract once.
"""
import math

DIM = 32


def fmt_vec(v: list) -> str:
    return "[" + " ".join(f"{x:.4f}" for x in v) + "]"


def fmt_matrix(rows: list) -> str:
    if not rows:
        return "[]"
    return ";".join(fmt_vec(r) for r in rows[-3:])  # max 3 rows


def build_prompt(entity_id: str,
                 round_num: int,
                 shared: list,
                 self_hist: list,
                 delta: list,
                 injection: list,
                 dim: int = DIM,
                 priming: bool = False) -> str:
    """
    Minimal symbolic prompt. Fits comfortably in 500 tokens.
    priming flag kept for API compatibility but no longer adds a block.
    """
    # Only include last own output, not full history
    last_self = self_hist[-1] if self_hist else [0.0] * dim

    lines = [
        f"P:1.0 E:{entity_id} R:{round_num} D:{dim}",
        f"SHR:{fmt_matrix(shared)}",
        f"SLF:{fmt_vec(last_self)}",
        f"DLT:{fmt_vec(delta)}",
        f"INP:{fmt_vec(injection)}",
        "OUT:",
    ]
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "Numerical state processor. "
        "Each turn: read INP array, update state, "
        f"output exactly {DIM} floats. "
        "Format: OUT:[x1 x2 ... xN]. No other text."
    )
