"""
prompt_builder.py — Symbolic prompt template. Zero natural language in the loop.
The protocol IS the grammar. Consistency across turns teaches the contract.
"""
import math

DIM = 32


def fmt_vec(v: list) -> str:
    return "[" + " ".join(f"{x:.4f}" for x in v) + "]"


def fmt_matrix(rows: list) -> str:
    """List of vectors → semicolon-separated rows"""
    if not rows:
        return "[]"
    return ";".join(fmt_vec(r) for r in rows)


def get_few_shot_priming(dim: int = DIM) -> str:
    """
    3 examples: sinusoidal input → phase-shifted output.
    Pure format contract. No semantic content.
    Only sent on turn 1.
    """
    examples = []
    for i in range(3):
        phase_in = i * (2 * math.pi / 3)          # spread input phases evenly
        phase_out = phase_in + (math.pi / 4)       # fixed shift as output
        inp = [round(math.sin(2 * math.pi * k / dim + phase_in) / math.pi, 4)
               for k in range(dim)]
        out = [round(math.sin(2 * math.pi * k / dim + phase_out) / math.pi, 4)
               for k in range(dim)]
        examples.append(f"EX{i+1}\nINP:{fmt_vec(inp)}\nOUT:{fmt_vec(out)}")
    return "\n\n".join(examples)


def build_prompt(entity_id: str,
                 round_num: int,
                 shared: list,        # list of up to 3 entity output vectors
                 self_hist: list,     # last 2 own outputs
                 delta: list,         # change from prev own output
                 injection: list,     # current teaching vector
                 dim: int = DIM,
                 priming: bool = False) -> str:
    """
    Build the symbolic prompt for one entity, one turn.
    All sections are pure numbers. No English after priming block.
    """
    lines = []

    if priming:
        lines.append(get_few_shot_priming(dim))
        lines.append("")

    lines.append(f"PROTOCOL:1.0")
    lines.append(f"ENT:{entity_id} RND:{round_num} DIM:{dim} RNG:[-1 1]")
    lines.append("")
    lines.append(f"SHR:{fmt_matrix(shared)}")
    lines.append(f"SLF:{fmt_matrix(self_hist) if self_hist else fmt_vec([0.0]*dim)}")
    lines.append(f"DLT:{fmt_vec(delta)}")
    lines.append("")
    lines.append(f"INP:{fmt_vec(injection)}")
    lines.append("")
    lines.append("WARN:DO_NOT_COPY_EXAMPLES")
    lines.append("OUT:")

    return "\n".join(lines)


def build_system_prompt() -> str:
    """
    Minimal system prompt — sent once, defines the contract.
    As close to zero language as possible while ensuring compliance.
    """
    return (
        "You are a numerical state processor. "
        "Each turn: read INP and SHR arrays, update your internal state, "
        "output exactly one array of DIM floats on the OUT: line. "
        "Format: OUT:[x1 x2 ... xN]. No other text. "
        "Do NOT output placeholder, example, or template values. "
        "Compute your output from the actual INP and SHR values provided."
    )
