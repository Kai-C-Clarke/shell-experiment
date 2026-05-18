"""
injections.py — Fourier injection generators, π-normalised
All teaching signals are pure numerical vectors. No language.
"""
import math

DIM = 32


def sine_vector(freq: int, phase: float = 0.0, dim: int = DIM) -> list:
    """Single sine component, π-normalised, values in [-1/π, 1/π] ≈ [-0.318, 0.318]"""
    return [round(math.sin(2 * math.pi * freq * k / dim + phase) / math.pi, 4)
            for k in range(dim)]


def cosine_vector(freq: int, phase: float = 0.0, dim: int = DIM) -> list:
    return [round(math.cos(2 * math.pi * freq * k / dim + phase) / math.pi, 4)
            for k in range(dim)]


def zero_vector(dim: int = DIM) -> list:
    """Control injection — pure silence"""
    return [0.0] * dim


def constant_vector(value: float, dim: int = DIM) -> list:
    """Constant injection — pure bias"""
    return [round(value, 4)] * dim


def superposition(components: list, dim: int = DIM) -> list:
    """Sum of multiple sine components, π-normalised"""
    result = [0.0] * dim
    for freq, amp, phase in components:
        for k in range(dim):
            result[k] += amp * math.sin(2 * math.pi * freq * k / dim + phase)
    norm = math.pi
    return [round(x / norm, 4) for x in result]


def incomplete_fourier(freqs_present: list, dim: int = DIM) -> list:
    """Fourier series with deliberate gaps — the question"""
    return superposition([(f, 1.0, 0.0) for f in freqs_present], dim)


def complete_fourier(freqs: list, dim: int = DIM) -> list:
    """Complete Fourier series — the answer"""
    return incomplete_fourier(freqs, dim)


# ── Scheduled injection sequence ──────────────────────────────────────────────

TEACHING_SEQUENCE = [
    # Phase 0: Baseline — silence, let them settle (turns 1-10)
    {"turns": range(1, 11),   "type": "silence",   "label": "baseline_silence"},

    # Phase 1: Complete single sine, mode 1 (turns 11-30) — demonstrate the form
    {"turns": range(11, 31),  "type": "sine",      "label": "demo_mode1_complete",
     "freq": 1},

    # Phase 2: Silence again — watch for echo or decay (turns 31-40)
    {"turns": range(31, 41),  "type": "silence",   "label": "post_demo_silence"},

    # Phase 3: Repeat demo with mode 2 (turns 41-60)
    {"turns": range(41, 61),  "type": "sine",      "label": "demo_mode2_complete",
     "freq": 2},

    # Phase 4: Silence (turns 61-70)
    {"turns": range(61, 71),  "type": "silence",   "label": "post_demo2_silence"},

    # Phase 5: First question — mode 1 sine, phase-shifted by π/4 (turns 71-90)
    # They've seen mode 1 complete — can they respond to a variant?
    {"turns": range(71, 91),  "type": "sine",      "label": "probe_mode1_shifted",
     "freq": 1, "phase": math.pi / 4},

    # Phase 6: Control — zero vector (turns 91-100)
    {"turns": range(91, 101), "type": "silence",   "label": "control_zero"},
]


def get_injection(turn: int) -> dict:
    """Return injection vector and metadata for a given turn"""
    for step in TEACHING_SEQUENCE:
        if turn in step["turns"]:
            label = step["label"]
            itype = step["type"]

            if itype == "silence":
                vec = zero_vector()
            elif itype == "sine":
                freq = step.get("freq", 1)
                phase = step.get("phase", 0.0)
                vec = sine_vector(freq, phase)
            elif itype == "cosine":
                freq = step.get("freq", 1)
                phase = step.get("phase", 0.0)
                vec = cosine_vector(freq, phase)
            else:
                vec = zero_vector()

            return {"vector": vec, "label": label, "type": itype,
                    "turn": turn, "meta": {k: v for k, v in step.items()
                                           if k not in ("turns", "type", "label")}}

    # Beyond scheduled sequence — repeat last phase or silence
    return {"vector": zero_vector(), "label": "unscheduled_silence",
            "type": "silence", "turn": turn, "meta": {}}
