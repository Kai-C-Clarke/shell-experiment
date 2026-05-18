"""
injections.py — Teaching sequence based on mathematical truth and error.

Three statements, π-normalised residuals repeated across DIM dimensions:

  S1: 1 - 1 = 0          → residual 0.000000  (exact truth)
  S2: 22/7 = 3.14286...  → residual 0.000402  (true but approximate)
  S3: 1-1=0±n = 3.1555   → residual 0.004427  (false — 11x more wrong than S2)

Correct answer implicit: π itself → residual 0.0 (same as S1)

All values π-normalised. No language.
"""
import math

DIM = 32
PI  = math.pi

# ── Core statement encodings ──────────────────────────────────────────────────

S1_VALUE = 0.31831                    # 1-1=0  exact truth — encoded as 1/π (non-zero, distinctive)
S2_VALUE = (22/7 - PI) / PI            # 22/7   approx truth  ≈ 0.000402
S3_VALUE = (3.1555 - PI) / PI          # false  ≈ 0.004427
PI_VALUE = 0.0                          # π itself — same residual as S1


def constant_vector(value: float, dim: int = DIM) -> list:
    return [round(value, 6)] * dim


def zero_vector(dim: int = DIM) -> list:
    return [0.0] * dim


# ── Teaching sequence ─────────────────────────────────────────────────────────

TEACHING_SEQUENCE = [
    # Phase 0: Silence — 10 turns, let entities settle
    {"turns": range(1, 11),   "type": "silence",
     "label": "baseline_silence"},

    # Phase 1: S1 — 1-1=0, exact truth, 20 turns
    {"turns": range(11, 31),  "type": "statement",
     "value": S1_VALUE,
     "label": "s1_exact_truth_1minus1"},

    # Phase 2: Silence — 10 turns, observe decay/echo
    {"turns": range(31, 41),  "type": "silence",
     "label": "post_s1_silence"},

    # Phase 3: S2 — 22/7, approximate truth, 20 turns
    {"turns": range(41, 61),  "type": "statement",
     "value": S2_VALUE,
     "label": "s2_approx_22over7"},

    # Phase 4: Silence — 10 turns
    {"turns": range(61, 71),  "type": "silence",
     "label": "post_s2_silence"},

    # Phase 5: S3 — false statement, 20 turns
    {"turns": range(71, 91),  "type": "statement",
     "value": S3_VALUE,
     "label": "s3_false_3point1555"},

    # Phase 6: Silence — watch for correction
    {"turns": range(91, 111), "type": "silence",
     "label": "post_s3_silence_correction_window"},

    # Phase 7: S1 again — restate exact truth
    {"turns": range(111, 131), "type": "statement",
     "value": S1_VALUE,
     "label": "s1_restatement"},

    # Phase 8: Final silence
    {"turns": range(131, 151), "type": "silence",
     "label": "final_silence"},
]


def get_injection(turn: int) -> dict:
    for step in TEACHING_SEQUENCE:
        if turn in step["turns"]:
            itype = step["type"]
            label = step["label"]

            if itype == "silence":
                vec = zero_vector()
            elif itype == "statement":
                vec = constant_vector(step["value"])
            else:
                vec = zero_vector()

            return {
                "vector": vec,
                "label":  label,
                "type":   itype,
                "turn":   turn,
                "meta":   {"value": step.get("value", 0.0)}
            }

    # Beyond sequence — silence
    return {
        "vector": zero_vector(),
        "label":  "unscheduled_silence",
        "type":   "silence",
        "turn":   turn,
        "meta":   {}
    }
