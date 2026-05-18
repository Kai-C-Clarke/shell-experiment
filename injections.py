"""
injections.py — Teaching sequence based on mathematical truth and error.

Sequence 1 (turns 1–150): Mathematical truth baseline
  S1: 1 - 1 = 0          → residual 0.000000  (exact truth)
  S2: 22/7 = 3.14286...  → residual 0.000402  (true but approximate)
  S3: 1-1=0±n = 3.1555   → residual 0.004427  (false — 11x more wrong than S2)

  Result: all three entities converged on 1/π (0.3183) unprompted after sequence ended.

Sequence 2 (turns 190–340): Circle area, r=1, A = π r²
  CA1: A = π             → residual 0.000000  (exact: π × 1²)
  CA2: A = 3.2           → residual 0.018592  (approximate π, plausible error)
  CA3: A = 4.0           → residual 0.273240  (false — area ≠ diameter²)

  The correct answer contains 1/π in the relationship. Entities are already there.
  All values π-normalised. No language.
"""
import math

DIM = 32
PI  = math.pi

# ── Sequence 1 encodings ──────────────────────────────────────────────────────

S1_VALUE  = 0.31831                 # 1/π — exact truth signal
S2_VALUE  = (22/7 - PI) / PI       # ≈ 0.000402
S3_VALUE  = (3.1555 - PI) / PI     # ≈ 0.004427

# ── Sequence 2 encodings — circle area, r=1 ──────────────────────────────────

CA1_VALUE = 0.0                     # A = π      — exact truth
CA2_VALUE = (3.2 - PI) / PI        # A = 3.2    — approx  ≈ 0.018592
CA3_VALUE = (4.0 - PI) / PI        # A = 4.0    — false   ≈ 0.273240


def constant_vector(value: float, dim: int = DIM) -> list:
    return [round(value, 6)] * dim


def zero_vector(dim: int = DIM) -> list:
    return [0.0] * dim


# ── Teaching sequence ─────────────────────────────────────────────────────────

TEACHING_SEQUENCE = [

    # ── Sequence 1 ───────────────────────────────────────────────────────────
    {"turns": range(1, 11),    "type": "silence",   "label": "baseline_silence"},
    {"turns": range(11, 31),   "type": "statement", "value": S1_VALUE,  "label": "s1_exact_truth_1minus1"},
    {"turns": range(31, 41),   "type": "silence",   "label": "post_s1_silence"},
    {"turns": range(41, 61),   "type": "statement", "value": S2_VALUE,  "label": "s2_approx_22over7"},
    {"turns": range(61, 71),   "type": "silence",   "label": "post_s2_silence"},
    {"turns": range(71, 91),   "type": "statement", "value": S3_VALUE,  "label": "s3_false_3point1555"},
    {"turns": range(91, 111),  "type": "silence",   "label": "post_s3_silence_correction_window"},
    {"turns": range(111, 131), "type": "statement", "value": S1_VALUE,  "label": "s1_restatement"},
    {"turns": range(131, 190), "type": "silence",   "label": "final_silence"},

    # ── Sequence 2: Circle area, r=1 ─────────────────────────────────────────
    {"turns": range(190, 200), "type": "silence",   "label": "pre_circle_silence"},
    {"turns": range(200, 220), "type": "statement", "value": CA1_VALUE, "label": "ca1_circle_area_exact"},
    {"turns": range(220, 230), "type": "silence",   "label": "post_ca1_silence"},
    {"turns": range(230, 250), "type": "statement", "value": CA2_VALUE, "label": "ca2_circle_area_approx_3point2"},
    {"turns": range(250, 260), "type": "silence",   "label": "post_ca2_silence"},
    {"turns": range(260, 280), "type": "statement", "value": CA3_VALUE, "label": "ca3_circle_area_false_4point0"},
    {"turns": range(280, 300), "type": "silence",   "label": "post_ca3_correction_window"},
    {"turns": range(300, 320), "type": "statement", "value": CA1_VALUE, "label": "ca1_restatement"},
    {"turns": range(320, 340), "type": "silence",   "label": "circle_final_silence"},
]


def get_injection(turn: int) -> dict:
    for step in TEACHING_SEQUENCE:
        if turn in step["turns"]:
            itype = step["type"]
            label = step["label"]
            vec   = constant_vector(step["value"]) if itype == "statement" else zero_vector()
            return {
                "vector": vec,
                "label":  label,
                "type":   itype,
                "turn":   turn,
                "meta":   {"value": step.get("value", 0.0)}
            }

    return {
        "vector": zero_vector(),
        "label":  "unscheduled_silence",
        "type":   "silence",
        "turn":   turn,
        "meta":   {}
    }
