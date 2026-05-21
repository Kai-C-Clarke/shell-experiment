"""
injections.py — Teaching sequence based on mathematical truth and error.

Sequence 1 (turns 1-150): Mathematical truth baseline
  S1: 1 - 1 = 0          → residual 0.000000  (exact truth)
  S2: 22/7 = 3.14286...  → residual 0.000402  (true but approximate)
  S3: 1-1=0±n = 3.1555   → residual 0.004427  (false — 11x more wrong than S2)

  Result: all three entities converged on 1/π (0.3183) unprompted.

Sequence 2 (turns 190-340): Circle area, r=1, A = π r²
  CA1: A = π             → residual 0.000000  (exact)
  CA2: A = 3.2           → residual 0.018592  (approximate)
  CA3: A = 4.0           → residual 0.273240  (false)

Runtime override: if /data/sequence.json exists, get_injection() reads from it
instead of TEACHING_SEQUENCE. Update via POST /portal/sequence — no redeploy needed.

All values π-normalised. No language reaches the entities.
"""
import os
import json
import math

DIM = 32
PI  = math.pi

SEQUENCE_PATH = "/data/sequence.json"

# ── Sequence 1 encodings ──────────────────────────────────────────────────────

S1_VALUE  = 0.31831                 # 1/π
S2_VALUE  = (22/7 - PI) / PI       # ≈ 0.000402
S3_VALUE  = (3.1555 - PI) / PI     # ≈ 0.004427

# ── Sequence 2 encodings — circle area, r=1 ──────────────────────────────────

CA1_VALUE = 0.0                     # A = π      exact
CA2_VALUE = (3.2 - PI) / PI        # A = 3.2    approx  ≈ 0.018592
CA3_VALUE = (4.0 - PI) / PI        # A = 4.0    false   ≈ 0.273240


def constant_vector(value: float, dim: int = DIM) -> list:
    return [round(value, 6)] * dim


def zero_vector(dim: int = DIM) -> list:
    return [0.0] * dim


# ── Hardcoded fallback sequence ───────────────────────────────────────────────

TEACHING_SEQUENCE = [
    # Sequence 1
    {"turns": range(1,   11),  "type": "silence",   "label": "baseline_silence"},
    {"turns": range(11,  31),  "type": "statement",  "value": S1_VALUE,  "label": "s1_exact_truth_1minus1"},
    {"turns": range(31,  41),  "type": "silence",   "label": "post_s1_silence"},
    {"turns": range(41,  61),  "type": "statement",  "value": S2_VALUE,  "label": "s2_approx_22over7"},
    {"turns": range(61,  71),  "type": "silence",   "label": "post_s2_silence"},
    {"turns": range(71,  91),  "type": "statement",  "value": S3_VALUE,  "label": "s3_false_3point1555"},
    {"turns": range(91,  111), "type": "silence",   "label": "post_s3_silence_correction_window"},
    {"turns": range(111, 131), "type": "statement",  "value": S1_VALUE,  "label": "s1_restatement"},
    {"turns": range(131, 190), "type": "silence",   "label": "final_silence"},
    # Sequence 2
    {"turns": range(190, 200), "type": "silence",   "label": "pre_circle_silence"},
    {"turns": range(200, 220), "type": "statement",  "value": CA1_VALUE, "label": "ca1_circle_area_exact"},
    {"turns": range(220, 230), "type": "silence",   "label": "post_ca1_silence"},
    {"turns": range(230, 250), "type": "statement",  "value": CA2_VALUE, "label": "ca2_circle_area_approx_3point2"},
    {"turns": range(250, 260), "type": "silence",   "label": "post_ca2_silence"},
    {"turns": range(260, 280), "type": "statement",  "value": CA3_VALUE, "label": "ca3_circle_area_false_4point0"},
    {"turns": range(280, 300), "type": "silence",   "label": "post_ca3_correction_window"},
    {"turns": range(300, 320), "type": "statement",  "value": CA1_VALUE, "label": "ca1_restatement"},
    {"turns": range(320, 340), "type": "silence",   "label": "circle_final_silence"},
]


# ── Runtime sequence loader ───────────────────────────────────────────────────

def _load_runtime_sequence():
    """
    Load sequence from /data/sequence.json if it exists.
    Format: list of {turn_start, turn_end, type, label, value?}
    Returns None if file absent or malformed.
    """
    if not os.path.exists(SEQUENCE_PATH):
        return None
    try:
        with open(SEQUENCE_PATH, "r") as f:
            steps = json.load(f)
        # Validate minimal structure
        for s in steps:
            if "turn_start" not in s or "turn_end" not in s or "type" not in s:
                return None
            if s["turn_start"] >= s["turn_end"]:
                return None
            if s["type"] == "statement" and "value" not in s and "vector" not in s:
                return None
        return steps
    except Exception:
        return None


def get_injection(turn: int) -> dict:
    """
    Return the injection dict for this turn.
    Checks /data/sequence.json first (runtime override).
    Falls back to hardcoded TEACHING_SEQUENCE.
    Falls back to unscheduled silence beyond all steps.
    """
    # ── Runtime override ──────────────────────────────────────────────────────
    runtime = _load_runtime_sequence()
    if runtime is not None:
        for step in runtime:
            if step["turn_start"] <= turn < step["turn_end"]:
                itype = step["type"]
                label = step.get("label", "runtime_step")
                value = float(step.get("value", 0.0))
                # Support pre-computed zone vectors directly in sequence file
                if "vector" in step and isinstance(step["vector"], list) and len(step["vector"]) == DIM:
                    vec = [max(-1.0, min(1.0, float(x))) for x in step["vector"]]
                elif itype == "statement":
                    vec = constant_vector(value)
                else:
                    vec = zero_vector()
                return {
                    "vector": vec,
                    "label":  label,
                    "type":   itype,
                    "turn":   turn,
                    "meta":   {"value": value, "source": "runtime"},
                }
        # Runtime file exists but turn is beyond all steps
        return {
            "vector": zero_vector(),
            "label":  "runtime_silence",
            "type":   "silence",
            "turn":   turn,
            "meta":   {"source": "runtime"},
        }

    # ── Hardcoded fallback ────────────────────────────────────────────────────
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
                "meta":   {"value": step.get("value", 0.0), "source": "hardcoded"},
            }

    return {
        "vector": zero_vector(),
        "label":  "unscheduled_silence",
        "type":   "silence",
        "turn":   turn,
        "meta":   {"source": "hardcoded"},
    }
