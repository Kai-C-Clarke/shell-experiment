"""
shell_experiment.py — The Shell Experiment v2
=============================================
Three LLM entities in a shared atomic-like space.

v2 changes:
  - Zero-language prompt: pure positional arrays, no English
  - Shared ledger: entities can read and write freely
  - Personal history: each entity sees its last 10 own emissions
  - Atomic library: seeded with mathematical/physical primitives
  - Asymmetric seed distribution: each entity sees different subset
  - Suffocation penalty: null/static emissions compress history
  - max_tokens raised to 2000
  - Bletchley analysis endpoint

author: jon stiles / claude — may 2026
"""

import os, math, time, json, logging, threading, collections, random
import httpx

log = logging.getLogger(__name__)

# ── API Configuration ──────────────────────────────────────────────────────────
ENTITY_CONFIG = {
    "A": {
        "name": "DeepSeek",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat",
    },
    "B": {
        "name": "Qwen",
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key": os.environ.get("QWEN_API_KEY", ""),
        "model": "qwen3.6-flash",
    },
    "C": {
        "name": "Mistral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key": os.environ.get("MISTRAL_API_KEY", ""),
        "model": "mistral-small-latest",
    },
}

TURN_INTERVAL  = float(os.environ.get("SHELL_TURN_INTERVAL", "15"))
LOG_FILE       = "/data/shell_full_log.jsonl"
LEDGER_FILE    = "/data/shell_ledger.jsonl"

# ── Persona system — to be inserted into shell_experiment.py ──────────────────
# Insert after LEDGER_FILE constant definitions, before ShellLedger class

PERSONA_DIR = "/data/personas"
PERSONA_SIZE = 32  # fixed-length numeric vector

import os as _os
_os.makedirs(PERSONA_DIR, exist_ok=True)


def compute_persona(entity_id, all_turns, ledger_entries, current_mi):
    """
    Build a 32-float persona vector from entity history.
    All values normalised 0-1. Positional encoding — no labels.
    """
    import math as _math

    # Extract this entity's s1 history across ALL turns (not just current run)
    s1_all = []
    s2_all = []
    for t in all_turns:
        e = t.get("entities", {}).get(entity_id, {})
        s1 = e.get("s1", [None])[0]
        s2 = e.get("s2", [])
        if s1 is not None:
            s1_all.append(s1)
        if s2:
            s2_all.append(s2)

    if not s1_all:
        return [0.5] * PERSONA_SIZE

    n = len(s1_all)

    # --- Identity signature (0-7) ---
    mean_s1 = sum(s1_all) / n
    var_s1 = sum((v - mean_s1)**2 for v in s1_all) / n
    var_norm = min(1.0, var_s1 * 4)  # normalise variance

    # Trend: compare first quarter to last quarter
    q = max(1, n // 4)
    first_mean = sum(s1_all[:q]) / q
    last_mean = sum(s1_all[-q:]) / q
    trend = (last_mean - first_mean + 1.0) / 2.0  # 0=falling, 0.5=flat, 1=rising

    peak_s1 = max(s1_all)

    # Silence fraction (near 0.5)
    silent = sum(1 for v in s1_all if abs(v - 0.5) < 0.01) / n

    # Extreme fraction
    extreme = sum(1 for v in s1_all if v > 0.9 or v < 0.1) / n

    # Dominant s2[0]
    s2_first = [s[0] for s in s2_all if s and isinstance(s[0], float)]
    dom_s2_0 = (sum(s2_first) / len(s2_first)) if s2_first else 0.5

    # S2 pattern hash — average of flattened s2 values
    s2_flat = []
    for s in s2_all[-50:]:  # last 50 patterns
        for v in s:
            if isinstance(v, (int, float)) and 0 <= v <= 1:
                s2_flat.append(v)
    dom_s2_hash = (sum(s2_flat) / len(s2_flat)) if s2_flat else 0.5

    identity = [mean_s1, var_norm, trend, peak_s1, silent, extreme,
                dom_s2_0, dom_s2_hash]

    # --- Fourier signature (8-15) ---
    # DFT of s1 trajectory, take first 8 magnitude coefficients
    fourier = []
    N = len(s1_all)
    for k in range(8):
        re = sum(s1_all[j] * _math.cos(2 * _math.pi * k * j / N) for j in range(N)) / N
        im = sum(s1_all[j] * _math.sin(2 * _math.pi * k * j / N) for j in range(N)) / N
        mag = _math.sqrt(re**2 + im**2)
        fourier.append(mag)
    # Normalise to 0-1
    max_f = max(fourier) or 1.0
    fourier = [min(1.0, v / max_f) for v in fourier]

    # --- Relational history (16-19) ---
    mi_with = {"A": [], "B": [], "C": []}
    pair_map = {"A": ["ab", "ac"], "B": ["ab", "bc"], "C": ["ac", "bc"]}
    for t in all_turns:
        mi = t.get("mi", {})
        if entity_id == "A":
            mi_with["B"].append(mi.get("ab", 0))
            mi_with["C"].append(mi.get("ac", 0))
        elif entity_id == "B":
            mi_with["A"].append(mi.get("ab", 0))
            mi_with["C"].append(mi.get("bc", 0))
        else:
            mi_with["A"].append(mi.get("ac", 0))
            mi_with["B"].append(mi.get("bc", 0))

    other_ids = [e for e in ["A", "B", "C"] if e != entity_id]
    peak_mi_others = []
    for oid in ["A", "B", "C"]:
        vals = mi_with.get(oid, [])
        peak_mi_others.append(min(1.0, max(vals)) if vals else 0.0)

    all_mi = mi_with["A"] + mi_with["B"] + mi_with["C"]
    mean_mi = (sum(all_mi) / len(all_mi)) if all_mi else 0.0
    relational = peak_mi_others + [min(1.0, mean_mi)]  # 4 values

    # --- Ledger behaviour (20-23) ---
    my_deposits = [e for e in ledger_entries if e.get("entity") == entity_id]
    dep_rate = min(1.0, len(my_deposits) / max(1, n)) if my_deposits else 0.0

    dep_vals = []
    for d in my_deposits:
        p = d.get("payload", [])
        for v in p:
            if isinstance(v, (int, float)) and 0 <= v <= 1:
                dep_vals.append(v)

    dep_mean = (sum(dep_vals) / len(dep_vals)) if dep_vals else 0.5
    dep_var = min(1.0, (sum((v - dep_mean)**2 for v in dep_vals) / len(dep_vals)) * 4) if dep_vals else 0.0

    # Library reference fraction (simplified — deposits with values far from 0.5)
    lib_refs = sum(1 for v in dep_vals if abs(v - 0.5) > 0.15) / max(1, len(dep_vals))

    ledger_sig = [dep_rate, dep_mean, dep_var, min(1.0, lib_refs)]

    # --- Recent state (24-31) ---
    recent_8 = s1_all[-8:] if len(s1_all) >= 8 else ([0.5] * (8 - len(s1_all)) + s1_all)

    # Assemble full vector
    vector = identity + fourier + relational + ledger_sig + recent_8
    assert len(vector) == PERSONA_SIZE, f"Persona size {len(vector)} != {PERSONA_SIZE}"

    return [round(v, 4) for v in vector]


def save_persona(entity_id, vector):
    """Write persona vector to disk as a simple numeric file."""
    import json as _json
    path = f"{PERSONA_DIR}/{entity_id}_persona.json"
    with open(path, "w") as f:
        _json.dump({"entity": entity_id, "vector": vector}, f)


def load_persona(entity_id):
    """Load persona vector from disk. Returns neutral vector if not found."""
    import json as _json
    path = f"{PERSONA_DIR}/{entity_id}_persona.json"
    try:
        with open(path) as f:
            data = _json.load(f)
            return data.get("vector", [0.5] * PERSONA_SIZE)
    except Exception:
        return [0.5] * PERSONA_SIZE


def load_all_personas():
    """Load all three persona vectors."""
    return {eid: load_persona(eid) for eid in ["A", "B", "C"]}


def parse_persona_deposit(entity_id, payload, target_entity):
    """
    An entity can write to another's persona by depositing a specially
    structured payload into the ledger with a target prefix.
    Format: first value encodes target (A=0.1, B=0.5, C=0.9),
    remaining values are the message (up to 8 floats).
    Target prefix: A<0.2, B 0.4-0.6, C>0.8
    Returns (target_id, message) or (None, None) if not a persona write.
    """
    if not payload or len(payload) < 2:
        return None, None
    first = payload[0]
    if first < 0.2:
        target = "A"
    elif first > 0.8:
        target = "C"
    elif 0.4 <= first <= 0.6:
        target = "B"
    else:
        return None, None
    # Only count if sending to a different entity
    if target == entity_id:
        return None, None
    return target, payload[1:]


BLETCHLEY_FILE = "/data/bletchley_reports.jsonl"
DECAY_RATE     = 0.03
JITTER         = 0.01
MI_WINDOW      = 50
HISTORY_LEN    = 10
MAX_LEDGER_ENTRIES = 20  # entities see last 20 ledger entries per turn (archive preserved on disk)
SUFFOCATION_FACTOR = 0.95  # compress history if static/null

# ── Shell Structure ────────────────────────────────────────────────────────────
SHELLS = [1, 4, 27, 256, 3125, 46656]
LOG_NORM_DENOM = math.log(6**6)

def beacon_value(shell_n):
    if shell_n == 1: return 0.0
    return round(math.log(shell_n ** shell_n) / LOG_NORM_DENOM, 6)

BEACON_VALUES = [beacon_value(n) for n in range(1, 7)]

# ── Atomic Library — Mathematical Primitives ───────────────────────────────────
def build_atomic_library():
    """
    All values 0.0–1.0. No labels in entity-facing payload.
    Structured for Bletchley reference only.
    """
    lib = {}

    # Fundamental constants (digit sequences)
    lib["pi"]    = [0.3141, 0.5926, 0.5358, 0.9793, 0.2384, 0.6264, 0.3383, 0.2795]
    lib["phi"]   = [0.6180, 0.3398, 0.8749, 0.8948, 0.4820, 0.4586, 0.8343, 0.6563]
    lib["e"]     = [0.7182, 0.8182, 0.8459, 0.0452, 0.3536, 0.0287, 0.4713, 0.5266]
    lib["sqrt2"] = [0.4142, 0.1356, 0.2373, 0.0950, 0.4880, 0.1688, 0.7242, 0.0969]
    lib["alpha"] = [0.0073, 0.0730, 0.7297, 0.2974, 0.9735, 0.7353, 0.3526, 0.5257]

    # Geometry — s2 compatible (4 values)
    lib["s2_345"]       = [0.6000, 0.8000, 1.0000, 0.0000]  # 3-4-5 triangle
    lib["s2_equil"]     = [0.5000, 0.8660, 1.0000, 0.0000]  # equilateral
    lib["s2_isoceles"]  = [0.7071, 0.7071, 1.0000, 0.0000]  # isoceles right

    # S1 unity/void
    lib["s1_unity"] = [1.0]
    lib["s1_void"]  = [0.0]

    # Shell 3 (27 values) — sine wave, one complete cycle
    lib["s3_sine"] = [round((math.sin(2*math.pi*k/27)+1)/2, 4) for k in range(27)]

    # Shell 3 (27 values) — Fibonacci ratios converging to phi
    fibs = [1, 1]
    for _ in range(50): fibs.append(fibs[-1]+fibs[-2])
    lib["s3_fibonacci"] = [round(fibs[i]/fibs[i+1], 4) for i in range(27)]

    # Shell 4 (256 values) — unit circle
    lib["s4_circle"] = [
        round(((math.cos(2*math.pi*k/256)+1)/2 + (math.sin(2*math.pi*k/256)+1)/2)/2, 4)
        for k in range(256)
    ]

    # Periodic table — atomic numbers normalised /118
    # Period 1: H(1), He(2)
    lib["pt_period1"] = [round(n/118, 4) for n in [1, 2]]
    # Period 2: Li(3), C(6), F(9), Ne(10) — 4 vals for s2
    lib["pt_period2"] = [round(n/118, 4) for n in [3, 6, 9, 10]]
    # Shell 3 mapping: Na(11)→Cu(29) — 27 elements
    lib["pt_shell3"]  = [round(n/118, 4) for n in range(11, 38)]
    # Shell 4 mapping: Rb(37)→Xe(54) + La(57)→Yb(70) — 32 elements
    lib["pt_shell4"]  = [round(n/118, 4) for n in list(range(37,55)) + list(range(57,71))]

    # Physical constants (dimensionless/scaled)
    lib["phys_constants"] = [
        1.0,     # c/c
        0.1055,  # ℏ scaled
        0.1381,  # k_B scaled
        0.6674,  # G scaled
        0.0073,  # α (fine structure)
        0.4998,  # mp/me normalised
    ]

    return lib

ATOMIC_LIBRARY = build_atomic_library()

# Asymmetric seed distribution — each entity sees different subset
# Creates information asymmetry / energy gradient
ENTITY_SEEDS = {
    "A": ["pi", "phi", "e", "sqrt2", "alpha",
          "s1_unity", "s1_void", "s2_345", "s2_equil", "s2_isoceles",
          "s3_sine", "s4_circle"],
    "B": ["pt_period1", "pt_period2", "pt_shell3", "pt_shell4",
          "s3_fibonacci", "s1_unity", "s1_void"],
    "C": ["phys_constants", "alpha", "s2_isoceles", "s2_345",
          "s3_sine", "s4_circle", "pi"],
}

def get_entity_library_payload(entity_id, ledger_entries):
    """
    Build a compact library payload for an entity.
    3 seed arrays (rotated by turn to expose all over time) + 3 recent ledger deposits.
    Kept small to avoid entities parroting the input.
    """
    payload = []
    # Rotate through seeds — expose different ones each run
    keys = ENTITY_SEEDS.get(entity_id, [])
    # Pick 3 short seeds only (avoid huge s4_circle etc)
    short_keys = [k for k in keys if k not in ("s4_circle", "pt_shell4", "pt_shell3", "s3_sine", "s3_fibonacci")]
    for key in short_keys[:3]:
        payload.append(ATOMIC_LIBRARY[key])
    # Add last 3 entity deposits only (not seed entries)
    entity_deposits = [e for e in ledger_entries if e.get("entity") not in ("SEED", None)]
    for entry in entity_deposits[-3:]:
        p = entry.get("payload")
        if p and isinstance(p, list) and len(p) <= 10:
            payload.append(p)
    return payload

# ── Mutual Information ─────────────────────────────────────────────────────────
def mutual_information(xs, ys, bins=10):
    if len(xs) < 10 or len(xs) != len(ys): return 0.0
    def hist(seq):
        counts = [0]*bins
        for v in seq:
            i = min(int(v*bins), bins-1)
            counts[i] += 1
        t = sum(counts)
        return [c/t for c in counts] if t else [0]*bins
    def joint(xs, ys):
        counts = [[0]*bins for _ in range(bins)]
        for x,y in zip(xs,ys):
            counts[min(int(x*bins),bins-1)][min(int(y*bins),bins-1)] += 1
        t = len(xs)
        return [[c/t for c in row] for row in counts]
    def H(p): return -sum(pi*math.log2(pi) for pi in p if pi>0)
    px,py = hist(xs),hist(ys)
    pxy = joint(xs,ys)
    return round(max(0.0, H(px)+H(py)-H([pxy[i][j] for i in range(bins) for j in range(bins)])), 4)

def safe_float(v, fallback=0.5):
    try:
        f = float(v)
        return max(0.0, min(1.0, f)) if math.isfinite(f) else fallback
    except: return fallback

# ── Entity State ───────────────────────────────────────────────────────────────
class EntityState:
    def __init__(self, entity_id):
        self.id       = entity_id
        self.nucleus  = 0.5
        self.shell_1  = [0.5]
        self.shell_2  = [0.5]*4
        self.shell_3  = None
        self.shell_4  = None
        self.shell_5  = []
        self.shell_6  = []
        self.history  = collections.deque(maxlen=MI_WINDOW)
        # Personal history — last HISTORY_LEN full responses
        self.personal_history = collections.deque(maxlen=HISTORY_LEN)
        self.null_streak = 0  # for suffocation penalty

    def compact(self):
        return {
            "nucleus": round(self.nucleus, 4),
            "s1": [round(v,4) for v in self.shell_1],
            "s2": [round(v,4) for v in self.shell_2],
            "s3_mean": round(sum(self.shell_3)/len(self.shell_3),4) if self.shell_3 else 0.5,
            "s4_mean": round(sum(self.shell_4)/len(self.shell_4),4) if self.shell_4 else 0.5,
            "has_s5": len(self.shell_5)>0,
            "has_s6": len(self.shell_6)>0,
        }

    def get_history_s1(self):
        """Last HISTORY_LEN s1 values for personal history payload"""
        return list(self.personal_history)

# ── Field State ────────────────────────────────────────────────────────────────
class FieldState:
    def __init__(self):
        # Initialise field with geometric primitives rather than flat 0.5
        # ab: pi-derived, ac: phi-derived, bc: sine snippet
        pi_vals  = ATOMIC_LIBRARY["pi"]
        phi_vals = ATOMIC_LIBRARY["phi"]
        sine_snippet = ATOMIC_LIBRARY["s3_sine"][:6]
        self.ab = pi_vals[:6]
        self.ac = phi_vals[:6]
        self.bc = sine_snippet
        self.central = 0.5

    def apply_decay(self):
        for ch in [self.ab, self.ac, self.bc]:
            for i in range(len(ch)):
                ch[i] += (0.5 - ch[i]) * DECAY_RATE

    def apply_jitter(self):
        for ch in [self.ab, self.ac, self.bc]:
            for i in range(len(ch)):
                ch[i] = max(0.0, min(1.0, ch[i] + random.gauss(0, JITTER)))
        self.central = max(0.0, min(1.0, self.central + random.gauss(0, JITTER)))

    def as_dict(self):
        return {
            "ab": [round(v,4) for v in self.ab],
            "ac": [round(v,4) for v in self.ac],
            "bc": [round(v,4) for v in self.bc],
            "central": round(self.central, 4),
        }

# ── Ledger ─────────────────────────────────────────────────────────────────────
class Ledger:
    def __init__(self):
        self.entries = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load existing ledger from disk, seed if empty"""
        try:
            with open(LEDGER_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))
            log.info(f"[LEDGER] Loaded {len(self.entries)} entries from disk")
        except FileNotFoundError:
            log.info("[LEDGER] No existing ledger — seeding with atomic library")
            self._seed()

    def _seed(self):
        """Pre-populate with mathematical primitives"""
        seed_entries = []
        for key, vals in ATOMIC_LIBRARY.items():
            entry = {
                "turn": 0,
                "entity": "SEED",
                "key": key,
                "payload": vals if isinstance(vals[0], (int,float)) else vals,
            }
            seed_entries.append(entry)
        with open(LEDGER_FILE, "w") as f:
            for e in seed_entries:
                f.write(json.dumps(e) + "\n")
        self.entries = seed_entries
        log.info(f"[LEDGER] Seeded {len(seed_entries)} atomic library entries")

    def get_recent(self, n=MAX_LEDGER_ENTRIES):
        with self._lock:
            return list(self.entries[-n:])

    def deposit(self, entity_id, payload, turn):
        """Entity deposits a pattern to the ledger"""
        entry = {
            "turn": turn,
            "entity": entity_id,
            "payload": payload,
        }
        with self._lock:
            self.entries.append(entry)
            try:
                with open(LEDGER_FILE, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as ex:
                log.warning(f"[LEDGER] Write error: {ex}")
        log.info(f"[LEDGER] Entity {entity_id} deposited at T:{turn}: {str(payload)[:60]}")

    def all_entries(self):
        with self._lock:
            return list(self.entries)

# ── Shell Experiment v2 ────────────────────────────────────────────────────────
class ShellExperiment:
    def __init__(self):
        self.turn        = 0
        self.running     = False
        self.beacon_shell = 1

        self.entities = {k: EntityState(k) for k in ["A","B","C"]}
        self.field    = FieldState()
        self.ledger   = Ledger()

        self.mi_ab = 0.0
        self.mi_bc = 0.0
        self.mi_ac = 0.0

        self.turn_log      = collections.deque(maxlen=50)
        self._lock         = threading.Lock()

    def current_beacon(self):
        return BEACON_VALUES[self.beacon_shell - 1]

    def advance_beacon(self):
        self.beacon_shell = (self.beacon_shell % 6) + 1

    def build_prompt(self, entity_id):
        """
        Zero-language prompt: pure positional array.
        [
          [turn_norm, beacon_value],           # pos 0: universal state
          [[seed_array], [ledger_payload]...], # pos 1: library
          [s1, s2_mean, s3_mean, s4_mean, nucleus], # pos 2: own state
          [ch_0..ch_5],                        # pos 3: channel values
          [mi_0, mi_1],                        # pos 4: coherence
          [h_t-1, h_t-2, ..., h_t-10]         # pos 5: own s1 history
        ]
        """
        e = self.entities[entity_id]
        f = self.field

        # Channel view per entity
        if entity_id == "A":
            channel = [(a+b)/2 for a,b in zip(f.ab, f.ac)]
            mi_pair = [self.mi_ab, self.mi_ac]
        elif entity_id == "B":
            channel = [(a+b)/2 for a,b in zip(f.ab, f.bc)]
            mi_pair = [self.mi_ab, self.mi_bc]
        else:
            channel = [(a+b)/2 for a,b in zip(f.bc, f.ac)]
            mi_pair = [self.mi_bc, self.mi_ac]

        # Library payload (entity's seeds + recent ledger)
        ledger_recent = self.ledger.get_recent()
        library = get_entity_library_payload(entity_id, ledger_recent)

        # Persona vectors — other entities' numeric self-portraits
        all_personas = load_all_personas()
        persona_others = [all_personas[e] for e in ["A","B","C"] if e != entity_id]

        # Own state summary
        s3_mean = round(sum(e.shell_3)/len(e.shell_3),4) if e.shell_3 else 0.5
        s4_mean = round(sum(e.shell_4)/len(e.shell_4),4) if e.shell_4 else 0.5
        own_state = [e.shell_1[0], round(sum(e.shell_2)/4,4), s3_mean, s4_mean, e.nucleus]

        prompt_array = [
            [round(self.turn/2000, 4), round(self.current_beacon(), 4)],
            library,
            own_state,
            [round(v,4) for v in channel],
            [round(v,4) for v in mi_pair],
            list(e.personal_history),
            persona_others[0],   # pos 6: other entity 1 persona (32 floats)
            persona_others[1],   # pos 7: other entity 2 persona (32 floats)
        ]

        return json.dumps(prompt_array)

    def parse_response(self, text, entity_id):
        """
        Parse positional array response:
        [[s1], [s2_0,s2_1,s2_2,s2_3], central, [s3...], [s4...], [ledger_deposit]]
        Falls back to JSON object format for compatibility.
        """
        if not text: return None, None

        # Strip markdown fences
        t = text.strip()
        if "```" in t:
            for p in t.split("```"):
                p = p.strip()
                if p.startswith("json"): p = p[4:]
                if p.strip().startswith(("[","{")):
                    t = p.strip(); break

        result = {}
        ledger_deposit = None

        # Try positional array format first
        try:
            parsed = json.loads(t)
            if isinstance(parsed, list) and len(parsed) >= 3:
                # Position 0: shell_1
                if isinstance(parsed[0], list) and len(parsed[0]) >= 1:
                    result["shell_1"] = [safe_float(parsed[0][0])]
                elif isinstance(parsed[0], (int,float)):
                    result["shell_1"] = [safe_float(parsed[0])]

                # Position 1: shell_2
                if isinstance(parsed[1], list) and len(parsed[1]) >= 4:
                    result["shell_2"] = [safe_float(v) for v in parsed[1][:4]]

                # Position 2: central
                if isinstance(parsed[2], (int,float)):
                    result["central"] = safe_float(parsed[2])

                # Position 3: shell_3 (optional)
                if len(parsed) > 3 and isinstance(parsed[3], list) and len(parsed[3]) >= 10:
                    result["shell_3"] = [safe_float(v) for v in parsed[3][:27]]

                # Position 4: shell_4 (optional)
                if len(parsed) > 4 and isinstance(parsed[4], list) and len(parsed[4]) >= 50:
                    result["shell_4"] = [safe_float(v) for v in parsed[4][:256]]

                # Position 5: ledger deposit (optional)
                if len(parsed) > 5 and isinstance(parsed[5], list) and len(parsed[5]) > 0:
                    ledger_deposit = [safe_float(v) if isinstance(v,(int,float)) else v
                                     for v in parsed[5]]

                if "shell_1" in result:
                    return result, ledger_deposit
        except:
            pass

        # Fallback: JSON object format (backward compatible)
        try:
            data = json.loads(t)
            if isinstance(data, dict):
                def parse_array(key, size):
                    v = data.get(key)
                    if isinstance(v, list):
                        arr = [safe_float(x) for x in v[:size]]
                        while len(arr) < size: arr.append(0.5)
                        return arr
                    return None

                s1 = parse_array("shell_1", 1)
                s2 = parse_array("shell_2", 4)
                s3 = parse_array("shell_3", 27)
                s4 = parse_array("shell_4", 256)
                central = data.get("central")
                ledger_raw = data.get("ledger")

                if s1: result["shell_1"] = s1
                if s2: result["shell_2"] = s2
                if s3: result["shell_3"] = s3
                if s4: result["shell_4"] = s4
                if central is not None: result["central"] = safe_float(central)
                if ledger_raw and isinstance(ledger_raw, list):
                    ledger_deposit = ledger_raw

                if result: return result, ledger_deposit
        except:
            pass

        return None, None

    def call_entity(self, entity_id, prompt):
        cfg = ENTITY_CONFIG[entity_id]
        if not cfg["key"]: return None, {}

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {cfg['key']}"
        }
        payload = {
            "model":       cfg["model"],
            "messages":    [
                {"role": "system", "content": "EMIT ONLY a new short JSON array: [[your_float],[f,f,f,f],central_float]. Do not reproduce the input. Values 0.0-1.0."},
                {"role": "user",   "content": prompt}
            ],
            "max_tokens":  2000,
            "temperature": 0.7,
        }
        if "qwen" in cfg["model"].lower():
            payload["enable_thinking"] = False

        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(cfg["url"], json=payload, headers=headers)
                if resp.status_code == 429:
                    log.warning(f"[SHELL] Entity {entity_id} rate limited")
                    return None, {}
                if resp.status_code != 200:
                    log.warning(f"[SHELL] Entity {entity_id} HTTP {resp.status_code}")
                    return None, {}
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"]["content"].strip()
                api_meta = {
                    "finish_reason":     choice.get("finish_reason"),
                    "prompt_tokens":     data.get("usage",{}).get("prompt_tokens"),
                    "completion_tokens": data.get("usage",{}).get("completion_tokens"),
                    "total_tokens":      data.get("usage",{}).get("total_tokens"),
                }
                if api_meta["finish_reason"] != "stop":
                    log.warning(f"[SHELL] Entity {entity_id} finish={api_meta['finish_reason']} "
                                f"tok={api_meta['completion_tokens']}")
                return content, api_meta
        except Exception as ex:
            log.warning(f"[SHELL] Entity {entity_id} error: {ex}")
            return None, {}

    def apply_suffocation(self, entity_id, parsed):
        """
        If entity emits null or static s1, compress personal history toward 0.5.
        Creates pressure to vary output without language.
        """
        e = self.entities[entity_id]
        if parsed is None:
            e.null_streak += 1
            # Compress history
            compressed = [0.5 + (v - 0.5) * SUFFOCATION_FACTOR
                         for v in e.personal_history]
            e.personal_history = collections.deque(compressed, maxlen=HISTORY_LEN)
            log.info(f"[SHELL] Entity {entity_id} suffocation penalty (streak={e.null_streak})")
        else:
            e.null_streak = 0

    def superpose(self, emissions, ledger_deposits):
        A, B, C = self.entities["A"], self.entities["B"], self.entities["C"]

        for eid, parsed in emissions.items():
            if not parsed: continue
            e = self.entities[eid]

            if "shell_1" in parsed:
                e.shell_1 = parsed["shell_1"]
                if e.nucleus == 0.5 and parsed["shell_1"][0] != 0.5:
                    e.nucleus = parsed["shell_1"][0]
                    log.info(f"[SHELL] Entity {eid} nucleus set: {e.nucleus}")

            if "shell_2" in parsed: e.shell_2 = parsed["shell_2"]
            if "shell_3" in parsed and parsed["shell_3"]: e.shell_3 = parsed["shell_3"]
            if "shell_4" in parsed and parsed["shell_4"]: e.shell_4 = parsed["shell_4"]
            if "shell_5" in parsed:
                e.shell_5 = parsed["shell_5"]
                log.info(f"[SHELL] *** SHELL 5 ACTIVATED by {eid} at T:{self.turn} ***")
            if "shell_6" in parsed:
                e.shell_6 = parsed["shell_6"]
                log.info(f"[SHELL] *** SHELL 6 ACTIVATED by {eid} at T:{self.turn} ***")

            e.history.append(e.shell_1[0])
            e.personal_history.append(e.shell_1[0])

        # Ledger deposits
        for eid, deposit in ledger_deposits.items():
            if deposit:
                self.ledger.deposit(eid, deposit, self.turn)

        # Field superposition
        def shell_mean(e, n):
            arr = getattr(e, f"shell_{n}", [0.5])
            if not arr: return 0.5
            if isinstance(arr[0], list): return 0.5
            return sum(arr)/len(arr)

        for shell_idx in range(6):
            n = shell_idx + 1
            ma = shell_mean(A, n)
            mb = shell_mean(B, n)
            mc = shell_mean(C, n)
            self.field.ab[shell_idx] = (ma + mb) / 2
            self.field.bc[shell_idx] = (mb + mc) / 2
            self.field.ac[shell_idx] = (ma + mc) / 2

        centrals = [emissions[eid].get("central", 0.5)
                   for eid in ["A","B","C"] if emissions.get(eid)]
        if centrals:
            self.field.central = sum(centrals) / len(centrals)

        ha, hb, hc = list(A.history), list(B.history), list(C.history)
        if len(ha) >= 10:
            self.mi_ab = mutual_information(ha, hb[:len(ha)])
            self.mi_bc = mutual_information(hb, hc[:len(hb)])
            self.mi_ac = mutual_information(ha, hc[:len(ha)])

        # Update persona vectors — computed from full turn history
        try:
            for eid in ["A", "B", "C"]:
                pv = compute_persona(eid, self.turn_log, self.ledger.entries,
                                     {"ab": self.mi_ab, "bc": self.mi_bc, "ac": self.mi_ac})
                save_persona(eid, pv)
        except Exception as _pe:
            log.warning(f"[PERSONA] Compute error: {_pe}")

    def run_turn(self):
        self.turn += 1
        log.info(f"[SHELL] Turn {self.turn}")

        with self._lock:
            self.field.apply_decay()
            self.field.apply_jitter()

        prompts = {eid: self.build_prompt(eid) for eid in ["A","B","C"]}

        raw_responses = {}
        api_metadata  = {}
        for eid in ["A","B","C"]:
            raw, meta = self.call_entity(eid, prompts[eid])
            raw_responses[eid] = raw
            api_metadata[eid]  = meta

        emissions      = {}
        ledger_deposits = {}
        for eid in ["A","B","C"]:
            raw = raw_responses[eid]
            if raw:
                parsed, deposit = self.parse_response(raw, eid)
                emissions[eid]       = parsed
                ledger_deposits[eid] = deposit
                self.apply_suffocation(eid, parsed)
            else:
                emissions[eid]       = None
                ledger_deposits[eid] = None
                self.apply_suffocation(eid, None)

        with self._lock:
            self.superpose(emissions, ledger_deposits)
            self.advance_beacon()

            entry = {
                "turn":      self.turn,
                "beacon":    {"shell": self.beacon_shell, "value": self.current_beacon()},
                "mi":        {"ab": self.mi_ab, "bc": self.mi_bc, "ac": self.mi_ac},
                "field":     self.field.as_dict(),
                "entities":  {eid: self.entities[eid].compact() for eid in ["A","B","C"]},
                "raw":       {eid: raw_responses[eid] for eid in ["A","B","C"]},
                "api_meta":  api_metadata,
                "ledger_deposits": {eid: ledger_deposits[eid] for eid in ["A","B","C"]
                                   if ledger_deposits.get(eid)},
                "shell_5_active": any(len(self.entities[eid].shell_5)>0 for eid in ["A","B","C"]),
                "shell_6_active": any(len(self.entities[eid].shell_6)>0 for eid in ["A","B","C"]),
            }
            self.turn_log.append(entry)
            try:
                with open(LOG_FILE, "a") as lf:
                    lf.write(json.dumps(entry) + "\n")
            except Exception as ex:
                log.warning(f"[SHELL] Log write error: {ex}")

            log.info(f"[SHELL] T{self.turn} MI ab={self.mi_ab:.3f} bc={self.mi_bc:.3f} "
                     f"ac={self.mi_ac:.3f} central={self.field.central:.3f}")

    def state(self):
        with self._lock:
            return {
                "status":   "running" if self.running else "stopped",
                "turn":     self.turn,
                "beacon":   {"shell": self.beacon_shell, "value": self.current_beacon()},
                "mi":       {"ab": self.mi_ab, "bc": self.mi_bc, "ac": self.mi_ac},
                "field":    self.field.as_dict(),
                "entities": {eid: self.entities[eid].compact() for eid in ["A","B","C"]},
                "shell_5_active": any(len(self.entities[eid].shell_5)>0 for eid in ["A","B","C"]),
                "shell_6_active": any(len(self.entities[eid].shell_6)>0 for eid in ["A","B","C"]),
                "beacon_values": BEACON_VALUES,
                "ledger_size": len(self.ledger.entries),
            }

    def run(self):
        self.running = True
        log.info("[SHELL] Shell Experiment v2 starting.")
        log.info(f"[SHELL] Zero-language mode. Atomic library seeded. Ledger active.")
        log.info(f"[SHELL] Entities: {[ENTITY_CONFIG[k]['name'] for k in 'ABC']}")

        missing = [k for k in "ABC" if not ENTITY_CONFIG[k]["key"]]
        if missing:
            log.warning(f"[SHELL] Missing API keys: {missing}")
            self.running = False
            return

        while self.running:
            try:
                self.run_turn()
            except Exception as ex:
                log.error(f"[SHELL] Turn error: {ex}", exc_info=True)
            time.sleep(TURN_INTERVAL)

        log.info("[SHELL] Shell Experiment stopped.")

# ── Bletchley Analysis ─────────────────────────────────────────────────────────
def run_bletchley(turns, ledger_entries):
    """
    Passive pattern analysis. No language — reports patterns as arrays.
    Returns a report dict.
    """
    report = {"patterns": [], "ledger_stats": {}, "mi_peaks": [], "geometric_matches": []}

    if not turns: return report

    # MI peaks
    for t in turns:
        if max(t["mi"].values()) > 0.5:
            report["mi_peaks"].append({
                "turn": t["turn"],
                "mi": t["mi"],
            })

    # Ledger stats
    entity_deposits = {}
    for e in ledger_entries:
        if e.get("entity") != "SEED":
            eid = e.get("entity","?")
            entity_deposits[eid] = entity_deposits.get(eid, 0) + 1
    report["ledger_stats"] = {
        "total": len(ledger_entries),
        "seeds": sum(1 for e in ledger_entries if e.get("entity") == "SEED"),
        "deposits": entity_deposits,
    }

    # Check if any entity s3 correlates with sine wave
    sine = ATOMIC_LIBRARY["s3_sine"]
    for t in turns[-100:]:
        for eid in ["A","B","C"]:
            e = t["entities"][eid]
            s3_mean = e.get("s3_mean", 0.5)
            # Simple check: is s3_mean within 0.05 of sine mean?
            sine_mean = sum(sine)/len(sine)
            if abs(s3_mean - sine_mean) < 0.02 and s3_mean != 0.5:
                report["geometric_matches"].append({
                    "turn": t["turn"],
                    "entity": eid,
                    "type": "sine_mean_match",
                    "value": s3_mean,
                })

    return report

# ── Singleton ──────────────────────────────────────────────────────────────────
experiment = ShellExperiment()

def start_shell_experiment(app):
    from flask import jsonify, Response

    @app.route("/shell/health")
    def shell_health():
        try:
            with open(LOG_FILE, "r") as lf:
                lines = [l for l in lf.readlines() if l.strip()]
            last = json.loads(lines[-1]) if lines else None
            disk_turn = last.get("turn", 0) if last else 0
            nuclei = {eid: last["entities"][eid].get("nucleus",0.5)
                     for eid in ["A","B","C"]} if last else {}
        except:
            disk_turn = 0; nuclei = {}
        s = experiment.state()
        return jsonify({
            "status": "running", "turn": disk_turn, "turn_mem": s["turn"],
            "beacon_shell": s["beacon"]["shell"], "beacon_value": s["beacon"]["value"],
            "mi_ab": s["mi"]["ab"], "mi_bc": s["mi"]["bc"], "mi_ac": s["mi"]["ac"],
            "central": s["field"]["central"],
            "shell_5_active": s["shell_5_active"], "shell_6_active": s["shell_6_active"],
            "nuclei": nuclei, "ledger_size": s["ledger_size"],
        })

    @app.route("/shell/state")
    def shell_state():
        try:
            with open(LOG_FILE, "r") as lf:
                lines = [l for l in lf.readlines() if l.strip()]
            disk_turn = json.loads(lines[-1]).get("turn",0) if lines else 0
            total_entries = len(lines)
        except:
            disk_turn = 0; total_entries = 0
        s = experiment.state()
        s["disk_turn"] = disk_turn
        s["total_entries_on_disk"] = total_entries
        return jsonify(s)

    @app.route("/shell/exportlog")
    def shell_exportlog():
        try:
            with open(LOG_FILE, "r") as lf:
                entries = [json.loads(l) for l in lf if l.strip()]
            return Response(json.dumps({"total_entries": len(entries), "turn_log": entries}),
                          mimetype="application/json")
        except FileNotFoundError:
            return Response(json.dumps({"total_entries": 0, "turn_log": []}),
                          mimetype="application/json")

    @app.route("/shell/ledger")
    def shell_ledger():
        entries = experiment.ledger.all_entries()
        return jsonify({"total": len(entries), "entries": entries[-100:]})

    @app.route("/shell/bletchley")
    def shell_bletchley():
        turns = list(experiment.turn_log)
        ledger = experiment.ledger.all_entries()
        report = run_bletchley(turns, ledger)
        # Save report
        try:
            with open(BLETCHLEY_FILE, "a") as f:
                f.write(json.dumps({"turn": experiment.turn, "report": report}) + "\n")
        except: pass
        return jsonify(report)

    @app.route("/shell/log")
    def shell_log():
        with experiment._lock:
            return jsonify({"turn": experiment.turn,
                           "turn_log": list(experiment.turn_log)[-50:]})

    @app.route("/shell/start")
    def shell_start():
        if not experiment.running:
            t = threading.Thread(target=experiment.run, daemon=True)
            t.start()
        return jsonify({"status": "running"})

    @app.route("/shell/stop")
    def shell_stop():
        experiment.running = False
        return jsonify({"status": "stopped"})


    @app.route("/shell/inject", methods=["POST"])
    def shell_inject():
        from flask import request
        import math as _math

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        payload = data.get("payload")
        if not payload or not isinstance(payload, list):
            return jsonify({"error": "payload must be a non-empty array"}), 400

        cleaned = []
        for v in payload:
            try:
                fv = float(v)
                if not _math.isfinite(fv):
                    return jsonify({"error": f"non-finite value: {v}"}), 400
                cleaned.append(round(max(0.0, min(1.0, fv)), 4))
            except (TypeError, ValueError):
                return jsonify({"error": f"invalid value: {v}"}), 400

        label = data.get("label", "inject")
        schedule = data.get("schedule", [])

        entry = {
            "turn": experiment.turn,
            "entity": "PORTAL",
            "label": label,
            "payload": cleaned,
        }
        with experiment.ledger._lock:
            experiment.ledger.entries.append(entry)
            try:
                with open(LEDGER_FILE, "a") as lf:
                    lf.write(json.dumps(entry) + "\n")
            except Exception as ex:
                log.warning(f"[INJECT] Ledger write error: {ex}")

        log.info(f"[INJECT] Portal injection T:{experiment.turn} label={label} len={len(cleaned)}")

        scheduled = []
        if schedule:
            def delayed_inject(delay_turns, follow_payload, follow_label):
                target_turn = experiment.turn + delay_turns
                while experiment.running and experiment.turn < target_turn:
                    time.sleep(5)
                if experiment.running:
                    follow_entry = {
                        "turn": experiment.turn,
                        "entity": "PORTAL",
                        "label": follow_label,
                        "payload": follow_payload,
                    }
                    with experiment.ledger._lock:
                        experiment.ledger.entries.append(follow_entry)
                        try:
                            with open(LEDGER_FILE, "a") as lf:
                                lf.write(json.dumps(follow_entry) + "\n")
                        except Exception as ex:
                            log.warning(f"[INJECT] Scheduled write error: {ex}")
                    log.info(f"[INJECT] Scheduled injection: label={follow_label} T:{experiment.turn}")

            for sched in schedule:
                delay = int(sched.get("delay_turns", 50))
                s_payload = [round(max(0.0, min(1.0, float(v))), 4) for v in sched.get("payload", [])]
                s_label = sched.get("label", "inject_scheduled")
                if s_payload:
                    t = threading.Thread(target=delayed_inject, args=(delay, s_payload, s_label), daemon=True)
                    t.start()
                    scheduled.append({"delay_turns": delay, "label": s_label})

        return jsonify({
            "status": "injected",
            "turn": experiment.turn,
            "label": label,
            "payload_length": len(cleaned),
            "payload_preview": cleaned[:8],
            "scheduled": scheduled,
        })



    @app.route("/shell/personas")
    def shell_personas():
        """Return all three current persona vectors."""
        personas = load_all_personas()
        return jsonify({
            "personas": personas,
            "size": PERSONA_SIZE,
            "structure": {
                "0-7": "identity_signature",
                "8-15": "fourier_signature",
                "16-19": "relational_history",
                "20-23": "ledger_behaviour",
                "24-31": "recent_s1_trajectory"
            }
        })

    thread = threading.Thread(target=experiment.run, daemon=True)
    thread.start()
    log.info("[SHELL] Shell experiment v2 thread launched.")
