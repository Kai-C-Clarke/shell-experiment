"""
memory.py — Persistence layer for entity and collective memory.

Individual:  /data/entity_history_{A,B,C}.json
Collective:  /data/collective_memory.json

All reads are best-effort — failures return empty and never crash the experiment.
All writes are atomic (tmp + os.replace), consistent with the existing codebase pattern.
"""
import os
import json
import logging
import threading

log = logging.getLogger(__name__)

DATA_DIR          = os.environ.get("PORTAL_DATA_DIR", "/data")
DIM               = 32
MAX_TURNS_PER_RUN = 200
MAX_RUNS_RETAINED = 3

CONVERGENCE_COSINE = 0.85   # all-pair cosine must exceed this
LARGE_DELTA_NORM   = 0.40   # any entity delta_norm must exceed this
HIGH_MI            = 0.50   # any cross-entity mi_approx must exceed this

_write_lock = threading.Lock()


# ── Path helpers ───────────────────────────────────────────────────────────────

def _entity_path(entity_id):
    return os.path.join(DATA_DIR, f"entity_history_{entity_id}.json")

def _collective_path():
    return os.path.join(DATA_DIR, "collective_memory.json")


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning(f"Failed to load {path}: {e}")
        return {}


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── Individual entity memory ───────────────────────────────────────────────────

def load_entity_history(entity_id, run_id):
    """
    Return vectors for this entity+run, newest-first.
    Returns [] if file missing, corrupt, or run_id not present.
    """
    data = _load_json(_entity_path(entity_id))
    entries = data.get("runs", {}).get(run_id, [])
    return [e["vector"] for e in reversed(entries)]


def append_entity_history(entity_id, run_id, turn, vector):
    """Append one turn. Caps at MAX_TURNS_PER_RUN per run. Atomic write. Silent on failure."""
    path = _entity_path(entity_id)
    with _write_lock:
        try:
            data = _load_json(path)
            runs = data.get("runs", {})
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append({"turn": turn, "vector": vector})
            if len(runs[run_id]) > MAX_TURNS_PER_RUN:
                runs[run_id] = runs[run_id][-MAX_TURNS_PER_RUN:]
            if len(runs) > MAX_RUNS_RETAINED:
                del runs[sorted(runs.keys())[0]]
            _atomic_write(path, {"entity": entity_id, "runs": runs})
        except Exception as e:
            log.error(f"Entity history write failed [{entity_id}]: {e}")


# ── Collective memory ──────────────────────────────────────────────────────────

def load_collective_memory(run_id, k=5):
    """
    Return up to k centroid vectors for this run, oldest-first (chronological order
    so entities see the arc of the session in CMEM:).
    Returns [] on any failure.
    """
    data = _load_json(_collective_path())
    entries = data.get("runs", {}).get(run_id, [])
    recent = entries[-k:] if len(entries) >= k else entries
    return [e["centroid"] for e in recent]


def append_collective_memory(run_id, turn, tag, centroid):
    """Append one significant event. Atomic write. Silent on failure."""
    path = _collective_path()
    with _write_lock:
        try:
            data = _load_json(path)
            runs = data.get("runs", {})
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append({"turn": turn, "tag": tag, "centroid": centroid})
            if len(runs) > MAX_RUNS_RETAINED:
                del runs[sorted(runs.keys())[0]]
            _atomic_write(path, {"runs": runs})
        except Exception as e:
            log.error(f"Collective memory write failed: {e}")


# ── Significance check ─────────────────────────────────────────────────────────

def is_significant(metrics, cross, injection):
    """
    Pure function. Returns (True, tag) if this turn qualifies for collective memory.

    metrics:   {"A": {"delta_norm": float, ...}, "B": {...}, "C": {...}}
    cross:     {"AB": {"cosine": float, "mi_approx": float}, ...}
    injection: the injection dict for this turn

    Priority: INJ > CON > DST > MUT (first match wins).
    """
    if injection.get("type") == "statement":
        return True, "INJ"

    cosines = [cross.get(p, {}).get("cosine", 0.0) for p in ("AB", "AC", "BC")]
    if cosines and min(cosines) > CONVERGENCE_COSINE:
        return True, "CON"

    if any(m.get("delta_norm", 0.0) > LARGE_DELTA_NORM for m in metrics.values()):
        return True, "DST"

    if any(cross.get(p, {}).get("mi_approx", 0.0) > HIGH_MI for p in ("AB", "AC", "BC")):
        return True, "MUT"

    return False, ""


def compute_centroid(outputs):
    """Mean vector of A, B, C outputs."""
    vecs = [outputs[e] for e in ("A", "B", "C") if e in outputs]
    if not vecs:
        return [0.0] * DIM
    n = len(vecs)
    return [round(sum(v[i] for v in vecs) / n, 6) for i in range(len(vecs[0]))]


def get_memory_snapshot(run_id, entity_history_copy):
    """Return a serializable snapshot for the /portal/memory endpoint."""
    collective_entries = []
    if run_id:
        data = _load_json(_collective_path())
        collective_entries = data.get("runs", {}).get(run_id, [])

    return {
        "entity_history": {
            e: {
                "count":  len(entity_history_copy.get(e, [])),
                "recent": entity_history_copy.get(e, [])[:5],
            }
            for e in ("A", "B", "C")
        },
        "collective_memory": {
            "count":   len(collective_entries),
            "entries": collective_entries[-10:],
        },
    }
