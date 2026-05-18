"""
analysis.py — Measurement layer.
Delta, cosine similarity, mutual information, heard-it scoring.
All results logged to SQLite. Injection impact is first-class.
"""
import math
import sqlite3
import json
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)

DB_PATH = os.environ.get("PORTAL_DB", "/tmp/portal.db")

DIM = 32
HEARD_IT_DELTA_THRESHOLD = 0.10
HEARD_IT_CORR_THRESHOLD  = 0.20


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT,
            turn        INTEGER,
            entity      TEXT,
            output      TEXT,   -- JSON array
            delta_norm  REAL,
            cosine_prev REAL,
            inj_corr    REAL,
            heard_it    INTEGER,
            timestamp   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS injections (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id  TEXT,
            turn    INTEGER,
            label   TEXT,
            itype   TEXT,
            vector  TEXT,   -- JSON array
            meta    TEXT    -- JSON
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS cross_entity (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT,
            turn        INTEGER,
            pair        TEXT,   -- AB, AC, BC
            cosine      REAL,
            mi_approx   REAL
        )
    """)
    conn.commit()
    conn.close()


def log_injection(run_id: str, turn: int, injection: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO injections (run_id, turn, label, itype, vector, meta) VALUES (?,?,?,?,?,?)",
        (run_id, turn, injection["label"], injection["type"],
         json.dumps(injection["vector"]), json.dumps(injection.get("meta", {})))
    )
    conn.commit()
    conn.close()


def log_turn(run_id: str, turn: int, entity: str, output: list,
             prev_output: list, injection_vec: list):
    delta_n = euclidean_distance(output, prev_output)
    cos_p   = cosine_similarity(output, prev_output)
    diff    = vector_diff(output, prev_output)
    inj_c   = cosine_similarity(diff, injection_vec) if any(injection_vec) else 0.0
    heard   = int(delta_n > HEARD_IT_DELTA_THRESHOLD and inj_c > HEARD_IT_CORR_THRESHOLD)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO turns
           (run_id, turn, entity, output, delta_norm, cosine_prev, inj_corr, heard_it, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, turn, entity, json.dumps(output),
         round(delta_n, 6), round(cos_p, 6), round(inj_c, 6),
         heard, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return {
        "delta_norm": round(delta_n, 4),
        "cosine_prev": round(cos_p, 4),
        "inj_corr": round(inj_c, 4),
        "heard_it": bool(heard)
    }


def log_cross_entity(run_id: str, turn: int, outputs: dict):
    """outputs = {"A": [...], "B": [...], "C": [...]}"""
    pairs = [("AB", "A", "B"), ("AC", "A", "C"), ("BC", "B", "C")]
    conn = sqlite3.connect(DB_PATH)
    for pair, e1, e2 in pairs:
        if e1 in outputs and e2 in outputs:
            cos = cosine_similarity(outputs[e1], outputs[e2])
            mi  = mi_approx(outputs[e1], outputs[e2])
            conn.execute(
                "INSERT INTO cross_entity (run_id, turn, pair, cosine, mi_approx) VALUES (?,?,?,?,?)",
                (run_id, turn, pair, round(cos, 6), round(mi, 6))
            )
    conn.commit()
    conn.close()


# ── Metrics ───────────────────────────────────────────────────────────────────

def euclidean_distance(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return round(dot / (na * nb), 6)


def vector_diff(a: list, b: list) -> list:
    return [x - y for x, y in zip(a, b)]


def mi_approx(a: list, b: list, bins: int = 8) -> float:
    """
    Approximate mutual information via histogram binning.
    Fast, good enough for real-time monitoring.
    """
    n = len(a)
    if n == 0:
        return 0.0

    def bin_idx(x, lo=-1.0, hi=1.0):
        return min(bins - 1, int((x - lo) / (hi - lo) * bins))

    joint  = [[0] * bins for _ in range(bins)]
    margin_a = [0] * bins
    margin_b = [0] * bins

    for x, y in zip(a, b):
        i = bin_idx(x)
        j = bin_idx(y)
        joint[i][j] += 1
        margin_a[i] += 1
        margin_b[j] += 1

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i][j] > 0 and margin_a[i] > 0 and margin_b[j] > 0:
                p_ij = joint[i][j] / n
                p_i  = margin_a[i] / n
                p_j  = margin_b[j] / n
                mi  += p_ij * math.log(p_ij / (p_i * p_j))
    return max(0.0, mi)


def get_summary(run_id: str) -> dict:
    """Summary stats for /portal/health endpoint"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT MAX(turn) FROM turns WHERE run_id=?", (run_id,))
    row = c.fetchone()
    max_turn = row[0] if row and row[0] else 0

    c.execute("""SELECT entity, AVG(delta_norm), AVG(inj_corr), SUM(heard_it)
                 FROM turns WHERE run_id=? GROUP BY entity""", (run_id,))
    entity_stats = {}
    for row in c.fetchall():
        entity_stats[row[0]] = {
            "avg_delta": round(row[1], 4),
            "avg_inj_corr": round(row[2], 4),
            "heard_count": row[3]
        }

    c.execute("""SELECT pair, AVG(cosine), MAX(mi_approx)
                 FROM cross_entity WHERE run_id=? GROUP BY pair""", (run_id,))
    cross_stats = {}
    for row in c.fetchall():
        cross_stats[row[0]] = {
            "avg_cosine": round(row[1], 4),
            "peak_mi": round(row[2], 4)
        }

    c.execute("SELECT turn, label FROM injections WHERE run_id=? ORDER BY turn", (run_id,))
    injections = [{"turn": r[0], "label": r[1]} for r in c.fetchall()]

    conn.close()
    return {
        "run_id": run_id,
        "max_turn": max_turn,
        "entities": entity_stats,
        "cross": cross_stats,
        "injections": injections
    }


def get_recent_turns(run_id: str, n: int = 20) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT turn, entity, delta_norm, cosine_prev, inj_corr, heard_it
                 FROM turns WHERE run_id=? ORDER BY id DESC LIMIT ?""", (run_id, n))
    rows = [{"turn": r[0], "entity": r[1], "delta": r[2],
             "cos_prev": r[3], "inj_corr": r[4], "heard_it": bool(r[5])}
            for r in c.fetchall()]
    conn.close()
    return rows
