"""
portal_experiment.py — Main orchestrator.
Vector dynamical system. Three LLMs. Zero language in the loop.
Teaching via scheduled Fourier injections.
"""
import os
import json
import time
import uuid
import logging
import threading
import httpx
from datetime import datetime

from injections import get_injection, DIM
from prompt_builder import build_prompt, build_system_prompt
from parser import strict_parse_output, make_retry_prompt, fallback_vector
from analysis import (init_db, log_injection, log_turn, log_cross_entity,
                      get_summary, get_recent_turns)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TURN_INTERVAL  = int(os.environ.get("PORTAL_TURN_INTERVAL", os.environ.get("SHELL_TURN_INTERVAL", 30)))  # seconds
MAX_RETRIES    = 3
SHARED_HISTORY = 3   # how many past outputs in SHR block
SELF_HISTORY   = 2   # how many own past outputs in SLF block

ENTITIES = {
    "A": {
        "name": "DeepSeek",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model":   "deepseek-chat",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    },
    "B": {
        "name": "Qwen",
        "api_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model":   "qwen3.6-flash",
        "api_key": os.environ.get("QWEN_API_KEY", ""),
    },
    "C": {
        "name": "Mistral",
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "model":   "mistral-small-latest",
        "api_key": os.environ.get("MISTRAL_API_KEY", ""),
    },
}

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    "run_id":    None,
    "turn":      0,
    "status":    "stopped",
    "outputs":   {"A": [0.0] * DIM, "B": [0.0] * DIM, "C": [0.0] * DIM},
    "history":   {"A": [], "B": [], "C": []},
    "last_inj":  None,
    "last_metrics": {},
    "errors":    [],
}
lock = threading.Lock()
_thread = None


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(entity_id: str, prompt: str, retry_num: int = 0) -> str:
    cfg = ENTITIES[entity_id]
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  120,
        "temperature": 0.05,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(cfg["api_url"], headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"LLM call failed [{entity_id}] attempt {retry_num}: {e}")
        raise


def get_entity_output(entity_id: str, prompt: str, turn: int) -> list:
    """Call LLM, parse, retry up to MAX_RETRIES times"""
    last_err = "unknown"
    for attempt in range(MAX_RETRIES):
        try:
            if attempt == 0:
                raw = call_llm(entity_id, prompt)
            else:
                retry_p = make_retry_prompt(turn, DIM)
                raw = call_llm(entity_id, retry_p, retry_num=attempt)
            return strict_parse_output(raw, DIM)
        except ValueError as e:
            log.warning(f"Parse fail [{entity_id}] attempt {attempt+1}: {e}")
        except Exception as e:
            err_msg = str(e)
            log.warning(f"LLM error [{entity_id}] attempt {attempt+1}: {err_msg}")
            time.sleep(2)
            last_err = err_msg

    log.error(f"All retries failed for [{entity_id}] turn {turn} — using fallback")
    with lock:
        state["errors"].append({
            "turn": turn, "entity": entity_id,
            "msg": last_err if 'last_err' in dir() else "all retries failed",
            "ts": datetime.utcnow().isoformat()
        })
    return fallback_vector(DIM)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop():
    with lock:
        state["status"] = "running"
        state["run_id"] = f"portal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        state["turn"] = 0
        state["outputs"] = {"A": [0.0]*DIM, "B": [0.0]*DIM, "C": [0.0]*DIM}
        state["history"] = {"A": [], "B": [], "C": []}
        state["errors"] = []
        run_id = state["run_id"]

    init_db()
    log.info(f"Portal experiment started — run_id={run_id}")

    while True:
        with lock:
            if state["status"] != "running":
                break
            turn = state["turn"] + 1
            state["turn"] = turn
            prev_outputs = {e: list(v) for e, v in state["outputs"].items()}
            history = {e: list(h) for e, h in state["history"].items()}

        # Get injection for this turn
        injection = get_injection(turn)
        log_injection(run_id, turn, injection)
        inj_vec = injection["vector"]

        with lock:
            state["last_inj"] = {"turn": turn, "label": injection["label"]}

        # Build shared block — last outputs of all three entities
        shared = [prev_outputs[e] for e in ["A", "B", "C"]]

        new_outputs = {}
        metrics = {}

        for entity_id in ["A", "B", "C"]:
            self_hist = history[entity_id][-SELF_HISTORY:]
            delta = [a - b for a, b in zip(
                prev_outputs[entity_id],
                (self_hist[-1] if self_hist else [0.0] * DIM)
            )]

            prompt = build_prompt(
                entity_id=entity_id,
                round_num=turn,
                shared=shared,
                self_hist=self_hist,
                delta=delta,
                injection=inj_vec,
                dim=DIM,
                priming=(turn == 1)
            )

            output = get_entity_output(entity_id, prompt, turn)
            new_outputs[entity_id] = output

            m = log_turn(run_id, turn, entity_id, output,
                         prev_outputs[entity_id], inj_vec)
            metrics[entity_id] = m

            log.info(
                f"T:{turn} {entity_id} delta={m['delta_norm']:.4f} "
                f"inj_corr={m['inj_corr']:.4f} heard={m['heard_it']}"
            )

        log_cross_entity(run_id, turn, new_outputs)

        with lock:
            state["outputs"] = new_outputs
            for e in ["A", "B", "C"]:
                state["history"][e].append(new_outputs[e])
                if len(state["history"][e]) > 20:
                    state["history"][e] = state["history"][e][-20:]
            state["last_metrics"] = metrics

        time.sleep(TURN_INTERVAL)

    log.info(f"Portal experiment loop ended — run_id={run_id}")


# ── Flask endpoint registration ───────────────────────────────────────────────

def start_portal_experiment(app):
    from flask import jsonify, request

    @app.route("/portal/health")
    def portal_health():
        with lock:
            run_id = state["run_id"]
            turn   = state["turn"]
            status = state["status"]
            last_inj  = state["last_inj"]
            last_metrics = state["last_metrics"]
        summary = get_summary(run_id) if run_id else {}
        return jsonify({
            "status":       status,
            "run_id":       run_id,
            "turn":         turn,
            "last_injection": last_inj,
            "last_metrics": last_metrics,
            "summary":      summary,
        })

    @app.route("/portal/state")
    def portal_state():
        with lock:
            return jsonify({
                "run_id":  state["run_id"],
                "turn":    state["turn"],
                "status":  state["status"],
                "outputs": {e: state["outputs"][e] for e in ["A","B","C"]},
            })

    @app.route("/portal/log")
    def portal_log():
        with lock:
            run_id = state["run_id"]
        n = int(request.args.get("n", 30))
        return jsonify(get_recent_turns(run_id, n) if run_id else [])

    @app.route("/portal/inject", methods=["POST"])
    def portal_inject():
        """Manual injection endpoint — accepts {vector: [...], label: str}"""
        data = request.get_json()
        if not data or "vector" not in data:
            return jsonify({"error": "missing vector"}), 400
        vec = data["vector"]
        if len(vec) != DIM:
            return jsonify({"error": f"vector must be {DIM} floats"}), 400
        label = data.get("label", "manual_inject")
        with lock:
            run_id = state["run_id"]
            turn   = state["turn"]
        if run_id:
            inj = {"vector": vec, "label": label, "type": "manual",
                   "turn": turn, "meta": {}}
            log_injection(run_id, turn, inj)
        return jsonify({"ok": True, "label": label, "turn": turn})

    @app.route("/portal/start", methods=["POST"])
    def portal_start():
        global _thread
        with lock:
            if state["status"] == "running":
                return jsonify({"error": "already running"}), 400
        _thread = threading.Thread(target=run_loop, daemon=True)
        _thread.start()
        return jsonify({"ok": True, "msg": "portal experiment started"})

    @app.route("/portal/stop", methods=["POST"])
    def portal_stop():
        with lock:
            state["status"] = "stopped"
        return jsonify({"ok": True, "msg": "stopping after current turn"})

    @app.route("/portal/errors")
    def portal_errors():
        with lock:
            return jsonify(state["errors"])

    log.info("Portal experiment routes registered — /portal/* active")
