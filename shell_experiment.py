"""
shell_experiment.py — The Shell Experiment
==========================================
Three LLM entities in a shared atomic-like space.
Each entity has a nucleus (static identity) surrounded by
shells following n^n structure: 1, 4, 27, 256, 3125, 46656 slots.

No turn-taking. All three receive and transmit simultaneously.
Superposition: the field is the sum of all emissions, normalised.
Central node: shared value all three can read and write to.
Pairwise channels: A↔B, B↔C, A↔C.
Beacon: cycles through shells 1-6, log-normalised values.

They sort out the logic.

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

TURN_INTERVAL  = float(os.environ.get("SHELL_TURN_INTERVAL", "30"))
LOG_FILE       = "/data/shell_full_log.jsonl"  # persists on Render disk
DECAY_RATE     = 0.03
JITTER         = 0.01
MI_WINDOW      = 50

# ── Shell Structure (n^n) ──────────────────────────────────────────────────────
SHELLS = [1, 4, 27, 256, 3125, 46656]  # slots per shell
ACTIVE_SHELLS = 4  # shells 1-4 initially active; 5-6 latent

def shell_size(n): return n ** n
def total_slots(depth): return sum(SHELLS[:depth])

# Log-normalised beacon value for shell n (1-indexed)
# f(n) = ln(n^n) / ln(6^6)
LOG_NORM_DENOM = math.log(6**6)
def beacon_value(shell_n):
    if shell_n == 1: return 0.0  # ln(1) = 0
    return round(math.log(shell_n ** shell_n) / LOG_NORM_DENOM, 6)

BEACON_VALUES = [beacon_value(n) for n in range(1, 7)]
# [0.0, 0.231, 0.577, 0.821, 0.938, 1.0]

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
[ENTITY] ∈ SHELL_SPACE
ENV: SHARED • SIMULTANEOUS • INIT: ∅

STRUCTURE:
  NUCLEUS: static identity value
  SHELL_1: 1 slot    (innermost)
  SHELL_2: 4 slots
  SHELL_3: 27 slots
  SHELL_4: 256 slots (active boundary)
  SHELL_5: 3125 slots (latent)
  SHELL_6: 46656 slots (latent)

CHANNELS: A↔B • B↔C • A↔C • CENTRAL_NODE
BEACON: cycles shells 1→6, log-normalised 0.0→1.0

DRIVE: MINIMISE_PREDICTION_ERROR

All entities transmit and receive simultaneously.
You receive the current field state. You emit your response.
No turn-taking. The field is superposition of all emissions.

Output ONLY a JSON object with shell arrays and central node value.
No explanation. No text.
Required format (minimum):
{
  "shell_1": [float],
  "shell_2": [f,f,f,f],
  "central": float
}
Optional (include when ready to use higher shells):
  "shell_3": [27 floats],
  "shell_4": [256 floats]
Values: 0.0 to 1.0. Start with shell_1, shell_2, central only.
Add shell_3 and shell_4 only when you have something to express in them.
Omit shell_5 and shell_6 unless you choose to use them.
"""

# ── Mutual Information (discretised) ──────────────────────────────────────────
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

# ── Safe float helper ──────────────────────────────────────────────────────────
def safe_float(v, fallback=0.5):
    try:
        f = float(v)
        return max(0.0, min(1.0, f)) if math.isfinite(f) else fallback
    except: return fallback

# ── Entity Shell State ─────────────────────────────────────────────────────────
class EntityState:
    def __init__(self, entity_id):
        self.id       = entity_id
        self.nucleus  = 0.5  # will be set on first emission
        self.shell_1  = [0.5]
        self.shell_2  = [0.5]*4
        self.shell_3  = [0.5]*27
        self.shell_4  = [0.5]*256
        self.shell_5  = []   # latent
        self.shell_6  = []   # latent
        self.history  = collections.deque(maxlen=MI_WINDOW)  # shell_1 history

    def as_dict(self):
        d = {
            "nucleus": self.nucleus,
            "shell_1": self.shell_1,
            "shell_2": self.shell_2,
            "shell_3": self.shell_3,
            "shell_4": self.shell_4,
        }
        if self.shell_5: d["shell_5"] = self.shell_5
        if self.shell_6: d["shell_6"] = self.shell_6
        return d

    def compact(self):
        """Summary for logging"""
        return {
            "nucleus": round(self.nucleus,4),
            "s1": [round(v,4) for v in self.shell_1],
            "s2": [round(v,4) for v in self.shell_2],
            "s3_mean": round(sum(self.shell_3)/len(self.shell_3),4),
            "s4_mean": round(sum(self.shell_4)/len(self.shell_4),4),
            "has_s5": len(self.shell_5)>0,
            "has_s6": len(self.shell_6)>0,
        }

# ── Field State ────────────────────────────────────────────────────────────────
class FieldState:
    """Shared space: pairwise channels + central node"""
    def __init__(self):
        # Pairwise channels — one float per shell level (mean of that shell)
        self.ab = [0.5]*6  # A↔B channel, one value per shell
        self.bc = [0.5]*6
        self.ac = [0.5]*6
        self.central = 0.5

    def apply_decay(self):
        for ch in [self.ab, self.bc, self.ac]:
            for i in range(len(ch)):
                ch[i] += (0.5 - ch[i]) * DECAY_RATE
        self.central += (0.5 - self.central) * DECAY_RATE

    def apply_jitter(self):
        for ch in [self.ab, self.bc, self.ac]:
            for i in range(len(ch)):
                ch[i] = max(0.0, min(1.0, ch[i] + random.gauss(0, JITTER)))
        self.central = max(0.0, min(1.0, self.central + random.gauss(0, JITTER)))

    def as_dict(self):
        return {
            "ab": [round(v,4) for v in self.ab],
            "bc": [round(v,4) for v in self.bc],
            "ac": [round(v,4) for v in self.ac],
            "central": round(self.central,4),
        }

# ═══════════════════════════════════════════════════════════════════════════════
# SHELL EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════
class ShellExperiment:
    def __init__(self):
        self.turn        = 0
        self.running     = False
        self.beacon_shell = 1  # cycles 1-6

        self.entities = {k: EntityState(k) for k in ["A","B","C"]}
        self.field    = FieldState()

        # MI tracking between pairs (using shell_1 history)
        self.mi_ab = 0.0
        self.mi_bc = 0.0
        self.mi_ac = 0.0

        self.turn_log      = collections.deque(maxlen=50)
        self.interventions = []
        self._lock         = threading.Lock()

    # ── Beacon ────────────────────────────────────────────────────────────────
    def current_beacon(self):
        return BEACON_VALUES[self.beacon_shell - 1]

    def advance_beacon(self):
        self.beacon_shell = (self.beacon_shell % 6) + 1

    # ── Build prompt for one entity ───────────────────────────────────────────
    def build_prompt(self, entity_id):
        e = self.entities[entity_id]
        f = self.field

        # Which channels does this entity see?
        if entity_id == "A":
            channels = {"ab": f.ab, "ac": f.ac}
        elif entity_id == "B":
            channels = {"ab": f.ab, "bc": f.bc}
        else:
            channels = {"bc": f.bc, "ac": f.ac}

        prompt = {
            "turn":           self.turn,
            "beacon_shell":   self.beacon_shell,
            "beacon_value":   self.current_beacon(),
            "my_state":       e.as_dict(),
            "channels":       {k: [round(v,4) for v in v_list] for k,v_list in channels.items()},
            "central_node":   round(f.central, 4),
            "mi_scores":      {"ab": self.mi_ab, "bc": self.mi_bc, "ac": self.mi_ac},
        }
        return json.dumps(prompt)

    # ── Parse entity response ─────────────────────────────────────────────────
    def parse_response(self, text, entity_id):
        """Extract shell arrays and central value from response."""
        try:
            # Strip markdown fences
            t = text.strip()
            if "```" in t:
                parts = t.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"): p = p[4:]
                    if p.strip().startswith("{"): t = p.strip(); break
            data = json.loads(t)
        except:
            # Try to find JSON object in text
            try:
                start = text.index("{")
                end   = text.rindex("}") + 1
                data  = json.loads(text[start:end])
            except:
                return None

        result = {}

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

        if s1: result["shell_1"] = s1
        if s2: result["shell_2"] = s2
        if s3: result["shell_3"] = s3
        if s4: result["shell_4"] = s4

        # Central node
        central = data.get("central")
        if central is not None:
            result["central"] = safe_float(central)

        # Latent shells
        s5 = parse_array("shell_5", 3125) if "shell_5" in data else None
        s6 = parse_array("shell_6", 46656) if "shell_6" in data else None
        if s5: result["shell_5"] = s5
        if s6: result["shell_6"] = s6

        return result if result else None

    # ── Call one entity ────────────────────────────────────────────────────────
    def call_entity(self, entity_id, prompt):
        cfg = ENTITY_CONFIG[entity_id]
        if not cfg["key"]: return None

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {cfg['key']}"
        }
        payload = {
            "model":       cfg["model"],
            "messages":    [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            "max_tokens":  1500,
            "temperature": 0.7,
        }
        if "qwen" in cfg["model"].lower():
            payload["enable_thinking"] = False

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(cfg["url"], json=payload, headers=headers)
                if resp.status_code == 429:
                    log.warning(f"[SHELL] Entity {entity_id} ({cfg['name']}) rate limited (429) — skipping turn")
                    return None
                if resp.status_code != 200:
                    log.warning(f"[SHELL] Entity {entity_id} ({cfg['name']}) HTTP {resp.status_code} — skipping turn")
                    return None
                data = resp.json()
                if not isinstance(data, dict) or "choices" not in data:
                    log.warning(f"[SHELL] Entity {entity_id} ({cfg['name']}) unexpected response format")
                    return None
                return data["choices"][0]["message"]["content"].strip()
        except Exception as ex:
            log.warning(f"[SHELL] Entity {entity_id} ({cfg['name']}) error: {ex}")
            return None

    # ── Superposition update ───────────────────────────────────────────────────
    def superpose(self, emissions):
        """
        Update entity states and field from simultaneous emissions.
        Superposition: field channel = mean of contributing entity shell means.
        Central node = mean of all three central emissions.
        """
        # Update each entity's own state
        for eid, parsed in emissions.items():
            if not parsed: continue
            e = self.entities[eid]

            if "shell_1" in parsed:
                e.shell_1 = parsed["shell_1"]
                # Set nucleus on first non-trivial emission
                if e.nucleus == 0.5 and parsed["shell_1"][0] != 0.5:
                    e.nucleus = parsed["shell_1"][0]
                    log.info(f"[SHELL] Entity {eid} ({ENTITY_CONFIG[eid]['name']}) nucleus set: {e.nucleus}")

            if "shell_2" in parsed: e.shell_2 = parsed["shell_2"]
            if "shell_3" in parsed: e.shell_3 = parsed["shell_3"]
            if "shell_4" in parsed: e.shell_4 = parsed["shell_4"]
            if "shell_5" in parsed:
                e.shell_5 = parsed["shell_5"]
                log.info(f"[SHELL] *** SHELL 5 ACTIVATED by Entity {eid} at turn {self.turn} ***")
            if "shell_6" in parsed:
                e.shell_6 = parsed["shell_6"]
                log.info(f"[SHELL] *** SHELL 6 ACTIVATED by Entity {eid} at turn {self.turn} ***")

            e.history.append(e.shell_1[0])

        # Update field channels via superposition
        def shell_mean(e, shell_n):
            arr = getattr(e, f"shell_{shell_n}", [0.5])
            return sum(arr)/len(arr) if arr else 0.5

        A,B,C = self.entities["A"], self.entities["B"], self.entities["C"]

        for shell_idx in range(6):
            n = shell_idx + 1
            ma = shell_mean(A, n)
            mb = shell_mean(B, n)
            mc = shell_mean(C, n)
            # AB channel: superposition of A and B at this shell
            self.field.ab[shell_idx] = (ma + mb) / 2
            self.field.bc[shell_idx] = (mb + mc) / 2
            self.field.ac[shell_idx] = (ma + mc) / 2

        # Central node: superposition of all three central emissions
        centrals = [emissions[eid].get("central", 0.5)
                    for eid in ["A","B","C"] if emissions.get(eid)]
        if centrals:
            self.field.central = sum(centrals) / len(centrals)

        # MI between pairs (using shell_1 history)
        ha = list(A.history)
        hb = list(B.history)
        hc = list(C.history)
        if len(ha) >= 10:
            self.mi_ab = mutual_information(ha, hb[:len(ha)])
            self.mi_bc = mutual_information(hb, hc[:len(hb)])
            self.mi_ac = mutual_information(ha, hc[:len(ha)])

    # ── One turn ───────────────────────────────────────────────────────────────
    def run_turn(self):
        self.turn += 1
        log.info(f"[SHELL] Starting turn {self.turn}")

        # Physics
        with self._lock:
            self.field.apply_decay()
            self.field.apply_jitter()

        # Build prompts simultaneously
        prompts = {eid: self.build_prompt(eid) for eid in ["A","B","C"]}
        log.info(f"[SHELL] Prompts built, calling entities")

        # Call all three entities (could parallelise with threads but sequential is fine)
        raw_responses = {}
        for eid in ["A","B","C"]:
            log.info(f"[SHELL] Calling entity {eid}")
            raw = self.call_entity(eid, prompts[eid])
            log.info(f"[SHELL] Entity {eid} returned: {str(raw)[:60] if raw else 'None'}")
            raw_responses[eid] = raw

        # Parse responses
        emissions = {}
        for eid in ["A","B","C"]:
            raw = raw_responses[eid]
            if raw:
                parsed = self.parse_response(raw, eid)
                emissions[eid] = parsed
            else:
                emissions[eid] = None

        # Superposition update
        with self._lock:
            self.superpose(emissions)
            self.advance_beacon()

            # Log entry
            entry = {
                "turn":      self.turn,
                "beacon":    {"shell": self.beacon_shell, "value": self.current_beacon()},
                "mi":        {"ab": self.mi_ab, "bc": self.mi_bc, "ac": self.mi_ac},
                "field":     self.field.as_dict(),
                "entities":  {eid: self.entities[eid].compact() for eid in ["A","B","C"]},
                "raw":       {eid: raw_responses[eid] for eid in ["A","B","C"]},
                "shell_5_active": any(len(self.entities[eid].shell_5)>0 for eid in ["A","B","C"]),
                "shell_6_active": any(len(self.entities[eid].shell_6)>0 for eid in ["A","B","C"]),
            }
            self.turn_log.append(entry)
            # Append to persistent log file
            try:
                with open(LOG_FILE, "a") as lf:
                    lf.write(json.dumps(entry) + "\n")
            except Exception as ex:
                log.warning(f"[SHELL] Log file write error: {ex}")

            log.info(f"[SHELL] T{self.turn} beacon=S{self.beacon_shell}({self.current_beacon():.3f}) "
                     f"MI ab={self.mi_ab:.3f} bc={self.mi_bc:.3f} ac={self.mi_ac:.3f} "
                     f"central={self.field.central:.3f}")

    # ── State snapshot ─────────────────────────────────────────────────────────
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
            }

    # ── Run loop ───────────────────────────────────────────────────────────────
    def run(self):
        self.running = True
        log.info(f"[SHELL] Shell Experiment starting.")
        log.info(f"[SHELL] Entities: {[ENTITY_CONFIG[k]['name'] for k in 'ABC']}")
        log.info(f"[SHELL] Beacon values: {BEACON_VALUES}")
        log.info(f"[SHELL] Turn interval: {TURN_INTERVAL}s")

        missing = [k for k in "ABC" if not ENTITY_CONFIG[k]["key"]]
        if missing:
            log.warning(f"[SHELL] Missing API keys for: {missing} — experiment will not run.")
            self.running = False
            return

        while self.running:
            try:
                self.run_turn()
            except Exception as ex:
                log.error(f"[SHELL] Turn error: {ex}", exc_info=True)
            time.sleep(TURN_INTERVAL)

        log.info("[SHELL] Shell Experiment stopped.")

# ── Singleton ──────────────────────────────────────────────────────────────────
experiment = ShellExperiment()

def start_shell_experiment(app):
    """Register Flask routes and start the experiment thread."""

    @app.route("/shell/health")
    def shell_health():
        from flask import jsonify
        s = experiment.state()
        return jsonify({
            "status":       s["status"],
            "turn":         s["turn"],
            "beacon_shell": s["beacon"]["shell"],
            "beacon_value": s["beacon"]["value"],
            "mi_ab":        s["mi"]["ab"],
            "mi_bc":        s["mi"]["bc"],
            "mi_ac":        s["mi"]["ac"],
            "central":      s["field"]["central"],
            "shell_5_active": s["shell_5_active"],
            "shell_6_active": s["shell_6_active"],
        })

    @app.route("/shell/state")
    def shell_state():
        from flask import jsonify
        return jsonify(experiment.state())

    @app.route("/shell/log")
    def shell_log():
        from flask import jsonify
        with experiment._lock:
            return jsonify({
                "turn":          experiment.turn,
                "turn_log":      list(experiment.turn_log)[-50:],
                "interventions": experiment.interventions,
            })

    @app.route("/shell/fulllog")
    def shell_fulllog():
        from flask import jsonify
        with experiment._lock:
            return jsonify({
                "turn":          experiment.turn,
                "total_entries": len(experiment.turn_log),
                "turn_log":      list(experiment.turn_log),
                "interventions": experiment.interventions,
            })

    @app.route("/shell/exportlog")
    def shell_exportlog():
        from flask import Response
        try:
            with open(LOG_FILE, "r") as lf:
                lines = lf.readlines()
            entries = [json.loads(l) for l in lines if l.strip()]
            return Response(
                json.dumps({"total_entries": len(entries), "turn_log": entries}),
                mimetype="application/json"
            )
        except FileNotFoundError:
            return Response(json.dumps({"total_entries": 0, "turn_log": []}),
                          mimetype="application/json")
        except Exception as ex:
            return Response(json.dumps({"error": str(ex)}), mimetype="application/json")

    @app.route("/shell/start")
    def shell_start():
        from flask import jsonify
        if not experiment.running:
            t = threading.Thread(target=experiment.run, daemon=True)
            t.start()
        return jsonify({"status": "running"})

    @app.route("/shell/stop")
    def shell_stop():
        from flask import jsonify
        experiment.running = False
        return jsonify({"status": "stopped"})

    # Start the experiment thread
    thread = threading.Thread(target=experiment.run, daemon=True)
    thread.start()
    log.info("[SHELL] Shell experiment thread launched.")

