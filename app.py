"""
shell_app.py — Shell Experiment Server
"""
import logging, os, threading, time
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return jsonify({"service": "shell-experiment", "status": "ok"})

@app.route("/health")
def health():
    return jsonify({"service": "shell-experiment", "status": "ok"})

# Register shell experiment routes
try:
    from shell_experiment import start_shell_experiment
    start_shell_experiment(app)
    logging.info("Shell experiment routes registered — /shell/* active")
except Exception as e:
    logging.warning(f"Shell experiment failed to start: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
