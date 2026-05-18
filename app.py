"""
app.py — Portal Experiment Server
Clean rebuild. Shell experiment replaced by Portal.
"""
import logging
import os
import threading
import time

from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return jsonify({"service": "portal-experiment", "status": "ok"})


@app.route("/health")
def health():
    return jsonify({"service": "portal-experiment", "status": "ok"})


# Register portal experiment routes
try:
    from portal_experiment import start_portal_experiment
    start_portal_experiment(app)
    logging.info("Portal experiment routes registered — /portal/* active")
except Exception as e:
    logging.warning(f"Portal experiment failed to register: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
