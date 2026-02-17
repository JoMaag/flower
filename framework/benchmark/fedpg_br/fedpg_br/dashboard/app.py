"""Web dashboard for FedPG-BR distributed training.

Real-time visualization of federated learning with multiple clients.
"""

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from fedpg_br.benchmark.results_store import ResultsStore

app = Flask(__name__)
app.config["SECRET_KEY"] = "fedpg-br-secret"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
active_clients: Dict[str, dict] = {}
training_metrics: List[dict] = []
current_round = 0
results_store = None
_experiment_process: Optional[subprocess.Popen] = None
_experiment_thread: Optional[threading.Thread] = None


def init_dashboard(results_dir: str = "results"):
    """Initialize dashboard with results store."""
    global results_store
    results_store = ResultsStore(f"{results_dir}/.benchmark_db.sqlite")


@app.route("/")
def index():
    """Serve main dashboard page."""
    return render_template("dashboard.html")


@app.route("/experiment")
def experiment():
    """Serve experiment configuration dashboard."""
    return render_template("experiment.html")


@app.route("/api/strategies")
def get_strategies():
    """List all available aggregation strategies."""
    try:
        from fedpg_br.strategies import list_strategies
        return jsonify(list_strategies())
    except ImportError:
        # Fallback if strategies module not loaded
        return jsonify({
            "fedpg-br": "Byzantine filtering + SCSG variance reduction (paper's method)",
            "svrpg": "SCSG variance reduction, no Byzantine filtering",
            "gomdp": "Simple averaging, single gradient step (baseline)",
        })


@app.route("/api/status")
def get_status():
    """Get current training status."""
    return jsonify(
        {
            "active_clients": len(active_clients),
            "current_round": current_round,
            "total_rounds": 50,
            "status": "running" if active_clients else "idle",
        }
    )


@app.route("/api/clients")
def get_clients():
    """Get list of active clients."""
    return jsonify(list(active_clients.values()))


@app.route("/api/metrics")
def get_metrics():
    """Get training metrics history."""
    # Get latest metrics from database
    if results_store:
        # Get latest run
        runs = results_store.list_runs(limit=1)
        if runs:
            run_id = runs[0]["run_id"]
            metrics = results_store.get_metric_timeseries(run_id, "avg_reward")
            return jsonify(
                [{"round": r, "value": v} for r, v in metrics]
            )
    return jsonify(training_metrics)


@app.route("/api/runs")
def get_runs():
    """Get list of all training runs."""
    if results_store:
        runs = results_store.list_runs(limit=10)
        return jsonify(runs)
    return jsonify([])


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit("status", {"message": "Connected to FedPG-BR dashboard"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


@socketio.on("start_experiment")
def handle_start_experiment(data):
    """Launch a flower-simulation subprocess from the dashboard."""
    global _experiment_process, _experiment_thread

    # Kill any running experiment first
    _kill_experiment()

    env_name = data.get("env", "CartPole-v1")
    method = data.get("method", "fedpg-br")
    num_workers = int(data.get("num_workers", 10))
    num_byzantine = int(data.get("num_byzantine", 0))
    num_rounds = int(data.get("num_rounds", 312))
    attack_type = data.get("attack_type", "none")
    batch_size = data.get("batch_size")
    learning_rate = data.get("learning_rate")
    sigma = data.get("sigma")
    gamma = data.get("gamma")
    mini_batch_size = data.get("mini_batch_size")
    delta = data.get("delta")
    max_episode_len = data.get("max_episode_len")
    hidden_units = data.get("hidden_units")
    activation = data.get("activation")

    use_fedpg_br = "true" if method == "fedpg-br" else "false"
    attack_cfg = attack_type if attack_type != "none" else "random-noise"

    run_config = (
        f"env='{env_name}' method='{method}' "
        f"num-server-rounds={num_rounds} num-workers={num_workers} "
        f"num-byzantine={num_byzantine} use-fedpg-br={use_fedpg_br} "
        f"attack-type='{attack_cfg}'"
    )
    if batch_size:
        run_config += f" batch-size={int(batch_size)}"
    if learning_rate:
        run_config += f" lr={float(learning_rate)}"
    if sigma:
        run_config += f" sigma={float(sigma)}"
    if gamma:
        run_config += f" gamma={float(gamma)}"
    if mini_batch_size:
        run_config += f" mini-batch-size={int(mini_batch_size)}"
    if delta:
        run_config += f" delta={float(delta)}"
    if max_episode_len:
        run_config += f" max-episode-len={int(max_episode_len)}"
    if hidden_units:
        run_config += f" hidden-units='{hidden_units}'"
    if activation:
        run_config += f" activation='{activation}'"

    # Find the project root (where pyproject.toml is)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Find flower-simulation executable next to the running Python interpreter
    # On conda/venv Windows: python.exe is in env root, scripts are in Scripts/
    python_dir = Path(sys.executable).parent
    candidates = [
        python_dir / "Scripts" / "flower-simulation.exe",  # Windows conda/venv
        python_dir / "Scripts" / "flower-simulation",
        python_dir / "flower-simulation.exe",              # Some installations
        python_dir / "flower-simulation",                  # Linux/macOS
    ]
    flower_sim = None
    for c in candidates:
        if c.exists():
            flower_sim = str(c)
            break

    if not flower_sim:
        import shutil
        flower_sim = shutil.which("flower-simulation")

    if flower_sim:
        cmd = [
            flower_sim,
            "--app", str(project_root),
            "--num-supernodes", str(num_workers),
            "--run-config", run_config,
        ]
    else:
        socketio.emit("log", {"message": f"ERROR: Cannot find flower-simulation. Python at: {sys.executable}"})
        return

    print(f"[Dashboard] Launching: {cmd}")
    socketio.emit("log", {"message": f"CMD: {cmd[0]}"})
    socketio.emit("log", {"message": f"Launching {method.upper()} on {env_name} (K={num_workers}, B={num_byzantine}, rounds={num_rounds})"})

    try:
        _experiment_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root),
        )
    except Exception as e:
        socketio.emit("log", {"message": f"ERROR: Failed to launch: {e}"})
        return

    # Stream output in a background thread
    def stream_output():
        global _experiment_process
        proc = _experiment_process
        if proc is None or proc.stdout is None:
            return

        # Regex patterns for parsing flower output
        fit_progress_re = re.compile(
            r"fit progress: \((\d+), (-?[\d.]+), \{'server_avg_reward': np\.float64\((-?[\d.]+)\)\}, ([\d.]+)\)"
        )
        round_detail_re = re.compile(
            r"Round (\d+)(?: \[(\w[\w-]*)\])?: good_agents=(\d+), scsg_steps=(\d+), active=(\d+), skipped=(\d+)"
        )

        last_good_agents = 0
        last_scsg_steps = 0
        last_reward = 0.0

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue

            # Strip ANSI color codes for parsing
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

            # Parse round detail line
            m = round_detail_re.search(clean)
            if m:
                last_good_agents = int(m.group(3))
                last_scsg_steps = int(m.group(4))
                continue

            # Parse fit progress line
            m = fit_progress_re.search(clean)
            if m:
                round_num = int(m.group(1))
                reward = float(m.group(3))

                last_reward = reward
                socketio.emit("metrics", {
                    "round": round_num,
                    "server_avg_reward": reward,
                    "num_good_agents": last_good_agents,
                    "scsg_steps": last_scsg_steps,
                })
                continue

        # Process finished
        exit_code = proc.wait()
        socketio.emit("experiment_done", {
            "method": method,
            "exit_code": exit_code,
            "final_reward": last_reward,
        })
        socketio.emit("log", {"message": f"Experiment finished (exit code {exit_code})"})
        _experiment_process = None

    _experiment_thread = threading.Thread(target=stream_output, daemon=True)
    _experiment_thread.start()


@socketio.on("stop_experiment")
def handle_stop_experiment():
    """Stop the running experiment."""
    _kill_experiment()
    socketio.emit("log", {"message": "Experiment stopped by user."})


def _kill_experiment():
    """Terminate the running experiment process if any."""
    global _experiment_process
    if _experiment_process is not None:
        try:
            _experiment_process.terminate()
            _experiment_process.wait(timeout=5)
        except Exception:
            try:
                _experiment_process.kill()
            except Exception:
                pass
        _experiment_process = None


@socketio.on("register_client")
def handle_register_client(data):
    """Register a new training client."""
    client_id = data.get("client_id")
    client_info = {
        "id": client_id,
        "name": data.get("name", f"Client {client_id}"),
        "location": data.get("location", "Unknown"),
        "status": "active",
        "last_seen": datetime.now().isoformat(),
    }
    active_clients[client_id] = client_info

    # Broadcast update to all connected dashboards
    socketio.emit("client_update", {"clients": list(active_clients.values())})


@socketio.on("metrics_update")
def handle_metrics_update(data):
    """Handle real-time metrics update from training."""
    global current_round

    round_num = data.get("round")
    metrics = data.get("metrics", {})

    if round_num > current_round:
        current_round = round_num

    # Store metrics
    metric_entry = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    training_metrics.append(metric_entry)

    # Keep only last 100 entries
    if len(training_metrics) > 100:
        training_metrics.pop(0)

    # Broadcast to all connected dashboards
    socketio.emit("metrics", metric_entry)


def start_dashboard(host: str = "0.0.0.0", port: int = 5000, results_dir: str = "results"):
    """Start the web dashboard server.

    Args:
        host: Host address to bind to
        port: Port to listen on
        results_dir: Directory containing results
    """
    import webbrowser
    import threading

    init_dashboard(results_dir)

    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/experiment"
    print(f"\n🚀 FedPG-BR Dashboard starting at {url}")

    # Open browser after a short delay to let the server start
    if not os.environ.get("RUNNING_IN_DOCKER"):
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    start_dashboard()
