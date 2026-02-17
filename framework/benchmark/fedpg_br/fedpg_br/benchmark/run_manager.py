"""Run manager for executing benchmark experiments."""

import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fedpg_br.benchmark import metrics_registry
from fedpg_br.benchmark.metrics_collector import MetricsCollector
from fedpg_br.benchmark.results_store import ResultsStore
from fedpg_br.benchmark.utils import (
    create_run_directory,
    generate_run_id,
    get_system_metadata,
    save_jsonl,
)


class RunManager:
    """Manages the execution of benchmark runs.

    This class orchestrates the execution of experiments, including:
    - Setting up metrics collection
    - Running Flower simulations
    - Storing results to database
    - Saving run artifacts
    """

    def __init__(
        self,
        results_store: ResultsStore,
        results_dir: str = "results",
        enable_visualization: bool = False,
    ):
        """Initialize run manager.

        Args:
            results_store: ResultsStore instance
            results_dir: Base directory for results
            enable_visualization: Enable real-time visualization
        """
        self.results_store = results_store
        self.results_dir = Path(results_dir)
        self.enable_visualization = enable_visualization

    def run_experiment(
        self,
        config: Dict[str, Any],
        suite_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single experiment.

        Args:
            config: Experiment configuration
            suite_name: Name of benchmark suite (if applicable)
            tags: Tags for this run
            run_id: Optional run ID (generated if not provided)

        Returns:
            Dictionary with run results
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = generate_run_id()

        # Create run directory
        run_dir = create_run_directory(run_id, str(self.results_dir))

        # Collect metadata
        metadata = get_system_metadata()

        # Create metrics collector
        metrics_collector = MetricsCollector(run_id)

        # Register metrics collector globally
        metrics_registry.register_metrics_collector(metrics_collector)

        # Create run entry in database
        self.results_store.create_run(
            run_id=run_id,
            config=config,
            suite_name=suite_name,
            git_commit=metadata.get("git_commit"),
            metadata=metadata,
            tags=tags,
        )

        # Save config snapshot
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        try:
            # Prepare config with benchmark enabled
            run_config = config.copy()
            run_config["enable-benchmark"] = True
            run_config["benchmark-run-id"] = run_id
            run_config["benchmark-results-dir"] = str(run_dir.absolute())

            # Create temporary config file for this run
            temp_config_path = run_dir / "run_config.toml"
            self._write_toml_config(temp_config_path, run_config)

            # Optional: Connect visualizer
            if self.enable_visualization:
                try:
                    from fedpg_br.benchmark.visualizer import LiveVisualizer

                    visualizer = LiveVisualizer(metrics_collector, config, results_dir=run_dir)
                    visualizer.start()
                except ImportError:
                    print("Warning: Visualizer not available")

            # Record start time
            start_time = time.time()

            # Run Flower simulation using subprocess
            result = self._run_flower_simulation(temp_config_path)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Update run status
            if result["success"]:
                self.results_store.update_run_status(run_id, "completed")
            else:
                self.results_store.update_run_status(
                    run_id, "failed", error_message=result.get("error")
                )

            # Stop visualizer if running
            if self.enable_visualization:
                try:
                    visualizer.stop()
                except (NameError, UnboundLocalError):
                    pass

            # Read metrics from file (written by server process)
            metrics_live_path = run_dir / "metrics_live.jsonl"
            metric_records = []

            if metrics_live_path.exists():
                # Load metrics from file
                from fedpg_br.benchmark.metrics_collector import MetricRecord

                with open(metrics_live_path, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.load(io.StringIO(line))
                            metric_records.append(
                                MetricRecord(
                                    run_id=data["run_id"],
                                    round_num=data["round_num"],
                                    metric_name=data["metric_name"],
                                    metric_value=data["metric_value"],
                                    client_id=data.get("client_id"),
                                    timestamp=data["timestamp"],
                                )
                            )

                # Store metrics to database
                if metric_records:
                    self.results_store.store_metrics(metric_records)

                # Save metrics as JSONL for portability
                metrics_jsonl_path = run_dir / "metrics.jsonl"
                metrics_data = [
                    {
                        "run_id": r.run_id,
                        "round_num": r.round_num,
                        "metric_name": r.metric_name,
                        "metric_value": r.metric_value,
                        "client_id": r.client_id,
                        "timestamp": r.timestamp,
                    }
                    for r in metric_records
                ]
                save_jsonl(metrics_data, metrics_jsonl_path)

            # Get summary from metrics
            if metric_records:
                # Count unique rounds
                unique_rounds = len(set(r.round_num for r in metric_records))
                summary = {
                    "total_rounds": unique_rounds,
                    "elapsed_time": elapsed_time,
                }
            else:
                summary = metrics_collector.get_summary()
                # Add elapsed time if not present
                if "elapsed_time" not in summary:
                    summary["elapsed_time"] = elapsed_time

            return {
                "run_id": run_id,
                "success": result["success"],
                "run_dir": str(run_dir),
                "summary": summary,
                "error": result.get("error"),
            }

        except Exception as e:
            # Handle unexpected errors
            self.results_store.update_run_status(run_id, "failed", error_message=str(e))
            return {
                "run_id": run_id,
                "success": False,
                "error": str(e),
            }

        finally:
            # Clear metrics collector from registry
            metrics_registry.clear_metrics_collector()

    def _write_toml_config(self, path: Path, config: Dict[str, Any]) -> None:
        """Write configuration to TOML file.

        Args:
            path: Path to TOML file
            config: Configuration dictionary
        """
        # Simple TOML writer (can use tomli-w if available)
        try:
            import tomli_w
            with open(path, "wb") as f:
                tomli_w.dump(config, f)
        except ImportError:
            # Fallback: manual TOML writing for simple configs
            with open(path, "w") as f:
                for key, value in config.items():
                    if isinstance(value, bool):
                        f.write(f'{key} = {str(value).lower()}\n')
                    elif isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f'{key} = {value}\n')

    def _run_flower_simulation(self, config_path: Path) -> Dict[str, Any]:
        """Run Flower simulation using subprocess.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary with execution results
        """
        try:
            # Get the project root directory (where pyproject.toml is)
            project_root = Path.cwd()

            # Run flwr command directly from project root
            cmd = [
                "flwr",
                "run",
                ".",
                "--run-config",
                str(config_path.absolute()),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(project_root),
            )

            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Flower simulation failed: {e.stderr}",
                "stdout": e.stdout,
                "stderr": e.stderr,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }
