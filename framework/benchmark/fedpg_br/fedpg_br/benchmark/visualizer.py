"""Real-time visualization using Rich."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from fedpg_br.benchmark.metrics_collector import MetricsCollector


class LiveVisualizer:
    """Real-time visualization of training metrics using Rich.

    This class provides live updates of training progress, metrics, and
    Byzantine detection statistics during federated learning.
    """

    def __init__(self, metrics_collector: MetricsCollector, config: Dict[str, Any], results_dir: Path = None):
        """Initialize visualizer.

        Args:
            metrics_collector: MetricsCollector instance
            config: Experiment configuration
            results_dir: Directory where metrics_live.jsonl is written
        """
        self.metrics_collector = metrics_collector
        self.config = config
        self.console = Console()
        self.is_running = False
        self._live = None
        self._update_thread = None
        self.results_dir = results_dir

        # Extract config info
        self.env_name = config.get("env", "Unknown")
        self.num_workers = config.get("num-workers", 0)
        self.num_byzantine = config.get("num-byzantine", 0)
        self.attack_type = config.get("attack-type", "none")
        self.num_rounds = config.get("num-server-rounds", 0)

        # Metrics tracking
        self.current_round = 0
        self.best_reward = float("-inf")
        self.best_round = 0
        self.latest_metrics = {}
        self._last_file_size = 0

        # If results_dir is provided, use file-based polling
        # Otherwise fall back to callback-based updates
        if results_dir:
            self.metrics_file = results_dir / "metrics_live.jsonl"
        else:
            self.metrics_file = None
            self.metrics_collector.add_callback(self._on_metrics_update)

    def start(self) -> None:
        """Start the live visualization."""
        self.is_running = True

        # Create live display
        layout = self._create_layout()
        self._live = Live(layout, console=self.console, refresh_per_second=4)
        self._live.start()

        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop(self) -> None:
        """Stop the live visualization."""
        self.is_running = False

        if self._update_thread:
            self._update_thread.join(timeout=2)

        if self._live:
            self._live.stop()

    def _on_metrics_update(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Callback for metrics updates.

        Args:
            round_num: Round number
            metrics: Metrics dictionary
        """
        self.current_round = round_num

        # Track best reward
        server_metrics = metrics.get("server", {})
        if "avg_reward" in server_metrics:
            reward = server_metrics["avg_reward"]
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_round = round_num

    def _update_loop(self) -> None:
        """Update loop for live display."""
        while self.is_running:
            # Poll metrics file if using file-based updates
            if self.metrics_file:
                self._poll_metrics_file()

            if self._live:
                self._live.update(self._create_layout())
            time.sleep(0.25)  # Update 4 times per second

    def _poll_metrics_file(self) -> None:
        """Poll the metrics file for new data."""
        if not self.metrics_file:
            return

        if not self.metrics_file.exists():
            return

        try:
            # Read all lines from the file
            with open(self.metrics_file, "r") as f:
                lines = f.readlines()

            # Process each line
            lines_processed = 0
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    round_num = record["round_num"]
                    metric_name = record["metric_name"]
                    metric_value = record["metric_value"]

                    # Update current round
                    if round_num > self.current_round:
                        self.current_round = round_num

                    # Store metrics by round
                    if round_num not in self.latest_metrics:
                        self.latest_metrics[round_num] = {}
                    self.latest_metrics[round_num][metric_name] = metric_value

                    # Track best reward
                    if "avg_reward" in metric_name:
                        if metric_value > self.best_reward:
                            self.best_reward = metric_value
                            self.best_round = round_num
                    lines_processed += 1
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            # Silently ignore errors during polling
            pass

    def _create_layout(self) -> Layout:
        """Create the Rich layout for visualization.

        Returns:
            Layout object
        """
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
        )

        # Header
        header = self._create_header()
        layout["header"].update(header)

        # Body
        body_layout = Layout()
        body_layout.split_column(
            Layout(name="progress", size=5),
            Layout(name="metrics", size=10),
        )

        body_layout["progress"].update(self._create_progress_bar())
        body_layout["metrics"].update(self._create_metrics_table())

        layout["body"].update(body_layout)

        return layout

    def _create_header(self) -> Panel:
        """Create header panel with configuration info.

        Returns:
            Panel with header information
        """
        header_text = f"""[bold]FedPG-BR Training[/bold]
Environment: {self.env_name}
Workers: {self.num_workers} ({self.num_byzantine} Byzantine)
Attack: {self.attack_type}"""

        return Panel(header_text, border_style="cyan")

    def _create_progress_bar(self) -> Panel:
        """Create progress bar panel.

        Returns:
            Panel with progress bar
        """
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

        task = progress.add_task(
            "Round Progress",
            total=self.num_rounds,
            completed=self.current_round,
        )

        return Panel(progress, title="Progress", border_style="green")

    def _create_metrics_table(self) -> Table:
        """Create metrics table.

        Returns:
            Table with current metrics
        """
        table = Table(
            title="Metrics", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Current", justify="right")
        table.add_column("Best", justify="right")
        table.add_column("Round", justify="right")

        # Get current metrics (from file or callback)
        if self.current_round > 0:
            # Use latest_metrics if available (file-based), otherwise query collector
            if self.latest_metrics and self.current_round in self.latest_metrics:
                current_metrics = self.latest_metrics[self.current_round]
            else:
                server_metrics = self.metrics_collector.get_server_metrics(
                    self.current_round
                )
                client_metrics = self.metrics_collector.get_client_metrics(
                    self.current_round
                )
                current_metrics = {**server_metrics, **client_metrics}

            # Avg Reward
            reward = current_metrics.get("avg_reward", 0)
            table.add_row(
                "Avg Reward",
                f"{reward:.2f}",
                f"{self.best_reward:.2f}" if self.best_reward > float("-inf") else "N/A",
                str(self.best_round) if self.best_round > 0 else "N/A",
            )

            # Good Agents
            good_agents = current_metrics.get("num_good_agents", 0)
            table.add_row(
                "Good Agents",
                str(int(good_agents)) if good_agents else "0",
                "-",
                "-",
            )

            # SCSG Steps
            scsg_steps = current_metrics.get("scsg_steps", 0)
            table.add_row(
                "SCSG Steps",
                str(int(scsg_steps)) if scsg_steps else "0",
                "-",
                "-",
            )

            # Loss
            if "client_loss_mean" in current_metrics:
                loss = current_metrics["client_loss_mean"]
                table.add_row(
                    "Loss",
                    f"{loss:.4f}",
                    "-",
                    "-",
                )

        return table
