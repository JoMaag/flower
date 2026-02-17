"""Metrics collection and aggregation."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from flwr.common.typing import Scalar


@dataclass
class MetricRecord:
    """A single metric record."""

    run_id: str
    round_num: int
    metric_name: str
    metric_value: float
    client_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates metrics during federated training.

    This class buffers metrics from server and client, provides callbacks
    for real-time updates, and can forward metrics to storage.
    """

    def __init__(self, run_id: str, results_dir: Optional[str] = None):
        """Initialize metrics collector.

        Args:
            run_id: Unique identifier for this training run
            results_dir: Optional directory to save metrics to file
        """
        self.run_id = run_id
        self._server_metrics: dict[int, dict[str, float]] = {}
        self._client_metrics: dict[int, dict[int, dict[str, float]]] = {}
        self._callbacks: list[Callable[[int, dict[str, Any]], None]] = []
        self._start_time = time.time()

        # Optional file-based storage for cross-process communication
        self._metrics_file = None
        if results_dir:
            metrics_path = Path(results_dir) / "metrics_live.jsonl"
            self._metrics_file = open(metrics_path, "w")

    def record_server_metrics(
        self, round_num: int, metrics: dict[str, Scalar]
    ) -> None:
        """Record server-level metrics for a round.

        Args:
            round_num: Server round number
            metrics: Dictionary of metric name -> value
        """
        # Convert Scalar types to float, skipping None values
        float_metrics = {}
        for k, v in metrics.items():
            if v is None or (isinstance(v, str) and v.lower() == 'none'):
                continue  # Skip None values
            try:
                float_metrics[k] = float(v)
            except (ValueError, TypeError):
                # Skip values that can't be converted to float
                continue
        self._server_metrics[round_num] = float_metrics

        # Save to file if enabled
        if self._metrics_file:
            for metric_name, metric_value in float_metrics.items():
                record = {
                    "run_id": self.run_id,
                    "round_num": round_num,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "client_id": None,
                    "timestamp": time.time(),
                }
                self._metrics_file.write(json.dumps(record) + "\n")
                self._metrics_file.flush()

        # Trigger callbacks
        self._notify_callbacks(round_num, {"server": float_metrics})

    def record_client_metrics(
        self, round_num: int, client_id: int, metrics: dict[str, Scalar]
    ) -> None:
        """Record client-level metrics for a round.

        Args:
            round_num: Server round number
            client_id: Client identifier
            metrics: Dictionary of metric name -> value
        """
        if round_num not in self._client_metrics:
            self._client_metrics[round_num] = {}

        # Convert Scalar types to float, skipping None values
        float_metrics = {}
        for k, v in metrics.items():
            if v is None or (isinstance(v, str) and v.lower() == 'none'):
                continue  # Skip None values
            try:
                float_metrics[k] = float(v)
            except (ValueError, TypeError):
                # Skip values that can't be converted to float
                continue
        self._client_metrics[round_num][client_id] = float_metrics

        # Save to file if enabled
        if self._metrics_file:
            for metric_name, metric_value in float_metrics.items():
                record = {
                    "run_id": self.run_id,
                    "round_num": round_num,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "client_id": client_id,
                    "timestamp": time.time(),
                }
                self._metrics_file.write(json.dumps(record) + "\n")
                self._metrics_file.flush()

        # Trigger callbacks with aggregated client metrics
        aggregated = self._aggregate_client_metrics(round_num)
        self._notify_callbacks(round_num, {"clients": aggregated})

    def add_callback(self, callback: Callable[[int, dict[str, Any]], None]) -> None:
        """Add a callback function to be notified of new metrics.

        Args:
            callback: Function that takes (round_num, metrics_dict)
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, round_num: int, metrics: dict[str, Any]) -> None:
        """Notify all registered callbacks of new metrics."""
        for callback in self._callbacks:
            try:
                callback(round_num, metrics)
            except Exception as e:
                # Don't let callback errors break training
                print(f"Warning: Callback error in round {round_num}: {e}")

    def _aggregate_client_metrics(self, round_num: int) -> dict[str, float]:
        """Aggregate client metrics for a round (mean values).

        Args:
            round_num: Server round number

        Returns:
            Dictionary of aggregated metrics
        """
        if round_num not in self._client_metrics:
            return {}

        client_data = self._client_metrics[round_num]
        if not client_data:
            return {}

        # Collect all metric names
        all_metrics: set[str] = set()
        for metrics in client_data.values():
            all_metrics.update(metrics.keys())

        # Compute mean for each metric
        aggregated = {}
        for metric_name in all_metrics:
            values = [
                metrics[metric_name]
                for metrics in client_data.values()
                if metric_name in metrics
            ]
            if values:
                aggregated[f"client_{metric_name}_mean"] = sum(values) / len(values)

        return aggregated

    def get_server_metrics(self, round_num: int) -> dict[str, float]:
        """Get server metrics for a specific round.

        Args:
            round_num: Server round number

        Returns:
            Dictionary of metrics (empty if round not found)
        """
        return self._server_metrics.get(round_num, {})

    def get_client_metrics(
        self, round_num: int, client_id: Optional[int] = None
    ) -> dict[str, float]:
        """Get client metrics for a specific round.

        Args:
            round_num: Server round number
            client_id: Specific client ID (if None, returns aggregated)

        Returns:
            Dictionary of metrics
        """
        if round_num not in self._client_metrics:
            return {}

        if client_id is not None:
            return self._client_metrics[round_num].get(client_id, {})
        else:
            # Return aggregated
            return self._aggregate_client_metrics(round_num)

    def get_all_rounds(self) -> list[int]:
        """Get list of all recorded rounds."""
        return sorted(set(self._server_metrics.keys()) | set(self._client_metrics.keys()))

    def get_metric_history(self, metric_name: str, source: str = "server") -> list[tuple[int, float]]:
        """Get time series of a specific metric.

        Args:
            metric_name: Name of the metric
            source: "server" or "client" (for client, returns aggregated)

        Returns:
            List of (round_num, value) tuples
        """
        history = []
        for round_num in self.get_all_rounds():
            if source == "server":
                metrics = self.get_server_metrics(round_num)
            else:
                metrics = self.get_client_metrics(round_num)

            if metric_name in metrics:
                history.append((round_num, metrics[metric_name]))

        return history

    def get_elapsed_time(self) -> float:
        """Get elapsed time since collector initialization."""
        return time.time() - self._start_time

    def export_records(self) -> list[MetricRecord]:
        """Export all metrics as a list of MetricRecord objects.

        Returns:
            List of all metric records
        """
        records = []

        # Export server metrics
        for round_num, metrics in self._server_metrics.items():
            for metric_name, value in metrics.items():
                records.append(
                    MetricRecord(
                        run_id=self.run_id,
                        round_num=round_num,
                        metric_name=metric_name,
                        metric_value=value,
                        client_id=None,
                    )
                )

        # Export client metrics
        for round_num, clients in self._client_metrics.items():
            for client_id, metrics in clients.items():
                for metric_name, value in metrics.items():
                    records.append(
                        MetricRecord(
                            run_id=self.run_id,
                            round_num=round_num,
                            metric_name=metric_name,
                            metric_value=value,
                            client_id=client_id,
                        )
                    )

        return records

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        all_rounds = self.get_all_rounds()
        if not all_rounds:
            return {"total_rounds": 0, "elapsed_time": self.get_elapsed_time()}

        summary = {
            "total_rounds": len(all_rounds),
            "elapsed_time": self.get_elapsed_time(),
            "first_round": min(all_rounds),
            "last_round": max(all_rounds),
        }

        # Add final metrics from last round
        last_round = max(all_rounds)
        summary["final_server_metrics"] = self.get_server_metrics(last_round)
        summary["final_client_metrics"] = self.get_client_metrics(last_round)

        return summary

    def close(self) -> None:
        """Close the metrics file if open."""
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_file = None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
