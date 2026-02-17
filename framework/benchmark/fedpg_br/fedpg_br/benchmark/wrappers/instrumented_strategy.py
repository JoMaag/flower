"""Instrumented strategy wrapper for metrics collection."""

from typing import Dict, List, Optional, Tuple

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from fedpg_br.server_app import FedPGStrategy
from fedpg_br.benchmark.metrics_collector import MetricsCollector


class InstrumentedFedPGStrategy(FedPGStrategy):
    """Wrapper around FedPGStrategy that injects metrics collection hooks.

    This class extends FedPGStrategy to capture and record metrics during
    training without modifying the core FedPG-BR logic.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        env_name: str,
        num_agents: int,
        byzantine_ratio: float = 0.0,
        use_adaptive_batch: bool = False,
    ):
        """Initialize instrumented strategy.

        Args:
            metrics_collector: MetricsCollector instance for recording metrics
            env_name: Name of the Gymnasium environment
            num_agents: Total number of agents/workers
            byzantine_ratio: Fraction of Byzantine agents (0.0 to 1.0)
            use_adaptive_batch: Enable adaptive batch sizing (FedPG-BR)
        """
        super().__init__(env_name, num_agents, byzantine_ratio, use_adaptive_batch)
        self.metrics_collector = metrics_collector

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with metrics collection.

        This method wraps the parent aggregate_fit to capture both server-level
        and client-level metrics.

        Args:
            server_round: Current server round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failed client results

        Returns:
            Tuple of (updated_parameters, server_metrics)
        """
        # Collect client-level metrics BEFORE aggregation
        for client, fit_res in results:
            if fit_res.metrics:
                # Extract client ID from proxy (use hash if not available)
                client_id = hash(client.cid) % 10000
                self.metrics_collector.record_client_metrics(
                    server_round, client_id, fit_res.metrics
                )

        # Call parent (unchanged core logic)
        params, server_metrics = super().aggregate_fit(server_round, results, failures)

        # Collect server-level metrics AFTER aggregation
        if server_metrics:
            self.metrics_collector.record_server_metrics(server_round, server_metrics)

        return params, server_metrics

    def aggregate_evaluate(
        self, server_round: int, results, failures
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluate results with metrics collection.

        Args:
            server_round: Current server round number
            results: List of evaluation results
            failures: List of failed evaluations

        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        # Call parent
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Record evaluation metrics
        if metrics:
            self.metrics_collector.record_server_metrics(server_round, metrics)

        return loss, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model on server with metrics collection.

        Args:
            server_round: Current server round number
            parameters: Current model parameters

        Returns:
            Tuple of (loss, metrics) or None
        """
        # Call parent
        result = super().evaluate(server_round, parameters)

        # Record server evaluation metrics
        if result is not None:
            loss, metrics = result
            if metrics:
                # Prefix with "server_eval_" to distinguish from client eval
                prefixed_metrics = {f"server_eval_{k}": v for k, v in metrics.items()}
                self.metrics_collector.record_server_metrics(
                    server_round, prefixed_metrics
                )

        return result
