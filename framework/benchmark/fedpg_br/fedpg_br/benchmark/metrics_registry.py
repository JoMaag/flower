"""Global registry for metrics collector.

This module provides a simple global registry pattern to allow metrics collection
to be coordinated between the RunManager and server_fn, which is called by Flower.
"""

from typing import Optional
from fedpg_br.benchmark.metrics_collector import MetricsCollector

# Global registry
_metrics_collector: Optional[MetricsCollector] = None


def register_metrics_collector(collector: MetricsCollector) -> None:
    """Register a metrics collector for use by the server.

    Args:
        collector: MetricsCollector instance
    """
    global _metrics_collector
    _metrics_collector = collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the registered metrics collector.

    Returns:
        MetricsCollector instance or None if not registered
    """
    return _metrics_collector


def clear_metrics_collector() -> None:
    """Clear the registered metrics collector."""
    global _metrics_collector
    _metrics_collector = None
