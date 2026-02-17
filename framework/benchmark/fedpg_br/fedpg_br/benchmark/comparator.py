"""Comparator for analyzing multiple runs."""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from fedpg_br.benchmark.results_store import ResultsStore


class Comparator:
    """Compares and visualizes results from multiple benchmark runs."""

    def __init__(self, results_store: ResultsStore):
        """Initialize comparator.

        Args:
            results_store: ResultsStore instance
        """
        self.results_store = results_store
        self.console = Console()

    def compare_runs(
        self,
        run_ids: List[str],
        metric_name: str = "avg_reward",
        show_plot: bool = True,
    ) -> None:
        """Compare multiple runs and display results.

        Args:
            run_ids: List of run IDs to compare
            metric_name: Metric to compare
            show_plot: Whether to show learning curve plot
        """
        # Fetch run information
        runs_data = []
        for run_id in run_ids:
            run_info = self.results_store.get_run(run_id)
            if run_info is None:
                self.console.print(f"[yellow]Warning: Run not found: {run_id}[/yellow]")
                continue

            # Get metric time series
            timeseries = self.results_store.get_metric_timeseries(run_id, metric_name)

            if not timeseries:
                self.console.print(
                    f"[yellow]Warning: No {metric_name} data for {run_id}[/yellow]"
                )
                continue

            # Compute statistics
            values = [v for _, v in timeseries]
            final_value = values[-1] if values else None
            max_value = max(values) if values else None
            mean_value = sum(values) / len(values) if values else None

            runs_data.append(
                {
                    "run_id": run_id,
                    "run_info": run_info,
                    "timeseries": timeseries,
                    "final_value": final_value,
                    "max_value": max_value,
                    "mean_value": mean_value,
                }
            )

        if not runs_data:
            self.console.print("[red]No valid runs to compare[/red]")
            return

        # Display comparison table
        self._display_comparison_table(runs_data, metric_name)

        # Optionally display learning curves
        if show_plot:
            self._display_learning_curves(runs_data, metric_name)

    def _display_comparison_table(
        self, runs_data: List[Dict[str, Any]], metric_name: str
    ) -> None:
        """Display comparison table.

        Args:
            runs_data: List of run data dictionaries
            metric_name: Name of metric being compared
        """
        table = Table(title=f"Run Comparison - {metric_name}", show_header=True)
        table.add_column("Run ID", style="cyan")
        table.add_column("Config", style="dim")
        table.add_column("Final", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Status", justify="center")

        # Find baseline for % difference
        baseline = runs_data[0]["final_value"] if runs_data else None

        for data in runs_data:
            run_id = data["run_id"][:12]  # Truncate for display
            run_info = data["run_info"]
            config = run_info["config"]

            # Extract key config details
            config_str = f"{config.get('env', 'N/A')} | "
            config_str += f"workers={config.get('num-workers', 'N/A')} | "
            if config.get("num-byzantine", 0) > 0:
                config_str += f"byz={config.get('num-byzantine')} "
                config_str += f"({config.get('attack-type', 'N/A')})"
            else:
                config_str += "no attacks"

            # Final value with % difference from baseline
            final_str = f"{data['final_value']:.2f}"
            if baseline is not None and data['final_value'] is not None:
                diff_pct = ((data['final_value'] - baseline) / abs(baseline)) * 100
                if diff_pct >= 0:
                    final_str += f" [green](+{diff_pct:.1f}%)[/green]"
                else:
                    final_str += f" [red]({diff_pct:.1f}%)[/red]"

            max_str = f"{data['max_value']:.2f}" if data['max_value'] else "N/A"
            mean_str = f"{data['mean_value']:.2f}" if data['mean_value'] else "N/A"
            status = run_info["status"]
            status_str = (
                "[green]OK[/green]"
                if status == "completed"
                else "[red]FAIL[/red]"
            )

            table.add_row(
                run_id,
                config_str,
                final_str,
                max_str,
                mean_str,
                status_str,
            )

        self.console.print("\n")
        self.console.print(table)

    def _display_learning_curves(
        self, runs_data: List[Dict[str, Any]], metric_name: str
    ) -> None:
        """Display learning curves using plotext (optional).

        Args:
            runs_data: List of run data dictionaries
            metric_name: Name of metric being plotted
        """
        try:
            import plotext as plt

            plt.clear_figure()
            plt.title(f"Learning Curves - {metric_name}")
            plt.xlabel("Round")
            plt.ylabel(metric_name)

            for data in runs_data:
                run_id = data["run_id"][:8]  # Short ID for legend
                timeseries = data["timeseries"]

                rounds = [r for r, _ in timeseries]
                values = [v for _, v in timeseries]

                plt.plot(rounds, values, label=run_id)

            plt.show()

        except ImportError:
            self.console.print(
                "\n[yellow]Note: Install 'plotext' for terminal plots[/yellow]"
            )

    def export_comparison(
        self, run_ids: List[str], output_path: str, format: str = "csv"
    ) -> None:
        """Export comparison data to file.

        Args:
            run_ids: List of run IDs
            output_path: Output file path
            format: Export format ('csv', 'json')
        """
        import json

        # Gather all data
        comparison_data = []

        for run_id in run_ids:
            run_info = self.results_store.get_run(run_id)
            if run_info is None:
                continue

            comparison_data.append(
                {
                    "run_id": run_id,
                    "config": run_info["config"],
                    "status": run_info["status"],
                    "start_time": run_info["start_time"],
                    "end_time": run_info["end_time"],
                }
            )

        # Export based on format
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(comparison_data, f, indent=2)
            self.console.print(f"Exported to: {output_path}")

        elif format == "csv":
            try:
                import pandas as pd

                df = pd.DataFrame(comparison_data)
                df.to_csv(output_path, index=False)
                self.console.print(f"Exported to: {output_path}")
            except ImportError:
                self.console.print(
                    "[red]Error: pandas required for CSV export[/red]"
                )

        else:
            self.console.print(f"[red]Unknown format: {format}[/red]")
