"""Command-line interface for FedPG-BR benchmark."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from fedpg_br.benchmark.results_store import ResultsStore
from fedpg_br.benchmark.run_manager import RunManager
from fedpg_br.benchmark.suites import get_suite_path, list_available_suites
from fedpg_br.benchmark.utils import parse_duration_string

app = typer.Typer(
    name="fedpg-benchmark",
    help="FedPG-BR Benchmark Framework - Gym-like benchmarks for federated RL",
    add_completion=False,
)

console = Console()

# Subcommands
suite_app = typer.Typer(help="Manage benchmark suites")
app.add_typer(suite_app, name="suite")


@app.command()
def run(
    suite_or_config: str = typer.Argument(
        ..., help="Suite name or path to config file"
    ),
    live: bool = typer.Option(
        False, "--live", help="Enable real-time visualization"
    ),
    output_dir: str = typer.Option(
        "results", "--output-dir", "-o", help="Output directory for results"
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save results to database"
    ),
):
    """Run a benchmark suite or single experiment.

    Examples:
        fedpg-benchmark run byzantine-robustness --live
        fedpg-benchmark run my_config.toml
    """
    console.print(f"[bold]FedPG-BR Benchmark Runner[/bold]\n")

    # Check if it's a suite name or config file
    path = Path(suite_or_config)
    is_suite = False

    if not path.exists():
        # Try as suite name
        try:
            suite_path = get_suite_path(suite_or_config)
            if suite_path.exists():
                is_suite = True
                console.print(f"Loading suite: [cyan]{suite_or_config}[/cyan]")
            else:
                console.print(
                    f"[red]Error:[/red] Neither file nor suite found: {suite_or_config}"
                )
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    else:
        console.print(f"Loading config: [cyan]{path}[/cyan]")

    # Initialize results store
    results_store = ResultsStore(f"{output_dir}/.benchmark_db.sqlite")

    try:
        if is_suite:
            # Import suite manager
            from fedpg_br.benchmark.suite_manager import SuiteManager

            suite_manager = SuiteManager()
            suite = suite_manager.load_suite(get_suite_path(suite_or_config))

            console.print(f"\n[bold]{suite['name']}[/bold]")
            console.print(f"{suite['description']}\n")

            # Generate all run configs
            run_configs = suite_manager.expand_suite(suite)
            console.print(
                f"Suite contains [cyan]{len(run_configs)}[/cyan] experiments\n"
            )

            # Run all experiments
            run_manager = RunManager(
                results_store, results_dir=output_dir, enable_visualization=live
            )

            for i, config in enumerate(run_configs, 1):
                console.print(
                    f"[bold]Running experiment {i}/{len(run_configs)}[/bold]"
                )
                result = run_manager.run_experiment(
                    config, suite_name=suite["name"], tags=suite.get("tags", [])
                )

                if result["success"]:
                    console.print(
                        f"  [green]OK[/green] Completed: {result['run_id']}\n"
                    )
                else:
                    console.print(
                        f"  [red]✗[/red] Failed: {result.get('error', 'Unknown error')}\n"
                    )

            console.print(f"\n[bold green]Suite completed![/bold green]")
            console.print(
                f"View results: [cyan]fedpg-benchmark compare --suite {suite_or_config}[/cyan]"
            )

        else:
            # Single experiment from config file
            import tomli

            with open(path, "rb") as f:
                config_data = tomli.load(f)

            # Extract the actual run config
            # If it's a pyproject.toml, extract [tool.flwr.app.config]
            if "tool" in config_data and "flwr" in config_data["tool"]:
                config = config_data["tool"]["flwr"]["app"]["config"]
            else:
                # Otherwise use the whole file
                config = config_data

            run_manager = RunManager(
                results_store, results_dir=output_dir, enable_visualization=live
            )

            result = run_manager.run_experiment(config)

            if result["success"]:
                console.print(f"\n[bold green]OK Run completed![/bold green]")
                console.print(f"Run ID: [cyan]{result['run_id']}[/cyan]")
                console.print(f"Results saved to: {result['run_dir']}")

                # Display summary
                summary = result.get("summary", {})
                if summary:
                    console.print("\n[bold]Summary:[/bold]")
                    console.print(
                        f"  Total rounds: {summary.get('total_rounds', 'N/A')}"
                    )
                    console.print(
                        f"  Elapsed time: {summary.get('elapsed_time', 0):.1f}s"
                    )

                    final_server = summary.get("final_server_metrics", {})
                    if "avg_reward" in final_server:
                        console.print(
                            f"  Final reward: {final_server['avg_reward']:.2f}"
                        )

            else:
                console.print(f"\n[bold red]✗ Run failed![/bold red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")
                raise typer.Exit(1)

    finally:
        results_store.close()


@suite_app.command("list")
def suite_list():
    """List all available benchmark suites."""
    console.print("[bold]Available Benchmark Suites:[/bold]\n")

    suites = list_available_suites()

    if not suites:
        console.print("[yellow]No suites found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Suite Name", style="cyan")
    table.add_column("Description")

    for suite_name in sorted(suites):
        try:
            from fedpg_br.benchmark.suite_manager import SuiteManager

            suite_manager = SuiteManager()
            suite = suite_manager.load_suite(get_suite_path(suite_name))
            description = suite.get("description", "No description")
        except Exception:
            description = "Error loading suite"

        table.add_row(suite_name, description)

    console.print(table)


@suite_app.command("show")
def suite_show(suite_name: str = typer.Argument(..., help="Suite name")):
    """Show detailed information about a suite."""
    try:
        from fedpg_br.benchmark.suite_manager import SuiteManager

        suite_manager = SuiteManager()
        suite_path = get_suite_path(suite_name)

        if not suite_path.exists():
            console.print(f"[red]Error:[/red] Suite not found: {suite_name}")
            raise typer.Exit(1)

        suite = suite_manager.load_suite(suite_path)
        run_configs = suite_manager.expand_suite(suite)

        console.print(f"\n[bold]{suite['name']}[/bold]")
        console.print(f"{suite['description']}\n")

        console.print(f"[bold]Configuration:[/bold]")
        console.print(f"  Total experiments: [cyan]{len(run_configs)}[/cyan]")

        if "base_config" in suite:
            console.print("\n  Base config:")
            for key, value in suite["base_config"].items():
                console.print(f"    {key}: {value}")

        if "parameter_matrix" in suite:
            console.print("\n  Parameter matrix:")
            for key, values in suite["parameter_matrix"].items():
                console.print(f"    {key}: {values}")

        if "tags" in suite:
            console.print(f"\n  Tags: {', '.join(suite['tags'])}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def compare(
    runs: Optional[list[str]] = typer.Option(
        None, "--runs", "-r", help="Run IDs to compare"
    ),
    suite: Optional[str] = typer.Option(
        None, "--suite", "-s", help="Compare all runs from a suite"
    ),
    latest: int = typer.Option(
        None, "--latest", "-n", help="Compare latest N runs"
    ),
    metric: str = typer.Option(
        "avg_reward", "--metric", "-m", help="Metric to compare"
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", "-e", help="Export comparison to file"
    ),
):
    """Compare results from multiple runs.

    Examples:
        fedpg-benchmark compare --runs run_001 run_002 run_003
        fedpg-benchmark compare --suite byzantine-robustness --latest 3
    """
    console.print("[bold]Experiment Comparison[/bold]\n")

    # Initialize results store
    results_store = ResultsStore()

    try:
        # Determine which runs to compare
        run_ids = []

        if runs:
            run_ids = list(runs)
        elif suite and latest:
            all_runs = results_store.get_runs(suite_name=suite, status="completed")
            run_ids = [r["run_id"] for r in all_runs[:latest]]
        elif suite:
            all_runs = results_store.get_runs(suite_name=suite, status="completed")
            run_ids = [r["run_id"] for r in all_runs]
        elif latest:
            all_runs = results_store.get_runs(status="completed", limit=latest)
            run_ids = [r["run_id"] for r in all_runs]
        else:
            console.print(
                "[red]Error:[/red] Must specify --runs, --suite, or --latest"
            )
            raise typer.Exit(1)

        if not run_ids:
            console.print("[yellow]No runs found to compare[/yellow]")
            return

        console.print(f"Comparing [cyan]{len(run_ids)}[/cyan] runs\n")

        # Import comparator
        from fedpg_br.benchmark.comparator import Comparator

        comparator = Comparator(results_store)
        comparator.compare_runs(run_ids, metric_name=metric)

        # Export if requested
        if export:
            console.print(f"\nExporting comparison to: {export}")
            # TODO: Implement export

    finally:
        results_store.close()


@app.command()
def cleanup(
    older_than: str = typer.Option(
        "30d", "--older-than", help="Delete runs older than (e.g., 30d, 2w, 6m)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without deleting"
    ),
):
    """Clean up old benchmark runs.

    Examples:
        fedpg-benchmark cleanup --older-than 30d
        fedpg-benchmark cleanup --older-than 2w --dry-run
    """
    days = parse_duration_string(older_than)
    console.print(f"[bold]Cleanup:[/bold] Runs older than {days} days\n")

    results_store = ResultsStore()

    try:
        if dry_run:
            console.print("[yellow]Dry run mode - no data will be deleted[/yellow]\n")
            # TODO: Implement dry run listing

        deleted_count = results_store.cleanup_old_runs(days)
        console.print(f"Deleted [cyan]{deleted_count}[/cyan] runs")

    finally:
        results_store.close()


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
