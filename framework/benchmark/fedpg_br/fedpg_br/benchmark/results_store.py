"""Results storage using SQLite database."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fedpg_br.benchmark.metrics_collector import MetricRecord


class ResultsStore:
    """SQLite-based storage for benchmark runs and metrics.

    This class manages a SQLite database for persisting experiment results,
    including run metadata, time-series metrics, and tags.
    """

    def __init__(self, db_path: str = "results/.benchmark_db.sqlite"):
        """Initialize results store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                suite_name TEXT,
                config_hash TEXT,
                config_json TEXT,
                git_commit TEXT,
                start_time REAL,
                end_time REAL,
                status TEXT,
                error_message TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                round_num INTEGER,
                metric_name TEXT,
                metric_value REAL,
                client_id INTEGER,
                timestamp REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                run_id TEXT,
                tag TEXT,
                PRIMARY KEY (run_id, tag),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for efficient querying
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_suite ON runs(suite_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_round ON metrics(run_id, round_num)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(run_id, metric_name)"
        )

        self.conn.commit()

    def create_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        suite_name: Optional[str] = None,
        git_commit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Create a new run entry.

        Args:
            run_id: Unique run identifier
            config: Run configuration dictionary
            suite_name: Name of benchmark suite (if applicable)
            git_commit: Git commit hash
            metadata: Additional metadata
            tags: List of tags for this run
        """
        cursor = self.conn.cursor()

        # Compute config hash
        config_json = json.dumps(config, sort_keys=True)
        config_hash = str(hash(config_json))

        # Insert run
        cursor.execute(
            """
            INSERT INTO runs
            (run_id, suite_name, config_hash, config_json, git_commit,
             start_time, status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                suite_name,
                config_hash,
                config_json,
                git_commit,
                time.time(),
                "running",
                json.dumps(metadata or {}),
            ),
        )

        # Insert tags
        if tags:
            for tag in tags:
                cursor.execute(
                    "INSERT OR IGNORE INTO tags (run_id, tag) VALUES (?, ?)",
                    (run_id, tag),
                )

        self.conn.commit()

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update run status.

        Args:
            run_id: Run identifier
            status: New status ('running', 'completed', 'failed')
            error_message: Error message if status is 'failed'
        """
        cursor = self.conn.cursor()

        end_time = time.time() if status in ("completed", "failed") else None

        cursor.execute(
            """
            UPDATE runs
            SET status = ?, end_time = ?, error_message = ?
            WHERE run_id = ?
            """,
            (status, end_time, error_message, run_id),
        )

        self.conn.commit()

    def store_metrics(self, records: List[MetricRecord]) -> None:
        """Store metric records.

        Args:
            records: List of MetricRecord objects
        """
        cursor = self.conn.cursor()

        for record in records:
            cursor.execute(
                """
                INSERT INTO metrics
                (run_id, round_num, metric_name, metric_value, client_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.round_num,
                    record.metric_name,
                    record.metric_value,
                    record.client_id,
                    record.timestamp,
                ),
            )

        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run information.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with run information or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "run_id": row["run_id"],
            "suite_name": row["suite_name"],
            "config": json.loads(row["config_json"]),
            "git_commit": row["git_commit"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "status": row["status"],
            "error_message": row["error_message"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def get_runs(
        self,
        suite_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query runs with optional filtering.

        Args:
            suite_name: Filter by suite name
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of run dictionaries
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM runs WHERE 1=1"
        params = []

        if suite_name:
            query += " AND suite_name = ?"
            params.append(suite_name)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "run_id": row["run_id"],
                "suite_name": row["suite_name"],
                "config": json.loads(row["config_json"]),
                "git_commit": row["git_commit"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_metrics(
        self,
        run_id: str,
        metric_name: Optional[str] = None,
        client_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics for a run.

        Args:
            run_id: Run identifier
            metric_name: Filter by specific metric name
            client_id: Filter by client ID (None for server metrics)

        Returns:
            List of metric dictionaries
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM metrics WHERE run_id = ?"
        params = [run_id]

        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)
        else:
            query += " AND client_id IS NULL"

        query += " ORDER BY round_num"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "round_num": row["round_num"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "client_id": row["client_id"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_metric_timeseries(
        self, run_id: str, metric_name: str
    ) -> List[tuple[int, float]]:
        """Get time series for a specific metric.

        Args:
            run_id: Run identifier
            metric_name: Metric name

        Returns:
            List of (round_num, value) tuples
        """
        metrics = self.get_metrics(run_id, metric_name=metric_name)
        return [(m["round_num"], m["metric_value"]) for m in metrics]

    def get_runs_by_tag(self, tag: str) -> List[str]:
        """Get run IDs with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of run IDs
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT run_id FROM tags WHERE tag = ?", (tag,))
        return [row[0] for row in cursor.fetchall()]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated data.

        Args:
            run_id: Run identifier
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def cleanup_old_runs(self, days: int) -> int:
        """Delete runs older than specified days.

        Args:
            days: Number of days

        Returns:
            Number of runs deleted
        """
        cursor = self.conn.cursor()
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        cursor.execute(
            "SELECT run_id FROM runs WHERE start_time < ?", (cutoff_time,)
        )
        run_ids = [row[0] for row in cursor.fetchall()]

        for run_id in run_ids:
            self.delete_run(run_id)

        return len(run_ids)

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
