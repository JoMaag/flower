"""Utility functions for benchmark framework."""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp and short hash.

    Returns:
        Run ID in format: run_YYYYMMDD_HHMMSS_hash
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add a short random hash for uniqueness
    hash_val = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:4]
    return f"run_{timestamp}_{hash_val}"


def get_git_commit() -> Optional[str]:
    """Get current git commit hash.

    Returns:
        Git commit hash or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name.

    Returns:
        Branch name or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_system_metadata() -> Dict[str, Any]:
    """Collect system metadata for reproducibility.

    Returns:
        Dictionary with system information
    """
    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
    }

    # Try to get package versions
    try:
        import torch
        metadata["torch_version"] = torch.__version__
    except ImportError:
        pass

    try:
        import flwr
        metadata["flwr_version"] = flwr.__version__
    except ImportError:
        pass

    try:
        import numpy
        metadata["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    return metadata


def ensure_results_dir(results_dir: str = "results") -> Path:
    """Ensure results directory exists.

    Args:
        results_dir: Path to results directory

    Returns:
        Path object for results directory
    """
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_directory(run_id: str, results_dir: str = "results") -> Path:
    """Create directory for a specific run.

    Args:
        run_id: Run identifier
        results_dir: Base results directory

    Returns:
        Path to run directory
    """
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_jsonl(data: list, filepath: Path) -> None:
    """Save data as JSON Lines format.

    Args:
        data: List of dictionaries
        filepath: Path to save file
    """
    import json

    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(filepath: Path) -> list:
    """Load data from JSON Lines format.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries
    """
    import json

    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Hash string
    """
    import json

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def parse_duration_string(duration_str: str) -> int:
    """Parse duration string to days.

    Args:
        duration_str: Duration string (e.g., "30d", "2w", "6m")

    Returns:
        Number of days
    """
    duration_str = duration_str.lower().strip()

    if duration_str.endswith("d"):
        return int(duration_str[:-1])
    elif duration_str.endswith("w"):
        return int(duration_str[:-1]) * 7
    elif duration_str.endswith("m"):
        return int(duration_str[:-1]) * 30
    elif duration_str.endswith("y"):
        return int(duration_str[:-1]) * 365
    else:
        # Assume days if no unit
        return int(duration_str)
