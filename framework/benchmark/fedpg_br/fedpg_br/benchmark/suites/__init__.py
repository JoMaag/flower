"""Pre-defined benchmark suites."""

import os
from pathlib import Path

# Directory containing suite YAML files
SUITES_DIR = Path(__file__).parent

def get_suite_path(suite_name: str) -> Path:
    """Get the path to a suite YAML file."""
    return SUITES_DIR / f"{suite_name}.yaml"

def list_available_suites() -> list[str]:
    """List all available suite names."""
    return [
        f.stem for f in SUITES_DIR.glob("*.yaml")
    ]
