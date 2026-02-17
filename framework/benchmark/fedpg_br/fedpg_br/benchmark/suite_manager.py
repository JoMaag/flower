"""Suite manager for benchmark suites."""

import itertools
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, ValidationError


class SuiteConfig(BaseModel):
    """Pydantic model for suite configuration validation."""

    name: str = Field(..., description="Suite name")
    description: str = Field(..., description="Suite description")
    version: str = Field(default="1.0", description="Suite version")
    base_config: Dict[str, Any] = Field(..., description="Base configuration")
    parameter_matrix: Dict[str, List[Any]] = Field(
        default_factory=dict, description="Parameter matrix for cartesian product"
    )
    parameter_list: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of parameter combinations"
    )
    metrics_to_track: List[str] = Field(
        default_factory=list, description="Metrics to track"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for this suite")


class SuiteManager:
    """Manager for loading and expanding benchmark suites."""

    def load_suite(self, suite_path: Path) -> Dict[str, Any]:
        """Load a benchmark suite from YAML file.

        Args:
            suite_path: Path to suite YAML file

        Returns:
            Dictionary with suite configuration

        Raises:
            FileNotFoundError: If suite file doesn't exist
            ValidationError: If suite configuration is invalid
        """
        if not suite_path.exists():
            raise FileNotFoundError(f"Suite file not found: {suite_path}")

        with open(suite_path, "r") as f:
            suite_data = yaml.safe_load(f)

        # Validate with Pydantic
        try:
            suite_config = SuiteConfig(**suite_data)
            return suite_config.model_dump()
        except ValidationError as e:
            raise ValueError(f"Invalid suite configuration: {e}")

    def expand_suite(self, suite: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand suite into list of individual run configurations.

        Args:
            suite: Suite dictionary

        Returns:
            List of run configurations
        """
        base_config = suite.get("base_config", {})
        parameter_matrix = suite.get("parameter_matrix", {})
        parameter_list = suite.get("parameter_list", [])

        run_configs = []

        if parameter_matrix:
            # Generate cartesian product of parameter matrix
            keys = list(parameter_matrix.keys())
            values = [parameter_matrix[k] for k in keys]

            for combination in itertools.product(*values):
                config = base_config.copy()
                for key, value in zip(keys, combination):
                    config[key] = value
                run_configs.append(config)

        if parameter_list:
            # Use explicit parameter list
            for params in parameter_list:
                config = base_config.copy()
                config.update(params)
                run_configs.append(config)

        # If neither matrix nor list, return just base config
        if not run_configs:
            run_configs = [base_config]

        return run_configs

    def validate_suite(self, suite_path: Path) -> tuple[bool, str]:
        """Validate a benchmark suite.

        Args:
            suite_path: Path to suite YAML file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            suite = self.load_suite(suite_path)
            run_configs = self.expand_suite(suite)

            if not run_configs:
                return False, "Suite expands to zero run configurations"

            return True, f"Valid suite with {len(run_configs)} configurations"

        except Exception as e:
            return False, str(e)
