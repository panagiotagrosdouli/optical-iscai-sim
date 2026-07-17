"""Command-line runner for reproducible YAML-defined experiments."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from optical_iscai.evaluation import EvaluationConfiguration
from optical_iscai.experiment import (
    MonteCarloConfiguration,
    ParameterValue,
    SimulationFunction,
    run_monte_carlo,
)
from optical_iscai.reporting import ReportPaths, write_experiment_report
from optical_iscai.sequence import SequenceResult


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML document."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    required = ("simulator", "output")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing required configuration keys: {', '.join(missing)}")
    return config


def load_simulator(specification: str) -> SimulationFunction:
    """Resolve a ``module:function`` simulator reference."""
    if not isinstance(specification, str) or ":" not in specification:
        raise ValueError("simulator must use the form 'module:function'")
    module_name, function_name = specification.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("simulator must use the form 'module:function'")
    module = import_module(module_name)
    simulator = getattr(module, function_name)
    if not callable(simulator):
        raise TypeError("configured simulator is not callable")
    return simulator


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _build_sweep(value: Any) -> dict[str, tuple[ParameterValue, ...]]:
    raw = _mapping(value, "sweep")
    sweep: dict[str, tuple[ParameterValue, ...]] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("sweep parameter names must be non-empty strings")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"sweep parameter {name!r} must be a non-empty list")
        sweep[name] = tuple(values)
    return sweep


def run_from_config(
    config: Mapping[str, Any],
    *,
    overwrite: bool | None = None,
) -> ReportPaths:
    """Execute one parsed experiment configuration and persist its report."""
    data = dict(config)
    simulator = load_simulator(str(data["simulator"]))
    fixed_parameters = _mapping(data.get("parameters"), "parameters")

    def configured_simulator(
        swept_parameters: dict[str, ParameterValue],
        rng: np.random.Generator,
    ) -> SequenceResult:
        parameters = {**fixed_parameters, **swept_parameters}
        result = simulator(parameters, rng)
        if not isinstance(result, SequenceResult):
            raise TypeError("simulator must return a SequenceResult")
        return result

    monte_carlo = MonteCarloConfiguration(**_mapping(data.get("monte_carlo"), "monte_carlo"))
    evaluation = EvaluationConfiguration(**_mapping(data.get("evaluation"), "evaluation"))
    result = run_monte_carlo(
        configured_simulator,
        sweep=_build_sweep(data.get("sweep")),
        config=monte_carlo,
        evaluation_config=evaluation,
    )
    configured_overwrite = bool(data.get("overwrite", False))
    return write_experiment_report(
        result,
        Path(str(data["output"])),
        experiment_name=data.get("name"),
        monte_carlo_config=monte_carlo,
        metadata=_mapping(data.get("metadata"), "metadata"),
        overwrite=configured_overwrite if overwrite is None else overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Optical-ISCAI Monte Carlo experiment")
    parser.add_argument("config", type=Path, help="YAML experiment configuration")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing report files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_from_config(load_experiment_config(args.config), overwrite=args.overwrite or None)
    print(f"Experiment report written to {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
