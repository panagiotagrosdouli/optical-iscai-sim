"""Reproducible Monte Carlo experiment orchestration.

The engine is intentionally independent of a particular scenario sampler.  A caller
provides a simulation function accepting one parameter dictionary and one random
number generator and returning a :class:`SequenceResult`.  The engine expands a
Cartesian parameter grid, repeats every condition, evaluates each sequence, and
returns normalized per-run metrics suitable for tables and plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from optical_iscai.evaluation import EvaluationConfiguration, evaluate_sequence
from optical_iscai.sequence import SequenceResult

ParameterValue = int | float | str | bool
SimulationFunction = Callable[[dict[str, ParameterValue], np.random.Generator], SequenceResult]


@dataclass(frozen=True, slots=True)
class MonteCarloConfiguration:
    """Execution settings for a Cartesian parameter sweep."""

    repetitions: int = 10
    seed: int = 0

    def validate(self) -> None:
        if isinstance(self.repetitions, bool) or int(self.repetitions) != self.repetitions:
            raise ValueError("repetitions must be an integer")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least one")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Metrics and metadata for one independent simulation run."""

    condition_index: int
    repetition_index: int
    run_seed: int
    parameters: dict[str, ParameterValue]
    ground_truth_count: int
    track_count: int
    matched_count: int
    missed_count: int
    false_track_count: int
    precision: float
    recall: float
    f1_score: float
    range_rmse_m: float
    velocity_rmse_m_s: float

    def as_record(self) -> dict[str, ParameterValue | int | float]:
        """Return one flat record for DataFrame/CSV/Parquet export."""
        return {
            "condition_index": self.condition_index,
            "repetition_index": self.repetition_index,
            "run_seed": self.run_seed,
            **self.parameters,
            "ground_truth_count": self.ground_truth_count,
            "track_count": self.track_count,
            "matched_count": self.matched_count,
            "missed_count": self.missed_count,
            "false_track_count": self.false_track_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "range_rmse_m": self.range_rmse_m,
            "velocity_rmse_m_s": self.velocity_rmse_m_s,
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete collection of Monte Carlo runs."""

    runs: tuple[ExperimentRun, ...]
    parameter_names: tuple[str, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per run with parameters and evaluation metrics."""
        return pd.DataFrame([run.as_record() for run in self.runs])

    def summarize(self) -> pd.DataFrame:
        """Aggregate numeric metrics by swept parameter condition.

        The output contains mean, standard deviation and run count for every
        numeric metric.  With no swept parameters, a single global summary row is
        returned.
        """
        frame = self.to_dataframe()
        if frame.empty:
            return frame
        metric_columns = [
            "precision",
            "recall",
            "f1_score",
            "range_rmse_m",
            "velocity_rmse_m_s",
            "matched_count",
            "missed_count",
            "false_track_count",
        ]
        if self.parameter_names:
            grouped = frame.groupby(list(self.parameter_names), dropna=False, sort=True)
            summary = grouped[metric_columns].agg(["mean", "std", "count"]).reset_index()
            summary.columns = [
                column if isinstance(column, str) else "_".join(part for part in column if part)
                for column in summary.columns
            ]
            return summary

        values: dict[str, float | int] = {}
        for metric in metric_columns:
            values[f"{metric}_mean"] = float(frame[metric].mean())
            values[f"{metric}_std"] = float(frame[metric].std())
            values[f"{metric}_count"] = int(frame[metric].count())
        return pd.DataFrame([values])


def parameter_grid(
    sweep: Mapping[str, Sequence[ParameterValue]] | None,
) -> tuple[dict[str, ParameterValue], ...]:
    """Expand a deterministic Cartesian product of named sweep values."""
    if sweep is None or len(sweep) == 0:
        return ({},)

    names = tuple(sweep.keys())
    value_lists: list[tuple[ParameterValue, ...]] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("sweep parameter names must be non-empty strings")
        values = tuple(sweep[name])
        if not values:
            raise ValueError(f"sweep parameter {name!r} must contain at least one value")
        value_lists.append(values)

    return tuple(
        dict(zip(names, combination, strict=True))
        for combination in product(*value_lists)
    )


def _f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    return float(2.0 * precision * recall / denominator)


def run_monte_carlo(
    simulator: SimulationFunction,
    *,
    sweep: Mapping[str, Sequence[ParameterValue]] | None = None,
    config: MonteCarloConfiguration | None = None,
    evaluation_config: EvaluationConfiguration | None = None,
) -> ExperimentResult:
    """Execute and evaluate all conditions and repetitions reproducibly."""
    cfg = MonteCarloConfiguration() if config is None else config
    cfg.validate()
    conditions = parameter_grid(sweep)
    parameter_names = tuple(sweep.keys()) if sweep else ()

    total_runs = len(conditions) * int(cfg.repetitions)
    seed_sequence = np.random.SeedSequence(int(cfg.seed))
    child_sequences = seed_sequence.spawn(total_runs)

    outputs: list[ExperimentRun] = []
    run_index = 0
    for condition_index, parameters in enumerate(conditions):
        for repetition_index in range(int(cfg.repetitions)):
            child = child_sequences[run_index]
            run_index += 1
            run_seed = int(child.generate_state(1, dtype=np.uint32)[0])
            rng = np.random.default_rng(child)
            sequence = simulator(dict(parameters), rng)
            if not isinstance(sequence, SequenceResult):
                raise TypeError("simulator must return a SequenceResult")
            evaluation = evaluate_sequence(sequence, evaluation_config)
            outputs.append(
                ExperimentRun(
                    condition_index=condition_index,
                    repetition_index=repetition_index,
                    run_seed=run_seed,
                    parameters=dict(parameters),
                    ground_truth_count=evaluation.ground_truth_count,
                    track_count=evaluation.track_count,
                    matched_count=evaluation.matched_count,
                    missed_count=evaluation.missed_count,
                    false_track_count=evaluation.false_track_count,
                    precision=float(evaluation.precision),
                    recall=float(evaluation.recall),
                    f1_score=_f1_score(evaluation.precision, evaluation.recall),
                    range_rmse_m=float(evaluation.range_rmse_m),
                    velocity_rmse_m_s=float(evaluation.velocity_rmse_m_s),
                )
            )

    return ExperimentResult(runs=tuple(outputs), parameter_names=parameter_names)
