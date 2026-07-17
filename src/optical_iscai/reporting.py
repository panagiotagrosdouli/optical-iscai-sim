"""Persistence utilities for reproducible Monte Carlo experiment results.

The reporting layer writes normalized per-run metrics, grouped summaries, and a
small JSON manifest into one experiment directory.  It deliberately keeps figure
generation separate so numerical results remain usable in headless environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from optical_iscai.experiment import ExperimentResult, MonteCarloConfiguration


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """Paths created for one persisted experiment report."""

    root: Path
    runs_csv: Path
    runs_parquet: Path
    summary_csv: Path
    summary_parquet: Path
    manifest_json: Path


def _json_safe(value: Any) -> Any:
    """Convert nested values to strict JSON-compatible Python objects."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_experiment_report(
    result: ExperimentResult,
    output_directory: str | Path,
    *,
    experiment_name: str | None = None,
    monte_carlo_config: MonteCarloConfiguration | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> ReportPaths:
    """Write run-level metrics, summaries, and a reproducibility manifest.

    Parameters
    ----------
    result:
        Completed Monte Carlo result.
    output_directory:
        Directory to create or reuse.
    experiment_name:
        Optional human-readable experiment identifier stored in the manifest.
    monte_carlo_config:
        Optional execution configuration stored in the manifest.
    metadata:
        Additional JSON-compatible research metadata.
    overwrite:
        Permit replacing report files that already exist.
    """
    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")

    root = Path(output_directory)
    paths = ReportPaths(
        root=root,
        runs_csv=root / "runs.csv",
        runs_parquet=root / "runs.parquet",
        summary_csv=root / "summary.csv",
        summary_parquet=root / "summary.parquet",
        manifest_json=root / "manifest.json",
    )
    files = (
        paths.runs_csv,
        paths.runs_parquet,
        paths.summary_csv,
        paths.summary_parquet,
        paths.manifest_json,
    )
    existing = [path for path in files if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"report files already exist: {names}")

    root.mkdir(parents=True, exist_ok=True)
    runs = result.to_dataframe()
    summary = result.summarize()
    runs.to_csv(paths.runs_csv, index=False)
    runs.to_parquet(paths.runs_parquet, index=False)
    summary.to_csv(paths.summary_csv, index=False)
    summary.to_parquet(paths.summary_parquet, index=False)

    manifest = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "run_count": len(result.runs),
        "condition_count": len({run.condition_index for run in result.runs}),
        "parameter_names": list(result.parameter_names),
        "monte_carlo": (
            asdict(monte_carlo_config) if monte_carlo_config is not None else None
        ),
        "metadata": dict(metadata or {}),
        "files": {
            "runs_csv": paths.runs_csv.name,
            "runs_parquet": paths.runs_parquet.name,
            "summary_csv": paths.summary_csv.name,
            "summary_parquet": paths.summary_parquet.name,
        },
    }
    with paths.manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    return paths
