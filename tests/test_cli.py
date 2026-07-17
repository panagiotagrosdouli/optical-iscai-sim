from pathlib import Path

import pytest

from optical_iscai import cli
from optical_iscai.reporting import ReportPaths


def test_load_experiment_config(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "simulator: package.module:simulate\noutput: results/demo\n",
        encoding="utf-8",
    )

    config = cli.load_experiment_config(path)

    assert config["simulator"] == "package.module:simulate"
    assert config["output"] == "results/demo"


def test_load_experiment_config_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        cli.load_experiment_config(path)


def test_load_experiment_config_requires_keys(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("name: incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="simulator, output"):
        cli.load_experiment_config(path)


def test_load_simulator_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="module:function"):
        cli.load_simulator("not-a-reference")


def test_build_sweep_normalizes_lists() -> None:
    sweep = cli._build_sweep({"snr_db": [-10, 0, 10], "target_count": [1, 2]})

    assert sweep == {"snr_db": (-10, 0, 10), "target_count": (1, 2)}


def test_main_loads_configuration_and_reports_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "simulator: package.module:simulate\noutput: results/demo\n",
        encoding="utf-8",
    )
    root = tmp_path / "report"
    paths = ReportPaths(
        root=root,
        runs_csv=root / "runs.csv",
        runs_parquet=root / "runs.parquet",
        summary_csv=root / "summary.csv",
        summary_parquet=root / "summary.parquet",
        manifest_json=root / "manifest.json",
    )
    captured: dict[str, object] = {}

    def fake_run(config: dict[str, object], *, overwrite: bool | None = None) -> ReportPaths:
        captured["config"] = config
        captured["overwrite"] = overwrite
        return paths

    monkeypatch.setattr(cli, "run_from_config", fake_run)

    assert cli.main([str(config_path), "--overwrite"]) == 0
    assert captured["overwrite"] is True
    assert "Experiment report written to" in capsys.readouterr().out
