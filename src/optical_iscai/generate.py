"""Pilot dataset generator.

This module provides an end-to-end pipeline check. The equations are deliberately
simple and must not be treated as a validated optical ISCAI channel model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _uniform(rng: np.random.Generator, bounds: list[float], size: int) -> np.ndarray:
    low, high = map(float, bounds)
    return rng.uniform(low, high, size=size)


def _integers(rng: np.random.Generator, bounds: list[int], size: int) -> np.ndarray:
    low, high = map(int, bounds)
    return rng.integers(low, high + 1, size=size)


def generate_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Generate a pilot tabular dataset from a parsed configuration."""
    n = int(config["samples"])
    rng = np.random.default_rng(int(config["seed"]))

    scenario = config["scenario"]
    environment = config["environment"]
    transmitter = config["transmitter"]
    receiver = config["receiver"]

    distance_m = _uniform(rng, scenario["distance_m"], n)
    relative_speed_mps = _uniform(rng, scenario["relative_speed_mps"], n)
    target_count = _integers(rng, scenario["target_count"], n)
    visibility_m = _uniform(rng, environment["visibility_m"], n)
    ambient_lux = _uniform(rng, environment["ambient_illuminance_lux"], n)
    rain_rate = _uniform(rng, environment["rain_rate_mm_h"], n)
    transmit_power_w = _uniform(rng, transmitter["power_w"], n)
    divergence_mrad = _uniform(rng, transmitter["beam_divergence_mrad"], n)

    # Placeholder attenuation terms for pipeline validation only.
    geometric_loss_db = 20.0 * np.log10(np.maximum(distance_m, 1e-6))
    weather_loss_db = 4.343 * distance_m / np.maximum(visibility_m, 1.0)
    rain_loss_db = 0.002 * rain_rate * distance_m / 1000.0
    beam_penalty_db = 10.0 * np.log10(np.maximum(divergence_mrad, 1e-6))
    transmit_power_dbm = 10.0 * np.log10(transmit_power_w * 1000.0)

    received_power_dbm = (
        transmit_power_dbm
        - geometric_loss_db
        - weather_loss_db
        - rain_loss_db
        - beam_penalty_db
    )
    noise_floor_dbm = float(receiver["noise_floor_dbm"])
    snr_db = received_power_dbm - noise_floor_dbm

    # Smooth proxy metrics, not validated link-level equations.
    ber_proxy = 0.5 * np.exp(-np.maximum(10.0 ** (snr_db / 10.0), 0.0))
    detection_probability_proxy = 1.0 / (1.0 + np.exp(-(snr_db - 8.0) / 3.0))
    throughput_mbps_proxy = 1000.0 * np.clip(1.0 - ber_proxy, 0.0, 1.0)

    return pd.DataFrame(
        {
            "distance_m": distance_m,
            "relative_speed_mps": relative_speed_mps,
            "target_count": target_count,
            "visibility_m": visibility_m,
            "ambient_illuminance_lux": ambient_lux,
            "rain_rate_mm_h": rain_rate,
            "transmit_power_w": transmit_power_w,
            "beam_divergence_mrad": divergence_mrad,
            "received_power_dbm_proxy": received_power_dbm,
            "snr_db_proxy": snr_db,
            "ber_proxy": ber_proxy,
            "detection_probability_proxy": detection_probability_proxy,
            "throughput_mbps_proxy": throughput_mbps_proxy,
            "model_status": "unvalidated_pilot",
        }
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pilot Optical-ISCAI dataset")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = generate_dataset(config)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    print(f"Wrote {len(dataset):,} rows to {output}")


if __name__ == "__main__":
    main()
