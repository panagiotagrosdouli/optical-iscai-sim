# Optical-ISCAI-Sim

A reproducible simulation and dataset-generation framework for **optical Integrated Sensing, Communication and Illumination (ISCAI)** systems based on phase-coded FMCW automotive laser headlamps.

> **Status:** Early research prototype. The physical models, assumptions, and benchmark tasks are still being defined and must be validated against the literature and, where possible, experimental measurements.

## Research objective

Optical-ISCAI-Sim aims to generate large, structured synthetic datasets for studying joint adaptation of:

- optical communication performance;
- FMCW sensing performance;
- adaptive driving-beam illumination;
- energy and reliability trade-offs;
- waveform, power, coding, and beam-control decisions.

The project is intended to provide four connected outputs:

1. a modular simulator;
2. versioned synthetic datasets;
3. benchmark optimization and machine-learning tasks;
4. reproducible baseline results.

## Initial benchmark question

Given the vehicle state, propagation environment, optical-channel state, sensing requirements, and illumination constraints, select system parameters that maximize a configurable multi-objective utility while satisfying safety and reliability constraints.

Candidate decisions include:

- transmit power;
- chirp bandwidth and duration;
- phase-code configuration;
- modulation and coding mode;
- beam width and direction;
- sensing/communication/illumination resource priorities.

## Planned data groups

| Group | Example variables |
|---|---|
| Scenario | distance, relative speed, heading, road geometry, target count |
| Environment | visibility, fog, rain, ambient light, temperature |
| Optical channel | path loss, atmospheric attenuation, received power, noise |
| Communication | SNR, BER, PER, throughput, latency, goodput |
| Sensing | range error, velocity error, detection probability, false-alarm rate |
| Illumination | beam angle, beam width, illuminance, glare/safety constraints |
| Control | waveform, power, coding, beam and priority decisions |

## Repository structure

```text
configs/             Simulation configurations
src/optical_iscai/   Python package
  channel/           Optical propagation and noise models
  communication/     Link metrics and modulation/coding models
  sensing/           FMCW sensing models
  illumination/      Headlamp and ADB models
  optimization/      Labels, objectives, and policies
  data/              Dataset schemas and writers
tests/               Automated tests
docs/                Model assumptions and validation notes
```

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m optical_iscai.generate --config configs/example.yaml
pytest
```

The first implementation generates a small tabular synthetic dataset as an end-to-end pipeline check. It is **not yet a validated physical simulator**.

## Reproducibility principles

- Every dataset release must record its configuration and random seed.
- Variables must include units and documented valid ranges.
- Derived metrics must reference the equations and assumptions used.
- Training, validation, and test splits should be scenario-disjoint where appropriate.
- Dataset size alone is not treated as evidence of scientific quality.

## Roadmap

- [ ] Define the minimum scientifically defensible system model.
- [ ] Implement and test optical propagation/noise modules.
- [ ] Implement PC-FMCW sensing and communication metrics.
- [ ] Add illumination and ADB constraints.
- [ ] Define multi-objective oracle labels.
- [ ] Generate a pilot dataset and perform sanity checks.
- [ ] Add baseline optimization and ML models.
- [ ] Validate against published results.
- [ ] Publish a versioned benchmark dataset and model card.

## Citation

A citation entry will be added when the methodology and first public dataset release are stable.

## Contributing

The project is currently in the design phase. Issues should clearly separate:

- physical-model corrections;
- implementation work;
- dataset-design proposals;
- benchmark and evaluation proposals.
