# Paper-Derived System Parameters

Source paper: **Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination** (IEEE Photonics Technology Letters, accepted version, DOI: 10.1109/LPT.2025.3649597).

This document records only values and design choices explicitly stated in the paper. Parameters not reported by the authors are marked as `not specified` and must not be silently invented.

## Fixed waveform and communication parameters

| Parameter | Symbol | Paper value | SI value | Status |
|---|---:|---:|---:|---|
| Optical carrier frequency | `f_c` | 193.4 THz | `193.4e12 Hz` | Explicit |
| FMCW chirp bandwidth | `B` | 10 GHz | `10e9 Hz` | Explicit |
| Chirp period/duration | `T_chirp` | 10 us | `10e-6 s` | Explicit |
| Data rate | `R_b` | 1 Gbit/s | `1e9 bit/s` | Explicit |
| Data modulation | — | DPSK | — | Explicit |
| Waveform | — | Phase-coded FMCW | — | Explicit |
| Receiver architecture | — | Coherent detection | — | Explicit |
| Chirp slope | `K = B/T_chirp` | Derived | `1e15 Hz/s` | Derived |
| Symbol duration | `T_s = 1/R_b` | Derived | `1 ns` | Derived |
| Symbols per chirp | `T_chirp/T_s` | Derived | `10,000` | Derived |
| Carrier wavelength | `lambda = c/f_c` | Derived | approximately `1.550e-6 m` | Derived |
| Ideal range resolution | `Delta_R = c/(2B)` | Derived | approximately `0.015 m` | Derived baseline |

The reported maximum ranging error in the multi-target simulation is **3.8 cm**. This is a validation result, not the same quantity as the ideal bandwidth-limited range resolution.

## Signal-processing chain

1. DPSK symbols are embedded in the phase of an FMCW optical chirp.
2. The target receiver demodulates the communication stream.
3. Reflected light returns to the ego vehicle for sensing.
4. Coherent mixing with a local oscillator produces an IF signal.
5. A group-delay filter compensates phase-code perturbations.
6. A two-dimensional FFT forms a range-Doppler map.
7. Two-dimensional CA-CFAR performs adaptive detection.
8. Multidimensional Hough-transform TBD reconstructs trajectories.

## Sensing and detection parameters

| Parameter or method | Paper value | Status |
|---|---|---|
| Range-Doppler processing | 2D FFT | Explicit |
| Detector | 2D CA-CFAR | Explicit |
| Tracking strategy | Track-before-detect | Explicit |
| Track extractor | Multidimensional Hough Transform | Explicit |
| Hough projections | `xy`, `xt`, `yt` | Explicit |
| Accumulator smoothing | 3 x 3 mean filter | Explicit |
| Projection fusion | AND-logic intersection | Explicit |
| Motion handling | Rolling-window piecewise-linear reconstruction | Explicit |
| Association cost | Position and kinematic consistency | Explicit |
| CA-CFAR training cells | not specified | Missing |
| CA-CFAR guard cells | not specified | Missing |
| CA-CFAR target false-alarm probability | not specified | Missing |
| Hough vote threshold | not specified | Missing |
| Supporting-point distance threshold | not specified | Missing |
| Minimum common supporting points | not specified | Missing |
| Rolling-window length | not specified | Missing |
| Stitching threshold | not specified | Missing |
| Position/kinematic cost weights | not specified | Missing |
| Number of chirps | not specified | Missing |
| Coherent integration time | not specified | Missing |
| Sampling rate and FFT sizes | not specified | Missing |

## Illumination subsystem

| Parameter or method | Paper value | Status |
|---|---|---|
| Illumination mode | Adaptive Driving Beam | Explicit |
| Lighting conversion | Blue laser to white light using phosphor | Explicit |
| Glare-control basis | SAE J3069 | Explicit |
| Zones | Non-glare and transition zones | Explicit |
| Transition profile | Raised-cosine intensity adjustment | Explicit |
| Target localization inputs | Camera plus PC-FMCW range | Explicit |
| Minimum/maximum transition distances | not specified | Missing |
| Safety margin | not specified | Missing |
| Camera-headlamp lateral offset | not specified | Missing |
| Beam angular limits | not specified | Missing |
| Optical power split among functions | not specified | Missing |

## Validation scenarios and reported outcomes

### Adaptive illumination

- Oncoming-vehicle scenario:
  - Ego speed: `40 km/h`
  - Oncoming vehicle speed: `30 km/h`
  - Initial separation: `150 m`
- Multiple-preceding-vehicle scenario:
  - Ego speed: `50 km/h`
  - Two vehicles in adjacent lanes
  - Initial spacing: `30 m`

### Tracking

- Scenario 1 uses a `100 x 100` 2D space with two linear tracks, Gaussian noise, and random clutter.
- Scenario 2 includes one linear and one nonlinear track in dense clutter.
- Reported linear-track parameter errors: `0.1251` and `0.2348` units.
- Reported mean trajectory deviation: `1.6787` units.

## Important reproducibility limitation

The paper does **not** provide enough numerical parameters to reproduce every physical and signal-processing layer uniquely. The repository must therefore distinguish among:

- `paper_explicit`: directly stated by the authors;
- `paper_derived`: obtained algebraically from explicit values;
- `external_model`: supplied by a separately cited physical/channel standard;
- `scenario_assumption`: user-controlled simulation choice;
- `calibrated`: fitted to reproduce a reported figure or metric.

Every generated dataset column and configuration parameter should carry one of these provenance labels.