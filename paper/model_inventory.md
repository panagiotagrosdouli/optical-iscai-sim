# Model Inventory and Implementation Plan

This file translates the source paper into implementable simulator components without reproducing the paper verbatim.

## 1. PC-FMCW waveform

### Required inputs

- optical carrier frequency
- chirp bandwidth
- chirp duration
- sample rate
- number of chirps
- DPSK bit sequence and data rate
- initial phase and amplitude

### Required outputs

- local-oscillator complex waveform
- phase-coded transmit waveform
- delayed/Doppler-shifted target returns
- mixed intermediate-frequency waveform

### Paper coverage

The analytical signal structure and four fixed headline parameters are provided. Sampling rate, waveform amplitude, chirp count, phase-noise model, and hardware nonidealities are not provided.

## 2. Communication receiver

### Required operations

- carrier/bin extraction using an FFT
- differential phase demodulation
- bit recovery
- BER calculation against transmitted bits
- optional packetization for PER, throughput, and latency

### Missing choices

The paper reports a 1 Gbit/s data rate but does not provide a full noise/link-budget model or packet protocol. BER, PER, throughput under impairments, and latency therefore require separately documented models.

## 3. Sensing receiver

### Required operations

- coherent mixing
- low-pass filtering
- phase-code compensation using a group-delay filter
- range FFT across fast time
- Doppler FFT across chirps
- power range-Doppler map

### Derived baseline metrics

- chirp slope: `B/T_chirp`
- ideal range resolution: `c/(2B)`
- unambiguous range and velocity: to be derived only after sampling rate and slow-time configuration are selected

## 4. CA-CFAR detector

### Required inputs

- power range-Doppler map
- training-cell dimensions
- guard-cell dimensions
- target false-alarm probability or threshold multiplier
- edge handling policy

### Required outputs

- adaptive threshold map
- detection mask
- estimated noise map
- detections with range, Doppler, and power

The detector type is paper-derived, while its numerical window parameters must be explicit simulator configuration.

## 5. MHT track-before-detect

### Pipeline

1. Form points in spatiotemporal coordinates.
2. Project points onto `xy`, `xt`, and `yt` planes.
3. Perform a 2D Hough transform independently on each plane.
4. Smooth each accumulator with a 3 x 3 mean filter.
5. Select peaks using a configurable vote threshold.
6. Associate supporting points to each line using a distance threshold.
7. Validate candidates through AND-logic intersection across projections.
8. Process rolling windows for nonlinear/maneuvering tracks.
9. Stitch segments using positional and kinematic costs.

### Dataset outputs

- number of raw points
- clutter count/density
- number of Hough candidates by projection
- common-support count
- track detected flag
- trajectory deviation
- runtime and memory metrics

## 6. Adaptive Driving Beam

### Pipeline

- combine camera lateral localization with PC-FMCW range
- calculate a non-glare angular interval with a safety margin
- apply a smooth raised-cosine transition between dark and fully illuminated regions
- support one or multiple road users

### Dataset outputs

- target range and lateral offset
- shadow interval boundaries
- requested and applied intensity
- glare-violation flag
- retained road illumination ratio

## 7. CRLB evaluation

The paper evaluates lower bounds for delay and Doppler estimation under complex Gaussian noise. The implementation should expose:

- SNR
- bandwidth
- carrier wavelength
- chirp count
- coherent integration time
- delay variance lower bound
- Doppler variance lower bound
- equivalent range and velocity standard-deviation bounds

The exact convention used for waveform energy, SNR, and dimensional constants must be verified during implementation and covered by unit tests.

## 8. Provenance rules

Every model parameter must appear in configuration metadata with:

```yaml
value: 193.4e12
unit: Hz
provenance: paper_explicit
source: DOI:10.1109/LPT.2025.3649597
```

Allowed provenance values:

- `paper_explicit`
- `paper_derived`
- `external_model`
- `scenario_assumption`
- `calibrated`

## 9. Recommended implementation order

1. Parameter and provenance validation
2. PC-FMCW waveform generation
3. Single-target noiseless range estimation
4. Multi-target range-Doppler simulation
5. AWGN and coherent receiver noise
6. CA-CFAR
7. DPSK demodulation and BER
8. CRLB evaluation
9. MHT tracking
10. ADB illumination
11. Atmospheric optical-channel extensions
12. Large-scale dataset generation

No multi-million-row dataset should be released before the component models pass deterministic validation tests.