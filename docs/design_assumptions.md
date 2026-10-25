# Design assumptions

Labels: **[spec]** = from the project brief, **[datasheet]** = manufacturer data,
**[derived]** = computed here, **[assumed]** = engineering assumption chosen for this study.

## PV array
| Item | Value | Source |
|---|---|---|
| Module | Kyocera KC200GT, 54 cells, 200 W | [datasheet] |
| STC | Pmax 200 W, Vmp 26.3 V, Imp 7.61 A, Voc 32.9 V, Isc 8.21 A | [datasheet] |
| Coefficients | Kv = −0.123 V/K, Ki = 3.18 mA/K | [datasheet] |
| Ideality factor a | 1.3 | [assumed], following Villalva et al. (2009) |
| Rs, Rsh | 0.2308 Ω, 597.4 Ω | [derived] |
| Array | 8 series × 3 parallel = 24 modules, **4.80 kW** at STC | [derived] from the "≈5 kW" [spec] |
| Array MPP (STC) | 210.4 V, 22.83 A | [derived] |
| Array Voc | 263 V (25 °C), 273 V (15 °C), 224 V (65 °C) | [derived] |
| Bypass diodes | one per module, 0.7 V, ideal | [assumed] |

Why 8s×3p: the boost converter needs `v_pv < v_dc = 400 V` at all operating points, including
open circuit at the coldest test temperature (15 °C → 273 V); 12s×2p (4.8 kW) would give
≈ 409 V at 15 °C, above the bus. 8s×3p gives a nominal duty ≈ 0.47, well inside `[0, 0.9]`.

## Converters and bus
| Item | Value | Source |
|---|---|---|
| DC bus | 400 V | [spec] |
| Switching frequency | 20 kHz | [spec] |
| Boost inductance | 2 mH | [spec] |
| DC-link capacitance | 2 mF | [spec] |
| PV-side input capacitor | 220 µF | [assumed] |
| Boost duty limits | 0 … 0.90 | [assumed] |
| Battery-converter inductance | 1.5 mH, 20 mΩ | [assumed] |
| Battery-converter duty limits | 0 … 0.95 | [assumed] |
| Lossy boost variant | R_L 50 mΩ, R_on 60 mΩ, V_F 1.2 V, t_sw 100 ns | [assumed] |
| Controller sampling | 50 µs (also the RK4 step) | [spec] |

**Check of the 2 mH choice:** analytical inductor ripple at the STC MPP is 2.52 A p-p on a
22.8 A mean (11 %), comfortably CCM. Boost contribution to bus ripple ≈ 0.14 V p-p on 400 V.

## Battery
| Item | Value | Source |
|---|---|---|
| Capacity | 40 Ah | [spec] |
| Pack | 80 series cells, OCV 264–334 V, 302 V at 50 % SOC (≈300 V nominal) | [assumed] synthetic NMC-like OCV curve |
| R0, R1, τ1 | 0.12 Ω, 0.06 Ω, 30 s | [assumed] |
| SOC window | 10 % … 90 %, current derated to zero over 2 % SOC at each end | [assumed] |
| Current limits | 20 A charge (0.5C), 40 A discharge (1C) | [assumed] |
| Voltage window | 232 … 336 V (2.9–4.2 V/cell) | [assumed] |

## Energy management
Bus set-points 400 V (battery), 410 V (PV curtailment), 390 V (load shedding). PV_OFF after
PV power < 25 W for 0.1 s. Load is a constant-power load. [assumed]

## Profiles
All irradiance, temperature, load and shading profiles are **synthetic and deterministic**
(`src/profiles.py`, `scripts/run_experiments.py`). Ramps in experiment D (−350 and
+1400 W/m²/s) are faster than typical standard test ramps (EN 50530 uses up to 100 W/m²/s) so that
they fit into short averaged-model simulations; they stress the trackers more than field data.
Cell temperature is a prescribed input (no thermal model).

## Time-scale caveat
Simulations last 1–5 s, so SOC changes by < 0.01 %. SOC-limit behaviour (experiment G) is
exercised by **initialising** SOC inside the derating band (88.8 % and 10.5 %); a full-day
SOC trajectory would need a quasi-static model (not implemented).
