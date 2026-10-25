# SolarCon — PV boost converter with MPPT and battery storage on a 400 V DC bus

A reproducible, purely computational study of a ≈5 kW photovoltaic array feeding a 400 V DC bus
through a boost converter, with a 300 V / 40 Ah battery on a bidirectional buck–boost converter
and a supervisory energy-management controller. Conventional MPPT (Perturb & Observe,
Incremental Conductance) is compared with an offline-tuned **adaptive, drift-compensated P&O**
under identical conditions, including irradiance steps and ramps, temperature sweeps, partial
shading, SOC limits, load steps, and component/measurement uncertainty.

All models are **state-space averaged** (no switching ripple in the time series); all profiles
are **synthetic**. Every number below is produced by `scripts/run_experiments.py` and written
to `results/metrics/`.

## System

```
 PV array 8s×3p KC200GT ─ Cin ─ boost (L=2 mH, 20 kHz) ─┐
   4.80 kW STC                  MPPT → v_ref → PI v → PI i → d
                                                         ├─ DC bus 400 V, C = 2 mF ─ constant-power load
 Battery 80s, ≈300 V, 40 Ah ─ sync. buck–boost (1.5 mH) ─┘
                                PI v_dc → i_b,ref (EMS limits) → PI i → d_b
 Supervisor: battery regulates 400 V │ curtail PV above 410 V │ shed load below 390 V │ SOC-derated current limits
```

* **PV model:** single-diode, parameters extracted from the Kyocera KC200GT datasheet
  (Rs = 0.231 Ω, Rsh = 597 Ω, a = 1.3), explicit Lambert-W solution via the Wright-omega
  function (overflow-free), bypass-diode partial-shading model.
* **Control:** cascaded PI loops with feed-forward, duty saturation and back-calculation
  anti-windup; one bus-regulating loop at a time via non-overlapping voltage bands.
* **MPPT:** P&O, IncCond, dP-P&O (ablation), Adaptive dP-P&O (proposed), plus an optional
  global scan for partial shading. All four are tuned by the **same** grid search
  (`optimization/tune_mppt.py`) on training episodes that are not used for evaluation.

Equations: [`docs/model_equations.md`](docs/model_equations.md) · assumptions:
[`docs/design_assumptions.md`](docs/design_assumptions.md) · references:
[`docs/references.md`](docs/references.md) · report: [`report/technical_report.md`](report/technical_report.md)

## Results (ideal converters unless stated)

**PV model validation** against the datasheet (single module): STC Pmax 200.1 W (+0.07 %),
Isc and Voc exact by construction; the **independent NOCT point** (800 W/m², 47 °C) gives
142.2 W vs 142 W (+0.14 %), Voc −0.5 %, Vmp +1.2 %, Imp −1.2 %.

**MPPT comparison** — η_MPPT = ∫P_pv / ∫P_mpp (`results/metrics/J_comparison.csv`):

| Test | P&O | IncCond | dP-P&O (ablation) | **Adaptive dP-P&O** |
|---|---|---|---|---|
| A: STC, 0.3–1.5 s | 99.84 % | 99.84 % | 99.84 % | **99.99 %** |
| A: steady v_pv oscillation (p-p) | 8.7 V | 8.7 V | 8.7 V | **2.2 V** |
| B: steps 1000→600→300→800 W/m² | 99.83 % | 99.84 % | 99.83 % | **99.96 %** |
| D: ramps −350 / +1400 W/m²/s | 99.78 % | 99.79 % | 99.83 % | **99.99 %** |
| D: during +1400 W/m²/s ramp | 99.06 % | 99.10 % | 99.84 % | **99.99 %** |
| E: partial shading (global MPP) | 71.9 % | 71.9 % | 71.9 % | 72.1 % → **99.1 % with scan** |
| I: 8 perturbed plants + 0.5 % FS noise (mean) | **99.76 %** | **99.76 %** | 99.63 % | 99.70 % |

What the comparison shows:

* Without noise, the proposed tracker gains ~0.15 percentage points of tracking efficiency and
  cuts the steady-state voltage oscillation by 4× (its small step near the MPP); the drift
  compensation removes P&O's characteristic loss during fast ramps (ramp-up η 99.06 % → 99.99 %).
* All local trackers lock onto the high-voltage local maximum under partial shading (η ≈ 55 %
  in pattern 1); only the triggered global scan recovers the global MPP.
* **With 0.5 % full-scale measurement noise the advantage reverses**: fixed-step P&O/IncCond
  are the most robust (99.76 % vs 99.70 %). The dP estimate differences three noisy power
  samples, which amplifies noise; this is the main limitation of the proposed method.
* Most gains are in hundredths of a percent of energy: in an averaged model with exact
  measurements every tracker is already > 99.7 % efficient on uniform irradiance.

**DC bus and energy management** (Adaptive dP-P&O; `results/metrics/FGH_system.csv`):

| Test | Result |
|---|---|
| H: load steps 1→6→0.5→3 kW, PI + bus-current feed-forward | max deviation 0.78 V |
| H: same, PI only (ablation) | max deviation 20.9 V, settles to ±1 V in ≤ 146 ms |
| F: PV 1000→300→800→0 W/m², 3 kW load | battery moves charge→discharge→charge→discharge; bus within −0.81/+0.43 V |
| G1: SOC 88.8 %, load 1.5 kW → 0 | charge current saturates at its derated limit; bus peaks at 421 V then curtailment holds 410 V; 0 current/SOC violations |
| G2: SOC 10.5 %, load 3 → 6 kW | discharge current saturates (10 A); bus falls to 385.5 V then shedding holds 390 V; 0.51 Wh unserved over 1 s |
| A (lossy boost variant) | converter efficiency 98.66 %; analytical inductor ripple 2.52 A p-p (11 % of 22.8 A) |

Figures are in `results/figures/` (I–V/P–V curves, step responses, ramps, shading, system tests).

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                       # 22 tests, ~5 s
python scripts/run_experiments.py         # all experiments A–J, ~3 min on 2 cores
python optimization/tune_mppt.py          # optional: re-run MPPT tuning (~9 min); writes tuned_params.json
```

`run_experiments.py` uses the committed `optimization/tuned_params.json`, deterministic profiles
and fixed seeds, and rewrites `results/metrics/*.csv`, `results/metrics/tables.md` (all tables)
and `results/figures/*.png`. Tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0,
Matplotlib 3.11.

## Repository layout

```
src/            pv_model, boost_converter, battery_model, bidirectional_converter,
                mppt, energy_manager, control (PI), system (coupled simulation),
                profiles, metrics
scripts/        run_experiments.py
optimization/   tune_mppt.py, tuned_params.json, tuning_results.csv
tests/          pytest suite (models, trackers, closed-loop scenarios)
results/        figures/, metrics/
docs/, report/
```

## Limitations

* Averaged CCM models: no switching ripple, DCM, dead time, ADC/PWM delay or EMI; ripple and
  efficiency figures come from analytical and simplified conduction/switching-loss expressions.
* Synthetic battery OCV curve and parameters; synthetic irradiance profiles with ramps faster
  than standard test sequences; prescribed cell temperature (no thermal model).
* Seconds-long simulations: SOC-limit behaviour is shown by initialising SOC near the limits,
  not by long-term cycling.
* Partial shading assumes identical patterns in all strings and ideal bypass diodes; the scan
  trigger and range were set by hand, not tuned.
* MPPT tuning is a coarse grid; the best P&O/IncCond step (4 V) and the best adaptive
  `dv_max` (8 V) lie on the grid edge.
* The bus-voltage loop relies on a load-current feed-forward measurement; without it, deviations
  are ~20 V for 5 kW steps.
* No MATLAB/Simulink or OpenModelica model is included (neither was available in the build
  environment); the Python model is the single source of truth. No hardware validation.

## License

MIT — see [LICENSE](LICENSE).
