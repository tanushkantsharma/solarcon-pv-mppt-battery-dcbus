# Model equations

All converter models are **state-space averaged, continuous-conduction-mode (CCM)** models.
Switching ripple is not simulated; it is computed analytically (Section 6). Sign conventions:
battery current `i_b > 0` = discharge.

## 1. PV single-diode model (`src/pv_model.py`)

Module equation (Ns = 54 series cells, KC200GT):

    I = Iph − I0 [exp((V + I Rs) / (a Vt)) − 1] − (V + I Rs) / Rsh,   Vt = Ns k Tc / q

Here `a` is the diode ideality factor and `a·Vt` corresponds to `n·Ns·Vt,cell` in the project brief.

**Parameter extraction.** With `a = 1.3` fixed (Villalva et al., 2009), `Rs` and `Rsh` are solved
(`scipy.optimize.fsolve`) from two conditions at STC:

1. the curve passes through the datasheet point `I(Vmp) = Imp`;
2. that point is the maximum: `dP/dV = I + V dI/dV = 0` at `(Vmp, Imp)`, with
   `dI/dV = −g / (1 + Rs g)`, `g = I0/(a Vt) e^{(V+IRs)/(aVt)} + 1/Rsh`.

with `Iph,ref = Isc (Rs + Rsh)/Rsh` and `I0` set so that `I(Voc) = 0`. Result:
`Rs = 0.2308 Ω`, `Rsh = 597.4 Ω`, `Iph,ref = 8.2132 A`, `I0,ref = 9.76e-8 A`.

**Irradiance / temperature dependence** (`ΔT = Tc − 25 °C`):

    Iph(G, Tc) = (Iph,ref + Ki ΔT) · G / 1000
    I0(Tc)     = (Iph,ref + Ki ΔT − Voc(Tc)/Rsh) / (exp(Voc(Tc)/(a Vt(Tc))) − 1),   Voc(Tc) = Voc,ref + Kv ΔT

The `I0(T)` form is the Villalva approach generalised to include `Rsh`, so that the model's
open-circuit voltage at 1000 W/m² follows the datasheet `Kv` exactly. `Rs`, `Rsh` are held constant.

**Explicit solver.** The implicit equation is solved in closed form with the Lambert W function
(Jain & Kapoor, 2004):

    I = [Rsh (Iph + I0) − V] / (Rs + Rsh) − (a Vt / Rs) · W(θ)
    ln θ = ln[Rs Rsh I0 / (a Vt (Rs + Rsh))] + Rsh [Rs (Iph + I0) + V] / [a Vt (Rs + Rsh)]

`W(θ)` is evaluated as `wrightomega(ln θ)`, which never forms `exp(ln θ)` and therefore cannot
overflow (`ln θ` reaches several hundred near and beyond Voc). The analogous closed form for
`V(I)` is used to build shaded string curves. A unit test checks the residual of the implicit
equation is below 1e-8 A for −5 V ≤ V ≤ 34 V.

**Array and partial shading.** 8 modules in series × 3 strings. Uniform irradiance:
`I_array(V) = 3 · I_mod(V/8)`. With shading, each module has an ideal bypass diode
(drop 0.7 V); for a string current `I`, the string voltage is
`Σ_k max(V_mod(I, G·s_k), −0.7 V)`; the resulting `V(I)` table (3000 points) is inverted by
interpolation. All three strings are assumed to see the same pattern.

## 2. PV-side boost converter (`src/boost_converter.py`)

States: PV-side capacitor voltage `v_pv`, inductor current `i_L`.

    Cin dv_pv/dt = i_pv(v_pv) − i_L
    L   di_L/dt  = v_pv − (R_L + d R_on) i_L − (1 − d)(v_dc + V_F)
    i_out        = (1 − d) i_L − P_sw / v_dc,     P_sw = ½ v_dc i_L t_sw f_s

`d ∈ [0, 0.9]`. `i_L ≥ 0` is enforced (diode); DCM is not modelled. **Ideal baseline:**
`R_L = R_on = V_F = t_sw = 0`. **Lossy variant** (assumed values, not a specific BOM):
`R_L = 50 mΩ`, `R_on = 60 mΩ`, `V_F = 1.2 V`, `t_sw = 100 ns`. Steady state (ideal):
`d = 1 − v_pv/v_dc` (≈ 0.474 at the STC MPP).

## 3. Battery: first-order Thevenin model (`src/battery_model.py`)

    dSOC/dt = −i_b / (3600 Q)
    dV1/dt  = −V1 / (R1 C1) + i_b / C1,     C1 = τ1 / R1
    v_t     = OCV(SOC) − V1 − R0 i_b

`OCV(SOC)` = 80 × synthetic NMC-like cell curve (linear interpolation, 3.30–4.18 V/cell).

## 4. Battery-side bidirectional converter (`src/bidirectional_converter.py`)

Synchronous half-bridge, battery on the low-voltage side; one averaged equation for both
directions (synchronous rectification keeps CCM):

    L_b di_b/dt = v_t − R_Lb i_b − (1 − d_b) v_dc,     i_bus = (1 − d_b) i_b,   d_b ∈ [0, 0.95]

Boost mode (discharge) for `i_b > 0`, buck mode (charge) for `i_b < 0`.

## 5. DC bus and integration (`src/system.py`)

    C_dc dv_dc/dt = i_out + (1 − d_b) i_b − P_served / v_dc

The load is constant-power (`P_served` = demand minus EMS shedding). State vector
`[v_pv, i_L, v_dc, i_b, V1, SOC]`, classical RK4 with step `Ts = 50 µs` and zero-order-hold duty
cycles (control and integration share the 50 µs grid).

## 6. Control

PI with back-calculation anti-windup (Åström & Hägglund): `u_raw = Kp e + x + ff`,
`u = sat(u_raw)`, `x ← x + Ts (Ki e + Kaw (u − u_raw))`, `Kaw = Ki/Kp`.
Gains are set from loop crossover `ωc`: `Kp = (plant L or C)·ωc`, `Ki = Kp·ωc/10`.

| Loop | Output | Crossover | Feed-forward | Saturation |
|---|---|---|---|---|
| PV voltage | `i_L,ref = PI(v_pv − v_ref) + i_pv` | 200 Hz | measured `i_pv` | `0 … 1.3·3·Isc` |
| PV current | `u` → `d = 1 − (v_pv − u)/v_dc` | 1.5 kHz | `v_pv`, `v_dc` | `d ∈ [0, 0.9]` (bounds mapped onto `u`) |
| Bus voltage | `i_b,ref` | 40 Hz | `(P_load/v_dc − (1−d) i_L)·v_dc/v_t` | `[−I_ch,lim, I_dis,lim]` from EMS |
| Battery current | `u_b` → `d_b = 1 − (v_t − u_b)/v_dc` | 1 kHz | `v_t`, `v_dc` | `d_b ∈ [0, 0.95]` |
| Curtailment (EMS) | `Δv_ref = PI(v_dc − 410)` | — | — | `0 … 150 V` |
| Load shedding (EMS) | `P_shed = PI(390 − v_dc)` | — | — | `0 … P_load` |

**Coordination.** Only one loop regulates the bus at a time, selected by non-overlapping
voltage set-points (DC-bus signalling): battery at 400 V; if the battery charge current
saturates, the bus rises to 410 V where PV curtailment takes over (MPPT frozen); if the
discharge current saturates, the bus falls to 390 V where load shedding takes over. SOC limits are
enforced through current limits derated linearly to zero over 2 % SOC at each boundary
(`soc_min = 0.10`, `soc_max = 0.90`). PV_OFF is declared after PV power stays below 25 W for 0.1 s.

**Analytical ripple.** `Δi_L,pp = v_pv d / (L f_s)`; boost contribution to bus ripple
`Δv_dc,pp ≈ (1 − d) i_L d / (C_dc f_s)`.

## 7. MPPT (`src/mppt.py`)

All trackers output a PV-voltage reference once per MPPT period `T_p` and receive the same
measurement: PV voltage and current averaged over the **last quarter** of the period (after the
voltage loop has settled).

* **P&O** (Femia et al., 2005): reverse direction if `P_k < P_{k−1}`, fixed step `ΔV`.
* **IncCond** (Hussein et al., 1995): compare `dI/dV` with `−I/V`; tolerance band
  `|dI/dV + I/V| ≤ ε·I/V`, fixed step.
* **dP-P&O** (Sera et al., 2006): a mid-period measurement `P_x` (second quarter of the period)
  separates the perturbation effect from a linear irradiance trend:
  `dP = (P_x − P_k) − (P_{k+1} − P_x)`.
* **Adaptive dP-P&O (proposed):** dP-P&O decision with a variable step
  `ΔV_k = clip(M·|dP| / ΔV_{k−1}, ΔV_min, ΔV_max)` (variable-step idea as in Liu et al., 2008).
  With `M = 0` it is fixed-step dP-P&O (the ablation).
* **+ scan (partial-shading extension):** if `|P_k − P_{k−1}| > 0.15·max(P_{k−1}, 0.1 P_rated)`,
  sweep 12 references from 0.90 to 0.35 of `v_max` (one period each), restart from the best.

## 8. Metrics (`src/metrics.py`)

Logged every 0.5 ms (10 × Ts); integrals by the trapezoidal rule over the stated window.

| Metric | Definition |
|---|---|
| η_MPPT | `∫P_pv dt / ∫P_mpp dt`; `P_mpp(G,T,shade)` = global maximum of the static P–V curve (dense scan + bounded refinement) |
| PV energy | `∫P_pv dt` (Wh) |
| Reach time `t_r` | first time after an event that `|P_pv − P_mpp| ≤ 1 % P_mpp` |
| Settling time `t_s` | time after which `|P_pv − P_mpp| ≤ 1 % P_mpp` **and** `|v_pv − V_mpp| ≤ 2 % V_mpp` hold until the window end (≈ window length ⇒ never settled) |
| Oscillation | peak-to-peak `v_pv` and `(max−min)/mean` of `P_pv` in a steady window |
| Bus RMS error / overshoot | `sqrt(mean((v_dc − 400)^2))`; max positive / negative deviation |
| Converter efficiency | `∫v_dc i_out dt / ∫v_pv i_L dt` (=1 for the ideal model) |
| Battery | RMS `i_b`; time with `i_b` beyond its active limit by > 0.5 A; time with SOC outside `[0.1, 0.9]` or `v_t` outside `[232, 336] V` |
| Curtailment | `∫(P_mpp − P_pv) dt` while mode = CURTAIL |
| Unserved load | `∫(P_load − P_served) dt` |
| Runtime | wall-clock time of the simulation loop; real-time factor = runtime / simulated time |
