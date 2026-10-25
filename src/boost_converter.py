"""Averaged (state-space averaged, CCM) model of the PV-side boost converter.

States: v_pv (input capacitor), i_L (boost inductor). The DC-bus capacitor is
shared with the battery converter and is integrated in src/system.py.

    Cin dv_pv/dt = i_pv - i_L
    L   di_L/dt  = v_pv - (R_L + d R_on) i_L - (1 - d)(v_dc + V_F)
    i_out        = (1 - d) i_L - P_sw / v_dc

Ideal baseline: R_L = R_on = V_F = 0 and no switching loss.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BoostParams:
    L: float = 2e-3          # H
    Cin: float = 220e-6      # F (PV-side capacitor, design assumption)
    fs: float = 20e3         # Hz
    d_min: float = 0.0
    d_max: float = 0.90
    R_L: float = 0.0         # ohm, inductor winding resistance
    R_on: float = 0.0        # ohm, MOSFET on-resistance
    V_F: float = 0.0         # V, boost diode forward drop
    t_sw: float = 0.0        # s, (t_rise + t_fall) for hard-switching loss

    def lossy(self):
        """Simplified loss set (assumed component values, not a specific BOM)."""
        return replace(self, R_L=0.05, R_on=0.06, V_F=1.2, t_sw=100e-9)


def derivatives(v_pv, i_L, v_dc, d, i_pv, p: BoostParams):
    dv_pv = (i_pv - i_L) / p.Cin
    di_L = (v_pv - (p.R_L + d * p.R_on) * i_L - (1.0 - d) * (v_dc + p.V_F)) / p.L
    if i_L <= 0.0 and di_L < 0.0:      # diode blocks reverse current (no DCM model)
        di_L = 0.0
    return dv_pv, di_L


def switching_loss(i_L, v_dc, p: BoostParams):
    return 0.5 * v_dc * max(i_L, 0.0) * p.t_sw * p.fs


def output_current(i_L, v_dc, d, p: BoostParams):
    return (1.0 - d) * i_L - switching_loss(i_L, v_dc, p) / max(v_dc, 1.0)


def inductor_ripple_pp(v_pv, d, p: BoostParams):
    """Analytical peak-to-peak inductor current ripple in CCM, A."""
    return v_pv * d / (p.L * p.fs)


def ideal_duty(v_pv, v_dc):
    return 1.0 - v_pv / v_dc
