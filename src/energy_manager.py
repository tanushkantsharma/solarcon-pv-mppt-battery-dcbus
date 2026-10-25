"""Supervisory energy-management controller (DC-bus-signalling coordination).

Only ONE loop regulates the bus at any time, selected by non-overlapping
voltage thresholds:

* NORMAL   : battery converter regulates v_dc = V_ref (400 V); PV runs MPPT.
* CURTAIL  : battery charge current is at its limit (SOC near soc_max or
             I_ch_max reached) so v_dc rises; above V_curt (410 V) a PI loop
             adds a positive offset to the PV-voltage reference (moves the PV
             operating point right of the MPP) and the MPPT is frozen.
* SHED     : battery discharge current is at its limit (SOC near soc_min or
             I_dis_max reached) so v_dc falls; below V_shed (390 V) a PI loop
             disconnects part of the (constant-power) load.
* PV_OFF   : PV power below threshold for t_pv_off (night/heavy cloud): MPPT
             frozen and reference reset; battery alone serves the load.

Battery current limits are derated linearly to zero inside a SOC band of
width ``soc_band`` at each SOC boundary, so limits are continuous (no mode
chattering) and SOC can never be driven past soc_min/soc_max by the
controller.
"""
from dataclasses import dataclass

from .battery_model import BatteryParams
from .control import PI


@dataclass(frozen=True)
class EMSParams:
    V_ref: float = 400.0
    V_curt: float = 410.0
    V_shed: float = 390.0
    soc_band: float = 0.02
    p_pv_off: float = 25.0        # W
    t_pv_off: float = 0.1         # s, PV power must stay low this long
    curt_kp: float = 1.0          # V/V
    curt_ki: float = 150.0        # V/(V s)
    curt_max: float = 150.0       # V
    shed_kp: float = 300.0        # W/V
    shed_ki: float = 3.0e4        # W/(V s)


class EnergyManager:
    def __init__(self, ts, bat: BatteryParams, p: EMSParams = EMSParams()):
        self.p, self.bat = p, bat
        self.curt = PI(p.curt_kp, p.curt_ki, ts, 0.0, p.curt_max)
        self.shed = PI(p.shed_kp, p.shed_ki, ts, 0.0, 1e9)
        self.mode = "NORMAL"
        self.ts = ts
        self.t_low = 0.0

    def current_limits(self, soc):
        b, w = self.bat, self.p.soc_band
        f_ch = min(max((b.soc_max - soc) / w, 0.0), 1.0)
        f_dis = min(max((soc - b.soc_min) / w, 0.0), 1.0)
        return b.I_ch_max * f_ch, b.I_dis_max * f_dis

    def step(self, v_dc, soc, p_load, p_pv):
        """Return (I_ch_lim, I_dis_lim, dv_curtail, p_served, freeze_mppt)."""
        i_ch, i_dis = self.current_limits(soc)
        dv = self.curt.step(v_dc - self.p.V_curt)
        p_shed = self.shed.step(self.p.V_shed - v_dc, hi=p_load)
        p_served = p_load - p_shed
        self.t_low = self.t_low + self.ts if p_pv < self.p.p_pv_off else 0.0
        pv_off = self.t_low >= self.p.t_pv_off and dv <= 0.0
        if dv > 1e-6:
            self.mode = "CURTAIL"
        elif p_shed > 1e-3:
            self.mode = "SHED"
        elif pv_off:
            self.mode = "PV_OFF"
        else:
            self.mode = "NORMAL"
        return i_ch, i_dis, dv, p_served, (dv > 1e-6) or pv_off
