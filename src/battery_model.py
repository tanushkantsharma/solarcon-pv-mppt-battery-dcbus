"""First-order Thevenin battery pack model (sign: i > 0 discharge).

    dSOC/dt = -i / (3600 Q)
    dV1/dt  = -V1 / (R1 C1) + i / C1
    v_t     = OCV(SOC) - V1 - R0 i

The OCV curve is a SYNTHETIC, NMC-like per-cell curve scaled to 80 series cells
(about 300 V nominal). It is a design assumption, not measured data.
"""
from dataclasses import dataclass

import numpy as np

SOC_PTS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
CELL_OCV = np.array([3.30, 3.55, 3.62, 3.67, 3.72, 3.78, 3.86, 3.94, 4.02, 4.10, 4.18])


@dataclass(frozen=True)
class BatteryParams:
    Q_Ah: float = 40.0
    n_series: int = 80
    R0: float = 0.12          # ohm (pack)
    R1: float = 0.06          # ohm (pack)
    tau1: float = 30.0        # s
    soc_min: float = 0.10
    soc_max: float = 0.90
    I_ch_max: float = 20.0    # A (0.5C)
    I_dis_max: float = 40.0   # A (1C)
    V_min: float = 2.9 * 80
    V_max: float = 4.2 * 80

    @property
    def C1(self):
        return self.tau1 / self.R1


def ocv(soc, p: BatteryParams):
    return p.n_series * float(np.interp(soc, SOC_PTS, CELL_OCV))


def derivatives(soc, v1, i, p: BatteryParams):
    return -i / (3600.0 * p.Q_Ah), -v1 / p.tau1 + i / p.C1


def terminal_voltage(soc, v1, i, p: BatteryParams):
    return ocv(soc, p) - v1 - p.R0 * i
