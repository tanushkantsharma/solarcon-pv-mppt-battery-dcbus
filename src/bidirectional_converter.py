"""Averaged model of the battery-side bidirectional synchronous buck-boost
(half-bridge) converter. Battery on the low-voltage side, i_b > 0 = discharge
(boost mode), i_b < 0 = charge (buck mode). Synchronous rectification keeps
the converter in CCM in both directions, so one averaged equation covers both:

    L_b di_b/dt = v_t - R_Lb i_b - (1 - d_b) v_dc
    i_bus       = (1 - d_b) i_b
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BidirParams:
    L: float = 1.5e-3       # H (design assumption)
    R_L: float = 0.02       # ohm
    fs: float = 20e3
    d_min: float = 0.0
    d_max: float = 0.95


def di_dt(i_b, v_t, v_dc, d_b, p: BidirParams):
    return (v_t - p.R_L * i_b - (1.0 - d_b) * v_dc) / p.L


def bus_current(i_b, d_b):
    return (1.0 - d_b) * i_b


def mode(i_b, tol=0.2):
    return "discharge" if i_b > tol else "charge" if i_b < -tol else "idle"
