"""Reproducible, synthetic irradiance / temperature / load / shading profiles.

All profiles are piecewise-linear breakpoint tables (time in s). Steps are
represented by two breakpoints 1 microsecond apart. Nothing here is measured
weather data.
"""
from dataclasses import dataclass, field

import numpy as np


def pwl(*pts):
    """Breakpoints [(t, value), ...] -> (t_array, value_array)."""
    t, v = zip(*pts)
    return np.asarray(t, float), np.asarray(v, float)


def steps(t_end, *segs):
    """steps(t_end, (t0, v0), (t1, v1), ...) -> piecewise-constant pwl."""
    pts = []
    for k, (t, v) in enumerate(segs):
        if k:
            pts.append((t - 1e-6, segs[k - 1][1]))
        pts.append((t, v))
    pts.append((t_end, segs[-1][1]))
    return pwl(*pts)


@dataclass
class Scenario:
    name: str
    t_end: float
    G: tuple                         # pwl irradiance, W/m^2
    T: tuple                         # pwl cell temperature, degC
    load: tuple                      # pwl constant-power load, W
    shade: list = field(default_factory=list)   # [(t_start, pattern or None), ...]
    soc0: float = 0.5
    stiff_bus: bool = False          # True: ideal 400 V bus, PV side only
    eval_start: float = 0.2          # metrics integration start, s
    segments: list = field(default_factory=list)  # step times for settling analysis

    def at(self, t):
        g = float(np.interp(t, *self.G))
        tc = float(np.interp(t, *self.T))
        pl = float(np.interp(t, *self.load))
        sh = None
        for ts, pat in self.shade:
            if t >= ts:
                sh = pat
        return g, tc, pl, sh


def const(v, t_end):
    return pwl((0.0, v), (t_end, v))


def training_profile(seed=1, t_end=5.0):
    """Seeded random sequence of holds, steps and ramps (200-1000 W/m^2,
    ramp rates 100-800 W/m^2/s, 25->50 degC) used ONLY for offline MPPT
    parameter tuning; never used for evaluation."""
    rng = np.random.default_rng(seed)
    t, g = 0.0, 1000.0
    pts = [(0.0, g)]
    while t < t_end:
        g_new = float(np.clip(g + rng.uniform(-500, 500), 200, 1000))
        if rng.random() < 0.4:                       # step
            pts.append((t + 1e-6, g_new))
        else:                                         # ramp at a random rate
            t += abs(g_new - g) / float(rng.uniform(100, 800)) + 1e-6
            pts.append((t, g_new))
        t += float(rng.uniform(0.1, 0.4))             # hold
        pts.append((t, g_new))
        g = g_new
    t_arr, g_arr = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    G = (t_arr, g_arr)
    T = pwl((0.0, 25.0), (t_end, 50.0))
    return Scenario("training", t_end, G, T, const(0.0, t_end), stiff_bus=True)
