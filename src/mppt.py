"""MPPT algorithms. Each tracker is called once per MPPT period with the
PV voltage and current averaged over that period and returns the next
PV-voltage reference (V). The inner PV-voltage loop (src/system.py) tracks it.
"""
from __future__ import annotations


class Tracker:
    name = "base"

    def __init__(self, v0, v_min, v_max, period=0.01):
        self.v_ref = v0
        self.v_min, self.v_max = v_min, v_max
        self.period = period
        self.v_old = None
        self.p_old = None
        self.i_old = None

    def _clip(self, v):
        return min(max(v, self.v_min), self.v_max)

    def reset(self, v0):
        self.v_ref = v0
        self.v_old = self.p_old = self.i_old = None

    def update(self, v, i, vx=None, ix=None):
        raise NotImplementedError


class PerturbObserve(Tracker):
    """Classic fixed-step P&O (voltage-reference form)."""
    name = "P&O"

    def __init__(self, v0, v_min, v_max, period=0.01, step=2.0):
        super().__init__(v0, v_min, v_max, period)
        self.step = step
        self.direction = 1.0

    def update(self, v, i, vx=None, ix=None):
        p = v * i
        if self.p_old is not None and p < self.p_old:
            self.direction = -self.direction
        self.p_old = p
        self.v_ref = self._clip(self.v_ref + self.direction * self.step)
        return self.v_ref


class IncrementalConductance(Tracker):
    """Incremental conductance with a relative tolerance band eps on
    |dI/dV + I/V| / (I/V); eps = 0 reproduces the textbook rule."""
    name = "IncCond"

    def __init__(self, v0, v_min, v_max, period=0.01, step=2.0, eps=0.0):
        super().__init__(v0, v_min, v_max, period)
        self.step, self.eps = step, eps

    def update(self, v, i, vx=None, ix=None):
        if self.v_old is None:
            self.v_old, self.i_old = v, i
            self.v_ref = self._clip(self.v_ref + self.step)
            return self.v_ref
        dv, di = v - self.v_old, i - self.i_old
        g = i / v if v > 1e-6 else 0.0
        if abs(dv) < 1e-3:
            if abs(di) > 1e-3:
                self.v_ref += self.step if di > 0 else -self.step
        else:
            err = di / dv + g
            if abs(err) > self.eps * abs(g):
                self.v_ref += self.step if err > 0 else -self.step
        self.v_old, self.i_old = v, i
        self.v_ref = self._clip(self.v_ref)
        return self.v_ref


class AdaptivePO(Tracker):
    """Proposed method: adaptive-step, drift-compensated P&O.

    1. Drift compensation (dP-P&O, Sera et al. 2006): with a mid-period
       measurement P_x, the perturbation-induced power change is estimated as
       dP = (P_x - P_k) - (P_k+1 - P_x), removing a linear irradiance trend.
    2. Adaptive step: dV = clip(M |dP| / dV_prev, dv_min, dv_max) - large
       steps far from the MPP, small steps near it.
    Parameters (M, dv_min, dv_max, period) are tuned offline
    (optimization/tune_mppt.py) on a training profile not used for evaluation.
    With M = 0 the method reduces to fixed-step dP-P&O (ablation "dpo").

    Optional global scan (``scan=True``): when the power changes between two
    MPPT periods by more than ``scan_trigger`` x the previous power (floor:
    10 % of rated), the reference
    sweeps ``n_scan`` points between ``scan_lo`` and ``scan_hi`` (fractions of
    v_max) and the local search restarts from the best point (partial
    shading), at the cost of a transient loss during the sweep.
    """
    name = "Adaptive dP-P&O"

    def __init__(self, v0, v_min, v_max, period=0.01, M=0.05, dv_min=0.5,
                 dv_max=8.0, scan=False, p_rated=4800.0, scan_trigger=0.15,
                 n_scan=12, scan_lo=0.35, scan_hi=0.90):
        super().__init__(v0, v_min, v_max, period)
        self.M, self.dv_min, self.dv_max = M, dv_min, dv_max
        self.direction = 1.0
        self.last_step = dv_min
        self.scan, self.p_rated, self.scan_trigger = scan, p_rated, scan_trigger
        self.n_scan, self.scan_lo, self.scan_hi = n_scan, scan_lo, scan_hi
        self._queue, self._results = [], []
        self._scanning = False
        self.n_scans = 0
        if scan:
            self.name = "Adaptive dP-P&O + scan"
        elif M == 0:
            self.name = "dP-P&O (fixed step)"

    def reset(self, v0):
        super().reset(v0)
        self._scanning, self._queue = False, []

    def _start_scan(self):
        hi, lo = self.scan_hi * self.v_max, self.scan_lo * self.v_max
        n = self.n_scan
        self._queue = [hi - k * (hi - lo) / (n - 1) for k in range(n)]
        self._results = []
        self._scanning = True
        self.n_scans += 1
        self.v_ref = self._clip(self._queue.pop(0))
        return self.v_ref

    def update(self, v, i, vx=None, ix=None):
        p = v * i
        if self._scanning:
            self._results.append((p, v))      # measurement at previous scan point
            if self._queue:
                self.v_ref = self._clip(self._queue.pop(0))
                return self.v_ref
            self._scanning = False
            self.v_ref = self._clip(max(self._results)[1])
            self.p_old = None
            return self.v_ref
        if (self.scan and self.p_old is not None and abs(p - self.p_old)
                > self.scan_trigger * max(self.p_old, 0.1 * self.p_rated)):
            return self._start_scan()
        if self.p_old is None:
            step = self.dv_min
        else:
            if vx is not None:
                px = vx * ix
                dp = (px - self.p_old) - (p - px)
            else:
                dp = p - self.p_old
            if dp < 0:
                self.direction = -self.direction
            slope = abs(dp) / max(self.last_step, 1e-3)
            step = min(max(self.M * slope, self.dv_min), self.dv_max)
        self.p_old = p
        self.last_step = step
        self.v_ref = self._clip(self.v_ref + self.direction * step)
        return self.v_ref


def make_tracker(kind: str, v0, v_min, v_max, p_rated=4800.0, **kw):
    kinds = {
        "po": lambda: PerturbObserve(v0, v_min, v_max, **kw),
        "inc": lambda: IncrementalConductance(v0, v_min, v_max, **kw),
        "dpo": lambda: AdaptivePO(v0, v_min, v_max, p_rated=p_rated, M=0.0,
                                  dv_min=kw["step"], dv_max=kw["step"], period=kw["period"]),
        "apo": lambda: AdaptivePO(v0, v_min, v_max, p_rated=p_rated, **kw),
        "apo_scan": lambda: AdaptivePO(v0, v_min, v_max, p_rated=p_rated, scan=True, **kw),
    }
    return kinds[kind]()

LABELS = {"po": "P&O", "inc": "IncCond", "dpo": "dP-P&O (ablation)",
          "apo": "Adaptive dP-P&O", "apo_scan": "Adaptive dP-P&O + scan"}
