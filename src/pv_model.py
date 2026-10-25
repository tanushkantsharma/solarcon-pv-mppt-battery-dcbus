"""Single-diode photovoltaic module / array model.

Module data: Kyocera KC200GT datasheet (STC: 1000 W/m^2, 25 degC, AM1.5).
Unknown circuit parameters (Rs, Rsh) are extracted from the datasheet
(Vmp, Imp) point with the ideality factor fixed (a = 1.3), following the
approach of Villalva et al. (2009). See docs/model_equations.md.

The implicit diode equation is solved explicitly with the Lambert-W function,
evaluated through the Wright omega function to avoid exp() overflow:
W(exp(x)) = wrightomega(x).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.special import wrightomega

K_B = 1.380649e-23   # J/K
Q_E = 1.602176634e-19  # C
G_REF = 1000.0       # W/m^2
T_REF = 25.0         # degC


@dataclass(frozen=True)
class ModuleDatasheet:
    name: str
    Pmax: float
    Vmp: float
    Imp: float
    Voc: float
    Isc: float
    Kv: float   # V/K
    Ki: float   # A/K
    Ns: int     # series cells


# Kyocera KC200GT manufacturer datasheet values (STC).
KC200GT = ModuleDatasheet("Kyocera KC200GT", 200.0, 26.3, 7.61, 32.9, 8.21,
                          -0.123, 3.18e-3, 54)


def module_vt(Ns: int, t_c: float) -> float:
    """Thermal voltage of Ns series cells, V."""
    return Ns * K_B * (t_c + 273.15) / Q_E


@dataclass(frozen=True)
class ModuleParams:
    ds: ModuleDatasheet
    a: float
    Rs: float
    Rsh: float
    Iph_ref: float

    def at(self, G: float, t_c: float):
        """Return (Iph, I0, a*Vt) at irradiance G (W/m^2) and cell temperature t_c."""
        dT = t_c - T_REF
        avt = self.a * module_vt(self.ds.Ns, t_c)
        iph_stc_t = self.Iph_ref + self.ds.Ki * dT
        voc_t = self.ds.Voc + self.ds.Kv * dT
        # I0 chosen so that I(Voc(T)) = 0 at G_REF: Voc follows the datasheet Kv.
        i0 = (iph_stc_t - voc_t / self.Rsh) / np.expm1(voc_t / avt)
        iph = iph_stc_t * G / G_REF
        return iph, i0, avt


def _residuals(x, ds: ModuleDatasheet, a: float):
    rs, log_rsh = x
    rsh = np.exp(log_rsh)
    avt = a * module_vt(ds.Ns, T_REF)
    iph = ds.Isc * (rs + rsh) / rsh
    i0 = (iph - ds.Voc / rsh) / np.expm1(ds.Voc / avt)
    e = np.exp((ds.Vmp + ds.Imp * rs) / avt)
    f1 = iph - i0 * (e - 1) - (ds.Vmp + ds.Imp * rs) / rsh - ds.Imp
    g = i0 / avt * e + 1 / rsh
    didv = -g / (1 + rs * g)
    f2 = ds.Imp + ds.Vmp * didv          # dP/dV = 0 at MPP
    return [f1, f2]


@lru_cache(maxsize=8)
def extract_parameters(ds: ModuleDatasheet = KC200GT, a: float = 1.3) -> ModuleParams:
    """Solve for Rs, Rsh so the model passes through (Vmp, Imp) with dP/dV = 0."""
    sol, info, ier, msg = fsolve(_residuals, [0.2, np.log(300.0)], args=(ds, a),
                                 full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"PV parameter extraction failed: {msg}")
    rs, rsh = float(sol[0]), float(np.exp(sol[1]))
    if rs <= 0 or rsh <= 0:
        raise RuntimeError("Non-physical PV parameters extracted")
    return ModuleParams(ds, a, rs, rsh, ds.Isc * (rs + rsh) / rsh)


def module_current(v, G, t_c, p: ModuleParams):
    """Module current (A) at terminal voltage v (V); explicit Lambert-W solution."""
    iph, i0, avt = p.at(G, t_c)
    rs, rsh = p.Rs, p.Rsh
    v = np.asarray(v, dtype=float)
    lnth = (np.log(rs * rsh * i0 / (avt * (rs + rsh)))
            + rsh * (rs * (iph + i0) + v) / (avt * (rs + rsh)))
    w = np.real(wrightomega(lnth))
    return (rsh * (iph + i0) - v) / (rs + rsh) - avt / rs * w


def module_voltage(i, G, t_c, p: ModuleParams):
    """Module voltage (V) at current i (A); explicit Lambert-W solution."""
    iph, i0, avt = p.at(G, t_c)
    rs, rsh = p.Rs, p.Rsh
    i = np.asarray(i, dtype=float)
    lnph = np.log(i0 * rsh / avt) + rsh * (iph + i0 - i) / avt
    w = np.real(wrightomega(lnph))
    return (iph + i0 - i) * rsh - i * rs - avt * w


@dataclass
class PVArray:
    """Ns_mod series modules x Np_str parallel strings, one bypass diode per module.

    Uniform irradiance uses the explicit Lambert-W current. A shading pattern
    (per-series-module irradiance multipliers, identical in every string)
    uses a tabulated string I-V curve with ideal bypass diodes (drop vd_bypass).
    """
    n_series: int = 8
    n_parallel: int = 3
    params: ModuleParams = field(default_factory=extract_parameters)
    vd_bypass: float = 0.7
    _cache: dict = field(default_factory=dict, repr=False)

    def current(self, v, G, t_c, shade=None):
        if shade is None:
            return self.n_parallel * module_current(np.asarray(v) / self.n_series, G, t_c, self.params)
        vt, it = self._table(G, t_c, tuple(shade))
        return np.interp(v, vt, it, right=it[-1] - (it[-2] - it[-1]))  # mild extrapolation

    def _table(self, G, t_c, shade):
        key = (round(G, 6), round(t_c, 6), shade)
        tab = self._cache.get(key)
        if tab is not None:
            return tab
        p = self.params
        iph_max = max(p.at(G * s, t_c)[0] for s in shade)
        i_str = np.linspace(0.0, iph_max * 0.99999, 3000)
        v_str = np.zeros_like(i_str)
        for s in set(shade):
            n = shade.count(s)
            vm = module_voltage(i_str, max(G * s, 1e-3), t_c, p)
            v_str += n * np.maximum(vm, -self.vd_bypass)
        order = np.argsort(v_str)
        tab = (v_str[order], self.n_parallel * i_str[order])
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = tab
        return tab

    def voc(self, G, t_c):
        p = self.params
        dT = t_c - T_REF
        return self.n_series * max(p.ds.Voc + p.ds.Kv * dT, 1.0)

    def mpp(self, G, t_c, shade=None):
        """Global maximum power point (V, I, P) by dense scan + bounded refinement."""
        voc = self.voc(G, t_c) * 1.05
        v = np.linspace(0.0, voc, 400)
        pw = v * self.current(v, G, t_c, shade)
        k = int(np.argmax(pw))
        lo, hi = v[max(k - 1, 0)], v[min(k + 1, len(v) - 1)]
        r = minimize_scalar(lambda x: -x * float(self.current(x, G, t_c, shade)),
                            bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        vm = float(r.x)
        im = float(self.current(vm, G, t_c, shade))
        return vm, im, vm * im

    @property
    def rated_power(self):
        return self.n_series * self.n_parallel * self.params.ds.Pmax
