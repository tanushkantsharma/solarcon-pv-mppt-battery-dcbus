import numpy as np
import pytest

from src import battery_model as bm
from src.boost_converter import BoostParams, derivatives, ideal_duty, inductor_ripple_pp
from src.control import PI
from src.energy_manager import EnergyManager
from src.mppt import AdaptivePO, IncrementalConductance, PerturbObserve
from src.pv_model import KC200GT, PVArray, extract_parameters, module_current, module_voltage


@pytest.fixture(scope="module")
def p():
    return extract_parameters()


def test_parameters_physical(p):
    assert 0 < p.Rs < 1 and 50 < p.Rsh < 5000


def test_stc_datasheet_points(p):
    ds = KC200GT
    assert module_current(0.0, 1000, 25, p) == pytest.approx(ds.Isc, rel=1e-3)
    assert abs(module_current(ds.Voc, 1000, 25, p)) < 1e-6
    assert module_current(ds.Vmp, 1000, 25, p) == pytest.approx(ds.Imp, rel=1e-6)
    v = np.linspace(0, ds.Voc, 20001)
    pw = v * module_current(v, 1000, 25, p)
    assert v[np.argmax(pw)] == pytest.approx(ds.Vmp, abs=0.01)
    assert pw.max() == pytest.approx(ds.Pmax, rel=2e-3)


def test_noct_datasheet_point(p):
    """Independent check: datasheet NOCT values (800 W/m^2, 47 C): 142 W, Isc 6.62 A, Voc 29.9 V."""
    v = np.linspace(0, 32, 20001)
    i = module_current(v, 800, 47, p)
    assert (v * i).max() == pytest.approx(142.0, rel=0.02)
    assert module_current(0.0, 800, 47, p) == pytest.approx(6.62, rel=0.01)
    assert float(np.interp(0, -i, v)) == pytest.approx(29.9, rel=0.01)


def test_temperature_coefficient_voc(p):
    for t in (15, 45, 65):
        voc = KC200GT.Voc + KC200GT.Kv * (t - 25)
        assert abs(module_current(voc, 1000, t, p)) < 1e-6


def test_explicit_inverse_consistency(p):
    v = np.linspace(0, 32, 50)
    i = module_current(v, 700, 40, p)
    assert np.allclose(module_voltage(i, 700, 40, p), v, atol=1e-6)


def test_solution_satisfies_implicit_equation(p):
    v = np.linspace(-5, 34, 200)
    i = module_current(v, 600, 35, p)
    iph, i0, avt = p.at(600, 35)
    res = iph - i0 * np.expm1((v + i * p.Rs) / avt) - (v + i * p.Rs) / p.Rsh - i
    assert np.max(np.abs(res)) < 1e-8


def test_no_overflow_extreme_voltage(p):
    i = module_current(np.array([-100.0, 0.0, 60.0, 500.0]), 1000, 25, p)
    assert np.all(np.isfinite(i))


def test_array_scaling_and_shading():
    a = PVArray()
    vm, im, pm = a.mpp(1000, 25)
    assert pm == pytest.approx(8 * 3 * 200.0, rel=3e-3)
    assert vm == pytest.approx(8 * 26.3, rel=0.01)
    v = np.linspace(0, a.voc(1000, 25), 800)
    pw = v * a.current(v, 1000, 25, (1, 1, 1, 1, 1, 0.3, 0.3, 0.3))
    peaks = np.nonzero((pw[1:-1] > pw[:-2]) & (pw[1:-1] > pw[2:]))[0]
    assert len(peaks) >= 2                      # multiple local maxima
    _, _, pg = a.mpp(1000, 25, (1, 1, 1, 1, 1, 0.3, 0.3, 0.3))
    assert pg == pytest.approx(pw.max(), rel=0.01) and pg < pm


def test_boost_steady_state_duty():
    bp = BoostParams()
    d = ideal_duty(210.0, 400.0)
    dv, di = derivatives(210.0, 20.0, 400.0, d, 20.0, bp)
    assert abs(dv) < 1e-9 and abs(di) < 1e-9
    assert inductor_ripple_pp(210.0, d, bp) == pytest.approx(210 * d / (2e-3 * 20e3))


def test_boost_diode_blocks_reverse_current():
    _, di = derivatives(100.0, 0.0, 400.0, 0.0, 5.0, BoostParams())
    assert di == 0.0


def test_battery_ocv_and_soc_rate():
    b = bm.BatteryParams()
    assert 290 < bm.ocv(0.5, b) < 310
    assert np.all(np.diff([bm.ocv(s, b) for s in np.linspace(0, 1, 21)]) > 0)
    dsoc, dv1 = bm.derivatives(0.5, 0.0, 40.0, b)
    assert dsoc == pytest.approx(-40 / (3600 * 40))
    assert bm.terminal_voltage(0.5, 0.0, 10.0, b) == pytest.approx(bm.ocv(0.5, b) - 10 * b.R0)


def test_pi_antiwindup():
    c = PI(1.0, 100.0, 1e-3, lo=-1, hi=1)
    for _ in range(5000):
        u = c.step(10.0)
    assert u == 1.0 and c.x < 5.0            # integrator bounded by back-calculation
    for _ in range(20):
        u = c.step(-10.0)
    assert u == -1.0                          # recovers quickly


def test_ems_current_limits_derate():
    ems = EnergyManager(50e-6, bm.BatteryParams())
    assert ems.current_limits(0.5) == (20.0, 40.0)
    ich, idis = ems.current_limits(0.9)
    assert ich == 0.0 and idis == 40.0
    ich, idis = ems.current_limits(0.1)
    assert ich == 20.0 and idis == 0.0
    assert ems.current_limits(0.89)[0] == pytest.approx(10.0)


def _static_climb(tracker, a, n=400):
    for _ in range(n):
        v = tracker.v_ref
        tracker.update(v, float(a.current(v, 1000, 25)), v, float(a.current(v, 1000, 25)))
    return tracker.v_ref


@pytest.mark.parametrize("cls,kw", [(PerturbObserve, dict(step=2.0)),
                                    (IncrementalConductance, dict(step=2.0)),
                                    (AdaptivePO, dict(M=0.1, dv_min=0.5, dv_max=8))])
def test_trackers_converge_on_static_curve(cls, kw):
    a = PVArray()
    vm = a.mpp(1000, 25)[0]
    for v0 in (150.0, 250.0):
        t = cls(v0, 80.0, 284.0, **kw)
        assert abs(_static_climb(t, a) - vm) < 5.0


def test_global_scan_escapes_local_maximum():
    a = PVArray()
    sh = (1, 1, 1, 1, 1, 0.3, 0.3, 0.3)
    vg = a.mpp(1000, 25, sh)[0]
    t = AdaptivePO(225.0, 80.0, 284.0, M=0.1, dv_min=0.5, dv_max=8, scan=True)
    t._start_scan()
    for _ in range(100):
        v = t.v_ref
        t.update(v, float(a.current(v, 1000, 25, sh)))
    assert abs(t.v_ref - vg) < 10.0
