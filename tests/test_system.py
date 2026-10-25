"""Short closed-loop simulations (a few hundred ms each)."""
import numpy as np
import pytest

from src import metrics as M
from src.profiles import Scenario, const, steps
from src.system import SystemConfig, lossy, simulate

APO = dict(M=0.1, dv_min=1.0, dv_max=8.0, period=0.005)


def test_stc_tracking_and_bus_regulation():
    sc = Scenario("t", 0.6, const(1000, .6), const(25, .6), const(2000, .6))
    df, _ = simulate(sc, "apo", APO)
    assert M.tracking_efficiency(df, 0.3, 0.6) > 0.995
    assert M.bus_metrics(df, t0=0.1)["vdc_max_dev_V"] < 2.0
    # power balance: PV - load = battery bus power (ideal converters, steady state)
    w = df[df.t > 0.3]
    p_bat_bus = w.p_load.mean() - w.p_boost_out.mean()
    assert p_bat_bus == pytest.approx((w.i_b * (1 - w.d_b) * w.v_dc).mean(), rel=0.02)


def test_lossy_converter_efficiency_below_one():
    sc = Scenario("t", 0.5, const(1000, .5), const(25, .5), const(2000, .5))
    df, _ = simulate(sc, "po", dict(step=2.0, period=0.005), lossy())
    eff = M.converter_efficiency(df, 0.3, 0.5)
    assert 0.95 < eff < 0.999


def test_soc_high_triggers_curtailment_without_violation():
    sc = Scenario("t", 1.0, const(1000, 1), const(25, 1), const(0, 1), soc0=0.895)
    df, _ = simulate(sc, "apo", APO)
    assert (df["mode"] == "CURTAIL").any()
    b = M.battery_metrics(df, SystemConfig().bat)
    assert b["t_soc_violation_s"] == 0
    assert -df.i_b[df.t > 0.6].min() <= df.i_ch_lim[df.t > 0.6].max() + 0.5
    assert abs(df.v_dc[df.t > 0.6].mean() - 410) < 2.0


def test_soc_low_triggers_load_shedding():
    sc = Scenario("t", 1.0, const(0, 1), const(25, 1), const(4000, 1), soc0=0.105)
    df, _ = simulate(sc, "apo", APO)
    assert (df["mode"] == "SHED").any()
    assert M.curtailment_unserved(df)["unserved_Wh"] > 0
    assert df.i_b.max() <= df.i_dis_lim.max() + 0.5


def test_deterministic_with_noise():
    sc = Scenario("t", 0.2, steps(0.2, (0, 1000), (0.1, 500)), const(25, .2), const(1000, .2))
    cfg = SystemConfig(noise=0.005, seed=3)
    a, _ = simulate(sc, "po", dict(step=2.0, period=0.005), cfg)
    b, _ = simulate(sc, "po", dict(step=2.0, period=0.005), cfg)
    assert np.array_equal(a.v_pv.values, b.v_pv.values)
