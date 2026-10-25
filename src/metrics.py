"""Performance metrics computed from logged simulation data (sample period of
the log = log_every x Ts). Integrals use the trapezoidal rule over the stated
window [t0, t1]. See docs/model_equations.md for definitions."""
import numpy as np


def _w(df, t0=None, t1=None):
    m = np.ones(len(df), bool)
    if t0 is not None:
        m &= df.t.values >= t0
    if t1 is not None:
        m &= df.t.values < t1
    return df[m]


def energy_wh(df, col, t0=None, t1=None):
    w = _w(df, t0, t1)
    return float(np.trapezoid(w[col].values, w.t.values) / 3600.0)


def tracking_efficiency(df, t0=None, t1=None):
    """eta_MPPT = int P_pv dt / int P_mpp dt (P_mpp = global MPP of static curve)."""
    w = _w(df, t0, t1)
    den = np.trapezoid(w.p_mpp.values, w.t.values)
    return float(np.trapezoid(w.p_pv.values, w.t.values) / den) if den > 0 else float("nan")


def settling_time(df, t_step, t_end, p_band=0.01, v_band=0.02):
    """Time after t_step until P_pv stays within p_band x P_mpp of P_mpp AND
    v_pv stays within v_band x V_mpp of V_mpp, until t_end."""
    w = _w(df, t_step, t_end)
    err = ((np.abs(w.p_pv.values - w.p_mpp.values) > p_band * np.maximum(w.p_mpp.values, 1.0))
           | (np.abs(w.v_pv.values - w.v_mpp.values) > v_band * np.maximum(w.v_mpp.values, 1.0)))
    if not err.any():
        return 0.0
    last = np.nonzero(err)[0][-1]
    if last >= len(w) - 1:
        return float("nan")      # never settled inside the window
    return float(w.t.values[last + 1] - t_step)


def reach_time(df, t_step, t_end, p_band=0.01):
    """Time after t_step until P_pv first comes within p_band x P_mpp."""
    w = _w(df, t_step, t_end)
    ok = np.abs(w.p_pv.values - w.p_mpp.values) <= p_band * np.maximum(w.p_mpp.values, 1.0)
    return float(w.t.values[np.argmax(ok)] - t_step) if ok.any() else float("nan")


def steady_oscillation(df, t0, t1):
    """Peak-to-peak PV voltage (V) and power ripple (% of mean) in [t0, t1)."""
    w = _w(df, t0, t1)
    return (float(w.v_pv.max() - w.v_pv.min()),
            float(100 * (w.p_pv.max() - w.p_pv.min()) / max(w.p_pv.mean(), 1.0)))


def bus_metrics(df, v_ref=400.0, t0=None, t1=None):
    w = _w(df, t0, t1)
    e = w.v_dc.values - v_ref
    return {"vdc_rms_err_V": float(np.sqrt(np.mean(e ** 2))),
            "vdc_max_dev_V": float(np.max(np.abs(e))),
            "vdc_overshoot_V": float(max(e.max(), 0.0)),
            "vdc_undershoot_V": float(max(-e.min(), 0.0))}


def battery_metrics(df, bat, t0=None, t1=None, tol=0.5):
    w = _w(df, t0, t1)
    dt = float(np.median(np.diff(df.t.values)))
    i = w.i_b.values
    viol_i = (i > bat.I_dis_max + tol) | (-i > bat.I_ch_max + tol) \
        | (i > w.i_dis_lim.values + tol) | (-i > w.i_ch_lim.values + tol)
    viol_soc = (w.soc.values < bat.soc_min) | (w.soc.values > bat.soc_max)
    viol_v = (w.v_bat.values < bat.V_min) | (w.v_bat.values > bat.V_max)
    return {"ib_rms_A": float(np.sqrt(np.mean(i ** 2))),
            "soc_start": float(w.soc.values[0]), "soc_end": float(w.soc.values[-1]),
            "t_current_violation_s": float(viol_i.sum() * dt),
            "t_soc_violation_s": float(viol_soc.sum() * dt),
            "t_voltage_violation_s": float(viol_v.sum() * dt)}


def curtailment_unserved(df, t0=None, t1=None):
    w = _w(df, t0, t1)
    curt = np.where(w["mode"].values == "CURTAIL", w.p_mpp.values - w.p_pv.values, 0.0)
    uns = w.p_load.values - w.p_served.values
    return {"curtailed_Wh": float(np.trapezoid(curt, w.t.values) / 3600),
            "unserved_Wh": float(np.trapezoid(uns, w.t.values) / 3600)}


def converter_efficiency(df, t0=None, t1=None):
    w = _w(df, t0, t1)
    pin = np.trapezoid(w.p_boost_in.values, w.t.values)
    return float(np.trapezoid(w.p_boost_out.values, w.t.values) / pin) if pin > 0 else float("nan")
