"""Coupled averaged simulation of PV + boost + DC bus + battery converter + EMS.

States x = [v_pv, i_L, v_dc, i_b, v1, soc]. Plant integrated with classical
RK4 at the controller sampling period Ts (50 us) with zero-order-hold duty
cycles. Controllers execute once per Ts on (optionally noisy) measurements.

Control structure
  PV side   : MPPT (every T_mppt) -> v_ref ; PI voltage loop -> i_L_ref
              (+ i_pv feed-forward) ; PI current loop -> d (feed-forward of
              v_pv and v_dc).
  Battery   : PI bus-voltage loop -> i_b_ref (+ bus-current feed-forward),
              saturated to the EMS current limits ; PI current loop -> d_b.
  EMS       : src/energy_manager.py (curtailment / load shedding bands).
All PI loops use back-calculation anti-windup; all duties are saturated.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import battery_model as bm
from .bidirectional_converter import BidirParams
from .boost_converter import BoostParams, switching_loss
from .control import PI
from .energy_manager import EMSParams, EnergyManager
from .mppt import make_tracker
from .profiles import Scenario
from .pv_model import PVArray


@dataclass
class SystemConfig:
    ts: float = 50e-6
    C_dc: float = 2e-3
    boost: BoostParams = field(default_factory=BoostParams)
    bidir: BidirParams = field(default_factory=BidirParams)
    bat: bm.BatteryParams = field(default_factory=bm.BatteryParams)
    ems: EMSParams = field(default_factory=EMSParams)
    noise: float = 0.0            # measurement noise std, fraction of full scale
    bus_ff: bool = True           # bus-current feed-forward in the battery voltage loop
    seed: int = 0
    log_every: int = 10           # log decimation (10 x 50 us = 0.5 ms)
    # loop bandwidths (Hz) used for PI design
    f_i_pv: float = 1500.0
    f_v_pv: float = 200.0
    f_i_bat: float = 1000.0
    f_v_dc: float = 40.0
    full_scale: tuple = (300.0, 30.0, 500.0, 60.0, 400.0)  # v_pv, i_pv/i_L, v_dc, i_b, v_bat


def simulate(sc: Scenario, tracker_kind: str, mppt_params: dict,
             cfg: SystemConfig = SystemConfig(), pv: PVArray | None = None,
             mpp_cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    pv = pv or PVArray()
    bst, bdp, bat, ts = cfg.boost, cfg.bidir, cfg.bat, cfg.ts
    rng = np.random.default_rng(cfg.seed)
    Cdc = cfg.C_dc
    voc_stc = pv.voc(1000.0, 25.0)
    v_lo, v_hi = 0.30 * voc_stc, 1.00 * voc_stc * 1.08
    v_reset = 0.80 * voc_stc                   # reference after a PV_OFF period
    g, tc, pl, sh = sc.at(0.0)
    v_oc0 = pv.voc(g, tc) if g > 1 else 0.0    # Cin pre-charged to open circuit
    v0 = max(0.9 * v_oc0, v_lo)                # fractional-Voc start reference
    mp = dict(mppt_params)
    period = mp.pop("period", 0.01)
    n_mppt = max(1, int(round(period / ts)))
    trk = make_tracker(tracker_kind, v0, v_lo, v_hi, p_rated=pv.rated_power, period=period, **mp)
    ems = EnergyManager(ts, bat, cfg.ems)
    i_max = 1.3 * pv.n_parallel * pv.params.ds.Isc

    # PI design (crossover w = 2 pi f; integral corner one decade lower)
    w = 2 * math.pi
    pi_i = PI(bst.L * w * cfg.f_i_pv, bst.L * w * cfg.f_i_pv * w * cfg.f_i_pv / 10, ts)
    kpv = bst.Cin * w * cfg.f_v_pv
    pi_v = PI(kpv, kpv * w * cfg.f_v_pv / 10, ts, 0.0, i_max)
    pi_ib = PI(bdp.L * w * cfg.f_i_bat, bdp.L * w * cfg.f_i_bat * w * cfg.f_i_bat / 10, ts)
    kdc = Cdc * w * cfg.f_v_dc * 1.33
    pi_dc = PI(kdc, kdc * w * cfg.f_v_dc / 10, ts)

    # initial operating point: converter wakes up at PV open circuit, i_L = 0
    v_pv = v_oc0
    i_L = 0.0
    v_dc = cfg.ems.V_ref
    soc, v1 = sc.soc0, 0.0
    i_ch0, i_dis0 = ems.current_limits(soc)
    i_b = min(max((pl - v_pv * i_L) / bm.ocv(soc, bat), -i_ch0), i_dis0)   # respect limits
    v1 = bat.R1 * i_b
    pi_v.reset(0.0)
    d = 0.0
    d_b = 1 - bm.terminal_voltage(soc, v1, i_b, bat) / v_dc
    pi_dc.reset(0.0)

    fs_noise = [s * cfg.noise for s in cfg.full_scale]
    n_steps = int(round(sc.t_end / ts))
    acc_v = acc_i = ax_v = ax_i = 0.0
    acc_n = 0
    n_qa, n_qb, n_qc = n_mppt // 4, n_mppt // 2, (3 * n_mppt) // 4
    v_ref = v0
    dv_curt, p_served, freeze = 0.0, pl, False
    i_ch, i_dis = ems.current_limits(soc)
    prev_mode = "NORMAL"
    mpp_cache = {} if mpp_cache is None else mpp_cache
    rows = []
    stiff = sc.stiff_bus
    t_wall = time.perf_counter()

    def f(x, d, d_b, g, tc, sh, p_srv):
        v_pv_, i_L_, v_dc_, i_b_, v1_, soc_ = x
        i_pv = float(pv.current(v_pv_, g, tc, sh))
        dv_pv = (i_pv - i_L_) / bst.Cin
        di_L = (v_pv_ - (bst.R_L + d * bst.R_on) * i_L_ - (1 - d) * (v_dc_ + bst.V_F)) / bst.L
        if i_L_ <= 0.0 and di_L < 0.0:
            di_L = 0.0
        if stiff:
            return (dv_pv, di_L, 0.0, 0.0, 0.0, 0.0)
        vdc_s = max(v_dc_, 50.0)
        i_out = (1 - d) * i_L_ - switching_loss(i_L_, vdc_s, bst) / vdc_s
        v_t = bm.ocv(soc_, bat) - v1_ - bat.R0 * i_b_
        di_b = (v_t - bdp.R_L * i_b_ - (1 - d_b) * v_dc_) / bdp.L
        dv_dc = (i_out + (1 - d_b) * i_b_ - p_srv / vdc_s) / Cdc
        dsoc, dv1 = bm.derivatives(soc_, v1_, i_b_, bat)
        return (dv_pv, di_L, dv_dc, di_b, dv1, dsoc)

    x = [v_pv, i_L, v_dc, i_b, v1, soc]
    for k in range(n_steps):
        t = k * ts
        g, tc, pl, sh = sc.at(t)
        v_pv, i_L, v_dc, i_b, v1, soc = x
        i_pv = float(pv.current(v_pv, g, tc, sh))
        v_bt = bm.terminal_voltage(soc, v1, i_b, bat)
        if cfg.noise > 0:
            nz = rng.standard_normal(6)
            m_vpv = v_pv + fs_noise[0] * nz[0]
            m_ipv = i_pv + fs_noise[1] * nz[1]
            m_iL = i_L + fs_noise[1] * nz[2]
            m_vdc = v_dc + fs_noise[2] * nz[3]
            m_ib = i_b + fs_noise[3] * nz[4]
            m_vbt = v_bt + fs_noise[4] * nz[5]
        else:
            m_vpv, m_ipv, m_iL, m_vdc, m_ib, m_vbt = v_pv, i_pv, i_L, v_dc, i_b, v_bt

        # ---- supervisory EMS ----
        if not stiff:
            i_ch, i_dis, dv_curt, p_served, freeze = ems.step(m_vdc, soc, pl, m_vpv * m_ipv)
            if ems.mode == "PV_OFF" and prev_mode != "PV_OFF":
                trk.reset(v_reset)
            prev_mode = ems.mode

        # ---- MPPT ----
        # all trackers see the same settled measurement: average over the last
        # quarter of the MPPT period (window B); dP-P&O also gets window A
        # (second quarter) as its mid-period measurement.
        if n_qa <= acc_n < n_qb:
            ax_v += m_vpv
            ax_i += m_ipv
        elif acc_n >= n_qc:
            acc_v += m_vpv
            acc_i += m_ipv
        acc_n += 1
        if acc_n == n_mppt:
            if not freeze:
                trk.update(acc_v / (n_mppt - n_qc), acc_i / (n_mppt - n_qc),
                           ax_v / (n_qb - n_qa), ax_i / (n_qb - n_qa))
            acc_v = acc_i = ax_v = ax_i = 0.0
            acc_n = 0
        v_ref = min(trk.v_ref + dv_curt, v_hi)

        # ---- PV-side cascaded control ----
        vdc_c = max(m_vdc, 50.0)
        i_ref = pi_v.step(m_vpv - v_ref, ff=m_ipv)
        u_lo = m_vpv - (1 - bst.d_min) * vdc_c
        u_hi = m_vpv - (1 - bst.d_max) * vdc_c
        u = pi_i.step(i_ref - m_iL, lo=u_lo, hi=u_hi)
        d = 1 - (m_vpv - u) / vdc_c
        d = min(max(d, bst.d_min), bst.d_max)

        # ---- battery-side control ----
        if not stiff:
            vbt_c = max(m_vbt, 50.0)
            i_bus_ff = (p_served / vdc_c - (1 - d) * m_iL) if cfg.bus_ff else 0.0
            ib_ref = pi_dc.step(cfg.ems.V_ref - m_vdc, ff=i_bus_ff * vdc_c / vbt_c,
                                lo=-i_ch, hi=i_dis)
            ub_lo = m_vbt - (1 - bdp.d_min) * vdc_c
            ub_hi = m_vbt - (1 - bdp.d_max) * vdc_c
            ub = pi_ib.step(ib_ref - m_ib, lo=ub_lo, hi=ub_hi)
            d_b = 1 - (m_vbt - ub) / vdc_c
            d_b = min(max(d_b, bdp.d_min), bdp.d_max)
        else:
            ib_ref = 0.0

        if k % cfg.log_every == 0:
            key = (round(g * 2) / 2, round(tc * 20) / 20, sh)
            mp_ = mpp_cache.get(key)
            if mp_ is None:
                mp_ = pv.mpp(*key) if key[0] > 0 else (0.0, 0.0, 0.0)
                mpp_cache[key] = mp_
            v_mpp, p_mpp = mp_[0], mp_[2]
            vdc_l = max(v_dc, 50.0)
            rows.append((t, g, tc, v_pv, i_pv, v_pv * i_pv, p_mpp, v_mpp, v_ref, i_L, d,
                         v_dc, i_b, ib_ref, v_bt, soc, d_b, pl, p_served,
                         i_ch, i_dis, ems.mode if not stiff else "STIFF",
                         v_pv * i_L, v_dc * ((1 - d) * i_L) - switching_loss(i_L, vdc_l, bst)))

        # ---- plant integration (RK4, ZOH inputs) ----
        a = (d, d_b, g, tc, sh, p_served)
        k1 = f(x, *a)
        x2 = [xi + 0.5 * ts * ki for xi, ki in zip(x, k1)]
        k2 = f(x2, *a)
        x3 = [xi + 0.5 * ts * ki for xi, ki in zip(x, k2)]
        k3 = f(x3, *a)
        x4 = [xi + ts * ki for xi, ki in zip(x, k3)]
        k4 = f(x4, *a)
        x = [xi + ts / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
             for xi, a1, a2, a3, a4 in zip(x, k1, k2, k3, k4)]
        if x[1] < 0.0:
            x[1] = 0.0
        if stiff:
            x[2] = cfg.ems.V_ref

    runtime = time.perf_counter() - t_wall
    df = pd.DataFrame(rows, columns=[
        "t", "G", "T", "v_pv", "i_pv", "p_pv", "p_mpp", "v_mpp", "v_ref", "i_L", "d",
        "v_dc", "i_b", "i_b_ref", "v_bat", "soc", "d_b", "p_load", "p_served",
        "i_ch_lim", "i_dis_lim", "mode", "p_boost_in", "p_boost_out"])
    info = {"runtime_s": runtime, "sim_time_s": sc.t_end,
            "n_scans": getattr(trk, "n_scans", 0), "tracker": trk.name}
    return df, info


def lossy(cfg: SystemConfig = SystemConfig()) -> SystemConfig:
    return replace(cfg, boost=cfg.boost.lossy())
