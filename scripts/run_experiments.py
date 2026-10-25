"""Run every SolarCon experiment (A-J), write metrics CSVs and figures.

    python scripts/run_experiments.py            # uses optimization/tuned_params.json
    python scripts/run_experiments.py --quick    # shorter Monte-Carlo (CI smoke run)

All profiles are synthetic and deterministic; Monte-Carlo runs use fixed seeds.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as M  # noqa: E402
from src.boost_converter import inductor_ripple_pp  # noqa: E402
from src.mppt import LABELS  # noqa: E402
from src.profiles import Scenario, const, pwl, steps  # noqa: E402
from src.pv_model import KC200GT, PVArray, extract_parameters, module_current  # noqa: E402
from src.system import SystemConfig, lossy, simulate  # noqa: E402

FIG = ROOT / "results" / "figures"
MET = ROOT / "results" / "metrics"
BASE = ["po", "inc", "dpo", "apo"]
COLORS = {"po": "#2a78d6", "inc": "#eb6834", "dpo": "#1baf7a", "apo": "#c98500", "apo_scan": "#d55181"}
STYLES = {"po": "-", "inc": "--", "dpo": "-.", "apo": "-", "apo_scan": ":"}
INK, MUTED = "#3d3d3a", "#8a8980"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "lines.linewidth": 1.3, "legend.frameon": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})
SHADE1 = (1, 1, 1, 1, 1, 0.3, 0.3, 0.3)
SHADE2 = (1, 1, 0.6, 0.6, 0.6, 0.25, 0.25, 0.25)


# ------------------------------------------------------------------ scenarios
def scenario(name: str) -> Scenario:
    if name == "A":
        return Scenario("A", 1.5, const(1000, 1.5), const(25, 1.5), const(2000, 1.5), eval_start=0.3)
    if name == "B":
        return Scenario("B", 4.0, steps(4.0, (0, 1000), (1.0, 600), (2.0, 300), (3.0, 800)),
                        const(25, 4), const(2500, 4), segments=[0.0, 1.0, 2.0, 3.0, 4.0])
    if name.startswith("C_T"):
        tc = float(name[3:])
        return Scenario(name, 0.8, const(1000, .8), const(tc, .8), const(0, .8), stiff_bus=True, eval_start=0.3)
    if name.startswith("C_G"):
        g = float(name[3:])
        return Scenario(name, 0.8, const(g, .8), const(25, .8), const(0, .8), stiff_bus=True, eval_start=0.3)
    if name == "D":
        return Scenario("D", 4.0, pwl((0, 1000), (0.5, 1000), (2.5, 300), (3.0, 300), (3.5, 1000), (4.0, 1000)),
                        pwl((0, 25), (4, 45)), const(2500, 4), segments=[0.5, 2.5, 3.0, 3.5])
    if name == "E":
        return Scenario("E", 5.0, const(1000, 5), const(25, 5), const(2500, 5),
                        shade=[(0.0, None), (1.0, SHADE1), (3.0, SHADE2)], segments=[0.0, 1.0, 3.0, 5.0])
    if name == "F":
        return Scenario("F", 5.0, steps(5, (0, 1000), (1.5, 300), (3.0, 800), (4.0, 0)),
                        const(25, 5), const(3000, 5), segments=[0, 1.5, 3.0, 4.0, 5.0])
    if name == "G1":
        return Scenario("G1", 2.5, const(1000, 2.5), const(25, 2.5), steps(2.5, (0, 1500), (1.0, 0)),
                        soc0=0.888)
    if name == "G2":
        return Scenario("G2", 3.0, const(300, 3), const(25, 3), steps(3, (0, 3000), (1.0, 6000), (2.0, 2000)),
                        soc0=0.105)
    if name == "H":
        return Scenario("H", 4.0, const(1000, 4), const(25, 4),
                        steps(4, (0, 1000), (1.0, 6000), (2.0, 500), (3.0, 3000)), segments=[1.0, 2.0, 3.0, 4.0])
    raise KeyError(name)


def perturbed(seed: int):
    """Monte-Carlo plant: component tolerances + 0.5 % FS measurement noise."""
    rng = np.random.default_rng(1000 + seed)
    u = lambda r: 1 + rng.uniform(-r, r)  # noqa: E731
    cfg = SystemConfig(noise=0.005, seed=seed)
    cfg = replace(cfg, C_dc=cfg.C_dc * u(0.2),
                  boost=replace(cfg.boost, L=cfg.boost.L * u(0.2), Cin=cfg.boost.Cin * u(0.2)),
                  bidir=replace(cfg.bidir, L=cfg.bidir.L * u(0.2)),
                  bat=replace(cfg.bat, R0=cfg.bat.R0 * u(0.3), Q_Ah=cfg.bat.Q_Ah * u(0.1)))
    p = extract_parameters()
    pv_p = replace(p, Rs=p.Rs * u(0.1), Rsh=p.Rsh * u(0.1))
    return cfg, pv_p


# ------------------------------------------------------------------ job runner
def _job(spec):
    sid, scn, kind, params, variant, seed = spec
    sc = scenario(scn)
    pv = None
    if variant == "mc":
        cfg, pv_p = perturbed(seed)
        pv = PVArray(params=pv_p)
    elif variant == "lossy":
        cfg = lossy()
    elif variant == "noff":
        cfg = SystemConfig(bus_ff=False)
    else:
        cfg = SystemConfig()
    df, info = simulate(sc, kind, params, cfg, pv=pv)
    return sid, df, info, cfg


def run_jobs(specs, procs):
    with Pool(procs) as pool:
        out = pool.map(_job, specs)
    return {sid: (df, info, cfg) for sid, df, info, cfg in out}


# ------------------------------------------------------------------ helpers
def tracking_row(df, info, sc, kind, extra=None):
    t0 = sc.eval_start
    row = {"method": LABELS[kind], "eta_mppt": M.tracking_efficiency(df, t0, sc.t_end),
           "E_pv_Wh": M.energy_wh(df, "p_pv", t0, sc.t_end), "E_mpp_Wh": M.energy_wh(df, "p_mpp", t0, sc.t_end),
           "runtime_s": info["runtime_s"], "realtime_factor": info["runtime_s"] / sc.t_end}
    row.update(extra or {})
    return row


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fmt_table(df, floatfmt=4):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ PV validation
def pv_validation():
    arr = PVArray()
    p = arr.params
    ds = KC200GT
    v = np.linspace(0, ds.Voc * 1.02, 600)
    rows = []
    # STC datasheet point and NOCT datasheet point (800 W/m^2, cell 47 degC)
    for label, G, T, ref in [("STC", 1000, 25, (ds.Isc, ds.Voc, ds.Vmp, ds.Imp, ds.Pmax)),
                             ("NOCT (800 W/m2, 47 C)", 800, 47, (6.62, 29.9, 23.2, 6.13, 142.0))]:
        i = module_current(v, G, T, p)
        pw = v * i
        k = np.argmax(pw)
        isc = float(module_current(0.0, G, T, p))
        voc = float(np.interp(0, -i, v))
        mod = (isc, voc, v[k], i[k], pw[k])
        for name, m, r in zip(["Isc_A", "Voc_V", "Vmp_V", "Imp_A", "Pmax_W"], mod, ref):
            rows.append({"condition": label, "quantity": name, "datasheet": r, "model": round(float(m), 3),
                         "error_pct": round(100 * (m - r) / r, 2)})
    val = pd.DataFrame(rows)
    val.to_csv(MET / "pv_validation.csv", index=False)
    params = {"module": ds.name, "a": p.a, "Rs_ohm": p.Rs, "Rsh_ohm": p.Rsh, "Iph_ref_A": p.Iph_ref,
              "I0_ref_A": float(p.at(1000, 25)[1]), "array": f"{arr.n_series}s x {arr.n_parallel}p",
              "array_rated_W": arr.rated_power}
    (MET / "pv_parameters.json").write_text(json.dumps(params, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    va = np.linspace(0, arr.voc(1000, 15) * 1.01, 500)
    for G, c in zip([1000, 800, 600, 400, 200], ["#104281", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]):
        ia = arr.current(va, G, 25)
        ax[0].plot(va, np.clip(ia, 0, None), color=c, label=f"{G} W/m²")
        ax[1].plot(va, np.clip(va * ia, 0, None) / 1e3, color=c)
    for T, ls in [(15, ":"), (65, "--")]:
        ia = arr.current(va, 1000, T)
        ax[1].plot(va, np.clip(va * ia, 0, None) / 1e3, color=MUTED, ls=ls, label=f"1000 W/m², {T} °C")
    ax[0].plot([0, arr.n_series * ds.Vmp, arr.n_series * ds.Voc],
               [arr.n_parallel * ds.Isc, arr.n_parallel * ds.Imp, 0], "o", color=INK, ms=4,
               label="datasheet (STC, scaled)")
    ax[0].set(xlabel="Array voltage (V)", ylabel="Array current (A)", ylim=(0, 27), title="I–V, 25 °C")
    ax[1].set(xlabel="Array voltage (V)", ylabel="Array power (kW)", ylim=(0, 5.4), title="P–V")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    savefig(fig, "pv_iv_pv_curves")

    # partial-shading P-V curves
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for sh, lab, c in [(None, "uniform", "#2a78d6"), (SHADE1, "pattern 1", "#eb6834"), (SHADE2, "pattern 2", "#1baf7a")]:
        pw = va * arr.current(va, 1000, 25, sh)
        ax.plot(va, np.clip(pw, 0, None) / 1e3, color=c, label=lab)
        vm, _, pm = arr.mpp(1000, 25, sh)
        ax.plot(vm, pm / 1e3, "o", color=c, ms=5)
    ax.set(xlabel="Array voltage (V)", ylabel="Power (kW)", ylim=(0, 5.2), title="Partial shading (dots = global MPP)")
    ax.legend(fontsize=7)
    savefig(fig, "pv_partial_shading_curves")
    return val, params


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 Monte-Carlo seeds instead of 8")
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    tuned = json.loads((ROOT / "optimization" / "tuned_params.json").read_text())
    t_start = time.perf_counter()

    val, pvpar = pv_validation()

    specs = []
    for k in BASE:
        for s in ["A", "B", "D"]:
            specs.append((f"{s}|{k}", s, k, tuned[k], "ideal", 0))
        for T in range(15, 66, 10):
            specs.append((f"C_T{T}|{k}", f"C_T{T}", k, tuned[k], "ideal", 0))
        for G in range(200, 1001, 200):
            specs.append((f"C_G{G}|{k}", f"C_G{G}", k, tuned[k], "ideal", 0))
    for k in BASE + ["apo_scan"]:
        specs.append((f"E|{k}", "E", k, tuned[k], "ideal", 0))
    specs.append(("A|apo|lossy", "A", "apo", tuned["apo"], "lossy", 0))
    for s in ["F", "G1", "G2", "H"]:
        specs.append((f"{s}|apo", s, "apo", tuned["apo"], "ideal", 0))
    specs.append(("H|apo|lossy", "H", "apo", tuned["apo"], "lossy", 0))
    specs.append(("H|apo|noff", "H", "apo", tuned["apo"], "noff", 0))
    n_mc = 3 if args.quick else 8
    for k in BASE:
        for sd in range(n_mc):
            specs.append((f"I|{k}|{sd}", "B", k, tuned[k], "mc", sd))
    R = run_jobs(specs, args.procs)
    summary = {"pv_parameters": pvpar, "tuned_params": tuned, "python": platform.python_version()}
    tables = []

    # ---------- A: STC
    rows = []
    scA = scenario("A")
    for k in BASE:
        df, info, cfg = R[f"A|{k}"]
        vpp, prip = M.steady_oscillation(df, 1.0, 1.5)
        rows.append(tracking_row(df, info, scA, k, {
            "startup_reach_s": M.reach_time(df, 0.0, 1.5),
            "startup_settling_s": M.settling_time(df, 0.0, 1.5), "vpv_pp_V": vpp, "p_ripple_pct": prip,
            **M.bus_metrics(df, t0=0.3)}))
    A = pd.DataFrame(rows)
    df, info, cfg = R["A|apo|lossy"]
    w = df[df.t >= 0.3]
    lossyA = {"converter_eff_lossy": M.converter_efficiency(df, 0.3, 1.5),
              "converter_eff_ideal": M.converter_efficiency(R["A|apo"][0], 0.3, 1.5),
              "eta_mppt_lossy": M.tracking_efficiency(df, 0.3, 1.5),
              "iL_ripple_pp_A_analytical": float(inductor_ripple_pp(w.v_pv.mean(), w.d.mean(), cfg.boost)),
              "iL_mean_A": float(w.i_L.mean()),
              "vdc_ripple_pp_V_analytical": float(w.i_L.mean() * (1 - w.d.mean()) * w.d.mean() / (cfg.C_dc * cfg.boost.fs))}
    A.to_csv(MET / "A_standard.csv", index=False)
    summary["A_lossy"] = lossyA
    tables.append(("A — STC, 1000 W/m², 25 °C (window 0.3–1.5 s)",
                   A[["method", "eta_mppt", "startup_reach_s", "startup_settling_s", "vpv_pp_V", "p_ripple_pct", "vdc_rms_err_V"]]))

    # ---------- B: irradiance steps
    scB = scenario("B")
    rows = []
    for k in BASE:
        df, info, _ = R[f"B|{k}"]
        ex = {}
        for a, b in zip(scB.segments[:-1], scB.segments[1:]):
            g = int(df[(df.t >= a) & (df.t < b)].G.iloc[-1])
            ex[f"tr_{g}_s"] = M.reach_time(df, a, b)
            ex[f"ts_{g}_s"] = M.settling_time(df, a, b)
            ex[f"eta_{g}"] = M.tracking_efficiency(df, a + 0.05 if a else 0.2, b)
        rows.append(tracking_row(df, info, scB, k, {**ex, **M.bus_metrics(df, t0=0.2)}))
    B = pd.DataFrame(rows)
    B.to_csv(MET / "B_irradiance_steps.csv", index=False)
    tables.append(("B — irradiance steps 1000→600→300→800 W/m² (window 0.2–4 s). tr = first reach of ±1 % P_mpp; "
                   "ts = settled (stays within ±1 % P_mpp and ±2 % V_mpp until the next step; ≈1 s means never)",
                   B[["method", "eta_mppt", "tr_1000_s", "tr_600_s", "tr_300_s", "tr_800_s",
                      "ts_600_s", "ts_300_s", "ts_800_s", "vdc_max_dev_V"]]))
    fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    for k in BASE:
        df = R[f"B|{k}"][0]
        ax[0].plot(df.t, df.p_pv / 1e3, STYLES[k], color=COLORS[k], label=LABELS[k])
        ax[1].plot(df.t, df.v_pv, STYLES[k], color=COLORS[k])
        ax[2].plot(df.t, df.v_dc, STYLES[k], color=COLORS[k])
    ax[0].plot(df.t, df.p_mpp / 1e3, color=INK, lw=0.8, label="available P_mpp")
    ax[0].set_ylabel("PV power (kW)")
    ax[1].set_ylabel("PV voltage (V)")
    ax[2].set(ylabel="DC bus (V)", xlabel="Time (s)")
    ax[0].legend(ncol=5, fontsize=7, loc="lower left")
    ax[0].set_title("B: irradiance steps, full system (load 2.5 kW)")
    savefig(fig, "B_irradiance_steps")
    fig, axz = plt.subplots(1, 3, figsize=(10, 3))
    for j, (a, b) in enumerate([(0.95, 1.25), (1.95, 2.25), (2.95, 3.25)]):
        for k in BASE:
            df = R[f"B|{k}"][0]
            w = df[(df.t >= a) & (df.t < b)]
            axz[j].plot(w.t, w.p_pv / 1e3, STYLES[k], color=COLORS[k], label=LABELS[k])
        axz[j].plot(w.t, w.p_mpp / 1e3, color=INK, lw=0.8)
        axz[j].set(xlabel="Time (s)", title=f"step at {a + 0.05:.0f} s")
    axz[0].set_ylabel("PV power (kW)")
    axz[0].legend(fontsize=6)
    savefig(fig, "B_step_zoom")

    # ---------- C: temperature / irradiance sweeps
    rows = []
    for k in BASE:
        for T in range(15, 66, 10):
            df, info, _ = R[f"C_T{T}|{k}"]
            vpp, prip = M.steady_oscillation(df, 0.5, 0.8)
            rows.append({"sweep": "T_degC", "value": T, "method": LABELS[k],
                         "P_mpp_W": float(df.p_mpp.iloc[-1]), "eta_mppt": M.tracking_efficiency(df, 0.3, 0.8),
                         "vpv_pp_V": vpp})
        for G in range(200, 1001, 200):
            df, info, _ = R[f"C_G{G}|{k}"]
            vpp, prip = M.steady_oscillation(df, 0.5, 0.8)
            rows.append({"sweep": "G_Wm2", "value": G, "method": LABELS[k],
                         "P_mpp_W": float(df.p_mpp.iloc[-1]), "eta_mppt": M.tracking_efficiency(df, 0.3, 0.8),
                         "vpv_pp_V": vpp})
    C = pd.DataFrame(rows)
    C.to_csv(MET / "C_sweeps.csv", index=False)
    piv = C[C.sweep == "T_degC"].pivot(index="value", columns="method", values="eta_mppt").reset_index()
    piv.columns.name = None
    tables.append(("C — temperature sweep at 1000 W/m², η_MPPT (window 0.3–0.8 s, ideal 400 V bus)",
                   piv.rename(columns={"value": "T (°C)"})))
    piv = C[C.sweep == "G_Wm2"].pivot(index="value", columns="method", values="eta_mppt").reset_index()
    piv.columns.name = None
    tables.append(("C — irradiance sweep at 25 °C, η_MPPT (window 0.3–0.8 s)", piv.rename(columns={"value": "G (W/m²)"})))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for k in BASE:
        for j, sw in enumerate(["T_degC", "G_Wm2"]):
            c = C[(C.sweep == sw) & (C.method == LABELS[k])]
            ax[j].plot(c.value, 100 * c.eta_mppt, STYLES[k], marker="o", ms=4, color=COLORS[k], label=LABELS[k])
    ax[0].set(xlabel="Cell temperature (°C)", ylabel="η_MPPT (%)", title="1000 W/m²")
    ax[1].set(xlabel="Irradiance (W/m²)", title="25 °C")
    ax[0].legend(fontsize=7)
    savefig(fig, "C_sweeps")

    # ---------- D: ramps
    scD = scenario("D")
    rows = []
    for k in BASE:
        df, info, _ = R[f"D|{k}"]
        rows.append(tracking_row(df, info, scD, k, {
            "eta_ramp_down": M.tracking_efficiency(df, 0.5, 2.5), "eta_ramp_up": M.tracking_efficiency(df, 3.0, 3.5)}))
    D = pd.DataFrame(rows)
    D.to_csv(MET / "D_ramps.csv", index=False)
    tables.append(("D — ramps: −350 W/m²/s (0.5–2.5 s), +1400 W/m²/s (3.0–3.5 s), 25→45 °C",
                   D[["method", "eta_mppt", "eta_ramp_down", "eta_ramp_up"]]))
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    for k in BASE:
        df = R[f"D|{k}"][0]
        ax[0].plot(df.t, (df.p_mpp - df.p_pv), STYLES[k], color=COLORS[k], label=LABELS[k])
        ax[1].plot(df.t, df.v_pv, STYLES[k], color=COLORS[k])
    ax[0].set(ylabel="P_mpp − P_pv (W)", title="D: irradiance ramps — tracking loss")
    ax[1].set(ylabel="PV voltage (V)", xlabel="Time (s)")
    ax[0].legend(fontsize=7, ncol=4)
    savefig(fig, "D_ramps")

    # ---------- E: partial shading
    scE = scenario("E")
    rows = []
    for k in BASE + ["apo_scan"]:
        df, info, _ = R[f"E|{k}"]
        rows.append(tracking_row(df, info, scE, k, {
            "eta_uniform": M.tracking_efficiency(df, 0.2, 1.0), "eta_pattern1": M.tracking_efficiency(df, 1.0, 3.0),
            "eta_pattern2": M.tracking_efficiency(df, 3.0, 5.0), "n_scans": info["n_scans"]}))
    E = pd.DataFrame(rows)
    E.to_csv(MET / "E_partial_shading.csv", index=False)
    tables.append(("E — partial shading (global MPP reference), window 0.2–5 s",
                   E[["method", "eta_mppt", "eta_uniform", "eta_pattern1", "eta_pattern2", "n_scans"]]))
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    for k in BASE + ["apo_scan"]:
        df = R[f"E|{k}"][0]
        ax[0].plot(df.t, df.p_pv / 1e3, STYLES[k], color=COLORS[k], label=LABELS[k])
        ax[1].plot(df.t, df.v_pv, STYLES[k], color=COLORS[k])
    ax[0].plot(df.t, df.p_mpp / 1e3, color=INK, lw=0.8, label="global P_mpp")
    ax[0].set(ylabel="PV power (kW)", title="E: partial shading (pattern 1 at 1 s, pattern 2 at 3 s)")
    ax[1].set(ylabel="PV voltage (V)", xlabel="Time (s)")
    ax[0].legend(fontsize=7, ncol=3, loc="lower left")
    savefig(fig, "E_partial_shading")

    # ---------- F, G, H: system level (proposed controller)
    sysrows = []
    for s in ["F", "G1", "G2", "H"]:
        df, info, cfg = R[f"{s}|apo"]
        modes = df["mode"].value_counts(normalize=True).round(3).to_dict()
        sysrows.append({"experiment": s, "eta_mppt_excl_curtail": M.tracking_efficiency(df[df["mode"] != "CURTAIL"]),
                        **M.bus_metrics(df, t0=0.05), **M.battery_metrics(df, cfg.bat, t0=0.0),
                        **M.curtailment_unserved(df), "modes": json.dumps(modes), "runtime_s": info["runtime_s"]})
        fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        ax[0].plot(df.t, df.p_mpp / 1e3, color=MUTED, lw=0.8, label="available PV (MPP)")
        ax[0].plot(df.t, df.p_pv / 1e3, color=COLORS["po"], label="PV")
        ax[0].plot(df.t, df.p_load / 1e3, color=INK, ls=":", label="load demand")
        ax[0].plot(df.t, df.p_served / 1e3, color=COLORS["inc"], ls="--", label="load served")
        ax[0].set_ylabel("Power (kW)")
        ax[0].legend(fontsize=7, ncol=4, loc="upper right")
        ax[1].plot(df.t, df.v_dc, color=COLORS["po"])
        for v, lab in [(400, "V_ref"), (410, "V_curt"), (390, "V_shed")]:
            ax[1].axhline(v, color=MUTED, lw=0.6, ls="--")
            ax[1].text(df.t.iloc[-1], v, " " + lab, fontsize=7, color=MUTED, va="center")
        ax[1].set_ylabel("DC bus (V)")
        ax[2].plot(df.t, df.i_b, color=COLORS["po"], label="i_b (+ discharge)")
        ax[2].plot(df.t, df.i_dis_lim, color=MUTED, ls="--", lw=0.8, label="limits")
        ax[2].plot(df.t, -df.i_ch_lim, color=MUTED, ls="--", lw=0.8)
        ax[2].set_ylabel("Battery current (A)")
        ax[2].legend(fontsize=7)
        ax[3].plot(df.t, 100 * df.soc, color=COLORS["po"])
        ax[3].set(ylabel="SOC (%)", xlabel="Time (s)")
        ax[3].ticklabel_format(useOffset=False, axis="y")
        ax[0].set_title({"F": "F: battery charge/discharge, PV unavailable at 4 s",
                         "G1": "G1: SOC near upper limit (88.8 %), charge-current saturation → PV curtailment",
                         "G2": "G2: SOC near lower limit (10.5 %), discharge-current saturation → load shedding",
                         "H": "H: DC-bus load steps (ideal converters)"}[s])
        savefig(fig, f"{s}_system")
    dfl, infol, cfgl = R["H|apo|lossy"]
    sysrows.append({"experiment": "H (lossy boost)", "eta_mppt_excl_curtail": M.tracking_efficiency(dfl),
                    **M.bus_metrics(dfl, t0=0.05), **M.battery_metrics(dfl, cfgl.bat, t0=0.0),
                    **M.curtailment_unserved(dfl), "modes": json.dumps(dfl["mode"].value_counts(normalize=True).round(3).to_dict()),
                    "runtime_s": infol["runtime_s"], "converter_eff": M.converter_efficiency(dfl)})
    SYS = pd.DataFrame(sysrows)
    SYS.to_csv(MET / "FGH_system.csv", index=False)
    tables.append(("F/G/H — system-level tests with Adaptive dP-P&O",
                   SYS[["experiment", "vdc_rms_err_V", "vdc_overshoot_V", "vdc_undershoot_V", "ib_rms_A",
                        "t_current_violation_s", "t_soc_violation_s", "curtailed_Wh", "unserved_Wh"]]))
    # H per-step bus metrics
    hrows = []
    for var, lab in [("H|apo", "with feed-forward"), ("H|apo|noff", "PI only (no feed-forward)")]:
        dfh = R[var][0]
        for a, b in zip([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]):
            w = dfh[(dfh.t >= a) & (dfh.t < b)]
            e = np.abs(w.v_dc.values - 400)
            out = np.nonzero(e > 1.0)[0]
            hrows.append({"bus_control": lab, "step_at_s": a, "load_W": f"{w.p_load.iloc[0]:.0f}",
                          "max_dev_V": float(e.max()),
                          "settle_1V_ms": float(1e3 * (w.t.values[out[-1] + 1] - a)) if len(out) else 0.0})
    Hs = pd.DataFrame(hrows)
    Hs.to_csv(MET / "H_load_steps.csv", index=False)
    tables.append(("H — DC-bus response to load steps (settling to ±1 V of 400 V)", Hs))
    fig, ax = plt.subplots(figsize=(8, 3))
    for var, lab, c in [("H|apo", "with bus-current feed-forward", COLORS["po"]),
                        ("H|apo|noff", "PI only", COLORS["inc"])]:
        dfh = R[var][0]
        ax.plot(dfh.t, dfh.v_dc, color=c, label=lab)
    ax.set(xlabel="Time (s)", ylabel="DC bus (V)", title="H: bus voltage, load steps 1→6→0.5→3 kW at 1, 2, 3 s")
    ax.legend(fontsize=7)
    savefig(fig, "H_bus_feedforward_ablation")

    # ---------- I: Monte Carlo robustness on profile B
    rows = []
    for k in BASE:
        for sd in range(n_mc):
            df, info, _ = R[f"I|{k}|{sd}"]
            rows.append({"method": LABELS[k], "seed": sd, "eta_mppt": M.tracking_efficiency(df, 0.2, 4.0),
                         "ts_300_s": M.settling_time(df, 2.0, 3.0), "ts_800_s": M.settling_time(df, 3.0, 4.0),
                         **M.bus_metrics(df, t0=0.2)})
    I = pd.DataFrame(rows)
    I.to_csv(MET / "I_monte_carlo.csv", index=False)
    Ig = I.groupby("method", sort=False).agg(eta_mean=("eta_mppt", "mean"), eta_std=("eta_mppt", "std"),
                                             eta_min=("eta_mppt", "min"), vdc_rms_mean=("vdc_rms_err_V", "mean"),
                                             ts_800_median=("ts_800_s", "median")).reset_index()
    tables.append((f"I — {n_mc} Monte-Carlo plants (L, Cin, C_dc ±20 %, R0 ±30 %, Q ±10 %, PV Rs/Rsh ±10 %) "
                   "+ 0.5 % FS Gaussian measurement noise, profile B", Ig))

    # ---------- J: comparison
    J = pd.DataFrame({"method": [LABELS[k] for k in BASE],
                      "A_eta": A.eta_mppt.values, "A_startup_reach_s": A.startup_reach_s.values,
                      "A_vpv_pp_V": A.vpv_pp_V.values, "B_eta": B.eta_mppt.values,
                      "D_eta": D.eta_mppt.values, "E_eta": E.eta_mppt.values[:4],
                      "I_eta_mean": Ig.eta_mean.values})
    J.to_csv(MET / "J_comparison.csv", index=False)
    tables.insert(0, ("J — identical-condition comparison (ideal converters)", J))

    summary["total_wall_s"] = time.perf_counter() - t_start
    summary["n_simulations"] = len(specs)
    summary["sim_seconds_total"] = float(sum(i["sim_time_s"] for _, i, _ in R.values()))
    (MET / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    md = ["# SolarCon results (auto-generated by scripts/run_experiments.py)\n",
          f"{len(specs)} simulations, {summary['sim_seconds_total']:.1f} s simulated, "
          f"{summary['total_wall_s']:.0f} s wall time.\n",
          "## PV model validation (single module)\n", fmt_table(val), "\n",
          "## Boost converter, ideal vs modeled losses (A, Adaptive dP-P&O)\n",
          fmt_table(pd.DataFrame([lossyA])), "\n"]
    for title, t in tables:
        md += [f"## {title}\n", fmt_table(t), "\n"]
    (MET / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
