"""Offline grid-search tuning of MPPT parameters.

Every method (P&O, IncCond, dP-P&O, Adaptive dP-P&O) is tuned with the SAME
grid-search procedure on the SAME three training episodes, none of which is
used for evaluation (ideal 400 V bus, PV side only):
  1. seeded random irradiance steps/ramps (src/profiles.training_profile),
  2. the same profile with 0.5 % full-scale measurement noise (seed 99),
  3. a cold start at 900 W/m^2, 60 degC (MPP far from the start reference).
Objective: mean energy tracking efficiency over the episodes, each evaluated
over [0, t_end] so start-up convergence counts.

Usage:  python optimization/tune_mppt.py        (writes tuned_params.json)
"""
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import tracking_efficiency  # noqa: E402
from src.profiles import Scenario, const, training_profile  # noqa: E402
from src.system import SystemConfig, simulate  # noqa: E402

GRID = {
    "po": [dict(step=s, period=p) for s, p in itertools.product([0.5, 1, 2, 4], [0.005, 0.01, 0.02])],
    "inc": [dict(step=s, period=p, eps=e) for s, p, e in
            itertools.product([0.5, 1, 2, 4], [0.005, 0.01, 0.02], [0.0, 0.02])],
    "dpo": [dict(step=s, period=p) for s, p in itertools.product([0.5, 1, 2, 4], [0.005, 0.01, 0.02])],
    "apo": [dict(M=m, dv_min=a, dv_max=b, period=p) for m, a, b, p in
            itertools.product([0.01, 0.03, 0.1, 0.3], [0.25, 0.5, 1.0], [4.0, 8.0], [0.005, 0.01])],
}


def episodes():
    tr = training_profile()
    cold = Scenario("cold", 1.0, const(900, 1.0), const(60, 1.0), const(0, 1.0), stiff_bus=True)
    return [(tr, SystemConfig(log_every=20)),
            (tr, SystemConfig(log_every=20, noise=0.005, seed=99)),
            (cold, SystemConfig(log_every=20))]


def _run(args):
    kind, params = args
    etas = []
    for sc, cfg in episodes():
        df, _ = simulate(sc, kind, params, cfg)
        etas.append(tracking_efficiency(df, 0.0, sc.t_end))
    return kind, params, sum(etas) / len(etas), etas


def main():
    jobs = [(k, p) for k, ps in GRID.items() for p in ps]
    with Pool() as pool:
        res = pool.map(_run, jobs)
    rows = [dict(method=k, eta=e, eta_dyn=x[0], eta_noise=x[1], eta_cold=x[2], **p) for k, p, e, x in res]
    out = pd.DataFrame(rows).sort_values(["method", "eta"], ascending=[True, False])
    out.to_csv(ROOT / "optimization" / "tuning_results.csv", index=False)
    best = {}
    for k in GRID:
        r = max((x for x in res if x[0] == k), key=lambda x: x[2])
        best[k] = r[1]
        print(f"{k:4s} best mean eta={r[2]:.5f} episodes={[round(x, 5) for x in r[3]]} params={r[1]}")
    best["apo_scan"] = dict(best["apo"])
    (ROOT / "optimization" / "tuned_params.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
