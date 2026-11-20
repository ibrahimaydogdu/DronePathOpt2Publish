"""Fixed energy per dispatched drone (c) and battery inventory per drone (b): effect on the structural properties.
SYNTHETIC demand, n = 8, span 1500 m, M600 Pro. c is an UNSOURCED sensitivity parameter (Wh per dispatched drone); b in {1, 2, inf}.
Usage (from 03_Code_Execution): python ../04_Results/run_inventory.py <draws>   (NW env var <= 3)
Output: inventory_rows.csv: seed, b, c, k, E_slack (min E over the whole frontier), T_Emin, Tmin, E_at_Tmin, u_Emin (drones used at E_slack)
"""
import sys, os, csv
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))
from drone3d_d2.fleet import trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import PlanarLeg
sys.path.insert(0, HERE)
from run_phaseA import make_inst

TM = TimeModel(s_service=60.0, tau=180.0)
BS = (1, 2, None)
CS = (0.0, 15.0, 45.0)
KS = (1, 2, 3, 4)
N = 8


def one(seed):
    pts, reqs = make_inst(N, seed)
    leg = PlanarLeg(pts, TYPE_M600)
    trips = trip_table(leg, reqs, TYPE_M600, TM)
    rows = []
    for b in BS:
        h = one_drone_table(trips, N, TM, b)
        for c in CS:
            for k in KS:
                f = fleet_frontier([h] * k, N, c)
                if not f:
                    rows.append(dict(seed=seed, b=("inf" if b is None else b), c=c, k=k, E_slack="", T_Emin="", Tmin="", E_at_Tmin="", u_Emin=""))
                    continue
                e0, t0, a0 = f[0]; e1, t1, _ = f[-1]
                u = sum(1 for plan in a0 if plan)
                rows.append(dict(seed=seed, b=("inf" if b is None else b), c=c, k=k, E_slack=e0, T_Emin=t0, Tmin=t1, E_at_Tmin=e1, u_Emin=u))
    return rows


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p:
        res = p.map(one, [1001 + i for i in range(draws)])
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, "inventory_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))
