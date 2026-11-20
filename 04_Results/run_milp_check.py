"""Independent check of the exact fleet dynamic programme by a set-partitioning MILP (HiGHS via scipy.optimize.milp) on real instances.
Columns = (trip option, drone): a trip option is one (E, T, order) label of one request subset taken from the same trip table; the MILP covers every
request exactly once and keeps each drone's time sum(t_c + tau) <= T + tau (equivalent to sum t_c + tau (m-1) <= T for a used drone; an unused drone has 0).
It minimises energy for a bound T = Tmin + theta (T_Emin - Tmin) taken from the DP frontier. The MILP shares the trip table with the DP, so it checks the
combination stages (one-drone table, fleet merge, Pareto pruning), not the trip table. Output: milp_check_rows.csv, MILP_CHECK_SUMMARY_D2.txt.
Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_milp_check.py <draws>"""
import sys, os, csv, time, statistics as st
from multiprocessing import Pool
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier, min_energy_at
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0)
N = int(os.environ.get("NREQ", "8"))
ARMS = [("LA_base", "DLA4"), ("CHI_base", "DCH1"), ("AUS_base", "DAU1")]
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
KS = (2, 4); THETAS = (0.0, 0.25, 0.5)


def solve_milp(opts, k, T, n):
    """opts: list of (mask, E, t). Variables x[o, d]. Returns (E, seconds) or (None, seconds)."""
    t0 = time.time(); m = len(opts); nv = m * k
    c = np.array([e for (_, e, _) in opts for _ in range(k)])
    A = lil_matrix((n + k, nv)); lo = np.zeros(n + k); hi = np.zeros(n + k)
    for r in range(n):
        lo[r] = hi[r] = 1.0
    for j, (mask, e, t) in enumerate(opts):
        for d in range(k):
            v = j * k + d
            for r in range(n):
                if mask >> r & 1: A[r, v] = 1.0
            A[n + d, v] = t + TM.tau
    for d in range(k):
        lo[n + d] = -np.inf; hi[n + d] = T + TM.tau + 1e-6
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": 120})
    return (float(res.fun) if res.status == 0 else None), time.time() - t0, res.status


def one(args):
    (name, st_), seed = args
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, N, seed, 3000.0, 100.0, HF[name]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]
    t0 = time.time(); tt = trip_table(leg, R, TYPE_M600, TM); h = one_drone_table(tt, N, TM); t_trips = time.time() - t0
    opts = [(mask, e, t) for mask, L in tt.items() for (e, t, _) in L]
    rows = []
    for k in KS:
        t1 = time.time(); F = fleet_frontier([h] * k, N); t_dp = time.time() - t1
        tmin, temin = F[-1][1], F[0][1]
        for th in THETAS:
            T = tmin + th * (temin - tmin)
            ex = min_energy_at(F, T)
            e_m, sec, status = solve_milp(opts, k, T, N)
            rows.append(dict(arm=name, seed=seed, k=k, theta=th, T=round(T, 1), E_dp=round(ex[0], 4), E_milp=("" if e_m is None else round(e_m, 4)),
                             diff=("" if e_m is None else round(e_m - ex[0], 6)), status=status, n_cols=len(opts) * k, sec_milp=round(sec, 2), sec_dp_fleet=round(t_dp, 2), sec_trips=round(t_trips, 1)))
    return rows


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3); t0 = time.time()
    with Pool(nw) as p:
        res = p.map(one, [(a, 1001 + i) for a in ARMS for i in range(draws)], chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, f"milp_check_rows_n{N}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["E_milp"] != ""]
    bad = [r for r in ok if abs(float(r["diff"])) > 1e-4]
    out = [f"runs {len(rows)}; MILP solved {len(ok)}; status!=0 {len(rows)-len(ok)}; |E_milp - E_dp| > 1e-4 Wh in {len(bad)} runs; max |diff| {max(abs(float(r['diff'])) for r in ok):.2e} Wh"]
    for k in KS:
        g = [r for r in rows if r["k"] == k]
        out.append(f"k={k}: MILP median {st.median(float(r['sec_milp']) for r in g):.2f} s (max {max(float(r['sec_milp']) for r in g):.1f}), DP fleet merge median {st.median(float(r['sec_dp_fleet']) for r in g):.2f} s, columns median {st.median(r['n_cols'] for r in g):.0f}")
    out.append(f"trip table median {st.median(float(r['sec_trips']) for r in rows):.1f} s (shared by both)")
    open(os.path.join(HERE, f"MILP_CHECK_SUMMARY_n{N}_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out), "total sec", round(time.time() - t0))
