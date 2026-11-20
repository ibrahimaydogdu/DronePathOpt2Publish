"""Scale bridge on REAL instances: gap of the simulated-annealing heuristic (heur.anneal_multi, same cost model) to the exact frontier.
Base arms LA_base / CHI_base / AUS_base, n = NREQ (8 or 10), k in (2, 4), makespan bound T = Tmin + theta (T_Emin - Tmin), theta in THETAS.
Usage (from 03_Code_Execution): NREQ=8 NW=3 python -I ../04_Results/run_bridge_real.py <draws> [iters]   Output: bridge_real_n<NREQ>.csv"""
import sys, os, csv, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier, min_energy_at
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur import anneal_multi
TM = TimeModel(s_service=60.0, tau=180.0)
N = int(os.environ.get("NREQ", "8"))
KS = (2, 4)
THETAS = (0.0, 0.25, 0.5, 1.0)
ARMS = [("LA_base", "DLA4"), ("CHI_base", "DCH1"), ("AUS_base", "DAU1")]
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}


def one(args):
    (name, st_), seed, iters = args
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, N, seed, 3000.0, 100.0, HF[name]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]
    h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), N, TM)
    rows = []
    for k in KS:
        F = fleet_frontier([h] * k, N)
        if not F:
            continue
        tmin, temin = F[-1][1], F[0][1]
        for th in THETAS:
            T = tmin + th * (temin - tmin)
            ex = min_energy_at(F, T)
            t0 = time.time()
            r = anneal_multi(leg, R, TYPE_M600, TM, k, T, restarts=3, iters=iters, seed0=seed)
            rows.append(dict(arm=name, n=N, seed=seed, k=k, theta=th, T=round(T, 1), E_exact=round(ex[0], 3),
                             E_heur=(round(r[0], 3) if r else ""), mk_heur=(round(r[1], 1) if r else ""), sec=round(time.time() - t0, 1)))
    return rows


if __name__ == "__main__":
    draws = int(sys.argv[1]); iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    nw = min(int(os.environ.get("NW", "1")), 3)
    t0 = time.time()
    with Pool(nw) as p:
        res = p.map(one, [(a, 1001 + i, iters) for a in ARMS for i in range(draws)], chunksize=1)
    rows = [r for rr in res for r in rr]
    out = os.path.join(HERE, os.environ.get("OUT", f"bridge_real_n{N}.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out, len(rows), "sec", round(time.time() - t0))
