"""Epsilon-thinned weighted-sum reachability: consecutive Pareto points closer than 30 s in makespan AND 1 % in energy
are merged (greedy from the minimum-energy end, so the kept points are pairwise distinguishable), then the hull count is recomputed.
Output: scalarization_eps_rows.csv, SCALARIZATION_EPS_SUMMARY_D2.txt.  Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_scalarization_eps.py 30"""
import sys, os, csv, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from run_scalarization import hull_supported, ARMS, HF, TM
from drone3d_d2.realcity import make_real_instance
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600
from drone3d_d2.legs import CityLeg
DT, DE = 30.0, 0.01


def thin(F):
    P = sorted((e, t, a) for e, t, a in F); out = [P[0]]
    for e, t, a in P[1:]:
        if abs(out[-1][1] - t) >= DT or abs(e - out[-1][0]) / out[-1][0] >= DE:
            out.append((e, t, a))
    return out


def one(args):
    (name, st_), seed = args
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, 8, seed, 3000.0, 100.0, HF[name]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), 8, TM); out = []
    for k in (2, 3, 4):
        F = fleet_frontier([h] * k, 8)
        if not F: continue
        T = thin(F)
        out.append(dict(arm=name, seed=seed, k=k, npts=len(F), supported=hull_supported(F), npts_thin=len(T), supported_thin=hull_supported(T)))
    return out


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p: res = p.map(one, [(a, 1001 + i) for a in ARMS for i in range(draws)], chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, "scalarization_eps_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = [f"Thinning: merge neighbours closer than {DT:.0f} s and {100*DE:.0f} % energy."]
    for a, _ in ARMS:
        for k in (2, 3, 4):
            g = [r for r in rows if r["arm"] == a and r["k"] == k]
            tot = sum(r["npts"] for r in g); tt = sum(r["npts_thin"] for r in g)
            sup = sum(r["supported"] for r in g); st_ = sum(r["supported_thin"] for r in g)
            sh = [1 - r["supported_thin"] / r["npts_thin"] for r in g]
            out.append(f"{a} k={k}: points {tot} -> {tt} after thinning; supported {sup} ({100*sup/tot:.1f} %) -> {st_} ({100*st_/tt:.1f} %); unsupported share per draw (thinned) median {100*st.median(sh):.1f} %, max {100*max(sh):.1f} %")
    open(os.path.join(HERE, "SCALARIZATION_EPS_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
