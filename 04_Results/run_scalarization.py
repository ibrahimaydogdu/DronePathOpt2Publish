"""How many exact (E, makespan) Pareto points can a weighted-sum scalarisation (the objective form of Cheng et al. 2020 and Troudi et al. 2018,
which weight cost terms) ever return? Points off the lower-left convex hull are 'unsupported': no weight vector makes them optimal.
Real-demand base arms (LA_base, CHI_base, AUS_base), n = 8, k = 1..4, 30 draws. Output: scalarization_rows.csv, SCALARIZATION_SUMMARY_D2.txt.
Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_scalarization.py 30"""
import sys, os, csv, time, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0)
ARMS = [("LA_base", "DLA4"), ("CHI_base", "DCH1"), ("AUS_base", "DAU1")]
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}


def hull_supported(F):
    """F sorted by E ascending (T descending). Count points on the lower-left convex hull (supported by some weight w>=0 in min wE+(1-w)T)."""
    P = sorted((e, t) for e, t, _ in F)
    h = []
    for p in P:
        while len(h) >= 2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= 0:
            h.pop()
        h.append(p)
    return len(h)


def one(args):
    (name, st_), seed = args; hf = HF[name]()
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, 8, seed, 3000.0, 100.0, hf)
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]
    h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), 8, TM)
    out = []
    for k in (1, 2, 3, 4):
        F = fleet_frontier([h] * k, 8)
        out.append(dict(arm=name, seed=seed, k=k, npts=len(F), supported=hull_supported(F) if F else 0))
    return out


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p:
        res = p.map(one, [(a, 1001 + i) for a in ARMS for i in range(draws)], chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, "scalarization_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = []
    for a, _ in ARMS:
        for k in (1, 2, 3, 4):
            g = [r for r in rows if r["arm"] == a and r["k"] == k and r["npts"] > 0]
            tot = sum(r["npts"] for r in g); sup = sum(r["supported"] for r in g)
            share = [1 - r["supported"] / r["npts"] for r in g]
            out.append(f"{a} k={k}: draws={len(g)} Pareto points total {tot} (median {st.median([r['npts'] for r in g]):.0f} per draw); weighted-sum can return {sup} ({100*sup/tot:.1f} %); unsupported share per draw median {100*st.median(share):.1f} %, max {100*max(share):.1f} %; draws with >=1 unsupported {sum(1 for r in g if r['supported'] < r['npts'])}/{len(g)}")
    open(os.path.join(HERE, "SCALARIZATION_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))
