"""Exact frontier at 12 requests. The paper reports CAP = 12, i.e. no cap on trip size (exact); smaller caps were tried and rejected (optimal plans at 10 requests use trips of up to 7 requests). With CAP < n the optional check compares the capped frontier (max_size = CAP) with the uncapped one
in all VAL draws per station (k = 1..4). Then 12 requests, 30 draws per station: E_min, T_min(k=1), T_min(k=4), premium at k=4.
Usage (from 03_Code_Execution): CAP=12 NW=3 python -I ../04_Results/run_n12_cap.py <draws12> <draws_val10>   Output: n12_cap_rows.csv, n10_cap_validation.csv, N12_CAP_SUMMARY_D2.txt"""
import sys, os, csv, time, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0); CAP = int(os.environ.get("CAP", "12"))
ARMS = [("LA_base", "DLA4"), ("CHI_base", "DCH1"), ("AUS_base", "DAU1")]
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}


def build(name, st_, n, seed):
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, n, seed, 3000.0, 100.0, HF[name]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    return leg, [Req(*r) for r in reqs]


def val(args):
    (name, st_), seed = args
    leg, R = build(name, st_, 10, seed)
    t0 = time.time(); tt = trip_table(leg, R, TYPE_M600, TM); full = one_drone_table(tt, 10, TM); t_full = time.time() - t0
    t0 = time.time(); tc = trip_table(leg, R, TYPE_M600, TM, max_size=CAP); cap = one_drone_table(tc, 10, TM); t_cap = time.time() - t0
    same = True; maxd = 0.0
    for k in (1, 2, 3, 4):
        A = [(round(e, 5), round(t, 3)) for e, t, _ in fleet_frontier([full] * k, 10)]; B = [(round(e, 5), round(t, 3)) for e, t, _ in fleet_frontier([cap] * k, 10)]
        if A != B: same = False
    return dict(arm=name, seed=seed, identical=int(same), sec_full=round(t_full, 1), sec_cap=round(t_cap, 1))


def run12(args):
    (name, st_), seed = args
    leg, R = build(name, st_, 12, seed)
    t0 = time.time(); h = one_drone_table(trip_table(leg, R, TYPE_M600, TM, max_size=CAP), 12, TM); t1 = time.time() - t0
    F1 = fleet_frontier([h], 12); F4 = fleet_frontier([h] * 4, 12)
    emin = F4[0][0]; tmin4 = F4[-1][1]; tmin1 = F1[-1][1]; e_t = F4[-1][0]
    return dict(arm=name, seed=seed, E_min=round(emin, 3), Tmin_k1=round(tmin1, 1), Tmin_k4=round(tmin4, 1), ratio=round(tmin1 / tmin4, 3), premium=round(e_t / emin - 1, 5), npts_k4=len(F4), sec=round(t1, 1))


if __name__ == "__main__":
    n12 = int(sys.argv[1]); nv = int(sys.argv[2]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p:
        V = p.map(val, [(a, 1001 + i) for a in ARMS for i in range(nv)], chunksize=1)
        R12 = p.map(run12, [(a, 1001 + i) for a in ARMS for i in range(n12)], chunksize=1)
    for fn, rows in (("n10_cap_validation.csv", V), ("n12_cap_rows.csv", R12)):
        with open(os.path.join(HERE, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = [f"cap = {CAP} requests per trip. Validation at 10 requests: identical frontiers (k=1..4) in {sum(r['identical'] for r in V)} of {len(V)} draws; median seconds full {st.median(r['sec_full'] for r in V):.1f}, capped {st.median(r['sec_cap'] for r in V):.1f}"]
    for a, _ in ARMS:
        g = [r for r in R12 if r["arm"] == a]
        out.append(f"{a} n=12: draws {len(g)}; E_min median {st.median(r['E_min'] for r in g):.1f}; Tmin k=1 {st.median(r['Tmin_k1'] for r in g):.0f} -> k=4 {st.median(r['Tmin_k4'] for r in g):.0f} (ratio median {st.median(r['ratio'] for r in g):.2f}); premium median {100*st.median(r['premium'] for r in g):.1f} % max {100*max(r['premium'] for r in g):.1f} %; sec median {st.median(r['sec'] for r in g):.0f}")
    open(os.path.join(HERE, "N12_CAP_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
