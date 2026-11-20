"""Calibration arm for the unsourced parcel density (Round 08-10 point): stop masses are drawn from a lognormal fitted to the delivery-mass quantiles reported by Amaral et al. (2026)
(median 0.695 kg, quartiles 0.300 and 1.625 kg: mu = ln 0.695, sigma = ln(1.625/0.300)/(2*0.6745) = 1.256; implied mean 1.53 kg against the reported 1.65 kg), truncated at the payload limit by resampling.
Locations are drawn as in the base arms (same seeds, but the pool is not reduced by the density rule). n = NREQ, k = 1..4, 30 draws per station.
Usage (from 03_Code_Execution): NREQ=8 NW=3 python -I ../04_Results/run_rho_calibrated.py 30   Output: rho_cal_n<NREQ>.csv, RHO_CAL_SUMMARY_n<NREQ>_D2.txt"""
import sys, os, csv, math, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0); N = int(os.environ.get("NREQ", "8"))
MU, SIG = math.log(0.695), math.log(1.625 / 0.300) / (2 * 0.6745)
ARMS = [("LA", "DLA4"), ("CHI", "DCH1"), ("AUS", "DAU1")]
HF = {"LA": lambda: height_la, "CHI": lambda: height_chicago(3.5, 0), "AUS": lambda: height_austin(0.0)}


def mass(rng): return math.exp(rng.gauss(MU, SIG))


def one(args):
    (name, st_), seed = args
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, N, seed, 3000.0, 100.0, HF[name](), mass_fn=mass)
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), N, TM)
    F1 = fleet_frontier([h], N); F4 = fleet_frontier([h] * 4, N)
    return dict(arm=name, seed=seed, mean_q=round(sum(r.q for r in R) / N, 3), E_min=round(F4[0][0], 3), Tmin_k1=round(F1[-1][1], 1), Tmin_k4=round(F4[-1][1], 1), premium=round(F4[-1][0] / F4[0][0] - 1, 5))


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p: res = p.map(one, [(a, 1001 + i) for a in ARMS for i in range(draws)], chunksize=1)
    with open(os.path.join(HERE, f"rho_cal_n{N}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    out = [f"Lognormal mass (median 0.695 kg, sigma {SIG:.3f}), n = {N}, 30 draws per station"]
    for a, _ in ARMS:
        g = [r for r in res if r["arm"] == a]
        out.append(f"{a}: mean stop mass {st.mean(r['mean_q'] for r in g):.2f} kg; E_min median {st.median(r['E_min'] for r in g):.1f} Wh; Tmin k=1 {st.median(r['Tmin_k1'] for r in g):.0f} -> k=4 {st.median(r['Tmin_k4'] for r in g):.0f} (ratio of medians {st.median(r['Tmin_k1'] for r in g)/st.median(r['Tmin_k4'] for r in g):.2f}); premium median {100*st.median(r['premium'] for r in g):.1f} % (max {100*max(r['premium'] for r in g):.1f})")
    open(os.path.join(HERE, f"RHO_CAL_SUMMARY_n{N}_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
