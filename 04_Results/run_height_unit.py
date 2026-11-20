"""Building-height unit sensitivity: Los Angeles (HEIGHT) and Austin (max_height) heights are read as feet in the paper (x 0.3048 m). If the unit were metres, every
height would be 3.28 times larger. Arms: LA and AUS with scale 3.28084 on the height function; base arms (scale 1) are read from realdemand_rows.csv. n = 8, k = 1..4, 30 draws.
Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_height_unit.py 30   Output: height_unit_rows.csv, HEIGHT_UNIT_SUMMARY_D2.txt"""
import sys, os, csv, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0); SC = 1.0 / 0.3048
ARMS = [("LA_base", "DLA4"), ("AUS_base", "DAU1")]


def scaled(f, tag):
    def g(props): return f(props) * SC
    g.tag = (tag, SC)
    return g


HF = {"LA_base": lambda: scaled(height_la, "la"), "AUS_base": lambda: scaled(height_austin(0.0), "aus")}


def one(args):
    (name, st_), seed = args
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st_, 8, seed, 3000.0, 100.0, HF[name]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), 8, TM)
    F1 = fleet_frontier([h], 8); F4 = fleet_frontier([h] * 4, 8)
    return dict(arm=name, seed=seed, E_min=round(F4[0][0], 3), Tmin_k1=round(F1[-1][1], 1), Tmin_k4=round(F4[-1][1], 1), premium=round(F4[-1][0] / F4[0][0] - 1, 5))


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p: res = p.map(one, [(a, 1001 + i) for a in ARMS for i in range(draws)], chunksize=1)
    with open(os.path.join(HERE, "height_unit_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    base = list(csv.DictReader(open(os.path.join(HERE, "realdemand_rows.csv"), encoding="utf-8")))
    out = ["Heights multiplied by 3.28 (unit metres instead of feet) versus the base arm (feet), same draws"]
    for a, _ in ARMS:
        dE, dT, dP = [], [], []
        for r in [x for x in res if x["arm"] == a]:
            b4 = [x for x in base if x["arm"] == a and x["seed"] == str(r["seed"]) and x["k"] == "4"][0]
            dE.append(100 * (r["E_min"] / float(b4["E_min"]) - 1)); dT.append(100 * (r["Tmin_k4"] / float(b4["Tmin"]) - 1))
            dP.append(100 * (r["premium"] - (float(b4["E_Tmin"]) / float(b4["E_min"]) - 1)))
        out.append(f"{a}: E_min change median {st.median(dE):+.2f} % (range {min(dE):+.2f} to {max(dE):+.2f}); Tmin(k=4) change median {st.median(dT):+.2f} % (range {min(dT):+.2f} to {max(dT):+.2f}); premium change median {st.median(dP):+.2f} points (range {min(dP):+.2f} to {max(dP):+.2f}); premium median {100*st.median(x['premium'] for x in res if x['arm']==a):.1f} %")
    open(os.path.join(HERE, "HEIGHT_UNIT_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
