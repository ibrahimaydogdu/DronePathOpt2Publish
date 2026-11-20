"""Fixed energy per dispatched drone (c Wh) and battery inventory (b trips per drone) on the REAL-demand instances (LA_base, CHI_base; n = 8).
c and b are UNSOURCED sensitivity parameters. Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_realinventory.py <draws>
Output: realinventory_rows.csv (arm, seed, b, c, k, E_slack, T_Emin, Tmin, E_at_Tmin, u_Emin)."""
import sys, os, csv, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0)
ARMS = [("LA_base", "DLA4", None, None), ("CHI_base", "DCH1", 3.5, 0), ("AUS_base", "DAU1", "aus", 0.0)]
BS, CS, KS, N = (1, 2, None), (0.0, 15.0, 45.0), (1, 2, 3, 4), 8
_R = {}

def one(args):
    arm, seed = args
    name, st, ms, fill = arm
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(st, N, seed, 3000.0, 100.0, height_la if ms is None else (height_austin(fill) if ms == 'aus' else height_chicago(ms, fill)))
    if name not in _R: _R[name] = CityRaster(city, res=10.0)
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=_R[name]), TYPE_M600)
    R = [Req(*r) for r in reqs]; trips = trip_table(leg, R, TYPE_M600, TM); rows = []
    for b in BS:
        h = one_drone_table(trips, N, TM, b)
        for c in CS:
            for k in KS:
                f = fleet_frontier([h] * k, N, c); base = dict(arm=name, seed=seed, b=("inf" if b is None else b), c=c, k=k)
                if not f: rows.append(dict(base, E_slack="", T_Emin="", Tmin="", E_at_Tmin="", u_Emin="")); continue
                rows.append(dict(base, E_slack=f[0][0], T_Emin=f[0][1], Tmin=f[-1][1], E_at_Tmin=f[-1][0], u_Emin=sum(1 for p in f[0][2] if p)))
    return rows

if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p: res = p.map(one, [(a, 1001 + i) for a in ARMS if a[0].startswith(os.environ.get('SEL', '')) for i in range(draws)], chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, os.environ.get("OUT", "realinventory_rows.csv")), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))
