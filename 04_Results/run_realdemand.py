"""Real demand (Amazon Last Mile Routing Challenge, eval subset) + real building footprints (Chicago DCH1, Los Angeles DLA4).
Exact (energy, makespan) Pareto frontier for k = 1..4 identical M600 drones, n = 8 deliveries drawn from stops within 3 km of the station.
Sensitivity arms (UNSOURCED parameters): package density rho (kg/m3), metres per storey and unknown-storey fill (Chicago).
Usage (from 03_Code_Execution): python ../04_Results/run_realdemand.py <draws>   (NW env var <= 3). Output: realdemand_rows.csv
"""
import sys, os, csv, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg

TM = TimeModel(s_service=60.0, tau=180.0)
N, RADIUS = int(os.environ.get('NREQ', '8')), 3000.0
KS = (1, 2, 3, 4)
TGRID = (900, 1200, 1800, 2700, 3600)
ARMS = [  # name, station, rho, ms (metres per storey), fill (storeys assumed where unknown)
    ("LA_base", "DLA4", 100.0, None, None), ("LA_rho50", "DLA4", 50.0, None, None), ("LA_rho200", "DLA4", 200.0, None, None),
    ("CHI_base", "DCH1", 100.0, 3.5, 0), ("CHI_ms3", "DCH1", 100.0, 3.0, 0), ("CHI_ms4", "DCH1", 100.0, 4.0, 0),
    ("CHI_fill2", "DCH1", 100.0, 3.5, 2), ("CHI_rho50", "DCH1", 50.0, 3.5, 0), ("CHI_rho200", "DCH1", 200.0, 3.5, 0),
    ("AUS_base", "DAU1", 100.0, "aus", 0.0), ("AUS_fill6", "DAU1", 100.0, "aus", 6.4),   # Austin: max_height in feet; unknown (16 %) -> 0 or 6.4 m (median known)
    ("AUS_rho50", "DAU1", 50.0, "aus", 0.0), ("AUS_rho200", "DAU1", 200.0, "aus", 0.0),
    # buildings removed as obstacles (nodes keep their roof altitude): isolates the effect of the buildings themselves
    ("LA_nobld", "DLA4", 100.0, None, None, "nobld"), ("CHI_nobld", "DCH1", 100.0, 3.5, 0, "nobld"), ("AUS_nobld", "DAU1", 100.0, "aus", 0.0, "nobld"),
]
_R = {}


def one(args):
    arm, seed = args
    name, st, rho, ms, fill = arm[:5]
    from legs import LegCost
    from around import CityRaster
    hfn = height_la if ms is None else (height_austin(fill) if ms == 'aus' else height_chicago(ms, fill))
    city, nodes, reqs, meta = make_real_instance(st, N, seed, RADIUS, rho, hfn)
    if len(arm) > 5:
        import dataclasses, numpy as np
        city = dataclasses.replace(city, obstacle=np.zeros(len(city.obstacle), dtype=bool))
    rk = (name,)
    if rk not in _R:
        _R[rk] = CityRaster(city, res=10.0)
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=_R[rk]), TYPE_M600)
    R = [Req(*r) for r in reqs]
    h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), N, TM)
    rows = []
    for k in KS:
        F = fleet_frontier([h] * k, N)
        base = dict(arm=name, station=st, seed=seed, k=k, rho=rho, pool=meta["pool"], heavy=meta["heavy_excluded"],
                    unk_h=meta["unknown_height_hosts"], mean_q=round(meta["mean_q"], 3))
        if not F:
            rows.append(dict(base, E_min="", T_Emin="", Tmin="", E_Tmin="", npts=0, **{f"E_T{T}": "" for T in TGRID}))
            continue
        d = dict(base, E_min=round(F[0][0], 3), T_Emin=round(F[0][1], 1), Tmin=round(F[-1][1], 1), E_Tmin=round(F[-1][0], 3), npts=len(F))
        for T in TGRID:
            b = None
            for e, t, _ in F:
                if t <= T + 1e-9 and (b is None or e < b):
                    b = e
            d[f"E_T{T}"] = "" if b is None else round(b, 3)
        rows.append(d)
    return rows


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    sel = os.environ.get("ARMS", "")
    jobs = [(a, 1001 + i) for a in ARMS if not sel or a[0] in sel.split(',') or (',' not in sel and a[0].startswith(sel)) for i in range(draws)]
    t0 = time.time()
    with Pool(nw) as p:
        res = p.map(one, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, os.environ.get("OUT", "realdemand_rows.csv")), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("rows", len(rows), "sec", round(time.time() - t0))
