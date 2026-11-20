"""Real-geometry row (Manhattan, NYC Open Data footprints; demand SYNTHETIC, PLUTO-weighted): building-aware vs building-blind
straight-line legs under the same fleet model. Run from 03_Code_Execution: python ../04_Results/run_city.py [n] [draws]
"""
import sys, os, csv, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "03_Code_Execution"))
import drone3d_d2._core  # noqa: F401  (puts core_d1 on sys.path)
from drone3d_d2.fleet import one_drone_table, trip_table, fleet_frontier
from drone3d_d2.dtypes import DroneType, TimeModel, M600
from drone3d_d2.legs import CityLeg, PlanarLeg

DATA = os.path.join(ROOT, "Evidence", "data")
TGRID = (900, 1200, 1800, 2700, 3600)
_G = {}


def env():
    if "city" not in _G:
        from around import CityRaster
        from geometry import load_city
        from instances import load_pluto
        _G["city"] = load_city(os.path.join(DATA, "NYC_Building_Footprints_raw.json"))
        _G["pluto"] = load_pluto(os.path.join(DATA, "NYC_MapPLUTO_raw.json"))
        _G["ras"] = CityRaster(_G["city"], res=5.0)
    return _G["city"], _G["pluto"], _G["ras"]


def run_one(args):
    n, seed = args
    from instances import DEPOT_A, make_instance
    from legs import LegCost
    city, pluto, ras = env()
    inst = make_instance(city, pluto, DEPOT_A, n, seed)
    from drone3d_d2.fleet import Req
    reqs = [Req(r.kind, r.o, r.d, r.q) for r in inst.requests]
    dt = DroneType("M600", M600, v=10.0)
    tm = TimeModel()
    rows = []
    legs = {
        "aware": CityLeg(LegCost(city, inst.nodes, model="F", hw=M600, v=10.0, raster=ras), dt),
        "blind": PlanarLeg([(nd.x, nd.y, nd.z) for nd in inst.nodes], dt),
    }
    for name, leg in legs.items():
        h = one_drone_table(trip_table(leg, reqs, dt, tm), n, tm)
        for k in (1, 2, 3):
            F = fleet_frontier([h] * k, n)
            if not F:
                rows.append(dict(seed=seed, leg=name, k=k, T="", E_at_T="", E_min="", Tmin="")); continue
            Emin, Tmin = F[0][0], F[-1][1]
            for T in TGRID:
                best = None
                for e, t, _ in F:
                    if t <= T + 1e-9 and (best is None or e < best):
                        best = e
                rows.append(dict(seed=seed, leg=name, k=k, T=T, E_at_T=("" if best is None else round(best, 3)),
                                 E_min=round(Emin, 3), Tmin=round(Tmin, 1)))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "3"))) as p:
        out = p.map(run_one, [(n, 1000 + i) for i in range(1, draws + 1)], chunksize=1)
    rows = [r for o in out for r in o]
    fn = os.path.join(HERE, f"city_rows_n{n}.csv")
    with open(fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done", len(rows), "%.0f s" % (time.time() - t0))
