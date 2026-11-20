"""RQ3 Phase A synthetic experiment: mixed fleet M600 Pro + FlyCart 30 (SENTETIK instances, labelled).
Heavy parcels (mass U[6,25] kg, exceed M600 capacity 6 kg => FlyCart only) vs light (U[0.5,3] kg).
Run from 03_Code_Execution: python ../04_Results/run_mixed.py [n] [draws] [span]
"""
import sys, os, csv, random, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import DroneType, TimeModel, M600, FLYCART, FLYCART_P09
from drone3d_d2.legs import PlanarLeg

TGRID = (900, 1200, 1800, 2700, 3600)
FLEETS = [(a, b) for a in range(0, 5) for b in range(0, 4) if 1 <= a + b <= 5]


def make_inst(n, seed, rho, span):
    r = random.Random(seed); pts = [(0.0, 0.0, 20.0)]; reqs = []
    nheavy = round(rho * n)
    heavy_idx = set(r.sample(range(n), nheavy))
    for i in range(n):
        k = r.choice("DPB")
        q = round(r.uniform(6.0, 25.0), 1) if i in heavy_idx else round(r.uniform(0.5, 3.0), 1)
        pt = lambda: (r.uniform(-span, span), r.uniform(-span, span), r.uniform(15, 120))
        if k == "B":
            pts.append(pt()); pts.append(pt()); reqs.append(Req("B", len(pts) - 2, len(pts) - 1, q))
        elif k == "D":
            pts.append(pt()); reqs.append(Req("D", 0, len(pts) - 1, q))
        else:
            pts.append(pt()); reqs.append(Req("P", len(pts) - 1, 0, q))
    return pts, reqs


def run_one(args):
    n, seed, span = args
    rows = []
    tm = TimeModel()
    for rho in (0.0, 0.125, 0.25, 0.5):
        pts, reqs = make_inst(n, seed, rho, span)
        for vfc, variant in ((10.0, "base"), (15.0, "base"), (10.0, "p09")):
            tM = DroneType("M600", M600, v=10.0)
            tF = DroneType("FlyCart30", FLYCART if variant == "base" else FLYCART_P09, v=vfc)
            hM = one_drone_table(trip_table(PlanarLeg(pts, tM), reqs, tM, tm), n, tm)
            hF = one_drone_table(trip_table(PlanarLeg(pts, tF), reqs, tF, tm), n, tm)
            for (a, b) in FLEETS:
                F = fleet_frontier([hM] * a + [hF] * b, n)
                if not F:
                    rows.append(dict(seed=seed, rho=rho, vfc=vfc, var=variant, a=a, b=b, T="", E_at_T="", E_min="", Tmin="")); continue
                Emin, Tmin = F[0][0], F[-1][1]
                for T in TGRID:
                    best = None
                    for e, t, _ in F:
                        if t <= T + 1e-9 and (best is None or e < best):
                            best = e
                    rows.append(dict(seed=seed, rho=rho, vfc=vfc, var=variant, a=a, b=b, T=T, E_at_T=("" if best is None else round(best, 3)),
                                     E_min=round(Emin, 3), Tmin=round(Tmin, 1)))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "3"))) as p:
        out = p.map(run_one, [(n, 1000 + i, span) for i in range(1, draws + 1)])
    rows = [r for o in out for r in o]
    fn = os.path.join(HERE, f"mixed_rows_n{n}_span{int(span)}.csv")
    with open(fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done", len(rows), "%.0f s" % (time.time() - t0))
