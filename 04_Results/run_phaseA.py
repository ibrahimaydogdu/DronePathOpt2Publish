"""Phase A synthetic experiment (SENTETIK instances; not real city data). Output: 04_Results/phaseA_rows.csv, phaseA_rules.csv.
Run from 03_Code_Execution: python ../04_Results/run_phaseA.py [n] [draws]
"""
import sys, os, csv, random, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier, pareto
from drone3d_d2.dtypes import DroneType, TimeModel, M600
from drone3d_d2.legs import PlanarLeg
from drone3d_d2.analysis import plan_components, rule_one_trip_per_request, rule_nearest_neighbour_bagging, k_lower_bound
from drone3d_d2._core import Hardware
import dataclasses

TGRID = (900, 1200, 1800, 2700, 3600)
KMAX = 4


def make_inst(n, seed, kinds="DPB", span=1500.0, depot=(0.0, 0.0)):
    r = random.Random(seed); pts = [(depot[0], depot[1], 20.0)]; reqs = []
    for _ in range(n):
        k = r.choice(kinds); q = round(r.uniform(0.5, 3.0), 1)
        pt = lambda: (r.uniform(-span, span), r.uniform(-span, span), r.uniform(15, 120))
        if k == "B":
            pts.append(pt()); pts.append(pt()); reqs.append(Req("B", len(pts) - 2, len(pts) - 1, q))
        elif k == "D":
            pts.append(pt()); reqs.append(Req("D", 0, len(pts) - 1, q))
        else:
            pts.append(pt()); reqs.append(Req("P", len(pts) - 1, 0, q))
    return pts, reqs


def hw_alpha(alpha):
    """M600 hardware whose hover-power exponent equals alpha (t1 chosen from t0 and mass ratio; P ~ 1/t)."""
    b = M600
    t1 = b.t0_min * (b.m0 / b.mtow) ** alpha
    return dataclasses.replace(b, t1_min=t1)


CONFIGS = []
for s in (15, 30, 60, 120, 240):
    for tau in (0, 60, 180, 360, 600):
        CONFIGS.append(dict(name=f"s{s}_tau{tau}", s=s, tau=tau, kappa=1.0, alpha=None, v=10.0, hover=True))
for kap in (0.8, 0.9, 1.1, 1.2):
    CONFIGS.append(dict(name=f"kappa{kap}", s=60, tau=180, kappa=kap, alpha=None, v=10.0, hover=True))
for al in (1.0, 1.2, 1.6):
    CONFIGS.append(dict(name=f"alpha{al}", s=60, tau=180, kappa=1.0, alpha=al, v=10.0, hover=True))
for v in (8.0, 15.0):
    CONFIGS.append(dict(name=f"v{int(v)}", s=60, tau=180, kappa=1.0, alpha=None, v=v, hover=True))
for v in (8.0, 10.0, 15.0):
    CONFIGS.append(dict(name=f"kv_v{int(v)}", s=60, tau=180, kappa=1.0, alpha=None, v=v, hover=True, ind=0.8))
CONFIGS.append(dict(name="land_nohover", s=60, tau=180, kappa=1.0, alpha=None, v=10.0, hover=False))
CONFIGS.append(dict(name="s0", s=0, tau=180, kappa=1.0, alpha=None, v=10.0, hover=False))


def run_one(args):
    n, seed, span = args
    pts, reqs = make_inst(n, seed, span=span)
    rows, rules = [], []
    cache = {}
    for c in CONFIGS:
        hw = M600 if c["alpha"] is None else hw_alpha(c["alpha"])
        dt = DroneType("M600", hw, v=c["v"], kappa_c=c["kappa"], ind_share=c.get("ind"))
        tm = TimeModel(s_service=float(c["s"]), tau=float(c["tau"]), service_hover=c["hover"])
        tkey = (c["s"], c["kappa"], c["alpha"], c["v"], c["hover"], c.get("ind"))
        leg = PlanarLeg(pts, dt)
        if tkey not in cache:
            cache[tkey] = trip_table(leg, reqs, dt, tm)
        trips = cache[tkey]
        h = one_drone_table(trips, n, tm)
        Emin = None
        for k in range(1, KMAX + 1):
            F = fleet_frontier([h] * k, n)
            if not F:
                rows.append(dict(seed=seed, cfg=c["name"], k=k, T="", E_min="", E_at_T="", Tmin="", ncross=0)); continue
            Emin = F[0][0]; Tmin = F[-1][1]
            for T in TGRID:
                best = None
                for e, t, _ in F:
                    if t <= T + 1e-9 and (best is None or e < best):
                        best = e
                rows.append(dict(seed=seed, cfg=c["name"], k=k, T=T, E_min=round(Emin, 4), E_at_T=("" if best is None else round(best, 4)),
                                 Tmin=round(Tmin, 2), ncross=len(F)))
            if c["name"] == "s60_tau180":
                # decomposition at both ends of the frontier, rules, lower bound
                for tag, idx in (("Emin_end", 0), ("Tmin_end", -1)):
                    e, t, a = F[idx]
                    ec, es, nt, used = plan_components(leg, reqs, dt, tm, a)
                    rules.append(dict(seed=seed, k=k, kind="decomp_" + tag, E=round(e, 3), T=round(t, 1), cruise=round(ec, 3),
                                      service=round(es, 3), trips=nt, drones=used, extra=""))
                Te, Tm = F[0][1], F[-1][1]
                for th in (0.0, 0.25, 0.5, 0.75, 1.0):
                    Tt = Tm + th * (Te - Tm)
                    bestE = min(e for e, t, _ in F if t <= Tt + 1e-9)
                    rules.append(dict(seed=seed, k=k, kind="theta", E=round(bestE, 4), T=round(Tt, 2), cruise="", service="", trips="", drones="", extra=th))
                for rname, rf in (("one_trip_per_request", rule_one_trip_per_request), ("nn_bagging", rule_nearest_neighbour_bagging)):
                    res = rf(leg, reqs, dt, tm, k)
                    if res is None:
                        continue
                    er, tr = res
                    bestE = None
                    for e, t, _ in F:
                        if t <= tr + 1e-9 and (bestE is None or e < bestE):
                            bestE = e
                    rules.append(dict(seed=seed, k=k, kind="rule_" + rname, E=round(er, 3), T=round(tr, 1), cruise="", service="", trips="",
                                      drones="", extra=("" if bestE is None else round(er / bestE - 1, 4))))
                for T in TGRID:
                    W, klb = k_lower_bound(trips, n, tm, T)
                    rules.append(dict(seed=seed, k=k, kind="klb", E=T, T=round(W, 1), cruise="", service="", trips="", drones="", extra=klb))
    return rows, rules


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
    tag = sys.argv[4] if len(sys.argv) > 4 else f"n{n}_span{int(span)}"
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "3"))) as p:
        out = p.map(run_one, [(n, 1000 + i, span) for i in range(1, draws + 1)])
    rows = [r for o in out for r in o[0]]; rules = [r for o in out for r in o[1]]
    for name, data in ((f"phaseA_rows_{tag}.csv", rows), (f"phaseA_rules_{tag}.csv", rules)):
        with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print("done", len(rows), len(rules), "%.0f s" % (time.time() - t0))
