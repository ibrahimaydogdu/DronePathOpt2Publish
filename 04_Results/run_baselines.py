"""Published-assumption baselines on the real-demand instances (LA_base, CHI_base; n = 8, k = 4, M600).
Plans are optimised EXACTLY under a simplified objective that earlier work uses, then re-evaluated under the full model:
  DIST   : minimise total travelled distance (Troudi 2018; Cheng 2020 distance term) with the obstacle-aware shortest option;
           distance is scaled to Wh at the cruise energy per metre of the M600 at 1 kg (only to make the battery limit meaningful).
  LOADBL : payload-blind energy (every leg costed as unloaded; the linear / load-independent approximation criticised by Cheng 2020).
For makespan bound T = Tmin + theta (T_Emin - Tmin) (theta in 0, .25, .5, 1; Tmin, T_Emin from the TRUE k = 4 frontier), the baseline
takes its own minimum-objective plan with makespan <= T; regret = E_true(plan) / E_true_opt(T) - 1 (None if a trip violates the battery limit
under the true model). Output: baselines_rows.csv.  Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_baselines.py <draws>
"""
import sys, os, csv, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier, min_energy_at
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.analysis import eval_trip_order
from drone3d_d2._core import segment_energy_wh
import math

TM = TimeModel(s_service=60.0, tau=180.0)
TM_PROXY = TimeModel(s_service=60.0, tau=180.0, service_hover=False)
N, RADIUS, K = 8, 3000.0, 4
THETAS = (0.0, 0.25, 0.5, 1.0)
ARMS = [("LA_base", "DLA4", 100.0, None, None), ("CHI_base", "DCH1", 100.0, 3.5, 0), ("AUS_base", "DAU1", 100.0, "aus", 0.0)]
_R = {}
WH_PER_M = segment_energy_wh("F", 1000.0, 0.0, 1.0, TYPE_M600.hw, TYPE_M600.v, TYPE_M600.kappa()) / 1000.0


class LoadBlind:
    def __init__(self, leg): self.leg = leg
    def et(self, i, j, load): return self.leg.et(i, j, 0.0)


class DistLeg:
    def __init__(self, leg): self.leg, self._c = leg, {}
    def et(self, i, j, load):
        if i == j: return 0.0, 0.0
        r = self._c.get((i, j))
        if r is None:
            best = None
            for _, segs in self.leg.lc.options(i, j):
                d = sum(math.hypot(ell, dz) for ell, dz in segs)
                t = sum(max(ell / TYPE_M600.v, max(dz, 0) / TYPE_M600.hw.v_up, max(-dz, 0) / TYPE_M600.hw.v_down) for ell, dz in segs)
                if best is None or d < best[0]: best = (d, t)
            r = (best[0] * WH_PER_M, best[1]); self._c[(i, j)] = r
        return r


def true_energy(leg, R, assignment):
    tot = 0.0
    for plan in assignment:
        for (sub, _, _, order) in plan:
            S = tuple(i for i in range(len(R)) if sub >> i & 1)
            r = eval_trip_order(leg, R, TYPE_M600, TM, S, list(order))
            if r is None: return None
            tot += r[0]
    return tot


def one(args):
    arm, seed = args
    name, st, rho, ms, fill = arm
    from legs import LegCost
    from around import CityRaster
    hfn = height_la if ms is None else (height_austin(fill) if ms == 'aus' else height_chicago(ms, fill))
    city, nodes, reqs, meta = make_real_instance(st, N, seed, RADIUS, rho, hfn)
    if name not in _R: _R[name] = CityRaster(city, res=10.0)
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=_R[name]), TYPE_M600)
    R = [Req(*r) for r in reqs]
    F = fleet_frontier([one_drone_table(trip_table(leg, R, TYPE_M600, TM), N, TM)] * K, N)
    tmin, temin = F[-1][1], F[0][1]
    out = []
    for bname, bleg, btm in (("DIST", DistLeg(leg), TM_PROXY), ("LOADBL", LoadBlind(leg), TM)):
        FB = fleet_frontier([one_drone_table(trip_table(bleg, R, TYPE_M600, btm), N, btm)] * K, N)
        for th in THETAS:
            T = tmin + th * (temin - tmin)
            opt = min_energy_at(F, T)
            cand = [(e, t, a) for e, t, a in FB if t <= T + 1e-7]
            row = dict(arm=name, seed=seed, base=bname, theta=th, T=round(T, 1), E_opt=round(opt[0], 3))
            if not cand:
                row.update(E_true="", regret="", status="no_plan")
            else:
                e, t, a = min(cand, key=lambda x: x[0])
                et = true_energy(leg, R, a)
                row.update(E_true=("" if et is None else round(et, 3)), regret=("" if et is None else round(et / opt[0] - 1, 5)),
                           status="battery_infeasible" if et is None else "ok")
            out.append(row)
    return out


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    jobs = [(a, 1001 + i) for a in ARMS if a[0].startswith(os.environ.get('SEL', '')) for i in range(draws)]
    t0 = time.time()
    with Pool(nw) as p:
        res = p.map(one, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, os.environ.get("OUT", "baselines_rows.csv")), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("rows", len(rows), "sec", round(time.time() - t0))
