"""Large-n pilot WITHOUT exact reference (n = NREQ, default 20). Instances 1001-1002 of each arm, k in (2,4), theta in (0.25, 0.5, 1.0).
T = Tmin_h + theta (T_E - Tmin_h): Tmin_h = best makespan found by an unconstrained makespan-minimising annealing (heuristic, so T may be slightly loose),
T_E = makespan of the energy-minimising unlimited-fleet annealing solution. All methods use the same T. Reference for gaps = best value found by any method/run in the cell ('best known', not a bound).
Methods: SA_full (heur.anneal_multi), ALNS (heur_alns), ownABC (heur_pop, native), libABC (heur_keys, random keys). Budget 80000 evaluations, runs seeds 1001..1000+RUNS.
Usage (from 03_Code_Execution): NREQ=20 RUNS=10 NW=6 python -X utf8 ../04_Results/run_large_n.py    Output: large_n<N>/*.csv, ALL_DONE.txt"""
import sys, os, csv, time, math, random, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur import anneal, anneal_multi, _Eval, _cost, _neighbor, initial
from drone3d_d2.heur_alns import alns
from drone3d_d2.heur_pop import abc as own_abc
from drone3d_d2.heur_keys import FleetProblem, ALGORITHMS, PENALTY
TM = TimeModel(s_service=60.0, tau=180.0)
N = int(os.environ.get("NREQ", "20")); BUD = 80000; RUNS = int(os.environ.get("RUNS", "10"))
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
OUT = os.path.join(HERE, f"large_n{N}")


def min_makespan(ev_, reqs, n, k, iters=40000, seed=7):
    rng = random.Random(seed)
    cur = initial(ev_, n, k); cc = _cost(ev_, cur, 0.0)
    best = cc[2]; temp = max(1.0, 0.02 * cc[2]); alpha = (1e-3 / temp) ** (1.0 / iters)
    for _ in range(iters):
        s = _neighbor(cur, rng, reqs, n, k)
        if s is None: continue
        c = _cost(ev_, s, 0.0)
        if c is None: continue
        f1, f0 = c[2] + 1e-6 * c[1], cc[2] + 1e-6 * cc[1]
        if f1 <= f0 or rng.random() < math.exp(-(f1 - f0) / temp):
            cur, cc = s, c; best = min(best, c[2])
        temp = max(temp * alpha, 1e-3)
    return best


def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    arm, seed, k = job
    fn = os.path.join(OUT, f"{arm}_{seed}_k{k}.csv")
    if os.path.exists(fn): return fn
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(ST[arm], N, seed, 3000.0, 100.0, HF[arm]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]
    ev_ = _Eval(leg, R, TYPE_M600, TM)
    tmin = min_makespan(ev_, R, N, k)
    unl = anneal(leg, R, TYPE_M600, TM, N, 1e9, iters=40000, seed=seed, evaluator=ev_)
    tE = unl[1] if unl else tmin
    out = []
    for th in (0.25, 0.5, 1.0):
        T = tmin + th * max(0.0, tE - tmin) + 0.05
        for rs in range(1001, 1001 + RUNS):
            row = dict(arm=arm, inst=seed, n=N, k=k, theta=th, T=round(T, 1), run=rs)
            t0 = time.time(); r = anneal_multi(leg, R, TYPE_M600, TM, k, T, restarts=3, iters=20000, seed0=rs)
            row["SA_full"] = round(r[0], 3) if r else ""; row["sec_SA"] = round(time.time() - t0, 1)
            r = alns(leg, R, TYPE_M600, TM, k, T, BUD, rs); row["ALNS"] = round(r[0], 3) if r else ""
            r = own_abc(leg, R, TYPE_M600, TM, k, T, budget=BUD, seed=rs); row["ownABC"] = round(r[0], 3) if r else ""
            r = ALGORITHMS["ABC"](FleetProblem(leg, R, TYPE_M600, TM, k, T), BUD, seed=rs, N=30)
            row["libABC"] = round(r.best, 3) if r.best < PENALTY else ""
            out.append(row)
    with open(fn + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    os.replace(fn + ".tmp", fn); return fn


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(a, s, k) for a in ST for s in (1001, 1002) for k in (2, 4)]
    with Pool(int(os.environ.get("NW", "6"))) as p:
        for fn in p.imap_unordered(one, jobs): print("done", fn, flush=True)
    open(os.path.join(OUT, "ALL_DONE.txt"), "w").write("ok")
