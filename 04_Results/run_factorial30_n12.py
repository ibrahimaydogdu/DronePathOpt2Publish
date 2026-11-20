"""2x2 factorial + missing opponents, 30 runs (seeds 1001-1030), subset: instances 1001-1003 of each arm (72 cells), n=12, budget 80000, T = stored T + 0.05.
Methods: SAkeys (SA on random keys + DP decoder), TLBO, GWO (library, keys), ownABC, ownGA (native trip-list encoding; heur_pop.py). Reference SA_full/libABC/gABC from campaign30.
Usage (from 03_Code_Execution): NW=6 python -X utf8 ../04_Results/run_factorial30_n12.py   Output: factorial30/*.csv, ALL_DONE.txt"""
import sys, os, csv, time, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_keys import FleetProblem, ALGORITHMS, PENALTY
from drone3d_d2.heur_sakeys import sa_keys
from drone3d_d2.heur_pop import abc as own_abc, ga as own_ga
TM = TimeModel(s_service=60.0, tau=180.0)
N = 12; BUD = 80000
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
OUT = os.path.join(HERE, "factorial30")
RUNS = range(1001, 1031)


def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    (arm, seed, kk), cells = job
    fn = os.path.join(OUT, f"{arm}_{seed}_k{kk}.csv")
    if os.path.exists(fn):
        return fn
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(ST[arm], N, seed, 3000.0, 100.0, HF[arm]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; out = []
    for c in cells:
        k = int(c["k"]); T = float(c["T"]) + 0.05
        for rs in RUNS:
            row = dict(arm=arm, inst=seed, k=k, theta=c["theta"], run=rs, E_exact=c["E_exact"])
            r = sa_keys(FleetProblem(leg, R, TYPE_M600, TM, k, T), BUD, seed=rs)
            row["SAkeys"] = round(r.best, 3) if r.best < PENALTY else ""
            for nm in ("TLBO", "GWO"):
                r = ALGORITHMS[nm](FleetProblem(leg, R, TYPE_M600, TM, k, T), BUD, seed=rs, N=30)
                row[nm] = round(r.best, 3) if r.best < PENALTY else ""
            for nm, f in (("ownABC", own_abc), ("ownGA", own_ga)):
                r = f(leg, R, TYPE_M600, TM, k, T, budget=BUD, seed=rs)
                row[nm] = round(r[0], 3) if r else ""
            out.append(row)
    with open(fn + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    os.replace(fn + ".tmp", fn)
    return fn


if __name__ == "__main__":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "bridge_real_n12.csv"), encoding="utf-8")))
    g = {}
    for r in rows:
        if int(r["seed"]) in (1001, 1002, 1003):
            g.setdefault((r["arm"], int(r["seed"]), int(r["k"])), []).append(r)
    jobs = list(g.items())
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "6"))) as p:
        for fn in p.imap_unordered(one, jobs):
            print("done", fn, round(time.time() - t0), flush=True)
    open(os.path.join(OUT, "ALL_DONE.txt"), "w").write("ok")
