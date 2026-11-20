"""30-run campaign on the 240 n=12 cells (bridge_real_n12.csv). Methods: SA_full, SA_plain (no unl/p1/hard), libABC (random keys), gABC (guided ABC seed+repair).
Run seeds 1001-1030, equal budget 80000 evaluations (SA 4x20000). T = stored T + 0.05. Resumable: one CSV per (arm, instance seed) in campaign30/.
Usage (from 03_Code_Execution): NW=2 python -X utf8 ../04_Results/run_campaign30_n12.py     Output: campaign30/*.csv"""
import sys, os, csv, time, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_abl import anneal_multi_abl
from drone3d_d2.heur_keys import FleetProblem, ALGORITHMS, PENALTY
from drone3d_d2.heur_guided import guided_abc
TM = TimeModel(s_service=60.0, tau=180.0)
N = 12; BUD = 80000
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
OUT = os.path.join(HERE, "campaign30")
RUNS = range(1001, 1031)


def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    (arm, seed), cells = job
    fn = os.path.join(OUT, f"{arm}_{seed}.csv")
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
            for nm, unl in (("SA_full", True), ("SA_plain", False)):
                r = anneal_multi_abl(leg, R, TYPE_M600, TM, k, T, unl, unl, unl, restarts=3, iters=20000, seed0=rs)
                row[nm] = round(r[0], 3) if r else ""
            res = ALGORITHMS["ABC"](FleetProblem(leg, R, TYPE_M600, TM, k, T), BUD, seed=rs, N=30)
            row["libABC"] = round(res.best, 3) if res.best < PENALTY else ""
            g = guided_abc(leg, R, TYPE_M600, TM, k, T, BUD, rs, True, True)
            row["gABC"] = round(g, 3) if g is not None else ""
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
        g.setdefault((r["arm"], int(r["seed"])), []).append(r)
    jobs = list(g.items())
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "2"))) as p:
        for fn in p.imap_unordered(one, jobs):
            print("done", fn, round(time.time() - t0), flush=True)
    open(os.path.join(OUT, "ALL_DONE.txt"), "w").write("ok")
