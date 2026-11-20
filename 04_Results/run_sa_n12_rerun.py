"""Re-run only the cells of bridge_real_n12.csv where SA (20000 it) missed (no plan or gap > 0.1 %) with 100000 iterations. Reuses stored T and E_exact."""
import sys, os, csv, time, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur import anneal_multi
TM = TimeModel(s_service=60.0, tau=180.0)
N = 12; ITERS = int(os.environ.get("ITERS", "100000")); ALPHA = 0.9995 ** (20000.0 / ITERS) if os.environ.get("SCALE_ALPHA") else 0.9995; OUTF = os.environ.get("OUTF", "bridge_real_n12_rerun.csv")
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}

def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    (arm, seed), cells = job
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(ST[arm], N, seed, 3000.0, 100.0, HF[arm]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; out = []
    for c in cells:
        t0 = time.time()
        r = anneal_multi(leg, R, TYPE_M600, TM, int(c["k"]), float(c["T"]), restarts=3, iters=ITERS, seed0=seed, alpha=ALPHA)
        out.append(dict(arm=arm, seed=seed, k=c["k"], theta=c["theta"], T=c["T"], E_exact=c["E_exact"], E_old=c["E_heur"],
                        E_new=(round(r[0], 3) if r else ""), sec=round(time.time() - t0, 1)))
    return out

if __name__ == "__main__":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    rows = list(csv.DictReader(open(os.path.join(HERE, "bridge_real_n12.csv"), encoding="utf-8")))
    g = {}
    for r in rows:
        miss = r["E_heur"] == "" or float(r["E_heur"]) / float(r["E_exact"]) - 1 > 0.001
        if miss: g.setdefault((r["arm"], int(r["seed"])), []).append(r)
    jobs = list(g.items()); print("jobs", len(jobs), "cells", sum(len(v) for v in g.values()), flush=True)
    t0 = time.time()
    with Pool(2) as p: res = p.map(one, jobs, chunksize=1)
    rr = [x for y in res for x in y]
    with open(os.path.join(HERE, OUTF), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
    print("done", len(rr), "sec", round(time.time() - t0), flush=True)
