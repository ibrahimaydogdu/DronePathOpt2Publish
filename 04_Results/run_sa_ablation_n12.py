"""SA component ablation on the n=12 cells of bridge_real_n12.csv (reference = column E_heur of that file, full SA, restarts=3, 20000 iterations).
Variants (unl, p1, hard): no_unl (F,T,T), no_p1 (T,F,F), soft (T,T,F), plain (F,F,F).
Usage (from 03_Code_Execution): NW=2 python -X utf8 ../04_Results/run_sa_ablation_n12.py [max_jobs]   Output: sa_ablation_n12.csv"""
import sys, os, csv, time, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_abl import anneal_multi_abl
TM = TimeModel(s_service=60.0, tau=180.0)
N = 12
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
VARS = (("full", True, True, True), ("no_unl", False, True, True), ("no_p1", True, False, False), ("soft", True, True, False), ("plain", False, False, False))


def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    (arm, seed), cells = job
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(ST[arm], N, seed, 3000.0, 100.0, HF[arm]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; out = []
    for c in cells:
        row = dict(arm=arm, seed=seed, k=c["k"], theta=c["theta"], T=c["T"], E_exact=c["E_exact"], E_full=c["E_heur"])
        for nm, unl, p1, hard in VARS:
            t0 = time.time()
            r = anneal_multi_abl(leg, R, TYPE_M600, TM, int(c["k"]), (float(c["T"]) + 0.05), unl, p1, hard, restarts=3, iters=20000, seed0=seed)
            row["E_" + nm] = round(r[0], 3) if r else ""
            row["sec_" + nm] = round(time.time() - t0, 1)
        out.append(row)
    return out


if __name__ == "__main__":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    rows = list(csv.DictReader(open(os.path.join(HERE, "bridge_real_n12.csv"), encoding="utf-8")))
    g = {}
    for r in rows:
        if os.environ.get("THETA0") and float(r["theta"]) != 0.0:
            continue
        g.setdefault((r["arm"], int(r["seed"])), []).append(r)
    jobs = list(g.items())
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "2"))) as p:
        res = p.map(one, jobs, chunksize=1)
    rr = [x for y in res for x in y]
    with open(os.path.join(HERE, "sa_ablation_n12.csv" if not os.environ.get("THETA0") else "sa_ablation_n12_th0.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
        w.writeheader(); w.writerows(rr)
    print("done", len(rr), "sec", round(time.time() - t0), flush=True)
