"""Ablation of guidance on the library ABC (n=12 cells of bridge_real_n12.csv). Arms: seed, repair, both (base = column E_ABC of keys_n12_80000.csv, same seeds).
Usage (from 03_Code_Execution): BUD=80000 NW=2 python -X utf8 ../04_Results/run_ablation_n12.py [max_jobs]   Output: ablation_n12_<BUD>.csv"""
import sys, os, csv, time, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_guided import guided_abc
TM = TimeModel(s_service=60.0, tau=180.0)
N = 12; BUD = int(os.environ.get("BUD", "80000"))
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
ARMS = (("seed", True, False), ("repair", False, True), ("both", True, True))


def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    (arm, seed), cells = job
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(ST[arm], N, seed, 3000.0, 100.0, HF[arm]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; out = []
    for c in cells:
        row = dict(arm=arm, seed=seed, k=c["k"], theta=c["theta"], T=c["T"], E_exact=c["E_exact"], E_sa=c["E_sa"], E_ABC=c["E_ABC"])
        s = seed + 31 * int(float(c["theta"]) * 4) + 7 * int(c["k"])
        for nm, us, ur in ARMS:
            t0 = time.time()
            r = guided_abc(leg, R, TYPE_M600, TM, int(c["k"]), (float(c["T"]) + 0.05), BUD, s, us, ur)
            row["E_" + nm] = round(r, 3) if r is not None else ""
            row["sec_" + nm] = round(time.time() - t0, 1)
        out.append(row)
    return out


if __name__ == "__main__":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    rows = list(csv.DictReader(open(os.path.join(HERE, "keys_n12_80000.csv"), encoding="utf-8")))
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
    with open(os.path.join(HERE, f"ablation_n12_{BUD}" + ("_th0" if os.environ.get("THETA0") else "") + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
        w.writeheader(); w.writerows(rr)
    print("done", len(rr), "sec", round(time.time() - t0), flush=True)
