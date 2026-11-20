"""ALNS (heur_alns.py) on the 72 cells of the factorial (instances 1001-1003), 30 runs (seeds 1001-1030), budget 80000, T=stored T+0.05. Output alns30/*.csv"""
import sys, os, csv, time, ctypes
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from drone3d_d2.heur_alns import alns
TM = TimeModel(s_service=60.0, tau=180.0)
ST = {"LA_base": "DLA4", "CHI_base": "DCH1", "AUS_base": "DAU1"}
HF = {"LA_base": lambda: height_la, "CHI_base": lambda: height_chicago(3.5, 0), "AUS_base": lambda: height_austin(0.0)}
OUT = os.path.join(HERE, "alns30")

def one(job):
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    (arm, seed, kk), cells = job
    fn = os.path.join(OUT, f"{arm}_{seed}_k{kk}.csv")
    if os.path.exists(fn): return fn
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance(ST[arm], 12, seed, 3000.0, 100.0, HF[arm]())
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]; out = []
    for c in cells:
        for rs in range(1001, 1031):
            r = alns(leg, R, TYPE_M600, TM, int(c["k"]), float(c["T"]) + 0.05, 80000, rs)
            out.append(dict(arm=arm, inst=seed, k=c["k"], theta=c["theta"], run=rs, E_exact=c["E_exact"], ALNS=(round(r[0], 3) if r else "")))
    with open(fn + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    os.replace(fn + ".tmp", fn); return fn

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    g = {}
    for r in csv.DictReader(open(os.path.join(HERE, "bridge_real_n12.csv"), encoding="utf-8")):
        if int(r["seed"]) in (1001, 1002, 1003): g.setdefault((r["arm"], int(r["seed"]), int(r["k"])), []).append(r)
    with Pool(6) as p:
        for fn in p.imap_unordered(one, list(g.items())): print("done", fn, flush=True)
    open(os.path.join(OUT, "ALL_DONE.txt"), "w").write("ok")
