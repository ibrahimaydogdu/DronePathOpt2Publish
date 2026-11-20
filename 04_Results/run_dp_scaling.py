"""Exact-solver run time and memory against n on real instances (Los Angeles DLA4, 3 draws, k = 4): trip table (leg costs + per-subset order DP),
one-drone table, fleet merge. Peak resident memory via psutil if available else tracemalloc. Output: dp_scaling.csv, DP_SCALING_SUMMARY_D2.txt.
Usage (from 03_Code_Execution): python -I ../04_Results/run_dp_scaling.py <nmax>"""
import sys, os, time, csv, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from legs import LegCost
from around import CityRaster
import tracemalloc
TM = TimeModel(s_service=60.0, tau=180.0)
rows = []
for n in range(8, int(sys.argv[1]) + 1):
    for seed in (1001, 1002, 1003):
        city, nodes, reqs, meta = make_real_instance("DLA4", n, seed, 3000.0, 100.0, height_la)
        leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
        R = [Req(*r) for r in reqs]
        tracemalloc.start(); t0 = time.time()
        tt = trip_table(leg, R, TYPE_M600, TM); t1 = time.time()
        h = one_drone_table(tt, n, TM); t2 = time.time()
        F = fleet_frontier([h] * 4, n); t3 = time.time()
        peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
        rows.append(dict(n=n, seed=seed, sec_trips=round(t1 - t0, 1), sec_onedrone=round(t2 - t1, 1), sec_fleet=round(t3 - t2, 1), sec_total=round(t3 - t0, 1), peak_mb=round(peak), npts_k4=len(F)))
        print(rows[-1], flush=True)
with open(os.path.join(HERE, "dp_scaling.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
out = ["n: median total seconds (trips / one-drone / fleet), median Python-heap peak MB (tracemalloc; excludes numpy/C buffers), median Pareto points at k=4"]
for n in sorted({r["n"] for r in rows}):
    g = [r for r in rows if r["n"] == n]
    out.append(f"n={n}: total {st.median(r['sec_total'] for r in g):.1f} s ({st.median(r['sec_trips'] for r in g):.1f} / {st.median(r['sec_onedrone'] for r in g):.1f} / {st.median(r['sec_fleet'] for r in g):.1f}); peak {st.median(r['peak_mb'] for r in g):.0f} MB; points {st.median(r['npts_k4'] for r in g):.0f}")
open(os.path.join(HERE, "DP_SCALING_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
