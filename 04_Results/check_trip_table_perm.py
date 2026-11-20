"""Independent check of the trip table: for every subset of at most SIZE requests, enumerate all event orders (permutations), evaluate each with analysis.eval_trip_order
and compare the Pareto set of (E, T) with the label-correcting result of fleet.trip_pareto (real instances, n = 8). Usage (from 03_Code_Execution): python -I ../04_Results/check_trip_table_perm.py <draws> <SIZE>"""
import sys, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, pareto, EPS
from drone3d_d2.analysis import eval_trip_order
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
TM = TimeModel(s_service=60.0, tau=180.0)
ARMS = [("LA", "DLA4", height_la), ("CHI", "DCH1", height_chicago(3.5, 0)), ("AUS", "DAU1", height_austin(0.0))]
draws, SIZE = int(sys.argv[1]), int(sys.argv[2]); bad = tot = 0; ncmp = 0
from legs import LegCost
from around import CityRaster
for name, st, hf in ARMS:
    for seed in range(1001, 1001 + draws):
        city, nodes, reqs, meta = make_real_instance(st, 8, seed, 3000.0, 100.0, hf)
        leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
        R = [Req(*r) for r in reqs]; tt = trip_table(leg, R, TYPE_M600, TM)
        for mask in range(1, 256):
            S = tuple(i for i in range(8) if mask >> i & 1)
            if len(S) > SIZE: continue
            cand = []
            for perm in itertools.permutations(S):
                ev = [(r, "d", R[r].d) for r in perm]
                x = eval_trip_order(leg, R, TYPE_M600, TM, S, ev)
                if x is not None: cand.append(x)
            ref = [(round(e, 5), round(t, 3)) for e, t in pareto(cand, key=lambda z: z)]
            got = [(round(e, 5), round(t, 3)) for e, t, _ in tt[mask]]
            tot += 1
            if ref != got: bad += 1; print("MISMATCH", name, seed, S, ref[:3], got[:3])
print(f"subsets checked {tot}; mismatches {bad}")
