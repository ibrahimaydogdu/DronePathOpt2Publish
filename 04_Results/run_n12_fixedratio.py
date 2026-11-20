"""Premium at constant requests-per-drone: n = 12 with k = 3 and k = 6 (n/k = 4 and 2), to be compared with n = 8 with k = 2 and k = 4 (realdemand_rows.csv).
Same instances as run_n12_cap.py (CAP = unrestricted). Usage (from 03_Code_Execution): NW=3 python -I ../04_Results/run_n12_fixedratio.py <draws>   Output: n12_fixedratio_rows.csv, N12_FIXEDRATIO_SUMMARY_D2.txt"""
import sys, os, csv, time, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
import importlib.util
spec = importlib.util.spec_from_file_location("n12", os.path.join(HERE, "run_n12_cap.py")); n12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(n12)
from drone3d_d2.fleet import trip_table, one_drone_table, fleet_frontier


def one(args):
    (name, st_), seed = args
    leg, R = n12.build(name, st_, 12, seed)
    h = one_drone_table(trip_table(leg, R, n12.TYPE_M600, n12.TM), 12, n12.TM)
    rows = []
    F1 = fleet_frontier([h], 12); t1 = F1[-1][1]
    for k in (3, 6):
        F = fleet_frontier([h] * k, 12)
        rows.append(dict(arm=name, seed=seed, k=k, E_min=round(F[0][0], 3), Tmin=round(F[-1][1], 1), Tmin_k1=round(t1, 1), premium=round(F[-1][0] / F[0][0] - 1, 5)))
    return rows


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = min(int(os.environ.get("NW", "1")), 3)
    with Pool(nw) as p: res = p.map(one, [(a, 1001 + i) for a in n12.ARMS for i in range(draws)], chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(HERE, "n12_fixedratio_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = list(csv.DictReader(open(os.path.join(HERE, "realdemand_rows.csv"), encoding="utf-8")))
    out = []
    for a, _ in n12.ARMS:
        for k12, k8 in ((3, 2), (6, 4)):
            g = [r for r in rows if r["arm"] == a and r["k"] == k12]
            b = [float(r["E_Tmin"]) / float(r["E_min"]) - 1 for r in base if r["arm"] == a and r["k"] == str(k8) and r["E_Tmin"]]
            out.append(f"{a}: n=12,k={k12} premium median {100*st.median(r['premium'] for r in g):.1f} % (max {100*max(r['premium'] for r in g):.1f}); n=8,k={k8} premium median {100*st.median(b):.1f} % (max {100*max(b):.1f})")
    open(os.path.join(HERE, "N12_FIXEDRATIO_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
