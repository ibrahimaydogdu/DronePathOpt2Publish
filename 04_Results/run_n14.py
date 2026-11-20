"""Exact frontier at 14 requests (extends the trend of the premium with the number of requests): k = 1 and 4, base arms, DRAWS draws per station, unrestricted trips.
Runs at below-normal priority. Usage (from 03_Code_Execution): NW=4 python -I ../04_Results/run_n14.py <draws>   Output: n14_rows.csv, N14_SUMMARY_D2.txt"""
import sys, os, csv, time, ctypes, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
import importlib.util
spec = importlib.util.spec_from_file_location("n12", os.path.join(HERE, "run_n12_cap.py")); n12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(n12)
from drone3d_d2.fleet import trip_table, one_drone_table, fleet_frontier
N = 14


def low():
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception: pass


def one(args):
    (name, st_), seed = args
    t0 = time.time(); leg, R = n12.build(name, st_, N, seed)
    h = one_drone_table(trip_table(leg, R, n12.TYPE_M600, n12.TM), N, n12.TM); t1 = time.time() - t0
    F1 = fleet_frontier([h], N); t2 = time.time()
    F4 = fleet_frontier([h] * 4, N); t3 = time.time() - t2
    return dict(arm=name, seed=seed, E_min=round(F4[0][0], 3), Tmin_k1=round(F1[-1][1], 1), Tmin_k4=round(F4[-1][1], 1), ratio=round(F1[-1][1] / F4[-1][1], 3),
                premium=round(F4[-1][0] / F4[0][0] - 1, 5), npts_k4=len(F4), sec_tables=round(t1), sec_fleet4=round(t3))


if __name__ == "__main__":
    draws = int(sys.argv[1]); nw = int(os.environ.get("NW", "2"))
    with Pool(nw, initializer=low) as p:
        res = []
        for r in p.imap(one, [(a, 1001 + i) for i in range(draws) for a in n12.ARMS], chunksize=1):
            res.append(r); print(r, flush=True)
            with open(os.path.join(HERE, "n14_rows.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    out = [f"n = 14, k = 1 and 4, {draws} draws per station"]
    for a, _ in n12.ARMS:
        g = [r for r in res if r["arm"] == a]
        out.append(f"{a}: E_min median {st.median(r['E_min'] for r in g):.1f}; Tmin k=1 {st.median(r['Tmin_k1'] for r in g):.0f} -> k=4 {st.median(r['Tmin_k4'] for r in g):.0f} (ratio of medians {st.median(r['Tmin_k1'] for r in g)/st.median(r['Tmin_k4'] for r in g):.2f}); premium median {100*st.median(r['premium'] for r in g):.1f} % (max {100*max(r['premium'] for r in g):.1f}); sec tables median {st.median(r['sec_tables'] for r in g):.0f}, fleet(4) {st.median(r['sec_fleet4'] for r in g):.0f}")
    open(os.path.join(HERE, "N14_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
