"""Scale bridge (SYNTHETIC instances). Part 1: SA gap to the exact frontier at n <= 10. Part 2: SA-only runs at n = 20, 30, 50.
Usage (from 03_Code_Execution): python ../04_Results/run_scale_bridge.py gap <n> <draws> <iters> | scale <n> <draws> <iters>
NW env var = worker count (max 3 on this PC).
"""
import sys, os, csv, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))
from drone3d_d2.fleet import frontier_homogeneous, min_energy_at
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import PlanarLeg
from drone3d_d2.heur import anneal_multi
sys.path.insert(0, HERE)
from run_phaseA import make_inst

TM = TimeModel(s_service=60.0, tau=180.0)
KS = (1, 2, 3)
THETAS = (0.0, 0.25, 0.5, 1.0)


def gap_one(args):
    n, seed, iters = args
    pts, reqs = make_inst(n, seed)
    leg = PlanarLeg(pts, TYPE_M600)
    rows = []
    fr = {}
    for k in KS:
        f, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, TM, k)
        fr[k] = f
        if not f:
            continue
        tmin, temin = f[0][1], f[-1][1]   # f sorted by E asc: first = E_min (largest T), last = min T
        tmin, temin = f[-1][1], f[0][1]
        for th in THETAS:
            T = tmin + th * (temin - tmin)
            ex = min_energy_at(f, T)
            t0 = time.time()
            h = anneal_multi(leg, reqs, TYPE_M600, TM, k, T, restarts=3, iters=iters, seed0=seed)
            rows.append(dict(n=n, seed=seed, k=k, theta=th, T=round(T, 1), E_exact=ex[0], E_heur=(h[0] if h else ""),
                             sec=round(time.time() - t0, 1)))
    return rows


def scale_one(args):
    n, seed, iters = args
    pts, reqs = make_inst(n, seed)
    leg = PlanarLeg(pts, TYPE_M600)
    rows = []
    for k in (2, 4, 6):
        for T in (1800, 3600, 7200):
            t0 = time.time()
            h = anneal_multi(leg, reqs, TYPE_M600, TM, k, float(T), restarts=3, iters=iters, seed0=seed)
            rows.append(dict(n=n, seed=seed, k=k, T=T, E_heur=(h[0] if h else ""), mk=(h[1] if h else ""),
                             sec=round(time.time() - t0, 1)))
    return rows


if __name__ == "__main__":
    mode, n, draws, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    nw = min(int(os.environ.get("NW", "1")), 3)
    fn = gap_one if mode == "gap" else scale_one
    jobs = [(n, 1001 + i, iters) for i in range(draws)]
    with Pool(nw) as p:
        res = p.map(fn, jobs)
    rows = [r for rr in res for r in rr]
    out = os.path.join(HERE, f"bridge_{mode}_n{n}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out, len(rows))
