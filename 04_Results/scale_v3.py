"""Scale profile v3 with PRODUCTION code (drone3d_d2): time and peak working set per (mode, n, seed), K=3, tau/s base.
Usage: python scale_v3.py child <mode> <n> <seed> <span>   |   python scale_v3.py all
mode: noB (D,P only) | mix (D,P,B) | worstB (all B requests). Instances SYNTHETIC.
"""
import sys, os, time, subprocess, psutil, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution"))


def child(mode, n, seed, span):
    from drone3d_d2.fleet import Req, frontier_homogeneous
    from drone3d_d2.dtypes import TYPE_M600, TimeModel
    from drone3d_d2.legs import PlanarLeg
    kinds = {"noB": "DP", "mix": "DPB", "worstB": "B"}[mode]
    r = random.Random(seed); pts = [(0.0, 0.0, 20.0)]; reqs = []
    for _ in range(n):
        k = r.choice(kinds); q = round(r.uniform(0.5, 3.0), 1)
        pt = lambda: (r.uniform(-span, span), r.uniform(-span, span), r.uniform(15, 120))
        if k == "B":
            pts.append(pt()); pts.append(pt()); reqs.append(Req("B", len(pts) - 2, len(pts) - 1, q))
        elif k == "D":
            pts.append(pt()); reqs.append(Req("D", 0, len(pts) - 1, q))
        else:
            pts.append(pt()); reqs.append(Req("P", len(pts) - 1, 0, q))
    t0 = time.time()
    F, trips, h = frontier_homogeneous(PlanarLeg(pts, TYPE_M600), reqs, TYPE_M600, TimeModel(), 3)
    dt = time.time() - t0
    peak = psutil.Process().memory_info().peak_wset / 1e6
    print(f"RESULT {mode} n={n} seed={seed} span={span:.0f} time={dt:.1f}s peakMB={peak:.0f} frontier={len(F)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])); sys.exit()
    jobs = []
    for mode, ns in (("noB", (10, 11, 12, 13)), ("mix", (8, 9, 10, 11)), ("worstB", (7, 8, 9, 10))):
        for n in ns:
            for span in (400.0, 1500.0):
                for seed in (1001, 1002, 1003, 1004, 1005):
                    jobs.append((mode, n, seed, span))
    out = open(os.path.join(HERE, "SCALE_V3_raw.txt"), "a", encoding="utf-8")
    skip = {}
    for mode, n, seed, span in jobs:
        if n >= skip.get((mode, span), 999):
            continue
        try:
            p = subprocess.run([sys.executable, __file__, "child", mode, str(n), str(seed), str(span)], capture_output=True, text=True, timeout=300)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
            msg = line[0] if line else f"FAIL {mode} n={n} seed={seed} span={span:.0f} rc={p.returncode} {p.stderr[-120:]}"
        except subprocess.TimeoutExpired:
            msg = f"TIMEOUT>300s {mode} n={n} seed={seed} span={span:.0f}"
        out.write(msg + "\n"); out.flush(); print(msg, flush=True)
        if msg.startswith("TIMEOUT"):
            skip[(mode, span)] = n
