"""(1) Absolute makespan bounds on the real base arms: share of draws feasible with at most four drones, median smallest fleet k*, median premium at k* and at k = 4, for T = 900, 1200, 1800 s.
(2) Bootstrap 95 % intervals (10,000 resamples of the 30 draws, percentile) of the median premium and of the ratio of the shortest makespans at four and one drone, n = 8, 10, 12.
The draws overlap in the pools of Chicago and Austin, so the intervals describe the sampling of draws, not of stops. Output: ABS_T_CI_SUMMARY_D2.txt"""
import csv, os, random, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
def rd(f): return list(csv.DictReader(open(os.path.join(H, f), encoding="utf-8")))
R8, R10, R12 = rd("realdemand_rows.csv"), rd("realdemand_rows_n10.csv"), rd("n12_cap_rows.csv")
ARMS = [("LA_base", "Los Angeles"), ("CHI_base", "Chicago"), ("AUS_base", "Austin")]
out = ["Absolute bounds, n = 8 (30 draws): feasible with k<=4 | median k* | median premium at k* | median premium at k=4"]
def fl(x): return float(x) if x not in ("", None) else None
for n, R in ((8, R8), (10, R10)):
    out.append(f"-- n = {n}")
    for a, nm in ARMS:
        for T in (900, 1200, 1800):
            nf = 0; ks = []; pk = []; p4 = []
            for seed in sorted({r["seed"] for r in R if r["arm"] == a}):
                rows = {int(r["k"]): r for r in R if r["arm"] == a and r["seed"] == seed}
                emin = fl(rows[4]["E_min"]); ok = [k for k in (1, 2, 3, 4) if fl(rows[k][f"E_T{T}"]) is not None]
                if not ok: continue
                nf += 1; k0 = min(ok); ks.append(k0); pk.append(fl(rows[k0][f"E_T{T}"]) / emin - 1)
                if 4 in ok: p4.append(fl(rows[4][f"E_T{T}"]) / emin - 1)
            out.append(f"{nm} T={T}: feasible {nf}/30; median k* {st.median(ks):.1f}; premium at k* median {100*st.median(pk):.1f} % (share >1 %: {sum(x>0.01 for x in pk)}/{len(pk)}); at k=4 median {100*st.median(p4):.1f} %")
rng = random.Random(1001)
def boot(v, f=st.median):
    b = sorted(f([rng.choice(v) for _ in v]) for _ in range(10000)); return b[250], b[9749]
out.append("Bootstrap 95 % intervals over draws (median premium %, ratio of medians of Tmin k=1 / k=4)")
for n, R in ((8, R8), (10, R10)):
    for a, nm in ARMS:
        prem = []; t1 = []; t4 = []
        for seed in sorted({r["seed"] for r in R if r["arm"] == a}):
            rows = {int(r["k"]): r for r in R if r["arm"] == a and r["seed"] == seed}
            prem.append(fl(rows[4]["E_Tmin"]) / fl(rows[4]["E_min"]) - 1); t1.append(fl(rows[1]["Tmin"])); t4.append(fl(rows[4]["Tmin"]))
        lo, hi = boot(prem)
        idx = list(range(len(t1))); rb = []
        for _ in range(10000):
            ii = [rng.choice(idx) for _ in idx]; rb.append(st.median(t1[i] for i in ii) / st.median(t4[i] for i in ii))
        rb.sort()
        out.append(f"n={n} {nm}: premium median {100*st.median(prem):.1f} % [{100*lo:.1f}, {100*hi:.1f}]; ratio {st.median(t1)/st.median(t4):.2f} [{rb[250]:.2f}, {rb[9749]:.2f}]")
for a, nm in ARMS:
    g = [r for r in R12 if r["arm"] == a]; prem = [float(r["premium"]) for r in g]; t1 = [float(r["Tmin_k1"]) for r in g]; t4 = [float(r["Tmin_k4"]) for r in g]
    lo, hi = boot(prem); idx = list(range(len(g))); rb = []
    for _ in range(10000):
        ii = [rng.choice(idx) for _ in idx]; rb.append(st.median(t1[i] for i in ii) / st.median(t4[i] for i in ii))
    rb.sort()
    out.append(f"n=12 {nm}: premium median {100*st.median(prem):.1f} % [{100*lo:.1f}, {100*hi:.1f}]; ratio {st.median(t1)/st.median(t4):.2f} [{rb[250]:.2f}, {rb[9749]:.2f}]")
open(os.path.join(H, "ABS_T_CI_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
