"""AUS_base lines for BASELINES_SUMMARY_D2.txt and REALINVENTORY_SUMMARY_D2.txt (same format as the LA/CHI lines), from the merged CSVs."""
import csv, os, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
f = lambda x: float(x) if x not in ("", None) else None
bl, iv = [], []
rows = [r for r in csv.DictReader(open(os.path.join(H, "baselines_rows.csv"), encoding="utf-8")) if r["arm"] == "AUS_base"]
for base in ("DIST", "LOADBL"):
    for th in ("0.0", "0.25", "0.5", "1.0"):
        g = [r for r in rows if r["base"] == base and r["theta"] == th]
        ok = [float(r["regret"]) for r in g if r["status"] == "ok"]
        no = {s: sum(1 for r in g if r["status"] == s) for s in ("no_plan", "battery_infeasible")}
        bl.append(f"AUS_base {base} theta={th}: n={len(g)} not_ok={len(g)-len(ok)} (no_plan {no['no_plan']}, battery {no['battery_infeasible']}) regret>0 in {sum(1 for x in ok if x>1e-6)}/{len(ok)}; median {100*st.median(ok):.2f}% mean {100*st.mean(ok):.2f}% max {100*max(ok):.2f}%")
rows = [r for r in csv.DictReader(open(os.path.join(H, "realinventory_rows.csv"), encoding="utf-8")) if r["arm"] == "AUS_base"]
for b in ("1", "2", "inf"):
    for c in ("0.0", "15.0", "45.0"):
        by = {}
        for r in rows:
            if r["b"] == b and r["c"] == c: by.setdefault(r["seed"], {})[int(r["k"])] = r
        feas = tuple(sum(1 for v in by.values() if f(v[k]["E_slack"]) is not None) for k in (1, 2, 3, 4))
        ex = []; gap1 = []
        for k in (1, 2, 3):
            n = 0
            for v in by.values():
                e4, ek = f(v[4]["E_slack"]), f(v[k]["E_slack"])
                if e4 is not None and ek is not None and ek > e4 * 1.005:
                    n += 1
                    if k == 1: gap1.append(ek / e4 - 1)
            ex.append(n)
        prem = [f(v[4]["E_at_Tmin"]) / f(v[4]["E_slack"]) - 1 for v in by.values() if f(v[4]["E_slack"])]
        u = [f(v[4]["u_Emin"]) for v in by.values() if f(v[4]["u_Emin"]) is not None]
        iv.append(f"AUS_base b={b} c={c}: feasible(k=1..4)={feas}/30; slack-E above k=4 by >0.5% (k=1,2,3)={tuple(ex)}; k=1 max gap {100*max(gap1) if gap1 else 0.0:.1f}%; time-min premium at k=4 median {100*st.median(prem):.1f}%; drones used at E_min median {st.median(u):.1f}")
open(os.path.join(H, "BASELINES_SUMMARY_AUS_D2.txt"), "w", encoding="utf-8").write("\n".join(bl) + "\n")
open(os.path.join(H, "REALINVENTORY_SUMMARY_AUS_D2.txt"), "w", encoding="utf-8").write("\n".join(iv) + "\n")
print("\n".join(bl)); print("\n".join(iv))
