import csv, statistics as st, os
H = os.path.dirname(os.path.abspath(__file__)); out = []
for n in (8, 10):
    rows = list(csv.DictReader(open(os.path.join(H, f"bridge_real_n{n}.csv"), encoding="utf-8")))
    def blk(g, lab):
        miss = [r for r in g if r["E_heur"] == ""]
        gaps = [float(r["E_heur"]) / float(r["E_exact"]) - 1 for r in g if r["E_heur"] != ""]
        exact_hit = sum(1 for x in gaps if x <= 1e-4)
        out.append(f"n={n} {lab}: runs={len(g)} no feasible solution found={len(miss)}; among found: exact (gap<=0.01%) {exact_hit}/{len(gaps)}, mean gap {100*st.mean(gaps):.2f}%, p95 {100*sorted(gaps)[int(0.95*(len(gaps)-1))]:.2f}%, max {100*max(gaps):.2f}%; mean sec {st.mean(float(r['sec']) for r in g):.1f}")
    blk(rows, "all")
    for a in ("LA_base", "CHI_base", "AUS_base"): blk([r for r in rows if r["arm"] == a], a)
    for th in ("0.0", "0.25", "0.5", "1.0"): blk([r for r in rows if r["theta"] == th], f"theta={th}")
    for k in ("2", "4"): blk([r for r in rows if r["k"] == k], f"k={k}")
    bad = [r for r in rows if r["E_heur"] != "" and float(r["mk_heur"]) > float(r["T"]) + 1e-6]
    out.append(f"n={n}: heuristic makespan above bound in {len(bad)} runs (must be 0)")
open(os.path.join(H, "BRIDGE_REAL_SUMMARY_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
