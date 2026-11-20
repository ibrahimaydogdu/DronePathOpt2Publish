"""Summary of realdemand_rows.csv -> REALDEMAND_SUMMARY_D2.txt. Per arm: n instances, medians of E_min, Tmin(k=1,4), energy premium of the
time-minimal plan at k=4 over the energy-minimal plan, k needed to reach E_min at T in TGRID (share of draws), pool/heavy."""
import csv, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "realdemand_rows_n10.csv"), encoding="utf-8")))
f = lambda x: float(x) if x not in ("", None) else None
by = {}
for r in rows:
    by.setdefault((r["arm"], r["seed"]), {})[int(r["k"])] = r
arms = sorted({a for a, _ in by}, key=lambda a: [x[0] for x in dict.fromkeys((r["arm"],) for r in rows)].index(a) if False else a)
out = []
for a in dict.fromkeys(r["arm"] for r in rows):
    inst = [v for (aa, s), v in by.items() if aa == a]
    emin = [f(v[1]["E_min"]) for v in inst if f(v[1]["E_min"]) is not None]
    t1 = [f(v[1]["Tmin"]) for v in inst if f(v[1]["Tmin"]) is not None]
    t4 = [f(v[4]["Tmin"]) for v in inst if f(v[4]["Tmin"]) is not None]
    prem = [f(v[4]["E_Tmin"]) / f(v[4]["E_min"]) - 1 for v in inst if f(v[4]["E_min"])]
    # smallest k whose slack energy (E_min) equals the k=4 E_min within 0.5 % and share of draws needing k>1
    kslack = []
    for v in inst:
        e4 = f(v[4]["E_min"])
        if e4 is None: continue
        kslack.append(min(k for k in (1, 2, 3, 4) if f(v[k]["E_min"]) is not None and f(v[k]["E_min"]) <= e4 * 1.005))
    r0 = inst[0][1]
    out.append(f"{a}: draws={len(inst)} pool={r0['pool']} heavy_excl={r0['heavy']} mean_q(draw mean)~{sum(float(v[1]['mean_q']) for v in inst)/len(inst):.2f}")
    out.append(f"  feasible k=1: {len(emin)}/{len(inst)}; E_min median {st.median(emin):.1f} Wh [{min(emin):.1f}, {max(emin):.1f}]")
    out.append(f"  Tmin median k=1 {st.median(t1):.0f} s, k=4 {st.median(t4):.0f} s (ratio {st.median(t1)/st.median(t4):.2f})")
    out.append(f"  energy premium of the time-minimal plan at k=4 over E_min: median {100*st.median(prem):.1f} %, max {100*max(prem):.1f} %")
    out.append(f"  draws where k=1 slack energy exceeds k=4 by > 0.5 %: {sum(1 for k in kslack if k>1)}/{len(kslack)}")
    for T in (900, 1800, 3600):
        k_need = []
        for v in inst:
            ok = [k for k in (1, 2, 3, 4) if f(v[k][f"E_T{T}"]) is not None]
            k_need.append(min(ok) if ok else None)
        feas = [k for k in k_need if k]
        out.append(f"  T={T}: feasible at some k<=4 in {len(feas)}/{len(inst)} draws; median min k {st.median(feas) if feas else '-'}")
open(os.path.join(HERE, "REALDEMAND_SUMMARY_n10_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
