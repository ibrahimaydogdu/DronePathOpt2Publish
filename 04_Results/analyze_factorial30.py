"""2x2 factorial summary: merges factorial30/*.csv with campaign30 (same 72 cells, same run seeds)."""
import csv, glob, os, collections
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda pat: [r for fn in sorted(glob.glob(os.path.join(HERE, pat))) for r in csv.DictReader(open(fn, encoding="utf-8"))]
F = rd("factorial30/*.csv"); C = rd("campaign30/*.csv")
key = lambda r: (r["arm"], r["inst"], r["k"], r["theta"], r["run"])
cm = {key(r): r for r in C}
rows = []
for r in F:
    c = cm[key(r)]
    for m in ("SA_full", "SA_plain", "libABC", "gABC"): r[m] = c[m]
    rows.append(r)
M = ["SA_full", "SA_plain", "SAkeys", "libABC", "TLBO", "GWO", "gABC", "ownABC", "ownGA"]
mt = lambda r, k: r[k] != "" and float(r[k]) <= float(r["E_exact"]) + 1e-3 * max(1, abs(float(r["E_exact"])))
print("rows", len(rows), "cells", len({key(r)[:4] for r in rows}))
print("Layout: paradigm x encoding.  SA/native: SA_full(guided), SA_plain; SA/keys: SAkeys; ABC/keys: libABC; ABC/native: ownABC; GA/native: ownGA; TLBO/GWO keys")
print(f"{'method':9s} match% noplan% meangap%(feasible)")
for m in M:
    mm = sum(mt(r, m) for r in rows); npn = sum(r[m] == "" for r in rows)
    g = [max(0, (float(r[m]) - float(r["E_exact"])) / float(r["E_exact"])) for r in rows if r[m] != ""]
    print(f"{m:9s} {100*mm/len(rows):6.1f} {100*npn/len(rows):7.1f} {100*sum(g)/max(1,len(g)):8.2f}")
cells = collections.defaultdict(list)
for r in rows: cells[key(r)[:4]].append(r)
sr = {m: [sum(mt(r, m) for r in v) / len(v) for v in cells.values()] for m in M}
print("\nPaired Wilcoxon on per-cell success rate (SA_full vs other), Holm over", len(M) - 1, "comparisons")
ps = []
for m in M[1:]:
    d = [a - b for a, b in zip(sr["SA_full"], sr[m])]
    ps.append((wilcoxon(d).pvalue if any(d) else 1.0, m, sum(x > 0 for x in d), sum(x < 0 for x in d), sum(d) / len(d)))
ps.sort(); mx = 0
for i, (p, m, w, l, md) in enumerate(ps):
    mx = max(mx, min(1, (len(ps) - i) * p))
    print(f"  vs {m:9s} p_holm {mx:.2e}  SA better in {w}, worse in {l} cells, mean diff {100*md:.1f} pts")
