"""Analysis of campaign30/*.csv: success (exact match within 1e-3 rel.), no-plan, gap; per-cell Mann-Whitney U (gap, failures=+inf rank) SA_full vs others, Holm, A12."""
import csv, glob, os, sys, math, collections
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__))
M = ["SA_full", "SA_plain", "libABC", "gABC"]
rows = []
for fn in sorted(glob.glob(os.path.join(HERE, "campaign30", "*.csv"))):
    rows += list(csv.DictReader(open(fn, encoding="utf-8")))
print("rows", len(rows), "cells", len({(r["arm"], r["inst"], r["k"], r["theta"]) for r in rows}))


def gap(r, m):
    v = r[m]
    if v == "": return math.inf
    e = float(r["E_exact"]); return max(0.0, (float(v) - e) / e)


def match(r, m):
    return r[m] != "" and float(r[m]) <= float(r["E_exact"]) + 1e-3 * max(1, abs(float(r["E_exact"])))


print("\nOverall per-run: match% / noplan% / mean gap(%) over feasible")
for m in M:
    mt = sum(match(r, m) for r in rows); npn = sum(r[m] == "" for r in rows)
    g = [gap(r, m) for r in rows if r[m] != ""]
    print(f"  {m:9s} {100*mt/len(rows):5.1f} {100*npn/len(rows):5.1f} {100*sum(g)/max(1,len(g)):6.2f}")
print("\nBy theta (match %):")
for th in sorted({r["theta"] for r in rows}, key=float):
    s = [r for r in rows if r["theta"] == th]
    print("  theta", th, {m: round(100 * sum(match(r, m) for r in s) / len(s), 1) for m in M})
print("By k:")
for k in sorted({r["k"] for r in rows}):
    s = [r for r in rows if r["k"] == k]
    print("  k", k, {m: round(100 * sum(match(r, m) for r in s) / len(s), 1) for m in M})
# per-cell success rate over 30 runs
cells = collections.defaultdict(list)
for r in rows: cells[(r["arm"], r["inst"], r["k"], r["theta"])].append(r)
print("\nPer-cell (30 runs): cells where method matches exact in all/majority/never")
for m in M:
    sr = [sum(match(r, m) for r in v) / len(v) for v in cells.values()]
    print(f"  {m:9s} all {sum(x==1 for x in sr)}  >=50% {sum(x>=.5 for x in sr)}  never {sum(x==0 for x in sr)}  mean success {100*sum(sr)/len(sr):.1f}%")


def a12(x, y):  # P(x<y)+0.5P(x=y): prob that x better (smaller gap) than y
    n = 0.0
    for a in x:
        for b in y: n += (a < b) + 0.5 * (a == b)
    return n / (len(x) * len(y))


print("\nPer-cell MWU on gap (inf for no plan), SA_full vs each; Holm over cells within comparison; A12 = P(SA better)")
for m in M[1:]:
    ps, aa = [], []
    for key, v in cells.items():
        x = [gap(r, "SA_full") for r in v]; y = [gap(r, m) for r in v]
        if len(set(x + y)) == 1: ps.append(1.0); aa.append(0.5); continue
        ps.append(mannwhitneyu(x, y, alternative="two-sided").pvalue); aa.append(a12(x, y))
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); mx = 0
    for rnk, i in enumerate(order):
        mx = max(mx, min(1, (len(ps) - rnk) * ps[i])); adj[i] = mx
    sig_sa = sum(1 for p, a in zip(adj, aa) if p < .05 and a > .5); sig_o = sum(1 for p, a in zip(adj, aa) if p < .05 and a < .5)
    print(f"  SA_full vs {m:9s}: cells {len(ps)}; SA sig better {sig_sa}; other sig better {sig_o}; median A12 {sorted(aa)[len(aa)//2]:.2f}")
