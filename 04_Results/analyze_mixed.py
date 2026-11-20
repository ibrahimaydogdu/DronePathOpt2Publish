"""Summaries of mixed_rows_<tag>.csv (SYNTHETIC; heavy parcels are a SCENARIO). Usage: python analyze_mixed.py <tag>
Columns: seed, rho, vfc, var (FlyCart power base: base = E, p09 = 0.9E), a (M600 count), b (FlyCart count), T, E_at_T, E_min, Tmin
"""
import sys, os, csv, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1]
rng = np.random.default_rng(20261113)
rows = list(csv.DictReader(open(os.path.join(HERE, f"mixed_rows_{tag}.csv"), encoding="utf-8")))
fl = lambda x: None if x == "" else float(x)
D = collections.defaultdict(dict)   # (rho,vfc,var,seed,T) -> {(a,b): E_at_T or None}
EM = {}
for r in rows:
    var = r.get("var", "base")
    EM[(float(r["rho"]), float(r["vfc"]), var, int(r["seed"]), int(r["a"]), int(r["b"]))] = fl(r["E_min"])
    if r["T"] == "":
        continue
    D[(float(r["rho"]), float(r["vfc"]), var, int(r["seed"]), int(r["T"]))][(int(r["a"]), int(r["b"]))] = fl(r["E_at_T"])
RHOS = sorted({k[0] for k in D}); CFG = sorted({(k[1], k[2]) for k in D}); SEEDS = sorted({k[3] for k in D})


def boot(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    b = [np.median(rng.choice(x, len(x))) for _ in range(10000)]
    return (np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


out = [f"# RQ3 Phase A (SENTETİK, {tag}): karma filo özeti (UZANTI analizi)\nM600 Pro (6 kg) + FlyCart 30 (30 kg); {len(SEEDS)} çekim; hafif parsel U[0,5–3] kg, ağır parsel (SENARYO, kanıtsız) U[6–25] kg ⇒ zorunlu FlyCart; s/τ 60/180 s; filo ızgarası a ≤ 4 M600, b ≤ 3 FlyCart, a+b ≤ 5. FlyCart güç tabanı: base = E, p09 = 0,9E (duyarlılık).\n"]
w = out.append
w("## 1. FlyCart/M600 tek-drone enerji oranı (hafif parsel, ρ=0, T gevşek: E_min) ve türetme duyarlılığı\n")
w("| FlyCart hızı | güç tabanı | oran medyan [GA] |\n|---|---|---|")
for (v, var) in CFG:
    ratios = []
    for s in SEEDS:
        a = EM.get((0.0, v, var, s, 1, 0)); b = EM.get((0.0, v, var, s, 0, 1))
        if a and b:
            ratios.append(b / a)
    m, lo, hi = boot(ratios)
    w(f"| {v:.0f} m/s | {var} | {m:.2f} [{lo:.2f}, {hi:.2f}] |")
w("\nGA yalnız çekim örneklemesidir; model belirsizliği güç tabanı varyantı (E ve 0,9E), hız ve κ varsayımları arasındaki farktır (bu tablonun satırları).\n")

w("## 2. En az enerjili uygun filo bileşimi (T = 900 ve 1800 s)\n")
w("| T | ρ | FC hızı / taban | uygun oran | min-enerji filo: yalnız M600 / yalnız FC / karma | medyan E Wh | ızgara sınırında (a=4 veya b=3 veya a+b=5) oran |\n|---|---|---|---|---|---|---|")
for T in (900, 1800):
    for rho in RHOS:
        for (v, var) in CFG:
            best = []
            for s in SEEDS:
                d = D.get((rho, v, var, s, T), {})
                c = [(e, ab) for ab, e in d.items() if e is not None]
                if c:
                    best.append(min(c))
            if not best:
                w(f"| {T} | {rho} | {v:.0f}/{var} | 0,00 | — | — | — |"); continue
            m6 = np.mean([ab[1] == 0 for _, ab in best]); fc = np.mean([ab[0] == 0 for _, ab in best]); mx = np.mean([ab[0] >= 1 and ab[1] >= 1 for _, ab in best])
            edge = np.mean([ab[0] == 4 or ab[1] == 3 or sum(ab) == 5 for _, ab in best])
            w(f"| {T} | {rho} | {v:.0f}/{var} | {len(best)/len(SEEDS):.2f} | {m6:.2f} / {fc:.2f} / {mx:.2f} | {np.median([e for e, _ in best]):.0f} | {edge:.2f} |")

w("\n## 3. Yalnız-FlyCart filonun gereken boyutu ve karma filonun etkisi (T = 900 s)\n")
w("Sermaye maliyeti modellenmedi (kaynaklı fiyat yok); bu nedenle kırılma maliyet oranı r* BAŞLIK sonucu olarak raporlanmaz (r* = a/(k_F − b') tanım gereği ≈ 1 civarı çıkar, tautolojiktir).\n")
w("| ρ | FC hızı / taban | yalnız-FC gereken b (1/2/3/>3) | karma ile daha az toplam drone yeterli oranı |\n|---|---|---|---|")
for rho in RHOS:
    for (v, var) in CFG:
        cb = collections.Counter(); less = 0; tot = 0
        for s in SEEDS:
            d = D.get((rho, v, var, s, 900), {})
            fcb = [b for (a, b), e in d.items() if a == 0 and e is not None]
            kF = min(fcb) if fcb else 99
            cb[min(kF, 4)] += 1
            mixed = [a + b for (a, b), e in d.items() if e is not None and a >= 1 and b >= 1]
            tot += 1
            if mixed and min(mixed) < kF:
                less += 1
        w(f"| {rho} | {v:.0f}/{var} | " + " / ".join(f"{cb.get(i,0)/len(SEEDS):.2f}" for i in (1, 2, 3, 4)) + f" | {less/tot:.2f} |")
open(os.path.join(HERE, f"MIXED_SUMMARY_{tag}.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
